#!/usr/bin/env python
"""Hand surface mesh from board videos (camera poses known from the screen board, no SfM).

    python recon_board.py --videos "scans/*.mp4" --board-spec data/board_screen.json \
        --camera data/camera.json --out out/hand_scan.ply

Capture (agreed with Olex): open right hand ~15 cm in front of the on-screen ChArUco
board, forearm resting on books, phone (4K60, manual exposure) sweeping 10-15 s around
it. Take 1 (the reference frame of the output): palm facing the camera. Take 2: thumb
up, palm facing left. More takes are allowed; every take is registered to take 1.

Pipeline
 1. Frames: the sharpest frame in every 1/fps window (Laplacian variance), frames far
    below the take's median sharpness are dropped. Board corners (gripscan.board.detect)
    per frame; the video lens is calibrated from the board frames
    (gripscan.relaxed_pose.calibrate_from_frames); per-frame pose (pose_of), optionally
    with a per-frame focal refinement (autofocus breathing, --focal refine).
 2. Hand mask per frame: the screen image is rendered through the board pose (ray/plane
    per pixel, lens distortion included), a smooth photometric model (gain/offset per
    colour channel, vignetting) and the board blur are fitted, and every pixel is
    compared with the colour segment the board could show there (edge tolerant, so the
    skin colour separates from the grey board even on blurred square edges). The hand
    blob is picked with MediaPipe landmarks. Pixels outside the board area, and other
    foreground blobs (books ...), are UNKNOWN, never background.
 3. Per take: (a) robust visual hull: per view signed silhouette distance (mm at the
    voxel's depth), the k-th largest over the views that know the voxel (tolerates a few
    bad masks), coarse 2 mm -> 0.5 mm band. (b) takes 2.. are registered to take 1:
    initial rigid transform from the triangulated MediaPipe landmarks (robust Kabsch),
    refined by silhouette coherence (every silhouette ray of one take must hit the other
    take's hull). (c) joint robust hull from all views of all takes at 0.5 mm.
    (d) photo-consistency refinement for what silhouettes cannot see (palm hollow,
    finger webs, faces seen only head-on): a plane sweep along the hull normal with
    multi-view NCC of 5x5 tangent-plane patches, visibility from the hull, a smoothness
    -regularised displacement field (rims stay put), re-meshed from a signed distance
    field (always watertight). Optionally COLMAP dense points (--colmap-fused) replace
    the NCC depths. (e) wrist cut (plane at the MediaPipe wrist, normal towards the
    middle MCP), largest component, marching cubes, light Taubin smoothing.
 4. Outputs, next to --out: <name>.ply (watertight, board mm of take 1),
    <name>_hull.ply, <name>_landmarks.json (21 triangulated landmarks per take, like
    relaxed_pose.py, plus all takes mapped into the output frame), <name>_report.json
    (mask IoU vs reprojection per view, views per surface point, hull vs refined,
    calibration and focal breathing, registration), and --colmap DIR writes a COLMAP
    project with the known poses (see write_colmap for the exact commands).

Frames and units: board frame of gripscan.board (mm, origin at the top-left chessboard
corner, X right, Y down, Z into the screen, i.e. the hand in front of the screen has
negative Z). The output is in take 1's board frame.
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

def _find_gripscan():
    """Folder that contains the gripscan package: $GRIPSCAN_PATH, a parent of this file,
    a sibling checkout, or the session's copy."""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get("GRIPSCAN_PATH", "")]
    d = here
    for _ in range(4):
        cands += [d, os.path.join(d, "grip_scan"), os.path.join(d, "pg")]
        d = os.path.dirname(d)
    cands.append("/tmp/claude-0/hs/pg")
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "gripscan", "board.py")):
            return c
    raise SystemExit("gripscan package not found: set GRIPSCAN_PATH to the folder that contains gripscan/")


GRIPSCAN = _find_gripscan()
if GRIPSCAN not in sys.path:
    sys.path.insert(0, GRIPSCAN)
from gripscan.board import BoardSpec, Camera, Pose, detect, make_board  # noqa: E402
from gripscan.landmarks import NAMES  # noqa: E402
from gripscan.relaxed_pose import VIDEO_EXT, calibrate_from_frames  # noqa: E402
from gripscan.relaxed_pose import pose_of as pose_plain  # noqa: E402
from gripscan.triangulate import baseline_deg, robust_point  # noqa: E402

MP_MODEL_CANDIDATES = [
    os.path.join(GRIPSCAN, "models", "hand_landmarker.task"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task"),
    "/tmp/claude-0/hs/work/recon/models/hand_landmarker.task",
]
MIN_CORNERS = 12
MAX_POSE_RMS = 3.0
UNSEEN = 5.0  # field value (mm, "outside") for voxels no view knows


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================ geometry helpers

def rodrigues(rv):
    return cv2.Rodrigues(np.asarray(rv, float).reshape(3, 1))[0]


def project(X, R, t, K, dist, r2max=1.2):
    """Board points (N,3) -> pixel u, v and camera depth z (OpenCV 5-coef model).
    Points behind the camera or far outside the lens model get NaN."""
    Xc = X @ R.T + t
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = Xc[:, 0] / z
        y = Xc[:, 1] / z
    d = list(np.ravel(dist)) + [0.0] * 5
    k1, k2, p1, p2, k3 = d[:5]
    r2 = x * x + y * y
    rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = K[0, 0] * xd + K[0, 1] * yd + K[0, 2]
    v = K[1, 1] * yd + K[1, 2]
    bad = ~(z > 1e-6) | ~(r2 < r2max)
    u[bad] = np.nan
    v[bad] = np.nan
    return u, v, z


def pixel_rays(K, dist, W, H, step=8):
    """Undistorted normalized ray (x, y) per pixel, (H, W, 2) float32, from a coarse grid."""
    gw, gh = int(np.ceil(W / step)), int(np.ceil(H / step))
    gx = (np.arange(gw) + 0.5) * step - 0.5
    gy = (np.arange(gh) + 0.5) * step - 0.5
    P = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 1, 2).astype(np.float64)
    n = cv2.undistortPoints(P, K, np.ravel(dist)).reshape(gh, gw, 2).astype(np.float32)
    full = cv2.resize(n, (gw * step, gh * step), interpolation=cv2.INTER_LINEAR)
    return full[:H, :W]


def board_hit(rays, R, t):
    """Intersection of per-pixel rays with the board plane z=0: (bx, by, depth)."""
    Rt = R.T
    C = -Rt @ t
    d = rays[..., 0:1] * Rt[:, 0] + rays[..., 1:2] * Rt[:, 1] + Rt[:, 2]  # (H,W,3)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -C[2] / d[..., 2]
    bx = C[0] + s * d[..., 0]
    by = C[1] + s * d[..., 1]
    s = np.where(s > 0, s, np.nan)
    return bx.astype(np.float32), by.astype(np.float32), s.astype(np.float32)


def sample_image(img, u, v, border=np.nan):
    """Bilinear samples of a float32 image at (u, v) (any shape); NaN outside."""
    shp = u.shape
    n = u.size
    cols = 4096
    rows = int(np.ceil(n / cols)) if n else 1
    mu = np.full(rows * cols, -1e6, np.float32)
    mv = np.full(rows * cols, -1e6, np.float32)
    uu = np.ravel(u).astype(np.float32)
    vv = np.ravel(v).astype(np.float32)
    ok = np.isfinite(uu) & np.isfinite(vv)
    mu[:n][ok] = uu[ok]
    mv[:n][ok] = vv[ok]
    out = cv2.remap(img, mu.reshape(rows, cols), mv.reshape(rows, cols), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=float(border))
    return out.reshape(-1)[:n].reshape(shp)


def rasterize(u, v, z, faces, W, H, max_cand=3_000_000):
    """Z-buffer rasterization of projected triangles (pixel centres at integer coords).

    Returns face id (H, W) int32 (-1 empty), barycentric weights b0, b1 of the face's
    first two vertices (H, W) float32 and depth (H, W) float32 (inf empty)."""
    zb = np.full(H * W, np.inf, np.float32)
    fid = np.full(H * W, -1, np.int32)
    B0 = np.zeros(H * W, np.float32)
    B1 = np.zeros(H * W, np.float32)
    ok = np.isfinite(u) & np.isfinite(v) & (z > 0)
    f_all = np.arange(len(faces))
    good = ok[faces].all(1)
    F = faces[good]
    fids = f_all[good]
    xs, ys, zs = u[F], v[F], z[F]
    x0 = np.ceil(xs.min(1)).astype(np.int64)
    x1 = np.floor(xs.max(1)).astype(np.int64)
    y0 = np.ceil(ys.min(1)).astype(np.int64)
    y1 = np.floor(ys.max(1)).astype(np.int64)
    x0 = np.maximum(x0, 0)
    y0 = np.maximum(y0, 0)
    x1 = np.minimum(x1, W - 1)
    y1 = np.minimum(y1, H - 1)
    keep = (x1 >= x0) & (y1 >= y0)
    size = np.maximum(x1 - x0 + 1, y1 - y0 + 1)
    blocks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    lo = 0
    for B in blocks:
        sel = np.nonzero(keep & (size > lo) & (size <= B))[0]
        lo = B
        if len(sel) == 0:
            continue
        oy, ox = np.mgrid[0:B, 0:B]
        ox = ox.ravel()
        oy = oy.ravel()
        per = max(1, max_cand // (B * B))
        for c0 in range(0, len(sel), per):
            s = sel[c0:c0 + per]
            px = x0[s, None] + ox[None]
            py = y0[s, None] + oy[None]
            inb = (px <= x1[s, None]) & (py <= y1[s, None])
            xa, xb, xc = xs[s, 0:1], xs[s, 1:2], xs[s, 2:3]
            ya, yb, yc = ys[s, 0:1], ys[s, 1:2], ys[s, 2:3]
            den = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
            den = np.where(np.abs(den) < 1e-12, np.nan, den)
            l0 = ((yb - yc) * (px - xc) + (xc - xb) * (py - yc)) / den
            l1 = ((yc - ya) * (px - xc) + (xa - xc) * (py - yc)) / den
            l2 = 1 - l0 - l1
            e = -1e-7
            ins = inb & (l0 >= e) & (l1 >= e) & (l2 >= e)
            if not ins.any():
                continue
            zz = l0 * zs[s, 0:1] + l1 * zs[s, 1:2] + l2 * zs[s, 2:3]
            ii, jj = np.nonzero(ins)
            pix = py[ii, jj] * W + px[ii, jj]
            zv = zz[ii, jj].astype(np.float32)
            o = np.lexsort((zv, pix))
            pix, zv = pix[o], zv[o]
            first = np.ones(len(pix), bool)
            first[1:] = pix[1:] != pix[:-1]
            o2 = o[first]
            pix, zv = pix[first], zv[first]
            upd = zv < zb[pix]
            pix, zv, o2 = pix[upd], zv[upd], o2[upd]
            zb[pix] = zv
            fid[pix] = fids[s[ii[o2]]]
            B0[pix] = l0[ii[o2], jj[o2]]
            B1[pix] = l1[ii[o2], jj[o2]]
    return fid.reshape(H, W), B0.reshape(H, W), B1.reshape(H, W), zb.reshape(H, W)


def kabsch(A, B, w=None):
    """Rigid (R, t) with B ~ R A + t (weighted)."""
    w = np.ones(len(A)) if w is None else np.asarray(w, float)
    ca = (w[:, None] * A).sum(0) / w.sum()
    cb = (w[:, None] * B).sum(0) / w.sum()
    Hm = ((A - ca) * w[:, None]).T @ (B - cb)
    U, _, Vt = np.linalg.svd(Hm)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def robust_kabsch(A, B, iters=5):
    ok = np.all(np.isfinite(A), 1) & np.all(np.isfinite(B), 1)
    w = ok.astype(float)
    A0 = np.where(ok[:, None], A, 0)
    B0 = np.where(ok[:, None], B, 0)
    for _ in range(iters):
        R, t = kabsch(A0, B0, w + 1e-9)
        r = np.linalg.norm(A0 @ R.T + t - B0, axis=1)
        s = max(2.0, 2.5 * np.median(r[ok]))
        w = (ok & (r < s)).astype(float)
    return R, t, r


# ============================================================ screen model

class ScreenModel:
    """What the monitor shows, as a function of board coordinates (mm)."""

    def __init__(self, spec, known_margin_mm=8.0, mode="screen", inset_mm=10.0):
        self.spec = spec
        bw_mm = spec.squares_x * spec.square_mm
        bh_mm = spec.squares_y * spec.square_mm
        self.board_mm = (bw_mm, bh_mm)
        self.screen_rect = None
        tex = None
        try:
            import re
            m = re.search(r"screen (\d+)x(\d+), pitch ([0-9.]+)", spec.source)
            if m:
                from gripscan import screen_board as sb
                w, h, pitch = int(m.group(1)), int(m.group(2)), float(m.group(3))
                if (sb.SQUARES_X, sb.SQUARES_Y) == (spec.squares_x, spec.squares_y) and \
                        abs(sb.SQUARE_PX * pitch - spec.square_mm) < 0.05:
                    tex = sb.render(w, h, pitch)
                    bwp = sb.SQUARES_X * sb.SQUARE_PX
                    self.x0, self.y0 = (w - bwp) // 2, 40
                    self.ppm = 1.0 / pitch
                    self.screen_rect = (-self.x0 * pitch, -self.y0 * pitch,
                                        (w - self.x0) * pitch, (h - self.y0) * pitch)
        except Exception as e:  # pragma: no cover
            log("screen layout not reproduced:", e)
        if tex is None:  # printed board (or unknown layout): board + white margin only
            self.ppm = 8.0
            pad = int(np.ceil((known_margin_mm + 5) * self.ppm))
            board = make_board(spec)
            bw, bh = int(round(bw_mm * self.ppm)), int(round(bh_mm * self.ppm))
            img = board.generateImage((bw, bh), marginSize=0, borderBits=1)
            tex = np.full((bh + 2 * pad, bw + 2 * pad), 255, np.uint8)
            tex[pad:pad + bh, pad:pad + bw] = img
            self.x0 = self.y0 = pad
            m_ = pad / self.ppm
            self.screen_rect = (-m_, -m_, bw_mm + m_, bh_mm + m_)
        self.tex = tex.astype(np.float32) / 255.0
        sr = self.screen_rect
        self.known_rect = (max(sr[0], -known_margin_mm), max(sr[1], -known_margin_mm),
                           min(sr[2], bw_mm + known_margin_mm), min(sr[3], bh_mm + known_margin_mm))
        self.layout = tex is not None and self.ppm != 8.0
        if mode == "screen" and self.layout:  # whole screen image is reproduced exactly
            self.known_rect = (sr[0] + inset_mm, sr[1] + inset_mm, sr[2] - inset_mm, sr[3] - inset_mm)
        self._blur_cache = {}

    def tex_blurred(self, sigma):
        key = round(float(sigma), 1)
        if key not in self._blur_cache:
            self._blur_cache[key] = (cv2.GaussianBlur(self.tex, (0, 0), key) if key > 0.3
                                     else self.tex)
        return self._blur_cache[key]

    def to_tex(self, bx, by):
        return self.x0 + bx * self.ppm - 0.5, self.y0 + by * self.ppm - 0.5

    def known(self, bx, by):
        r = self.known_rect
        return (bx >= r[0]) & (bx <= r[2]) & (by >= r[1]) & (by <= r[3])


# ============================================================ views

class View:
    """One selected frame: images at processing scale, camera, board pose, mask."""
    __slots__ = ("take", "idx", "ts", "img", "gray", "K", "dist", "R", "t", "f_scale",
                 "rms", "ncorner", "sharp", "lm", "mask", "sdf", "info", "path")

    def C(self):
        return -self.R.T @ self.t

    def ext(self, T=None):
        """Extrinsics mapping OUTPUT-frame points into this camera (T: output -> take)."""
        if T is None:
            return self.R, self.t
        RT, tT = T
        return self.R @ RT, self.R @ tT + self.t


def read_video_best(path, fps, sharp_side=960):
    """Sharpest frame of every 1/fps window: [(ts_ms, img, sharpness)]."""
    if os.path.splitext(path)[1].lower() not in VIDEO_EXT:
        raise SystemExit(f"{path}: expected a video")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    win = max(1, int(round(src / fps)))
    out, best, i = [], None, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        sc = sharp_side / img.shape[1]
        g = cv2.cvtColor(cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        s = float(cv2.Laplacian(g, cv2.CV_32F).var())
        if best is None or s > best[2]:
            best = (int(i * 1000 / src), img, s)
        i += 1
        if i % win == 0:
            out.append(best)
            best = None
    if best is not None:
        out.append(best)
    cap.release()
    return out


def refine_focal(det, cam, pose, sigma=0.01):
    """Per-frame pose + focal scale (prior N(1, sigma)) from the board corners."""
    from scipy.optimize import least_squares
    obj, img = det

    def res(p):
        K = cam.K.copy()
        K[0, 0] *= p[6]
        K[1, 1] *= p[6]
        pr, _ = cv2.projectPoints(obj, p[:3], p[3:6], K, cam.dist)
        return np.concatenate([(pr.reshape(-1, 2) - img).ravel(), [(p[6] - 1) / sigma * 0.5]])

    p0 = np.concatenate([pose.rvec.ravel(), pose.tvec.ravel(), [1.0]])
    r = least_squares(res, p0, x_scale=np.r_[[1e-3] * 3, [1.0] * 3, [1e-3]], method="lm")
    p = r.x
    rr = res(p)[:-1].reshape(-1, 2)
    return p[:3].reshape(3, 1), p[3:6].reshape(3, 1), float(p[6]), float(np.sqrt(np.mean(np.sum(rr ** 2, 1))))


_GAMMA_LUT = (np.linspace(0, 1, 256) ** 2.4 * 255).astype(np.uint8)


def detect_robust(img, board):
    """Board corners, robust to screen glare / overexposure: the plain grey image, CLAHE
    and a darkening gamma are tried and the detection with most corners wins."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    best, how = None, None
    variants = [("gray", lambda g: g),
                ("clahe", lambda g: cv2.createCLAHE(3.0, (16, 16)).apply(g)),
                ("gamma", lambda g: cv2.LUT(g, _GAMMA_LUT)),
                ("gamma+clahe", lambda g: cv2.createCLAHE(4.0, (16, 16)).apply(cv2.LUT(g, _GAMMA_LUT)))]
    for name, fn in variants:
        d = detect(fn(gray), board)
        if d is not None and (best is None or len(d[0]) > len(best[0])):
            best, how = d, name
        if best is not None and len(best[0]) >= 60:
            break
    return best, how



def clean_det(det, cam, min_px=3.0, rounds=3):
    """Board pose with corner outlier rejection (glare/blur corrupt single corners).
    Returns (inlier detection, Pose) or (det, None)."""
    if det is None:
        return det, None
    obj, img = det
    keep = np.ones(len(obj), bool)
    pose = None
    for _ in range(rounds):
        if keep.sum() < max(6, int(0.6 * len(obj))):
            return det, None
        o, q = obj[keep], img[keep]
        if pose is None:
            ok, rv, tv = cv2.solvePnP(o, q, cam.K, cam.dist)
        else:
            ok, rv, tv = cv2.solvePnP(o, q, cam.K, cam.dist, pose.rvec.copy(), pose.tvec.copy(), True)
        if not ok:
            return det, None
        pr, _ = cv2.projectPoints(obj, rv, tv, cam.K, cam.dist)
        e = np.linalg.norm(pr.reshape(-1, 2) - img, axis=1)
        rms = float(np.sqrt(np.mean(e[keep] ** 2)))
        pose = Pose(rv, tv, rms, int(keep.sum()))
        thr = max(min_px, 3.0 * 1.4826 * np.median(e[keep]))
        new = e < thr
        if (new == keep).all():
            break
        keep = new
    return (obj[keep], img[keep]), pose


def pose_of(det, cam):
    return clean_det(det, cam)[1]

def calibrate_robust(dets, size, min_corners=20):
    """calibrate_from_frames, then drop views with high per-view error and refit."""
    cam, rms = calibrate_from_frames(dets, size)
    if cam is None:
        return None, None, 0
    views, errs = [], []
    for d in dets:
        if d is None or len(d[0]) < min_corners:
            continue
        d2, p = clean_det(d, cam)
        if p is None or len(d2[0]) < min_corners:
            continue
        views.append(d2)
        errs.append(p.reproj_px)
    errs = np.array(errs)
    keep = errs < max(1.5, 2.5 * np.median(errs))
    views = [d for d, k in zip(views, keep) if k]
    if len(views) < 8:
        return cam, rms, len(views)
    pick = np.linspace(0, len(views) - 1, min(60, len(views))).astype(int)
    objs = [views[i][0].astype(np.float32) for i in pick]
    imgs = [views[i][1].astype(np.float32) for i in pick]
    flags = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    K0 = None
    if len(views) < 15:  # few views: principal point at the centre, square pixels
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_USE_INTRINSIC_GUESS
        K0 = cam.K.copy()
        K0[0, 2], K0[1, 2] = (size[0] - 1) / 2, (size[1] - 1) / 2
        K0[1, 1] = K0[0, 0]
    rms2, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, K0, None, flags=flags)
    return Camera(K, dist, size), float(rms2), len(views)


def interp_pose(ts, poses, t, max_gap_ms):
    """Camera pose at time t from accepted poses (ts sorted): slerp/lerp, None if too far."""
    from scipy.spatial.transform import Rotation, Slerp
    if not len(ts):
        return None
    i = np.searchsorted(ts, t)
    if 0 < i < len(ts) and ts[i] - ts[i - 1] <= 2 * max_gap_ms:
        t0, t1 = ts[i - 1], ts[i]
        (R0, C0), (R1, C1) = poses[i - 1], poses[i]
        w = (t - t0) / max(t1 - t0, 1)
        R = Slerp([0, 1], Rotation.from_matrix([R0, R1]))([w]).as_matrix()[0]
        return R, (1 - w) * C0 + w * C1
    j = i - 1 if i == len(ts) or (i > 0 and t - ts[i - 1] < ts[i] - t) else i
    if abs(ts[j] - t) <= max_gap_ms / 2:
        return poses[j]
    return None


def _local_norm(g, sig):
    g = g.astype(np.float32)
    m = cv2.GaussianBlur(g, (0, 0), sig)
    d = g - m
    sd = np.sqrt(cv2.GaussianBlur(d * d, (0, 0), sig)) + 0.02
    return d / sd


def pose_direct(img_small, Ks, dist, prior, screen, lm=None, scale=0.5, min_cc=0.55):
    """Board pose without detected corners: the screen image is rendered at the prior
    pose (trajectory interpolation) and aligned to the frame by ECC (homography of the
    screen plane, coarse to fine) on locally contrast-normalized, undistorted images, so
    glare and overexposure matter little. Returns (R, t, ecc correlation) or None."""
    R0, C0 = prior
    t0 = -R0 @ C0
    g = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    K = scaled_K(Ks, scale)
    H, W = g.shape
    und = cv2.undistort(g, K, dist).astype(np.float32) / 255.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rays = np.stack([(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1]], -1)
    bx, by, depth = board_hit(rays, R0, t0)
    known = screen.known(bx, by) & np.isfinite(depth)
    if known.mean() < 0.05:
        return None
    foot = np.nanmedian(depth[known]) / K[0, 0] * screen.ppm
    sx, sy = screen.to_tex(bx, by)
    E = sample_image(screen.tex_blurred(0.45 * foot), sx, sy)
    E = np.where(known, E, 0.5).astype(np.float32)
    mask = known.copy()
    if lm is not None:  # the hand is not board
        P = cv2.undistortPoints(lm.reshape(-1, 1, 2) * scale, K, dist, P=K).reshape(-1, 2)
        hull = cv2.convexHull(P.astype(np.float32)).astype(np.int32)
        hm = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(hm, hull, 1)
        hm = cv2.dilate(hm, np.ones((5, 5), np.uint8), iterations=int(0.04 * W))
        mask &= hm == 0
    mask &= und < 0.985  # saturated glare carries nothing
    Hw = np.eye(3, dtype=np.float32)
    cc = 0.0
    for lev in (3, 2, 1, 0):
        f = 2 ** lev
        a_ = cv2.resize(_local_norm(E, 3), (W // f, H // f), interpolation=cv2.INTER_AREA)
        b_ = cv2.resize(_local_norm(und, 3), (W // f, H // f), interpolation=cv2.INTER_AREA)
        m_ = cv2.resize(mask.astype(np.uint8), (W // f, H // f), interpolation=cv2.INTER_NEAREST)
        S = np.diag([1.0 / f, 1.0 / f, 1.0]).astype(np.float32)
        Hl = S @ Hw @ np.linalg.inv(S)
        try:
            cc, Hl = cv2.findTransformECC(a_, b_, Hl.astype(np.float32), cv2.MOTION_HOMOGRAPHY,
                                          (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5),
                                          m_, 5)
        except cv2.error:
            return None
        Hw = np.linalg.inv(S) @ Hl @ S
    if cc < min_cc:
        return None
    # board points -> template pixels (prior, pinhole) -> frame pixels via Hw -> PnP
    gx, gy = np.meshgrid(np.linspace(0, screen.board_mm[0], 9), np.linspace(0, screen.board_mm[1], 7))
    B = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    p = (B @ R0.T + t0) @ K.T
    p = p[:, :2] / p[:, 2:]
    q = np.column_stack([p, np.ones(len(p))]) @ Hw.T
    q = q[:, :2] / q[:, 2:]
    ok, rv, tv = cv2.solvePnP(B, q, K, None, cv2.Rodrigues(R0)[0], t0.reshape(3, 1), True)
    if not ok:
        return None
    return rodrigues(rv), tv.ravel(), float(cc)


def find_mp_model(path):
    for p in ([path] if path else []) + MP_MODEL_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return None


def load_takes(paths, spec, a):
    """Frames, board detections, MediaPipe landmarks for every take."""
    board = make_board(spec)
    mp_path = find_mp_model(a.mp_model)
    if mp_path is None:
        log("WARNING: MediaPipe model not found; landmarks (registration init, blob pick, "
            "wrist cut) disabled")
    raw = []
    for ti, path in enumerate(paths):
        frames = read_video_best(path, a.fps)
        sh = np.array([f[2] for f in frames])
        med = np.median(sh) if len(sh) else 0
        lmk = None
        if mp_path:
            from gripscan.relaxed_pose import landmarker
            lmk = landmarker(mp_path)
        recs = []
        last_ts = -1
        for k, (ts, img, s) in enumerate(frames):
            rec = {"ts": ts, "sharp": s, "img_full": None, "det": None, "lm": None,
                   "blurry": bool(s < a.blur_frac * med)}
            if rec["blurry"]:
                recs.append(rec)
                continue
            rec["det"], rec["det_how"] = detect_robust(img, board)
            sc = a.scale
            small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            rec["img"] = small
            if lmk is not None:
                import mediapipe as mp
                ts_mp = max(ts, last_ts + 1)
                last_ts = ts_mp
                res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                    data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), ts_mp)
                if res.hand_landmarks:
                    h, w = small.shape[:2]
                    rec["lm"] = np.array([[q.x * w, q.y * h] for q in res.hand_landmarks[0]])
            recs.append(rec)
        if lmk is not None:
            lmk.close()
        size = (frames[0][1].shape[1], frames[0][1].shape[0]) if frames else None
        raw.append({"path": path, "size": size, "recs": recs})
        log(f"take {ti + 1} {os.path.basename(path)}: {len(recs)} frames, "
            f"{sum(r['blurry'] for r in recs)} blurry dropped, board in "
            f"{sum(r['det'] is not None for r in recs)}, hand landmarks in "
            f"{sum(r['lm'] is not None for r in recs)}")
    return raw


def load_scan_pkg(pkg_dir, spec, a):
    """Olex's PC-side package (make_pkg.py): frames.json with full-res board corners and
    MediaPipe landmarks per sampled frame, plus full-res hand crops. Returns the `raw`
    structure of load_takes (one take), the crops standing in for the frames."""
    meta = json.load(open(os.path.join(pkg_dir, "frames.json")))
    board = make_board(spec)
    obj_all = np.asarray(board.getChessboardCorners(), float).reshape(-1, 3)
    size = tuple(meta["size"])
    sh = np.array([f["sharpness"] for f in meta["frames"] if f.get("crop")])
    med = float(np.median(sh)) if len(sh) else 0.0
    recs = []
    for f in meta["frames"]:
        rec = {"ts": int(round(f["time_s"] * 1000)), "sharp": f.get("sharpness", 0.0), "det": None,
               "det_how": f.get("board_method"), "lm": None, "blurry": False}
        if f.get("corner_ids"):
            ids = np.asarray(f["corner_ids"], int)
            if len(ids) >= 6:
                rec["det"] = (obj_all[ids], np.asarray(f["corner_px"], float))
        c = f.get("crop")
        if c and rec["det"] is not None:
            img = cv2.imread(os.path.join(pkg_dir, c["file"]))
            if img is None:
                continue
            rec["img"] = cv2.resize(img, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
            rec["crop"] = (c["x0"], c["y0"])
            rec["blurry"] = bool(rec["sharp"] < a.blur_frac * med)
            if f.get("landmarks_px") is not None:
                L = np.asarray(f["landmarks_px"], float) - [c["x0"], c["y0"]]
                rec["lm"] = (L + 0.5) * a.scale - 0.5
        recs.append(rec)
    log(f"scan package {pkg_dir}: {len(recs)} sampled frames of {meta.get('video')}, board corners in "
        f"{sum(r['det'] is not None for r in recs)}, hand crops {sum('img' in r for r in recs)}")
    return [{"path": os.path.join(pkg_dir, meta.get("video", "video")), "size": size, "recs": recs}], meta


def camera_fallback(camera_json, size):
    """camera.json (photo mode) mapped to a video size: same lens, scaled by width, centre crop."""
    c = Camera.from_dict(json.load(open(camera_json)))
    s = size[0] / c.size[0]
    K = c.K.copy()
    K[:2] *= s
    K[1, 2] -= (c.size[1] * s - size[1]) / 2
    return Camera(K, c.dist, tuple(size))


def build_views(raw, a, report, screen):
    """Lens per video size, board pose per frame -> list of View per take."""
    cams = {}
    calib = {}
    for size in {t["size"] for t in raw if t["size"]}:
        dets = [r["det"] for t in raw if t["size"] == size for r in t["recs"]]
        cam, rms, nviews = calibrate_robust(dets, size)
        fb = camera_fallback(a.camera, size) if a.camera and os.path.exists(a.camera) else None
        if cam is None:
            if fb is None:
                raise SystemExit(f"not enough board views to calibrate {size} and no --camera")
            cam, rms = fb, None
            log(f"lens {size}: too few board views, using camera.json scaled (photo-mode lens; "
                "video modes usually crop, so expect errors)")
        else:
            log(f"lens {size[0]}x{size[1]} from {nviews} board frames: f={cam.K[0, 0]:.1f}px "
                f"c=({cam.K[0, 2]:.0f},{cam.K[1, 2]:.0f}) k1 {np.ravel(cam.dist)[0]:.3f} RMS {rms:.2f}px")
            if rms > 1.5:
                log("WARNING: lens calibration RMS above 1.5 px (glare? motion blur?)")
        cams[size] = cam
        calib[f"{size[0]}x{size[1]}"] = {
            "K": cam.K.tolist(), "dist": np.ravel(cam.dist).tolist(), "rms_px": rms, "views": nviews,
            "camera_json_scaled_f": float(fb.K[0, 0]) if fb is not None else None}
    report["calibration"] = calib
    takes = []
    focal_stats = []
    pose_stats = []
    for ti, t in enumerate(raw):
        cam = cams[t["size"]]
        sc = a.scale
        # pass 1: frames with enough corners; pass 2: few corners, with a trajectory prior
        acc = {}
        traj_only = {}
        for k, r in enumerate(t["recs"]):
            if r["det"] is None or len(r["det"][0]) < MIN_CORNERS:
                continue
            det2, pose = clean_det(r["det"], cam)
            if pose is not None:
                r["det"] = det2
            if r.get("img") is None:  # pose only (trajectory prior), no image to use
                if pose is not None and pose.reproj_px <= MAX_POSE_RMS:
                    traj_only[k] = pose
                continue
            if pose is not None and pose.reproj_px <= MAX_POSE_RMS:
                acc[k] = (pose, False)
        allp = {**{k: p for k, p in traj_only.items()}, **{k: v_[0] for k, v_ in acc.items()}}
        ks = sorted(allp)
        ts = np.array([t["recs"][k]["ts"] for k in ks], float)
        traj = [(allp[k].R, allp[k].camera_centre()) for k in ks]
        n_weak = n_rej = 0
        for k, r in enumerate(t["recs"]):
            if k in acc or r["det"] is None or a.weak_poses == "skip" or r.get("img") is None:
                continue
            prior = interp_pose(ts, traj, r["ts"], a.prior_gap_ms)
            if prior is None:
                n_rej += 1
                continue
            Rp, Cp = prior
            rv0 = cv2.Rodrigues(Rp)[0]
            tv0 = (-Rp @ Cp).reshape(3, 1)
            obj, img = r["det"]
            ok, rv, tv = cv2.solvePnP(obj, img, cam.K, cam.dist, rv0.copy(), tv0.copy(), True,
                                      cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                n_rej += 1
                continue
            pr, _ = cv2.projectPoints(obj, rv, tv, cam.K, cam.dist)
            rms = float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - img) ** 2, 1))))
            pose = Pose(rv, tv, rms, len(obj))
            dC = np.linalg.norm(pose.camera_centre() - Cp)
            dang = np.degrees(np.linalg.norm(cv2.Rodrigues(pose.R @ Rp.T)[0]))
            if rms <= 2.0 and dC <= a.prior_mm and dang <= a.prior_deg:
                acc[k] = (pose, True)
                n_weak += 1
            else:
                n_rej += 1
        # pass 3: no usable corners (glare): trajectory prior + direct alignment of the screen image
        n_direct = 0
        direct_cc = {}
        if a.direct_poses:
            Ks = scaled_K(cam.K, sc)
            for k, r in enumerate(t["recs"]):
                if k in acc or r["blurry"] or r.get("img") is None or r.get("crop") is not None:
                    continue
                prior = interp_pose(ts, traj, r["ts"], a.prior_gap_ms * 1.5)
                if prior is None:
                    continue
                res = pose_direct(r["img"], Ks, np.ravel(cam.dist), prior, screen, r["lm"])
                if res is None:
                    continue
                R_, t_, cc = res
                C_ = -R_.T @ t_
                dC = np.linalg.norm(C_ - prior[1])
                dang = np.degrees(np.linalg.norm(cv2.Rodrigues(R_ @ prior[0].T)[0]))
                if dC <= 1.5 * a.prior_mm and dang <= 1.5 * a.prior_deg:
                    acc[k] = (Pose(cv2.Rodrigues(R_)[0], t_.reshape(3, 1), float("nan"), 0), True)
                    direct_cc[k] = cc
                    n_direct += 1
        views = []
        for k in sorted(acc):
            r = t["recs"][k]
            pose, weak = acc[k]
            fs, rms = 1.0, pose.reproj_px
            rvec, tvec = pose.rvec, pose.tvec
            if a.focal != "fixed" and not weak:
                rv2, tv2, fs2, rms2 = refine_focal(r["det"], cam, pose, a.focal_sigma)
                focal_stats.append({"take": ti + 1, "frame": k, "f_scale": fs2,
                                    "rms_fixed": pose.reproj_px, "rms_refined": rms2,
                                    "corners": len(r["det"][0]),
                                    "dz_mm": float((tv2 - tvec)[2, 0])})
                if a.focal == "refine":
                    rvec, tvec, fs, rms = rv2, tv2, fs2, rms2
            v = View()
            v.take, v.idx, v.ts, v.sharp, v.path = ti, k, r["ts"], r["sharp"], t["path"]
            v.img = r["img"]
            v.gray = cv2.cvtColor(v.img, cv2.COLOR_BGR2GRAY)
            K = cam.K.copy()
            K[0, 0] *= fs
            K[1, 1] *= fs
            if r.get("crop") is not None:  # crop of the full frame
                K[0, 2] -= r["crop"][0]
                K[1, 2] -= r["crop"][1]
            K[:2] *= sc
            K[0, 2] += 0.5 * sc - 0.5  # pixel-centre convention under resizing
            K[1, 2] += 0.5 * sc - 0.5
            v.K, v.dist = K, np.ravel(cam.dist).astype(float)
            v.R, v.t = rodrigues(rvec), np.ravel(tvec).astype(float)
            v.f_scale, v.rms, v.ncorner = fs, rms, (len(r["det"][0]) if r["det"] is not None else 0)
            v.lm = r["lm"]
            v.info = {"weak_pose": weak, "detect": r.get("det_how"),
                      "pose_source": ("direct" if k in direct_cc else "prior+corners" if weak else "corners")}
            if k in direct_cc:
                v.info["ecc"] = direct_cc[k]
            views.append(v)
        n_nodet = sum(r["det"] is None and not r["blurry"] for r in t["recs"])
        pose_stats.append({"take": ti + 1, "frames": len(t["recs"]),
                           "blurry": sum(r["blurry"] for r in t["recs"]), "no_board": n_nodet,
                           "strong_pose": len(acc) - n_weak - n_direct, "weak_pose_with_prior": n_weak,
                           "direct_pose_no_corners": n_direct,
                           "rejected": n_rej,
                           "detect_variant": {h: sum(r.get("det_how") == h for r in t["recs"])
                                              for h in ["gray", "clahe", "gamma", "gamma+clahe"]}})
        log(f"take {ti + 1}: {len(views)} views with a board pose ({n_weak} from few corners + "
            f"trajectory prior, {n_direct} without corners by screen-image alignment; "
            f"{n_nodet} frames without board corners, {len(t['recs']) - len(views)} frames unused)")
        takes.append(views)
        for r in t["recs"]:
            r.pop("img", None)
    report["poses"] = pose_stats
    if focal_stats:
        fsv = np.array([s["f_scale"] for s in focal_stats])
        report["focal_breathing"] = {
            "mode": a.focal, "prior_sigma": a.focal_sigma,
            "f_scale_median": float(np.median(fsv)), "f_scale_p05": float(np.percentile(fsv, 5)),
            "f_scale_p95": float(np.percentile(fsv, 95)),
            "rms_fixed_median_px": float(np.median([s["rms_fixed"] for s in focal_stats])),
            "rms_refined_median_px": float(np.median([s["rms_refined"] for s in focal_stats])),
            "depth_shift_mm_p95": float(np.percentile(np.abs([s["dz_mm"] for s in focal_stats]), 95)),
            "per_frame": focal_stats}
        log(f"focal per frame: scale {np.percentile(fsv, 5):.4f}..{np.percentile(fsv, 95):.4f} "
            f"(5-95%), corner RMS fixed {report['focal_breathing']['rms_fixed_median_px']:.2f}px -> "
            f"refined {report['focal_breathing']['rms_refined_median_px']:.2f}px "
            f"({'applied' if a.focal == 'refine' else 'reported only'})")
    return takes


# ============================================================ masks

def _poly_basis(xn, yn):
    return np.stack([np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn], -1)


def _fill_tiles(P, valid, sig=1.2):
    """Fill invalid tile values by normalized Gaussian convolution (growing radius)."""
    P = P.copy()
    w = valid.astype(np.float32)
    for s_ in (sig, 3 * sig, 8 * sig):
        if w.min() > 0:
            break
        num = cv2.GaussianBlur(P * w[..., None], (0, 0), s_)
        den = cv2.GaussianBlur(w, (0, 0), s_)
        fill = (w == 0) & (den > 1e-3)
        P[fill] = num[fill] / den[fill][:, None]
        w = np.where(fill, 1.0, w).astype(np.float32)
    if w.min() == 0:
        P[w == 0] = np.median(P[w > 0], 0) if (w > 0).any() else 0.5
    return P


def _up(T, tile, H, W):
    full = cv2.resize(T, (T.shape[1] * tile, T.shape[0] * tile), interpolation=cv2.INTER_LINEAR)
    return full[:H, :W]


def local_photometric(E, img, usable, tile, iters=3):
    """Per-tile robust fit of what the screen shows: colour = black + (white - black) * E.

    Handles glare and vignetting (tiles of `tile` px). Returns maps black (H,W,3),
    white (H,W,3), noise sigma (H,W,3) and a fit cost."""
    H, W = E.shape
    ty, tx = (H + tile - 1) // tile, (W + tile - 1) // tile
    ys, xs = np.nonzero(usable[::2, ::2])
    ys, xs = ys * 2, xs * 2
    tid = (ys // tile) * tx + (xs // tile)
    e = E[ys, xs].astype(np.float64)
    C = img[ys, xs].astype(np.float64)
    nt = ty * tx
    w = np.ones(len(e))
    blk = np.zeros((nt, 3))
    wht = np.zeros((nt, 3))
    sig = np.full(3, 0.02)
    for it in range(iters):
        sw = np.bincount(tid, w, nt)
        se = np.bincount(tid, w * e, nt)
        see = np.bincount(tid, w * e * e, nt)
        with np.errstate(invalid="ignore", divide="ignore"):
            me = se / sw
            ve = see / sw - me ** 2
        res = np.empty_like(C)
        vb = np.zeros((nt, 3), bool)
        vw = np.zeros((nt, 3), bool)
        for c in range(3):
            sc_ = np.bincount(tid, w * C[:, c], nt)
            sec = np.bincount(tid, w * e * C[:, c], nt)
            with np.errstate(invalid="ignore", divide="ignore"):
                mc = sc_ / sw
                cov = sec / sw - me * mc
                slope = cov / ve
            two = (sw > 40) & (ve > 0.02)
            one = (sw > 40) & ~two
            b = np.where(two, mc - slope * me, np.nan)
            wv = np.where(two, mc + slope * (1 - me), np.nan)
            b = np.where(one & (me < 0.5), mc, b)
            wv = np.where(one & (me >= 0.5), mc, wv)
            blk[:, c], wht[:, c] = np.nan_to_num(b), np.nan_to_num(wv)
            vb[:, c], vw[:, c] = np.isfinite(b), np.isfinite(wv)
            res[:, c] = C[:, c] - (blk[tid, c] + (wht[tid, c] - blk[tid, c]) * e)
            # missing ends of single-sided tiles: approximate from the residual-free side
            bad = ~(vb[tid, c] & vw[tid, c])
            res[bad, c] = 0
        sig = np.maximum(1.4826 * np.median(np.abs(res[w > 0.5]), 0), 0.004)
        u = np.abs(res) / (4.685 * sig)
        w = np.prod(np.where(u < 1, (1 - u ** 2) ** 2, 0), 1) + 1e-6
    vb_all, vw_all = vb.all(1), vw.all(1)
    blkT = _fill_tiles(blk.reshape(ty, tx, 3).astype(np.float32), vb_all.reshape(ty, tx))
    whtT = _fill_tiles(wht.reshape(ty, tx, 3).astype(np.float32), vw_all.reshape(ty, tx))
    # local noise per tile (robust RMS of inliers), at least the global sigma
    r2 = np.bincount(tid, (w > 0.5) * np.sum(res ** 2, 1), nt) / np.maximum(np.bincount(tid, (w > 0.5) * 1.0, nt), 1) / 3
    sT = np.sqrt(r2).reshape(ty, tx)
    sT = np.maximum(sT, np.min(sig))
    okT = (np.bincount(tid, minlength=nt) > 40).reshape(ty, tx)
    sT = _fill_tiles(np.repeat(sT[..., None], 3, 2).astype(np.float32), okT)
    sT = cv2.GaussianBlur(sT, (0, 0), 1.0)
    Sg = np.maximum(_up(sT, tile, H, W), sig.astype(np.float32)[None, None])
    cost = float(np.sum(sig))
    return _up(blkT, tile, H, W), _up(whtT, tile, H, W), Sg, cost


def _score_fg(E, img, known, fit, a):
    """Distance of each pixel's colour to the colours the board can show there
    (segment between the darkest and brightest board value in a 3x3 window), in noise
    units, then hysteresis threshold + cleaning. Returns (fg, score)."""
    blk, wht, Sg, _ = fit
    # floor: phone video is denoised/compressed, so the fitted noise is unrealistically small
    # while the screen shows texture, moire and glare gradients of a few grey levels
    Sg = np.maximum(Sg, getattr(a, "noise_floor", 0.0))
    k = np.ones((3, 3), np.uint8)
    Emin = cv2.erode(E, k)[..., None]
    Emax = cv2.dilate(E, k)[..., None]
    P0 = (blk + (wht - blk) * Emin) / Sg
    P1 = (blk + (wht - blk) * Emax) / Sg
    Cn = img / Sg
    dP = P1 - P0
    tt = np.clip(np.sum((Cn - P0) * dP, -1) / np.maximum(np.sum(dP * dP, -1), 1e-6), 0, 1)
    score = np.linalg.norm(Cn - (P0 + tt[..., None] * dP), axis=-1) / np.sqrt(3)
    cs = getattr(a, "chroma_sigma", 0.0)
    if cs > 0:
        # skin is redder than anything the (neutral white / blue-black) screen can show:
        # chroma (R - B) / brightness against the reddest board colour possible here
        def chroma(c):
            return (c[..., 2] - c[..., 0]) / (c.mean(-1) + 0.04)
        ci = chroma(img)
        ref = known & (score < a.thr_lo) & (Emin[..., 0] > 0.7)
        if ref.sum() < 500:
            ref = known & (Emin[..., 0] > 0.7)
        cw = float(np.median(ci[ref])) if ref.any() else 0.0  # chroma of the screen's white
        sb = float(1.4826 * np.median(np.abs(ci[ref] - cw))) if ref.any() else cs
        # white balance varies between frames: scale so that typical skin of this frame
        # (pixels the photometric test already calls hand) scores well above threshold,
        # but never closer than 3 board-noise units
        core = known & (score > 2 * a.thr_hi)
        if core.sum() > 500:
            ck = float(np.median(ci[core])) - cw
            if ck > 0:
                cs = min(cs, max(ck / (1.5 * a.thr_hi), 3 * sb / a.thr_lo))
        score = np.maximum(score, (ci - cw) / cs)
    score[~known] = 0
    strong = score > a.thr_hi
    weak = score > a.thr_lo
    lab, n = ndimage.label(weak)
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[strong])] = True
    keep[0] = False
    fg = keep[lab]
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    return fg, score


def mask_to_sdf(mask):
    """Signed distance (px) to the hand/background boundary; NaN where unknown."""
    fg, bg = mask == 1, mask == 0
    d_fg = ndimage.distance_transform_edt(~fg)
    d_bg = ndimage.distance_transform_edt(~bg)
    sdf = np.where(fg, -(d_bg - 0.5), d_fg - 0.5).astype(np.float32)
    sdf[mask == 2] = np.nan
    return sdf


def hand_mask(v, screen, a):
    """Mask of view v: 0 background (board seen), 1 hand, 2 unknown. Also v.sdf (px)."""
    H, W = v.gray.shape
    rays = pixel_rays(v.K, v.dist, W, H)
    bx, by, depth = board_hit(rays, v.R, v.t)
    known = screen.known(bx, by) & np.isfinite(depth)
    # texture footprint (screen px per image px) -> prefilter
    mm_per_px = np.nanmedian(depth[known]) / v.K[0, 0] if known.any() else 0.5
    foot = mm_per_px * screen.ppm
    tex = screen.tex_blurred(0.45 * foot)
    sx, sy = screen.to_tex(bx, by)
    E0 = sample_image(tex, sx, sy, border=np.nan)
    known &= np.isfinite(E0)
    E0 = np.where(known, E0, 0.5).astype(np.float32)
    img = v.img.astype(np.float32) / 255.0
    excl = np.zeros((H, W), bool)
    if v.lm is not None:
        hull = cv2.convexHull(v.lm.astype(np.float32)).astype(np.int32)
        cv2.fillConvexPoly(excl.view(np.uint8), hull, 1)
        excl = cv2.dilate(excl.astype(np.uint8), np.ones((5, 5), np.uint8),
                          iterations=int(max(4, 0.03 * W / 2))).astype(bool)
    # pass 1: board appearance fitted away from the MediaPipe hand; pass 2: away from mask 1
    usable = known & ~excl
    if usable.sum() < 0.2 * known.sum():
        usable = known
    best = None
    for sig in a.board_blur:
        E = cv2.GaussianBlur(E0, (0, 0), sig) if sig > 0 else E0
        fit = local_photometric(E, img, usable, a.tile)
        if best is None or fit[3] < best[0]:
            best = (fit[3], sig, E, fit)
    _, sig, E, fit = best
    fg, score = _score_fg(E, img, known, fit, a)
    usable2 = known & ~cv2.dilate(fg.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=8).astype(bool)
    if usable2.sum() > 0.2 * known.sum():
        fit = local_photometric(E, img, usable2, a.tile)
        fg, score = _score_fg(E, img, known, fit, a)
    scale = fit[2]
    # holes: board-coloured patches enclosed by hand (or by hand + unknown border), e.g. glare
    # spots on the palm; up to hole_px or hole_frac of the hand area
    holes = ndimage.binary_fill_holes(fg | ~known) & ~fg & known
    hl, hn = ndimage.label(holes)
    if hn:
        sizes = ndimage.sum(holes, hl, np.arange(1, hn + 1))
        small = np.zeros(hn + 1, bool)
        small[1:] = sizes < max(a.hole_px, a.hole_frac * fg.sum())
        fg |= small[hl]
    # the hand blob(s); other blobs are occluders -> unknown
    lab, n = ndimage.label(fg)
    other = np.zeros((H, W), bool)
    if n:
        areas = ndimage.sum(fg, lab, np.arange(1, n + 1))
        sel = np.zeros(n + 1, bool)
        if v.lm is not None:
            pts = np.round(v.lm).astype(int)
            ok = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
            for (px, py) in pts[ok]:
                y0_, y1_ = max(0, py - 6), min(H, py + 7)
                x0_, x1_ = max(0, px - 6), min(W, px + 7)
                ls = np.unique(lab[y0_:y1_, x0_:x1_])
                sel[ls] = True
        if not sel[1:].any():
            sel[1 + int(np.argmax(areas))] = True
            sel[1:] |= areas >= 0.2 * areas.max()
        sel[0] = False
        hand = sel[lab]
        other = fg & ~hand
        fg = hand
    other = cv2.dilate(other.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~fg
    mask = np.zeros((H, W), np.uint8)
    mask[fg] = 1
    unknown = ~known | other
    mask[unknown & ~fg] = 2
    v.mask, v.sdf = mask, mask_to_sdf(mask)
    v.info.update(board_blur_px=float(sig), noise_median=float(np.median(scale)),
                  known_frac=float(known.mean()), fg_px=int(fg.sum()))


# ============================================================ volumetric hull

class Grid:
    def __init__(self, lo, hi, step):
        self.step = float(step)
        self.lo = np.asarray(lo, float)
        self.shape = tuple(int(np.ceil(x)) + 1 for x in (np.asarray(hi) - self.lo) / step)

    def points(self, idx=None):
        if idx is None:
            ii = np.indices(self.shape).reshape(3, -1).T
        else:
            ii = np.stack(np.unravel_index(idx, self.shape), 1)
        return self.lo + ii * self.step

    def coords(self, X):
        return ((X - self.lo) / self.step).T

    def hi(self):
        return self.lo + (np.array(self.shape) - 1) * self.step


def robust_field(views, X, exts, a, chunk=120_000):
    """k-th largest signed silhouette distance (mm) over the views that know each point."""
    N = len(X)
    out = np.empty(N, np.float32)
    nknown = np.empty(N, np.int16)
    nv = len(views)
    for c0 in range(0, N, chunk):
        P = X[c0:c0 + chunk]
        n = len(P)
        S = np.full((nv, n), -np.inf, np.float32)
        for j, (v, (R, t)) in enumerate(zip(views, exts)):
            u, vv, z = project(P, R, t, v.K, v.dist)
            s = sample_image(v.sdf, u, vv)
            mm = s * z / v.K[0, 0]
            S[j] = np.where(np.isfinite(mm), mm, -np.inf)
        nk = np.isfinite(S).sum(0)
        nbad = np.minimum(a.max_bad, np.floor(a.bad_frac * nk)).astype(int)
        S.sort(axis=0)  # ascending; -inf (unknown) first
        val = S[nv - 1 - nbad, np.arange(n)]
        val[nk < a.min_views] = UNSEEN
        out[c0:c0 + n] = val
        nknown[c0:c0 + n] = nk
    return out, nknown


def field_on_grid(views, exts, grid, a, coarse=None, band=2.5, clip=None):
    """Robust field on a grid; with `coarse` (callable X->value) only a band is evaluated.
    `clip(X)` is a second field intersected with it (e.g. the screen surface)."""
    X = grid.points()
    if coarse is None:
        F, _ = robust_field(views, X, exts, a)
    else:
        F = coarse(X).astype(np.float32)
        idx = np.nonzero(np.abs(F) < band)[0]
        F[idx], _ = robust_field(views, X[idx], exts, a)
    if clip is not None:
        F = np.maximum(F, clip(X).astype(np.float32))
    return F.reshape(grid.shape)


def screen_clip(Ts, off):
    """Nothing can be behind the screen glass: z (board frame of every take) <= -off."""
    def f(X):
        h = X[:, 2] + off
        for T in Ts:
            if T is not None:
                h = np.maximum(h, X @ T[0][2] + T[1][2] + off)
        return h
    return f


def interp_field(F, grid, X, outside=UNSEEN):
    c = grid.coords(X)
    return ndimage.map_coordinates(F, c, order=1, mode="constant", cval=outside)


def main_component(F, keep_pts=None, grid=None):
    occ = F < 0
    lab, n = ndimage.label(occ)
    if n <= 1:
        return F
    sizes = ndimage.sum(occ, lab, np.arange(1, n + 1))
    best = 1 + int(np.argmax(sizes))
    if keep_pts is not None and grid is not None:
        ii = np.round(grid.coords(keep_pts).T).astype(int)
        ok = np.all((ii >= 0) & (ii < np.array(grid.shape)), 1)
        ls = lab[tuple(ii[ok].T)]
        ls = ls[ls > 0]
        if len(ls):
            best = int(np.bincount(ls).argmax())
    F = F.copy()
    F[occ & (lab != best)] = np.abs(F[occ & (lab != best)]) + 0.5
    return F


def mesh_from_field(F, grid):
    import trimesh
    from skimage import measure
    Fp = np.pad(ndimage.gaussian_filter(F, 0.6), 1, constant_values=UNSEEN)
    vts, faces, _, _ = measure.marching_cubes(Fp, 0.0, spacing=(grid.step,) * 3, allow_degenerate=False)
    vts = vts - grid.step + grid.lo
    m = trimesh.Trimesh(vts, faces[:, ::-1], process=True)  # outward normals
    if m.volume < 0:
        m.invert()
    return m


def decimate(mesh, faces):
    """Quadric-collapse decimation with Blender (bpy) to about `faces` triangles."""
    if faces <= 0 or len(mesh.faces) <= faces:
        return mesh
    try:
        import bpy
        import trimesh
    except ImportError:
        log("bpy not available: no decimation")
        return mesh
    me = bpy.data.meshes.new("recon")
    me.from_pydata(np.asarray(mesh.vertices).tolist(), [], np.asarray(mesh.faces).tolist())
    ob = bpy.data.objects.new("recon", me)
    bpy.context.scene.collection.objects.link(ob)
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.ratio = faces / len(mesh.faces)
    mod.use_collapse_triangulate = True
    dg = bpy.context.evaluated_depsgraph_get()
    em = ob.evaluated_get(dg).to_mesh()
    V = np.array([v.co[:] for v in em.vertices])
    em.loop_triangles.update() if hasattr(em.loop_triangles, "update") else em.calc_loop_triangles()
    Fc = np.array([t.vertices[:] for t in em.loop_triangles])
    ob.evaluated_get(dg).to_mesh_clear()
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    out = trimesh.Trimesh(V, Fc, process=True)
    return out if out.is_watertight or not mesh.is_watertight else mesh


def take_hull(views, a, lm3d=None, box=None):
    """Per-take robust hull field on a 1 mm grid (own board frame)."""
    exts = [v.ext() for v in views]
    if box is None:
        if lm3d is not None and np.isfinite(lm3d).all():
            box = (lm3d.min(0) - 60, lm3d.max(0) + 60)
        else:
            bw, bh = views[0].info.get("board_mm", (400, 270))
            box = (np.array([-60, -60, -450.0]), np.array([bw + 60, bh + 60, -3.0]))
    g4 = Grid(box[0], box[1], 4.0)
    clip = screen_clip([None], a.screen_off)
    F4 = field_on_grid(views, exts, g4, a, clip=clip)
    if lm3d is not None:
        F4 = cut_field(F4, g4, lm3d, a.wrist_margin + 25)
    F4 = main_component(F4, lm3d, g4)
    occ = np.argwhere(F4 < 0)
    if len(occ) == 0:
        raise SystemExit("empty hull: masks or poses are wrong")
    lo = g4.lo + occ.min(0) * 4 - 8
    hi = g4.lo + occ.max(0) * 4 + 8
    g2 = Grid(lo, hi, 2.0)
    F2 = field_on_grid(views, exts, g2, a, coarse=lambda X: interp_field(F4, g4, X), band=7, clip=clip)
    g1 = Grid(lo, hi, 1.0)
    F1 = field_on_grid(views, exts, g1, a, coarse=lambda X: interp_field(F2, g2, X), band=3, clip=clip)
    if lm3d is not None:
        F1 = cut_field(F1, g1, lm3d, a.wrist_margin + 25)
    F1 = main_component(F1, lm3d, g1)
    return g1, F1



def drop_bad_masks(views, a, lm3d, report):
    """Masks that cut away part of the hand (glare, skin washed out) carve the hull for
    everyone. Build a tolerant hull, measure per view the fraction of its silhouette the
    view calls background, and drop views far above the others."""
    import copy
    b = copy.copy(a)
    b.bad_frac, b.max_bad = max(a.bad_frac, 0.15), max(a.max_bad, 15)
    exts = [v.ext() for v in views]
    box = (lm3d.min(0) - 60, lm3d.max(0) + 60) if lm3d is not None and np.isfinite(lm3d).all() else None
    if box is None:
        return views
    g = Grid(box[0], box[1], 2.0)
    F = field_on_grid(views, exts, g, b, clip=screen_clip([None], a.screen_off))
    F = cut_field(F, g, lm3d, a.wrist_margin)
    F = main_component(F, lm3d, g)
    try:
        m = mesh_from_field(F, g)
    except Exception:
        return views
    V, Fc = np.asarray(m.vertices), np.asarray(m.faces)
    carve = []
    for v, (R, t) in zip(views, exts):
        H, W = v.mask.shape
        u, vv, z = project(V, R, t, v.K, v.dist)
        fid, *_ = rasterize(u, vv, z, Fc, W, H)
        P = cv2.erode((fid >= 0).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        kn = P & (v.mask != 2)
        carve.append(float(((v.mask == 0) & kn).sum() / max(kn.sum(), 1)))
    carve = np.array(carve)
    med = np.median(carve)
    mad = 1.4826 * np.median(np.abs(carve - med))
    thr = max(a.mask_check, med + 4 * mad)
    keep = carve <= thr
    for v, c, k in zip(views, carve, keep):
        v.info["mask_carve_frac"] = c
        v.info["mask_dropped"] = bool(not k)
    report.setdefault("mask_check", []).append(
        {"take": views[0].take + 1, "threshold": float(thr), "median_carve": float(med),
         "dropped": [int(v.idx) for v, k in zip(views, keep) if not k]})
    log(f"take {views[0].take + 1}: mask check: {int((~keep).sum())} of {len(views)} views cut into the "
        f"hand (> {100 * thr:.1f}% of its outline called background) -> left out")
    return [v for v, k in zip(views, keep) if k]

def cut_field(F, grid, lm3d, margin):
    """Make everything more than `margin` mm proximal of the wrist landmark empty."""
    wrist, mid = lm3d[0], lm3d[9]
    n = mid - wrist
    n /= np.linalg.norm(n)
    X = grid.points()
    d = (X - wrist) @ n + margin  # >0 on the hand side
    return np.maximum(F.ravel(), -d).reshape(F.shape).astype(np.float32)


# ============================================================ landmarks

def triangulate_take(views):
    """21 landmarks (own board frame) by robust triangulation, like relaxed_pose.py."""
    Ps, xs, poses, fs = [], [], [], []
    for v in views:
        if v.lm is None:
            continue
        x = cv2.undistortPoints(v.lm.reshape(-1, 1, 2), v.K, v.dist).reshape(-1, 2)
        Ps.append(np.hstack([v.R, v.t.reshape(3, 1)]))
        xs.append(x)
        poses.append(Pose(cv2.Rodrigues(v.R)[0], v.t.reshape(3, 1), v.rms, v.ncorner))
        fs.append(v.K[0, 0])
    if len(Ps) < 5:
        return None, None
    f = float(np.median(fs))
    W, info = np.zeros((21, 3)), {}
    resid, used, base = [], [], []
    for j in range(21):
        X, keep, rms = robust_point(Ps, [x[j] for x in xs], f, thresh_px=12.0 * f / 2900)
        W[j] = X
        resid.append(rms)
        used.append(int(keep.sum()))
        base.append(baseline_deg(poses, keep))
    info = {"frames_used": len(Ps),
            "reprojection_px": {n: round(r, 2) for n, r in zip(NAMES, resid)},
            "views_per_joint_min": int(min(used)),
            "baseline_deg_min": round(float(min(base)), 1),
            "landmarks_board_mm": {n: [round(float(c), 2) for c in W[i]] for i, n in enumerate(NAMES)}}
    return W, info


# ============================================================ registration

def silhouette_rays(views, F, grid, n_per_view, rng, step=0.75):
    """Rays through pixels just inside each silhouette, sampled where the own hull is."""
    out = []
    for v in views:
        m = v.mask
        unk = cv2.dilate((m == 2).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        sel = (m == 1) & (v.sdf > -3.0) & (v.sdf < -0.5) & ~unk
        ys, xs = np.nonzero(sel)
        if len(xs) < 20:
            continue
        k = rng.choice(len(xs), min(n_per_view, len(xs)), replace=False)
        px = np.stack([xs[k], ys[k]], 1).astype(np.float64)
        nrm = cv2.undistortPoints(px.reshape(-1, 1, 2), v.K, v.dist).reshape(-1, 2)
        d = np.column_stack([nrm, np.ones(len(nrm))]) @ v.R  # board frame
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        C = v.C()
        lo, hi = grid.lo, grid.hi()
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - C) / d
            t2 = (hi - C) / d
        tn = np.nanmax(np.minimum(t1, t2), 1)
        tf = np.nanmin(np.maximum(t1, t2), 1)
        ok = tf > tn
        for i in np.nonzero(ok)[0]:
            ts = np.arange(tn[i], tf[i], step)
            X = C + ts[:, None] * d[i]
            h = interp_field(F, grid, X)
            keep = h < 1.0
            if keep.sum() < 2:
                continue
            out.append((X[keep], h[keep]))
    return out


def pack_rays(rays):
    L = max(len(r[0]) for r in rays)
    n = len(rays)
    X = np.zeros((n, L, 3))
    H = np.full((n, L), np.inf)
    for i, (x, h) in enumerate(rays):
        X[i, :len(x)] = x
        H[i, :len(x)] = h
    return X, H


def register_take(ref, mov, a, rng):
    """Rigid T (ref output frame -> mov take's board frame) by landmarks + silhouette coherence.

    ref/mov: dicts with views, grid, F (hull field, own frame), lm (21x3 or None)."""
    from scipy.optimize import least_squares
    info = {}
    if ref["lm"] is not None and mov["lm"] is not None:
        R0, t0, r = robust_kabsch(ref["lm"], mov["lm"])
        info["landmark_residual_mm"] = {"median": float(np.median(r)), "max": float(np.max(r))}
        info["init"] = "landmarks"
    else:  # no landmarks: hull centroids and principal axes (sign ambiguity -> try 4)
        R0, t0 = None, None
        info["init"] = "hull_pca"
    c_ref = (ref["lm"].mean(0) if ref["lm"] is not None else
             ref["grid"].points(np.nonzero(ref["F"].ravel() < 0)[0]).mean(0))
    rays_ref = pack_rays(silhouette_rays(ref["views"][::a.reg_view_step], ref["F"], ref["grid"],
                                         a.reg_rays, rng))
    rays_mov = pack_rays(silhouette_rays(mov["views"][::a.reg_view_step], mov["F"], mov["grid"],
                                         a.reg_rays, rng))
    if R0 is None:
        def pca(F, g):
            P = g.points(np.nonzero(F.ravel() < 0)[0])
            c = P.mean(0)
            w, V = np.linalg.eigh(np.cov((P - c).T))
            return c, V[:, ::-1]
        ca, Va = pca(ref["F"], ref["grid"])
        cb, Vb = pca(mov["F"], mov["grid"])
        cands = []
        for sx in (1, -1):
            for sy in (1, -1):
                Vb2 = Vb * np.array([sx, sy, sx * sy])
                Rc = Vb2 @ Va.T
                cands.append((Rc, cb - Rc @ ca))
    else:
        cands = [(R0, t0)]

    def T_of(p, Rb, tb):
        Rd = rodrigues(p[:3])
        R = Rd @ Rb
        cm = Rb @ c_ref + tb
        t = Rd @ (tb - cm) + cm + p[3:]
        return R, t

    def resid(p, Rb, tb):
        R, t = T_of(p, Rb, tb)
        Xr, Hr = rays_ref
        Hm = interp_field(mov["F"], mov["grid"], Xr.reshape(-1, 3) @ R.T + t).reshape(Hr.shape)
        r1 = np.min(np.maximum(Hr, Hm), 1)
        Xm, Hmm = rays_mov
        Hrr = interp_field(ref["F"], ref["grid"], (Xm.reshape(-1, 3) - t) @ R).reshape(Hmm.shape)
        r2 = np.min(np.maximum(Hmm, Hrr), 1)
        r = np.maximum(np.concatenate([r1, r2]), 0)
        prior = np.concatenate([p[:3] * a.reg_prior_rad, p[3:] * a.reg_prior_mm])
        return np.concatenate([r, prior])

    best = None
    for Rb, tb in cands:
        r0 = resid(np.zeros(6), Rb, tb)
        sol = least_squares(resid, np.zeros(6), args=(Rb, tb), loss="soft_l1", f_scale=1.0,
                            diff_step=np.r_[[2e-3] * 3, [0.05] * 3] / np.r_[[1] * 3, [1] * 3],
                            x_scale=np.r_[[0.02] * 3, [1.0] * 3], max_nfev=200)
        cost = float(np.mean(sol.fun[:-6] ** 2))
        if best is None or cost < best[0]:
            best = (cost, sol, Rb, tb, float(np.mean(r0[:-6] ** 2)))
    cost, sol, Rb, tb, cost0 = best
    R, t = T_of(sol.x, Rb, tb)
    rr = sol.fun[:-6]
    info.update({
        "rays": int(len(rr)),
        "unexplained_rays_frac_before": float(np.mean(resid(np.zeros(6), Rb, tb)[:-6] > 0.5)),
        "unexplained_rays_frac_after": float(np.mean(rr > 0.5)),
        "rms_before_mm": float(np.sqrt(cost0)), "rms_after_mm": float(np.sqrt(cost)),
        "refine_rotation_deg": float(np.degrees(np.linalg.norm(sol.x[:3]))),
        "refine_translation_mm": float(np.linalg.norm(sol.x[3:])),
        "R": R.tolist(), "t": t.tolist()})
    return (R, t), info


# ============================================================ multi-view stereo refinement

def visibility_maps(mesh, views, exts, shrink=2):
    """Per view depth map of the mesh at 1/shrink processing resolution."""
    out = []
    V, Fc = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    for v, (R, t) in zip(views, exts):
        H, W = v.gray.shape
        K = scaled_K(v.K, 1.0 / shrink)
        u, vv, z = project(V, R, t, K, v.dist)
        _, _, _, zb = rasterize(u, vv, z, Fc, W // shrink, H // shrink)
        out.append((zb, K))
    return out


def visible(X, N, v, ext, vis, tol=1.5, min_cos=0.2):
    R, t = ext
    zb, Ks = vis
    u, vv, z = project(X, R, t, Ks, v.dist)
    ui = np.round(u)
    vi = np.round(vv)
    h, w = zb.shape
    ok = np.isfinite(ui) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    zz = np.full(len(X), -np.inf)
    zz[ok] = zb[vi[ok].astype(int), ui[ok].astype(int)]
    C = -R.T @ t
    vd = C - X
    vd /= np.linalg.norm(vd, axis=1, keepdims=True)
    cos = np.sum(N * vd, 1)
    return ok & (z <= zz + tol + 0.004 * z) & (cos > min_cos), cos


def scaled_K(K, s):
    K = K.copy()
    K[:2] *= s
    K[0, 2] += 0.5 * s - 0.5
    K[1, 2] += 0.5 * s - 0.5
    return K


_CTX = {}


def pmap(fn, items, workers):
    """Parallel map over threads (OpenCV and large numpy operations release the GIL; works
    the same on Windows; forked processes deadlock inside OpenCV after its threads ran)."""
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(i) for i in items]
    from concurrent.futures import ThreadPoolExecutor
    nt = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(fn, items))
    finally:
        cv2.setNumThreads(nt)


def _mask_job(i):
    v = _CTX["views"][i]
    hand_mask(v, _CTX["screen"], _CTX["a"])
    return v.mask, v.sdf, v.info


def masks_parallel(views, screen, a):
    _CTX.update(views=views, screen=screen, a=a)
    res = pmap(_mask_job, range(len(views)), a.workers)
    for v, (m, s, info) in zip(views, res):
        v.mask, v.sdf, v.info = m, s, info


def _box(x, w):
    return cv2.boxFilter(x, -1, (w, w), normalize=True, borderType=cv2.BORDER_REFLECT)


def _depth_job(j):
    """Plane-sweep depth map of reference view j (camera z, mm) at MVS resolution."""
    c = _CTX
    a, views, exts, G, KS = c["a"], c["views"], c["exts"], c["G"], c["KS"]
    v, (R, t), Ks = views[j], exts[j], KS[j]
    I = G[j]
    Hs, Ws = I.shape
    zf, zb = c["zfront"][j], c["zback"][j]
    m = cv2.resize((v.mask == 1).astype(np.uint8), (Ws, Hs), interpolation=cv2.INTER_NEAREST)
    m = cv2.erode(m, np.ones((5, 5), np.uint8)).astype(bool) & np.isfinite(zf)
    nbrs = c["nbrs"][j]
    out = np.full((Hs, Ws), np.nan, np.float32)
    conf = np.zeros((Hs, Ws), np.float32)
    if m.sum() < 50 or len(nbrs) < 2:
        return out, conf
    ys, xs = np.nonzero(m)
    w = a.mvs_window
    pad = w + 2
    y0, y1 = max(0, ys.min() - pad), min(Hs, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(Ws, xs.max() + pad + 1)
    rays = pixel_rays(Ks, v.dist, Ws, Hs)[y0:y1, x0:x1]
    # screen plane along each ray (nothing can be behind the glass)
    vR, vt = v.R, v.t
    _, _, zs = board_hit(rays, vR, vt)
    zs = zs - a.screen_off - 0.5
    mm = m[y0:y1, x0:x1]
    zlo = zf[y0:y1, x0:x1] - a.mvs_margin
    zhi = np.fmin(np.fmin(zb[y0:y1, x0:x1] + a.mvs_margin, zs), zlo + a.mvs_range)
    zlo = np.where(mm, zlo, np.nan)
    Ic = I[y0:y1, x0:x1]
    mI = _box(Ic, w)
    vI = _box(Ic * Ic, w) - mI * mI
    tex_ok = vI > a.mvs_min_var
    Rt = R.T

    def score(z):
        """aggregated NCC (h, w) for per-pixel camera depths z."""
        z = z.astype(np.float32)
        Xc = np.stack([rays[..., 0] * z, rays[..., 1] * z, z], -1).reshape(-1, 3)
        X = (Xc - t.astype(np.float32)) @ R.astype(np.float32)  # == R^T (Xc - t)
        S = []
        for n in nbrs:
            Rn, tn = exts[n]
            u, vv, _ = project(X, Rn, tn, KS[n], views[n].dist)
            J = sample_image(G[n], u.reshape(z.shape), vv.reshape(z.shape))
            # NaN (outside the neighbour's image/crop) would poison OpenCV's running box sums
            bad = _box((~np.isfinite(J)).astype(np.float32), w) > 1e-3
            J = np.nan_to_num(J, nan=0.0).astype(np.float32)
            mJ = _box(J, w)
            vJ = _box(J * J, w) - mJ * mJ
            cIJ = _box(Ic * J, w) - mI * mJ
            with np.errstate(invalid="ignore", divide="ignore"):
                r = cIJ / np.sqrt(np.maximum(vI * vJ, 1e-12))
            r[~(vJ > a.mvs_min_var * 0.25) | bad] = np.nan
            S.append(r)
        S = np.stack(S)
        S = np.where(np.isfinite(S), S, -1.0)
        S = -np.sort(-S, 0)[: max(2, min(a.mvs_best, len(nbrs)))]
        S = S.mean(0)
        if a.mvs_smooth > 0:  # cost aggregation (cheap spatial regularization)
            S = cv2.GaussianBlur(S.astype(np.float32), (0, 0), a.mvs_smooth)
        return S

    nk = int(np.ceil(np.nanmax(zhi - zlo) / a.mvs_step)) + 1 if np.isfinite(np.nanmax(zhi - zlo)) else 0
    if nk <= 0:
        return out, conf
    nk = min(nk, 200)
    best = np.full(zlo.shape, -2.0, np.float32)
    bestk = np.zeros(zlo.shape, np.int32)
    vol = np.full((nk,) + zlo.shape, -2.0, np.float32)
    for k in range(nk):
        z = zlo + k * a.mvs_step
        sc = score(np.nan_to_num(z, nan=1000.0))
        sc[~(z <= zhi)] = -2
        vol[k] = sc
        upd = sc > best
        best[upd] = sc[upd]
        bestk[upd] = k
    second = vol.copy()
    kk = np.arange(nk)[:, None, None]
    second[np.abs(kk - bestk[None]) <= max(1, int(round(1.5 / a.mvs_step)))] = -2
    second = second.max(0)
    zc = zlo + bestk * a.mvs_step
    # fine sweep around the best depth
    offs = np.linspace(-a.mvs_step, a.mvs_step, 9)
    fs = np.stack([score(np.nan_to_num(zc + o, nan=1000.0)) for o in offs])
    fb = np.argmax(fs, 0)
    i1 = np.clip(fb, 1, len(offs) - 2)
    g = lambda i: np.take_along_axis(fs, i[None], 0)[0]
    sm, s0, sp = g(i1 - 1), g(i1), g(i1 + 1)
    den = sm - 2 * s0 + sp
    with np.errstate(invalid="ignore", divide="ignore"):
        off = np.where(den < -1e-6, 0.5 * (sm - sp) / den, 0)
    off = np.clip(np.nan_to_num(off), -0.5, 0.5)
    zbest = zc + offs[i1] + off * (offs[1] - offs[0])
    fbest = fs.max(0)
    good = mm & tex_ok & (fbest > a.mvs_min_ncc) & (fbest - second > a.mvs_min_margin) & np.isfinite(zbest)
    if c.get("debug") is not None:
        c["debug"][j] = dict(mm=mm, tex=tex_ok, fbest=fbest, second=second, zbest=zbest, box=(y0, y1, x0, x1),
                             zlo=zlo, zhi=zhi)
    out[y0:y1, x0:x1] = np.where(good, zbest, np.nan)
    conf[y0:y1, x0:x1] = np.where(good, fbest, 0)
    return out, conf


def choose_neighbours(views, exts, centre, refs, a):
    """Same-take partner views with a useful baseline for every reference view."""
    Cs = np.array([-R.T @ t for R, t in exts])
    dirs = Cs - centre
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    out = {}
    for j in refs:
        ang = np.degrees(np.arccos(np.clip(dirs @ dirs[j], -1, 1)))
        cand = [n for n in range(len(views)) if n != j and views[n].take == views[j].take
                and a.mvs_min_deg <= ang[n] <= a.mvs_max_deg
                and (a.mvs_max_dt_ms <= 0 or abs(views[n].ts - views[j].ts) <= a.mvs_max_dt_ms)]
        cand.sort(key=lambda n: abs(ang[n] - a.mvs_pref_deg))
        out[j] = cand[: a.mvs_neighbours]
    return out


def depth_maps(views, exts, hull, a, refs):
    """Depth maps for reference views (dict j -> (depth, conf)) at MVS resolution."""
    s = a.mvs_scale
    G, KS, ZF, ZB = [], [], [], []
    V, Fc = np.asarray(hull.vertices), np.asarray(hull.faces)
    for v, (R, t) in zip(views, exts):
        H, W = v.gray.shape
        Ws, Hs = int(round(W * s)), int(round(H * s))
        g = cv2.resize(v.gray, (Ws, Hs), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        G.append(g)
        KS.append(scaled_K(v.K, s))
    for j in range(len(views)):
        if j not in refs:
            ZF.append(None)
            ZB.append(None)
            continue
        R, t = exts[j]
        Hs, Ws = G[j].shape
        u, vv, z = project(V, R, t, KS[j], views[j].dist)
        _, _, _, zf = rasterize(u, vv, z, Fc, Ws, Hs)
        _, _, _, zb = rasterize(u, vv, -z, Fc, Ws, Hs)
        ZF.append(np.where(np.isfinite(zf), zf, np.nan).astype(np.float32))
        ZB.append(np.where(np.isfinite(zb), -zb, np.nan).astype(np.float32))
    nb = choose_neighbours(views, exts, hull.centroid, refs, a)
    _CTX.update(a=a, views=views, exts=exts, G=G, KS=KS, zfront=ZF, zback=ZB, nbrs=nb)
    res = pmap(_depth_job, refs, a.workers)
    return {j: r for j, r in zip(refs, res)}, KS


def depth_points(views, exts, KS, dm, j):
    """3D points (output frame) and viewing directions of depth map j."""
    D, conf = dm[j]
    R, t = exts[j]
    Hs, Ws = D.shape
    ys, xs = np.nonzero(np.isfinite(D))
    if len(xs) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), (ys, xs)
    n = cv2.undistortPoints(np.stack([xs, ys], 1).astype(np.float64).reshape(-1, 1, 2),
                            KS[j], views[j].dist).reshape(-1, 2)
    z = D[ys, xs]
    Xc = np.column_stack([n * z[:, None], z])
    X = (Xc - t) @ R
    C = -R.T @ t
    d = C - X
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return X, d, conf[ys, xs], (ys, xs)


def consistency_filter(views, exts, KS, dm, a):
    """Keep depth pixels confirmed by >= a.mvs_consistent other depth maps of the same take."""
    refs = sorted(dm)
    Cs = {j: -exts[j][0].T @ exts[j][1] for j in refs}
    kept = {}
    stats = []
    for j in refs:
        X, d, cf, (ys, xs) = depth_points(views, exts, KS, dm, j)
        if len(X) == 0:
            kept[j] = dm[j]
            continue
        others = [k for k in refs if k != j and views[k].take == views[j].take]
        others.sort(key=lambda k: np.linalg.norm(Cs[k] - Cs[j]))
        agree = np.zeros(len(X), np.int32)
        seen = np.zeros(len(X), np.int32)
        for k in others[: a.mvs_check_views]:
            R, t = exts[k]
            u, vv, z = project(X, R, t, KS[k], views[k].dist)
            Dk = dm[k][0]
            ui, vi = np.round(u), np.round(vv)
            ok = np.isfinite(ui) & (ui >= 0) & (ui < Dk.shape[1]) & (vi >= 0) & (vi < Dk.shape[0])
            dk = np.full(len(X), np.nan)
            dk[ok] = Dk[vi[ok].astype(int), ui[ok].astype(int)]
            has = np.isfinite(dk)
            seen += has
            agree += has & (np.abs(dk - z) < np.maximum(a.mvs_agree_mm, 0.003 * z))
        good = agree >= a.mvs_consistent
        D2 = np.full_like(dm[j][0], np.nan)
        C2 = np.zeros_like(dm[j][1])
        D2[ys[good], xs[good]] = dm[j][0][ys[good], xs[good]]
        C2[ys[good], xs[good]] = dm[j][1][ys[good], xs[good]]
        kept[j] = (D2, C2)
        stats.append((len(X), int(good.sum())))
    tot = np.array(stats) if stats else np.zeros((1, 2))
    return kept, {"depth_pixels": int(tot[:, 0].sum()), "consistent": int(tot[:, 1].sum())}


def take_points(views, exts, KS, dm, take, spacing=1.0):
    P, Dv = [], []
    for j in dm:
        if views[j].take != take:
            continue
        X, d, _, _ = depth_points(views, exts, KS, dm, j)
        P.append(X)
        Dv.append(d)
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3))
    P, Dv = np.concatenate(P), np.concatenate(Dv)
    key = np.round(P / spacing).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return P[first], Dv[first]


def normals_pca(P, view_dirs, k=12):
    tree = cKDTree(P)
    _, nn = tree.query(P, k=min(k, len(P)))
    Q = P[nn] - P[nn].mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", Q, Q)
    w, V = np.linalg.eigh(C)
    N = V[:, :, 0]
    N *= np.sign(np.sum(N * view_dirs, 1))[:, None]  # towards the camera = outwards
    flat = w[:, 0] / np.maximum(w.sum(1), 1e-9)
    return N, flat


def icp_refine(P1, D1, P2, D2, a):
    """Rigid (R, t) moving take-k surface points P2 onto take-1 points P1 (point-to-plane,
    trimmed, normal-compatible: only the overlap - hand sides, fingertips - pairs up)."""
    if len(P1) < 200 or len(P2) < 200:
        return None, {"pairs": 0}
    N1, _ = normals_pca(P1, D1)
    N2, _ = normals_pca(P2, D2)
    tree = cKDTree(P1)
    R, t = np.eye(3), np.zeros(3)
    thr = a.icp_max_mm
    info = {}
    for it in range(30):
        Q = P2 @ R.T + t
        Nq = N2 @ R.T
        d, i = tree.query(Q)
        ok = (d < thr) & (np.sum(Nq * N1[i], 1) > 0.6)
        if ok.sum() < 100:
            return None, {"pairs": int(ok.sum())}
        q, p, n = Q[ok], P1[i[ok]], N1[i[ok]]
        r = np.sum((q - p) * n, 1)
        s = max(0.3, 1.4826 * np.median(np.abs(r)))
        w = 1 / np.maximum(1, np.abs(r) / (1.5 * s))
        A = np.column_stack([np.cross(q, n), n])
        dx = np.linalg.solve((A * w[:, None]).T @ A + 1e-6 * np.eye(6), -(A * w[:, None]).T @ r)
        dR = rodrigues(dx[:3])
        R, t = dR @ R, dR @ t + dx[3:]
        thr = max(1.0, 0.8 * thr)
        info = {"pairs": int(ok.sum()), "residual_mm_median": float(np.median(np.abs(r))), "iterations": it + 1}
        if np.linalg.norm(dx) < 1e-4:
            break
    info["rotation_deg"] = float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))
    info["translation_mm"] = float(np.linalg.norm(t))
    return (R, t), info


def align_to_texture(mesh, views, exts, a, iters=2):
    """Per-view 2D correction of the camera (principal point shift = small rotation) so the
    hand's skin texture lines up across views: the board pose is good to ~1-2 px at the hand,
    but the hand also jitters and drifts against the board. A consensus texture is sampled on
    the current surface (median over views), rendered into each view and aligned by ECC."""
    s = a.mvs_scale
    V, N = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
    Fc = np.asarray(mesh.faces)
    shifts_total = np.zeros((len(views), 2))
    LN = []
    for v in views:
        g = cv2.resize(v.gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        LN.append(_local_norm(g, 4))
    info = {}
    for it in range(iters):
        KS = [scaled_K(v.K, s) for v in views]
        cols = np.full((len(views), len(V)), np.nan, np.float32)
        vis_px = []
        for j, v in enumerate(views):
            R, t = exts[j]
            Hs, Ws = LN[j].shape
            u, vv, z = project(V, R, t, KS[j], v.dist)
            _, _, _, zb = rasterize(u, vv, z, Fc, Ws, Hs)
            ui, vi = np.round(u), np.round(vv)
            ok = np.isfinite(ui) & (ui >= 0) & (ui < Ws) & (vi >= 0) & (vi < Hs)
            zz = np.full(len(V), -np.inf)
            zz[ok] = zb[vi[ok].astype(int), ui[ok].astype(int)]
            C = -R.T @ t
            d = C - V
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            vis = ok & (z <= zz + 1.0) & (np.sum(N * d, 1) > 0.4)
            cols[j, vis] = sample_image(LN[j], u[vis], vv[vis])
            vis_px.append((vis, u, vv, z))
        nv = np.isfinite(cols).sum(0)
        tex = np.nanmedian(np.where(nv[None] >= 3, cols, np.nan), 0) if (nv >= 3).any() else None
        if tex is None:
            break
        shifts = []
        for j, v in enumerate(views):
            vis, u, vv, z = vis_px[j]
            sel = vis & np.isfinite(tex)
            Hs, Ws = LN[j].shape
            T = np.zeros(Hs * Ws, np.float32)
            W_ = np.zeros(Hs * Ws, np.float32)
            idx = (np.round(vv[sel]) * Ws + np.round(u[sel])).astype(int)
            np.add.at(T, idx, tex[sel])
            np.add.at(W_, idx, 1)
            T, W_ = T.reshape(Hs, Ws), W_.reshape(Hs, Ws)
            Tb = cv2.GaussianBlur(T, (0, 0), 0.7)
            Wb = cv2.GaussianBlur(W_, (0, 0), 0.7)
            tmpl = np.where(Wb > 0.05, Tb / np.maximum(Wb, 1e-6), 0).astype(np.float32)
            m = cv2.erode((Wb > 0.3).astype(np.uint8), np.ones((5, 5), np.uint8))
            if m.sum() < 500:
                shifts.append((0.0, 0.0))
                continue
            ys, xs = np.nonzero(m)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                cc, warp = cv2.findTransformECC(tmpl[y0:y1, x0:x1], LN[j][y0:y1, x0:x1], warp,
                                                cv2.MOTION_TRANSLATION,
                                                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4),
                                                m[y0:y1, x0:x1], 3)
            except cv2.error:
                shifts.append((0.0, 0.0))
                continue
            dx, dy = float(warp[0, 2]), float(warp[1, 2])
            if cc < 0.2 or np.hypot(dx, dy) > 6:
                shifts.append((0.0, 0.0))
                continue
            shifts.append((dx, dy))
            v.K[0, 2] += dx / s
            v.K[1, 2] += dy / s
        sh = np.array(shifts) / s
        shifts_total += sh
        info[f"iter{it + 1}_shift_px_median"] = float(np.median(np.hypot(sh[:, 0], sh[:, 1])))
        log(f"texture alignment {it + 1}: view shifts median {info[f'iter{it + 1}_shift_px_median']:.2f}px, "
            f"max {np.hypot(sh[:, 0], sh[:, 1]).max():.2f}px (processing scale)")
    for v, sh in zip(views, shifts_total):
        v.info["texture_shift_px"] = [round(float(sh[0]), 2), round(float(sh[1]), 2)]
    info["total_shift_px_median"] = float(np.median(np.hypot(shifts_total[:, 0], shifts_total[:, 1])))
    return info


def tsdf_field(grid, Fhull, views, exts, KS, dm, a):
    """Truncated signed distance from the depth maps (camera-depth units, + = in front of
    the measured surface = outside), weighted by NCC; NaN where too few maps saw it."""
    F = Fhull.ravel()
    idx = np.nonzero(F < 1.5)[0]
    X = grid.points(idx)
    mu = a.tsdf_mu
    acc = np.zeros(len(idx), np.float32)
    wsum = np.zeros(len(idx), np.float32)
    nsee = np.zeros(len(idx), np.int16)
    for j, (D, conf) in dm.items():
        R, t = exts[j]
        u, vv, z = project(X, R, t, KS[j], views[j].dist)
        ui, vi = np.round(u), np.round(vv)
        ok = np.isfinite(ui) & (ui >= 0) & (ui < D.shape[1]) & (vi >= 0) & (vi < D.shape[0])
        d = np.full(len(X), np.nan, np.float32)
        c = np.zeros(len(X), np.float32)
        d[ok] = D[vi[ok].astype(int), ui[ok].astype(int)]
        c[ok] = conf[vi[ok].astype(int), ui[ok].astype(int)]
        s = d - z
        use = np.isfinite(s) & (s > -mu)
        acc[use] += c[use] * np.minimum(s[use], mu)
        wsum[use] += c[use]
        nsee[use] += 1
    out = np.full(F.shape, np.nan, np.float32)
    good = (nsee >= a.tsdf_min_views) & (wsum > 0)
    out[idx[good]] = acc[good] / wsum[good]
    return out.reshape(grid.shape), int(good.sum())


def depth_from_points(points, views, exts, KS, refs, radius_px=1):
    """Depth maps from an external dense point cloud (COLMAP fused.ply) by z-buffering."""
    dm = {}
    for j in refs:
        R, t = exts[j]
        Hs, Ws = int(round(views[j].gray.shape[0] * KS[j][1, 1] / views[j].K[1, 1])), \
            int(round(views[j].gray.shape[1] * KS[j][0, 0] / views[j].K[0, 0]))
        u, vv, z = project(points, R, t, KS[j], views[j].dist)
        D = np.full(Hs * Ws, np.inf, np.float32)
        for dy in range(-radius_px, radius_px + 1):
            for dx in range(-radius_px, radius_px + 1):
                ui, vi = np.round(u) + dx, np.round(vv) + dy
                ok = np.isfinite(ui) & (ui >= 0) & (ui < Ws) & (vi >= 0) & (vi < Hs)
                np.minimum.at(D, (vi[ok] * Ws + ui[ok]).astype(int), z[ok].astype(np.float32))
        D = D.reshape(Hs, Ws)
        D[~np.isfinite(D)] = np.nan
        dm[j] = (D, np.where(np.isfinite(D), 1.0, 0.0).astype(np.float32))
    return dm


# ============================================================ unobserved side: thickness prior

TEMPLATE_CANDIDATES = ["/mnt/project-files/hand_scan_rig/standin/standin_open.ply"]
LABELS = ["photo", "silhouette", "prior", "screen", "cut"]
LABEL_COLORS = {"photo": (225, 180, 150), "silhouette": (250, 230, 120), "prior": (110, 150, 240),
                "screen": (170, 170, 170), "cut": (120, 120, 120)}


HAND_SIGN = {"scan": 1.0}  # -1 when the filmed hand is a left hand (--hand left)


def hand_axes(W, sign=1.0):
    """origin (wrist), x (pinky->index across), y (wrist->middle MCP), n = sign * x cross y
    (sign +1 for a right hand, -1 for a left hand: n is the palm normal, out of the palm)."""
    o = W[0]
    y = W[9] - W[0]
    y /= np.linalg.norm(y)
    x = W[5] - W[17]
    x -= (x @ y) * y
    x /= np.linalg.norm(x)
    return o, x, y, sign * np.cross(x, y)


def load_template(path):
    import trimesh
    for p in ([path] if path else []) + TEMPLATE_CANDIDATES:
        if p and os.path.exists(p):
            j = os.path.join(os.path.dirname(p), "joints.json")
            if not os.path.exists(j):
                continue
            L = json.load(open(j))["landmarks"]
            return trimesh.load(p, process=False), np.array([L[n] for n in NAMES], float), p
    return None, None, None


def height_maps(mesh, o, x, y, n, res=1.0, pad=4):
    """Top (max h) and bottom (min h) of a mesh over the (u, v) palm plane."""
    V = np.asarray(mesh.vertices) - o
    u, v, h = V @ x, V @ y, V @ n
    u0, v0 = u.min() - pad, v.min() - pad
    Wd, Hd = int((u.max() - u0 + pad) / res) + 1, int((v.max() - v0 + pad) / res) + 1
    pu, pv = (u - u0) / res, (v - v0) / res
    Fc = np.asarray(mesh.faces)
    _, _, _, top = rasterize(pu, pv, 1000 - h, Fc, Wd, Hd)
    _, _, _, bot = rasterize(pu, pv, 1000 + h, Fc, Wd, Hd)
    top = np.where(np.isfinite(top), 1000 - top, np.nan)
    bot = np.where(np.isfinite(bot), bot - 1000, np.nan)
    return top, bot, (u0, v0, res)


def _fill_nan(M, sig):
    w = np.isfinite(M).astype(np.float32)
    num = cv2.GaussianBlur(np.nan_to_num(M).astype(np.float32), (0, 0), sig)
    den = cv2.GaussianBlur(w, (0, 0), sig)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.05, num / den, np.nan)


def thickness_prior(grid, F, lm, exts, a, tpl_path=None):
    """Back surface for a hand seen from one side only: observed front height minus the
    template's thickness (stand-in hand from Olex's measurements), mapped by a thin-plate
    spline through the 21 landmarks, smoothed. Returns (field, info) or (None, info)."""
    from scipy.interpolate import RBFInterpolator
    info = {}
    tm, TW, tp = load_template(tpl_path)
    if tm is None:
        info["skipped"] = "no template (pass --template standin_open.ply with joints.json next to it)"
        return None, info
    o, x, y, n = hand_axes(lm, HAND_SIGN["scan"])
    Cs = np.array([-R.T @ t for R, t in exts])
    side = np.sign((Cs - lm.mean(0)) @ n)
    frac = float(np.mean(side > 0))
    info.update(template=tp, cameras_on_palm_side=frac)
    if 0.1 < frac < 0.9:
        info["skipped"] = "both sides of the hand were filmed"
        return None, info
    s = 1.0 if frac >= 0.9 else -1.0  # +1: palm side observed, back unobserved
    to, tx, ty, tn = hand_axes(TW)
    top, bot, (tu0, tv0, tres) = height_maps(tm, to, tx, ty, tn)
    Tt = top - bot
    # observed (u, v) -> template (u, v) through the landmarks (thin-plate spline)
    uv_o = np.column_stack([(lm - o) @ x, (lm - o) @ y])
    uv_t = np.column_stack([(TW - to) @ tx, (TW - to) @ ty])
    tps = RBFInterpolator(uv_o, uv_t, kernel="thin_plate_spline", smoothing=2.0)
    scale = np.linalg.norm(lm[9] - lm[0]) / np.linalg.norm(TW[9] - TW[0])
    # observed front heights per column, from the reconstruction so far
    occ = grid.points(np.nonzero(F.ravel() < 0)[0])
    if len(occ) == 0:
        return None, info
    P = occ - o
    uu, vv, hh = P @ x, P @ y, P @ n
    res = 1.0
    u0, v0 = uu.min() - 3, vv.min() - 3
    Wd, Hd = int((uu.max() - u0 + 3) / res) + 1, int((vv.max() - v0 + 3) / res) + 1
    gu, gv = np.meshgrid(u0 + np.arange(Wd) * res, v0 + np.arange(Hd) * res)
    hs = np.arange(hh.min() - 2, hh.max() + 2, 0.5)
    hf = np.full(gu.shape, np.nan)
    order = hs[::-1] if s > 0 else hs  # from the observed side inwards
    found = np.zeros(gu.shape, bool)
    for hval in order:
        X = o + gu[..., None] * x + gv[..., None] * y + hval * n
        f = interp_field(F, grid, X.reshape(-1, 3)).reshape(gu.shape)
        new = (f < 0) & ~found
        hf[new] = hval
        found |= new
    # template thickness at the mapped columns
    m = np.isfinite(hf)
    T = np.full(gu.shape, np.nan)
    tuv = tps(np.column_stack([gu[m], gv[m]]))
    ti = (tuv[:, 0] - tu0) / tres
    tj = (tuv[:, 1] - tv0) / tres
    Tt_f = np.where(np.isfinite(Tt), Tt, 0).astype(np.float32)
    T[m] = ndimage.map_coordinates(Tt_f, [tj, ti], order=1, mode="constant", cval=0.0) * scale
    T = _fill_nan(np.where(m, T, np.nan), 1.5)
    hb = hf - s * T
    hb = np.where(m, _fill_nan(hb, 1.0), np.nan)
    # field on the grid: outside beyond the back surface
    X = grid.points()
    P = X - o
    pu, pv, ph = (P @ x - u0) / res, (P @ y - v0) / res, P @ n
    hbg = ndimage.map_coordinates(np.nan_to_num(hb, nan=np.nan), [pv, pu], order=1, mode="constant", cval=np.nan)
    Fp = (s * (hbg - ph)).astype(np.float32)
    info.update(observed_side="palm" if s > 0 else "back", thickness_scale=float(scale),
                thickness_median_mm=float(np.nanmedian(T[m])), columns=int(m.sum()))
    return Fp.reshape(grid.shape), info


def hand_regions(V, N, lm):
    """fingers / thumb / palm (palm side) / back (dorsum) / palm_edge / wrist per vertex."""
    o, x, y, n = hand_axes(lm, HAND_SIGN["scan"])
    W = lm
    segs, kinds = [], []
    for f in ["index", "middle", "ring", "pinky"]:
        i0 = NAMES.index(f + "_mcp")
        for i in range(3):
            segs.append((W[i0 + i], W[i0 + i + 1]))
            kinds.append("fingers")
        segs.append((W[0], W[i0]))
        kinds.append("palm")
    for a_, b_, k in [(2, 3, "thumb"), (3, 4, "thumb"), (1, 2, "palm"), (0, 1, "palm")]:
        segs.append((W[a_], W[b_]))
        kinds.append(k)
    D = []
    for a_, b_ in segs:
        ab = b_ - a_
        t = np.clip((V - a_) @ ab / (ab @ ab), 0, 1)
        D.append(np.linalg.norm(V - a_ - t[:, None] * ab, axis=1))
    kind = np.array(kinds, object)[np.argmin(np.array(D), 0)]
    out = kind.copy()
    palm = kind == "palm"
    cn = N @ n
    out[palm & (cn > 0.3)] = "palm"
    out[palm & (cn < -0.3)] = "back"
    out[palm & (np.abs(cn) <= 0.3)] = "palm_edge"
    out[(V - o) @ y < 5] = "wrist"
    return out


def coverage_labels(mesh, grid, Fj, Fprior, depth_pts, Ts, a, cut=None):
    """What determines each output vertex: photo (depth maps), silhouette (hull),
    prior (thickness prior), screen (glass), cut (wrist cap)."""
    V = np.asarray(mesh.vertices)
    lab = np.full(len(V), "silhouette", object)
    if cut is not None:
        p0, nrm = cut
        lab[np.abs((V - p0) @ nrm) < 0.6] = "cut"
    scr = screen_clip(Ts, a.screen_off)(V)
    lab[np.abs(scr) < 0.6] = "screen"
    if Fprior is not None:
        fp = interp_field(np.nan_to_num(Fprior, nan=99.0), grid, V, outside=99.0)
        fh = interp_field(Fj, grid, V)
        lab[(np.abs(fp) < 0.6) & (fh < -0.3)] = "prior"
    if depth_pts is not None and len(depth_pts):
        d, _ = cKDTree(depth_pts).query(V)
        lab[d < a.obs_mm] = "photo"
    return lab


# ============================================================ report helpers

def mask_iou(mesh, views, exts, lm_wrist_mid=None, shrink=2):
    out = []
    V, Fc = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    for v, (R, t) in zip(views, exts):
        H, W = v.gray.shape
        K = v.K.copy()
        K[:2] /= shrink
        K[0, 2] += 0.5 / shrink - 0.5
        K[1, 2] += 0.5 / shrink - 0.5
        u, vv, z = project(V, R, t, K, v.dist)
        fid, _, _, _ = rasterize(u, vv, z, Fc, W // shrink, H // shrink)
        rend = fid >= 0
        m = cv2.resize(v.mask, (W // shrink, H // shrink), interpolation=cv2.INTER_NEAREST)
        roi = cv2.dilate(rend.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=6).astype(bool)
        roi &= m != 2
        if lm_wrist_mid is not None:  # drop the forearm side of the wrist cut plane
            p0, nrm = lm_wrist_mid
            ys, xs = np.nonzero(roi)
            uw, vw, _ = project(np.array([p0, p0 + nrm * 30]), R, t, K, v.dist)
            dvec = np.array([uw[1] - uw[0], vw[1] - vw[0]])
            if np.all(np.isfinite(dvec)):
                side = (xs - uw[0]) * dvec[0] + (ys - vw[0]) * dvec[1]
                roi[ys[side < 0], xs[side < 0]] = False
        fg = (m == 1) & roi
        rd = rend & roi
        inter, union = (fg & rd).sum(), (fg | rd).sum()
        out.append(float(inter / union) if union else float("nan"))
    return out


def views_per_vertex(mesh, views, exts):
    V = np.asarray(mesh.vertices)
    N = np.asarray(mesh.vertex_normals)
    vis = visibility_maps(mesh, views, exts, shrink=4)
    cnt = np.zeros(len(V), np.int32)
    for j, v in enumerate(views):
        ok, _ = visible(V, N, v, exts[j], vis[j], tol=2.0, min_cos=0.1)
        cnt += ok
    return cnt


# ============================================================ COLMAP export

def rot_to_quat(R):
    q = np.empty(4)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q[:] = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q) * np.sign(q[0] if q[0] != 0 else 1)


def write_colmap(views, exts, out_dir, depth_range, points=None):
    """COLMAP project with the known poses (world = output board frame, mm).

    Images are written undistorted (PINHOLE), so image_undistorter only builds the
    workspace. Commands (COLMAP >= 3.8 with CUDA), from out_dir:
      colmap image_undistorter --image_path images --input_path sparse --output_path dense --output_type COLMAP
      colmap patch_match_stereo --workspace_path dense --workspace_format COLMAP \
          --PatchMatchStereo.depth_min D0 --PatchMatchStereo.depth_max D1 \
          --PatchMatchStereo.window_radius 7 --PatchMatchStereo.geom_consistency true \
          --PatchMatchStereo.filter_min_ncc 0.2
      colmap stereo_fusion --workspace_path dense --workspace_format COLMAP --input_type geometric \
          --output_path dense/fused.ply --StereoFusion.min_num_pixels 3
    then rerun recon_board.py with --colmap-fused out_dir/dense/fused.ply.
    """
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "sparse"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    cams, imgs = [], []
    for i, (v, (R, t)) in enumerate(zip(views, exts)):
        name = f"t{v.take + 1}_{v.idx:04d}.jpg"
        und = cv2.undistort(v.img, v.K, v.dist, None, v.K)
        cv2.imwrite(os.path.join(out_dir, "images", name), und, [cv2.IMWRITE_JPEG_QUALITY, 95])
        mk = cv2.undistort((v.mask == 1).astype(np.uint8) * 255, v.K, v.dist, None, v.K)
        cv2.imwrite(os.path.join(out_dir, "masks", name + ".png"), mk)
        H, W = v.gray.shape
        cams.append(f"{i + 1} PINHOLE {W} {H} {v.K[0, 0]:.6f} {v.K[1, 1]:.6f} {v.K[0, 2]:.6f} {v.K[1, 2]:.6f}")
        q = rot_to_quat(R)
        imgs.append(f"{i + 1} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                    f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {i + 1} {name}\n")
    with open(os.path.join(out_dir, "sparse", "cameras.txt"), "w") as f:
        f.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n" + "\n".join(cams) + "\n")
    with open(os.path.join(out_dir, "sparse", "images.txt"), "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] (empty)\n" + "\n".join(imgs))
    with open(os.path.join(out_dir, "sparse", "points3D.txt"), "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        if points is not None:
            for k, p in enumerate(points):
                f.write(f"{k + 1} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} 200 150 130 0.5\n")
    d0, d1 = depth_range
    cmd = f"""# run from {out_dir} (COLMAP with CUDA). World units = mm (board frame of take 1).
colmap image_undistorter --image_path images --input_path sparse --output_path dense --output_type COLMAP
colmap patch_match_stereo --workspace_path dense --workspace_format COLMAP --PatchMatchStereo.depth_min {d0:.0f} --PatchMatchStereo.depth_max {d1:.0f} --PatchMatchStereo.window_radius 7 --PatchMatchStereo.geom_consistency true --PatchMatchStereo.filter_min_ncc 0.2
colmap stereo_fusion --workspace_path dense --workspace_format COLMAP --input_type geometric --output_path dense/fused.ply --StereoFusion.min_num_pixels 3
# masks/ holds the hand masks (undistorted, name.jpg.png) if you want --StereoFusion.mask_path masks
# then: python recon_board.py ... --colmap-fused {os.path.join(out_dir, 'dense', 'fused.ply')}
"""
    with open(os.path.join(out_dir, "commands.sh"), "w") as f:
        f.write(cmd)
    return cmd



# ============================================================ compact package (PC -> cloud)

def export_package(takes, out_dir, a, report, budget_mb=80.0):
    """Per-frame camera (intrinsics adjusted to the crop, board pose) + hand-cropped JPEGs
    and masks, so the reconstruction can be rerun elsewhere (--package DIR)."""
    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
    frames, crops = [], []
    for views in takes:
        for v in views:
            H, W = v.gray.shape
            ys, xs = np.nonzero(v.mask == 1)
            if len(xs) == 0:
                continue
            bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
            mx, my = int(max(40, 0.3 * bw)), int(max(40, 0.3 * bh))
            x0, x1 = max(0, xs.min() - mx), min(W, xs.max() + mx + 1)
            y0, y1 = max(0, ys.min() - my), min(H, ys.max() + my + 1)
            crops.append((v, x0, y0, x1, y1))
    for q in (92, 85, 78, 70, 60, 50):
        total = 0
        for v, x0, y0, x1, y1 in crops:
            ok, buf = cv2.imencode(".jpg", v.img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, q])
            total += len(buf)
        if total < budget_mb * 0.9e6:
            break
    for v, x0, y0, x1, y1 in crops:
        name = f"t{v.take + 1}_{v.idx:04d}"
        cv2.imwrite(os.path.join(out_dir, "frames", name + ".jpg"), v.img[y0:y1, x0:x1],
                    [cv2.IMWRITE_JPEG_QUALITY, q])
        cv2.imwrite(os.path.join(out_dir, "frames", name + "_mask.png"), v.mask[y0:y1, x0:x1])
        K = v.K.copy()
        K[0, 2] -= x0
        K[1, 2] -= y0
        frames.append({"name": name, "take": v.take, "frame": v.idx, "ts_ms": v.ts, "sharpness": v.sharp,
                       "crop_xywh_processing_px": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                       "K": K.tolist(), "dist": np.ravel(v.dist).tolist(),
                       "R": v.R.tolist(), "t": np.ravel(v.t).tolist(), "f_scale": v.f_scale,
                       "board_rms_px": None if not np.isfinite(v.rms) else v.rms, "corners": v.ncorner,
                       "pose_source": v.info.get("pose_source"),
                       "landmarks_px": None if v.lm is None else (v.lm - [x0, y0]).tolist()})
    meta = {"format": "recon_board package v1", "units": "mm, px of the cropped JPEGs",
            "pose": "x_cam = R X_board + t (OpenCV), board frame of gripscan.board",
            "mask": "0 board background, 1 hand, 2 unknown", "processing_scale": a.scale,
            "jpeg_quality": q, "board_spec": json.load(open(a.board_spec)),
            "inputs": report.get("inputs"), "calibration": report.get("calibration"),
            "poses": report.get("poses"), "frames": frames}
    with open(os.path.join(out_dir, "package.json"), "w") as f:
        json.dump(meta, f, indent=1)
    size = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fs in os.walk(out_dir) for fn in fs)
    log(f"package: {len(frames)} frames, JPEG quality {q}, {size / 1e6:.1f} MB -> {out_dir}")
    return size


def load_package(pkg_dir, use_masks=False):
    meta = json.load(open(os.path.join(pkg_dir, "package.json")))
    takes = {}
    for fr in meta["frames"]:
        v = View()
        v.take, v.idx, v.ts, v.sharp, v.path = fr["take"], fr["frame"], fr["ts_ms"], fr["sharpness"], pkg_dir
        v.img = cv2.imread(os.path.join(pkg_dir, "frames", fr["name"] + ".jpg"))
        v.gray = cv2.cvtColor(v.img, cv2.COLOR_BGR2GRAY)
        v.K, v.dist = np.array(fr["K"]), np.array(fr["dist"])
        v.R, v.t = np.array(fr["R"]), np.array(fr["t"])
        v.f_scale, v.rms, v.ncorner = fr["f_scale"], fr["board_rms_px"] or float("nan"), fr["corners"]
        v.lm = None if fr["landmarks_px"] is None else np.array(fr["landmarks_px"])
        v.info = {"pose_source": fr.get("pose_source"), "weak_pose": fr.get("pose_source") != "corners"}
        v.mask = v.sdf = None
        if use_masks:
            v.mask = cv2.imread(os.path.join(pkg_dir, "frames", fr["name"] + "_mask.png"), cv2.IMREAD_GRAYSCALE)
            v.sdf = mask_to_sdf(v.mask)
            v.info["fg_px"] = int((v.mask == 1).sum())
        takes.setdefault(v.take, []).append(v)
    return [takes[k] for k in sorted(takes)], meta

# ============================================================ main

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", nargs="+", default=[], help="take videos; first = reference take")
    p.add_argument("--package", default=None, help="input: a package written by --export-package (instead of --videos)")
    p.add_argument("--scan-pkg", default=None, help="input: make_pkg.py package (frames.json + crops/) instead of --videos")
    p.add_argument("--package-masks", action="store_true", help="with --package: use its masks instead of recomputing")
    p.add_argument("--export-package", default=None, help="write a compact package (poses + hand crops) to this folder")
    p.add_argument("--package-mb", type=float, default=80.0)
    p.add_argument("--package-only", action="store_true", help="stop after writing the package")
    p.add_argument("--board-spec", required=True)
    p.add_argument("--camera", default=None, help="camera.json (fallback lens, and comparison)")
    p.add_argument("--out", required=True, help="output mesh .ply (other outputs next to it)")
    p.add_argument("--fps", type=float, default=6.0, help="frames kept per second of video")
    p.add_argument("--scale", type=float, default=0.5, help="processing scale of the frames")
    p.add_argument("--blur-frac", type=float, default=0.35, help="drop frames sharper than this x median")
    p.add_argument("--focal", choices=["fixed", "refine", "report"], default="report",
                   help="per-frame focal refinement (autofocus breathing)")
    p.add_argument("--focal-sigma", type=float, default=0.01)
    p.add_argument("--weak-poses", choices=["use", "skip"], default="use",
                   help="frames with 6-11 board corners: pose with a trajectory prior")
    p.add_argument("--prior-gap-ms", type=float, default=600)
    p.add_argument("--no-direct-poses", dest="direct_poses", action="store_false",
                   help="do not pose frames without board corners by screen-image alignment")
    p.add_argument("--prior-mm", type=float, default=30.0)
    p.add_argument("--prior-deg", type=float, default=8.0)
    p.add_argument("--mp-model", default=None)
    p.add_argument("--known", choices=["screen", "board"], default="screen",
                   help="background known on the whole screen image (screen_board layout) or only the board")
    p.add_argument("--screen-inset", type=float, default=10.0, help="mm of the screen edge treated as unknown")
    p.add_argument("--known-margin", type=float, default=8.0, help="with --known board: mm of white screen around the board")
    p.add_argument("--board-blur", type=float, nargs="+", default=[0.0, 0.7, 1.2, 2.0, 3.0])
    p.add_argument("--thr-hi", type=float, default=7.0)
    p.add_argument("--thr-lo", type=float, default=4.0)
    p.add_argument("--hole-px", type=int, default=250)
    p.add_argument("--hole-frac", type=float, default=0.02, help="fill enclosed holes up to this fraction of the hand area")
    p.add_argument("--tile", type=int, default=48, help="px tiles of the local board photometric model")
    p.add_argument("--bad-frac", type=float, default=0.05, help="fraction of views allowed to disagree per voxel")
    p.add_argument("--max-bad", type=int, default=3)
    p.add_argument("--min-views", type=int, default=3)
    p.add_argument("--voxel", type=float, default=0.5)
    p.add_argument("--target-faces", type=int, default=300000, help="decimate the output (0: keep)")
    p.add_argument("--noise-floor", type=float, default=0.03,
                   help="minimum colour noise (0..1) used for the hand/board decision")
    p.add_argument("--chroma-sigma", type=float, default=0.025,
                   help="skin-redness evidence: (R-B)/brightness above the screen's, in these units "
                        "(0 = off)")
    p.add_argument("--mvs-max-dt-ms", type=float, default=2500,
                   help="only pair views filmed within this time (hand moves during long scans); 0 = any")
    p.add_argument("--mask-check", type=float, default=0.0,
                   help="leave out views whose mask calls more than this fraction (or median+4 MAD) of a "
                        "tolerant hull's outline background; 0 = off")
    p.add_argument("--time-window", type=lambda x: tuple(float(y) for y in x.split(",")), default=None,
                   help="T0,T1: use only frames between these video times (s); the hand changes "
                        "posture slightly during long scans")
    p.add_argument("--hand", choices=["right", "left"], default="right",
                   help="which hand was filmed (decides which side is the palm; a left hand is also "
                        "written mirrored as <out>_as_right.ply for the right-hand rig)")
    p.add_argument("--prior", choices=["auto", "off"], default="auto",
                   help="thickness prior for the side no camera saw (auto: when only one side was filmed)")
    p.add_argument("--template", default=None, help="template hand mesh (joints.json next to it), default the stand-in")
    p.add_argument("--obs-mm", type=float, default=1.0, help="vertex counts as photo-observed within this of a depth point")
    p.add_argument("--wrist-margin", type=float, default=10.0, help="mm of forearm kept proximal of the wrist landmark")
    p.add_argument("--reg-rays", type=int, default=250)
    p.add_argument("--reg-view-step", type=int, default=2)
    p.add_argument("--reg-prior-mm", type=float, default=0.03)
    p.add_argument("--reg-prior-rad", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4, help="processes for masks and depth maps")
    p.add_argument("--no-photo", action="store_true", help="hull only (no depth maps)")
    p.add_argument("--mvs-scale", type=float, default=0.5, help="depth map resolution vs processing frames")
    p.add_argument("--mvs-ref-step", type=int, default=2, help="every n-th view is a depth reference")
    p.add_argument("--mvs-window", type=int, default=7)
    p.add_argument("--mvs-step", type=float, default=1.5, help="mm depth step of the coarse sweep")
    p.add_argument("--mvs-range", type=float, default=60.0)
    p.add_argument("--mvs-margin", type=float, default=3.0)
    p.add_argument("--mvs-neighbours", type=int, default=6)
    p.add_argument("--mvs-best", type=int, default=3, help="average of the best n neighbour NCCs")
    p.add_argument("--mvs-smooth", type=float, default=0.0, help="px sigma of cost aggregation")
    p.add_argument("--mvs-min-deg", type=float, default=3.0)
    p.add_argument("--mvs-max-deg", type=float, default=40.0)
    p.add_argument("--mvs-pref-deg", type=float, default=15.0)
    p.add_argument("--mvs-min-ncc", type=float, default=0.5)
    p.add_argument("--mvs-min-margin", type=float, default=0.05)
    p.add_argument("--mvs-min-var", type=float, default=2e-5)
    p.add_argument("--mvs-check-views", type=int, default=8)
    p.add_argument("--mvs-consistent", type=int, default=2)
    p.add_argument("--mvs-agree-mm", type=float, default=1.0)
    p.add_argument("--icp-max-mm", type=float, default=3.0)
    p.add_argument("--align-iters", type=int, default=0, help="texture alignment rounds (experimental; made the synthetic test worse)")
    p.add_argument("--tsdf-mu", type=float, default=3.0)
    p.add_argument("--tsdf-voxel", type=float, default=1.0)
    p.add_argument("--tsdf-min-views", type=int, default=2)
    p.add_argument("--tsdf-smooth", type=float, default=1.5, help="mm Gaussian on the fused depth field")
    p.add_argument("--colmap", default=None, help="write a COLMAP project with known poses here")
    p.add_argument("--colmap-fused", default=None, help="COLMAP fused.ply to use instead of NCC depths")
    p.add_argument("--debug-dir", default=None, help="write mask overlays here")
    p.add_argument("--save-state", default=None, help="npz with fields for later inspection")
    p.add_argument("--cache", default=None, help="pickle of frames/poses/masks (reused if it exists)")
    return p.parse_args(argv)


def run(a):
    import trimesh
    t_start = time.time()
    paths = [f for g in a.videos for f in (sorted(glob.glob(g)) or [g])]
    spec = BoardSpec.load(a.board_spec)
    screen = ScreenModel(spec, a.known_margin, a.known, a.screen_inset)
    a.screen_off = float(spec.surface_offset_mm)
    HAND_SIGN["scan"] = -1.0 if a.hand == "left" else 1.0
    report = {"inputs": paths, "board_spec": a.board_spec, "args": vars(a)}
    if not paths and not a.package and not a.scan_pkg:
        raise SystemExit("give --videos or --package")
    import pickle
    if a.cache and os.path.exists(a.cache):  # frames, poses and masks of an earlier run
        takes, rep0 = pickle.load(open(a.cache, "rb"))
        report.update(rep0)
        log(f"loaded {sum(map(len, takes))} views from {a.cache}")
    else:
        if a.package:  # frames, poses (and masks) from a package written by --export-package
            takes, meta = load_package(a.package, a.package_masks)
            report.update({k: meta.get(k) for k in ("calibration", "poses")})
            log(f"package {a.package}: {sum(map(len, takes))} frames")
            if not a.package_masks:
                masks_parallel([v for views in takes for v in views], screen, a)
        else:
            if a.scan_pkg:  # board corners + landmarks + hand crops from Olex's PC (make_pkg.py)
                raw, meta = load_scan_pkg(a.scan_pkg, spec, a)
                report["inputs"] = [a.scan_pkg]
            else:
                raw = load_takes(paths, spec, a)
            takes = build_views(raw, a, report, screen)
            del raw
            masks_parallel([v for views in takes for v in views], screen, a)
        if a.cache:
            pickle.dump((takes, {k: report[k] for k in ("calibration", "poses", "focal_breathing", "inputs")
                                 if k in report}), open(a.cache, "wb"), protocol=4)
    for views in takes:
        for v in views:
            v.info["board_mm"] = screen.board_mm
        for v in list(views):
            if v.info.get("fg_px", 0) < 500:
                views.remove(v)
            elif a.time_window and not (a.time_window[0] * 1000 <= v.ts <= a.time_window[1] * 1000):
                views.remove(v)
    log("masks done:", [len(v) for v in takes], "views"
        + (f" (time window {a.time_window[0]}-{a.time_window[1]} s)" if a.time_window else ""))
    if a.export_package:
        report["package_bytes"] = export_package(takes, a.export_package, a, report, a.package_mb)
        if a.package_only:
            return report
    if a.debug_dir:
        os.makedirs(a.debug_dir, exist_ok=True)
        for views in takes:
            for v in views[::max(1, len(views) // 6)]:
                ov = v.img.copy()
                ov[v.mask == 1] = (0.5 * ov[v.mask == 1] + [0, 0, 127]).astype(np.uint8)
                ov[v.mask == 2] = (0.5 * ov[v.mask == 2]).astype(np.uint8)
                cv2.imwrite(os.path.join(a.debug_dir, f"mask_t{v.take + 1}_{v.idx:03d}.jpg"), ov)
    # per take: landmarks + hull
    per = []
    lm_report = {"units": "mm", "frame": "board frame of each take (gripscan.board)", "takes": []}
    for ti, views in enumerate(takes):
        lm, info = triangulate_take(views)
        if lm is not None:
            log(f"take {ti + 1}: landmarks triangulated from {info['frames_used']} views, "
                f"reprojection median {np.median(list(info['reprojection_px'].values())):.1f}px")
        if a.mask_check > 0 and len(views) >= 12:
            views = drop_bad_masks(views, a, lm, report)
            takes[ti] = views
        g, F = take_hull(views, a, lm)
        log(f"take {ti + 1}: hull grid {g.shape}, volume {(F < 0).sum() * g.step ** 3 / 1000:.0f} ml")
        per.append({"views": views, "grid": g, "F": F, "lm": lm})
        lm_report["takes"].append({"input": (report.get("inputs") or paths or ["?"])[min(ti, len(report.get("inputs") or paths or ["?"]) - 1)], **(info or {"landmarks": None})})
    # registration into take 1
    rng = np.random.default_rng(0)
    Ts = [None]
    report["registration"] = []
    for k in range(1, len(per)):
        T, info = register_take(per[0], per[k], a, rng)
        Ts.append(T)
        report["registration"].append({"take": k + 1, **info})
        log(f"take {k + 1} -> take 1: init {info['init']}, unexplained silhouette rays "
            f"{info['unexplained_rays_frac_before']:.3f} -> {info['unexplained_rays_frac_after']:.3f}, "
            f"refinement {info['refine_rotation_deg']:.2f} deg / {info['refine_translation_mm']:.2f} mm")
    lm1 = per[0]["lm"]
    if lm1 is None:  # take 1 without landmarks: any take's landmarks mapped back
        for k in range(1, len(per)):
            if per[k]["lm"] is not None:
                R, t = Ts[k]
                lm1 = (per[k]["lm"] - t) @ R
                break
    def coarse_joint(X):
        h = interp_field(per[0]["F"], per[0]["grid"], X)
        for k in range(1, len(per)):
            R, t = Ts[k]
            h = np.maximum(h, interp_field(per[k]["F"], per[k]["grid"], X @ R.T + t))
        return h

    def gather():
        V, E = [], []
        for k, p in enumerate(per):
            for v in p["views"]:
                V.append(v)
                E.append(v.ext(Ts[k]))
        return V, E

    def coarse_hull_mesh():
        g1 = per[0]["grid"]
        Fc = coarse_joint(g1.points()).reshape(g1.shape).astype(np.float32)
        Fc = np.maximum(Fc, screen_clip(Ts, a.screen_off)(g1.points()).reshape(g1.shape))
        Fc = main_component(Fc, lm1, g1)
        return g1, Fc, mesh_from_field(Fc, g1)

    def fuse(g, Fh, dm, KS):
        gt = Grid(g.lo, g.hi(), a.tsdf_voxel)
        Fh_t = interp_field(Fh, g, gt.points()).reshape(gt.shape).astype(np.float32)
        Ft, nvox = tsdf_field(gt, Fh_t, all_views, all_exts, KS, dm, a)
        valid = np.isfinite(Ft).astype(np.float32)
        Ftv = np.nan_to_num(Ft)
        if a.tsdf_smooth > 0:  # normalized Gaussian over the voxels with data: removes pits from depth noise
            sg = a.tsdf_smooth / a.tsdf_voxel
            wsum = ndimage.gaussian_filter(valid, sg)
            with np.errstate(invalid="ignore", divide="ignore"):
                Ftv = np.where(valid > 0, ndimage.gaussian_filter(Ftv * valid, sg) / np.maximum(wsum, 1e-6), 0)
        Xf = g.points()
        w_ = interp_field(valid, gt, Xf, outside=0)
        f_ = interp_field(Ftv, gt, Xf, outside=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            f_ = np.where(w_ > 0.5, f_ / np.maximum(w_, 1e-6), np.nan)
        Fr = Fh.ravel().copy()
        ok = np.isfinite(f_)
        Fr[ok] = np.maximum(Fr[ok], f_[ok])
        return Fr.reshape(g.shape), nvox

    def mvs(hull_c):
        refs = list(range(0, len(all_views), max(1, a.mvs_ref_step)))
        t0 = time.time()
        dm, KS = depth_maps(all_views, all_exts, hull_c, a, refs)
        dm, cst = consistency_filter(all_views, all_exts, KS, dm, a)
        log(f"depth maps: {len(refs)} reference views, {cst['consistent']}/{cst['depth_pixels']} "
            f"depth pixels confirmed by >= {a.mvs_consistent} other maps ({time.time() - t0:.0f}s)")
        return dm, KS, {"source": "plane-sweep NCC depth maps", "reference_views": len(refs), **cst}

    all_views, all_exts = gather()
    base = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    dm, KS, mvs_stats = None, None, {}
    if not a.no_photo:
        g1, Fc, hull_c = coarse_hull_mesh()
        if a.colmap_fused:
            cpts = np.asarray(trimesh.load(a.colmap_fused).vertices)
            KS = [scaled_K(v.K, a.mvs_scale) for v in all_views]
            dm = depth_from_points(cpts, all_views, all_exts, KS, list(range(len(all_views))))
            mvs_stats["source"] = f"COLMAP {a.colmap_fused}: {len(cpts)} points"
        else:
            dm, KS, mvs_stats = mvs(hull_c)
            if a.align_iters > 0:
                Fb1, _ = fuse(g1, Fc, dm, KS)
                mesh0 = mesh_from_field(main_component(Fb1, lm1, g1), g1)
                mvs_stats["texture_alignment"] = align_to_texture(mesh0, all_views, all_exts, a, a.align_iters)
                for p in per:  # silhouettes again with the corrected cameras
                    p["grid"], p["F"] = take_hull(p["views"], a, p["lm"])
                g1, Fc, hull_c = coarse_hull_mesh()
                dm, KS, st2 = mvs(hull_c)
                mvs_stats.update({k + "_after_alignment": v for k, v in st2.items() if k != "source"})
        # registration refinement on the photo surfaces (overlap of the takes: sides, tips)
        for k in range(1, len(per)):
            P1, D1 = take_points(all_views, all_exts, KS, dm, 0)
            Pk, Dk = take_points(all_views, all_exts, KS, dm, k)
            dT, info = icp_refine(P1, D1, Pk, Dk, a)
            report["registration"][k - 1]["icp"] = info
            if dT is not None and info.get("translation_mm", 99) < a.icp_max_mm * 2:
                Rd, td = dT
                R, t = Ts[k]
                Ts[k] = (R @ Rd.T, t - R @ Rd.T @ td)
                report["registration"][k - 1]["R"] = Ts[k][0].tolist()
                report["registration"][k - 1]["t"] = Ts[k][1].tolist()
                log(f"take {k + 1}: ICP on the depth surfaces, {info['pairs']} pairs, correction "
                    f"{info['rotation_deg']:.2f} deg / {info['translation_mm']:.2f} mm, residual "
                    f"{info['residual_mm_median']:.2f} mm")
            else:
                log(f"take {k + 1}: ICP skipped ({info})")
        all_views, all_exts = gather()
    # joint hull at the fine voxel size
    g1 = per[0]["grid"]
    occ = g1.points(np.nonzero(coarse_joint(g1.points()) < 0)[0])
    lo, hi = occ.min(0) - 3, occ.max(0) + 3
    gf = Grid(lo, hi, a.voxel)
    log(f"joint hull: {len(all_views)} views, grid {gf.shape} at {a.voxel} mm")
    Fj = field_on_grid(all_views, all_exts, gf, a, coarse=coarse_joint, band=2.0,
                       clip=screen_clip(Ts, a.screen_off))
    if lm1 is not None:
        Fj = cut_field(Fj, gf, lm1, a.wrist_margin)
    Fj = main_component(Fj, lm1, gf)
    hull = mesh_from_field(Fj, gf)
    log(f"hull mesh: {len(hull.vertices)} vertices, watertight {hull.is_watertight}")
    hull.export(base + "_hull.ply")
    final = hull
    photo_stats = None
    Fbase = Fj
    if dm is not None:
        Fbase, nvox = fuse(gf, Fj, dm, KS)
        photo_stats = {**mvs_stats, "tsdf_voxels_with_data": nvox}
        if a.save_state:
            np.savez_compressed(a.save_state + "_depth.npz",
                                **{f"d{j}": dm[j][0] for j in dm}, **{f"c{j}": dm[j][1] for j in dm})
    # the side no camera saw: thickness prior from the measured (stand-in) hand
    Fprior = None
    if lm1 is not None and a.prior != "off":
        Fprior, pinfo = thickness_prior(gf, main_component(Fbase, lm1, gf), lm1, all_exts, a, a.template)
        report["thickness_prior"] = pinfo
        if Fprior is not None:
            okp = np.isfinite(Fprior)
            Fbase = Fbase.copy()
            Fbase[okp] = np.maximum(Fbase[okp], Fprior[okp])
            if pinfo.get("observed_side") == "back":
                log(f"WARNING: cameras saw the BACK of a {a.hand} hand. If the video shows the palm, "
                    f"the hand is the other one: rerun with --hand {'left' if a.hand == 'right' else 'right'}")
            log(f"thickness prior ({pinfo.get('observed_side')} side observed): template "
                f"{os.path.basename(pinfo['template'])} x{pinfo['thickness_scale']:.3f}, median thickness "
                f"{pinfo['thickness_median_mm']:.1f} mm")
        else:
            log(f"thickness prior skipped: {pinfo.get('skipped')}")
    if Fbase is not Fj:
        if lm1 is not None:
            Fbase = cut_field(Fbase, gf, lm1, a.wrist_margin)
        Fbase = main_component(Fbase, lm1, gf)
        final = mesh_from_field(Fbase, gf)
    import trimesh.smoothing
    trimesh.smoothing.filter_taubin(final, lamb=0.5, nu=-0.53, iterations=6)
    if not final.is_watertight:
        trimesh.repair.fill_holes(final)
    final_full = final
    final = decimate(final, a.target_faces)
    final.export(a.out)
    log(f"final mesh: {len(final.vertices)} vertices, watertight {final.is_watertight}, "
        f"volume {final.volume / 1000:.1f} ml -> {a.out}")
    if a.hand == "left":  # mirror (board X -> -X) so the right-hand rig can use it
        mir = final.copy()
        mir.vertices[:, 0] *= -1
        mir.faces = mir.faces[:, ::-1]
        mir.export(base + "_as_right.ply")
        log(f"left hand: mirrored copy (board X negated) -> {base}_as_right.ply")
    # landmarks in the output frame
    fused = {}
    for k, p in enumerate(per):
        if p["lm"] is None:
            continue
        if k == 0:
            L = p["lm"]
        else:
            R, t = Ts[k]
            L = (p["lm"] - t) @ R
        lm_report["takes"][k]["landmarks_output_frame_mm"] = {
            n: [round(float(c), 2) for c in L[i]] for i, n in enumerate(NAMES)}
        fused[k] = L
    if fused:
        Lm = np.median(np.stack(list(fused.values())), 0)
        lm_report["landmarks_output_frame_mm"] = {n: [round(float(c), 2) for c in Lm[i]]
                                                  for i, n in enumerate(NAMES)}
        if len(fused) > 1:
            dd = np.linalg.norm(fused[0] - fused[max(fused)], axis=1)
            lm_report["take_disagreement_mm"] = {n: round(float(x), 2) for n, x in zip(NAMES, dd)}
    with open(base + "_landmarks.json", "w") as f:
        json.dump(lm_report, f, indent=1)
    # quality report
    wm = None
    if lm1 is not None:
        nrm = lm1[9] - lm1[0]
        nrm /= np.linalg.norm(nrm)
        wm = (lm1[0] - a.wrist_margin * nrm, nrm)
    iou = mask_iou(final, all_views, all_exts, wm)
    iou_h = mask_iou(decimate(hull, a.target_faces), all_views, all_exts, wm)
    vpv = views_per_vertex(final, all_views, all_exts)
    Vh = np.asarray(hull.vertices)
    tree = cKDTree(Vh)
    dd, nn = tree.query(np.asarray(final.vertices))
    sgn = np.sign(np.sum((np.asarray(final.vertices) - Vh[nn]) * np.asarray(hull.vertex_normals)[nn], 1))
    diff = dd * sgn
    # coverage: what determines each output vertex, per hand region
    dpts = None
    if dm is not None:
        dpts = np.concatenate([take_points(all_views, all_exts, KS, dm, k, spacing=0.5)[0]
                               for k in range(len(per))])
    labels = coverage_labels(final, gf, Fj, Fprior, dpts, Ts, a, wm)
    coverage = {}
    if lm1 is not None:
        regs = hand_regions(np.asarray(final.vertices), np.asarray(final.vertex_normals), lm1)
        areas = np.zeros(len(final.vertices))
        np.add.at(areas, np.asarray(final.faces).ravel(), np.repeat(final.area_faces / 3, 3))
        for r in ["fingers", "thumb", "palm", "palm_edge", "back", "wrist", "all"]:
            sel = np.ones(len(regs), bool) if r == "all" else regs == r
            if sel.sum() == 0:
                continue
            tot = areas[sel].sum()
            coverage[r] = {"area_cm2": round(float(tot / 100), 1),
                           **{l: round(float(areas[sel & (labels == l)].sum() / tot), 3) for l in LABELS}}
            coverage[r]["observed"] = round(coverage[r]["photo"] + coverage[r]["silhouette"], 3)
            coverage[r]["filled_in"] = round(coverage[r]["prior"] + coverage[r]["screen"] + coverage[r]["cut"], 3)
        log("coverage (area fraction): " + "; ".join(
            f"{r} photo {c['photo']:.2f} silhouette {c['silhouette']:.2f} prior {c['prior']:.2f}"
            for r, c in coverage.items() if r in ("fingers", "thumb", "palm", "back")))
    cov_mesh = final.copy()
    cov_mesh.visual.vertex_colors = np.array([LABEL_COLORS[l] + (255,) for l in labels], np.uint8)
    cov_mesh.export(base + "_coverage.ply")
    np.save(base + "_coverage_labels.npy", np.array([LABELS.index(l) for l in labels], np.uint8))
    report["coverage"] = {"labels": LABELS, "meaning": {
        "photo": "measured by multi-view stereo (depth maps)",
        "silhouette": "bounded by the visual hull: exact on the sides (rims), an upper bound elsewhere",
        "prior": "unobserved: front surface minus the template thickness",
        "screen": "unobserved: limited by the screen glass", "cut": "wrist cut"},
        "per_region_area_fraction": coverage}
    report.update({
        "views": [{"take": v.take + 1, "frame": v.idx, "ts_ms": v.ts, "f_scale": v.f_scale,
                   "board_rms_px": v.rms, "corners": v.ncorner, "sharpness": v.sharp,
                   "landmarks": v.lm is not None, "iou_final": iou[i], "iou_hull": iou_h[i], **v.info}
                  for i, v in enumerate(all_views)],
        "mask_iou": {"final_median": float(np.nanmedian(iou)), "final_p10": float(np.nanpercentile(iou, 10)),
                     "hull_median": float(np.nanmedian(iou_h)),
                     "worst_views": [f"t{all_views[i].take + 1}_{all_views[i].idx}"
                                     for i in np.argsort(np.nan_to_num(iou, nan=1))[:5]]},
        "views_per_surface_point": {"median": float(np.median(vpv)), "p05": float(np.percentile(vpv, 5)),
                                    "frac_below_3": float(np.mean(vpv < 3)),
                                    "median_where_seen": float(np.median(vpv[vpv > 0])) if (vpv > 0).any() else 0.0,
                                    "frac_never_seen": float(np.mean(vpv == 0))},
        "hull_vs_refined_mm": {"median_abs": float(np.median(np.abs(diff))),
                               "p95_abs": float(np.percentile(np.abs(diff), 95)),
                               "max_inward": float(-diff.min())},
        "photo": photo_stats,
        "mesh": {"vertices": int(len(final.vertices)), "faces": int(len(final.faces)),
                 "watertight": bool(final.is_watertight), "volume_ml": float(final.volume / 1000),
                 "bounds_mm": final.bounds.tolist()},
        "wrist_cut": None if wm is None else {"point": wm[0].tolist(), "normal": wm[1].tolist()},
        "runtime_s": round(time.time() - t_start, 1)})
    if a.colmap:
        D = [np.linalg.norm(v.C() if Ts[v.take] is None else
                            ((v.C() - Ts[v.take][1]) @ Ts[v.take][0]) - final.centroid) for v in all_views]
        report["colmap_commands"] = write_colmap(all_views, all_exts, a.colmap,
                                                 (max(50, min(D) - 150), max(D) + 150),
                                                 np.asarray(final.vertices)[::200])
        log("COLMAP project ->", a.colmap)
    with open(base + "_report.json", "w") as f:
        json.dump(report, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log(f"mask IoU median {report['mask_iou']['final_median']:.3f} (hull {report['mask_iou']['hull_median']:.3f}), "
        f"views per seen point median {report['views_per_surface_point']['median_where_seen']:.0f} "
        f"({100 * report['views_per_surface_point']['frac_never_seen']:.0f}% of surface never seen), "
        f"hull->refined median {report['hull_vs_refined_mm']['median_abs']:.2f} mm, "
        f"total {report['runtime_s']:.0f}s")
    if a.save_state:
        np.savez_compressed(a.save_state + "_fields.npz", Fj=Fj, lo=gf.lo, step=gf.step,
                            T=np.array([np.r_[np.ravel(T[0]), T[1]] if T else np.zeros(12) for T in Ts]))
    return report


if __name__ == "__main__":
    run(parse_args())
