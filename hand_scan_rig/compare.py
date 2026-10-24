"""Module D: compare the folded scan hand with the photogrammetry data, and tune the skin.

    python compare.py --rig RIG.blend --relaxed build/gen/data/hand_relaxed.json \
        --videos "build/gen/photos/relaxed/*.mp4" --board-spec data/board_screen.json --out DIR
    python compare.py synth [render|run|eval|all]     # synthetic test (no real data needed)

Two levels.

1. Landmark level (--relaxed only).  The rig's pose (all 21 joint DOFs of skin.Rig, inside
   anatomical limits) plus a rigid root transform (and optionally one global scale) is fitted to
   the 21 triangulated landmarks of every take with a robust loss (soft-L1, 2 mm).  Output: the
   relaxed pose on the scan rig (angles in the relaxed_pose / SPEC definitions, pose.json format),
   per-joint residuals, and the scale test: a scan whose size is off shows up as a free-scale
   fit far from 1 together with much smaller residuals than the fixed-scale fit.

2. Surface level (--videos).  Per take (one video): board corners -> lens calibration of the
   video mode and camera pose per frame (the gripscan.relaxed_pose functions); hand silhouette
   per frame = pixels that differ from the board pattern warped into the frame by the estimated
   camera (photometric model of the screen fitted locally), plus a colour (chroma) cue outside
   the board, morphology, and the connected blob that holds the hand landmarks (MediaPipe 2D,
   a --landmarks2d sidecar, or the triangulated 3D landmarks projected).  The arm is cut off by
   an image line at --roi-cut-mm proximal of the wrist (the scan's forearm stub is 40 mm).
   The board difference fails for a pale hand over white squares or a dark / shaded hand over
   black ones, so the cut-out is refined with GrabCut around the rig placed at the level-1 pose
   (sure hand inside it, sure background > 9 mm outside it, colour decides between; CIELAB with
   the colour axes x4, two passes), and every frame is checked against that model silhouette
   (IoU >= 0.8 on the hand side of the arm cut) and the 2D landmarks (>= 85% inside); frames
   that disagree are dropped and listed.
   The posed + deformed scan mesh (skin.py numpy backend: bulge keys, DQS/LBS, Corrective
   Smooth) is projected with the same camera and rasterised.  Score per frame = symmetric
   contour distance in mm at the hand depth (truncated at 10 mm) + 0.05 x landmark reprojection
   (mm); IoU reported.  Small pose corrections per take (CMA-ES, step 0.05 sigma, root at 0.3 of
   that) and the skin parameters are optimised in alternating stages, regularised toward the
   defaults / the landmark-level pose.  Skin candidates (--tune widths: finger MCP / PIP / DIP
   zone widths + MCP pad bulge; --tune all adds offsets, tilts, PIP/DIP bulges, cs_factor) are
   first profiled (skin_sensitivity); only those whose score slope or curvature stands above
   the frame-to-frame noise are tuned, the rest stay at the defaults, and every one gets a
   confidence half-width.  The frames not used inside the optimiser are scored separately
   (held out); a held-out score that gets worse, or (synthetic test) a tuned score below the
   true hand's, is reported as a warning: the tuned values are best fits, not measurements.
   Lens: k1, k2 and, with >= 15 board views, k3 when it lowers the hold-out reprojection; the
   per-frame board reprojection is reported (> 0.5 px: stabilisation probably on).

Synthetic test scenarios (SCENARIOS): main (fixed true skin, top light / light skin side
light), seed21 / seed33 (random true skins, dark / light skin tones and side light), seed7q,
and autoweights (an out-of-family truth: Blender bone-heat weights + plain LBS).
    python compare.py synth --scenario main render|run|eval
    python compare.py synth multi          # every other scenario + multi_summary.json
    python compare.py synth --scenario main masks    # silhouette test across skin tones / light

Conventions: 1 unit = 1 mm, right hand, board frame of gripscan.board (OpenCV ChArUco: X right,
Y down the board, Z into the board, i.e. points in front of the screen have negative Z).
Angles: relaxed_pose.angles definitions, thumb MCP/IP + toward the palm (skin.spec_angles).
Pose vector x = skin.Rig.fk layout (skin.MEAS order: arch_imr, arch_mrp, 3 spreads, 12 finger
flexions, thumb_cmc, thumb_cmc_elev, thumb_mcp, thumb_ip), each entry ~ the change of that angle
from the scan's rest pose.  Root = (rotation vector, translation, log scale): world =
R (s (p - wrist_rest)) + wrist_rest + t.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/tmp/claude-0/hs/pg")
import skin  # noqa: E402
from skin import (MEAS, F4, measurements, load_params, compute_weights, skin_dqs,  # noqa: E402
                  skin_lbs, CorrectiveSmooth, params_key)
from gripscan.landmarks import NAMES  # noqa: E402
from gripscan.board import BoardSpec, Camera, Pose, detect, make_board  # noqa: E402
from gripscan.triangulate import projection, normalize, robust_point  # noqa: E402

OUT = "/mnt/project-files/hand_scan_rig/compare"
WORK = "/tmp/claude-0/hs/work/compare"
STANDIN_BLEND = "/mnt/project-files/hand_scan_rig/skin/standin_rigged.blend"
BOARD_SPEC = "/mnt/project-files/hand_scan_rig/data/board_screen.json"
CAMERA_JSON = "/tmp/claude-0/hs/pg/data/camera.json"
REAL_EXTRACT = "/mnt/project-files/hand_scan_rig/data/hand_relaxed_extract.json"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".3gp"}
MIN_CORNERS = 12
MAX_POSE_RMS = 3.0
PALM_IDX = [0, 1, 5, 9, 13, 17]
NI = {n: i for i, n in enumerate(NAMES)}

# ------------------------------------------------------------------ anatomical limits
# absolute angles (SPEC definitions, degrees).  Spreads / thumb CMC are limited relative to the
# rest pose (their x entries are rotations, not the unsigned spread itself).
LIMITS = {"mcp": (-25.0, 95.0), "pip": (-5.0, 110.0), "dip": (-10.0, 90.0),
          "thumb_mcp": (-15.0, 70.0), "thumb_ip": (-25.0, 85.0), "arch": (-10.0, 30.0)}
REL_LIMITS = {"spread": 25.0, "thumb_cmc": 40.0, "thumb_cmc_elev": 40.0}


def limit_of(k):
    if k.startswith("arch"):
        return LIMITS["arch"]
    if k in ("thumb_mcp", "thumb_ip"):
        return LIMITS[k]
    for j in ("mcp", "pip", "dip"):
        if k.endswith("_" + j) and not k.startswith("thumb"):
            return LIMITS[j]
    return None


def x_bounds(rest):
    lo, hi = np.zeros(len(MEAS)), np.zeros(len(MEAS))
    for i, k in enumerate(MEAS):
        lim = limit_of(k)
        if lim is not None:
            lo[i], hi[i] = lim[0] - rest[k], lim[1] - rest[k]
        else:
            r = REL_LIMITS["spread" if k.startswith("spread") else k]
            lo[i], hi[i] = -r, r
        lo[i], hi[i] = min(lo[i], -0.5), max(hi[i], 0.5)
    return lo, hi


# ------------------------------------------------------------------ small math

def rotvec_mat(r):
    return cv2.Rodrigues(np.asarray(r, float).reshape(3, 1))[0]


def mat_rotvec(R):
    return cv2.Rodrigues(np.asarray(R, float))[0].ravel()


def kabsch(A, B, w=None):
    """R, c with B ~ R A + c."""
    w = np.ones(len(A)) if w is None else np.asarray(w, float)
    ca, cb = (w[:, None] * A).sum(0) / w.sum(), (w[:, None] * B).sum(0) / w.sum()
    H = ((A - ca) * w[:, None]).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def apply_root(Y, root, w0):
    """root = [rotvec(3), t(3), log s] -> R (s (Y - w0)) + w0 + t."""
    R = rotvec_mat(root[:3])
    s = np.exp(root[6]) if len(root) > 6 else 1.0
    return (s * (Y - w0)) @ R.T + w0 + root[3:6]


# ------------------------------------------------------------------ hand model (numpy only after load)

def decimate_mesh(V, F, ratio):
    import bpy
    me = bpy.data.meshes.new("dec")
    me.from_pydata(V.tolist(), [], F.tolist())
    ob = bpy.data.objects.new("dec", me)
    bpy.context.scene.collection.objects.link(ob)
    m = ob.modifiers.new("dec", "DECIMATE")
    m.ratio = float(ratio)
    dg = bpy.context.evaluated_depsgraph_get()
    e = ob.evaluated_get(dg).to_mesh()
    e.calc_loop_triangles()
    Vd = np.zeros(len(e.vertices) * 3)
    e.vertices.foreach_get("co", Vd)
    Fd = np.zeros(len(e.loop_triangles) * 3, np.int32)
    e.loop_triangles.foreach_get("vertices", Fd)
    ob.evaluated_get(dg).to_mesh_clear()
    bpy.data.objects.remove(ob)
    return Vd.reshape(-1, 3), Fd.reshape(-1, 3).astype(np.int64)


class HandModel3D:
    """The rigged scan: skin.Skinner loads the .blend (bpy) once; posing is numpy only
    (same algorithm as skin.Skinner._pose_numpy, verified there against Blender)."""

    def __init__(self, blend, params=None, decimate=0.2):
        sk = skin.Skinner(blend, params)
        self.blend = os.path.abspath(blend)
        self.rig = sk.rig
        self.base_params = copy.deepcopy(sk.params)
        self.Vf, self.Ff = sk.V0.copy(), sk.F.copy()
        if decimate and decimate < 1:
            self.Vd, self.Fd = decimate_mesh(self.Vf, self.Ff, decimate)
        else:
            self.Vd, self.Fd = self.Vf, self.Ff
        self.w0 = self.rig.L["wrist"].copy()
        self.rest = {k: v for k, v in self.rig.rest_meas.items()}
        self.xlo, self.xhi = x_bounds(self.rest)
        # what Skinner.local_flex needs (bulge from the hinge angle, as the Blender drivers)
        self.amw, self.rest_mat = sk.amw.copy(), dict(sk.rest_mat)
        self.order = sk.order
        self.fixed_weights = None
        if sk.params["global"]["weights"] == "blend":   # out-of-family truth: the .blend's own groups
            self.fixed_weights = (sk.W.copy(), sk.order)
        self._cache = {}
        del sk

    def mesh_arrays(self, full):
        return (self.Vf, self.Ff) if full else (self.Vd, self.Fd)

    def skinning(self, params, full=False):
        key = params_key(params) + ("F" if full else "d")
        if key not in self._cache:
            V, F = self.mesh_arrays(full)
            if self.fixed_weights is not None:
                if not full and len(V) != len(self.Vf):
                    raise ValueError("a .blend-weights model is only evaluated on the full mesh")
                g = params["global"]
                cs = CorrectiveSmooth(V, F, np.zeros(len(V)), g["cs_factor"], g["cs_iterations"])
                self._cache[key] = (self.fixed_weights[0], cs, {})
                return self._cache[key]
            r = compute_weights(V, F, self.rig, params)
            g = params["global"]
            cs = CorrectiveSmooth(V, F, r["cs_mask"], g["cs_factor"], g["cs_iterations"])
            if len(self._cache) > 6:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = (r["W"], cs, r["bulge"])
        return self._cache[key]

    def landmarks(self, x):
        M, P = self.rig.fk(np.asarray(x, float))
        return np.array([P[n] for n in NAMES]), M, P

    def angles(self, x):
        P = self.rig.fk(np.asarray(x, float))[1]
        m = measurements(P)
        # + thumb_cmc_azimuth: the signed thumb angle that makes a pose.json unambiguous (skin.Rig.solve)
        return {k: float(m[k]) for k in MEAS + ["thumb_cmc_azimuth"]}

    def mesh(self, params, x, full=False):
        W, cs, bulge = self.skinning(params, full)
        V, F = self.mesh_arrays(full)
        M, P = self.rig.fk(np.asarray(x, float))
        X = V.copy()
        if bulge:
            # same as skin.Skinner.bulge_values(M=M): hinge flexion -> bulge_curve
            fl = skin.Skinner.local_flex(self, M)
            ref = params["global"]["bulge_ref_deg"]
            for j, d in bulge.items():
                a = skin.bulge_curve(fl.get(j, 0.0) / ref)
                if a > 0:
                    X += a * d
        order = self.fixed_weights[1] if self.fixed_weights is not None else None
        X = skin_dqs(X, W, M, order) if params["global"]["dqs"] else skin_lbs(X, W, M)
        return cs(X)

    def x_from_angles(self, angles):
        """SPEC angles dict -> x (skin.Rig.solve)."""
        x = self.rig.solve({"angles": angles})[2]["x"]
        return np.asarray(x)[:len(MEAS)]     # thumb pronation (entry len(MEAS)) stays at rest


# ------------------------------------------------------------------ level 1: landmark fit

def fit_landmarks(model, Lobs, fit_scale=False, joint_w=None, f_scale=2.0):
    """Robust fit of pose x (21, inside limits) + root (+ log scale) to 21x3 landmarks (mm)."""
    from scipy.optimize import least_squares
    Lobs = np.asarray(Lobs, float)
    w = np.ones(21) if joint_w is None else np.asarray(joint_w, float)
    sw = np.sqrt(w)[:, None]
    lo, hi = model.xlo, model.xhi
    m_obs = measurements({n: Lobs[i] for i, n in enumerate(NAMES)})
    xa = np.zeros(len(MEAS))
    for i, k in enumerate(MEAS):
        if limit_of(k) is not None:
            xa[i] = np.clip(m_obs[k] - model.rest[k], lo[i] + 1e-3, hi[i] - 1e-3)
    best = None
    for x0 in (xa, np.zeros(len(MEAS))):
        P0 = model.landmarks(x0)[0]
        R, c = kabsch(P0[PALM_IDX] - model.w0, Lobs[PALM_IDX])
        q0 = np.r_[x0, mat_rotvec(R), c - model.w0]
        if fit_scale:
            q0 = np.r_[q0, 0.0]
        nx = len(MEAS)

        def res(q):
            root = q[nx:] if fit_scale else np.r_[q[nx:], 0.0]
            P = apply_root(model.landmarks(q[:nx])[0], root, model.w0)
            return np.r_[((P - Lobs) * sw).ravel(), 0.003 * q[:nx]]

        blo = np.r_[lo, [-np.inf] * 6] if not fit_scale else np.r_[lo, [-np.inf] * 6, -0.4]
        bhi = np.r_[hi, [np.inf] * 6] if not fit_scale else np.r_[hi, [np.inf] * 6, 0.4]
        xs = np.r_[np.ones(nx), [0.02] * 3, [1.0] * 3] if not fit_scale else np.r_[np.ones(nx), [0.02] * 3, [1.0] * 3, 0.01]
        sol = least_squares(res, q0, bounds=(blo, bhi), loss="soft_l1", f_scale=f_scale, x_scale=xs,
                            max_nfev=400)
        if best is None or sol.cost < best.cost:
            best = sol
    q = best.x
    x = q[:len(MEAS)]
    root = q[len(MEAS):] if fit_scale else np.r_[q[len(MEAS):], 0.0]
    P = apply_root(model.landmarks(x)[0], root, model.w0)
    err_all = np.linalg.norm(P - Lobs, axis=1)
    err = np.where(w > 0, err_all, 0.0)        # summary stats over the joints that were fitted
    ang = model.angles(x)
    at_limit = [MEAS[i] for i in range(len(MEAS)) if min(x[i] - lo[i], hi[i] - x[i]) < 0.05]
    return {"x": x.tolist(), "root": root.tolist(), "scale": float(np.exp(root[6])),
            "angles": ang, "residual_mm": {n: float(e) for n, e in zip(NAMES, err_all)},
            "median_mm": float(np.median(err_all[w > 0])), "rms_mm": float(np.sqrt(np.mean(err_all[w > 0] ** 2))),
            "max_mm": float(err.max()), "worst": NAMES[int(np.argmax(err))],
            "at_limit": at_limit, "landmarks_fit": P.tolist()}


def pose_json(angles, root, w0):
    R = rotvec_mat(root[:3])
    from scipy.spatial.transform import Rotation
    qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
    # skin.Rig.fk root: rotation about the wrist then translation
    return {"angles": {k: round(float(v), 2) for k, v in angles.items()},
            "root": {"location": [float(v) for v in root[3:6]], "rotation_quat": [qw, qx, qy, qz],
                     "scale_about_wrist": float(np.exp(root[6])) if len(root) > 6 else 1.0}}


WRIST_OUTLIER_MM = 10.0


def level1(model, takes, key, out_dir, joint_w=None):
    res = []
    for ti, t in enumerate(takes):
        src = t.get(key) or t.get("landmarks_board_mm")
        Lobs = np.array([src[n] for n in NAMES], float)
        jw = np.ones(21) if joint_w is None else np.asarray(joint_w, float).copy()
        fixed = fit_landmarks(model, Lobs, False, jw)
        # the wrist and thumb CMC are the least well-defined landmarks (MediaPipe's wrist is not the
        # rig's wrist joint); one far off the rig's palm layout is dropped instead of bending the MCPs
        dropped = [n for n in ("wrist", "thumb_cmc") if fixed["residual_mm"][n] > WRIST_OUTLIER_MM]
        if dropped:
            jw[[NI[n] for n in dropped]] = 0.0
            fixed = fit_landmarks(model, Lobs, False, jw)
            print(f"[level1] take {ti}: {', '.join(dropped)} more than {WRIST_OUTLIER_MM:.0f} mm off the rig's palm "
                  f"layout: left out of the fit (residual now {', '.join('%.1f' % fixed['residual_mm'][n] for n in dropped)} mm)")
        free = fit_landmarks(model, Lobs, True, jw)
        fixed["dropped_joints"] = free["dropped_joints"] = dropped
        res.append({"take": ti, "input": t.get("input", f"take{ti}"), "fixed_scale": fixed, "free_scale": free,
                    "observed_landmarks": Lobs.tolist(),
                    "observed_angles": {k: float(v) for k, v in measurements(
                        {n: Lobs[i] for i, n in enumerate(NAMES)}).items() if k in MEAS}})
        print(f"[level1] take {ti}: fixed scale median {fixed['median_mm']:.2f} mm, rms {fixed['rms_mm']:.2f}, "
              f"max {fixed['max_mm']:.2f} ({fixed['worst']}); free scale {free['scale']:.4f} -> rms {free['rms_mm']:.2f}")
    sc = np.array([r["free_scale"]["scale"] for r in res])
    rf = np.array([r["fixed_scale"]["rms_mm"] for r in res])
    rs = np.array([r["free_scale"]["rms_mm"] for r in res])
    flag = bool(abs(np.median(sc) - 1) > 0.025 and np.median(rs) < 0.85 * np.median(rf))
    use = "free_scale" if flag else "fixed_scale"
    angles_med = {k: float(np.median([r[use]["angles"][k] for r in res])) for k in res[0][use]["angles"]}
    summary = {"scale_free_median": float(np.median(sc)), "scale_free_per_take": sc.tolist(),
               "rms_fixed_median": float(np.median(rf)), "rms_free_median": float(np.median(rs)),
               "scale_mismatch_flag": flag,
               "scale_note": ("scan looks %+.1f%% off in size vs the landmarks (rig would need x%.3f)"
                              % ((1 / np.median(sc) - 1) * 100, np.median(sc))) if flag else
               "no significant scale mismatch (|scale-1| <= 2.5% or no residual gain)",
               "angles_median": angles_med, "angles_from": use}
    out = {"landmark_key": key, "takes": res, "summary": summary}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        json.dump(out, open(os.path.join(out_dir, "level1_landmarks.json"), "w"), indent=1)
        r0 = res[0][use]
        pj = pose_json(angles_med, np.array(r0["root"]), model.w0)
        pj["note"] = f"median angles over takes ({use} fits); root of take 0 (board frame); compare.py level 1"
        json.dump(pj, open(os.path.join(out_dir, "relaxed_pose.json"), "w"), indent=1)
    print(f"[level1] scale free median {summary['scale_free_median']:.4f}; flag {flag}: {summary['scale_note']}")
    return out


# ------------------------------------------------------------------ rasteriser

def raster(P2, F, H, W, depth=None):
    """Union of projected triangles over pixel centres (integer coords).  With depth (per vertex)
    also returns the visible face id per pixel (-1 none)."""
    fin = np.isfinite(P2).all(1)
    orig = None
    if not fin.all():
        keep = fin[F].all(1)
        orig = np.nonzero(keep)[0]
        F = F[keep]
        P2 = np.where(fin[:, None], P2, 0.0)
    T = P2[F]
    mnx = np.ceil(T[:, :, 0].min(1)).astype(np.int64)
    mxx = np.floor(T[:, :, 0].max(1)).astype(np.int64)
    mny = np.ceil(T[:, :, 1].min(1)).astype(np.int64)
    mxy = np.floor(T[:, :, 1].max(1)).astype(np.int64)
    mnx, mny = np.maximum(mnx, 0), np.maximum(mny, 0)
    mxx, mxy = np.minimum(mxx, W - 1), np.minimum(mxy, H - 1)
    w, h = mxx - mnx + 1, mxy - mny + 1
    ok = (w > 0) & (h > 0)
    S = np.maximum(w, h)
    mask = np.zeros(H * W, bool)
    pix_all, z_all, f_all = [], [], []
    v0 = T[:, 0]
    e1 = T[:, 1] - v0
    e2 = T[:, 2] - v0
    den = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    ok &= np.abs(den) > 1e-12
    for lo_, hi_ in [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 1 << 14)]:
        sel = np.nonzero(ok & (S > lo_) & (S <= hi_))[0]
        if not len(sel):
            continue
        size = hi_ if hi_ <= 128 else int(S[sel].max())
        oy, ox = np.divmod(np.arange(size * size), size)
        chunk = max(1, int(3e6 // (size * size)))
        for c0 in range(0, len(sel), chunk):
            s = sel[c0:c0 + chunk]
            X = mnx[s, None] + ox[None]
            Y = mny[s, None] + oy[None]
            valid = (ox[None] < w[s, None]) & (oy[None] < h[s, None])
            px = X - v0[s, 0, None]
            py = Y - v0[s, 1, None]
            d = den[s, None]
            l1 = (px * e2[s, 1, None] - py * e2[s, 0, None]) / d
            l2 = (e1[s, 0, None] * py - e1[s, 1, None] * px) / d
            ins = valid & (l1 >= -1e-7) & (l2 >= -1e-7) & (l1 + l2 <= 1 + 1e-7)
            pix = (Y * W + X)[ins]
            mask[pix] = True
            if depth is not None:
                zf = depth[F[s]]
                z = (zf[:, 0, None] * (1 - l1 - l2) + zf[:, 1, None] * l1 + zf[:, 2, None] * l2)[ins]
                pix_all.append(pix)
                z_all.append(z)
                f_all.append(np.broadcast_to(s[:, None], ins.shape)[ins])
    mask = mask.reshape(H, W)
    if depth is None:
        return mask
    fid = np.full(H * W, -1, np.int64)
    if pix_all:
        pix, z, f = np.concatenate(pix_all), np.concatenate(z_all), np.concatenate(f_all)
        o = np.lexsort((z, pix))
        pix, f = pix[o], f[o]
        first = np.r_[True, pix[1:] != pix[:-1]]
        fid[pix[first]] = f[first] if orig is None else orig[f[first]]
    return mask, fid.reshape(H, W)


_RMAX = {}


def _rmax(K, dist):
    """Normalised (undistorted) radius a little beyond the image corners; the lens polynomial is
    not trusted (it folds back) outside it.  Image size taken as 2 x principal point."""
    key = (tuple(np.round(np.ravel(K), 3)), tuple(np.round(np.ravel(dist), 6)))
    if key not in _RMAX:
        K = np.asarray(K, float)
        W, H = 2 * (K[0, 2] + 0.5), 2 * (K[1, 2] + 0.5)
        c = np.array([[0, 0], [W, 0], [0, H], [W, H]], np.float64).reshape(-1, 1, 2)
        n = cv2.undistortPoints(c, K, np.asarray(dist, float)).reshape(-1, 2)
        _RMAX[key] = 1.3 * float(np.linalg.norm(n, axis=1).max())
    return _RMAX[key]


def project(V, cam_K, dist, rvec, tvec, safe=True):
    """OpenCV projection; with safe=True points behind the camera (z < 5 mm) or far outside the
    field of view (where the distortion polynomial is meaningless) come back as NaN."""
    V = np.asarray(V, np.float64).reshape(-1, 3)
    p, _ = cv2.projectPoints(V.reshape(-1, 1, 3), rvec, tvec, cam_K, dist)
    p = p.reshape(-1, 2)
    if safe:
        R = rotvec_mat(rvec)
        Xc = V @ R.T + np.asarray(tvec, float).ravel()
        z = Xc[:, 2]
        r = np.hypot(Xc[:, 0], Xc[:, 1]) / np.maximum(z, 1e-9)
        bad = (z < 5.0) | (r > _rmax(cam_K, dist))
        if bad.any():
            p[bad] = np.nan
    return p


def scale_K(K, s):
    K = np.array(K, float).copy()
    K[0, 0] *= s
    K[1, 1] *= s
    K[0, 2] = (K[0, 2] + 0.5) * s - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return K


SCORE_VERSION = 3  # bump when the score changes: a level2_state.json of another version is not resumed
CLOSE_PX = 7       # morphological closing of the video mask (full-resolution px); the model mask gets
                   # the same closing at the comparison scale, so narrow finger gaps are treated alike
HOLE_MM2 = 25.0    # holes in a silhouette smaller than this (mm^2 at the hand) are filled, in the video
                   # mask and in the model mask alike (real gaps between fingers are bigger)


def fill_small_holes(m, max_px):
    """Fill background components not touching the border with area <= max_px."""
    inv = (~m).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    if n <= 1:
        return m
    H, W = m.shape
    small = st[:, cv2.CC_STAT_AREA] <= max_px
    touch = (st[:, cv2.CC_STAT_LEFT] == 0) | (st[:, cv2.CC_STAT_TOP] == 0) | \
        (st[:, cv2.CC_STAT_LEFT] + st[:, cv2.CC_STAT_WIDTH] >= W) | (st[:, cv2.CC_STAT_TOP] + st[:, cv2.CC_STAT_HEIGHT] >= H)
    fill = small & ~touch
    fill[0] = False
    return m | fill[lab]


def boundary(m):
    er = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)
    return m & (er == 0)


# ------------------------------------------------------------------ board / video helpers

def board_pattern(spec, ppm=4.0):
    board = make_board(spec)
    wmm, hmm = spec.squares_x * spec.square_mm, spec.squares_y * spec.square_mm
    size = (int(round(wmm * ppm)), int(round(hmm * ppm)))
    img = board.generateImage(size, marginSize=0, borderBits=1)
    return img.astype(np.float32) / 255.0, (size[0] / wmm, size[1] / hmm), (wmm, hmm)


_RAYS = {}


def pixel_rays(K, dist, size):
    key = (tuple(np.round(np.ravel(K), 4)), tuple(np.round(np.ravel(dist), 6)), tuple(size))
    if key not in _RAYS:
        W, H = size
        xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        pts = np.stack([xx.ravel(), yy.ravel()], 1).reshape(-1, 1, 2)
        n = cv2.undistortPoints(pts, np.asarray(K, float), np.asarray(dist, float), None, None,
                                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-9)).reshape(-1, 2)
        _RAYS.clear()
        _RAYS[key] = np.c_[n, np.ones(len(n))].astype(np.float64)
    return _RAYS[key]


def board_plane_coords(K, dist, size, rvec, tvec):
    """Board-frame XY (mm) where each pixel's ray meets the board plane Z=0, and validity."""
    W, H = size
    rays = pixel_rays(K, dist, size)
    R = rotvec_mat(rvec)
    C = (-R.T @ np.asarray(tvec, float).reshape(3)).ravel()
    d = rays @ R
    t = -C[2] / np.where(np.abs(d[:, 2]) < 1e-9, 1e-9, d[:, 2])
    XY = C[:2] + t[:, None] * d[:, :2]
    return XY.reshape(H, W, 2).astype(np.float32), (t > 0).reshape(H, W)


def iter_video(path, fps=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src / fps))) if fps else 1
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield i, int(i * 1000 / src), img
        i += 1
    cap.release()


def file_key(*parts):
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, str) and os.path.exists(p):
            st = os.stat(p)
            p = f"{os.path.abspath(p)}:{st.st_size}:{st.st_mtime}"
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def scan_video(path, board, fps, sidecar=None, mp_model=None, work=WORK):
    """Board corners + 2D hand landmarks for every sampled frame (cached)."""
    cache = os.path.join(work, f"scan_{file_key(path, fps, bool(sidecar), mp_model or '')}.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    lm = None
    side = None
    if sidecar is not None:
        side = sidecar.get(os.path.basename(path), {})
    else:
        try:
            from gripscan.relaxed_pose import landmarker
            lm = landmarker(mp_model)
        except Exception as e:  # noqa: BLE001
            print(f"[frames] MediaPipe unavailable ({e}); the blob is picked with projected 3D landmarks")
    recs, size = [], None
    for fi, ts, img in iter_video(path, fps):
        size = (img.shape[1], img.shape[0])
        det = detect(img, board)
        px = None
        if side is not None:
            v = side.get(str(fi))
            px = np.array(v, float) if v is not None else None
        elif lm is not None:
            import mediapipe as mpi
            r = lm.detect_for_video(mpi.Image(image_format=mpi.ImageFormat.SRGB,
                                              data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), ts)
            if r.hand_landmarks:
                px = np.array([[q.x * size[0], q.y * size[1]] for q in r.hand_landmarks[0]])
        recs.append({"fi": fi, "ts": ts, "det": det, "px": px})
    if lm is not None:
        lm.close()
    out = {"path": path, "size": size, "recs": recs}
    os.makedirs(work, exist_ok=True)
    pickle.dump(out, open(cache, "wb"))
    print(f"[frames] {os.path.basename(path)}: {len(recs)} frames, board in "
          f"{sum(r['det'] is not None for r in recs)}, 2D hand in {sum(r['px'] is not None for r in recs)}")
    return out


def _monotone_radial(dist, r_max):
    """True when r (1 + k1 r^2 + k2 r^4 + k3 r^6) keeps increasing up to r_max (no fold-back)."""
    k = np.ravel(dist)
    k1, k2, k3 = k[0], k[1], (k[4] if len(k) > 4 else 0.0)
    r = np.linspace(0, r_max, 200)
    d = 1 + 3 * k1 * r ** 2 + 5 * k2 * r ** 4 + 7 * k3 * r ** 6
    return bool((d > 0).all())


def calibrate_lens(dets, size, max_views=40, min_views_k3=15):
    """Lens of one video mode from the board views: relaxed_pose.calibrate_from_frames (k1, k2;
    k3 and tangential fixed at 0) and, with at least min_views_k3 views, also with k3 free (review
    C5).  The k3 model is kept when it lowers the hold-out reprojection (every 4th view left out of
    the calibration and posed with each lens) and stays monotone over the image."""
    from gripscan.relaxed_pose import calibrate_from_frames, pose_of
    cam, rms = calibrate_from_frames(dets, size, max_views)
    info = {"model": "k1,k2", "rms": rms}
    views = [d for d in dets if d is not None and len(d[0]) >= 20]
    if cam is None or len(views) < min_views_k3:
        return cam, rms, info
    pick = np.linspace(0, len(views) - 1, min(max_views + max_views // 3, len(views))).astype(int)
    vv = [views[i] for i in pick]
    held = vv[3::4]
    train = [v for i, v in enumerate(vv) if i % 4 != 3]
    objs = [v[0].astype(np.float32) for v in train]
    imgs = [v[1].astype(np.float32) for v in train]
    res = {}
    for name, flags in (("k1,k2", cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST),
                        ("k1,k2,k3", cv2.CALIB_ZERO_TANGENT_DIST)):
        r_, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None, flags=flags)
        c = Camera(K, dist, size)
        e = [p.reproj_px for p in (pose_of(h, c) for h in held) if p is not None]
        res[name] = (c, float(r_), float(np.median(e)) if e else np.inf)
    info.update({"holdout_px": {k: v[2] for k, v in res.items()}, "train_rms": {k: v[1] for k, v in res.items()}})
    c3 = res["k1,k2,k3"][0]
    if res["k1,k2,k3"][2] < 0.95 * res["k1,k2"][2] and _monotone_radial(c3.dist, _rmax(c3.K, np.zeros(5)) / 1.3):
        objs = [v[0].astype(np.float32) for v in vv]
        imgs = [v[1].astype(np.float32) for v in vv]
        r_, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None, flags=cv2.CALIB_ZERO_TANGENT_DIST)
        info.update(model="k1,k2,k3", rms=float(r_))
        return Camera(K, dist, size), float(r_), info
    return cam, rms, info


LENS_INFO = {}


def calibrate_and_pose(scans, camera_json=None):
    """Video lens (calibrate_lens, or a given camera.json) + board poses."""
    from gripscan.relaxed_pose import pose_of
    cams = {}
    for size in {s["size"] for s in scans}:
        if camera_json:
            cam = Camera.from_dict(json.load(open(camera_json)))
            if tuple(cam.size) != tuple(size):
                cam = cam.scaled_to(size)
            rms = None
        else:
            dets = [r["det"] for s in scans if s["size"] == size for r in s["recs"]]
            cam, rms, info = calibrate_lens(dets, size)
            if cam is None:
                raise SystemExit(f"not enough board views to calibrate the {size} video mode")
            LENS_INFO[f"{size[0]}x{size[1]}"] = info
            print(f"[frames] lens model {info['model']} (hold-out reprojection px: {info.get('holdout_px')})")
        cams[size] = (cam, rms)
        print(f"[frames] lens {size[0]}x{size[1]}: f={cam.K[0, 0]:.1f}px "
              f"calibration RMS {rms if rms is None else round(rms, 3)}")
    for s in scans:
        cam = cams[s["size"]][0]
        s["cam"] = cam
        for r in s["recs"]:
            r["pose"] = None
            if r["det"] is None or len(r["det"][0]) < MIN_CORNERS:
                continue
            p = pose_of(r["det"], cam)
            if p is not None and p.reproj_px <= MAX_POSE_RMS:
                r["pose"] = p
        e = [r["pose"].reproj_px for r in s["recs"] if r["pose"] is not None]
        if e:
            s["board_rms_px"] = {"median": float(np.median(e)), "p90": float(np.percentile(e, 90))}
            msg = (f"[frames] {os.path.basename(s['path'])}: board reprojection per frame median "
                   f"{np.median(e):.2f} px, p90 {np.percentile(e, 90):.2f} px")
            if np.median(e) > 0.5:
                msg += ("  WARNING: above 0.5 px -- one fixed lens does not fit this take (electronic "
                        "stabilisation / crop changing per frame?); film with stabilisation off")
            print(msg)
    return cams


# ------------------------------------------------------------------ silhouette extraction

def _local_mean(val, m, k):
    m = m.astype(np.float32)
    num = cv2.boxFilter(val * m, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)
    den = cv2.boxFilter(m, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)
    return num / np.maximum(den, 1e-4), den


def model_prior(model, params, x, root, cam, pose, size, cut_mm):
    """Silhouette of the rig at the level-1 pose (default skin) in a frame (full resolution):
    {"mask", "distal" (hand side of the arm cut line), "mm_px"} -- the prior for extract_mask."""
    W, H = size
    V = apply_root(model.mesh(params, x), root, model.w0)
    F = model.Fd
    P2 = project(V, cam.K, cam.dist, pose.rvec, pose.tvec)
    ok = np.isfinite(P2).all(1)
    if ok.sum() < 10:
        return None
    lo_, hi_ = np.floor(P2[ok].min(0)).astype(int) - 2, np.ceil(P2[ok].max(0)).astype(int) + 2
    x0, y0 = max(lo_[0], 0), max(lo_[1], 0)
    x1, y1 = min(hi_[0], W - 1), min(hi_[1], H - 1)
    mask = np.zeros((H, W), bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1 + 1, x0:x1 + 1] = raster(P2 - [x0, y0], F, y1 - y0 + 1, x1 - x0 + 1)
    L, _, P = model.landmarks(x)
    prox = model.w0 + cut_mm * skin.unit(model.w0 - P["knuckle"])
    w3 = apply_root(model.w0[None], root, model.w0)[0]
    p3 = apply_root(prox[None], root, model.w0)[0]
    pw = project(w3[None], cam.K, cam.dist, pose.rvec, pose.tvec)[0]
    pp = project(p3[None], cam.K, cam.dist, pose.rvec, pose.tvec)[0]
    u = skin.unit(pw - pp)
    yy, xx = np.mgrid[0:H, 0:W]
    distal = (xx - pp[0]) * u[0] + (yy - pp[1]) * u[1] >= 0
    R = rotvec_mat(pose.rvec)
    depth = float((R @ V.mean(0) + np.asarray(pose.tvec, float).ravel())[2])
    return {"mask": mask & distal, "distal": distal, "mm_px": depth / float(cam.K[0, 0])}


def grabcut_refine(img, cand, prior, in_mm=3.5, out_mm=9.0, iters=4, gc_img=None):
    """Re-segment the hand around the model prior with GrabCut (colour GMMs + edges): the
    prior's core (eroded in_mm) is sure hand, beyond out_mm of it (hand side of the arm cut)
    sure background, the band in between is decided by colour -- a hand whose colour or shading
    is close to one of the board's squares is still found where the pattern difference fails."""
    H, W = cand.shape
    pm, distal, mm_px = prior["mask"], prior["distal"], prior["mm_px"]
    if pm.sum() < 200:
        return cand
    r_in, r_out = max(1, int(round(in_mm / mm_px))), max(2, int(round(out_mm / mm_px)))
    ys, xs = np.nonzero(pm)
    pad = r_out + 8
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    sl = (slice(y0, y1), slice(x0, x1))
    pmc, dc, cc = pm[sl].astype(np.uint8), distal[sl], cand[sl]
    ell = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))  # noqa: E731
    near = cv2.dilate(pmc, ell(r_out)) > 0
    core = cv2.erode(pmc, ell(r_in)) > 0
    gc = np.full(pmc.shape, cv2.GC_PR_BGD, np.uint8)
    gc[(cc | (pmc > 0)) & near] = cv2.GC_PR_FGD
    gc[~dc] = np.where(cc[~dc], cv2.GC_PR_FGD, cv2.GC_PR_BGD)
    gc[~near & dc] = cv2.GC_BGD
    gc[core & dc] = cv2.GC_FGD
    if not (gc == cv2.GC_FGD).any() or not ((gc == cv2.GC_BGD) | (gc == cv2.GC_PR_BGD)).any():
        return cand
    src = np.ascontiguousarray((img if gc_img is None else gc_img)[sl])
    bgm, fgm = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(src, gc, None, bgm, fgm, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return cand
    out = cand.copy()
    out[sl] = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    return out


def mask_agreement(hand, prior, lm_px):
    """Per-frame check of an extracted mask against the model prior (level-1 pose) and the 2D
    landmarks: IoU on the hand side of the arm cut, mean contour distance (mm) and the fraction
    of landmarks inside the mask (2 mm tolerance)."""
    H, W = hand.shape
    d, pm, mm_px = prior["distal"], prior["mask"], prior["mm_px"]
    hm = hand & d
    iou = float((hm & pm).sum() / max((hm | pm).sum(), 1))
    bh, bp = boundary(hm), boundary(pm)
    if bh.any() and bp.any():
        dtp = cv2.distanceTransform((~bp).astype(np.uint8), cv2.DIST_L2, 5)
        dth = cv2.distanceTransform((~bh).astype(np.uint8), cv2.DIST_L2, 5)
        cap = 10.0 / mm_px
        cd = 0.5 * (np.minimum(dtp[bh], cap).mean() + np.minimum(dth[bp], cap).mean()) * mm_px
    else:
        cd = 10.0
    inside = np.nan
    if lm_px is not None:
        r = max(1, int(round(2.0 / mm_px)))
        hd = cv2.dilate(hand.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        q = np.asarray(lm_px, float)
        q = q[np.isfinite(q).all(1)]
        xi, yi = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        inside = float(np.mean(np.where(ok, hd[np.clip(yi, 0, H - 1), np.clip(xi, 0, W - 1)] > 0, False))) if len(q) else np.nan
    return {"prior_iou": iou, "prior_contour_mm": float(cd), "landmarks_inside": inside}


GC_MODE = os.environ.get("COMPARE_GC_MODE", "labraw4")   # GrabCut feature image (see extract_mask)
GC_PASSES = int(os.environ.get("COMPARE_GC_PASSES", "2"))
PRIOR_NOISE = float(os.environ.get("MASK_PRIOR_NOISE", "0"))   # synth masks: level-1 pose error (deg / mm)
MASK_MIN_PRIOR_IOU = 0.80     # extracted mask vs the level-1 model silhouette (hand side of the cut)
MASK_MIN_LM_INSIDE = 0.85     # fraction of the 21 2D landmarks inside the mask


def extract_mask(img, cam, pose, pattern, ppm, board_mm, lm_px, diff_thr=0.22, chroma_thr=0.12,
                 return_debug=False, min_agree=0.93, edge_px=3, prior=None, gc_mode=None,
                 min_prior_iou=MASK_MIN_PRIOR_IOU, min_lm_inside=MASK_MIN_LM_INSIDE):
    """Hand silhouette in a frame: difference to the board pattern warped by the camera pose
    (local black/white levels of the screen), colour (chroma) cue, morphology, landmark blob.
    With prior (model_prior: the rig at the level-1 pose): GrabCut refinement around it
    (grabcut_refine) and a per-frame agreement check (mask_agreement) that rejects the frame
    when the mask and the model / landmarks disagree (wrong segmentation, not a skin change)."""
    H, W = img.shape[:2]
    XY, front = board_plane_coords(cam.K, cam.dist, (W, H), pose.rvec, pose.tvec)
    inset = 1.0
    inb = front & (XY[..., 0] > inset) & (XY[..., 1] > inset) & (XY[..., 0] < board_mm[0] - inset) & \
        (XY[..., 1] < board_mm[1] - inset)
    mx = XY[..., 0] * ppm[0] - 0.5
    my = XY[..., 1] * ppm[1] - 0.5
    p = cv2.remap(pattern, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.5)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f32 = img.astype(np.float32)
    chroma = f32.max(2) - f32.min(2)
    dark, bright = inb & (p < 0.15), inb & (p > 0.85)
    if dark.sum() < 100 or bright.sum() < 100:
        return None
    b0, w0 = np.median(g[dark]), np.median(g[bright])
    rng0 = max(w0 - b0, 10.0)
    ok0 = np.abs(g - (b0 + (w0 - b0) * p)) < 0.3 * rng0
    k = max(31, (min(H, W) // 16) | 1)
    b, db = _local_mean(g, dark & ok0, k)
    w, dw = _local_mean(g, bright & ok0, k)
    b = np.where(db > 1e-3, b, b0)
    w = np.where(dw > 1e-3, w, w0)
    rng = np.maximum(w - b, 10.0)
    pred = b + (w - b) * p
    ker = np.ones((edge_px, edge_px), np.uint8)
    hi, lo = cv2.dilate(pred, ker), cv2.erode(pred, ker)
    diff = np.maximum(np.maximum(g - hi, lo - g), 0) / rng
    c_ref = np.median(chroma[inb & ok0]) if (inb & ok0).any() else 0.0
    cex = (chroma - c_ref) / np.median(rng[inb]) if inb.any() else chroma / 255.0
    cand = np.where(inb, (diff > diff_thr) | ((cex > chroma_thr) & (diff > 0.5 * diff_thr)) | (cex > 2 * chroma_thr),
                    cex > chroma_thr)
    # quality gate: away from the hand, the warped board must explain the screen (a wrong board pose,
    # a lens model failing at the frame corners or a grazing view all show up here)
    if lm_px is not None:
        hull = np.zeros((H, W), np.uint8)
        q = np.asarray(lm_px, float)
        q = q[np.isfinite(q).all(1)]
        if len(q) >= 3:
            cv2.fillConvexPoly(hull, cv2.convexHull(q.astype(np.int32)), 1)
            span = float(np.ptp(q, 0).max())
            hull = cv2.dilate(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(span) | 1, int(span) | 1)))
            far = inb & (hull == 0)
            if far.sum() > 2000:
                agree = float(1.0 - (diff[far] > diff_thr).mean())
                if agree < min_agree:
                    return (None, {"agree": agree}) if return_debug else None
    m = cand.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_PX, CLOSE_PX)))
    gc_mode = GC_MODE if gc_mode is None else gc_mode
    if prior is not None and gc_mode and gc_mode != "none":
        gimg = None
        if gc_mode == "resid":
            # GrabCut on the residual to the predicted screen (board: 128 + image - prediction,
            # so black and white squares look alike; outside the board: the image itself)
            res = f32 - pred[..., None] + 128.0
            gimg = np.clip(np.where(inb[..., None], res, f32), 0, 255).astype(np.uint8)
        elif gc_mode.startswith("labraw"):
            # CIELAB with the colour axes amplified (skin is warm, the screen and walls neutral);
            # labrawN: x N, default x 2
            amp = float(gc_mode[6:] or 2)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            gimg = np.clip(np.dstack([lab[..., 0], 128 + amp * (lab[..., 1] - 128), 128 + amp * (lab[..., 2] - 128)]),
                           0, 255).astype(np.uint8)
        elif gc_mode == "lab":
            # channel 0: grey residual to the predicted screen on the board (squares alike), the
            # grey level elsewhere; channels 1-2: CIELAB a*, b* x 2 (skin is warm, the screen neutral)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            c0 = np.where(inb, g - pred + 128.0, g)
            gimg = np.clip(np.dstack([c0, 128 + 2 * (lab[..., 1] - 128), 128 + 2 * (lab[..., 2] - 128)]),
                           0, 255).astype(np.uint8)
        m = grabcut_refine(img, m > 0, prior, gc_img=gimg)
        for _ in range(GC_PASSES - 1):
            # re-centre the band on the last result (closer to the hand than the level-1 model
            # when the landmark pose is a few degrees off) and cut again with a narrower band
            p2 = dict(prior, mask=m & prior["distal"])
            m = grabcut_refine(img, m, p2, in_mm=2.0, out_mm=5.0, gc_img=gimg)
        m = m.astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    votes = np.zeros(n)
    if lm_px is not None:
        for x, y in np.asarray(lm_px, float):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < W and 0 <= yi < H:
                win = lab[max(0, yi - 3):yi + 4, max(0, xi - 3):xi + 4].ravel()
                win = win[win > 0]
                if len(win):
                    votes[np.bincount(win).argmax()] += 1
    if votes.max() > 0:
        keep = int(np.argmax(votes))
    else:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    hand = lab == keep
    # fill small holes only (missed skin pixels); real see-through gaps stay.  The hand distance
    # is unknown here: use the board distance as a (conservative, larger) proxy
    mm_px = float(np.linalg.norm(np.asarray(pose.tvec, float)) / cam.K[0, 0])
    hand = fill_small_holes(hand, HOLE_MM2 / mm_px ** 2)
    if hand.mean() > 0.35:      # the 'hand' swallowed the background: board model failed
        return (None, {"area_frac": float(hand.mean())}) if return_debug else None
    q = None
    if prior is not None:
        q = mask_agreement(hand, prior, lm_px)
        bad = q["prior_iou"] < min_prior_iou or (np.isfinite(q["landmarks_inside"])
                                                  and q["landmarks_inside"] < min_lm_inside)
        if bad:
            return (None, {"rejected": "disagrees with the model / landmarks", **q}) if return_debug else None
    if return_debug:
        return hand, {"diff": diff, "cex": cex, "inb": inb, "p": p, "agreement": q}
    return hand


def add_mask_noise(mask, rng, jitter_px=1.0, n_blobs=3):
    """Synthetic test noise: smooth random boundary displacement (+-jitter_px) and a few blobs
    stuck to / cut from the boundary."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return mask
    y0, y1, x0, x1 = max(0, ys.min() - 40), min(H, ys.max() + 40), max(0, xs.min() - 40), min(W, xs.max() + 40)
    sub = mask[y0:y1, x0:x1].astype(np.float32)
    h, w = sub.shape
    fields = []
    for _ in range(2):
        f = rng.normal(size=(max(2, h // 24), max(2, w // 24))).astype(np.float32)
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC)
        f /= max(np.abs(f).max(), 1e-6)
        fields.append(f * jitter_px)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    sub = cv2.remap(sub, gx + fields[0], gy + fields[1], cv2.INTER_LINEAR) > 0.5
    out = mask.copy()
    out[y0:y1, x0:x1] = sub
    by, bx = np.nonzero(boundary(out))
    for i in range(n_blobs):
        j = rng.integers(len(by))
        ax = (int(rng.uniform(3, 9)), int(rng.uniform(3, 7)))
        blob = np.zeros((H, W), np.uint8)
        cv2.ellipse(blob, (int(bx[j]), int(by[j])), ax, float(rng.uniform(0, 180)), 0, 360, 1, -1)
        out = (out | (blob > 0)) if i % 2 == 0 else (out & ~(blob > 0))
    return out


# ------------------------------------------------------------------ frame observations / scoring

class FrameObs:
    pass


def prepare_frames(a, takes, l1, model, work):
    """Per take: frames with camera, observed mask (work scale), 2D landmarks, ROI."""
    paths = [f for g in a.videos for f in (sorted(glob.glob(g)) or [g])]
    paths = [p for p in paths if os.path.splitext(p)[1].lower() in VIDEO_EXT]
    if not paths:
        raise SystemExit("no videos found")
    spec = BoardSpec.load(a.board_spec)
    board = make_board(spec)
    pattern, ppm, board_mm = board_pattern(spec)
    sidecar = json.load(open(a.landmarks2d)) if a.landmarks2d else None
    scans = [scan_video(p, board, a.fps, sidecar, a.mp_model, work) for p in paths]
    calibrate_and_pose(scans, a.camera)
    # match videos to takes
    names = [os.path.basename(str(t.get("input", ""))).replace("\\", "/").split("/")[-1] for t in takes]
    names = [n.split("\\")[-1] for n in names]
    match = {}
    for si, s in enumerate(scans):
        b = os.path.basename(s["path"])
        if b in names:
            match[si] = names.index(b)
    if len(match) < len(scans):
        if len(scans) == len(takes):
            match = {i: i for i in range(len(scans))}
        else:
            match = {i: min(i, len(takes) - 1) for i in range(len(scans))}
    rng = np.random.default_rng(a.seed)
    ws = a.work_scale
    out, rej, agree = [], [], []
    for si, s in enumerate(scans):
        ti = match[si]
        fit = l1_fit(l1, ti, a)
        root = np.array(fit["root"])
        x = np.array(fit["x"])
        P3 = apply_root(model.landmarks(x)[0], root, model.w0)
        usable = [r for r in s["recs"] if r["pose"] is not None]
        if len(usable) < 3:
            print(f"[frames] {os.path.basename(s['path'])}: only {len(usable)} usable frames, skipped")
            continue
        pick = np.unique(np.linspace(0, len(usable) - 1, min(a.max_frames, len(usable))).astype(int))
        want = {usable[i]["fi"]: usable[i] for i in pick}
        cam = s["cam"]
        Kw = scale_K(cam.K, ws)
        frames = []
        for fi, ts, img in iter_video(s["path"], a.fps):
            r = want.get(fi)
            if r is None:
                continue
            pose = r["pose"]
            proj3 = project(P3, cam.K, cam.dist, pose.rvec, pose.tvec)
            lm_pick = r["px"] if r["px"] is not None else proj3
            H, W = img.shape[:2]
            pr = model_prior(model, model.base_params, x, root, cam, pose, (W, H), a.roi_cut_mm) \
                if a.mask_prior else None
            res = extract_mask(img, cam, pose, pattern, ppm, board_mm, lm_pick, prior=pr, return_debug=True)
            m, dbg = res if isinstance(res, tuple) else (res, {})
            dbg = dbg or {}
            if m is None or m.sum() < 500:
                rej.append({"take": ti, "frame": fi, **{k: v for k, v in (dbg or {}).items()
                                                        if isinstance(v, (int, float, str))}})
                continue
            if dbg.get("agreement"):
                agree.append({"take": ti, "frame": fi, **dbg["agreement"]})
            if a.mask_noise > 0:
                m = add_mask_noise(m, rng, a.mask_noise, 3)
            Hw, Ww = int(round(H * ws)), int(round(W * ws))
            fo = FrameObs()
            fo.take, fo.fi, fo.video = ti, fi, s["path"]
            fo.mask = cv2.resize(m.astype(np.float32), (Ww, Hw), interpolation=cv2.INTER_AREA) > 0.5
            fo.gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (Ww, Hw), interpolation=cv2.INTER_AREA)
            fo.K, fo.dist, fo.rvec, fo.tvec = Kw, cam.dist.copy(), pose.rvec.copy(), pose.tvec.copy()
            fo.size = (Ww, Hw)
            fo.close_px = CLOSE_PX * ws
            fo.lm2d = None if r["px"] is None else (np.asarray(r["px"], float) + 0.5) * ws - 0.5
            fo.board_rms = pose.reproj_px
            frames.append(fo)
        print(f"[frames] {os.path.basename(s['path'])} -> take {ti}: {len(frames)} frames with a hand mask, "
              f"{sum(1 for q in rej if q['take'] == ti)} rejected by the mask checks")
        out.append({"take": ti, "video": s["path"], "frames": frames, "cam": cam})
    MASK_REPORT.clear()
    MASK_REPORT.update(rejected=rej, agreement=agree)
    if agree:
        iou = np.array([q["prior_iou"] for q in agree])
        print(f"[frames] mask vs level-1 model silhouette: IoU median {np.median(iou):.3f} min {iou.min():.3f}; "
              f"{len(rej)} frames rejected")
    return out


MASK_REPORT = {}


class Scorer:
    """Silhouette + landmark score of one frame (ROI fixed from the initial pose)."""

    def __init__(self, fo, P3_init, w0_init, prox_init, cut_mm, cap_mm=10.0):
        self.fo = fo
        W, H = fo.size
        pw = project(w0_init[None], fo.K, fo.dist, fo.rvec, fo.tvec)[0]
        pp = project(prox_init[None], fo.K, fo.dist, fo.rvec, fo.tvec)[0]
        u = pw - pp
        u /= max(np.linalg.norm(u), 1e-9)
        yy, xx = np.mgrid[0:H, 0:W]
        sd = (xx - pp[0]) * u[0] + (yy - pp[1]) * u[1]
        self.roi = sd >= 0
        inner = np.zeros((H, W), bool)
        inner[2:-2, 2:-2] = True
        self.cvalid = (sd >= 2.0) & inner
        R = rotvec_mat(fo.rvec)
        cen = (R @ P3_init.mean(0) + fo.tvec.ravel())
        self.mm_px = float(cen[2] / fo.K[0, 0])
        self.cap_px = cap_mm / self.mm_px
        self.hole_px = HOLE_MM2 / self.mm_px ** 2
        k = int(round(getattr(fo, "close_px", 0)))
        k = k + 1 if k % 2 == 0 else k
        self.close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) if k >= 3 else None
        obs = fill_small_holes(fo.mask & self.roi, self.hole_px)
        self.obs = obs
        oc = boundary(obs) & self.cvalid
        self.oc_yx = np.nonzero(oc)
        self.dt_obs = cv2.distanceTransform((~oc).astype(np.uint8), cv2.DIST_L2, 5)
        ys, xs = np.nonzero(obs)
        self.obox = (xs.min(), xs.max(), ys.min(), ys.max()) if len(xs) else (0, W - 1, 0, H - 1)
        self.n_obs = int(obs.sum())

    def predict_mask(self, V, F):
        fo = self.fo
        W, H = fo.size
        P2 = project(V, fo.K, fo.dist, fo.rvec, fo.tvec)
        lo_, hi_ = np.nanmin(P2, 0), np.nanmax(P2, 0)
        x0 = int(max(0, min(self.obox[0], np.floor(lo_[0])) - 6))
        x1 = int(min(W - 1, max(self.obox[1], np.ceil(hi_[0])) + 6))
        y0 = int(max(0, min(self.obox[2], np.floor(lo_[1])) - 6))
        y1 = int(min(H - 1, max(self.obox[3], np.ceil(hi_[1])) + 6))
        sub = raster(P2 - [x0, y0], F, y1 - y0 + 1, x1 - x0 + 1)
        return sub, (x0, x1, y0, y1)

    def score(self, V, F, L3=None, full_mask=False):
        sub, (x0, x1, y0, y1) = self.predict_mask(V, F)
        roi = self.roi[y0:y1 + 1, x0:x1 + 1]
        pm = sub
        if self.close_k is not None:
            pm = cv2.morphologyEx(np.pad(pm, 4).astype(np.uint8), cv2.MORPH_CLOSE, self.close_k)[4:-4, 4:-4] > 0
        pm = fill_small_holes(pm & roi, self.hole_px)
        pc = boundary(pm) & self.cvalid[y0:y1 + 1, x0:x1 + 1]
        py, px = np.nonzero(pc)
        if len(py) == 0 or len(self.oc_yx[0]) == 0:
            return {"contour_mm": 10.0, "iou": 0.0, "lm_mm": np.nan, "score": 10.0}
        d_po = np.minimum(self.dt_obs[py + y0, px + x0], self.cap_px)
        dtp = cv2.distanceTransform((~pc).astype(np.uint8), cv2.DIST_L2, 5)
        oy, ox = self.oc_yx
        ins = (oy >= y0) & (oy <= y1) & (ox >= x0) & (ox <= x1)
        d_op = np.full(len(oy), self.cap_px)
        d_op[ins] = np.minimum(dtp[oy[ins] - y0, ox[ins] - x0], self.cap_px)
        c = 0.5 * (d_po.mean() + d_op.mean()) * self.mm_px
        obs = self.obs[y0:y1 + 1, x0:x1 + 1]
        inter = (pm & obs).sum()
        union = (pm | obs).sum() + (self.n_obs - obs.sum())
        iou = inter / max(union, 1)
        lm = np.nan
        if L3 is not None and self.fo.lm2d is not None:
            p = project(L3, self.fo.K, self.fo.dist, self.fo.rvec, self.fo.tvec)
            e = np.linalg.norm(p - self.fo.lm2d, axis=1) * self.mm_px
            e = np.where(np.isfinite(e), e, 10.0)
            lm = float(np.mean(np.minimum(e, 10.0)))
        s = c + (0.05 * lm if np.isfinite(lm) else 0.0)
        out = {"contour_mm": float(c), "iou": float(iou), "lm_mm": lm, "score": float(s)}
        if full_mask:
            full = np.zeros(self.roi.shape, bool)
            full[y0:y1 + 1, x0:x1 + 1] = pm
            out["mask"] = full
        return out


# ------------------------------------------------------------------ parameter spaces

FIELD_SPEC = {  # kind, sigma
    "width_mm": ("log", 0.25), "offset_mm": ("add", 4.0), "tilt_mm": ("add", 5.0), "bulge_mm": ("add", 1.5),
}
SKIN_GROUPS = {
    "finger_mcp": [f"{f}_mcp" for f in F4], "finger_pip": [f"{f}_pip" for f in F4],
    "finger_dip": [f"{f}_dip" for f in F4], "thumb_mcp": ["thumb_mcp"], "thumb_ip": ["thumb_ip"],
}


def skin_space(groups, cs=True, scale=False):
    dims = []
    for gname in groups:
        for field in FIELD_SPEC:
            dims.append(("joint", gname, field))
    if cs:
        dims.append(("global", "cs_factor", None))
    if scale:
        dims.append(("scale", "log_scale", None))
    return dims


def apply_skin(z, dims, base):
    p = copy.deepcopy(base)
    log_s = 0.0
    for zi, (kind, name, field) in zip(z, dims):
        if kind == "joint":
            k, sig = FIELD_SPEC[field]
            for j in SKIN_GROUPS[name]:
                q = p["joints"][j]
                if k == "log":
                    q[field] = float(base["joints"][j][field] * np.exp(sig * zi))
                else:
                    q[field] = float(base["joints"][j][field] + sig * zi)
                if field == "bulge_mm":
                    q[field] = max(0.0, q[field])
                if field == "width_mm":
                    q[field] = max(6.0, q[field])
        elif kind == "global":
            p["global"]["cs_factor"] = float(np.clip(base["global"]["cs_factor"] + 0.25 * zi, 0.0, 1.0))
        elif kind == "scale":
            log_s = 0.02 * zi
    return p, log_s


POSE_SIG_DEG, ROOT_SIG_DEG, ROOT_SIG_MM = 2.5, 1.5, 1.5


def apply_pose(z, x0, root0, model):
    x = np.clip(x0 + POSE_SIG_DEG * z[:len(MEAS)], model.xlo, model.xhi)
    dR = rotvec_mat(np.radians(ROOT_SIG_DEG) * z[len(MEAS):len(MEAS) + 3])
    R0 = rotvec_mat(root0[:3])
    root = np.r_[mat_rotvec(dR @ R0), root0[3:6] + ROOT_SIG_MM * z[len(MEAS) + 3:len(MEAS) + 6], root0[6]]
    return x, root


# ------------------------------------------------------------------ level 2 optimisation

G = {}   # state shared with forked workers (read-only)


def eval_takes(params, poses, log_s=0.0, take_ids=None, full=False, detail=False):
    """Mean score over the frames of the given takes.  poses: {take: (x, root)}."""
    model, scorers = G["model"], G["scorers"]
    tot, n, per = 0.0, 0, []
    F = model.Ff if full else model.Fd
    for ti in (take_ids if take_ids is not None else list(scorers)):
        x, root = poses[ti]
        root = np.array(root, float).copy()
        root[6] = root[6] + log_s
        V = apply_root(model.mesh(params, x, full), root, model.w0)
        L3 = apply_root(model.landmarks(x)[0], root, model.w0)
        for sc in scorers[ti]:
            r = sc.score(V, F, L3)
            tot += r["score"]
            n += 1
            if detail:
                per.append({"take": ti, "frame": sc.fo.fi, **r})
    return (tot / max(n, 1), per) if detail else tot / max(n, 1)


def landmark_prior(x, root, ti, log_s=0.0):
    """Pose prior from level 1: robust mean squared distance (mm^2 / sigma^2) of the rig joints to
    the triangulated landmarks of the take (sigma = LM3D_SIGMA, capped at 3 sigma)."""
    Lobs = G.get("lobs", {}).get(ti)
    if Lobs is None:
        return 0.0
    r = np.array(root, float).copy()
    r[6] += log_s
    L = apply_root(G["model"].landmarks(x)[0], r, G["model"].w0)
    e = np.linalg.norm(L - Lobs, axis=1) / LM3D_SIGMA
    w = G["lobs_w"][ti]
    return float(np.sum(w * np.minimum(e, 3.0) ** 2) / w.sum())


LM3D_SIGMA = 2.0


def _w_pose(args):
    z, ti, params, x0, root0, reg, log_s = args
    x, root = apply_pose(np.asarray(z), x0, root0, G["model"])
    return (eval_takes(params, {ti: (x, root)}, log_s, [ti]) + reg * float(np.sum(np.square(z)))
            + G["lm3d_w"] * landmark_prior(x, root, ti, log_s))


def _w_skin(args):
    z, dims, base, poses, reg = args
    p, log_s = apply_skin(np.asarray(z), dims, base)
    return eval_takes(p, poses, log_s) + reg * float(np.sum(np.square(z)))


def run_cma(pool, make_args, worker, dim, budget, sigma0, seed, label, z_init=None, stds=None):
    import cma
    z0 = np.zeros(dim) if z_init is None else np.clip(np.asarray(z_init, float), -2.9, 2.9)
    f0 = pool.map(worker, [make_args(z0)])[0]
    best_z, best_f = z0, f0
    opts = {"bounds": [-3, 3], "maxfevals": budget, "seed": seed,
            "verbose": -9, "popsize": max(10, 4 + int(3 * np.log(dim)))}
    if stds is not None:
        opts["CMA_stds"] = list(np.asarray(stds, float))
    es = cma.CMAEvolutionStrategy(z0, sigma0, opts)
    t0, ne = time.time(), 1
    while not es.stop():
        X = es.ask()
        fs = pool.map(worker, [make_args(np.asarray(z)) for z in X])
        es.tell(X, fs)
        ne += len(X)
        i = int(np.argmin(fs))
        if fs[i] < best_f:
            best_f, best_z = fs[i], np.asarray(X[i])
    print(f"[cma] {label}: {f0:.4f} -> {best_f:.4f} ({ne} evals, {time.time() - t0:.0f} s)")
    if best_f >= f0:
        print(f"[cma] WARNING {label}: no sample beat the start ({ne} evals): the stage changed nothing "
              f"(step size too large for this objective, or already at its optimum)")
    return best_z, best_f, f0, ne


def l1_fit(l1, ti, a):
    """Level-1 fit a level-2 run starts from: the free-scale one when the scale is being fitted."""
    return l1["takes"][ti]["free_scale" if getattr(a, "fit_scale", False) is True else "fixed_scale"]


def build_scorers(a, model, l1, frames_by_take):
    poses0, scorers = {}, {}
    for tk in frames_by_take:
        ti = tk["take"]
        fit = l1_fit(l1, ti, a)
        x0, root0 = np.array(fit["x"]), np.array(fit["root"])
        poses0[ti] = (x0, root0)
        L, _, P = model.landmarks(x0)
        knuckle = P["knuckle"]
        prox = model.w0 + a.roi_cut_mm * skin.unit(model.w0 - knuckle)
        w3 = apply_root(model.w0[None], root0, model.w0)[0]
        p3 = apply_root(prox[None], root0, model.w0)[0]
        L3 = apply_root(L, root0, model.w0)
        scorers[ti] = [Scorer(fo, L3, w3, p3, a.roi_cut_mm) for fo in tk["frames"]]
    return poses0, scorers


def _w_skin_frames(args):
    z, dims, base, poses = args
    p, log_s = apply_skin(np.asarray(z), dims, base)
    _, per = eval_takes(p, poses, log_s, detail=True)
    return np.array([q["score"] for q in per])


TUNE_SETS = {
    # review C2: offsets, tilts, PIP/DIP bulges and cs_factor move the 3D skin by 0.2-0.8 mm but
    # the silhouette score by less than its noise, so by default only zone widths (+ the MCP pad
    # bulge, the largest silhouette effect) are candidates; skin_sensitivity then freezes any
    # candidate the frames cannot measure
    "widths": [("joint", g, "width_mm") for g in ("finger_mcp", "finger_pip", "finger_dip")]
    + [("joint", "finger_mcp", "bulge_mm")],
    "all": None,
}
SENS_MIN_CURV = 0.005     # mm of score per sigma^2: below this a parameter is not measurable


def skin_sensitivity(pool, dims, base, poses):
    """Profile of the score along each skin dimension at +-1 sigma (per frame).  Slope
    g = (f(+1) - f(-1)) / 2 and curvature c = (f(+1) + f(-1) - 2 f(0)) / 2 with their standard
    errors over frames; noise floor nf = SE of the per-frame score change.  measurable: the slope
    or the curvature is significant (> 2 SE) and larger than both nf and SENS_MIN_CURV.
    Confidence half-width (sigma units): sqrt(nf / c) when curved, nf / |g| otherwise."""
    zs = [np.zeros(len(dims))]
    for i in range(len(dims)):
        for sgn in (1.0, -1.0):
            z = np.zeros(len(dims))
            z[i] = sgn
            zs.append(z)
    fs = pool.map(_w_skin_frames, [(z, dims, base, poses) for z in zs])
    f0 = fs[0]
    n = len(f0)
    out = []
    for i, d in enumerate(dims):
        dp, dm = fs[1 + 2 * i] - f0, fs[2 + 2 * i] - f0
        c, g = 0.5 * (dp + dm), 0.5 * (dp - dm)
        C, seC = float(c.mean()), float(c.std(ddof=1) / np.sqrt(n))
        G, seG = float(g.mean()), float(g.std(ddof=1) / np.sqrt(n))
        nf = float(0.5 * (dp.std(ddof=1) + dm.std(ddof=1)) / np.sqrt(n))
        # measurable: a consistent (over frames) slope or curvature, larger than the noise floor
        signal = max(abs(G), max(C, 0.0))
        ok = bool((abs(G) > 2 * seG or C > 2 * seC) and signal > max(SENS_MIN_CURV, nf))
        # confidence: move whose score change equals the noise (quadratic if curved, else linear)
        hw = float(np.sqrt(nf / C)) if C > 2 * seC else (float(nf / abs(G)) if abs(G) > 0 else float("inf"))
        out.append({"dim": list(d), "curvature_per_sigma2": C, "curvature_se": seC, "slope_per_sigma": G, "slope_se": seG,
                    "noise_floor": nf, "confidence_sigma": hw, "measurable": ok,
                    "score_plus": float(fs[1 + 2 * i].mean()), "score_minus": float(fs[2 + 2 * i].mean()),
                    "score_0": float(f0.mean())})
    return out


def dim_value(params, d):
    kind, name, field = d
    if kind == "joint":
        return float(np.mean([params["joints"][j][field] for j in SKIN_GROUPS[name]]))
    if kind == "global":
        return float(params["global"]["cs_factor"])
    return 0.0


def dim_sigma_phys(base, d, hw):
    """Confidence half-width hw (sigma units) in the parameter's own unit."""
    kind, name, field = d
    if not np.isfinite(hw):
        return float("inf")
    if kind == "joint":
        k, sig = FIELD_SPEC[field]
        v = dim_value(base, d)
        return float(v * (np.exp(sig * hw) - 1)) if k == "log" else float(sig * hw)
    if kind == "global":
        return float(0.25 * hw)
    return float(0.02 * hw)


def eval_meshes(meshes, poses, take_ids=None, detail=True):
    """Score fixed meshes (board frame, e.g. the synthetic truth) with the frame scorers."""
    model, scorers = G["model"], G["scorers"]
    per = []
    for ti in (take_ids if take_ids is not None else list(scorers)):
        V, F = meshes[ti]
        x, root = poses[ti]
        L3 = apply_root(model.landmarks(x)[0], np.asarray(root, float), model.w0)
        for sc in scorers[ti]:
            per.append({"take": ti, "frame": sc.fo.fi, **sc.score(V, F, L3)})
    return float(np.mean([q["score"] for q in per])), per


def split_summary(per, fit_keys):
    fit = [q for q in per if (q["take"], q["frame"]) in fit_keys]
    held = [q for q in per if (q["take"], q["frame"]) not in fit_keys]
    return {"fit_frames": summarise_frames(fit) if fit else None,
            "held_out_frames": summarise_frames(held) if held else None}


def level2(a, model, l1, frames_by_take, out_dir):
    base = copy.deepcopy(model.base_params)
    if a.tune == "all":
        groups = ["finger_mcp", "finger_pip", "finger_dip"] + (["thumb_mcp", "thumb_ip"] if a.thumb else [])
        cand = skin_space(groups, cs=True, scale=False)
    else:
        cand = [tuple(d) for d in TUNE_SETS["widths"]]
        if a.thumb:
            cand += [("joint", "thumb_mcp", "width_mm"), ("joint", "thumb_ip", "width_mm")]
    poses0, scorers = build_scorers(a, model, l1, frames_by_take)
    # optimisation uses an evenly spread subset of frames per take; the rest are held out
    fit_sc = {ti: [scs[i] for i in np.unique(np.linspace(0, len(scs) - 1, min(a.fit_frames, len(scs))).astype(int))]
              for ti, scs in scorers.items()}
    fit_keys = {(ti, sc.fo.fi) for ti, scs in fit_sc.items() for sc in scs}
    jw = np.ones(21)
    jw[[0, 1]] = a.wrist_cmc_weight
    lobs_w = {}
    for ti in poses0:
        wt = jw.copy()
        wt[[NI[n] for n in l1["takes"][ti]["fixed_scale"].get("dropped_joints", [])]] = 0.0
        lobs_w[ti] = wt
    G.update(model=model, scorers=scorers, lm3d_w=a.lm3d_weight, lobs_w=lobs_w,
             lobs={ti: np.array(l1["takes"][ti]["observed_landmarks"]) for ti in poses0})
    before, per_before = eval_takes(base, poses0, 0.0, detail=True)
    print(f"[level2] before (all {len(per_before)} frames, {len(fit_keys)} used for fitting): score {before:.4f}")
    truth_score, per_truth, tr = None, None, None
    if getattr(a, "truth", None):
        tr = json.load(open(a.truth))
        tposes = {ti: (np.array(tr["takes"][ti]["x"])[:len(MEAS)], np.array(tr["takes"][ti]["root"])) for ti in poses0}
        if tr.get("true_mesh_npz"):
            z = np.load(tr["true_mesh_npz"])
            meshes = {ti: (z[f"V{ti}"], z["F"]) for ti in poses0}
            _, per_truth = eval_meshes(meshes, tposes)
        else:
            _, per_truth = eval_takes(tr["true_params"], tposes, 0.0, full=True, detail=True)
        truth_score = summarise_frames(per_truth)
        print(f"[level2] noise floor (true hand + true pose through the same pipeline, family "
              f"{tr.get('truth_family', 'param')}): score {truth_score['score']:.4f}, contour {truth_score['contour_mm']:.3f} mm")
    G.update(scorers=fit_sc)
    state_path = os.path.join(out_dir, "level2_state.json")
    sig = file_key(SCORE_VERSION, CLOSE_PX, HOLE_MM2, a.rig, a.relaxed, sorted(a.videos), a.fit_frames, a.budget_pose,
                   a.budget_skin, a.thumb, a.lm3d_weight, a.fit_scale, a.reg_pose, a.reg_skin, a.seed, a.mask_noise,
                   a.work_scale, a.decimate, a.params or "", a.tune, a.pose_sigma, a.root_std, GC_MODE,
                   MASK_MIN_PRIOR_IOU)
    params, log_s = base, 0.0
    dims, z_skin, sens = None, None, None
    z_pose = {ti: np.zeros(len(MEAS) + 6) for ti in poses0}
    poses = dict(poses0)
    hist, pose_only, start, warns = [], None, 0, []
    if os.path.exists(state_path) and not a.no_resume:
        st = json.load(open(state_path))
        if st.get("sig") == sig:
            start = st["round"]
            dims = [tuple(d) for d in st["dims"]]
            z_skin = np.array(st["z_skin"])
            sens = st.get("sensitivity")
            z_pose = {int(k): np.array(v) for k, v in st["z_pose"].items()}
            hist, pose_only, warns = st["history"], st.get("pose_only"), st.get("warnings", [])
            params, log_s = apply_skin(z_skin, dims, base)
            poses = {ti: apply_pose(z_pose[ti], *poses0[ti], model) for ti in poses0}
            print(f"[level2] resumed after round {start} from {state_path}")
    pose_stds = np.r_[np.ones(len(MEAS)), np.full(6, a.root_std)]
    ctx = mp.get_context("fork")
    pool = ctx.Pool(a.workers)
    try:
        for rnd in range(start, a.rounds):
            for ti in sorted(poses):
                x0, root0 = poses0[ti]
                bz, bf, f0, ne = run_cma(
                    pool, lambda z, ti=ti, x0=x0, root0=root0: (z, ti, params, x0, root0, a.reg_pose, log_s),
                    _w_pose, len(MEAS) + 6, a.budget_pose, a.pose_sigma if rnd == 0 else 0.7 * a.pose_sigma,
                    a.seed + rnd * 10 + ti, f"round {rnd + 1} pose take {ti}", z_init=z_pose[ti], stds=pose_stds)
                if bf >= f0:
                    warns.append(f"round {rnd + 1} pose take {ti}: CMA returned the start (no sample better)")
                z_pose[ti] = bz
                poses[ti] = apply_pose(bz, x0, root0, model)
                hist.append({"round": rnd + 1, "stage": f"pose take {ti}", "before": f0, "after": bf, "evals": ne})
            if rnd == 0:
                G.update(scorers=scorers)
                pose_only = eval_takes(base, poses, 0.0)
                G.update(scorers=fit_sc)
                print(f"[level2] after pose-only refinement (default skin, all frames): {pose_only:.4f}")
                # which skin parameters can the frames measure? (review C2)
                t0 = time.time()
                sens = skin_sensitivity(pool, cand, base, poses)
                for q in sens:
                    print(f"[sens] {q['dim'][1]}.{q['dim'][2] or ''}: curvature {q['curvature_per_sigma2']:+.4f} "
                          f"+- {q['curvature_se']:.4f} mm/sigma^2, slope {q['slope_per_sigma']:+.4f} +- {q['slope_se']:.4f}, "
                          f"confidence +-{q['confidence_sigma']:.2f} sigma -> "
                          f"{'TUNED' if q['measurable'] else 'frozen (not measurable)'}")
                print(f"[sens] {len(cand)} candidates profiled in {time.time() - t0:.0f} s")
                dims = [tuple(q["dim"]) for q in sens if q["measurable"]]
                if a.fit_scale:
                    dims.append(("scale", "log_scale", None))
                z_skin = np.zeros(len(dims))
            if dims:
                bz, bf, f0, ne = run_cma(pool, lambda z: (z, dims, base, poses, a.reg_skin), _w_skin, len(dims),
                                         a.budget_skin, 0.6 if rnd == 0 else 0.35, a.seed + 100 + rnd,
                                         f"round {rnd + 1} skin", z_init=z_skin)
                z_skin = bz
                params, log_s = apply_skin(z_skin, dims, base)
                hist.append({"round": rnd + 1, "stage": "skin", "before": f0, "after": bf, "evals": ne})
            else:
                print("[level2] no skin parameter is measurable from these frames: skin left at the defaults")
                if "no measurable skin parameter" not in " ".join(warns):
                    warns.append("no measurable skin parameter: skin left at the defaults")
            json.dump({"sig": sig, "round": rnd + 1, "z_skin": list(map(float, z_skin)), "dims": dims,
                       "log_scale": log_s, "sensitivity": sens, "warnings": warns,
                       "z_pose": {str(t): z.tolist() for t, z in z_pose.items()}, "pose_only": pose_only,
                       "history": hist}, open(state_path, "w"), indent=1)
    finally:
        pool.close()
        pool.join()
    G.update(scorers=scorers)
    after, per_after = eval_takes(params, poses, log_s, detail=True)
    print(f"[level2] after (all frames): score {after:.4f}")
    split = {"before": split_summary(per_before, fit_keys), "after": split_summary(per_after, fit_keys)}
    if per_truth is not None:
        split["truth"] = split_summary(per_truth, fit_keys)
    ha, hb = split["after"]["held_out_frames"], split["before"]["held_out_frames"]
    fa = split["after"]["fit_frames"]
    if ha and hb:
        print(f"[level2] held-out frames: {hb['score']:.4f} -> {ha['score']:.4f}; fit frames -> {fa['score']:.4f}")
        if ha["score"] > hb["score"]:
            warns.append(f"held-out frames got worse ({hb['score']:.4f} -> {ha['score']:.4f}): the pose/skin "
                         f"refinement fitted the fit frames only and does not generalise")
    if per_truth is not None:
        ht = split["truth"]["held_out_frames"] or split["truth"]["fit_frames"]
        hx = ha or fa
        if hx["score"] < ht["score"]:
            warns.append(f"tuned score {hx['score']:.4f} is BELOW the true hand's {ht['score']:.4f} on "
                         f"{'held-out' if ha else 'fit'} frames: the fit explains noise / pipeline bias (overfit), "
                         f"the tuned values are not measurements")
    for w in warns:
        print("[level2] WARNING " + w)
    conf = {}
    for q in sens or []:
        d = tuple(q["dim"])
        key = f"{d[1]}.{d[2]}" if d[2] else d[1]
        conf[key] = {"default": dim_value(base, d), "tuned": dim_value(params, d) if q["measurable"] else None,
                     "measurable": q["measurable"], "confidence_half_width": dim_sigma_phys(base, d, q["confidence_sigma"]),
                     "curvature_mm_per_sigma2": q["curvature_per_sigma2"], "curvature_se": q["curvature_se"]}
    return {"before": before, "pose_only": pose_only, "after": after, "per_frame_before": per_before,
            "per_frame_after": per_after, "params": params, "z_skin": list(map(float, z_skin if z_skin is not None else [])),
            "dims": dims, "log_scale": log_s, "poses0": poses0, "poses": poses, "history": hist,
            "truth_score": truth_score, "split": split, "sensitivity": sens, "confidence": conf, "warnings": warns}


def summarise_frames(per):
    k = ["contour_mm", "iou", "lm_mm", "score"]
    return {kk: float(np.nanmean([p[kk] for p in per])) for kk in k} | {
        "contour_mm_median": float(np.median([p["contour_mm"] for p in per])), "frames": len(per)}


# ------------------------------------------------------------------ picture

def overlay_picture(model, scorers_frames, cases, path, title, note=None):
    """cases: list of (label, params, poses, log_s).  Rows = frames, columns: whole hand for the
    first case, then a zoom on the folded fingers for every case.  In the zoom, pixels where the
    folded scan hand and the video silhouette disagree are coloured."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    rows, cols = len(scorers_frames), 1 + len(cases)
    fig, axs = plt.subplots(rows, cols, figsize=(4.4 * cols, 5.0 * rows + 1.0), squeeze=False)
    fing = [NI[f"{f}_{j}"] for f in F4 for j in ("mcp", "pip", "dip", "tip")] + [NI["thumb_ip"], NI["thumb_tip"]]
    for r, (ti, sc) in enumerate(scorers_frames):
        fo = sc.fo
        g = fo.gray.astype(float) / 255.0
        ys, xs = np.nonzero(sc.obs)
        X0, X1 = max(0, xs.min() - 20), min(fo.size[0], xs.max() + 20)
        Y0, Y1 = max(0, ys.min() - 20), min(fo.size[1], ys.max() + 20)
        res_all = []
        for (label, params, poses, log_s) in cases:
            x, root = poses[ti]
            root = np.array(root, float).copy()
            root[6] += log_s
            V = apply_root(model.mesh(params, x, False), root, model.w0)
            L3 = apply_root(model.landmarks(x)[0], root, model.w0)
            res_all.append((sc.score(V, model.Fd, L3, full_mask=True), L3))
        # zoom window: the four fingers and the thumb tip of the first case
        q = project(res_all[0][1][fing], fo.K, fo.dist, fo.rvec, fo.tvec)
        q = q[np.isfinite(q).all(1)]
        cx0, cy0 = q.min(0) - 14
        cx1, cy1 = q.max(0) + 14
        side = max(cx1 - cx0, cy1 - cy0)
        cxm, cym = 0.5 * (cx0 + cx1), 0.5 * (cy0 + cy1)
        zx0, zx1 = int(max(0, cxm - side / 2)), int(min(fo.size[0], cxm + side / 2))
        zy0, zy1 = int(max(0, cym - side / 2)), int(min(fo.size[1], cym + side / 2))
        ax = axs[r, 0]
        rgb = np.dstack([g * 0.8] * 3)
        om = sc.obs
        rgb[om] = rgb[om] * 0.5 + np.array([0.2, 0.5, 0.9]) * 0.5
        ax.imshow(rgb[Y0:Y1, X0:X1], interpolation="nearest")
        ax.add_patch(Rectangle((zx0 - X0, zy0 - Y0), zx1 - zx0, zy1 - zy0, fill=False, ec="k", lw=1.2, ls="--"))
        ax.set_title(f"Video frame (take {ti + 1}, frame {fo.fi})\nblue = hand silhouette found", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for c, ((label, *_), (res, _)) in enumerate(zip(cases, res_all)):
            ax = axs[r, c + 1]
            pm = res["mask"][zy0:zy1, zx0:zx1]
            ob = sc.obs[zy0:zy1, zx0:zx1]
            gz = g[zy0:zy1, zx0:zx1]
            rgb = np.dstack([0.35 + 0.5 * gz] * 3)
            both = pm & ob
            rgb[both] = rgb[both] * 0.35 + np.array([0.62, 0.62, 0.62]) * 0.65
            rgb[ob & ~pm] = (0.10, 0.45, 0.95)
            rgb[pm & ~ob] = (0.95, 0.25, 0.10)
            ax.imshow(rgb, interpolation="nearest")
            ax.set_title(f"{label}\ncontour error {res['contour_mm']:.2f} mm, IoU {res['iou']:.3f}", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title, fontsize=14)
    handles = [Patch(color=(0.62, 0.62, 0.62), label="both agree (hand)"),
               Patch(color=(0.10, 0.45, 0.95), label="video only (model too thin)"),
               Patch(color=(0.95, 0.25, 0.10), label="model only (model too fat)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, 0.025 if note else 0.005))
    if note:
        fig.text(0.5, 0.004, note, ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), h_pad=2.0)
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print("WROTE", path)


# ------------------------------------------------------------------ run (real-data CLI)

def run(a):
    out_dir = a.out
    work = os.path.join(WORK, file_key(os.path.abspath(out_dir))[:8])
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    model = HandModel3D(a.rig, a.params, a.decimate)
    print(f"[rig] {a.rig}: {len(model.Vf)} verts, compare mesh {len(model.Vd)} verts ({time.time() - t0:.1f} s)")
    relaxed = json.load(open(a.relaxed))
    takes = relaxed["takes"]
    jw = np.ones(21)
    jw[[0, 1]] = a.wrist_cmc_weight
    l1 = level1(model, takes, a.landmark_key, out_dir, jw)
    if a.fit_scale == "auto":
        a.fit_scale = bool(l1["summary"]["scale_mismatch_flag"])
        if a.fit_scale:
            print("[level2] level 1 flags a scale mismatch: the scan scale is fitted too (--fit-scale auto)")
    else:
        a.fit_scale = a.fit_scale == "on"
    if not a.videos:
        return {"level1": l1}
    frames_by_take = prepare_frames(a, takes, l1, model, work)
    frames_by_take = [t for t in frames_by_take if t["frames"]]
    if not frames_by_take:
        raise SystemExit("no frames with a hand mask")
    r2 = level2(a, model, l1, frames_by_take, out_dir)
    sb, sa = summarise_frames(r2["per_frame_before"]), summarise_frames(r2["per_frame_after"])
    tuned = r2["params"]
    doc = {"_note": ("tuned by compare.py level 2 (CMA over shared finger groups, regularised to the "
                     "defaults); base = " + str(a.params or skin.PARAMS_PATH) + ".  Only the parameters in "
                     "_confidence marked measurable were changed; they are best fits to the silhouettes "
                     "+- the given half-width, not measurements of the hand.  Everything else is the default."),
           "_confidence": r2["confidence"], "_warnings": r2["warnings"],
           "global": tuned["global"], "joints": tuned["joints"]}
    json.dump(doc, open(os.path.join(out_dir, "skin_params_tuned.json"), "w"), indent=1)
    poses_out = {}
    for ti, (x, root) in r2["poses"].items():
        pj = pose_json(model.angles(x), np.r_[root[:6], root[6] + r2["log_scale"]], model.w0)
        poses_out[str(ti)] = pj
    res = {
        "level2": {"before": sb, "pose_only_default_skin_score": r2["pose_only"], "after": sa,
                   "noise_floor_true_model": r2["truth_score"],
                   "fit_vs_held_out": r2["split"], "sensitivity": r2["sensitivity"],
                   "parameter_confidence": r2["confidence"], "warnings": r2["warnings"],
                   "lens": LENS_INFO,
                   "mask_checks": {"rejected_frames": MASK_REPORT.get("rejected", []),
                                   "agreement_per_frame": MASK_REPORT.get("agreement", [])},
                   "history": r2["history"], "skin_dims": r2["dims"], "z_skin": r2["z_skin"],
                   "log_scale": r2["log_scale"],
                   "scale": float(np.exp(np.median([p[1][6] for p in r2["poses"].values()]) + r2["log_scale"])),
                   "scale_note": "factor the scan must be multiplied by to match the video (1 = scan size right)",
                   "per_frame_before": r2["per_frame_before"], "per_frame_after": r2["per_frame_after"],
                   "refined_poses": poses_out,
                   "pose_change_deg": {str(ti): {k: float(model.angles(r2["poses"][ti][0])[k]
                                                         - model.angles(r2["poses0"][ti][0])[k]) for k in MEAS}
                                       for ti in r2["poses"]}},
        "settings": {k: v for k, v in vars(a).items() if k not in ("func",)},
    }
    json.dump(res, open(os.path.join(out_dir, "level2_surface.json"), "w"), indent=1, default=float)
    print(f"[level2] contour {sb['contour_mm']:.3f} -> {sa['contour_mm']:.3f} mm, IoU {sb['iou']:.4f} -> "
          f"{sa['iou']:.4f}, landmarks {sb['lm_mm']:.3f} -> {sa['lm_mm']:.3f} mm")
    # picture: 3 frames spread over the first take
    sc_all = [(ti, sc) for ti, scs in G["scorers"].items() for sc in scs]
    pick = [sc_all[i] for i in np.unique(np.linspace(0, len(sc_all) - 1, 3).astype(int))]
    cases = [("Before: default skin, landmark pose", model.base_params, r2["poses0"], 0.0),
             ("After: tuned skin + refined pose", tuned, r2["poses"], r2["log_scale"])]
    overlay_picture(model, pick, cases, os.path.join(out_dir, "before_after.png"),
                    a.picture_title or "Folded scan hand vs video silhouette", a.picture_note)
    print(f"[run] total {time.time() - t0:.0f} s")
    return {"level1": l1, "level2": res, "r2": r2}


def add_run_args(p):
    p.add_argument("--rig", required=True, help="rigged scan .blend (SPEC layout)")
    p.add_argument("--relaxed", required=True, help="hand_relaxed.json (relaxed_pose.py output)")
    p.add_argument("--videos", nargs="*", default=[], help="glob(s) of the relaxed videos")
    p.add_argument("--board-spec", default=BOARD_SPEC)
    p.add_argument("--out", default=os.path.join(OUT, "real"))
    p.add_argument("--params", default=None, help="skin_params.json to start from (default: skin module's)")
    p.add_argument("--landmark-key", default="model_landmarks_board_mm",
                   help="take key with the 21 landmarks (model_landmarks_board_mm or landmarks_board_mm)")
    p.add_argument("--wrist-cmc-weight", type=float, default=0.5,
                   help="weight of the wrist and thumb CMC landmarks (least anatomical in MediaPipe)")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--camera", default=None, help="fixed camera.json instead of calibrating the video lens")
    p.add_argument("--mp-model", default=None, help="MediaPipe hand_landmarker.task")
    p.add_argument("--landmarks2d", default=None, help="sidecar json {video basename: {frame: 21x2 px}}")
    p.add_argument("--max-frames", type=int, default=30, help="frames per take compared (before/after report)")
    p.add_argument("--fit-frames", type=int, default=12, help="of these, frames per take used inside the optimiser")
    p.add_argument("--no-resume", action="store_true", help="ignore level2_state.json from an earlier run")
    p.add_argument("--truth", default=None, help="(synthetic test) truth.json: also score the true skin + pose")
    p.add_argument("--work-scale", type=float, default=0.5, help="image scale of the silhouette comparison")
    p.add_argument("--decimate", type=float, default=0.2, help="mesh ratio used for the comparison")
    p.add_argument("--roi-cut-mm", type=float, default=20.0, help="arm cut, mm proximal of the wrist")
    p.add_argument("--mask-noise", type=float, default=0.0, help="(tests) boundary jitter px + blobs")
    p.add_argument("--no-mask-prior", dest="mask_prior", action="store_false",
                   help="silhouettes from the board difference only (no GrabCut around the level-1 model, "
                        "no per-frame model/landmark agreement check)")
    p.add_argument("--fit-scale", nargs="?", const="on", default="auto", choices=["on", "off", "auto"],
                   help="fit one global scan scale at level 2, starting from the level-1 free-scale fit "
                        "(auto: only when level 1 flags a scale mismatch)")
    p.add_argument("--thumb", action="store_true", help="also tune the thumb MCP/IP groups")
    p.add_argument("--tune", choices=["widths", "all"], default="widths",
                   help="candidate skin parameters (widths: finger zone widths + MCP pad bulge; all: also "
                        "offsets, tilts, PIP/DIP bulges, cs_factor).  Candidates the frames cannot measure "
                        "(skin_sensitivity) are frozen at the defaults either way")
    p.add_argument("--pose-sigma", type=float, default=0.05,
                   help="CMA step of the pose stage (x 2.5 deg per joint; the old 0.3 never improved)")
    p.add_argument("--root-std", type=float, default=0.3, help="root rotation/translation step relative to the joints")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--budget-pose", type=int, default=250)
    p.add_argument("--budget-skin", type=int, default=400)
    p.add_argument("--reg-pose", type=float, default=0.004)
    p.add_argument("--lm3d-weight", type=float, default=0.1,
                   help="weight of the level-1 landmark prior on the pose refinement (per (2 mm)^2)")
    p.add_argument("--reg-skin", type=float, default=0.004)
    p.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--picture-title", default=None)
    p.add_argument("--picture-note", default=None)


# ------------------------------------------------------------------ synthetic test

SYN = os.path.join(WORK, "synth")
SYN_OUT = os.path.join(OUT, "synthetic")
SYN_SIZE = (1080, 1920)
SYN_TAKES = [
    {"index_mcp": 68, "index_pip": 43, "index_dip": 25, "middle_mcp": 69, "middle_pip": 68, "middle_dip": 23,
     "ring_mcp": 65, "ring_pip": 78, "ring_dip": 34, "pinky_mcp": 54, "pinky_pip": 85, "pinky_dip": 30,
     "spread_index_middle": 5.0, "spread_middle_ring": 4.0, "spread_ring_pinky": 7.0,
     "thumb_cmc": 36.0, "thumb_cmc_elev": 22.0, "thumb_mcp": 14.0, "thumb_ip": 6.0,
     "arch_imr": 4.0, "arch_mrp": 6.0},
    {"index_mcp": 63, "index_pip": 47, "index_dip": 21, "middle_mcp": 72, "middle_pip": 64, "middle_dip": 27,
     "ring_mcp": 66, "ring_pip": 81, "ring_dip": 30, "pinky_mcp": 58, "pinky_pip": 82, "pinky_dip": 34,
     "spread_index_middle": 7.0, "spread_middle_ring": 3.0, "spread_ring_pinky": 8.0,
     "thumb_cmc": 33.0, "thumb_cmc_elev": 25.0, "thumb_mcp": 18.0, "thumb_ip": 3.0,
     "arch_imr": 5.0, "arch_mrp": 7.0},
]
SYN_FRAMES_PER_TAKE = 26
SYN_LM_NOISE_PX = 1.5


def true_params(base, seed=7):
    """'True' skin of the synthetic test.  seed 7: the fixed set (clearly different from the
    defaults: narrower zones, shifted creases, bigger pads, less smoothing, finger widths
    individually jittered).  Other seeds: a random set around the defaults (each group's width
    x exp(N(0, 0.25)), offset +N(0, 3), tilt +N(0, 4), bulge +N(0, 1.2) mm, cs_factor U(0.2, 0.8))."""
    p = copy.deepcopy(base)
    rng = np.random.default_rng(seed)
    if seed != 7:
        for j in ("mcp", "pip", "dip"):
            dw, do, dt, db = np.exp(rng.normal(0, 0.25)), rng.normal(0, 3), rng.normal(0, 4), rng.normal(0, 1.2)
            for f in F4:
                q = p["joints"][f"{f}_{j}"]
                q["width_mm"] = float(max(8.0, q["width_mm"] * dw * rng.uniform(0.95, 1.05)))
                q["offset_mm"] = float(q["offset_mm"] + do)
                q["tilt_mm"] = float(q["tilt_mm"] + dt)
                q["bulge_mm"] = float(max(0.0, q["bulge_mm"] + db))
        p["global"]["cs_factor"] = float(rng.uniform(0.2, 0.8))
        return p
    tgt = {"mcp": dict(width_mm=44.0, offset_mm=7.0, tilt_mm=4.0, bulge_mm=4.2),
           "pip": dict(width_mm=30.0, offset_mm=-1.5, tilt_mm=1.0, bulge_mm=1.9),
           "dip": dict(width_mm=24.0, offset_mm=1.5, tilt_mm=0.0, bulge_mm=1.2)}
    for f in F4:
        for j, d in tgt.items():
            q = p["joints"][f"{f}_{j}"]
            q.update(d)
            q["width_mm"] = float(d["width_mm"] * rng.uniform(0.93, 1.07))
    p["global"]["cs_factor"] = 0.25
    return p


def synth_camera():
    cj = json.load(open(CAMERA_JSON))
    K0, d0 = np.array(cj["K"]), np.array(cj["dist"])
    s = SYN_SIZE[1] / cj["size"][0]           # long side of the sensor -> 1920 px
    K = np.array([[K0[1, 1] * s, 0, K0[1, 2] * s - (cj["size"][1] * s - SYN_SIZE[0]) / 2],
                  [0, K0[0, 0] * s, K0[0, 2] * s],
                  [0, 0, 1.0]])
    dist = np.array([d0[0], d0[1], d0[3], d0[2], d0[4]])   # portrait: tangential terms swap
    return Camera(K, dist, SYN_SIZE)


def look_at(pos, target, roll_deg, board_x=np.array([1.0, 0, 0])):
    z = skin.unit(target - pos)
    y = board_x - (board_x @ z) * z       # image down = board +X (portrait over a landscape screen)
    y = skin.unit(y)
    y = rotvec_mat(z * np.radians(roll_deg)) @ y
    x = np.cross(y, z)
    R = np.vstack([x, y, z])
    return mat_rotvec(R).reshape(3, 1), (-R @ pos).reshape(3, 1)


def arm_tube(model, length=300.0, n_ring=36, n_len=24):
    """Forearm continuing the scan's stub (stand-in: ends 40 mm proximal of the wrist)."""
    V = model.Vf
    rig = model.rig
    along, across, nrm = rig.along, rig.across, rig.n
    s = (V - model.w0) @ along
    sl = V[(s > -38) & (s < -34)]
    a_ = (sl - model.w0) @ across
    b_ = (sl - model.w0) @ nrm
    ca, cb = 0.5 * (a_.max() + a_.min()), 0.5 * (b_.max() + b_.min())
    ra, rb = 0.5 * (a_.max() - a_.min()) * 0.97, 0.5 * (b_.max() - b_.min()) * 0.97
    th = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    rows = []
    for i, t in enumerate(np.linspace(-36.0, -36.0 - length, n_len)):
        k = 1.0 + 0.12 * (i / (n_len - 1))
        rows.append(model.w0 + t * along + (ca + k * ra * np.cos(th))[:, None] * across
                    + (cb + k * rb * np.sin(th))[:, None] * nrm)
    Va = np.concatenate(rows)
    F = []
    for i in range(n_len - 1):
        for j in range(n_ring):
            a0, a1 = i * n_ring + j, i * n_ring + (j + 1) % n_ring
            b0, b1 = a0 + n_ring, a1 + n_ring
            F += [[a0, b0, a1], [a1, b0, b1]]
    return Va, np.array(F)


SKIN_LOOKS = {   # synthetic hand appearance: BGR colour, light direction (board frame, None = headlight), ambient
    "orange_headlight": dict(col=(112, 148, 205), light=None, amb=0.45),
    "light_skin": dict(col=(185, 200, 235), light=None, amb=0.45),
    "dark_skin": dict(col=(55, 75, 105), light=None, amb=0.45),
    "orange_side_light": dict(col=(112, 148, 205), light=(0.8, 0.0, -0.6), amb=0.25),
    "light_side_light": dict(col=(185, 200, 235), light=(0.8, 0.0, -0.6), amb=0.25),
    "dark_side_light": dict(col=(55, 75, 105), light=(0.8, 0.0, -0.6), amb=0.25),
    "medium_top_light": dict(col=(95, 125, 170), light=(0.0, -0.6, -0.8), amb=0.3),
}


def render_synthetic(cam, rvec, tvec, Vh, Fh, Va, Fa, pattern, ppm, board_mm, rng, distractors, look=None,
                     shadow=0.0, shadow_off=(25, 15)):
    W, H = cam.size
    XY, front = board_plane_coords(cam.K, cam.dist, (W, H), rvec, tvec)
    inb = front & (XY[..., 0] >= 0) & (XY[..., 1] >= 0) & (XY[..., 0] <= board_mm[0]) & (XY[..., 1] <= board_mm[1])
    bez = front & (XY[..., 0] >= -15) & (XY[..., 1] >= -15) & (XY[..., 0] <= board_mm[0] + 15) & \
        (XY[..., 1] <= board_mm[1] + 35)
    p = cv2.remap(pattern, XY[..., 0] * ppm[0] - 0.5, XY[..., 1] * ppm[1] - 0.5, cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=0.5)
    # screen: black 28, white 228 with a slow brightness falloff and a slight blue tint
    fall = 1.0 - 0.12 * np.clip(np.hypot(XY[..., 0] - board_mm[0] / 2, XY[..., 1] - board_mm[1] / 2) / 250, 0, 1)
    scr = (28 + 200 * p * fall)[..., None] * np.array([1.04, 1.0, 0.97])[None, None]
    wall = np.array([138, 146, 150], float) + 10 * np.sin(XY[..., :1] / 70.0) * np.cos(XY[..., 1:] / 90.0)
    img = np.where(inb[..., None], scr, np.where(bez[..., None], 18.0, wall))
    img = np.where(front[..., None], img, 120.0)
    # distractors on the wall (coloured objects, not the hand)
    for (cx, cy, ax, by, col) in distractors:
        pts = np.array([[cx + ax * np.cos(t), cy + by * np.sin(t), 0] for t in np.linspace(0, 2 * np.pi, 40)])
        q = project(pts, cam.K, cam.dist, rvec, tvec)
        if not np.isfinite(q).all():
            continue
        q = q.astype(np.int32)
        tmp = np.zeros((H, W), np.uint8)
        cv2.fillPoly(tmp, [q], 1)
        img[tmp > 0] = col
    # hand + arm, 2x supersampled with a z-buffer, Lambert shading
    V = np.concatenate([Vh, Va])
    F = np.concatenate([Fh, Fa + len(Vh)])
    if shadow > 0:      # soft cast shadow of hand + arm on the screen
        sm = raster(project(V, cam.K, cam.dist, rvec, tvec), F, H, W).astype(np.float32)
        sm = cv2.warpAffine(sm, np.float32([[1, 0, shadow_off[0]], [0, 1, shadow_off[1]]]), (W, H))
        img = img * (1 - shadow * cv2.GaussianBlur(sm, (0, 0), 6))[..., None]
    K2 = scale_K(cam.K, 2.0)
    P2 = project(V, K2, cam.dist, rvec, tvec)
    R = rotvec_mat(rvec)
    Vc = V @ R.T + tvec.ravel()
    cov, fid = raster(P2, F, 2 * H, 2 * W, depth=Vc[:, 2])
    t = Vc[F]
    fn = skin.unit(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]) + 1e-12)
    vd = skin.unit(t.mean(1))
    lk = SKIN_LOOKS["orange_headlight"] if look is None else (SKIN_LOOKS[look] if isinstance(look, str) else look)
    amb = lk.get("amb", 0.45)
    if lk.get("light") is None:
        shade = amb + (1 - amb) * np.abs(np.sum(fn * vd, 1))
    else:
        ld = skin.unit(R @ np.asarray(lk["light"], float))
        fnc = np.where((np.sum(fn * vd, 1) > 0)[:, None], -fn, fn)    # normal facing the camera
        shade = amb + (1 - amb) * np.clip(np.sum(fnc * ld, 1), 0, 1)
    col = np.array(lk["col"], float)
    hand2 = np.zeros((2 * H, 2 * W, 3), np.float32)
    ok = fid >= 0
    hand2[ok] = (shade[fid[ok]][:, None] * col[None]).astype(np.float32)
    alpha = cv2.resize(ok.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    hcol = cv2.resize(hand2, (W, H), interpolation=cv2.INTER_AREA) / np.maximum(alpha, 1e-6)[..., None]
    img = img * (1 - alpha[..., None]) + hcol * alpha[..., None]
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.6)
    img = img + rng.normal(0, 2.0, img.shape)
    # true silhouette of the hand mesh alone (no arm), for the extraction check
    Ph = project(Vh, cam.K, cam.dist, rvec, tvec)
    true_hand = raster(Ph, Fh, H, W)
    Pall = project(V, cam.K, cam.dist, rvec, tvec)
    true_all = raster(Pall, F, H, W)
    return np.clip(img, 0, 255).astype(np.uint8), true_hand, true_all


def synth_render(a):
    os.makedirs(SYN, exist_ok=True)
    os.makedirs(os.path.join(SYN, "photos", "relaxed"), exist_ok=True)
    t0 = time.time()
    model = HandModel3D(a.rig, None, 1.0)
    base = model.base_params
    tp = true_params(base, a.true_seed)
    truth_model = model
    if a.family == "autoweights":
        # out-of-family truth (review C4): Blender bone-heat weights of the same mesh, plain LBS,
        # no bulge keys, no corrective smooth -- a different deformer from the fitted model
        truth_model = HandModel3D(skin.AUTO_BLEND, {"global": {"weights": "blend", "dqs": False, "cs_factor": 0.0}}, 1.0)
        if len(truth_model.Vf) != len(model.Vf):
            raise SystemExit("autoweights rig has another mesh; rebuild it with skin.py build --auto")
        tp = truth_model.base_params
    looks = (a.looks or "orange_headlight").split(",")
    cam = synth_camera()
    spec = BoardSpec.load(BOARD_SPEC)
    pattern, ppm, board_mm = board_pattern(spec)
    real = json.load(open(REAL_EXTRACT))["takes_model_landmarks_board_mm"]
    Va0, Fa = arm_tube(model)
    rng = np.random.default_rng(11)
    truth = {"true_params": tp, "takes": [], "camera": cam.to_dict(), "lm_noise_px": SYN_LM_NOISE_PX,
             "truth_family": a.family, "true_seed": a.true_seed, "looks": looks,
             "true_mesh_npz": os.path.join(SYN, "true_meshes.npz")}
    true_meshes = {"F": model.Ff}
    sidecar = {}
    for ti, ang in enumerate(SYN_TAKES):
        x = model.x_from_angles(ang)       # same angles for an out-of-family truth (same skeleton)
        L = model.landmarks(x)[0]
        Lreal = np.array([real[ti][n] for n in NAMES])
        R, c = kabsch(L[PALM_IDX] - model.w0, Lreal[PALM_IDX])
        root = np.r_[mat_rotvec(R), c - model.w0, 0.0]
        V = apply_root(truth_model.mesh(tp, x, full=True), root, model.w0)
        # keep the hand in front of the screen (Z < -25) and centred over the board
        shift = np.array([board_mm[0] / 2 - V[:, 0].mean(), board_mm[1] / 2 + 20 - V[:, 1].mean(),
                          min(0.0, -25.0 - V[:, 2].max())])
        root[3:6] += shift
        V = V + shift
        L3 = apply_root(L, root, model.w0)
        Va = apply_root(Va0, root, model.w0)
        true_meshes[f"V{ti}"] = V
        centre = V.mean(0)
        n = SYN_FRAMES_PER_TAKE
        vid = os.path.join(SYN, "photos", "relaxed", f"relaxed_take{ti + 1}.mp4")
        vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 10, SYN_SIZE)
        cams, side, ious = [], {}, []
        distract = [(30 + 300 * rng.random(), -60 - 30 * rng.random(), 18, 9, (60, 60, 190)),
                    (board_mm[0] + 40, 60 + 150 * rng.random(), 12, 30, (40, 150, 200))]
        for i in range(n):
            u = i / (n - 1)
            az = np.radians(-45 + 90 * u + (8 if ti else 0))
            el = np.radians(-28 + 20 * np.sin(2 * np.pi * u + ti))
            d = 300 + 100 * (0.5 + 0.5 * np.sin(3 * np.pi * u + 1.0 + ti))
            dirv = np.array([np.sin(az) * np.cos(el), np.sin(el), -np.cos(az) * np.cos(el)])
            pos = centre + d * dirv
            tgt = centre + rng.normal(0, 8, 3)
            rvec, tvec = look_at(pos, tgt, rng.normal(0, 8))
            img, th, tall = render_synthetic(cam, rvec, tvec, V, model.Ff, Va, Fa, pattern, ppm, board_mm, rng, distract,
                                             look=looks[ti % len(looks)])
            vw.write(img)
            lm = project(L3, cam.K, cam.dist, rvec, tvec) + rng.normal(0, SYN_LM_NOISE_PX, (21, 2))
            side[str(i)] = lm.tolist()
            cams.append({"frame": i, "rvec": rvec.ravel().tolist(), "tvec": tvec.ravel().tolist(),
                         "distance_mm": float(d)})
            np.savez_compressed(os.path.join(SYN, f"true_mask_t{ti}_f{i}.npz"), hand=th, all=tall)
            if i == n // 2:
                cv2.imwrite(os.path.join(SYN, f"frame_t{ti}_f{i}.jpg"), img)
        vw.release()
        sidecar[os.path.basename(vid)] = side
        truth["takes"].append({"angles": ang, "x": x.tolist(), "root": root.tolist(), "video": vid,
                               "cameras": cams, "landmarks_board_mm": L3.tolist(),
                               "measured_angles": model.angles(x)})
        print(f"[synth] take {ti}: {n} frames rendered ({time.time() - t0:.0f} s)")
    np.savez_compressed(truth["true_mesh_npz"], **true_meshes)
    json.dump(sidecar, open(os.path.join(SYN, "landmarks2d.json"), "w"))
    json.dump(truth, open(os.path.join(SYN, "truth.json"), "w"), indent=1)
    # hand_relaxed.json the way relaxed_pose.py makes it: board pose per frame + 2D joints, triangulated
    board = make_board(spec)
    scans = [scan_video(t["video"], board, 10.0, sidecar, None, SYN) for t in truth["takes"]]
    cams_ = calibrate_and_pose(scans)
    takes = []
    for ti, s in enumerate(scans):
        cam_e = s["cam"]
        Ps, xs, poses = [], [], []
        for r in s["recs"]:
            if r["pose"] is None or r["px"] is None:
                continue
            Ps.append(projection(cam_e, r["pose"]))
            xs.append(normalize(r["px"], cam_e))
            poses.append(r["pose"])
        f = float(cam_e.K[0, 0])
        Wt = np.array([robust_point(Ps, [x_[j] for x_ in xs], f)[0] for j in range(21)])
        Lt = np.array(truth["takes"][ti]["landmarks_board_mm"])
        e = np.linalg.norm(Wt - Lt, axis=1)
        print(f"[synth] take {ti}: triangulated from {len(Ps)} views, landmark error median {np.median(e):.2f} "
              f"max {e.max():.2f} mm")
        lmd = {n: [round(float(v), 3) for v in Wt[i]] for i, n in enumerate(NAMES)}
        takes.append({"input": s["path"], "frames_used": len(Ps), "landmarks_board_mm": lmd,
                      "model_landmarks_board_mm": lmd,
                      "angles": {k: float(v) for k, v in skin.spec_angles(Wt).items()},
                      "synthetic_triangulation_error_mm": {"median": float(np.median(e)), "max": float(e.max())}})
        truth["takes"][ti]["triangulation_error_mm"] = {"median": float(np.median(e)), "max": float(e.max()),
                                                        "per_joint": dict(zip(NAMES, e.tolist()))}
        truth["takes"][ti]["board_pose_rms_px"] = float(np.median([p.reproj_px for p in poses]))
    truth["calibration"] = {str(k): {"K": v[0].K.tolist(), "dist": v[0].dist.ravel().tolist(), "rms": v[1]}
                            for k, v in cams_.items()}
    json.dump({"units": "degrees (angles), mm (positions)", "method": "synthetic (compare.py synth)",
               "takes": takes}, open(os.path.join(SYN, "hand_relaxed.json"), "w"), indent=1)
    json.dump(truth, open(os.path.join(SYN, "truth.json"), "w"), indent=1)
    print(f"[synth] render done in {time.time() - t0:.0f} s")


def synth_run_args(a, out, rig, extra=()):
    p = argparse.ArgumentParser()
    add_run_args(p)
    argv = ["--rig", rig, "--relaxed", os.path.join(SYN, "hand_relaxed.json"),
            "--videos", os.path.join(SYN, "photos", "relaxed", "*.mp4"), "--board-spec", BOARD_SPEC,
            "--out", out, "--landmarks2d", os.path.join(SYN, "landmarks2d.json"), "--mask-noise", "1.0",
            "--truth", os.path.join(SYN, "truth.json"),
            "--picture-title", "Synthetic test: folded stand-in hand vs rendered video silhouette",
            "--picture-note", "Synthetic frames (stand-in hand, true skin parameters differ from the defaults)."]
    return p.parse_args(argv + list(extra))


def synth_run(a):
    ra = synth_run_args(a, SYN_OUT, a.rig, a.extra)
    res = run(ra)
    pickle.dump({"r2": {k: v for k, v in res["r2"].items() if k not in ("per_frame_before", "per_frame_after")},
                 "level1": res["level1"]}, open(os.path.join(SYN, "run_result.pkl"), "wb"))


def scaled_blend(src, dst, s):
    import bpy
    bpy.ops.wm.open_mainfile(filepath=src)
    for name in ("hand scan", "hand rig"):
        ob = bpy.data.objects[name]
        if ob.parent is None:
            ob.matrix_world = __import__("mathutils").Matrix.Scale(s, 4) @ ob.matrix_world
    bpy.ops.wm.save_as_mainfile(filepath=dst)


def synth_eval(a):
    """Compare the synthetic run with the truth; wrong-scale test."""
    truth = json.load(open(os.path.join(SYN, "truth.json")))
    l1 = json.load(open(os.path.join(SYN_OUT, "level1_landmarks.json")))
    l2 = json.load(open(os.path.join(SYN_OUT, "level2_surface.json")))
    rr = pickle.load(open(os.path.join(SYN, "run_result.pkl"), "rb"))["r2"]
    model = HandModel3D(a.rig, None, 1.0)
    base = model.base_params
    tp = truth["true_params"]
    rep = {"level1": [], "level2": {}, "mask_extraction": {}, "scale_test": {}}
    # level 1: angles vs truth
    for ti, t in enumerate(truth["takes"]):
        tru = t["measured_angles"]
        fit = l1["takes"][ti]["fixed_scale"]["angles"]
        ref = l2["level2"]["refined_poses"][str(ti)]["angles"]
        e1 = {k: fit[k] - tru[k] for k in MEAS}
        e2 = {k: ref[k] - tru[k] for k in MEAS}
        obs = l1["takes"][ti]["observed_angles"]
        e0 = {k: obs[k] - tru[k] for k in MEAS if k in obs}
        flex = [k for k in MEAS if k.endswith(("_mcp", "_pip", "_dip", "_ip"))]
        rep["level1"].append({
            "take": ti, "triangulation_error_mm": t["triangulation_error_mm"],
            "landmark_residual_median_mm": l1["takes"][ti]["fixed_scale"]["median_mm"],
            "landmark_residual_max_mm": l1["takes"][ti]["fixed_scale"]["max_mm"],
            "angle_err_level1_deg": e1, "angle_err_after_level2_deg": e2, "angle_err_direct_from_landmarks_deg": e0,
            "flex_abs_err_level1": {"median": float(np.median(np.abs([e1[k] for k in flex]))),
                                    "max": float(np.max(np.abs([e1[k] for k in flex]))),
                                    "worst": max(flex, key=lambda k: abs(e1[k]))},
            "all_abs_err_level1": {"median": float(np.median(np.abs(list(e1.values())))),
                                   "max": float(np.max(np.abs(list(e1.values())))),
                                   "worst": max(e1, key=lambda k: abs(e1[k]))},
            "flex_abs_err_level2": {"median": float(np.median(np.abs([e2[k] for k in flex]))),
                                    "max": float(np.max(np.abs([e2[k] for k in flex]))),
                                    "worst": max(flex, key=lambda k: abs(e2[k]))},
            "all_abs_err_level2": {"median": float(np.median(np.abs(list(e2.values())))),
                                   "max": float(np.max(np.abs(list(e2.values())))),
                                   "worst": max(e2, key=lambda k: abs(e2[k]))},
        })
    # level 2: 3D surface error vs the true deformed mesh, at the TRUE pose (isolates the skin params)
    tuned = rr["params"]
    fam = truth.get("truth_family", "param")
    tm = np.load(truth["true_mesh_npz"]) if truth.get("true_mesh_npz") else None
    surf = {}
    rows = (("defaults", base), ("tuned", tuned)) + ((("truth_params", tp),) if fam == "param" else ())
    for label, prm in rows:
        errs = []
        for ti, t in enumerate(truth["takes"]):
            x, root = np.array(t["x"])[:len(MEAS)], np.array(t["root"])
            Vt = tm[f"V{ti}"] if tm is not None else apply_root(model.mesh(tp, x, full=True), root, model.w0)
            Vp = apply_root(model.mesh(prm, x, full=True), root, model.w0)
            errs.append(np.linalg.norm(Vp - Vt, axis=1))
        e = np.concatenate(errs)
        surf[label] = {"mean_mm": float(e.mean()), "p95_mm": float(np.percentile(e, 95)), "max_mm": float(e.max())}
    rep["level2"]["surface_vs_truth_at_true_pose"] = surf
    rep["truth_family"], rep["true_seed"], rep["looks"] = fam, truth.get("true_seed"), truth.get("looks")
    # params
    ptab = {}
    conf = l2["level2"].get("parameter_confidence", {})
    for g in ("mcp", "pip", "dip"):
        for fld in FIELD_SPEC:
            vals = lambda P: float(np.mean([P["joints"][f"{f}_{g}"][fld] for f in F4]))  # noqa: E731
            k = f"finger_{g}.{fld}"
            c = conf.get(k, {})
            ptab[k] = {"default": vals(base), "true": vals(tp) if fam == "param" else None, "tuned": vals(tuned),
                       "measurable": c.get("measurable"), "confidence_half_width": c.get("confidence_half_width")}
            if fam == "param" and c.get("measurable"):
                ptab[k]["error_default"] = ptab[k]["default"] - ptab[k]["true"]
                ptab[k]["error_tuned"] = ptab[k]["tuned"] - ptab[k]["true"]
                ptab[k]["moved_toward_truth"] = bool(abs(ptab[k]["error_tuned"]) < abs(ptab[k]["error_default"]))
    ptab["cs_factor"] = {"default": base["global"]["cs_factor"],
                         "true": tp["global"]["cs_factor"] if fam == "param" else None,
                         "tuned": tuned["global"]["cs_factor"]}
    rep["level2"]["params"] = ptab
    rep["level2"]["fit_vs_held_out"] = l2["level2"].get("fit_vs_held_out")
    rep["level2"]["warnings"] = l2["level2"].get("warnings")
    rep["level2"]["noise_floor_true_model"] = l2["level2"].get("noise_floor_true_model")
    rep["level2"]["score_before"] = l2["level2"]["before"]
    rep["level2"]["score_after"] = l2["level2"]["after"]
    rep["level2"]["pose_only_default_skin_score"] = l2["level2"]["pose_only_default_skin_score"]
    # silhouette error of the TRUE model through the same pipeline (noise floor)
    rep["level2"]["history"] = l2["level2"]["history"]
    # mask extraction accuracy: extracted vs true (hand + arm) silhouette, full resolution
    spec = BoardSpec.load(BOARD_SPEC)
    pattern, ppm, board_mm = board_pattern(spec)
    board = make_board(spec)
    side = json.load(open(os.path.join(SYN, "landmarks2d.json")))
    scans = [scan_video(t["video"], board, 10.0, side, None, SYN) for t in truth["takes"]]
    calibrate_and_pose(scans)
    ious, bd, rejected = [], [], []
    for ti, s in enumerate(scans):
        recs = {r["fi"]: r for r in s["recs"]}
        for fi, ts, img in iter_video(s["path"]):
            if fi % 5:
                continue
            r = recs[fi]
            if r["pose"] is None:
                continue
            f1 = l1["takes"][ti]["fixed_scale"]
            H_, W_ = img.shape[:2]
            pr = model_prior(model, base, np.array(f1["x"]), np.array(f1["root"]), s["cam"], r["pose"], (W_, H_), 20.0)
            m = extract_mask(img, s["cam"], r["pose"], pattern, ppm, board_mm, r["px"], prior=pr)
            tm = np.load(os.path.join(SYN, f"true_mask_t{ti}_f{fi}.npz"))["all"] & pr["distal"]
            if m is not None:
                m = m & pr["distal"]
            if m is None:
                rejected.append([ti, fi])
                continue
            ious.append(float((m & tm).sum() / max((m | tm).sum(), 1)))
            dt = cv2.distanceTransform((~boundary(tm)).astype(np.uint8), cv2.DIST_L2, 5)
            bd.append(float(dt[boundary(m)].mean()))
    rep["mask_extraction"] = {"frames": len(ious), "rejected_by_quality_gate": rejected, "iou_mean": float(np.mean(ious)), "iou_min": float(np.min(ious)),
                              "boundary_dist_px_mean": float(np.mean(bd)), "boundary_dist_px_max": float(np.max(bd))}
    print(f"[eval] mask extraction IoU mean {np.mean(ious):.4f} min {np.min(ious):.4f}, boundary {np.mean(bd):.2f} px")
    rep["calibration"] = truth.get("calibration")
    json.dump(rep, open(os.path.join(SYN_OUT, "synthetic_eval.json"), "w"), indent=1)
    print(json.dumps({"surface": surf, "level1": [r["all_abs_err_level1"] for r in rep["level1"]],
                      "level2": [r["all_abs_err_level2"] for r in rep["level1"]]}, indent=1))


def synth_scale(a):
    """Wrong scan scale (+5%): landmark level with free scale, and surface score with/without scale."""
    dst = os.path.join(SYN, "standin_rigged_x105.blend")
    if not os.path.exists(dst):
        scaled_blend(a.rig, dst, 1.05)
    out = os.path.join(SYN_OUT, "scale_test")
    extra = ["--rounds", "1", "--budget-pose", "250", "--budget-skin", "200", "--fit-scale", "auto", "--max-frames", "14"]
    ra = synth_run_args(a, out, dst, extra)
    ra.picture_title = "Synthetic test, scan 5% too large: scale fitted at the surface level"
    ra.truth = None
    res = run(ra)
    l1 = res["level1"]["summary"]
    l2 = res["level2"]["level2"]
    rep = {"scan_scale_true": 1.05, "expected_fit_scale": 1 / 1.05,
           "level1_free_scale_median": l1["scale_free_median"], "level1_flag": l1["scale_mismatch_flag"],
           "level1_rms_fixed_mm": l1["rms_fixed_median"], "level1_rms_free_mm": l1["rms_free_median"],
           "level1_note": l1["scale_note"],
           "level2_scale": l2["scale"], "level2_contour_before_mm": l2["before"]["contour_mm"],
           "level2_contour_after_mm": l2["after"]["contour_mm"]}
    json.dump(rep, open(os.path.join(SYN_OUT, "scale_test.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


def synth_picture(a, out=None, rig=None, extra=()):
    """Redraw before_after.png from a finished run (no optimisation)."""
    out = out or SYN_OUT
    ra = synth_run_args(a, out, rig or a.rig, extra)
    rr = pickle.load(open(os.path.join(SYN, "run_result.pkl"), "rb"))["r2"]
    model = HandModel3D(ra.rig, ra.params, ra.decimate)
    l1 = json.load(open(os.path.join(out, "level1_landmarks.json")))
    takes = json.load(open(ra.relaxed))["takes"]
    work = os.path.join(WORK, file_key(os.path.abspath(out))[:8])
    fbt = [t for t in prepare_frames(ra, takes, l1, model, work) if t["frames"]]
    poses0, scorers = build_scorers(ra, model, l1, fbt)
    sc_all = [(ti, sc) for ti, scs in scorers.items() for sc in scs]
    pick = [sc_all[i] for i in np.unique(np.linspace(0, len(sc_all) - 1, 3).astype(int))]
    cases = [("Before: default skin, landmark pose", model.base_params, rr["poses0"], 0.0),
             ("After: tuned skin + refined pose", rr["params"], rr["poses"], rr["log_scale"])]
    overlay_picture(model, pick, cases, os.path.join(out, "before_after.png"), ra.picture_title, ra.picture_note)


def synth_masks(a):
    """Silhouette extraction across skin tones and lighting (review C1): re-render frames of the
    synthetic takes with each SKIN_LOOKS appearance (+ a screen shadow case), true camera and
    board pose, and compare extract_mask without / with the model prior (GrabCut + agreement
    check) against the true silhouette.  'contour floor' = Scorer contour error of the TRUE
    deformed mesh against the extracted mask (what the extraction alone adds)."""
    truth = json.load(open(os.path.join(SYN, "truth.json")))
    relaxed = json.load(open(os.path.join(SYN, "hand_relaxed.json")))
    side = json.load(open(os.path.join(SYN, "landmarks2d.json")))
    model = HandModel3D(a.rig, None, 0.2)
    tp = truth["true_params"]
    cam = Camera.from_dict(truth["camera"])
    spec = BoardSpec.load(BOARD_SPEC)
    pattern, ppm, board_mm = board_pattern(spec)
    Va0, Fa = arm_tube(model)
    jw = np.ones(21)
    jw[[0, 1]] = 0.5
    cases = [(k, k, 0.0) for k in SKIN_LOOKS] + [("orange_screen_shadow_25", "orange_headlight", 0.25),
                                                 ("dark_screen_shadow_25", "dark_skin", 0.25)]
    only = [c for c in a.extra if not c.startswith("-")]
    frames = [int(v) for v in os.environ.get("MASK_FRAMES", "0,6,13,19,25").split(",")]
    W, H = cam.size
    out = {}
    t0 = time.time()
    for name, look, shadow in cases:
        if only and not any(o in name for o in only):
            continue
        rng = np.random.default_rng(3)
        res = {"plain": [], "check_only": [], "prior": []}
        for ti in (0, 1):
            t = truth["takes"][ti]
            x, root = np.array(t["x"])[:len(MEAS)], np.array(t["root"])
            Vh = apply_root(model.mesh(tp, x, full=True), root, model.w0)
            Va = apply_root(Va0, root, model.w0)
            Lobs = np.array([relaxed["takes"][ti]["model_landmarks_board_mm"][n] for n in NAMES])
            f1 = fit_landmarks(model, Lobs, False, jw)
            x1, r1 = np.array(f1["x"]), np.array(f1["root"])
            if PRIOR_NOISE:
                # a worse level-1 pose (real MediaPipe joints are not joint centres): every joint
                # angle +-PRIOR_NOISE deg, the root +-PRIOR_NOISE mm
                prng = np.random.default_rng(100 + ti)
                x1 = np.clip(x1 + prng.normal(0, PRIOR_NOISE, len(x1)), model.xlo, model.xhi)
                r1 = r1.copy()
                r1[3:6] += prng.normal(0, PRIOR_NOISE, 3)
            Lr, _, Pr = model.landmarks(x)
            prox = model.w0 + 20.0 * skin.unit(model.w0 - Pr["knuckle"])
            w3 = apply_root(model.w0[None], root, model.w0)[0]
            p3 = apply_root(prox[None], root, model.w0)[0]
            for fi in frames:
                c = t["cameras"][fi]
                rvec, tvec = np.array(c["rvec"]).reshape(3, 1), np.array(c["tvec"]).reshape(3, 1)
                img, th, tall = render_synthetic(cam, rvec, tvec, Vh, model.Ff, Va, Fa, pattern, ppm, board_mm,
                                                 rng, [], look=look, shadow=shadow)
                pose = Pose(rvec, tvec, 0.0, 0)
                lm = np.array(side[os.path.basename(t["video"])][str(fi)], float)
                pr = model_prior(model, model.base_params, x1, r1, cam, pose, (W, H), 20.0)
                for mode in ("plain", "check_only", "prior"):
                    tt = time.time()
                    m = extract_mask(img, cam, pose, pattern, ppm, board_mm, lm,
                                     prior=None if mode == "plain" else pr,
                                     gc_mode="none" if mode == "check_only" else GC_MODE)
                    dt_s = time.time() - tt
                    if m is None:
                        res[mode].append({"take": ti, "frame": fi, "rejected": True})
                        continue
                    d = pr["distal"]
                    tm, mm = tall & d, m & d
                    iou = float((mm & tm).sum() / max((mm | tm).sum(), 1))
                    dtt = cv2.distanceTransform((~boundary(tm)).astype(np.uint8), cv2.DIST_L2, 5)
                    bd = float(dtt[boundary(mm)].mean()) if boundary(mm).any() else 99.0
                    fo = FrameObs()
                    ws = 0.5
                    Hw, Wv = int(round(H * ws)), int(round(W * ws))
                    fo.K, fo.dist, fo.rvec, fo.tvec = scale_K(cam.K, ws), cam.dist, rvec, tvec
                    fo.size, fo.close_px, fo.lm2d, fo.fi = (Wv, Hw), CLOSE_PX * ws, None, fi
                    sco = {}
                    for lab, mk in (("extracted", m), ("true", tall)):
                        fo.mask = cv2.resize(mk.astype(np.float32), (Wv, Hw), interpolation=cv2.INTER_AREA) > 0.5
                        scr = Scorer(fo, apply_root(Lr, root, model.w0), w3, p3, 20.0)
                        sco[lab] = scr.score(Vh, model.Ff)["contour_mm"]
                    res[mode].append({"take": ti, "frame": fi, "iou_hand_side": iou, "boundary_px": bd,
                                      "contour_floor_mm": sco["extracted"], "contour_floor_truemask_mm": sco["true"],
                                      "seconds": dt_s})
                if fi == 13 and ti == 0:
                    vis = img.copy()
                    m = extract_mask(img, cam, pose, pattern, ppm, board_mm, lm, prior=pr, gc_mode=GC_MODE)
                    if m is not None:
                        vis[m & ~tall] = (0, 0, 255)
                        vis[tall & ~m] = (255, 0, 0)
                    os.makedirs(os.path.join(SYN_OUT, "masks"), exist_ok=True)
                    cv2.imwrite(os.path.join(SYN_OUT, "masks", f"mask_{name}.jpg"), vis)
        summ = {}
        for mode, rr in res.items():
            ok = [r for r in rr if not r.get("rejected")]
            summ[mode] = {"frames": len(rr), "rejected": len(rr) - len(ok)}
            if ok:
                summ[mode].update({k: float(np.mean([r[k] for r in ok])) for k in
                                   ("iou_hand_side", "boundary_px", "contour_floor_mm", "contour_floor_truemask_mm",
                                    "seconds")})
                summ[mode]["iou_hand_side_min"] = float(np.min([r["iou_hand_side"] for r in ok]))
                summ[mode]["contour_floor_mm_max"] = float(np.max([r["contour_floor_mm"] for r in ok]))
        out[name] = {"summary": summ, "frames": res}
        print(f"[masks] {name}: " + "; ".join(
            f"{m_}: rej {v['rejected']}/{v['frames']}" + (f", IoU {v['iou_hand_side']:.3f} (min {v['iou_hand_side_min']:.3f}), "
                                                       f"boundary {v['boundary_px']:.2f} px, contour floor {v['contour_floor_mm']:.3f} mm "
                                                       f"(true mask {v['contour_floor_truemask_mm']:.3f})"
                                                       if 'iou_hand_side' in v else "") for m_, v in summ.items())
            + f" ({time.time() - t0:.0f} s)", flush=True)
        os.makedirs(SYN_OUT, exist_ok=True)
        path = os.path.join(SYN_OUT, "mask_robustness.json" if GC_MODE == "labraw4" and not PRIOR_NOISE and GC_PASSES == 2
                            else f"mask_robustness_{GC_MODE}_{PRIOR_NOISE}_p{GC_PASSES}.json")
        prev = json.load(open(path)) if os.path.exists(path) else {}
        prev[name] = out[name]
        json.dump(prev, open(path, "w"), indent=1)


# synthetic scenarios: main = the full run shown in the README; the others check that the method
# generalises (3 random true skins, one out-of-family truth) with a cheaper optimisation
_QUICK = ["--rounds", "1", "--fit-frames", "10", "--max-frames", "20", "--budget-pose", "150", "--budget-skin", "150"]
SCENARIOS = {
    "main": dict(family="param", true_seed=7, looks="medium_top_light,light_side_light"),
    "seed21": dict(family="param", true_seed=21, looks="dark_skin,orange_side_light", run_args=_QUICK),
    "seed33": dict(family="param", true_seed=33, looks="light_skin,dark_side_light", run_args=_QUICK),
    "seed7q": dict(family="param", true_seed=7, looks="orange_headlight,medium_top_light", run_args=_QUICK),
    "autoweights": dict(family="autoweights", true_seed=7, looks="medium_top_light,light_side_light", run_args=_QUICK),
}


def synth_multi(a):
    """Run (render -> run -> eval) every non-main scenario in its own process (skips finished
    ones) and summarise the spread of parameter recovery and surface error."""
    rows = {}
    for name in [n for n in SCENARIOS if n != "main"]:
        out = os.path.join(OUT, "synthetic", name)
        ev = os.path.join(out, "synthetic_eval.json")
        if not os.path.exists(ev):
            for st in ("render", "run", "eval"):
                if st == "render" and os.path.exists(os.path.join(WORK, "synth", name, "hand_relaxed.json")):
                    continue
                if st == "run" and os.path.exists(os.path.join(out, "level2_surface.json")) and \
                        os.path.exists(os.path.join(WORK, "synth", name, "run_result.pkl")):
                    continue
                print(f"[multi] {name}: {st}", flush=True)
                import subprocess
                subprocess.run([sys.executable, os.path.abspath(__file__), "synth", "--scenario", name,
                                "--rig", a.rig, st], check=True)
        rows[name] = json.load(open(ev))
    summ = {}
    for name, r in rows.items():
        l2 = r["level2"]
        sv = l2["surface_vs_truth_at_true_pose"]
        tuned_dims = {k: v for k, v in l2["params"].items() if v.get("measurable")}
        summ[name] = {"truth_family": r.get("truth_family"), "looks": r.get("looks"),
                      "surface_mean_mm": {k: v["mean_mm"] for k, v in sv.items()},
                      "surface_p95_mm": {k: v["p95_mm"] for k, v in sv.items()},
                      "tuned_params": {k: {kk: v.get(kk) for kk in ("default", "true", "tuned", "confidence_half_width",
                                                                     "moved_toward_truth")} for k, v in tuned_dims.items()},
                      "held_out": l2.get("fit_vs_held_out"), "warnings": l2.get("warnings"),
                      "mask_extraction": r.get("mask_extraction")}
    improve = [v["surface_mean_mm"]["defaults"] - v["surface_mean_mm"]["tuned"] for v in summ.values()]
    toward = [p["moved_toward_truth"] for v in summ.values() for p in v["tuned_params"].values()
              if p.get("moved_toward_truth") is not None]
    summary = {"scenarios": summ,
               "surface_mean_improvement_mm": {"per_scenario": dict(zip(summ, improve)),
                                               "min": float(min(improve)), "max": float(max(improve))},
               "tuned_params_moved_toward_truth": f"{sum(toward)}/{len(toward)}"}
    json.dump(summary, open(os.path.join(OUT, "synthetic", "multi_summary.json"), "w"), indent=1, default=float)
    print(json.dumps({k: {"surface_mean_mm": v["surface_mean_mm"], "tuned": list(v["tuned_params"])}
                      for k, v in summ.items()}, indent=1))
    print("moved toward truth:", summary["tuned_params_moved_toward_truth"])


def synth_real_l1(a):
    """Level 1 on Olex's real triangulated relaxed landmarks (extract) with the given rig."""
    ex = json.load(open(REAL_EXTRACT))
    takes = [{"input": f"take{i + 1}", "model_landmarks_board_mm": t} for i, t in enumerate(ex["takes_model_landmarks_board_mm"])]
    model = HandModel3D(a.rig, None, 1.0)
    jw = np.ones(21)
    jw[[0, 1]] = 0.5
    out = os.path.join(OUT, "real_extract_level1")
    r = level1(model, takes, "model_landmarks_board_mm", out, jw)
    ref = ex["joints"]
    cmp_ = {}
    for k in MEAS:
        if k in ref:
            cmp_[k] = {"capture_handmodel": ref[k]["deg"], "scan_rig_fit": r["summary"]["angles_median"][k]}
    json.dump({"note": "level 1 of compare.py on the per-take model_landmarks_board_mm of data/hand_relaxed_extract.json "
                       "with the stand-in rig (real scan not available yet)", "angles": cmp_},
              open(os.path.join(out, "angles_vs_capture.json"), "w"), indent=1)
    print(json.dumps(cmp_, indent=0))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "synth":
        p = argparse.ArgumentParser(description="synthetic test")
        p.add_argument("synth")
        p.add_argument("stage", choices=["render", "run", "eval", "scale", "picture", "real_l1", "masks", "multi", "all"])
        p.add_argument("--rig", default=STANDIN_BLEND)
        p.add_argument("--scenario", default="main",
                       help="name of this synthetic set (work + output subfolder); see SCENARIOS")
        p.add_argument("--family", choices=["param", "autoweights"], default=None,
                       help="true hand: this module's deformer with other parameters, or the bone-heat rig with LBS")
        p.add_argument("--true-seed", type=int, default=None, help="true skin parameter set (7 = the fixed one)")
        p.add_argument("--looks", default=None, help="comma list of SKIN_LOOKS, one per take")
        p.add_argument("extra", nargs=argparse.REMAINDER)
        a = p.parse_args()
        sc = SCENARIOS.get(a.scenario, {})
        a.family = a.family or sc.get("family", "param")
        a.true_seed = a.true_seed if a.true_seed is not None else sc.get("true_seed", 7)
        a.looks = a.looks or sc.get("looks")
        a.extra = list(sc.get("run_args", [])) + list(a.extra) if a.stage in ("run", "all", "picture") else a.extra
        global SYN, SYN_OUT
        SYN = os.path.join(WORK, "synth", a.scenario)
        SYN_OUT = os.path.join(OUT, "synthetic", a.scenario)
        os.makedirs(SYN_OUT, exist_ok=True)
        if a.stage == "multi":
            return synth_multi(a)
        stages = ["render", "run", "eval", "scale", "real_l1"] if a.stage == "all" else [a.stage]
        for s in stages:
            {"render": synth_render, "run": synth_run, "eval": synth_eval, "scale": synth_scale,
             "picture": synth_picture, "masks": synth_masks,
             "real_l1": synth_real_l1}[s](a)
        return
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(p)
    run(p.parse_args())


if __name__ == "__main__":
    main()
