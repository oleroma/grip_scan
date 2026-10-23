#!/usr/bin/env python
"""Synthetic board videos of a known hand mesh, and evaluation of recon_board.py on them.

    python recon_synth.py make --out /tmp/claude-0/hs/work/recon/synth          # videos + truth
    python recon_board.py --videos "<out>/scans/*.mp4" --board-spec <out>/data/board_screen.json \
        --camera <out>/data/camera.json --out <out>/recon/hand_scan.ply --fps 5
    python recon_synth.py eval --synth <out> --recon <out>/recon/hand_scan.ply --picture x.png

The scene: the screen (board_screen.json layout, white/black with a slightly blue
white, vignetting) and the hand mesh (standin_open.ply if present, else a capsule
hand) with skin shading and procedural texture dots, plus a forearm cylinder going to
the books. Phone: camera.json lens mapped to 3840x2160 video, focal breathing per frame,
sweeps like the relaxed videos; frames get defocus, some motion blur, noise, mp4
compression; the hand gets 1-2 px pose jitter against the board per frame and a slow
drift within the take.

Take 1: palm facing the camera, thumb up, fingers sideways, ~3 cm from the screen.
Take 2: back of the hand to the camera, re-placed a few mm / deg differently.
Camera 310-430 mm from the hand, -52..+26 deg left-right, -30..+48 deg up-down (the
ranges of Olex's first real video); ~40% of the frames have screen glare/overexposure.
"""
import argparse
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recon_board as rb  # noqa: E402

PG = rb.GRIPSCAN
STANDIN_CANDIDATES = ["/mnt/project-files/hand_scan_rig/standin/standin_open.ply",
                      "/tmp/claude-0/hs/work/recon/truth/standin_open.ply"]
W, H = 3840, 2160
RANGE = (435.0, 75.0)  # camera distance from the hand: mean, amplitude (Olex's video: 360-510 mm)


# ------------------------------------------------------------------ truth

def load_truth(path=None):
    import trimesh
    for p in ([path] if path else []) + STANDIN_CANDIDATES:
        if os.path.exists(p):
            m = trimesh.load(p, process=True)
            j = os.path.join(os.path.dirname(p), "joints.json")
            joints = json.load(open(j))["landmarks"] if os.path.exists(j) else None
            return m, joints, p
    return capsule_hand()


def capsule_hand():
    """Fallback: capsule fingers + rounded palm with a palm hollow, SDF -> mesh (hand frame)."""
    import trimesh
    from skimage import measure
    J = {"wrist": [0, 0, 0], "thumb_cmc": [30, 16, -5], "thumb_mcp": [55, 38, 4], "thumb_ip": [75, 57, 14],
         "thumb_tip": [88, 70, 22]}
    xs = {"index": 24, "middle": 0, "ring": -19, "pinky": -35}
    ys = {"index": 87, "middle": 88, "ring": 79, "pinky": 64}
    L = {"index": (42, 24, 20), "middle": (46, 26, 20), "ring": (43, 26, 20), "pinky": (33, 20, 18)}
    for f in xs:
        p = np.array([xs[f], ys[f], 0.0])
        J[f + "_mcp"] = p.tolist()
        for n, l in zip(["pip", "dip", "tip"], L[f]):
            p = p + np.array([0.08 * np.sign(xs[f]) * l, l, 0.05 * l])
            J[f + "_" + n] = p.tolist()
    J = {k: np.array(v, float) for k, v in J.items()}
    g = np.mgrid[-60:105:1.0, -45:185:1.0, -30:40:1.0]
    P = g.reshape(3, -1).T

    def seg(P, a, b, r):
        ab = b - a
        t = np.clip((P - a) @ ab / (ab @ ab), 0, 1)
        return np.linalg.norm(P - a - t[:, None] * ab, axis=1) - r

    def smin(a, b, k=4.0):
        h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
        return b * (1 - h) + a * h - k * h * (1 - h)
    # palm: rounded box
    q = np.abs(P - [0, 45, 2]) - [38, 40, 9]
    palm = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0) - 6
    hollow = np.linalg.norm((P - [0, 48, 26]) / [1, 1.2, 1], axis=1) - 20  # dent on the palm side
    palm = np.maximum(palm, -hollow)
    d = palm
    d = smin(d, seg(P, J["wrist"] + [0, -50, 0], J["wrist"] + [0, 5, 2], 24) , 6)
    for f in xs:
        r0 = {"index": 9.5, "middle": 9.5, "ring": 9, "pinky": 8}[f]
        names = ["mcp", "pip", "dip", "tip"]
        for i in range(3):
            d = smin(d, seg(P, J[f + "_" + names[i]], J[f + "_" + names[i + 1]], r0 - 0.8 * i), 2.5)
    for a_, b_, r in [("thumb_cmc", "thumb_mcp", 12), ("thumb_mcp", "thumb_ip", 10), ("thumb_ip", "thumb_tip", 9)]:
        d = smin(d, seg(P, J[a_], J[b_], r), 4)
    D = d.reshape(g.shape[1:])
    v, f, _, _ = measure.marching_cubes(D, 0, spacing=(1, 1, 1))
    v += [-60, -45, -30]
    m = trimesh.Trimesh(v, f[:, ::-1])
    if m.volume < 0:
        m.invert()
    return m, {k: v.tolist() for k, v in J.items()}, "capsule hand (fallback)"


def regions(mesh, joints):
    """Region per vertex: fingers, thumb, palm_hollow, palm_other, back, wrist (hand frame)."""
    V = np.asarray(mesh.vertices)
    N = np.asarray(mesh.vertex_normals)
    J = {k: np.array(v) for k, v in joints.items()}
    segs, kinds = [], []
    for f in ["index", "middle", "ring", "pinky"]:
        ch = [f + "_mcp", f + "_pip", f + "_dip", f + "_tip"]
        for i in range(3):
            segs.append((J[ch[i]], J[ch[i + 1]]))
            kinds.append("fingers")
        segs.append((J["wrist"], J[f + "_mcp"]))
        kinds.append("palm")
    for a_, b_, k in [("thumb_mcp", "thumb_ip", "thumb"), ("thumb_ip", "thumb_tip", "thumb"),
                      ("thumb_cmc", "thumb_mcp", "palm"), ("wrist", "thumb_cmc", "palm")]:
        segs.append((J[a_], J[b_]))
        kinds.append(k)
    D = []
    for a_, b_ in segs:
        ab = b_ - a_
        t = np.clip((V - a_) @ ab / (ab @ ab), 0, 1)
        D.append(np.linalg.norm(V - a_ - t[:, None] * ab, axis=1))
    near = np.argmin(np.array(D), 0)
    kind = np.array(kinds)[near].astype(object)
    out = np.empty(len(V), object)
    out[:] = kind
    palm = kind == "palm"
    xi, xp = J["index_mcp"][0], J["pinky_mcp"][0]
    ymcp = min(J["index_mcp"][1], J["middle_mcp"][1], J["ring_mcp"][1])
    central = (V[:, 0] < xi - 6) & (V[:, 0] > xp + 8) & (V[:, 1] > 25) & (V[:, 1] < ymcp - 10)
    out[palm & (N[:, 2] > 0.3) & central] = "palm_hollow"
    out[palm & (N[:, 2] > 0.3) & ~central] = "palm_other"
    out[palm & (N[:, 2] < -0.3)] = "back"
    out[palm & (np.abs(N[:, 2]) <= 0.3)] = "palm_edge"
    out[V[:, 1] < 12] = "wrist"
    return out


# ------------------------------------------------------------------ scene

def video_camera(camera_json):
    c = rb.camera_fallback(camera_json, (W, H))
    return c.K, np.ravel(c.dist)


def rot(axis, deg):
    return rb.rodrigues(np.asarray(axis, float) / np.linalg.norm(axis) * np.radians(deg))


def hand_pose(take):
    """Hand frame (wrist origin, +Y fingers, +Z palm normal) -> board frame.

    Like Olex's real scan: the hand ~2-4 cm in front of the screen near board (197,158),
    fingers sideways. Take 1: palm to the camera, thumb up (a right hand then points its
    fingers to the camera's left). Take 2: back of the hand to the camera (forearm turned
    180 deg: thumb down), placed a few mm / deg differently."""
    if take == 1:
        R = np.column_stack([[0, -1, 0], [-1, 0, 0], [0, 0, -1.0]])
        R = rot([0, 1, 0], 8) @ rot([1, 0, 0], -5) @ R
        c_board = np.array([197.0, 158, -38])
    else:
        R = np.column_stack([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])
        R = rot([0, 1, 0], -6) @ rot([0, 0, 1], 7) @ R
        c_board = np.array([190.0, 150, -40])
    c_hand = np.array([0, 70, 5.0])
    t = c_board - R @ c_hand
    return R, t


def look_at(C, target, roll_deg):
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross([0, 1.0, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    R = rot([0, 0, 1], roll_deg) @ R
    return R, -R @ C


def camera_path(take, n, rng):
    Rh, th = hand_pose(take)
    centre = Rh @ np.array([0, 80, 0.0]) + th
    s = np.linspace(0, 1, n)
    # Olex's real sweep: -52..+26 deg left-right, -30..+48 deg up-down, 300-500 mm
    ph = 0.3 if take == 1 else 2.0
    az = -13 + 39 * np.sin(2 * np.pi * s + ph)
    el = 9 + 39 * np.sin(2 * np.pi * 1.5 * s + 1.0 + ph)
    r = RANGE[0] + RANGE[1] * np.sin(2 * np.pi * 2 * s + ph)
    out = []
    for i in range(n):
        a_, e_ = np.radians(az[i]), np.radians(el[i])
        d = np.array([np.sin(a_) * np.cos(e_), -np.sin(e_), -np.cos(a_) * np.cos(e_)])
        C = centre + r[i] * d
        tgt = centre + rng.normal(0, 12, 3)
        R, t = look_at(C, tgt, rng.normal(0, 6))
        out.append((R, t))
    return out


def forearm_mesh(mesh, take):
    """Elliptic cylinder from the wrist stub towards the books (render only, hand frame)."""
    import trimesh
    V = np.asarray(mesh.vertices)
    stub = V[(V[:, 1] < -25) & (V[:, 1] > -38)]
    c = stub.mean(0)
    ax_x = (np.percentile(stub[:, 0], 98) - np.percentile(stub[:, 0], 2)) / 2 - 1
    ax_z = (np.percentile(stub[:, 2], 98) - np.percentile(stub[:, 2], 2)) / 2 - 1
    cyl = trimesh.creation.cylinder(radius=1.0, height=320, sections=64)
    cyl = cyl.subdivide_to_size(6.0)
    P = np.asarray(cyl.vertices).copy()
    # cylinder axis z -> hand -y, scaled to the stub's section, slightly flaring
    zz = P[:, 2] + 160  # 0..320
    flare = 1 + 0.0015 * zz
    Q = np.column_stack([P[:, 0] * ax_x * flare, -zz, P[:, 1] * ax_z * flare])
    bend = rot([1, 0, 0], -20) @ rot([0, 0, 1], -25) if take == 1 else rot([1, 0, 0], 20) @ rot([0, 0, 1], 25)
    Q = (Q @ bend.T) + [c[0], c[1] + 8, c[2]]
    return trimesh.Trimesh(Q, cyl.faces, process=False)


def screen_texture(spec):
    scr = rb.ScreenModel(spec, 8.0)
    return scr


def hash3(ix, iy, iz, k):
    h = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64) * 19349663) ^ (iz.astype(np.int64) * 83492791) ^ (k * 2654435761)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float64) / float(0xFFFFFF)


def albedo(Ph):
    """Skin colour (BGR, 0..1) with low-frequency variation and texture dots, hand frame."""
    base = np.array([0.50, 0.60, 0.82])
    low = 1 + 0.06 * np.sin(Ph[:, 0] * 0.11 + 1.3 * np.sin(Ph[:, 1] * 0.07)) + 0.05 * np.sin(Ph[:, 2] * 0.13 + Ph[:, 1] * 0.05)
    col = base[None] * low[:, None]
    fac = np.ones(len(Ph))
    for cell, rmin, rmax, dark, prob, k in [(1.6, 0.3, 0.55, 0.55, 0.7, 1), (0.8, 0.15, 0.3, 0.8, 0.6, 2)]:
        q = Ph / cell
        i = np.floor(q)
        off = q - i
        cx = 0.3 + 0.4 * hash3(i[:, 0], i[:, 1], i[:, 2], k)
        cy = 0.3 + 0.4 * hash3(i[:, 0], i[:, 1], i[:, 2], k + 10)
        cz = 0.3 + 0.4 * hash3(i[:, 0], i[:, 1], i[:, 2], k + 20)
        rr = (rmin + (rmax - rmin) * hash3(i[:, 0], i[:, 1], i[:, 2], k + 30)) / cell
        on = hash3(i[:, 0], i[:, 1], i[:, 2], k + 40) < prob
        dk = dark + (1 - dark) * 0.5 * hash3(i[:, 0], i[:, 1], i[:, 2], k + 50)
        d = np.sqrt((off[:, 0] - cx) ** 2 + (off[:, 1] - cy) ** 2 + (off[:, 2] - cz) ** 2)
        w = np.clip((rr - d) / (0.3 * rr), 0, 1) * on
        fac *= 1 - (1 - dk) * w
    col *= fac[:, None]
    col[:, 2] *= 1 + 0.08 * (1 - fac)  # dots slightly redder (freckle-like)
    return col


def render_frame(job):
    """One 4K frame. job: dict with all inputs (picklable)."""
    import trimesh  # noqa: F401
    rng = np.random.default_rng(job["seed"])
    K, dist = job["K"], job["dist"]
    R, t = job["R"], job["t"]
    scr = job["screen"]
    # background: screen through the lens
    rays = rb.pixel_rays(K, dist, W, H)
    bx, by, depth = rb.board_hit(rays, R, t)
    foot = np.nanmedian(depth) / K[0, 0] * scr.ppm
    tex = scr.tex_blurred(0.4 * foot)
    sx, sy = scr.to_tex(bx, by)
    E = rb.sample_image(tex, sx, sy, border=np.nan)
    r_ = scr.screen_rect
    inscreen = (bx >= r_[0]) & (bx <= r_[2]) & (by >= r_[1]) & (by <= r_[3]) & np.isfinite(E)
    white = np.array([0.93, 0.90, 0.86])   # BGR, bluish white
    black = np.array([0.075, 0.07, 0.07])
    Eg = np.clip(np.nan_to_num(E), 0, 1) ** 1.1
    bg = black[None, None] + (white - black)[None, None] * Eg[..., None]
    bezel = (bx >= r_[0] - 12) & (bx <= r_[2] + 12) & (by >= r_[1] - 12) & (by <= r_[3] + 12) & ~inscreen
    wall = 0.35 + 0.08 * np.sin(np.nan_to_num(bx) * 0.02) * np.cos(np.nan_to_num(by) * 0.015)
    wallc = np.stack([wall * 0.8, wall * 0.9, wall], -1)
    if job.get("glare", 0) > 0:  # reflection of a lamp/window on the glass + overexposure
        gx, gy, gs = job["glare_xy"][0] * W, job["glare_xy"][1] * H, job["glare_sigma"]
        yy_, xx_ = np.mgrid[0:H, 0:W].astype(np.float32)
        G = job["glare"] * np.exp(-((xx_ - gx) ** 2 + (yy_ - gy) ** 2) / (2 * gs * gs))
        bg = bg * job["glare_gain"] + G[..., None] * np.array([1.0, 0.98, 0.95])
        bg = np.minimum(bg, 1.0)
    bg = np.where(inscreen[..., None], bg, np.where(bezel[..., None], 0.06, wallc))
    # hand
    Vb, Nb, Ph, faces = job["V"], job["N"], job["Ph"], job["F"]
    u, v, z = rb.project(Vb, R, t, K, dist)
    fid, b0, b1, _ = rb.rasterize(u, v, z, faces, W, H)
    cov = (fid >= 0).astype(np.float32)
    ys, xs = np.nonzero(fid >= 0)
    f = faces[fid[ys, xs]]
    w0, w1 = b0[ys, xs][:, None], b1[ys, xs][:, None]
    w2 = 1 - w0 - w1
    P = w0 * Vb[f[:, 0]] + w1 * Vb[f[:, 1]] + w2 * Vb[f[:, 2]]
    n = w0 * Nb[f[:, 0]] + w1 * Nb[f[:, 1]] + w2 * Nb[f[:, 2]]
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    ph = w0 * Ph[f[:, 0]] + w1 * Ph[f[:, 1]] + w2 * Ph[f[:, 2]]
    col = albedo(ph)
    Ld = np.array([-0.3, -0.75, -0.6])
    Ld /= np.linalg.norm(Ld)
    C = -R.T @ t
    vd = C - P
    vd /= np.linalg.norm(vd, axis=1, keepdims=True)
    diff = np.clip(n @ Ld, 0, 1)
    back = np.clip(n @ np.array([0, 0, 1.0]), 0, 1)
    hv = (Ld[None] + vd)
    hv /= np.linalg.norm(hv, axis=1, keepdims=True)
    spec = 0.05 * np.clip(np.sum(n * hv, 1), 0, 1) ** 30
    shade = 0.32 + 0.6 * diff + 0.2 * back
    rgb = col * shade[:, None] * 0.95 + spec[:, None]
    hand = np.zeros((H, W, 3), np.float32)
    hand[ys, xs] = rgb
    # optics: board defocus, hand slight defocus, vignetting, motion blur, noise
    sb, sh = job["blur_board"], job["blur_hand"]
    bgb = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), sb)
    al = cv2.GaussianBlur(cov, (0, 0), sh)
    hc = cv2.GaussianBlur(hand * cov[..., None], (0, 0), sh) / np.maximum(al, 1e-4)[..., None]
    img = bgb * (1 - al[..., None]) + hc * al[..., None]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (W / 2) ** 2
    img *= (1 - 0.12 * r2)[..., None]
    if job["motion"] > 0:
        L = int(job["motion"])
        k = np.zeros((L, L), np.float32)
        k[L // 2, :] = 1
        M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), job["motion_ang"], 1)
        k = cv2.warpAffine(k, M, (L, L))
        img = cv2.filter2D(img, -1, k / k.sum())
    img = img + rng.normal(0, 0.008, img.shape).astype(np.float32) + \
        rng.normal(0, 0.012, img.shape).astype(np.float32) * np.sqrt(np.clip(img, 0, 1))
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def make(a):
    import trimesh
    from multiprocessing import Pool
    out = a.out
    for d in ["scans", "data", "truth", "frames"]:
        os.makedirs(os.path.join(out, d), exist_ok=True)
    shutil.copy(os.path.join(PG, "data", "board_screen.json"), os.path.join(out, "data", "board_screen.json"))
    shutil.copy(os.path.join(PG, "data", "camera.json"), os.path.join(out, "data", "camera.json"))
    spec = rb.BoardSpec.load(os.path.join(out, "data", "board_screen.json"))
    scr = screen_texture(spec)
    mesh, joints, src = load_truth(a.truth_mesh)
    print("truth mesh:", src, len(mesh.vertices), "vertices")
    rmesh = mesh.subdivide_to_size(1.0) if mesh.edges_unique_length.mean() > 1.2 else mesh
    rmesh = trimesh.Trimesh(rmesh.vertices, rmesh.faces, process=True)
    K0, dist = video_camera(os.path.join(out, "data", "camera.json"))
    truth = {"source": src, "takes": [], "K_video": K0.tolist(), "dist": dist.tolist(), "size": [W, H]}
    rng = np.random.default_rng(a.seed)
    takes = [int(x) for x in a.takes.split(",")]
    for take in takes:
        Rh, th = hand_pose(take)
        fa = forearm_mesh(mesh, take)
        cams = camera_path(take, a.frames, rng)
        jobs = []
        s = np.linspace(0, 1, a.frames)
        for i, (R, t) in enumerate(cams):
            # slow drift of the hand within the take (zero mean) + per-frame jitter (px)
            dr = a.drift * np.array([np.sin(2 * np.pi * s[i] + 0.5), np.sin(2 * np.pi * s[i] * 0.7 + 2.0),
                                     np.sin(2 * np.pi * s[i] * 1.3 + 4.0)])
            drot = rot([0.3, 1, 0.2], a.drift_deg * np.sin(2 * np.pi * s[i] + 1.0))
            fsc = 1 + a.breathing * np.sin(2 * np.pi * s[i] * 1.3 + rng.uniform(0, 6)) + rng.normal(0, a.breathing / 3)
            K = K0.copy()
            K[0, 0] *= fsc
            K[1, 1] *= fsc
            jit = rng.normal(0, a.jitter_px / K[0, 0], 3)  # small rotation about the camera centre
            Rj = rb.rodrigues(jit) @ R
            tj = rb.rodrigues(jit) @ t
            # hand vertices: hand frame -> board (with drift) -> camera with jittered pose
            ctr = Rh @ np.array([0, 70, 0.0]) + th
            Rh_i = drot @ Rh
            th_i = drot @ (th - ctr) + ctr + dr
            Vh = np.concatenate([np.asarray(rmesh.vertices), np.asarray(fa.vertices)])
            Nh = np.concatenate([np.asarray(rmesh.vertex_normals), np.asarray(fa.vertex_normals)])
            F = np.concatenate([np.asarray(rmesh.faces), np.asarray(fa.faces) + len(rmesh.vertices)])
            Vb = Vh @ Rh_i.T + th_i
            Nb = Nh @ Rh_i.T
            # express the jittered hand pose in the board camera: X_cam = Rj Xb + tj
            # (equivalent hand-in-board for rendering with the true board camera)
            Vb_eff = (Vb @ Rj.T + tj - t) @ R  # board coords that the true camera sees there
            Nb_eff = (Nb @ Rj.T) @ R
            motion = 0
            if rng.uniform() < a.motion_frac:
                motion = int(rng.uniform(14, 30))
            jobs.append({"seed": int(rng.integers(1 << 30)), "K": K, "dist": dist, "R": R, "t": t,
                         "screen": scr, "V": Vb_eff, "N": Nb_eff, "Ph": Vh, "F": F,
                         "blur_board": rng.uniform(1.0, 2.2), "blur_hand": rng.uniform(0.6, 1.0),
                         "motion": motion, "motion_ang": rng.uniform(0, 180)})
            if rng.uniform() < a.glare_frac:
                jobs[-1].update(glare=rng.uniform(0.5, a.glare_max), glare_xy=rng.uniform(0.15, 0.85, 2),
                                glare_sigma=rng.uniform(300, 900), glare_gain=rng.uniform(1.0, a.glare_gain_max))
            truth_frame = {"R": R.tolist(), "t": t.tolist(), "f_scale": float(fsc),
                           "glare": float(jobs[-1].get("glare", 0)),
                           "jitter_rad": jit.tolist(), "drift_mm": dr.tolist(), "motion_blur_px": motion}
            jobs[-1]["truth"] = truth_frame
        t0 = time.time()
        vpath = os.path.join(out, "scans", f"take{take}.mp4")
        vw = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
        with Pool(a.workers) as pool:
            for i, img in enumerate(pool.imap(render_frame, jobs, chunksize=1)):
                vw.write(img)
                if i % 10 == 0:
                    cv2.imwrite(os.path.join(out, "frames", f"take{take}_{i:03d}.jpg"),
                                cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA))
                    print(f"take {take}: frame {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
        vw.release()
        truth["takes"].append({"take": take, "video": vpath, "hand_R": Rh.tolist(), "hand_t": th.tolist(),
                               "frames": [j["truth"] for j in jobs]})
        print(f"take {take}: {len(jobs)} frames -> {vpath} ({time.time() - t0:.0f}s)")
    # truth in take-1 board frame
    Rh, th = hand_pose(takes[0])
    tm = trimesh.Trimesh(np.asarray(mesh.vertices) @ Rh.T + th, mesh.faces, process=False)
    tm.export(os.path.join(out, "truth", "true_take1.ply"))
    reg = regions(mesh, joints)
    np.save(os.path.join(out, "truth", "regions.npy"), reg.astype(str))
    truth["landmarks_take1"] = {k: (Rh @ np.array(v) + th).tolist() for k, v in joints.items()}
    if len(takes) > 1:
        R2, t2 = hand_pose(takes[1])
        # T maps take-1 board coords -> take-2 board coords
        RT = R2 @ Rh.T
        truth["T_take1_to_take2"] = {"R": RT.tolist(), "t": (t2 - RT @ th).tolist()}
    with open(os.path.join(out, "truth", "truth.json"), "w") as f:
        json.dump(truth, f, indent=1)
    print("truth ->", os.path.join(out, "truth"))


# ------------------------------------------------------------------ evaluation

def surface_distance(P, mesh, spacing=0.15):
    """Signed distance (mm) from points to a mesh surface (+ outside), dense-sample approx."""
    import trimesh
    from scipy.spatial import cKDTree
    n = int(mesh.area / spacing ** 2)
    S, fi = trimesh.sample.sample_surface_even(mesh, n, radius=spacing * 0.7, seed=0)
    Nf = mesh.face_normals[fi]
    d, i = cKDTree(S).query(P)
    sgn = np.sign(np.sum((P - S[i]) * Nf[i], 1))
    return d * sgn


def evaluate(a):
    import trimesh
    syn = a.synth
    truth = json.load(open(os.path.join(syn, "truth", "truth.json")))
    tm = trimesh.load(os.path.join(syn, "truth", "true_take1.ply"), process=False)
    reg = np.load(os.path.join(syn, "truth", "regions.npy"), allow_pickle=True)
    res = {}
    meshes = {}
    for label, path in [("hull", os.path.splitext(a.recon)[0] + "_hull.ply"), ("refined", a.recon)]:
        if not os.path.exists(path):
            continue
        m = trimesh.load(path, process=False)
        meshes[label] = m
        d = surface_distance(np.asarray(tm.vertices), m)  # + : truth outside recon (recon too small)
        d2 = surface_distance(np.asarray(m.vertices), tm)  # + : recon outside truth (too big)
        out = {}
        for r in ["fingers", "thumb", "palm_hollow", "palm_other", "palm_edge", "back"]:
            sel = reg == r
            if sel.sum() == 0:
                continue
            ad = np.abs(d[sel])
            out[r] = {"median": round(float(np.median(ad)), 3), "p95": round(float(np.percentile(ad, 95)), 3),
                      "signed_median": round(float(np.median(-d[sel])), 3), "n": int(sel.sum())}
        allsel = reg != "wrist"
        out["all"] = {"median": round(float(np.median(np.abs(d[allsel]))), 3),
                      "p95": round(float(np.percentile(np.abs(d[allsel]), 95)), 3)}
        # recon -> truth on the hand side of the truth wrist
        lmw = np.array(truth["landmarks_take1"]["wrist"])
        lmm = np.array(truth["landmarks_take1"]["middle_mcp"])
        nrm = (lmm - lmw) / np.linalg.norm(lmm - lmw)
        selr = (np.asarray(m.vertices) - lmw) @ nrm > 12
        out["recon_to_true"] = {"median": round(float(np.median(np.abs(d2[selr]))), 3),
                                "p95": round(float(np.percentile(np.abs(d2[selr]), 95)), 3)}
        res[label] = out
        res[label + "_true_dist"] = d
    # landmarks + registration vs truth
    lmj = os.path.splitext(a.recon)[0] + "_landmarks.json"
    rep = os.path.splitext(a.recon)[0] + "_report.json"
    extra = {}
    if os.path.exists(lmj):
        L = json.load(open(lmj))
        tl = truth["landmarks_take1"]
        if L.get("landmarks_output_frame_mm"):
            e = [np.linalg.norm(np.array(L["landmarks_output_frame_mm"][k]) - np.array(tl[k])) for k in tl]
            extra["landmarks_vs_standin_joints_mm"] = {"median": float(np.median(e)), "max": float(np.max(e)),
                                                       "note": "MediaPipe points vs the stand-in's joint centres"}
    if os.path.exists(rep) and "T_take1_to_take2" in truth:
        R = json.load(open(rep))
        if R.get("registration"):
            r0 = R["registration"][0]
            RT, tT = np.array(r0["R"]), np.array(r0["t"])
            Rt, tt = np.array(truth["T_take1_to_take2"]["R"]), np.array(truth["T_take1_to_take2"]["t"])
            P = np.asarray(tm.vertices)
            dd = np.linalg.norm(P @ RT.T + tT - (P @ Rt.T + tt), axis=1)
            extra["registration_error_mm"] = {"median": float(np.median(dd)), "max": float(dd.max())}
        extra["report"] = {k: R.get(k) for k in ["mask_iou", "views_per_surface_point", "hull_vs_refined_mm",
                                                 "photo", "runtime_s"]}
        if R.get("focal_breathing"):
            fb = R["focal_breathing"]
            tr = {(t["take"], i): f["f_scale"] for t in truth["takes"] for i, f in enumerate(t["frames"])}
            est = [(tr.get((p["take"], p["frame"])), p["f_scale"]) for p in fb["per_frame"]]
            est = np.array([e for e in est if e[0] is not None])
            if len(est):
                calib_f = np.array(list(R["calibration"].values())[0]["K"])[0, 0] / truth["K_video"][0][0]
                extra["focal"] = {"true_scale_range": [float(est[:, 0].min()), float(est[:, 0].max())],
                                  "calibrated_f_over_true_mean": float(calib_f),
                                  "corr_est_true": float(np.corrcoef(est[:, 0], est[:, 1])[0, 1]),
                                  "est_error_rms": float(np.sqrt(np.mean((est[:, 1] * calib_f - est[:, 0]) ** 2)))}
    summary = {k: v for k, v in res.items() if not k.endswith("_true_dist")}
    summary.update(extra)
    js = a.json or os.path.splitext(a.recon)[0] + "_eval.json"
    with open(js, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    if a.picture:
        picture(tm, meshes, res, reg, a.picture, truth)
    return summary


def shade_mesh(mesh, R, t, K, Wi, Hi, colors=None):
    V = np.asarray(mesh.vertices)
    N = np.asarray(mesh.vertex_normals)
    u, v, z = rb.project(V, R, t, K, np.zeros(5))
    fid, b0, b1, _ = rb.rasterize(u, v, z, np.asarray(mesh.faces), Wi, Hi)
    img = np.full((Hi, Wi, 3), 255, np.float32)
    ys, xs = np.nonzero(fid >= 0)
    f = np.asarray(mesh.faces)[fid[ys, xs]]
    w0, w1 = b0[ys, xs][:, None], b1[ys, xs][:, None]
    w2 = 1 - w0 - w1
    n = w0 * N[f[:, 0]] + w1 * N[f[:, 1]] + w2 * N[f[:, 2]]
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    nc = n @ R.T
    L = np.array([-0.4, -0.5, -0.75])
    L /= np.linalg.norm(L)
    s = 0.35 + 0.65 * np.clip(nc @ L, 0, 1)
    if colors is None:
        c = np.array([150, 175, 215], np.float32)[None] * np.ones((len(s), 1))
    else:
        c = w0 * colors[f[:, 0]] + w1 * colors[f[:, 1]] + w2 * colors[f[:, 2]]
    img[ys, xs] = c * s[:, None]
    return img.astype(np.uint8)


def err_colors(d, vmax=3.0):
    import matplotlib
    cm = matplotlib.colormaps["RdBu_r"]
    x = np.clip(d / vmax, -1, 1) * 0.5 + 0.5
    rgb = cm(x)[:, :3] * 255
    return rgb[:, ::-1].astype(np.float32)


def picture(tm, meshes, res, reg, path, truth):
    """Rows: views; columns: true, hull (error colours), refined (error colours)."""
    import trimesh  # noqa: F401
    Wi, Hi = 640, 720
    c = tm.bounds.mean(0)
    views = []
    for name, d in [("palm side", np.array([0, 0, -1.0])), ("back side", np.array([0.0, 0.25, 1.0])),
                    ("thumb side", np.array([1.0, 0, -0.4]))]:
        d /= np.linalg.norm(d)
        C = c + 520 * d
        z = c - C
        z /= np.linalg.norm(z)
        x = np.cross([0, 1.0, 0], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        views.append((name, R, -R @ C))
    K = np.array([[1150.0, 0, Wi / 2], [0, 1150.0, Hi / 2], [0, 0, 1]])
    cols = [("true hand (stand-in)", tm, None)]
    for lab, title in [("hull", "visual hull only"), ("refined", "hull + photo refinement")]:
        if lab in meshes:
            m = meshes[lab]
            d = surface_distance(np.asarray(m.vertices), tm)
            cols.append((title, m, err_colors(d)))
    rows = []
    for name, R, t in views:
        tiles = []
        for title, m, colr in cols:
            im = shade_mesh(m, R, t, K, Wi, Hi, colr)
            cv2.putText(im, f"{title} - {name}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            tiles.append(im)
        rows.append(np.concatenate(tiles, 1))
    img = np.concatenate(rows, 0)
    # legend + numbers
    leg = np.full((150, img.shape[1], 3), 255, np.uint8)
    bar = err_colors(np.linspace(-3, 3, 600))
    leg[20:45, 30:630] = bar[None].astype(np.uint8)
    for x, lab in [(30, "-3 mm (inside true)"), (300, "0"), (530, "+3 mm (outside)")]:
        cv2.putText(leg, lab, (x, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    y = 30
    for lab in ["hull", "refined"]:
        if lab in res:
            r = res[lab]
            txt = (f"{lab}: median / 95% distance to true surface (mm)   fingers {r['fingers']['median']:.2f} / {r['fingers']['p95']:.2f}"
                   f"   palm hollow {r.get('palm_hollow', {}).get('median', float('nan')):.2f} / {r.get('palm_hollow', {}).get('p95', float('nan')):.2f}"
                   f"   back {r['back']['median']:.2f} / {r['back']['p95']:.2f}")
            cv2.putText(leg, txt, (680, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            y += 35
    cv2.putText(leg, "colour = distance of the reconstruction to the true surface; synthetic videos",
                (680, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    img = np.concatenate([img, leg], 0)
    cv2.imwrite(path, img)
    print("picture ->", path)



def real_picture(a):
    """Real scan: real video crop | reconstruction from the same camera | coverage from other sides."""
    import pickle
    import trimesh
    import __main__
    __main__.View = rb.View
    takes, _ = pickle.load(open(a.cache, "rb"))
    views = [v for t in takes for v in t]
    mesh = trimesh.load(a.recon, process=False)
    base = os.path.splitext(a.recon)[0]
    labels = np.load(base + "_coverage_labels.npy", allow_pickle=True)
    rep = json.load(open(base + "_report.json"))
    names = rb.LABELS
    lab = np.array([names[int(x)] if not isinstance(x, str) else x for x in labels])
    col = np.array([rb.LABEL_COLORS[x][::-1] for x in lab], np.float32)  # BGR
    v = min(views, key=lambda v: abs(v.ts - a.ts))
    R, t = v.ext()
    H, W = v.img.shape[:2]
    sz = 620
    f = sz / W
    K = v.K.copy()
    K[:2] *= f
    real = cv2.resize(v.img, (sz, sz), interpolation=cv2.INTER_AREA)
    skin = np.tile(np.array([[150, 175, 215]], np.float32), (len(mesh.vertices), 1))
    rec = shade_mesh(mesh, R, t, K, sz, sz, skin)
    cov = shade_mesh(mesh, R, t, K, sz, sz, col)
    # overlay outline of the reconstruction on the real crop
    u, w, z = rb.project(np.asarray(mesh.vertices), R, t, K, np.zeros(5))
    fid, *_ = rb.rasterize(u, w, z, np.asarray(mesh.faces), sz, sz)
    cs, _ = cv2.findContours((fid >= 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ov = real.copy()
    cv2.drawContours(ov, cs, -1, (0, 255, 255), 2)
    tiles = [(real, "real video frame (crop)"), (ov, "yellow line = reconstruction outline"),
             (rec, "reconstruction, same camera"), (cov, "what each part is based on")]
    # other directions around the hand centre, in the hand's own axes
    c = mesh.bounds.mean(0)
    Cs = np.array([-Rv.T @ tv for Rv, tv in (vv.ext() for vv in views)])
    toward = Cs.mean(0) - c
    toward /= np.linalg.norm(toward)
    Kv = np.array([[1100.0, 0, sz / 2], [0, 1100.0, sz / 2], [0, 0, 1]])
    up = np.array([0, -1.0, 0])
    sideways = np.cross(up, toward)
    sideways /= np.linalg.norm(sideways)
    for name, d in [("seen from the screen (never filmed)", -toward),
                    ("from the edge: thickness", 0.25 * toward + sideways),
                    ("from below", 0.3 * toward - up)]:
        d = d / np.linalg.norm(d)
        C = c + 420 * d
        zc = (c - C) / np.linalg.norm(c - C)
        x = np.cross(up, zc)
        if np.linalg.norm(x) < 1e-3:
            x = np.array([1.0, 0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(zc, x)
        Rv = np.stack([x, y, zc])
        tiles.append((shade_mesh(mesh, Rv, -Rv @ C, Kv, sz, sz, col), name))
    tiles = tiles[:6] if len(tiles) >= 6 else tiles
    ims = []
    for im, title in tiles:
        im = im.copy()
        cv2.rectangle(im, (0, 0), (sz, 38), (255, 255, 255), -1)
        cv2.putText(im, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
        ims.append(im)
    while len(ims) % 3:
        ims.append(np.full((sz, sz, 3), 255, np.uint8))
    img = np.concatenate([np.concatenate(ims[i:i + 3], 1) for i in range(0, len(ims), 3)], 0)
    leg = np.full((210, img.shape[1], 3), 255, np.uint8)
    y = 30
    for key, txt in [("photo", "measured: depth from several video frames (stereo)"),
                     ("silhouette", "outline: limited by the hand's outline in the video frames"),
                     ("prior", "filled in: never seen, back = front minus typical hand thickness"),
                     ("cut", "wrist cut")]:
        cv2.rectangle(leg, (20, y - 18), (50, y + 4), tuple(int(x) for x in rb.LABEL_COLORS[key][::-1]), -1)
        cv2.putText(leg, txt, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y += 32
    cov_all = rep.get("coverage", {}).get("per_region_area_fraction", {}).get("all", {})
    vp = rep.get("views_per_surface_point", {})
    lines = [f"real scan, {len(rep.get('views', views))} video frames used; mesh volume {rep['mesh']['volume_ml']:.0f} ml; "
             f"outline agreement with video (IoU) {rep['mask_iou']['final_median']:.2f}",
             f"whole surface: measured by stereo {100 * cov_all.get('photo', 0):.0f}%, outline only "
             f"{100 * cov_all.get('silhouette', 0):.0f}%, filled in {100 * cov_all.get('filled_in', 0):.0f}%",
             f"lens calibration error {list(rep['calibration'].values())[0]['rms_px']:.1f} px; "
             f"the filmed hand is a LEFT hand (palm towards the camera)"]
    y = 40
    for ln in lines:
        cv2.putText(leg, ln, (img.shape[1] // 2 - 300, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y += 34
    img = np.concatenate([img, leg], 0)
    cv2.imwrite(a.picture, img)
    print("picture ->", a.picture)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--out", required=True)
    m.add_argument("--frames", type=int, default=50)
    m.add_argument("--takes", default="1,2")
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--workers", type=int, default=4)
    m.add_argument("--drift", type=float, default=0.5, help="mm amplitude of the hand drift within a take")
    m.add_argument("--drift-deg", type=float, default=0.3)
    m.add_argument("--jitter-px", type=float, default=1.2, help="hand vs board pose jitter per frame (px, 1 sigma per axis)")
    m.add_argument("--breathing", type=float, default=0.004, help="focal breathing amplitude (fraction)")
    m.add_argument("--motion-frac", type=float, default=0.1)
    m.add_argument("--glare-frac", type=float, default=0.4, help="frames with screen glare/overexposure")
    m.add_argument("--glare-max", type=float, default=2.0)
    m.add_argument("--glare-gain-max", type=float, default=1.8)
    m.add_argument("--truth-mesh", default=None, help="hand mesh to film (joints.json next to it)")
    m.add_argument("--fps", type=float, default=6.0)
    e = sub.add_parser("eval")
    e.add_argument("--synth", required=True)
    e.add_argument("--recon", required=True)
    e.add_argument("--picture", default=None)
    e.add_argument("--json", default=None)
    r = sub.add_parser("real-picture", help="picture of a reconstruction from real video")
    r.add_argument("--recon", required=True)
    r.add_argument("--cache", required=True, help="recon_board.py --cache file of that run")
    r.add_argument("--picture", required=True)
    r.add_argument("--ts", type=float, default=0.0, help="time (ms) of the frame to show")
    a = p.parse_args()
    {"make": make, "eval": evaluate, "real-picture": real_picture}[a.cmd](a)


if __name__ == "__main__":
    main()
