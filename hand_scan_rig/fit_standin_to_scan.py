"""Fit standin.py's smooth parametric hand to Olex's real hand scan (board video).

    P=/tmp/claude-0/hs/venv/bin/python
    $P fit_standin_to_scan.py fit --stage 1|2|3 [--rounds N --nfev N --views clean|all|clean_early|clean_late
                                 --scan3d all|stereo|off --label L --from L --scale-bound B]
    $P fit_standin_to_scan.py info       # full-covariance determinedness at the exported theta -> STATE/info.json
    $P fit_standin_to_scan.py export     # 0.5 mm mesh + joints.json + params json -> OUT
    $P fit_standin_to_scan.py metrics    # per-frame IoU / contour distance (all / clean / gated-out), 3D distance
    $P fit_standin_to_scan.py validate [--prefix val|val2 --train .. --test ..]   # held-out camera sector
    $P fit_standin_to_scan.py sheet      # fitted_sheet.jpg (needs run_all.py output + grip_views.py)

States, flags and checks live in STATE (/tmp/claude-0/hs/work/fitted_v2; v1 states in WORK).
Frames: only views passing the mask-quality gate (mask_quality(): no bleeding into the board, no
holes) are used; 39 of the 70.  Priors are Gaussian (gauss_under_soft_l1).

Why: the photogrammetry scan (recon/real/hand_scan.ply) has the right outline and size but a rough,
pitted surface and only ~10% stereo-measured skin, so it cannot be the rig skin.  Instead the
stand-in's smooth signed-distance hand (standin.py: capsules, ellipsoids, palm slab) is given
free shape parameters and fitted to the real data; the result is a clean, watertight hand that
fit_rig / skin / run_all treat like a scan.

Data
 * cameras + per-frame hand masks: the recon_board.py cache (frames 0-9.2 s = one hand posture,
   70 views; mask 0 board, 1 hand, 2 unknown), extracted once to WORK/views.pkl.
 * scan: hand_scan_as_right.ply (the LEFT-hand scan with board X negated) + per-vertex source
   labels hand_scan_coverage_labels.npy (0 stereo-measured, 1 outline-limited (visual hull),
   2 filled from the stand-in template, 3 screen-limited, 4 wrist cut).
 * triangulated MediaPipe landmarks (hand_scan_landmarks.json), mirrored the same way.

Frames.  The model is built in the stand-in's canonical RIGHT-hand frame (wrist joint at the
origin, fingers +Y, palm +Z, thumb +X) and placed by a rigid transform (rotation vector,
translation, optional global scale) into the "as_right" frame = board frame with X negated.
The video cameras are MIRRORED instead of the hand: a board point X_b = M X_a with
M = diag(-1, 1, 1), so the camera extrinsic for as_right points is (R_cam M, t_cam).  The
pixels and masks are the real, unflipped video data; nothing about the hand is mirrored except
the frame it is expressed in, so the fitted model stays a right hand (the SPEC side).

Model parameters (all with an anatomical prior = the stand-in default and a prior sigma):
 shape: per finger the skin radius scale at MCP / PIP / DIP / tip (width, segment taper) and the
   thickness/width ratio of each segment (proximal, middle, distal); the same for the thumb
   (CMC / MCP / IP / tip, metacarpal / proximal / distal); palm breadth, dorsal and palmar
   thickness, outline offsets (thumb side, little-finger side, knuckle row, wrist), thenar size /
   height / shift, hypothenar size / height, MCP pad size, palmar hollow depth, the three
   finger webs and the thumb web (how far up the proximal segment they reach).
 pose: per finger the in-palm direction of the proximal segment (fan), MCP / PIP / DIP flexion
   and a metacarpal direction offset (keeps the metacarpal length); thumb CMC angle, palmar
   elevation, axial roll, MCP and IP flexion; palm arch; rigid placement.  Bone lengths are the
   rig's (hold_rig.blend); a global scale is only fitted in a separate check (--free-scale).

Objective (scipy least_squares, soft-L1, residuals in units of the robust scale):
 (a) silhouettes, the most trustworthy data: for sampled pixels near the mask boundary, the
     signed distance (mm) of the pixel's ray to the model surface (min of the SDF along the
     ray, sampled at 1 mm with a parabolic refinement) must equal the pixel's signed distance
     to the mask boundary (mm at the hand depth); both clipped at +-6 mm.  Unknown pixels and
     the forearm side of the wrist cut are not used.
 (b) 3D: SDF of the model at scan vertices, weight 1.0 stereo-measured, 0.3 outline-limited,
     0 filled-from-model / screen / cut.
 (c) weak landmark term (triangulated MediaPipe joints, sigma 6 mm, wrist / CMC 15 mm).
 (e) neighbouring fingers keep >= 3.5 mm of model clearance from the PIP to the tip (the fingers are
     smooth-unioned with k = 6 mm; closer, they bridge and the mesh gets a handle).
 Outline-limited (visual hull) scan points are an upper bound: weight 0.3 where the model sticks
 out of them, 0.075 where it stays inside.
 Stages: 0 landmarks only (rigid + pose), 1 "before" = plain stand-in shape, pose + placement
 fitted, 2 everything, 3 everything + global scale (adopted only with STATE/use_scale.flag, see
 STATE/scale_check.json; the 2026-10-21 v2 run adopted 0.970, the -3% bound).
 (d) Gaussian priors toward the stand-in's typical anatomy for every shape parameter, weak priors
     on the pose.  How well each parameter is determined: cmd_info (variance reduction from the
     full Gauss-Newton covariance, at the exported theta).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import standin as S  # noqa: E402
from standin import (NAMES, FINGERS4, SEGS4, unit, rot, smin, smax, frame_from, Prim,  # noqa: E402
                     round_cone, ellipsoid, tri_slab, palm_slab, group)

REAL = "/mnt/project-files/hand_scan_rig/recon/real"
RECON_CACHE = "/tmp/claude-0/hs/work/recon/real/cache.pkl"
OUT = "/mnt/project-files/hand_scan_rig/fitted"
WORK = "/tmp/claude-0/hs/work/fitted"                    # views.pkl + frame images (shared)
STATE = os.environ.get("FIT_STATE_DIR", "/tmp/claude-0/hs/work/fitted_v2")   # fit states, info, flags
TIME_WINDOW_MS = (0, 9200)
# mask quality gate (per view, measured against the landmark-only stand-in, see mask_quality):
# spill = mask area 4-30 mm outside the model per mm of model outline (mask bleeding into the board),
# miss = board area > 4 mm inside the model per mm of outline (holes / hand cut short)
GATE_SPILL_MM, GATE_MISS_MM = 0.5, 0.3
SCAN3D_MODES = {"all": {0: 1.0, 1: 0.3}, "stereo": {0: 1.0}, "off": {}}
M_MIRROR = np.diag([-1.0, 1.0, 1.0])
Y_CUT = -10.0            # canonical y of the model's wrist cut (the scan is cut ~10 mm proximal of the wrist)
CUT_MARGIN_MM = 8.0      # pixels closer than this to the projected cut (and beyond it) are not used
CLIP_MM = 6.0            # silhouette residual clip
ROBUST_MM = 1.5          # soft-L1 scale
LABEL_W = {0: 1.0, 1: 0.3, 2: 0.0, 3: 0.0, 4: 0.0}
LABEL_NAMES = {0: "stereo-measured", 1: "outline-limited", 2: "filled-from-model", 3: "screen-limited", 4: "wrist cut"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))


# ============================================================ parameters
# (name, default, prior sigma, lo, hi, group, meaning)
def param_table():
    P = []
    OP = S.OPEN_POSE
    for f in FINGERS4:
        for j in ("mcp", "pip", "dip", "tip"):
            P.append((f"w_{f}_{j}", 1.0, 0.12, 0.6, 1.5, "shape", f"{f}: skin radius scale at the {j.upper()} end"))
        for s in SEGS4:
            P.append((f"t_{f}_{s}", S.THICK_RATIO, 0.08, 0.6, 1.1, "shape", f"{f} {s}: thickness / width"))
    for j in ("cmc", "mcp", "ip", "tip"):
        P.append((f"w_thumb_{j}", 1.0, 0.12, 0.6, 1.5, "shape", f"thumb: skin radius scale at the {j.upper()} end"))
    for s, d in (("metacarpal", 0.9), ("proximal", S.THICK_RATIO), ("distal", S.THICK_RATIO)):
        P.append((f"t_thumb_{s}", d, 0.08, 0.6, 1.1, "shape", f"thumb {s}: thickness / width"))
    P += [
        ("palm_breadth", 1.0, 0.08, 0.8, 1.2, "shape", "palm breadth scale (outline, eminences across the palm)"),
        ("palm_thick_dorsal", 1.0, 0.12, 0.7, 1.3, "shape", "palm: depth of the back of the hand below the mid-plane"),
        ("palm_thick_palmar", 1.0, 0.12, 0.7, 1.3, "shape", "palm: height of the palm surface above the mid-plane"),
        ("off_radial", 0.0, 4.0, -12, 12, "shape", "palm outline offset, thumb side (mm, + outward)"),
        ("off_ulnar", 0.0, 4.0, -12, 12, "shape", "palm outline offset, little-finger side (mm, + outward)"),
        ("off_knuckle", 0.0, 3.0, -10, 10, "shape", "palm outline offset at the knuckle row (mm, + distal)"),
        ("off_wrist", 0.0, 4.0, -12, 12, "shape", "palm half-width offset at the wrist (mm, + outward)"),
        ("thenar_size", 1.0, 0.15, 0.6, 1.6, "shape", "thenar eminence in-plane size"),
        ("thenar_height", 1.0, 0.2, 0.4, 1.8, "shape", "thenar eminence height on the palm side"),
        ("thenar_dx", 0.0, 4.0, -12, 12, "shape", "thenar centre shift toward the thumb (mm)"),
        ("hyp_size", 1.0, 0.15, 0.6, 1.6, "shape", "hypothenar eminence in-plane size"),
        ("hyp_height", 1.0, 0.2, 0.4, 1.8, "shape", "hypothenar eminence height on the palm side"),
        ("pad_size", 1.0, 0.2, 0.5, 1.6, "shape", "MCP palmar pads size"),
        ("hollow_depth", 0.0, 2.0, -6, 6, "shape", "palmar hollow: extra depth (mm, + deeper)"),
        ("web_index", 0.30, 0.08, 0.1, 0.6, "shape", "web index-middle: reach along the proximal segment (fraction)"),
        ("web_middle", 0.30, 0.08, 0.1, 0.6, "shape", "web middle-ring: reach (fraction)"),
        ("web_ring", 0.30, 0.08, 0.1, 0.6, "shape", "web ring-little: reach (fraction)"),
        ("web_thumb", 0.33, 0.10, 0.05, 0.8, "shape", "thumb web: reach along the thumb proximal segment (fraction)"),
        ("web_thumb_index", 0.80, 0.08, 0.5, 1.0, "shape", "thumb web: reach toward the index MCP (fraction of wrist->MCP)"),
    ]
    for f in FINGERS4:
        P += [(f"fan_{f}", OP["fan"][f], 15.0, -40, 40, "pose", f"{f}: in-palm direction of the proximal segment (deg, + thumb side)"),
              (f"mcp_{f}", OP["mcp_flex"], 30.0, -40, 90, "pose", f"{f}: MCP flexion (deg)"),
              (f"pip_{f}", OP["pip_flex"], 30.0, -20, 100, "pose", f"{f}: PIP flexion (deg)"),
              (f"dip_{f}", OP["dip_flex"], 30.0, -25, 90, "pose", f"{f}: DIP flexion (deg)"),
              (f"meta_{f}", 0.0, 3.0, -10, 10, "pose", f"{f}: metacarpal direction offset in the palm plane (deg)")]
    P += [("thumb_cmc", OP["thumb_cmc"], 25.0, -10, 90, "pose", "thumb metacarpal angle from the index metacarpal (deg)"),
          ("thumb_elev", OP["thumb_elev"], 25.0, -40, 80, "pose", "thumb abduction direction palmar elevation (deg)"),
          ("thumb_roll", 0.0, 20.0, -60, 60, "pose", "thumb axial roll of the bending plane (deg)"),
          ("thumb_mcp", OP["thumb_mcp_flex"], 30.0, -25, 80, "pose", "thumb MCP flexion (deg)"),
          ("thumb_ip", OP["thumb_ip_flex"], 30.0, -40, 90, "pose", "thumb IP flexion (deg)"),
          ("arch_keep", OP["arch_keep"], 0.3, -0.2, 1.0, "pose", "fraction of the rig's palm arch kept")]
    for i, ax in enumerate("xyz"):
        P.append((f"rot_{ax}", 0.0, 1e3, -10, 10, "rigid", "placement rotation vector (rad)"))
    for i, ax in enumerate("xyz"):
        P.append((f"tr_{ax}", 0.0, 1e5, -1e4, 1e4, "rigid", "placement translation (mm)"))
    P.append(("log_scale", 0.0, 0.015, -0.03, 0.03, "scale", "global scale (log), fitted only with --free-scale"))
    return P


PT = param_table()
PNAMES = [p[0] for p in PT]
PIDX = {n: i for i, n in enumerate(PNAMES)}
DEFAULT = np.array([p[1] for p in PT], float)
SIGMA = np.array([p[2] for p in PT], float)
LO = np.array([p[3] for p in PT], float)
HI = np.array([p[4] for p in PT], float)
GROUP = np.array([p[5] for p in PT])


def pdict(theta):
    return {n: float(theta[i]) for i, n in enumerate(PNAMES)}


# ============================================================ model: joints
_LAY = None


def layout():
    global _LAY
    if _LAY is None:
        _LAY = S.palm_layout(S.read_rig())
    return _LAY


def build_joints_p(p, lay=None):
    """standin.build_joints generalised: per-finger angles, metacarpal offsets, thumb roll.
    Identical to standin.build_joints(lay, OPEN_POSE) at the default parameters."""
    lay = lay or layout()
    Z = np.array([0.0, 0.0, 1.0])
    J = {"wrist": np.zeros(3)}
    for f in FINGERS4:
        m = lay["mcp"][f].copy()
        m[2] *= p["arch_keep"]
        if p[f"meta_{f}"]:
            m = rot(m, Z, -p[f"meta_{f}"])           # + toward the thumb side (+X)
        J[f"{f}_mcp"] = m
        t = np.radians(p[f"fan_{f}"])
        d0 = np.array([np.sin(t), np.cos(t), 0.0])
        ax = np.cross(d0, Z)
        q = m
        flex = 0.0
        for s, jn, key in zip(SEGS4, ["pip", "dip", "tip"], ["mcp", "pip", "dip"]):
            flex += p[f"{key}_{f}"]
            d = rot(d0, ax, flex)
            q = q + lay["len"][f][s] * d
            J[f"{f}_{jn}"] = q
    cmc = lay["cmc"].copy()
    J["thumb_cmc"] = cmc
    u_idx = unit(J["index_mcp"])
    perp = unit(np.cross(u_idx, Z))
    if perp[0] < 0:
        perp = -perp
    e = np.radians(p["thumb_elev"])
    c = np.radians(p["thumb_cmc"])
    dm = unit(np.cos(c) * u_idx + np.sin(c) * (np.cos(e) * perp + np.sin(e) * Z))
    bend = -perp * 0.7 + Z * 0.7
    bend = unit(bend - (bend @ dm) * dm)
    if p["thumb_roll"]:
        bend = rot(bend, dm, p["thumb_roll"])
    tax = np.cross(dm, bend)
    lt = lay["len"]["thumb"]
    q = cmc + lt["metacarpal"] * dm
    J["thumb_mcp"] = q
    d1 = rot(dm, tax, p["thumb_mcp"])
    q = q + lt["proximal"] * d1
    J["thumb_ip"] = q
    d2 = rot(dm, tax, p["thumb_mcp"] + p["thumb_ip"])
    J["thumb_tip"] = q + lt["distal"] * d2
    return {n: np.asarray(J[n], float) for n in NAMES}, unit(bend)


# ============================================================ model: skin primitives
def build_prims_p(J, tbend, p):
    """standin.build_prims with free shape parameters (identical at the defaults)."""
    Z = np.array([0.0, 0.0, 1.0])
    P = []
    m = {f: J[f"{f}_mcp"] for f in FINGERS4}
    pb = p["palm_breadth"]
    I, M_, R_, K = m["index"], m["middle"], m["ring"], m["pinky"]
    br = 10.0 * pb
    orad, oul, okn, ow = p["off_radial"], p["off_ulnar"], p["off_knuckle"], p["off_wrist"]
    V = [(-21 * pb - ow, -50), (21 * pb + ow, -50), (27 * pb + 0.5 * (ow + orad), 6),
         (I[0] + 11.5 * pb + orad, I[1] - 28), (I[0] + br + orad, I[1] - 4 + 0.5 * okn),
         (I[0] + 2, I[1] + 4 + okn), (M_[0], M_[1] + 4 + okn), (R_[0], R_[1] + 4 + okn), (K[0] - 2, K[1] + 4 + okn),
         (K[0] - br - oul, K[1] - 6 + 0.5 * okn), (K[0] - 9 * pb - oul, K[1] - 34), (-30 * pb - 0.5 * (ow + oul), 6)]
    ymcp = M_[1]
    td, tp = p["palm_thick_dorsal"], p["palm_thick_palmar"]

    def taper(Pp):
        return 0.3 * 6.0 * np.logaddexp(0, -Pp[:, 1] / 6.0)

    def zb(Pp):
        t = np.clip(Pp[:, 1] / ymcp, 0, 1)
        return (-17 + 6 * t) * td + taper(Pp)

    def zt(Pp):
        t = np.clip(Pp[:, 1] / ymcp, 0, 1)
        return (17 - 1.5 * t ** 2) * tp - taper(Pp)
    rr = 12.0 * min(td, tp, 1.0)
    P.append(palm_slab(V, zb, zt, r=rr, k=0, name="palm"))
    P.append(round_cone([0, 12, 0], [0, -60, 0], 27 * pb, 30.5 * pb, Z, squash=0.62 * 0.5 * (td + tp), k=16,
                        name="forearm"))
    hs, hh = p["hyp_size"], p["hyp_height"]
    hyp_c = np.array([K[0] + 4.5 - 4.5 * (hs - 1), 0.42 * K[1], 6.0 * tp * hh])
    hyp_c[0] = hyp_c[0] * pb
    P.append(ellipsoid(hyp_c, frame_from(np.array([-0.18, 1, 0]), Z), [13 * pb * hs, 12 * tp * hh, 32 * hs],
                       k=14, name="hypothenar"))
    cmc, tm = J["thumb_cmc"], J["thumb_mcp"]
    dt = tm - cmc
    dtp = unit(np.array([dt[0], dt[1], 0.0]))
    ts_, th_ = p["thenar_size"], p["thenar_height"]
    then_c = cmc + 0.42 * dt + np.array([-10.0 + p["thenar_dx"], -3.0, 0.0])
    then_c[2] = 7.0 * tp * th_
    P.append(ellipsoid(then_c, frame_from(dtp, Z), [19 * pb * ts_, 15.5 * tp * th_, 30 * ts_], k=12, name="thenar"))
    d_tp = unit(J["thumb_ip"] - tm)
    P.append(tri_slab(cmc + np.array([-4, 4, 3.0]), tm + p["web_thumb"] * np.linalg.norm(J["thumb_ip"] - tm) * d_tp
                      + np.array([-1.5, 0, 1.0]), I * p["web_thumb_index"] + np.array([0, 0, 2.0]), 5.0, k=8,
                      name="thumbweb"))
    rad = {}
    for f in FINGERS4:
        w = S.WIDTH_PIP[f] * S.RADIUS_FIT * 0.5
        rad[f] = (w * 1.07 * p[f"w_{f}_mcp"], w * p[f"w_{f}_pip"], w * 0.92 * p[f"w_{f}_dip"], w * 0.85 * p[f"w_{f}_tip"])
    for f in FINGERS4:
        d = unit(J[f"{f}_pip"] - m[f])
        w = S.WIDTH_PIP[f] * p[f"w_{f}_mcp"]
        ps = p["pad_size"]
        P.append(ellipsoid(m[f] + np.array([0, -4.0, 9.0 * tp]) - 1.0 * d,
                           frame_from(d, Z), [0.5 * w * ps, 5.5 * tp * ps, 0.5 * w * ps], k=10, name=f"pad_{f}"))
        P.append(ellipsoid(m[f] + np.array([0, -1.0, -3.0 * td]), frame_from(d, Z),
                           [0.47 * w, 6.5 * td, 0.5 * w], k=6, name=f"knuckle_{f}"))
    for f in FINGERS4:
        r_mcp, r_pip, r_dip, r_tip = rad[f]
        sq = [p[f"t_{f}_{s}"] for s in SEGS4]
        a, b, c, t = m[f], J[f"{f}_pip"], J[f"{f}_dip"], J[f"{f}_tip"]
        dd = unit(t - c)
        tip_end = t - dd * (r_tip - S.TIP_INSET)
        G = [round_cone(a, b, r_mcp, r_pip * 0.97, Z, squash=sq[0], k=0),
             round_cone(b, c, r_pip * 0.97, r_dip * 0.97, Z, squash=sq[1], k=0),
             round_cone(c, tip_end, r_dip * 0.97, r_tip, Z, squash=sq[2], k=0)]
        for jp, r, sqj in [(b, r_pip, 0.5 * (sq[0] + sq[1])), (c, r_dip, 0.5 * (sq[1] + sq[2]))]:
            dj = unit(t - jp)
            G.append(ellipsoid(jp - 1.0 * Z, frame_from(dj, Z),
                               [r * 0.98, r * 0.78 * sqj / S.THICK_RATIO, r * 0.7], k=3.0))
        for (p0, p1, r, sz, fr), sqs in zip([(a, b, r_mcp, 0.45, 0.5), (b, c, r_pip, 0.4, 0.5),
                                             (c, tip_end, r_dip, 0.5, 0.62)], sq):
            dj = unit(p1 - p0)
            L = np.linalg.norm(p1 - p0)
            ts = sqs / S.THICK_RATIO
            G.append(ellipsoid(p0 + dj * L * fr + Z * r * 0.3 * ts, frame_from(dj, Z),
                               [r * 0.78, r * 0.6 * ts, L * sz], k=3.0))
        P.append(group(G, k=6.0, name=f"finger_{f}"))
    for fa, fb in [("index", "middle"), ("middle", "ring"), ("ring", "pinky")]:
        fr = p[f"web_{fa}"]
        pa_ = m[fa] + fr * (J[f"{fa}_pip"] - m[fa]) + Z * 2.5
        pb_ = m[fb] + fr * (J[f"{fb}_pip"] - m[fb]) + Z * 2.5
        P.append(round_cone(pa_, pb_, 5.0, 5.0, Z, squash=0.8, k=5, name=f"web_{fa}"))
        qa, qb = m[fa] + Z * 2.0, m[fb] + Z * 2.0
        P.append(tri_slab(qa, qb, pb_, 5.0, k=5, name=f"webfill_{fa}"))
        P.append(tri_slab(qa, pb_, pa_, 5.0, k=5, name=f"webfill_{fa}"))
    wt = S.WIDTH_PIP["thumb"] * S.RADIUS_FIT * 0.5
    rc, rm, ri, rt = (wt * 0.95 * p["w_thumb_cmc"], wt * 1.02 * p["w_thumb_mcp"], wt * 0.98 * p["w_thumb_ip"],
                      wt * 0.86 * p["w_thumb_tip"])
    sqm, sqp, sqd = p["t_thumb_metacarpal"], p["t_thumb_proximal"], p["t_thumb_distal"]
    ip, tt = J["thumb_ip"], J["thumb_tip"]
    dd = unit(tt - ip)
    tip_end = tt - dd * (rt - S.TIP_INSET)
    rpip = wt * p["w_thumb_ip"]
    G = [round_cone(cmc, tm, rc, rm, tbend, squash=sqm, k=0),
         round_cone(tm, ip, rm, ri, tbend, squash=sqp, k=0),
         round_cone(ip, tip_end, ri, rt, tbend, squash=sqd, k=0),
         ellipsoid(ip - 1.0 * tbend, frame_from(dd, tbend),
                   [0.98 * rpip, 0.78 * rpip * 0.5 * (sqp + sqd) / S.THICK_RATIO, 0.7 * rpip], k=3.0)]
    for p0, p1, r, sqs in [(tm, ip, wt * p["w_thumb_mcp"], sqp), (ip, tip_end, rpip, sqd)]:
        dj = unit(p1 - p0)
        L = np.linalg.norm(p1 - p0)
        ts = sqs / S.THICK_RATIO
        G.append(ellipsoid(p0 + dj * L * 0.6 + tbend * r * 0.3 * ts, frame_from(dj, tbend),
                           [r * 0.8, r * 0.6 * ts, L * 0.5], k=3.0))
    P.append(group(G, k=8.0, name="thumb"))
    hol = np.array([(I[0] + K[0]) * 0.5 - 2, 0.55 * ymcp, 17 * tp + 5.0 - p["hollow_depth"]])
    P.append(ellipsoid(hol, np.eye(3), [22 * pb, 24, 7.0], k=10, op="sub", name="hollow"))
    return P


def model(theta):
    p = pdict(theta)
    J, tb = build_joints_p(p)
    return J, tb, build_prims_p(J, tb, p)


def eval_fast(prims, P, cut_y=Y_CUT, cell=28.0):
    """standin.eval_sdf evaluated per spatial cell (each cell culls the primitives whose box +
    blend margin misses it), cut flat at y = cut_y.  Points far from every primitive get 1e3."""
    P = np.asarray(P, float)
    out = np.empty(len(P))
    key = np.floor(P / cell).astype(np.int64)
    key = (key[:, 0] + 4096) * 67108864 + (key[:, 1] + 4096) * 8192 + (key[:, 2] + 4096)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    cuts = np.flatnonzero(np.diff(ks)) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(ks)]
    kmax = max(pr.k for pr in prims) + 3.0     # a primitive matters wherever any later blend can reach it
    boxes = [(pr.lo - kmax, pr.hi + kmax) for pr in prims]
    for s0, s1 in zip(starts, ends):
        idx = order[s0:s1]
        Q = P[idx]
        lo, hi = Q.min(0), Q.max(0)
        d = np.full(len(Q), 1e3)
        for pr, (blo, bhi) in zip(prims, boxes):
            if np.any(blo > hi) or np.any(bhi < lo):
                continue
            if pr.op == "union":
                d = smin(d, pr.fn(Q), pr.k)
            else:
                d = smax(d, -pr.fn(Q), pr.k)
        out[idx] = d
    return np.maximum(out, cut_y - P[:, 1])


# ============================================================ placement
def rotvec_mat(r):
    return cv2.Rodrigues(np.asarray(r, float).reshape(3, 1))[0]


def placement(theta):
    """canonical -> as_right: X = s R x + t."""
    p = pdict(theta)
    R = rotvec_mat([p["rot_x"], p["rot_y"], p["rot_z"]])
    t = np.array([p["tr_x"], p["tr_y"], p["tr_z"]])
    return R, t, float(np.exp(p["log_scale"]))


def to_canon(X, R, t, s):
    return ((X - t) @ R) / s


def to_world(x, R, t, s):
    return s * (x @ R.T) + t


def sdf_world(theta, X, prims=None):
    if prims is None:
        _, _, prims = model(theta)
    R, t, s = placement(theta)
    return s * eval_fast(prims, to_canon(X, R, t, s))


def mesh_of(theta, res=1.0, world=True, smooth=True):
    """Marching cubes of the model SDF (canonical grid, res mm); returns trimesh (world = as_right)."""
    import trimesh
    from skimage.measure import marching_cubes
    J, tb, prims = model(theta)
    pts = np.array(list(J.values()))
    lo = np.floor(np.minimum(pts.min(0) - 26, [-45, Y_CUT - 3, -30]))
    hi = np.ceil(np.maximum(pts.max(0) + 26, [45, 0, 35]))
    xs, ys, zs = (np.arange(lo[i], hi[i] + res, res) for i in range(3))
    G = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    vol = np.empty(len(G), np.float32)
    for c0 in range(0, len(G), 400_000):
        vol[c0:c0 + 400_000] = eval_fast(prims, G[c0:c0 + 400_000])
    vol = vol.reshape(len(xs), len(ys), len(zs))
    V, Fc, _, _ = marching_cubes(vol, 0.0, spacing=(res, res, res))
    V += lo
    m = trimesh.Trimesh(V, Fc, process=False)
    if smooth:
        trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=6)
    if world:
        R, t, s = placement(theta)
        m = trimesh.Trimesh(to_world(np.asarray(m.vertices), R, t, s), m.faces, process=False)
    return m


# ============================================================ data
def mask_to_sdf(mask):
    from scipy import ndimage
    fg, bg = mask == 1, mask == 0
    d_fg = ndimage.distance_transform_edt(~fg)
    d_bg = ndimage.distance_transform_edt(~bg)
    sdf = np.where(fg, -(d_bg - 0.5), d_fg - 0.5).astype(np.float32)
    sdf[mask == 2] = np.nan
    return sdf


def load_views():
    """Frames 0-9.2 s of the recon cache: cameras (mirrored for as_right points) + masks."""
    path = os.path.join(WORK, "views.pkl")
    if not os.path.exists(path):
        import recon_board
        import __main__
        __main__.View = recon_board.View
        takes, _ = pickle.load(open(RECON_CACHE, "rb"))
        vs = [v for v in takes[0] if v.info.get("fg_px", 0) >= 500
              and TIME_WINDOW_MS[0] <= v.ts <= TIME_WINDOW_MS[1]]
        os.makedirs(os.path.join(WORK, "imgs"), exist_ok=True)
        pickle.dump([dict(idx=v.idx, ts=v.ts, K=v.K, dist=np.ravel(v.dist), R=v.R, t=np.ravel(v.t),
                          mask=v.mask, lm=v.lm) for v in vs], open(path, "wb"), protocol=4)
        for v in vs:
            cv2.imwrite(os.path.join(WORK, "imgs", f"f{v.idx:03d}.jpg"), v.img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    vs = pickle.load(open(path, "rb"))
    for v in vs:
        v["Ra"] = v["R"] @ M_MIRROR             # as_right point -> camera
        v["C"] = -v["Ra"].T @ v["t"]
        v["f"] = float(v["K"][0, 0])
        v["sdf"] = mask_to_sdf(v["mask"])
    return vs


def mask_quality(views):
    """Per-view mask quality against a reference hand: state 'gate_ref' = the plain stand-in shape
    with pose and placement fitted (robust soft-L1) to every 3rd frame of all 70, i.e. the first
    run's s1_before.  The landmark-only pose (s0) is too far off (spill/miss 0.4-0.6 mm on good
    frames).  The gate looks only for gross mask errors (> 4 mm), so the reference's default finger
    widths do not matter.  Cached in STATE/mask_quality.json."""
    from scipy import ndimage
    path = os.path.join(STATE, "mask_quality.json")
    if os.path.exists(path):
        q = json.load(open(path))
        if all(str(v["idx"]) in q for v in views):
            return q
    th = load_state("gate_ref")
    mesh = mesh_of(th, 1.0)
    q = {}
    for v in views:
        H, W = v["mask"].shape
        r, zb = render_mask(mesh, v)
        mmpp = float(np.median(zb[r])) / v["f"]
        keep = wrist_keep(th, v, (H, W))
        din = ndimage.distance_transform_edt(r) * mmpp
        dout = ndimage.distance_transform_edt(~r) * mmpp
        fg, bg = v["mask"] == 1, v["mask"] == 0
        edge = r & ~cv2.erode(r.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & keep
        L = max(edge.sum() * mmpp, 1.0)
        spill = (fg & (dout > 4) & (dout < 30) & keep).sum() * mmpp ** 2 / L
        miss = (bg & (din > 4) & keep).sum() * mmpp ** 2 / L
        q[str(v["idx"])] = {"ts_ms": int(v["ts"]), "spill_mm": round(float(spill), 3), "miss_mm": round(float(miss), 3),
                            "clean": bool(spill < GATE_SPILL_MM and miss < GATE_MISS_MM)}
    jdump(q, path)
    return q


def select_views(views, mode):
    """all | clean (mask gate) | clean_early (clean, ts < 2 s) | clean_late (clean, ts >= 3 s)."""
    if mode == "all":
        return views
    q = mask_quality(views)
    vs = [v for v in views if q[str(v["idx"])]["clean"]]
    if mode == "clean_early":
        vs = [v for v in vs if v["ts"] < 2000]
    elif mode == "clean_late":
        vs = [v for v in vs if v["ts"] >= 3000]
    return vs


def load_scan():
    import trimesh
    m = trimesh.load(os.path.join(REAL, "hand_scan_as_right.ply"), process=False)
    lab = np.load(os.path.join(REAL, "hand_scan_coverage_labels.npy"))
    lmj = json.load(open(os.path.join(REAL, "hand_scan_landmarks.json")))
    L = {n: M_MIRROR @ np.array(x) for n, x in lmj["landmarks_output_frame_mm"].items()}
    rep = lmj["takes"][0]["reprojection_px"]
    return m, lab, L, rep


def project(X, R, t, K, dist):
    import recon_board
    return recon_board.project(X, R, t, K, dist)


def render_mask(mesh, v, shrink=1):
    import recon_board
    H, W = v["mask"].shape
    K = v["K"].copy()
    if shrink != 1:
        K[:2] /= shrink
        K[0, 2] += 0.5 / shrink - 0.5
        K[1, 2] += 0.5 / shrink - 0.5
    u, vv, z = project(np.asarray(mesh.vertices), v["Ra"], v["t"], K, v["dist"])
    fid, _, _, zb = recon_board.rasterize(u, vv, z, np.asarray(mesh.faces), W // shrink, H // shrink)
    return fid >= 0, zb


def wrist_keep(theta, v, shape, margin_mm=CUT_MARGIN_MM):
    """Pixels on the hand side of the projected wrist cut, margin_mm away from it."""
    R, t, s = placement(theta)
    P = to_world(np.array([[0, Y_CUT, 0], [0, Y_CUT + 30, 0]], float), R, t, s)
    u, vv, z = project(P, v["Ra"], v["t"], v["K"], v["dist"])
    d = np.array([u[1] - u[0], vv[1] - vv[0]])
    d /= np.linalg.norm(d)
    H, W = shape
    ys, xs = np.mgrid[0:H, 0:W]
    side = (xs - u[0]) * d[0] + (ys - vv[0]) * d[1]
    return side > margin_mm * v["f"] / z[0]


def pixel_rays(v, px):
    """Undistorted ray directions (as_right frame, camera z component 1) for pixels (N,2)."""
    n = cv2.undistortPoints(px.reshape(-1, 1, 2).astype(np.float64), v["K"], v["dist"]).reshape(-1, 2)
    dc = np.c_[n, np.ones(len(n))]
    return dc @ v["Ra"]          # = (Ra^T dc^T)^T


# ============================================================ problem
class Problem:
    """Residuals of one round: fixed pixel rays (with depth windows), scan points, landmarks."""

    def __init__(self, theta, views, scan, free, rng, lam_sil=0.05, lam_3d=0.05, lam_lm=1.0,
                 n_band=None, n_out=None, n_scan=(4000, 4000), use_sil=True, use_3d=True, lm_only=False,
                 scan3d="all"):
        self.theta0 = theta.copy()
        self.free = np.asarray(free)
        self.lam = (lam_sil, lam_3d, lam_lm)
        self.lm_only = lm_only
        mesh, lab, L, rep = scan
        self.L = np.array([L[n] for n in NAMES])
        sig = np.full(21, 6.0)
        sig[NAMES.index("wrist")] = 15.0
        sig[NAMES.index("thumb_cmc")] = 15.0
        self.lm_sig = sig
        # scan points
        V = np.asarray(mesh.vertices)
        P, W = [], []
        lw = SCAN3D_MODES[scan3d]
        for labv, n in ((0, n_scan[0]), (1, n_scan[1])):
            idx = np.flatnonzero(lab == labv)
            idx = rng.choice(idx, min(n, len(idx)), replace=False)   # same draw in every mode
            if labv not in lw:
                idx = idx[:0]
            P.append(V[idx])
            W.append(np.full(len(idx), lw.get(labv, 0.0)))
        use_3d = use_3d and len(lw) > 0
        self.scanP = np.concatenate(P) if use_3d else np.zeros((0, 3))
        self.scanW = np.concatenate(W) if use_3d else np.zeros(0)
        self.scanHull = (np.concatenate([np.zeros(len(P[0]), bool), np.ones(len(P[1]), bool)])
                         if use_3d else np.zeros(0, bool))
        self.rayX = np.zeros((0, 3))
        self.nrays = 0
        if use_sil and not lm_only:
            # ~11.5k rays in total whatever the number of views (24 views x 480 in the first run)
            n_band = n_band or int(round(9600 / len(views)))
            n_out = n_out or int(round(1920 / len(views)))
            self._sample_rays(theta, views, rng, n_band, n_out)

    def _sample_rays(self, theta, views, rng, n_band, n_out):
        from scipy import ndimage
        mesh = mesh_of(theta, res=2.0, smooth=False)
        S_, Tgt, Dep, Vid = [], [], [], []
        for vi, v in enumerate(views):
            H, W = v["mask"].shape
            rend, zb = render_mask(mesh, v)
            if rend.sum() < 100:
                continue
            zmed = float(np.median(zb[rend]))
            mmpp = zmed / v["f"]
            roi = cv2.dilate(rend.astype(np.uint8), np.ones((3, 3), np.uint8),
                             iterations=int(12 / mmpp)).astype(bool)
            keep = roi & np.isfinite(v["sdf"]) & wrist_keep(theta, v, (H, W))
            s_mm = v["sdf"] * mmpp
            band = keep & (np.abs(s_mm) < 7)
            out = keep & (s_mm >= 7)
            # depth of the nearest rendered pixel (the silhouette depth for pixels off the model)
            _, (iy, ix) = ndimage.distance_transform_edt(~rend, return_indices=True)
            zn = zb[iy, ix]
            for sel, n in ((band, n_band), (out, n_out)):
                ys, xs = np.nonzero(sel)
                if not len(ys):
                    continue
                k = rng.choice(len(ys), min(n, len(ys)), replace=False)
                ys, xs = ys[k], xs[k]
                S_.append(np.c_[xs, ys].astype(float))
                Tgt.append(s_mm[ys, xs])
                Dep.append(zn[ys, xs])
                Vid.append(np.full(len(ys), vi))
        px = np.concatenate(S_)
        self.tgt = np.clip(np.concatenate(Tgt), -CLIP_MM, CLIP_MM)
        dep = np.concatenate(Dep)
        vid = np.concatenate(Vid)
        offs = np.arange(-9.0, 18.01, 1.5)
        self.nsamp = len(offs)
        X = np.empty((len(px), len(offs), 3))
        for vi in np.unique(vid):
            v = views[vi]
            sel = vid == vi
            d = pixel_rays(v, px[sel])
            X[sel] = v["C"][None, None] + d[:, None, :] * (dep[sel][:, None] + offs[None])[..., None]
        self.rayX = X.reshape(-1, 3)
        self.ray_vid = vid
        self.ray_px = px
        self.nrays = len(px)

    def ray_min(self, d):
        """min along each ray with a parabolic refinement."""
        D = d.reshape(self.nrays, self.nsamp)
        D = np.minimum(D, 50.0)
        i = np.clip(np.argmin(D, 1), 1, self.nsamp - 2)
        r = np.arange(self.nrays)
        a, b, c = D[r, i - 1], D[r, i], D[r, i + 1]
        den = a - 2 * b + c
        with np.errstate(divide="ignore", invalid="ignore"):
            off = np.where(den > 1e-9, 0.5 * (a - c) / den, 0.0)
        off = np.clip(off, -1, 1)
        m = b - 0.25 * (a - c) * off
        return np.minimum(m, D.min(1))

    def residuals(self, x, parts=False):
        th = self.theta0.copy()
        th[self.free] = x
        J, tb, prims = model(th)
        R, t, s = placement(th)
        Jw = to_world(np.array([J[n] for n in NAMES]), R, t, s)
        r_lm = ((Jw - self.L) / self.lm_sig[:, None]).ravel() * np.sqrt(self.lam[2])
        pri = (th - DEFAULT) / SIGMA
        pmask = np.isin(GROUP, ["shape", "pose", "scale"])
        r_pr = pri[pmask]
        out = {"lm": r_lm, "prior": gauss_under_soft_l1(r_pr), "gap": gauss_under_soft_l1(finger_gap_residuals(th, J))}
        if not self.lm_only:
            xc = to_canon(np.concatenate([self.rayX, self.scanP]), R, t, s)
            d = s * eval_fast(prims, xc)
            if len(self.rayX):
                nr = len(self.rayX)
                m = self.ray_min(d[:nr])
                r_sil = (np.clip(m, -CLIP_MM, CLIP_MM) - self.tgt) / ROBUST_MM * np.sqrt(self.lam[0])
                out["sil"] = r_sil
                d = d[nr:]
            if len(self.scanP):
                yc = xc[-len(self.scanP):, 1]
                w = self.scanW * (yc > Y_CUT + 10)
                # outline-limited points are the visual hull = an upper bound of the skin: full weight
                # (0.3) where the model sticks out of it (scan point inside the model, d < 0), a quarter
                # of that where the model stays inside it
                w = np.where(self.scanHull & (d > 0), 0.25 * w, w)
                out["scan"] = np.clip(d, -20, 20) / ROBUST_MM * np.sqrt(w * self.lam[1])
        if parts:
            return out
        return np.concatenate([out[k] for k in ("sil", "scan", "lm", "prior", "gap") if k in out])


def gauss_under_soft_l1(r):
    """least_squares applies soft-L1 to every residual; the priors (and the finger-gap hinge) are
    meant to be Gaussian: return z with soft_l1(z) = 2(sqrt(1+z^2)-1) == r^2, so their cost stays
    quadratic.  Under plain soft-L1 a prior grows only linearly far from its mean, and weakly
    observed parameters (back of the hand, wrist outline, thumb CMC radius, palm hollow) ran from
    bound to bound between fits on different frame sets (2026-10-21 v1 and v2 soft-prior runs)."""
    r = np.asarray(r, float)
    if not GAUSS_WRAP:
        return r
    return np.sign(r) * np.sqrt((1.0 + 0.5 * r * r) ** 2 - 1.0)


GAUSS_WRAP = True        # cmd_info turns it off to get the plain Gaussian prior rows


MIN_GAP_MM = 3.5          # fingers are smooth-unioned with k = 6: closer than k/2 they bridge


def finger_gap_residuals(th, J=None):
    """Neighbouring fingers must not touch in the open pose (a skin bridge = a handle in the mesh,
    which breaks rigging): hinge penalty on (centre-line distance - both skin half-widths) below
    MIN_GAP_MM, from the PIP to the tip (proximal of the PIP the webs join them anyway)."""
    p = pdict(th)
    if J is None:
        J, _ = build_joints_p(p)
    t = np.linspace(0, 1, 7)[:, None]
    lines = {}
    for f in FINGERS4:
        w = S.WIDTH_PIP[f] * S.RADIUS_FIT * 0.5
        rr = [w * p[f"w_{f}_pip"], w * 0.92 * p[f"w_{f}_dip"], w * 0.85 * p[f"w_{f}_tip"]]
        pts, rad = [], []
        for (a_, b_), (ra, rb) in zip([("pip", "dip"), ("dip", "tip")], [(rr[0], rr[1]), (rr[1], rr[2])]):
            pts.append(J[f"{f}_{a_}"] + t * (J[f"{f}_{b_}"] - J[f"{f}_{a_}"]))
            rad.append(ra + t[:, 0] * (rb - ra))
        lines[f] = (np.concatenate(pts), np.concatenate(rad) * 1.06)    # joint bulges ~6% wider
    r = []
    for fa, fb in [("index", "middle"), ("middle", "ring"), ("ring", "pinky")]:
        (A, ra), (B, rb) = lines[fa], lines[fb]
        D = np.linalg.norm(A[:, None] - B[None], axis=2) - ra[:, None] - rb[None]
        r.append(max(0.0, MIN_GAP_MM - float(D.min())) / 0.2)
    return np.array(r)


_PB = None


def _res_worker(x):
    return _PB.residuals(x)


def solve(pb, x0, lo, hi, xs, max_nfev, workers=4, label=""):
    """least_squares (trf, soft-L1) with a parallel forward-difference Jacobian."""
    from scipy.optimize import least_squares
    global _PB
    _PB = pb
    h = 0.05 * xs
    pool = mp.get_context("fork").Pool(workers) if workers > 1 else None
    t0 = time.time()
    ncall = [0]

    def fun(x):
        ncall[0] += 1
        return pb.residuals(x)

    def jac(x):
        f0 = pb.residuals(x)
        xx = []
        for i in range(len(x)):
            y = x.copy()
            hi_ = h[i] if x[i] + h[i] <= hi[i] else -h[i]
            y[i] += hi_
            xx.append(y)
        F = pool.map(_res_worker, xx, chunksize=max(1, len(xx) // (4 * workers))) if pool else [pb.residuals(y) for y in xx]
        Jm = np.empty((len(f0), len(x)))
        for i, (y, f) in enumerate(zip(xx, F)):
            Jm[:, i] = (f - f0) / (y[i] - x[i])
        return Jm
    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
    res = least_squares(fun, x0, jac=jac, bounds=(lo, hi), x_scale=xs, loss="soft_l1", f_scale=1.0,
                        max_nfev=max_nfev, xtol=1e-4, ftol=1e-5, verbose=0)
    if pool:
        pool.close()
    log(f"{label}: cost {res.cost:.2f} nfev {res.nfev} status {res.status} {time.time() - t0:.0f}s")
    return res


# ============================================================ stages
def x_scale_of():
    xs = SIGMA.copy()
    for i, n in enumerate(PNAMES):
        if GROUP[i] == "pose":
            xs[i] = 5.0 if n != "arch_keep" else 0.1
            if n.startswith("meta_"):
                xs[i] = 1.0
        elif n.startswith("rot_"):
            xs[i] = 0.02
        elif n.startswith("tr_"):
            xs[i] = 2.0
        elif n == "log_scale":
            xs[i] = 0.005
    return xs


XS = x_scale_of()


def kabsch(A, B, w=None):
    w = np.ones(len(A)) if w is None else np.asarray(w, float)
    ca = (w[:, None] * A).sum(0) / w.sum()
    cb = (w[:, None] * B).sum(0) / w.sum()
    Hm = ((A - ca) * w[:, None]).T @ (B - cb)
    U, _, Vt = np.linalg.svd(Hm)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def init_theta(L):
    th = DEFAULT.copy()
    J, _ = build_joints_p(pdict(th))
    names = ["wrist", "thumb_cmc", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp",
             "index_pip", "middle_pip", "ring_pip", "pinky_pip"]
    w = [0.3, 0.3, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]
    R, t = kabsch(np.array([J[n] for n in names]), np.array([L[n] for n in names]), w)
    th[PIDX["rot_x"]:PIDX["rot_x"] + 3] = cv2.Rodrigues(R)[0].ravel()
    th[PIDX["tr_x"]:PIDX["tr_x"] + 3] = t
    return th


def free_idx(groups, extra=()):
    return np.array([i for i, g in enumerate(GROUP) if g in groups or PNAMES[i] in extra])


def run_stage(theta, views, scan, groups, rounds, max_nfev, seed, label, workers=4, **kw):
    free = free_idx(groups)
    for r in range(rounds):
        rng = np.random.default_rng(seed + r)
        t0 = time.time()
        pb = Problem(theta, views, scan, free, rng, **kw)
        log(f"{label} round {r}: {pb.nrays} rays x {getattr(pb, 'nsamp', 0)} samples, {len(pb.scanP)} scan pts "
            f"({time.time() - t0:.0f}s setup)")
        res = solve(pb, theta[free], LO[free], HI[free], XS[free], max_nfev, workers, f"{label} r{r}")
        theta = theta.copy()
        theta[free] = res.x
        st = {"theta": theta.tolist(), "names": PNAMES, "stage": label, "round": r, "cost": float(res.cost)}
        jdump(st, os.path.join(STATE, f"state_{label}.json"))
        jdump(st, os.path.join(STATE, f"state_{label}_r{r}.json"))     # per round (ray-seed spread)
    return theta


def load_state(label):
    p = os.path.join(STATE, f"state_{label}.json")
    if not os.path.exists(p):
        return None
    st = json.load(open(p))
    th = DEFAULT.copy()
    for n, v in zip(st["names"], st["theta"]):
        if n in PIDX:
            th[PIDX[n]] = v
    return th


def cmd_fit(a):
    """Stages: 0 landmarks only; 1 plain stand-in shape (rigid + pose); 2 everything; 3 everything +
    global scale.  --views picks the frames (default: the mask-quality gate 'clean', every view
    used), --scan3d which scan points enter the 3D term, --label / --from rename the states (for
    validation and scale variants), --scale-bound widens the log-scale bound (diagnostics only)."""
    os.makedirs(STATE, exist_ok=True)
    views = load_views()
    scan = load_scan()
    sub = select_views(views, a.views)[::a.view_step]
    log(f"{len(views)} views in {TIME_WINDOW_MS} ms, views '{a.views}': {len(sub)} used in the optimiser "
        f"(frames {[int(v['idx']) for v in sub]}), scan 3D '{a.scan3d}'")
    if a.scale_bound:
        LO[PIDX["log_scale"]], HI[PIDX["log_scale"]] = -a.scale_bound, a.scale_bound
    st = a.stage
    preset = {"0": ("s0_landmarks", None, ["rigid", "pose"], 0),
              "1": ("s1_before", "s0_landmarks", ["rigid", "pose"], 100),
              "2": ("s2_after", "s1_before", ["rigid", "pose", "shape"], 200),
              "3": ("s3_scale", "s2_after", ["rigid", "pose", "shape", "scale"], 300)}[st]
    label = a.label or preset[0]
    src = getattr(a, "from") or preset[1]
    if st == "0":
        th = init_theta(scan[2])
        th = run_stage(th, sub, scan, ["rigid", "pose"], 1, 60, 0, label, workers=1, lm_only=True)
    else:
        th = load_state(label) if (a.resume and load_state(label) is not None) else load_state(src)
        if th is None:
            raise SystemExit(f"no start state {src}")
        th = run_stage(th, sub, scan, preset[2], a.rounds, a.nfev, preset[3] + 10 * a.seed_off, label,
                       workers=a.workers, scan3d=a.scan3d)
    log("theta", json.dumps({n: round(float(v), 3) for n, v in zip(PNAMES, th)}))


# ============================================================ metrics
def silhouette_metrics(meshes, views, theta_cut, roi_mm=10.0, clip_mm=10.0):
    """Per view and per mesh: IoU and symmetric contour distance (mm at the hand depth, each
    contour pixel's distance to the other contour clipped at clip_mm, mean of both directions).
    One common region of interest per view for all meshes: known pixels (not 'unknown'), on the
    hand side of the wrist cut of theta_cut (8 mm margin), within roi_mm of any of the meshes."""
    from scipy import ndimage
    out = {k: [] for k in meshes}
    for v in views:
        H, W = v["mask"].shape
        rend = {k: render_mask(m, v) for k, m in meshes.items()}
        anyr = np.zeros((H, W), bool)
        zs = []
        for k, (r, zb) in rend.items():
            anyr |= r
            zs.append(np.median(zb[r]))
        mmpp = float(np.median(zs)) / v["f"]
        roi = cv2.dilate(anyr.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=int(roi_mm / mmpp)).astype(bool)
        known = v["mask"] != 2
        roi &= known & wrist_keep(theta_cut, v, (H, W))
        fg = (v["mask"] == 1)
        er = np.ones((3, 3), np.uint8)
        mb = fg & ~cv2.erode(fg.astype(np.uint8), er).astype(bool)
        # mask boundary pixels next to unknown pixels are not a real hand outline
        near_unk = cv2.dilate((~known).astype(np.uint8), er, iterations=2).astype(bool)
        mb &= roi & ~near_unk
        dt_m = ndimage.distance_transform_edt(~(fg & ~cv2.erode(fg.astype(np.uint8), er).astype(bool)))
        for k, (r, _) in rend.items():
            inter = (fg & r & roi).sum()
            union = ((fg | r) & roi).sum()
            rb = r & ~cv2.erode(r.astype(np.uint8), er).astype(bool)
            dt_r = ndimage.distance_transform_edt(~rb)
            rb &= roi & ~near_unk
            d1 = np.minimum(dt_m[rb] * mmpp, clip_mm)
            d2 = np.minimum(dt_r[mb] * mmpp, clip_mm)
            out[k].append({"frame": int(v["idx"]), "ts_ms": int(v["ts"]), "iou": float(inter / max(union, 1)),
                           "contour_mm": float(0.5 * (d1.mean() + d2.mean())) if len(d1) and len(d2) else float("nan"),
                           "model_to_mask_mm": float(d1.mean()) if len(d1) else float("nan"),
                           "mask_to_model_mm": float(d2.mean()) if len(d2) else float("nan")})
    return out


def summarize(rows):
    iou = np.array([r["iou"] for r in rows])
    cd = np.array([r["contour_mm"] for r in rows])
    wi, wc = int(np.argmin(iou)), int(np.nanargmax(cd))
    return {"iou_median": float(np.median(iou)), "iou_p10": float(np.percentile(iou, 10)),
            "iou_worst": float(iou[wi]), "iou_worst_frame": rows[wi]["frame"],
            "contour_mm_median": float(np.nanmedian(cd)), "contour_mm_p90": float(np.nanpercentile(cd, 90)),
            "contour_mm_worst": float(cd[wc]), "contour_worst_frame": rows[wc]["frame"], "frames": len(rows)}


def scan_distance(theta, scan, mesh_world):
    """Distance of the scan's stereo-measured (and outline-limited) vertices to the model surface."""
    from scipy.spatial import cKDTree
    V = np.asarray(scan[0].vertices)
    lab = scan[1]
    R, t, s = placement(theta)
    yc = to_canon(V, R, t, s)[:, 1]
    _, _, prims = model(theta)
    tree = cKDTree(np.asarray(mesh_world.vertices))
    out = {}
    for labv in (0, 1):
        sel = (lab == labv) & (yc > Y_CUT + 10)
        P = V[sel]
        d, _ = tree.query(P)
        sg = np.sign(s * eval_fast(prims, to_canon(P, R, t, s)))
        sd = d * sg
        out[LABEL_NAMES[labv]] = {"points": int(sel.sum()), "abs_median_mm": float(np.median(d)),
                                  "abs_p90_mm": float(np.percentile(d, 90)),
                                  "signed_median_mm": float(np.median(sd)),
                                  "note": "+ = scan point outside the model skin"}
    return out



# ============================================================ determinedness, export
def final_label():
    use_scale = os.path.exists(os.path.join(STATE, "use_scale.flag"))
    return "s3_scale" if use_scale and load_state("s3_scale") is not None else "s2_after"


def final_theta():
    return load_state(final_label())


def cmd_info(a):
    """How well each parameter is determined, at the exported theta, on the views the fit used.

    Gauss-Newton information J^T J of the full robust objective (soft-L1 IRLS weights; silhouettes,
    scan 3D, landmarks, priors and the finger-gap rows) over all free parameters.  The posterior
    covariance is its inverse, so correlations between parameters are accounted for.  For each
    shape/pose parameter: variance_reduction = 1 - posterior variance / prior variance (prior rows
    Gaussian, as in the fit since gauss_under_soft_l1).  >= 0.7 observed, 0.3-0.7 partly
    observed, < 0.3 prior-driven; a parameter on its bound is 'at limit' (Gauss-Newton says nothing
    there).  Pixels are correlated, so the data information is optimistic; 'vr_sil_info_quartered'
    repeats the computation with the silhouette information divided by 4 as a sensitivity check."""
    views = select_views(load_views(), a.views)
    scan = load_scan()
    th = final_theta()
    use_scale = abs(th[PIDX["log_scale"]]) > 1e-12
    free = free_idx(["rigid", "pose", "shape"] + (["scale"] if use_scale else []))
    pb = Problem(th, views, scan, free, np.random.default_rng(900), scan3d=a.scan3d)
    global _PB, GAUSS_WRAP
    _PB = pb
    GAUSS_WRAP = False          # prior / gap rows as plain Gaussian residuals (set before the fork)
    x = th[free]
    parts = pb.residuals(x, parts=True)
    keys = [k for k in ("sil", "scan", "lm", "prior", "gap") if k in parts]     # = residuals() order
    sizes = [len(parts[k]) for k in keys]
    f0 = np.concatenate([parts[k] for k in keys])
    h = 0.05 * XS[free]
    xx = []
    for i in range(len(x)):
        y = x.copy()
        y[i] += h[i] if x[i] + h[i] <= HI[free][i] else -h[i]
        xx.append(y)
    with mp.get_context("fork").Pool(a.workers) as pool:
        F = pool.map(_res_worker, xx, chunksize=4)
    assert all(len(f) == len(f0) for f in F), "residual length mismatch"
    Jm = np.stack([(f - f0) / (y[i] - x[i]) for i, (y, f) in enumerate(zip(xx, F))], 1)
    wts = 1.0 / np.sqrt(np.sqrt(1 + f0 ** 2))          # soft-L1 IRLS weights (sqrt), data rows only
    nd = sum(len(parts[k]) for k in keys if k in ("sil", "scan", "lm"))
    wts[nd:] = 1.0
    Jw = Jm * wts[:, None]
    edges = np.cumsum([0] + sizes)
    blocks = {k: Jw[edges[i]:edges[i + 1]] for i, k in enumerate(keys)}
    Ip = (blocks["prior"] ** 2).sum(0)
    per = {k: (blocks[k] ** 2).sum(0) for k in keys if k != "prior"}
    Idata = sum(per.values())

    def post_var(sil_factor):
        JTJ = sum(blocks[k].T @ blocks[k] * (sil_factor if k == "sil" else 1.0) for k in keys)
        return np.clip(np.diag(np.linalg.inv(JTJ + 1e-9 * np.eye(len(x)))), 0, None)
    pv, pv4 = post_var(1.0), post_var(0.25)
    info = {}
    for j, i in enumerate(free):
        n = PNAMES[i]
        at = "lo" if x[j] <= LO[i] + 1e-6 else "hi" if x[j] >= HI[i] - 1e-6 else None
        e = {"posterior_sigma": float(np.sqrt(pv[j])),
             "data_share_diag": float(Idata[j] / max(Idata[j] + Ip[j], 1e-12)),
             "from_silhouettes": float(per.get("sil", np.zeros(len(x)))[j] / max(Idata[j], 1e-12)),
             "from_scan_3d": float(per.get("scan", np.zeros(len(x)))[j] / max(Idata[j], 1e-12)),
             "from_landmarks": float(per.get("lm", np.zeros(len(x)))[j] / max(Idata[j], 1e-12)),
             "from_finger_gap": float(per.get("gap", np.zeros(len(x)))[j] / max(Idata[j], 1e-12)),
             "at_bound": at}
        if Ip[j] > 0:
            e["variance_reduction"] = float(1 - pv[j] * Ip[j])
            e["vr_sil_info_quartered"] = float(1 - pv4[j] * Ip[j])
            vr = e["variance_reduction"]
            e["determined_by"] = ("at limit" if at else "observed" if vr >= 0.7 else
                                  "partly observed" if vr >= 0.3 else "prior-driven")
        info[n] = e
    jdump({"theta": th, "views": [int(v["idx"]) for v in views], "info": info,
           "residual_rows": dict(zip(keys, sizes))}, os.path.join(STATE, "info.json"))
    cnt = {}
    for e in info.values():
        if "determined_by" in e:
            cnt[e["determined_by"]] = cnt.get(e["determined_by"], 0) + 1
    log("info written", os.path.join(STATE, "info.json"), cnt)


def make_mesh_fast(theta, res=0.5, target=60000):
    """Final skin: canonical-frame marching cubes of the fitted SDF, Taubin, Blender decimate (as standin)."""
    import trimesh
    m = mesh_of(theta, res=res, world=False, smooth=True)
    V2, F2 = S.decimate_bpy(np.asarray(m.vertices), np.asarray(m.faces), target)
    mesh = trimesh.Trimesh(V2, F2, process=False)
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def cmd_export(a):
    import trimesh
    os.makedirs(OUT, exist_ok=True)
    th = final_theta()
    t0 = time.time()
    mesh = make_mesh_fast(th, 0.5)
    R, t, s = placement(th)
    mesh = trimesh.Trimesh(np.asarray(mesh.vertices) * s, mesh.faces, process=False)   # global scale baked in
    log(f"mesh {len(mesh.vertices)} verts, watertight {mesh.is_watertight}, {time.time() - t0:.0f}s")
    mesh.export(os.path.join(OUT, "hand_fitted_open.ply"))
    Vw = to_world(np.asarray(mesh.vertices), R, t, 1.0)
    trimesh.Trimesh(Vw, mesh.faces, process=False).export(os.path.join(OUT, "hand_fitted_in_scan_frame_as_right.ply"))
    p = pdict(th)
    J, tb = build_joints_p(p)
    J = {n: v * s for n, v in J.items()}
    across = unit(J["index_mcp"] - J["pinky_mcp"])
    along = unit(J["middle_mcp"] - J["wrist"])
    pn = unit(np.cross(across, along))
    jj = {"side": "right", "units": "mm", "pose": "open",
          "landmarks": {n: [round(float(c), 3) for c in J[n]] for n in NAMES},
          "palm_normal": [round(float(c), 5) for c in pn],
          "lengths": S.lengths_of(J),
          "angles_deg": S.joint_angles(J),
          "source": {"generator": "fit_standin_to_scan.py", "rig": S.RIG_BLEND,
                     "frame": "stand-in canonical (wrist at origin, fingers +Y, palm +Z, thumb +X)",
                     "open_pose": "the scanned hand's open pose (frames 0-9.2 s), fitted",
                     "global_scale": s,
                     "placement_to_scan_as_right": {"R": R, "t_mm": t,
                                                    "how": "X_as_right = R @ x + t (x = these landmarks / this mesh, scale already applied); board (left hand) = diag(-1,1,1) @ X_as_right"}}}
    jdump(jj, os.path.join(OUT, "joints.json"))
    info = json.load(open(os.path.join(STATE, "info.json")))["info"] if os.path.exists(os.path.join(STATE, "info.json")) else {}
    params = {}
    for i, (n, d, sg, lo, hi, g, meaning) in enumerate(PT):
        if g == "scale" and abs(th[i]) < 1e-12:
            continue
        e = {"value": round(float(th[i]), 4), "prior": d, "prior_sigma": sg if g != "rigid" else None,
             "group": g, "meaning": meaning}
        if n in info:
            ii = info[n]
            e["posterior_sigma"] = round(ii["posterior_sigma"], 4)
            for k in ("variance_reduction", "vr_sil_info_quartered"):
                if k in ii:
                    e[k] = round(ii[k], 3)
            if "determined_by" in ii:
                e["determined_by"] = ii["determined_by"]
            e["main_data"] = max(("silhouettes", ii["from_silhouettes"]), ("scan 3D", ii["from_scan_3d"]),
                                 ("landmarks", ii["from_landmarks"]), ("finger gap", ii.get("from_finger_gap", 0)),
                                 key=lambda kv: kv[1])[0]
        params[n] = e
    # measured skin sections (same definition as standin mesh_stats section_mm) for the fitted and default hands
    sec_fit = {k: (None if v is None else round(v * s, 1)) for k, v in sections(th).items()}
    th0 = DEFAULT.copy()
    th0[GROUP == "pose"] = th[GROUP == "pose"]
    sec_def = sections(th0)
    # the same sections at each optimisation round of the final stage (different ray samples): spread
    sec_rounds = {}
    for r in range(4):
        thr = load_state(f"{final_label()}_r{r}")
        if thr is not None:
            sr = float(np.exp(thr[PIDX["log_scale"]]))
            sec_rounds[f"round_{r}"] = {k: (None if v is None else round(v * sr, 1)) for k, v in sections(thr).items()}
    jdump({"parameters": params,
           "sections_mm": {"fitted": sec_fit, "standin_defaults_same_pose": sec_def,
                           "fitted_per_round_of_final_stage": sec_rounds},
           "global_scale": {"value": s, "check": json.load(open(os.path.join(STATE, "scale_check.json")))
                            if os.path.exists(os.path.join(STATE, "scale_check.json")) else None},
           "views_used": json.load(open(os.path.join(STATE, "info.json"))).get("views")
           if os.path.exists(os.path.join(STATE, "info.json")) else None,
           "notes": {"determined_by": "variance_reduction = 1 - posterior / prior variance from the full Gauss-Newton "
                                      "covariance (all parameters jointly, soft-L1 weights) at the exported theta; "
                                      "observed >= 0.7, partly 0.3-0.7, prior-driven < 0.3, 'at limit' = on its bound",
                     "vr_sil_info_quartered": "the same with the silhouette information divided by 4 (pixels are "
                                              "correlated, so the full value is optimistic)",
                     "posterior_sigma": "from the same covariance; optimistic (lower bound)"}},
          os.path.join(OUT, "fitted_params.json"))
    log("exported", OUT)


def sections(theta):
    """Skin widths / thicknesses across the PIP (thumb IP) and palm, measured on the SDF."""
    p = pdict(theta)
    J, tb, prims = model(theta)
    Zv = np.array([0.0, 0, 1])

    def span(q, d, L=60.0):
        tt = np.linspace(-L, L, 4801)
        f = eval_fast(prims, q[None] + tt[:, None] * d[None])
        c = len(tt) // 2
        if f[c] >= 0:
            return None
        o_r = np.where(f[c:] >= 0)[0]
        o_l = np.where(f[:c][::-1] >= 0)[0]
        if not len(o_r) or not len(o_l):
            return None
        return round(float(tt[c + o_r[0]] - tt[c - 1 - o_l[0]]), 1)
    out = {"breadth_mcp_row": span(J["middle_mcp"] - [0, 8, 0], np.array([1.0, 0, 0]), 80),
           "thick_mid_palm": span(np.array([-3, 0.5 * J["middle_mcp"][1], 0]), Zv)}
    for f in ["thumb"] + FINGERS4:
        segs = [("pip", "ip"), ] if f == "thumb" else [("mcp", None), ("pip", None), ("dip", None)]
        for jn, alt in ([("mcp", None), ("ip", None)] if f == "thumb" else segs):
            if f == "thumb":
                q = J[f"thumb_{jn}"]
                nxt = J["thumb_ip"] if jn == "mcp" else J["thumb_tip"]
                q = q + 0.35 * (nxt - q) if jn == "mcp" else q
            else:
                q = J[f"{f}_{jn}"]
                nxt = {"mcp": "pip", "pip": "dip", "dip": "tip"}[jn]
                if jn == "mcp":
                    q = q + 0.5 * (J[f"{f}_pip"] - q)       # mid proximal phalanx (MCP itself is in the palm)
                nxt = J[f"{f}_{nxt}"]
            d = unit(nxt - q) if np.linalg.norm(nxt - q) > 1 else unit(J[f"{f}_tip"] - J[f"{f}_mcp"])
            side = unit(np.cross(d, tb if f == "thumb" else Zv))
            lab = {"mcp": "proximal_mid", "pip": "pip", "dip": "dip", "ip": "ip"}[jn] if f != "thumb" else \
                {"mcp": "proximal_mid", "ip": "ip"}[jn]
            out[f"width_{f}_{lab}"] = span(q, side, 30)
            out[f"thick_{f}_{lab}"] = span(q, unit(np.cross(side, d)), 30)
    return out



def cmd_metrics(a):
    """Per-frame silhouette IoU / contour distance (all 70 frames of 0-9.2 s), plain stand-in
    (pose + placement fitted, default shape) vs fitted, plus the raw scan as reference, summarised
    separately for the clean-mask frames (the ones the fit used) and the gated-out frames; 3D distance."""
    views = load_views()
    scan = load_scan()
    q = mask_quality(views)
    th_b, th_a = load_state("s1_before"), final_theta()
    meshes = {"before_plain_standin": mesh_of(th_b, 1.0), "after_fitted": mesh_of(th_a, 1.0),
              "raw_scan_reference": scan[0]}
    t0 = time.time()
    rows = silhouette_metrics(meshes, views, th_a)
    for k in rows:
        for r in rows[k]:
            r["clean_mask"] = q[str(r["frame"])]["clean"]
    log(f"silhouettes {time.time() - t0:.0f}s")
    summ = {}
    for sub, sel in (("all_70", lambda r: True), ("clean_used_by_fit", lambda r: r["clean_mask"]),
                     ("gated_out_bad_masks", lambda r: not r["clean_mask"])):
        summ[sub] = {k: summarize([r for r in v if sel(r)]) for k, v in rows.items()}
    res = {"frames": "all views of 0-9.2 s (one hand posture), 70 frames; the fit used the clean-mask frames "
                     f"(mask gate: spill < {GATE_SPILL_MM} mm and miss < {GATE_MISS_MM} mm, see STATE/mask_quality.json)",
           "roi": "known pixels, hand side of the wrist cut (8 mm margin), within 10 mm of any of the three meshes",
           "summary": summ,
           "per_frame": rows,
           "scan_3d_distance": {"before_plain_standin": scan_distance(th_b, scan, mesh_of(th_b, 0.5, smooth=False)),
                                "after_fitted": scan_distance(th_a, scan, mesh_of(th_a, 0.5, smooth=False))}}
    res["held_out_validation"] = {}
    for pre in ("val", "val2"):
        val = os.path.join(STATE, f"validation_{pre}.json")
        if os.path.exists(val):
            vj = json.load(open(val))
            res["held_out_validation"][pre] = {"train_frames": vj["train_frames"], "test_frames": vj["test_frames"],
                                               **vj["summary"]}
    os.makedirs(OUT, exist_ok=True)
    jdump(res, os.path.join(OUT, "fit_metrics.json"))
    log(json.dumps(res["summary"], indent=1))
    log(json.dumps(res["scan_3d_distance"], indent=1))


def cmd_validate(a):
    """Held-out check on a separate camera sector: the fit chain run on one block of clean frames
    only (states <prefix>_s1 = plain shape, <prefix>_s2 = everything, optional <prefix>_s3 = + scale),
    scored on another block.  val: train clean frames before 2 s, test clean frames after 3 s
    (cameras 17-62 deg from every training camera); val2: the reverse."""
    views = load_views()
    test = select_views(views, a.test)
    train = select_views(views, a.train)
    pre = a.prefix
    th_b, th_a = load_state(f"{pre}_s1"), load_state(f"{pre}_s2")
    meshes = {"before_plain_standin": mesh_of(th_b, 1.0), "after_fitted": mesh_of(th_a, 1.0)}
    if load_state(f"{pre}_s3") is not None:          # + free global scale (stage 3 on the same frames)
        meshes["after_fitted_free_scale"] = mesh_of(load_state(f"{pre}_s3"), 1.0)
    rows = silhouette_metrics(meshes, test, th_a)
    R, t, s = placement(th_a)
    dirn = lambda v: unit(t - v["C"])
    DT = np.array([dirn(v) for v in train])
    ang = {int(v["idx"]): float(np.degrees(np.arccos(np.clip(DT @ dirn(v), -1, 1))).min()) for v in test}
    for k in rows:
        for r in rows[k]:
            r["deg_to_nearest_train_camera"] = round(ang[r["frame"]], 1)
    b, af = rows["before_plain_standin"], rows["after_fitted"]
    better = sum(1 for x, y in zip(b, af) if y["iou"] > x["iou"] and y["contour_mm"] < x["contour_mm"])
    out = {"what": cmd_validate.__doc__.strip(), "train_frames": [int(v["idx"]) for v in train],
           "test_frames": [int(v["idx"]) for v in test],
           "summary": {**{k: summarize(v) for k, v in rows.items()},
                       "s3_scale": (float(np.exp(load_state(f"{pre}_s3")[PIDX["log_scale"]]))
                                        if load_state(f"{pre}_s3") is not None else None),
                       "after_better_on_both": f"{better}/{len(af)}",
                       "test_deg_to_nearest_train_camera": [min(ang.values()), max(ang.values())]},
           "per_frame": rows}
    jdump(out, os.path.join(STATE, f"validation_{pre}.json"))
    log(json.dumps(out["summary"], indent=1))


def cmd_sheet(a):
    """fitted_sheet.jpg for Olex.  Row 1: the best-matching and a typical clean video frame, each
    plain and with the fitted hand's outline.  Row 2: old cylinder hand vs new fitted hand on the
    whole controller (grip_views.py: see-through plastic and skin, skin inside the plastic red),
    the deepest spot ringed red and the palm-swell point ringed blue."""
    import subprocess
    from PIL import Image, ImageDraw, ImageFont
    FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    FONT_B = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    views = load_views()
    th = final_theta()
    mesh = mesh_of(th, 1.0)
    fm = json.load(open(os.path.join(OUT, "fit_metrics.json")))
    rows = [r for r in fm["per_frame"]["after_fitted"] if r.get("clean_mask")]
    rows.sort(key=lambda r: -r["iou"])
    other = [r for r in rows if r["ts_ms"] >= 3000]          # a camera on the other side of the hand
    picks = [(rows[0]["frame"], "best match"), (other[len(other) // 2]["frame"], "typical match")]
    if a.frames:
        picks = [(int(x), "") for x in a.frames.split(",")]
    P = 460
    panels1 = []
    for fi, what in picks:
        v = [w for w in views if w["idx"] == fi][0]
        img = cv2.imread(os.path.join(WORK, "imgs", f"f{fi:03d}.jpg"))
        r, _ = render_mask(mesh, v)
        ov = img.copy()
        ov[r] = (0.55 * ov[r] + 0.45 * np.array([60, 200, 255])).astype(np.uint8)
        cs, _ = cv2.findContours(r.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cs, -1, (0, 255, 255), max(3, img.shape[0] // 160), cv2.LINE_AA)
        for im, title in ((img, f"video frame {fi}"), (ov, f"fitted hand in yellow ({what})")):
            pil = Image.fromarray(cv2.cvtColor(cv2.resize(im, (P, P), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            panels1.append((pil, title))
    gv = os.path.join(a.run_dir, "grip_views")
    if not os.path.exists(os.path.join(gv, "grip_views.json")) or a.rerender:
        subprocess.run([sys.executable, os.path.join(HERE, "grip_views.py"), os.path.join(OUT, "hand_on_grip_fitted.blend"),
                        gv, "--samples", "24"], check=True, stdout=subprocess.DEVNULL)
    meta = json.load(open(os.path.join(gv, "grip_views.json")))
    pad_ = 14
    con = json.load(open(os.path.join(OUT, "contacts.json")))
    P2 = (4 * P + 5 * pad_ - 3 * pad_) // 2
    panels2 = []
    for vi in (0, 1):
        for row, title in (("old", "OLD: cylinder hand"), ("new", "NEW: your fitted hand")):
            im = Image.open(os.path.join(gv, f"{row}_{vi}.png")).convert("RGB")
            m = meta[f"{row}_{vi}"]
            d = ImageDraw.Draw(im)
            rr = 46
            for key, col in (("deepest_px", (220, 0, 0)), ("palm_swell_px", (0, 70, 230))):
                if m.get(key):
                    x, y = m[key]
                    d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=col, width=7)
            sc_ = P2 / im.size[0]
            panels2.append((im.resize((P2, int(im.size[1] * sc_)), Image.LANCZOS), f"{title}, {m['view']}"))
    f_t = ImageFont.truetype(FONT_B, 44)
    f_h = ImageFont.truetype(FONT_B, 30)
    f_c = ImageFont.truetype(FONT, 30)
    pad, titleh, headh = 14, 74, 44
    W = 4 * P + 5 * pad
    h2 = max(p.size[1] for p, _ in panels2)
    H = titleh + headh + P + 2 * pad + 2 * (headh + h2 + pad) + 150
    sh = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(sh)
    dr.text((pad, 14), "Your real hand: smooth hand fitted to your scan video", fill=(15, 15, 15), font=f_t)
    y = titleh
    for i, (im, title) in enumerate(panels1):
        x = pad + i * (P + pad)
        ft = f_h if dr.textlength(title, font=f_h) <= P else ImageFont.truetype(FONT_B, 28)
        dr.text((x, y + 4), title, fill=(25, 25, 25), font=ft)
        sh.paste(im, (x, y + headh))
    y += headh + P + 2 * pad
    for i, (im, title) in enumerate(panels2):
        x = pad + (i % 2) * (P2 + pad)
        yy = y + (i // 2) * (headh + h2 + pad)
        dr.text((x, yy + 4), title, fill=(25, 25, 25), font=f_h)
        sh.paste(im, (x, yy + headh))
    o, n = con["old_cylinder_hand"], con["new_scan_hand"]
    cap = [f"Top: yellow outline = fitted hand on your video.  Bottom: red = skin inside the plastic; red ring = deepest spot "
           f"(old {o['penetration_max_mm']:.1f} mm, new {n['penetration_max_mm']:.1f} mm).",
           f"Blue ring = palm swell: gap to the skin old {o['palm_swell_gap_mm']:.1f} mm, new {n['palm_swell_gap_mm']:.1f} mm.  "
           f"Same hold pose for both hands."]
    for i, c in enumerate(cap):
        dr.text((pad, H - 130 + i * 46), c, fill=(40, 40, 40), font=f_c)
    out = os.path.join(OUT, "fitted_sheet.jpg")
    sh.save(out, quality=90)
    log("WROTE", out, sh.size, picks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit")
    f.add_argument("--stage", default="0", choices=["0", "1", "2", "3"])
    f.add_argument("--rounds", type=int, default=2)
    f.add_argument("--nfev", type=int, default=15)
    f.add_argument("--view-step", type=int, default=1)
    f.add_argument("--workers", type=int, default=4)
    f.add_argument("--resume", action="store_true")
    f.add_argument("--seed-off", type=int, default=0)
    f.add_argument("--views", default="clean", choices=["all", "clean", "clean_early", "clean_late"])
    f.add_argument("--scan3d", default="all", choices=list(SCAN3D_MODES))
    f.add_argument("--label", default=None)
    f.add_argument("--from", default=None)
    f.add_argument("--scale-bound", type=float, default=None)
    for name in ("export", "metrics"):
        sp.add_parser(name)
    va = sp.add_parser("validate")
    va.add_argument("--train", default="clean_early")
    va.add_argument("--test", default="clean_late")
    va.add_argument("--prefix", default="val")
    fi = sp.add_parser("info")
    fi.add_argument("--views", default="clean", choices=["all", "clean"])
    fi.add_argument("--scan3d", default="all", choices=list(SCAN3D_MODES))
    fi.add_argument("--workers", type=int, default=4)
    sh = sp.add_parser("sheet")
    sh.add_argument("--frames", default="")
    sh.add_argument("--run-dir", default=os.path.join(OUT, "run"))
    sh.add_argument("--rerender", action="store_true")
    a = ap.parse_args()
    {"fit": cmd_fit}.get(a.cmd, lambda a: globals()["cmd_" + a.cmd](a))(a)


if __name__ == "__main__":
    main()
