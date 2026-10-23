"""Module A: stand-in hand. Builds an OPEN-pose right-hand skin mesh from Olex's
measurements, standing in for the photogrammetry scan until it exists.

    /tmp/claude-0/hs/venv/bin/python standin.py                 # base hand -> standin/
    /tmp/claude-0/hs/venv/bin/python standin.py --variant 3     # perturbed -> standin/variant_3/
    options: --no-render, --res 0.5 (mm), --target-verts 60000, --out DIR

Frame (SPEC): 1 unit = 1 mm, right hand, wrist joint centre at the origin,
fingers point +Y, palm faces +Z (palm up), thumb side = +X.

Joint centres: palm layout (MCP heads, thumb CMC, metacarpal lengths) and all
bone lengths come from the rest pose of 'hand rig' in hold_rig.blend, expressed
in a palm frame (y = wrist -> middle MCP, x = pinky MCP -> index MCP orthogonalised,
z = x cross y = palmar). The palm arch is flattened (normal offsets x0.3), the
fingers are straightened to a natural open pose (MCP 8, PIP 7, DIP 5 deg flexion),
spread 8 deg between neighbouring proximal segments, thumb metacarpal 40 deg from
the index metacarpal (relaxed out to the side, 15 deg palmar elevation of the
abduction direction).

Angle convention (as gripscan/relaxed_pose.py): flexion 0 = straight, positive =
towards the palm (+Z here). In hold_rig.blend the palm faces -X (world) and
positive flexion curls toward it; the bones' local X axes are not consistently
signed (e.g. index_proximal x = (0.2,-0.12,-0.97), index_distal x = (-0.19,-0.4,0.89)),
so bone-local rotation signs must be checked per bone by the rig module; this
module only produces joint centres + skin.

Skin: smooth signed distance field (polynomial smooth-min union of elliptical
tapered capsules for finger segments, rounded extruded palm slab, ellipsoids for
thenar / hypothenar / MCP pads / knuckles / fingertip pulps, a thin web slab
between thumb and index, palmar webs between fingers, a smooth-subtracted
central palmar hollow, a forearm stub cut flat 40 mm proximal of the wrist).
Marching cubes (skimage) at 0.5 mm, Taubin smoothing, quadric-ish decimation
with Blender's Decimate (collapse) to ~60k vertices.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RIG_BLEND = "/tmp/claude-0/hs/hold_rig.blend"
OUT_ROOT = "/mnt/project-files/hand_scan_rig/standin"
WORK = "/tmp/claude-0/hs/work/standin"

NAMES = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_mcp", "index_pip",
         "index_dip", "index_tip", "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
         "ring_mcp", "ring_pip", "ring_dip", "ring_tip", "pinky_mcp", "pinky_pip", "pinky_dip",
         "pinky_tip"]
FINGERS4 = ["index", "middle", "ring", "pinky"]
SEGS = {"thumb": ["metacarpal", "proximal", "distal"]}
SEGS4 = ["proximal", "middle", "distal"]

# ---------------------------------------------------------------- parameters
OPEN_POSE = {
    "mcp_flex": 8.0, "pip_flex": 7.0, "dip_flex": 5.0,
    # absolute in-palm-plane direction of each proximal segment, deg from +Y toward +X
    "fan": {"index": 12.0, "middle": 4.0, "ring": -4.0, "pinky": -12.0},
    "thumb_cmc": 40.0,          # angle thumb metacarpal vs index metacarpal
    "thumb_elev": 15.0,         # palmar elevation of the abduction direction
    "thumb_mcp_flex": 10.0, "thumb_ip_flex": 10.0,
    "arch_keep": 0.3,           # fraction of the rig's palm-arch (normal offsets) kept
}
WIDTH_PIP = {"index": 19.5, "middle": 19.5, "ring": 18.0, "pinky": 16.0, "thumb": 21.5}
THICK_RATIO = 0.85
TIP_INSET = 2.5                 # tip landmark lies this far inside the skin tip
# primitive radius factor so that the measured skin width AT the PIP/IP joint (incl. the
# joint bulge and smooth-union swell) equals WIDTH_PIP (calibrated: 20.5 -> 19.5 mm)
RADIUS_FIT = 0.952


def unit(v):
    return v / np.linalg.norm(v)


def rot(v, axis, deg):
    """Rodrigues rotation of v about unit axis."""
    a = unit(axis)
    t = np.radians(deg)
    return v * np.cos(t) + np.cross(a, v) * np.sin(t) + a * (a @ v) * (1 - np.cos(t))


# ---------------------------------------------------------------- rig
def read_rig(path=RIG_BLEND):
    """Bone heads/tails (world mm) of 'hand rig' rest pose. Cached as json."""
    cache = os.path.join(WORK, "rig_rest.json")
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
        return json.load(open(cache))
    import bpy
    bpy.ops.wm.open_mainfile(filepath=path)
    arm = bpy.data.objects["hand rig"]
    mw = arm.matrix_world
    out = {b.name: {"head": list(mw @ b.head_local), "tail": list(mw @ b.tail_local),
                    "length": b.length} for b in arm.data.bones}
    os.makedirs(WORK, exist_ok=True)
    json.dump(out, open(cache, "w"), indent=1)
    return out


def palm_layout(R):
    """MCP heads and thumb CMC in the palm frame, plus bone lengths."""
    W = np.array(R["palm"]["head"])
    mcp = {f: np.array(R[f + "_metacarpal"]["tail"]) for f in FINGERS4}
    y = unit(mcp["middle"] - W)
    a = mcp["index"] - mcp["pinky"]
    x = unit(a - (a @ y) * y)
    z = np.cross(x, y)
    B = np.stack([x, y, z])
    lay = {"mcp": {f: B @ (mcp[f] - W) for f in FINGERS4},
           "cmc": B @ (np.array(R["thumb_metacarpal"]["head"]) - W),
           "len": {}}
    for f in FINGERS4:
        lay["len"][f] = {s: R[f"{f}_{s}"]["length"] for s in SEGS4}
    lay["len"]["thumb"] = {s: R[f"thumb_{s}"]["length"] for s in SEGS["thumb"]}
    return lay


# ---------------------------------------------------------------- joints
def build_joints(lay, pose=OPEN_POSE, len_scale=None):
    """Open-pose joint centres. len_scale: {finger:{segment:f}} incl. 'metacarpal'."""
    ls = len_scale or {}
    Z = np.array([0.0, 0.0, 1.0])
    J = {"wrist": np.zeros(3)}
    dirs = {}
    for f in FINGERS4:
        m = lay["mcp"][f].copy()
        m[2] *= pose["arch_keep"]
        m *= ls.get(f, {}).get("metacarpal", 1.0)
        J[f"{f}_mcp"] = m
        t = np.radians(pose["fan"][f])
        d0 = np.array([np.sin(t), np.cos(t), 0.0])
        ax = np.cross(d0, Z)            # rotating d0 about ax by +theta moves it toward +Z
        p = m
        flex = 0.0
        for s, jn, fl in zip(SEGS4, ["pip", "dip", "tip"],
                             [pose["mcp_flex"], pose["pip_flex"], pose["dip_flex"]]):
            flex += fl
            d = rot(d0, ax, flex)
            L = lay["len"][f][s] * ls.get(f, {}).get(s, 1.0)
            p = p + L * d
            J[f"{f}_{jn}"] = p
            dirs[(f, s)] = d
    # thumb
    cmc = lay["cmc"].copy()
    J["thumb_cmc"] = cmc
    u_idx = unit(J["index_mcp"])
    perp = unit(np.cross(u_idx, Z))     # in-plane, pointing to +X side of index meta
    if perp[0] < 0:
        perp = -perp
    e = np.radians(pose["thumb_elev"])
    c = np.radians(pose["thumb_cmc"])
    dm = unit(np.cos(c) * u_idx + np.sin(c) * (np.cos(e) * perp + np.sin(e) * Z))
    # thumb bends toward the palm midline and the palm side
    bend = -perp * 0.7 + Z * 0.7
    bend = unit(bend - (bend @ dm) * dm)
    tax = np.cross(dm, bend)            # rotating dm about tax moves it toward `bend`
    lt = lay["len"]["thumb"]
    tl = ls.get("thumb", {})
    p = cmc + lt["metacarpal"] * tl.get("metacarpal", 1.0) * dm
    J["thumb_mcp"] = p
    d1 = rot(dm, tax, pose["thumb_mcp_flex"])
    p = p + lt["proximal"] * tl.get("proximal", 1.0) * d1
    J["thumb_ip"] = p
    d2 = rot(dm, tax, pose["thumb_mcp_flex"] + pose["thumb_ip_flex"])
    J["thumb_tip"] = p + lt["distal"] * tl.get("distal", 1.0) * d2
    return {n: np.asarray(J[n], float) for n in NAMES}, unit(bend)


def joint_angles(J):
    """Angles with gripscan.relaxed_pose.angles (same convention); sign made curl-positive."""
    sys.path.insert(0, "/tmp/claude-0/hs/pg")
    from gripscan.relaxed_pose import angles
    W = np.array([J[n] for n in NAMES])
    a = angles(W)
    pip = np.median([v for k, v in a.items() if k.endswith("_pip")])
    if pip < 0:
        a = {k: (v if (k.startswith("spread") or k == "thumb_cmc") else -v) for k, v in a.items()}
    return {k: round(v, 2) for k, v in a.items()}


def lengths_of(J):
    out = {}
    for f in ["thumb"] + FINGERS4:
        chain = ([f"thumb_{n}" for n in ["cmc", "mcp", "ip", "tip"]] if f == "thumb"
                 else ["wrist"] + [f"{f}_{n}" for n in ["mcp", "pip", "dip", "tip"]])
        segs = SEGS["thumb"] if f == "thumb" else ["metacarpal"] + SEGS4
        out[f] = {s: round(float(np.linalg.norm(J[chain[i + 1]] - J[chain[i]])), 2)
                  for i, s in enumerate(segs)}
    return out


# ---------------------------------------------------------------- SDF primitives
def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(k - np.abs(a - b), 0, None) / k
    return np.minimum(a, b) - h * h * k * 0.25


def smax(a, b, k):
    return -smin(-a, -b, k)


def frame_from(w, v_hint):
    w = unit(w)
    v = unit(v_hint - (v_hint @ w) * w)
    u = np.cross(v, w)
    return np.stack([u, v, w])          # rows u, v, w


class Prim:
    def __init__(self, fn, lo, hi, k, op="union", name=""):
        self.fn, self.lo, self.hi, self.k, self.op, self.name = fn, np.asarray(lo), np.asarray(hi), k, op, name


def round_cone(a, b, ra, rb, v_hint, squash=THICK_RATIO, k=3.0, name=""):
    """Tapered capsule a->b with elliptical cross-section (u: r, v: squash*r)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    F = frame_from(b - a, v_hint)
    h = np.linalg.norm(b - a)
    bb = (ra - rb) / h
    aa = np.sqrt(max(1 - bb * bb, 1e-6))
    sc = 0.5 * (1 + squash)

    def fn(P):
        q = (P - a) @ F.T
        rho = np.sqrt(q[:, 0] ** 2 + (q[:, 1] / squash) ** 2)
        y = q[:, 2]
        kk = -bb * rho + aa * y
        d = np.where(kk < 0, np.sqrt(rho ** 2 + y ** 2) - ra,
                     np.where(kk > aa * h, np.sqrt(rho ** 2 + (y - h) ** 2) - rb,
                              aa * rho + bb * y - ra))
        return d * sc
    r = max(ra, rb)
    return Prim(fn, np.minimum(a, b) - r, np.maximum(a, b) + r, k, name=name)


def ellipsoid(c, F, semi, k=6.0, op="union", name=""):
    c = np.asarray(c, float)
    s = np.asarray(semi, float)

    def fn(P):
        q = (P - c) @ F.T
        k0 = np.linalg.norm(q / s, axis=1)
        k1 = np.linalg.norm(q / (s * s), axis=1)
        return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)
    r = s.max()
    return Prim(fn, c - r, c + r, k, op, name)


def tri_slab(A, B, C, half, k=6.0, name=""):
    A, B, C = (np.asarray(x, float) for x in (A, B, C))
    ba, cb, ac = B - A, C - B, A - C
    nor = np.cross(ba, ac)

    def dot2(v):
        return np.einsum("ij,ij->i", v, v)

    def fn(P):
        pa, pb, pc = P - A, P - B, P - C
        s = (np.sign(np.cross(ba, nor) @ pa.T) + np.sign(np.cross(cb, nor) @ pb.T)
             + np.sign(np.cross(ac, nor) @ pc.T))
        e1 = ba[None] * np.clip((pa @ ba) / (ba @ ba), 0, 1)[:, None] - pa
        e2 = cb[None] * np.clip((pb @ cb) / (cb @ cb), 0, 1)[:, None] - pb
        e3 = ac[None] * np.clip((pc @ ac) / (ac @ ac), 0, 1)[:, None] - pc
        edge = np.minimum(np.minimum(dot2(e1), dot2(e2)), dot2(e3))
        plane = (pa @ nor) ** 2 / (nor @ nor)
        return np.sqrt(np.where(s < 2, edge, plane)) - half
    lo = np.minimum(np.minimum(A, B), C) - half
    hi = np.maximum(np.maximum(A, B), C) + half
    return Prim(fn, lo, hi, k, name=name)


def poly2d(xy, V):
    """Signed distance to a closed 2D polygon (iq sdPolygon), negative inside."""
    n = len(V)
    d = np.sum((xy - V[0]) ** 2, axis=1)
    s = np.ones(len(xy))
    j = n - 1
    for i in range(n):
        e = V[j] - V[i]
        w = xy - V[i]
        b = w - e[None] * np.clip((w @ e) / (e @ e), 0, 1)[:, None]
        d = np.minimum(d, np.sum(b * b, axis=1))
        c1 = xy[:, 1] >= V[i][1]
        c2 = xy[:, 1] < V[j][1]
        c3 = e[0] * w[:, 1] > e[1] * w[:, 0]
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        s = np.where(flip, -s, s)
        j = i
    return s * np.sqrt(d)


def group(prims, k, name=""):
    """Sub-union evaluated on its own, then blended into the hand with k."""
    def fn(P):
        d = np.full(len(P), 1e3)
        for pr in prims:
            d = smin(d, pr.fn(P), pr.k)
        return d
    lo = np.min([p.lo for p in prims], 0)
    hi = np.max([p.hi for p in prims], 0)
    return Prim(fn, lo, hi, k, name=name)


def palm_slab(V, zb_fn, zt_fn, r, k=8.0, name="palm"):
    V = np.asarray(V, float)

    def fn(P):
        d2 = poly2d(P[:, :2], V) + r
        zb, zt = zb_fn(P), zt_fn(P)
        dz = np.maximum(zb + r - P[:, 2], P[:, 2] - (zt - r))
        w0, w1 = d2, dz
        return (np.minimum(np.maximum(w0, w1), 0)
                + np.sqrt(np.clip(w0, 0, None) ** 2 + np.clip(w1, 0, None) ** 2) - r)
    lo = np.r_[V.min(0), -30]
    hi = np.r_[V.max(0), 30]
    return Prim(fn, lo, hi, k, name=name)


# ---------------------------------------------------------------- hand model
def build_prims(J, tbend, wscale=None, palm_scale=1.0):
    ws = wscale or {}
    Z = np.array([0.0, 0.0, 1.0])
    X = np.array([1.0, 0.0, 0.0])
    P = []
    m = {f: J[f"{f}_mcp"] for f in FINGERS4}
    pt = palm_scale              # thickness / breadth scale of the palm

    # palm slab outline (mid-thickness), from MCP centres
    I, M, R_, K = m["index"], m["middle"], m["ring"], m["pinky"]
    br = 10.0 * pt
    V = [(-21 * pt, -50), (21 * pt, -50), (27 * pt, 6), (I[0] + 11.5 * pt, I[1] - 28), (I[0] + br, I[1] - 4),
         (I[0] + 2, I[1] + 4), (M[0], M[1] + 4), (R_[0], R_[1] + 4), (K[0] - 2, K[1] + 4),
         (K[0] - br, K[1] - 6), (K[0] - 9 * pt, K[1] - 34), (-30 * pt, 6)]
    ymcp = M[1]

    def taper(Pp):  # proximal of the wrist the slab thins out inside the forearm stub
        return 0.3 * 6.0 * np.logaddexp(0, -Pp[:, 1] / 6.0)   # smooth ramp, no crease

    def zb(Pp):   # dorsal surface
        t = np.clip(Pp[:, 1] / ymcp, 0, 1)
        return (-17 + 6 * t) * pt + taper(Pp)

    def zt(Pp):   # palmar surface
        t = np.clip(Pp[:, 1] / ymcp, 0, 1)
        return (17 - 1.5 * t ** 2) * pt - taper(Pp)
    P.append(palm_slab(V, zb, zt, r=12.0 * pt, k=0, name="palm"))

    # forearm stub (elliptical round cone), cut flat at y = -40 later
    P.append(round_cone([0, 12, 0], [0, -60, 0], 27 * pt, 30.5 * pt, Z, squash=0.62, k=16,
                        name="forearm"))

    # hypothenar eminence
    hyp_c = np.array([K[0] + 4.5, 0.42 * K[1], 6.0]) * np.array([pt, 1, pt])
    P.append(ellipsoid(hyp_c, frame_from(np.array([-0.18, 1, 0]), Z)[[0, 1, 2]],
                       [13 * pt, 12 * pt, 32], k=14, name="hypothenar"))

    # thenar eminence (along the thumb metacarpal, in plane)
    cmc, tm = J["thumb_cmc"], J["thumb_mcp"]
    dt = tm - cmc
    dtp = unit(np.array([dt[0], dt[1], 0.0]))
    then_c = cmc + 0.42 * dt + np.array([-10.0, -3.0, 0.0])
    then_c[2] = 7.0 * pt
    P.append(ellipsoid(then_c, frame_from(dtp, Z), [19 * pt, 15.5 * pt, 30], k=12, name="thenar"))

    # web between thumb and index (first web space)
    d_tp = unit(J["thumb_ip"] - tm)
    P.append(tri_slab(cmc + np.array([-4, 4, 3.0]), tm + 0.33 * np.linalg.norm(J["thumb_ip"] - tm) * d_tp
                      + np.array([-1.5, 0, 1.0]), I * 0.80 + np.array([0, 0, 2.0]), 5.0, k=8, name="thumbweb"))

    # MCP palmar pads and dorsal knuckles
    for f in FINGERS4:
        d = unit(J[f"{f}_pip"] - m[f])
        w = WIDTH_PIP[f] * ws.get(f, 1.0)
        P.append(ellipsoid(m[f] + np.array([0, -4.0, 9.0 * pt]) - 1.0 * d,
                           frame_from(d, Z), [0.5 * w, 5.5 * pt, 0.5 * w], k=10, name=f"pad_{f}"))
        P.append(ellipsoid(m[f] + np.array([0, -1.0, -3.0]), frame_from(d, Z),
                           [0.47 * w, 6.5, 0.5 * w], k=6, name=f"knuckle_{f}"))

    # fingers: each finger is its own sub-union, then blended into the hand
    for f in FINGERS4:
        w = WIDTH_PIP[f] * ws.get(f, 1.0)
        w = w * RADIUS_FIT
        r_mcp, r_pip, r_dip, r_tip = 0.5 * w * 1.07, 0.5 * w, 0.5 * w * 0.92, 0.5 * w * 0.85
        a, b, c, t = m[f], J[f"{f}_pip"], J[f"{f}_dip"], J[f"{f}_tip"]
        dd = unit(t - c)
        tip_end = t - dd * (r_tip - TIP_INSET)
        G = [round_cone(a, b, r_mcp, r_pip * 0.97, Z, k=0),
             round_cone(b, c, r_pip * 0.97, r_dip * 0.97, Z, k=0),
             round_cone(c, tip_end, r_dip * 0.97, r_tip, Z, k=0)]
        # joints slightly wider than the shafts, dorsally biased
        for jp, r in [(b, r_pip), (c, r_dip)]:
            dj = unit(t - jp)
            G.append(ellipsoid(jp - 1.0 * Z, frame_from(dj, Z), [r * 0.98, r * 0.78, r * 0.7], k=3.0))
        # palmar pulps of the phalanges (soft, low)
        for p0, p1, r, sz, fr in [(a, b, r_mcp, 0.45, 0.5), (b, c, r_pip, 0.4, 0.5), (c, tip_end, r_dip, 0.5, 0.62)]:
            dj = unit(p1 - p0)
            L = np.linalg.norm(p1 - p0)
            G.append(ellipsoid(p0 + dj * L * fr + Z * r * 0.3, frame_from(dj, Z),
                               [r * 0.78, r * 0.6, L * sz], k=3.0))
        P.append(group(G, k=6.0, name=f"finger_{f}"))
    # palmar webs between neighbouring fingers
    for fa, fb in [("index", "middle"), ("middle", "ring"), ("ring", "pinky")]:
        pa_ = m[fa] + 0.30 * (J[f"{fa}_pip"] - m[fa]) + Z * 2.5
        pb_ = m[fb] + 0.30 * (J[f"{fb}_pip"] - m[fb]) + Z * 2.5
        P.append(round_cone(pa_, pb_, 5.0, 5.0, Z, squash=0.8, k=5, name=f"web_{fa}"))
        # fill between the web line and the knuckle row (no tunnel under the web)
        qa, qb = m[fa] + Z * 2.0, m[fb] + Z * 2.0
        P.append(tri_slab(qa, qb, pb_, 5.0, k=5, name=f"webfill_{fa}"))
        P.append(tri_slab(qa, pb_, pa_, 5.0, k=5, name=f"webfill_{fa}"))

    # thumb
    wt = WIDTH_PIP["thumb"] * ws.get("thumb", 1.0) * RADIUS_FIT
    ip, tt = J["thumb_ip"], J["thumb_tip"]
    dd = unit(tt - ip)
    rt = 0.5 * wt * 0.86
    tip_end = tt - dd * (rt - TIP_INSET)
    G = [round_cone(cmc, tm, 0.5 * wt * 0.95, 0.5 * wt * 1.02, tbend, squash=0.9, k=0),
         round_cone(tm, ip, 0.5 * wt * 1.02, 0.5 * wt * 0.98, tbend, k=0),
         round_cone(ip, tip_end, 0.5 * wt * 0.98, rt, tbend, k=0),
         ellipsoid(ip - 1.0 * tbend, frame_from(dd, tbend), [0.49 * wt, 0.39 * wt, 0.35 * wt], k=3.0)]
    for p0, p1, r in [(tm, ip, 0.5 * wt), (ip, tip_end, 0.5 * wt)]:
        dj = unit(p1 - p0)
        L = np.linalg.norm(p1 - p0)
        G.append(ellipsoid(p0 + dj * L * 0.6 + tbend * r * 0.3, frame_from(dj, tbend),
                           [r * 0.8, r * 0.6, L * 0.5], k=3.0))
    P.append(group(G, k=8.0, name="thumb"))

    # central palmar hollow (smooth subtraction)
    hol = np.array([(I[0] + K[0]) * 0.5 - 2, 0.55 * ymcp, 17 * pt + 5.0])
    P.append(ellipsoid(hol, np.eye(3), [22 * pt, 24, 7.0], k=10, op="sub", name="hollow"))
    return P


def eval_sdf(P, prims, cut_y=-40.0):
    lo, hi = P.min(0), P.max(0)
    d = np.full(len(P), 1e3, np.float64)
    for pr in prims:
        m = pr.k + 3.0
        if pr.op == "union":
            if np.any(pr.lo - m > hi) or np.any(pr.hi + m < lo):
                continue
            d = smin(d, pr.fn(P), pr.k)
        else:
            if np.any(pr.lo - m > hi) or np.any(pr.hi + m < lo):
                continue
            d = smax(d, -pr.fn(P), pr.k)
    return np.maximum(d, cut_y - P[:, 1])     # flat forearm cut


def sdf_grid(prims, J, res=0.5, cut_y=-40.0):
    pts = np.array(list(J.values()))
    lo = np.floor(np.minimum(pts.min(0) - 26, [-45, cut_y - 3, -30]))
    hi = np.ceil(np.maximum(pts.max(0) + 26, [45, 0, 35]))
    xs = np.arange(lo[0], hi[0] + res, res)
    ys = np.arange(lo[1], hi[1] + res, res)
    zs = np.arange(lo[2], hi[2] + res, res)
    vol = np.empty((len(xs), len(ys), len(zs)), np.float32)
    XX, ZZ = np.meshgrid(xs, zs, indexing="ij")
    step = max(1, int(6 / res))
    for j0 in range(0, len(ys), step):
        yy = ys[j0:j0 + step]
        Pc = np.stack(np.broadcast_arrays(XX[:, None, :], yy[None, :, None], ZZ[:, None, :]), -1)
        sh = Pc.shape[:3]
        vol[:, j0:j0 + step, :] = eval_sdf(Pc.reshape(-1, 3), prims, cut_y).reshape(sh)
    return vol, np.array([xs[0], ys[0], zs[0]])


def decimate_bpy(V, F, target):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    me = bpy.data.meshes.new("m")
    me.from_pydata(V.tolist(), [], F.tolist())
    ob = bpy.data.objects.new("m", me)
    bpy.context.scene.collection.objects.link(ob)
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.ratio = min(1.0, target / len(V))
    mod.use_collapse_triangulate = True
    dg = bpy.context.evaluated_depsgraph_get()
    em = ob.evaluated_get(dg).to_mesh()
    V2 = np.array([v.co[:] for v in em.vertices])
    em.calc_loop_triangles()
    F2 = np.array([t.vertices[:] for t in em.loop_triangles])
    return V2, F2


def make_mesh(prims, J, res=0.5, target=60000):
    import trimesh
    from skimage.measure import marching_cubes
    t0 = time.time()
    vol, org = sdf_grid(prims, J, res)
    t1 = time.time()
    V, F, _, _ = marching_cubes(vol, 0.0, spacing=(res, res, res))
    V += org
    mesh = trimesh.Trimesh(V, F, process=False)   # process=True merges/deletes MC slivers -> holes
    print("MC mesh watertight", mesh.is_watertight, "euler", mesh.euler_number)
    t2 = time.time()
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=6)
    n_mc = len(mesh.vertices)
    V2, F2 = decimate_bpy(np.asarray(mesh.vertices), np.asarray(mesh.faces), target)
    mesh = trimesh.Trimesh(V2, F2, process=False)
    mesh.remove_unreferenced_vertices()
    print("decimated watertight", mesh.is_watertight, "euler", mesh.euler_number)
    trimesh.repair.fix_normals(mesh)
    if mesh.volume < 0:
        mesh.invert()
    t3 = time.time()
    info = {"grid_shape": list(vol.shape), "res_mm": res, "mc_vertices": n_mc,
            "t_sdf_s": round(t1 - t0, 1), "t_mc_s": round(t2 - t1, 1), "t_post_s": round(t3 - t2, 1)}
    return mesh, info


# ---------------------------------------------------------------- render
def render_views(mesh_path, J, out_png, samples=24):
    """Cycles CPU, orthographic palm / back / side views, labelled with OpenCV."""
    import bpy
    import cv2
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    bpy.ops.wm.ply_import(filepath=mesh_path)
    ob = bpy.context.selected_objects[0]
    for p in ob.data.polygons:
        p.use_smooth = True
    mat = bpy.data.materials.new("skin")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.80, 0.58, 0.46, 1)
    bsdf.inputs["Roughness"].default_value = 0.55
    ob.data.materials.append(mat)
    world = bpy.data.worlds.new("w")
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.9, 0.9, 0.92, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.45
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = False
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    V = np.array([v.co[:] for v in ob.data.vertices])
    c = 0.5 * (V.min(0) + V.max(0))
    ext = V.max(0) - V.min(0)
    views = [("Palm (+Z)", (0, 0, 1), (0, 1, 0)), ("Back (-Z)", (0, 0, -1), (0, 1, 0)),
             ("Side, little-finger side (-X)", (-1, 0, 0), (0, 1, 0))]
    imgs = []
    cam_d = bpy.data.cameras.new("cam")
    cam_d.type = "ORTHO"
    cam = bpy.data.objects.new("cam", cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam
    sun_d = bpy.data.lights.new("sun", "SUN")
    sun_d.energy = 3.5
    sun_d.angle = 0.15
    sun = bpy.data.objects.new("sun", sun_d)
    sc.collection.objects.link(sun)
    from mathutils import Vector
    H, Wd = 900, 520
    sc.render.resolution_x, sc.render.resolution_y = Wd, H
    span = max(ext[1] * 1.08, ext[0] * 1.08 * H / Wd)
    cam_d.ortho_scale = span
    for title, vdir, up in views:
        from mathutils import Matrix
        vd = Vector(vdir)
        cy = Vector(up)
        cx = cy.cross(vd)
        cam.matrix_world = Matrix.Translation(Vector(c) + vd * 400) @ Matrix(
            [[cx[i], cy[i], vd[i], 0] for i in range(3)] + [[0, 0, 0, 1]])
        # key light from upper-left of the viewer: light travels along -ld
        ld = (vd * 1.0 + cy * 0.7 - cx * 0.6).normalized()
        sz = ld
        sx = Vector((0, 0, 1)).cross(sz) if abs(sz[2]) < 0.9 else Vector((0, 1, 0)).cross(sz)
        sx.normalize()
        sy = sz.cross(sx)
        sun.matrix_world = Matrix([[sx[i], sy[i], sz[i], 0] for i in range(3)] + [[0, 0, 0, 1]])
        cam_d.clip_end = 1000
        sc.render.filepath = os.path.join(WORK, f"view_{os.getpid()}.png")
        bpy.ops.render.render(write_still=True)
        im = cv2.imread(sc.render.filepath)
        im = cv2.copyMakeBorder(im, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(im, title, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 2, cv2.LINE_AA)
        imgs.append(im)
    out = np.concatenate([np.pad(i, ((0, 0), (4, 4), (0, 0)), constant_values=255) for i in imgs], 1)
    bar = np.full((40, out.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, "Stand-in right hand, open pose (orthographic, same scale, Y up = fingers)",
                (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.imwrite(out_png, np.concatenate([bar, out], 0))
    return out_png


# ---------------------------------------------------------------- main
def variant_scales(k):
    rng = np.random.default_rng(1000 + k)
    ws = {f: float(rng.uniform(0.9, 1.1)) for f in ["thumb"] + FINGERS4}
    ls = {f: {s: float(rng.uniform(0.95, 1.05)) for s in ["metacarpal"] + SEGS4} for f in FINGERS4}
    ls["thumb"] = {s: float(rng.uniform(0.95, 1.05)) for s in SEGS["thumb"]}
    palm = float(rng.uniform(0.9, 1.1))
    return ws, ls, palm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, default=None)
    ap.add_argument("--res", type=float, default=0.5)
    ap.add_argument("--target-verts", type=int, default=60000)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    out = a.out or (OUT_ROOT if a.variant is None else os.path.join(OUT_ROOT, f"variant_{a.variant}"))
    os.makedirs(out, exist_ok=True)

    lay = palm_layout(read_rig())
    ws, ls, palm = ({}, {}, 1.0) if a.variant is None else variant_scales(a.variant)
    J, tbend = build_joints(lay, OPEN_POSE, ls)
    prims = build_prims(J, tbend, ws, palm)
    mesh, info = make_mesh(prims, J, a.res, a.target_verts)

    ply = os.path.join(out, "standin_open.ply")
    mesh.export(ply)
    mesh.export(os.path.join(out, "standin_open.obj"))
    # palm normal: out of the palm (+Z by construction; recomputed from joints)
    across = unit(J["index_mcp"] - J["pinky_mcp"])
    along = unit(J["middle_mcp"] - J["wrist"])
    pn = unit(np.cross(across, along))
    jj = {"side": "right", "units": "mm", "pose": "open",
          "landmarks": {n: [round(float(c), 3) for c in J[n]] for n in NAMES},
          "palm_normal": [round(float(c), 5) for c in pn],
          "lengths": lengths_of(J),
          "angles_deg": joint_angles(J),
          "source": {"rig": RIG_BLEND, "generator": "standin.py",
                     "variant": a.variant,
                     "width_pip_mm": {f: round(WIDTH_PIP[f] * ws.get(f, 1.0), 2) for f in WIDTH_PIP},
                     "thickness_ratio": THICK_RATIO, "palm_scale": palm,
                     "length_scale": ls, "tip_inset_mm": TIP_INSET, "open_pose": OPEN_POSE}}
    json.dump(jj, open(os.path.join(out, "joints.json"), "w"), indent=1)

    # checks
    from scipy.spatial import cKDTree
    Vv = np.asarray(mesh.vertices)
    tree = cKDTree(Vv)
    tips = [f"{f}_tip" for f in ["thumb"] + FINGERS4]
    pts = np.array([J[n] for n in NAMES])
    field = eval_sdf(pts, prims)
    tip_depth = {t: round(float(tree.query(J[t])[0]), 2) for t in tips}
    stats = dict(info, vertices=len(mesh.vertices), faces=len(mesh.faces),
                 watertight=bool(mesh.is_watertight), winding_consistent=bool(mesh.is_winding_consistent),
                 volume_ml=round(mesh.volume / 1000, 1), euler=int(mesh.euler_number),
                 genus=int(round(1 - mesh.euler_number / 2)), components=len(mesh.split(only_watertight=False)),
                 bbox_min=np.round(Vv.min(0), 1).tolist(), bbox_max=np.round(Vv.max(0), 1).tolist(),
                 joints_inside_field=int((field < 0).sum()),
                 tip_to_skin_mm=tip_depth)

    def span(p, d, L=60.0):
        """Skin-to-skin extent of the SDF along the line p + t d."""
        t = np.linspace(-L, L, 4801)
        f = eval_sdf(p[None] + t[:, None] * d[None], prims)
        c = len(t) // 2
        if f[c] >= 0:
            return None
        out_r = np.where(f[c:] >= 0)[0]
        out_l = np.where(f[:c][::-1] >= 0)[0]
        if not len(out_r) or not len(out_l):
            return None
        return round(float(t[c + out_r[0]] - t[c - 1 - out_l[0]]), 1)
    Zv = np.array([0.0, 0, 1])
    th = {"mcp_row(middle_mcp -5)": span(J["middle_mcp"] - [0, 5, 0], Zv),
          "mid_palm": span(np.array([-3, 0.5 * J["middle_mcp"][1], 0]), Zv),
          "thenar": span(np.r_[(0.5 * (J["thumb_cmc"] + J["thumb_mcp"]))[:2] + [-8, 0], 0], Zv),
          "wrist": span(np.zeros(3), Zv),
          "breadth_mcp_row": span(J["middle_mcp"] - [0, 8, 0], np.array([1.0, 0, 0]), 80)}
    for f in ["thumb"] + FINGERS4:
        p = J[f"{f}_ip"] if f == "thumb" else J[f"{f}_pip"]
        d = unit(J[f"{f}_tip"] - J[f"{f}_mcp"])
        side = unit(np.cross(d, tbend if f == "thumb" else Zv))
        th[f"width_{f}_pip"] = span(p, side, 30)
        th[f"thick_{f}_pip"] = span(p, unit(np.cross(side, d)), 30)
    stats["section_mm"] = th
    json.dump(stats, open(os.path.join(out, "mesh_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))
    print(json.dumps(jj["angles_deg"]))
    if not a.no_render:
        render_views(ply, J, os.path.join(out, "standin_views.png"))
    print("WROTE", out)


if __name__ == "__main__":
    main()
