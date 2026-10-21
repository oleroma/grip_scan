"""Re-solve the HOLD pose of the fitted hand on the grip, with the real skin.

    P=/tmp/claude-0/hs/venv/bin/python
    $P hold_solve.py solve   [--params P] [--tag T]     # optimise wrist placement + joint angles -> WORK/solve_<tag>.json
    $P hold_solve.py export  [--tag T]                  # .blend + pose.json + contacts.json + metrics.json -> OUT
    $P hold_solve.py render --solve WORK/solve_<tag>.json   # before / after renders + ring positions
    $P hold_solve.py sheet                              # hold_v2_sheet.jpg (before / after, plain-language)
    $P hold_solve.py skincheck [--params P]             # flipped triangles + index->thumb leak for a params file

Why.  run_all.py's hold step copies the joint angles of the old cylinder rig and fits the
placement on the cylinder rig's JOINTS.  The fitted (scan-based) hand has a different skin
thickness, so with the same joints the palm stood 10.8 mm off the grip's palm swell, skin went
3.9 mm into the plastic and 60 palm triangles flipped.  Here the placement (wrist rotation +
translation) and the joint angles are solved against the SKIN itself (numpy backend of skin.py,
the same deformation Blender evaluates), staying close to the reference hold.

Unknowns (30): the 22 skin.Rig DOFs (palm arch x2, spreads x3, finger MCP/PIP/DIP x12, thumb
CMC azimuth / elevation / MCP / IP, thumb pronation), the two thumb axial twists that
run_all.refine_on_joints uses, and the rigid placement (rotation vector about the hand's
current orientation, translation).  Start = run_all's old placement ("before").

Objective (scipy least_squares, trf, analytic-free: parallel forward differences):
  grip penetration   every skin vertex: 4 * max(0, -sd - 0.4 mm) * sqrt(area share); sd = signed
                     distance to the grip (precomputed 0.5 mm grid, trilinear), so skin may press
                     0.4 mm into the plastic for free (real skin squashes), deeper costs a lot.
  palm on the swell  soft-min distance from the 'palm swell' marker (on the grip surface) to the
                     palm skin = 0.3 mm (weight 4).
  buttons            soft-min distance from 'R1 target' / 'R2 press' / 'R4 press' to the skin of the
                     index / middle / ring DISTAL segment (as contacts.json measures) = 0.3 mm.
  pinky wraps        its distal pad touches the grip (min sd of pinky distal skin = 0.3 mm, weight 2)
                     and its angles stay near the reference.
  thumb              thumb MCP / IP / tip joint positions stay at the reference (sigma 3 mm).
  reference hold     every SPEC angle near the reference (sigma 6-12 deg, wider for the joints
                     that must move to reach the buttons), every joint near its reference position
                     (sigma 12 mm), wrist sigma 6 mm.
  anatomical limits  hinge penalties outside LIMITS (deg, SPEC definitions, + = toward the palm).
  skin quality       faces whose normal turned more than ~80 deg against the rotation of their
                     bone (fold-overs / flips): max(0, 0.2 - dot), weight 3.
The 2026-10-21 hold_v2 result (fitted_hold/): skin_params_hold_v2.json (defaults + thumb_pure=true,
thumb_softness_mm=24), solved in two runs: --w '{"thumb":3}' 40 evaluations from the old placement,
then --w '{"thumb":3,"swell":8,"flips":6}' 60 more (--start).  Why those skin parameters: with the
defaults every thumb phalanx vertex carried up to 12% index weight (the wide thumb/finger logistic
never reaches 1), so bending the index MCP 20 deg moved the thumb skin 1.5 mm (now 0.15 mm), and the
palm centre carried 33% thumb-metacarpal weight, so the solver could only bring the palm onto the
swell by swinging the thumb.  The palm flips of the old pose (pinky/ring metacarpal) go away with
the new pose; the ~20 left are isolated sub-millimetre triangles (0.02-0.2 mm2) at the PIP/DIP
creases that Corrective Smooth does not fully unfold; no skin_params change tried (CS factor /
iterations / zone, weight smoothing, PIP width / bulge, palm softness) lowered them.
Grip signed distance = nearest-face-normal sign (as grip_views.py); the final numbers come from
run_all.py's contacts step (0.5 mm skin samples, exact BVH distance + winding-number sign).
"""
import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FIT_RUN = "/mnt/project-files/hand_scan_rig/fitted/run"
FIT_BLEND = os.path.join(FIT_RUN, "fit", "hand_rigged.blend")
HOLD_RIG = "/tmp/claude-0/hs/hold_rig.blend"
PARAMS = "/mnt/project-files/hand_scan_rig/skin/skin_params.json"
OUT = "/mnt/project-files/hand_scan_rig/fitted_hold"
WORK = "/tmp/claude-0/hs/work/hold_v2"
GRIP = "grip (current GripV)"
MARKERS = ["R1 target", "R2 press", "R4 press", "palm swell"]
BUTTON = {"R1 target": "index", "R2 press": "middle", "R4 press": "ring"}
GRID_MM = 1.0
FREE_PRESS_MM = 0.4        # skin may sit this deep in the plastic without cost (squash)
TOUCH_MM = 0.3             # target distance marker -> skin (touching, slightly pressed)

# anatomical limits (deg, SPEC definitions: flexion + toward the palm; thumb sign per SPEC)
LIMITS = {"mcp": (-20.0, 95.0), "pip": (0.0, 110.0), "dip": (-10.0, 85.0),
          "thumb_mcp": (-15.0, 65.0), "thumb_ip": (-20.0, 85.0), "thumb_cmc_elev": (-10.0, 70.0),
          "spread": (-25.0, 30.0), "arch": (-5.0, 25.0)}
# prior sigma (deg) of each angle around the reference hold
SIG_DEFAULT = 8.0
SIG = {"index_mcp": 15.0, "index_pip": 15.0, "index_dip": 15.0,
       "middle_mcp": 12.0, "middle_pip": 12.0, "middle_dip": 15.0,
       "ring_mcp": 12.0, "ring_pip": 12.0, "ring_dip": 15.0,
       "pinky_mcp": 10.0, "pinky_pip": 10.0, "pinky_dip": 12.0,
       "arch_imr": 4.0, "arch_mrp": 4.0, "spread_index_middle": 6.0, "spread_middle_ring": 6.0,
       "spread_ring_pinky": 6.0, "thumb_cmc_azimuth": 5.0, "thumb_cmc_elev": 5.0, "thumb_mcp": 5.0,
       "thumb_ip": 6.0}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))


def limit_of(k):
    if k.startswith("spread"):
        return LIMITS["spread"]
    if k.startswith("arch"):
        return LIMITS["arch"]
    if k in LIMITS:
        return LIMITS[k]
    j = k.split("_")[-1]
    return LIMITS.get(j)


# ------------------------------------------------------------------ scene

def grip_sdf_grid(Vg, Fg, lo, hi, step, cache):
    """Signed distance (mm, - inside) to the closed grip on a regular grid, nearest-face-normal sign."""
    if os.path.exists(cache):
        d = np.load(cache)
        if np.allclose(d["lo"], lo) and np.allclose(d["hi"], hi) and float(d["step"]) == step:
            return d["sdf"]
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    bvh = BVHTree.FromPolygons(Vg.tolist(), Fg.tolist())
    axes = [np.arange(lo[i], hi[i] + 1e-9, step) for i in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    sdf = np.empty(len(G), np.float32)
    t0 = time.time()
    for i, q in enumerate(G):
        loc, nrm, _, dist = bvh.find_nearest(Vector(q))
        sdf[i] = dist if (q - np.array(loc)) @ np.array(nrm) >= 0 else -dist
    sdf = sdf.reshape([len(a) for a in axes])
    log(f"grip sdf grid {sdf.shape} in {time.time() - t0:.0f} s")
    np.savez_compressed(cache, sdf=sdf, lo=lo, hi=hi, step=step)
    return sdf


class Scene:
    """Fitted rigged hand (numpy skinning), the grip as a signed-distance grid, markers and the
    reference hold (joints of the cylinder rig in hold_rig.blend)."""

    def __init__(self, params=None, fit_blend=FIT_BLEND, hold_rig=HOLD_RIG):
        import bpy
        import skin
        from run_all import world_mesh
        os.makedirs(WORK, exist_ok=True)
        bpy.ops.wm.open_mainfile(filepath=hold_rig)
        self.H = {k: np.asarray(v, float) for k, v in
                  skin._landmarks_from_armature(bpy.data.objects["hand rig"]).items()}
        self.Vg, self.Fg = world_mesh(bpy.data.objects[GRIP])
        self.markers = {n: np.array(bpy.data.objects[n].matrix_world.translation) for n in MARKERS}
        self.params_path = params or PARAMS
        self.sk = skin.Skinner(fit_blend, self.params_path)
        sk = self.sk
        self.F = sk.F
        self.rig = sk.rig
        self.ref = skin.measurements(self.H)
        # labels by strongest bone (as run_all's contacts step)
        self.lab = np.array(skin.BONES)[np.argmax(sk.W, axis=1)]
        A = sk.V0[self.F]
        area = 0.5 * np.linalg.norm(np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]), axis=1)
        va = np.zeros(len(sk.V0))
        np.add.at(va, self.F.ravel(), np.repeat(area / 3, 3))
        self.vw = np.sqrt(va / va.mean())
        self.face_bone = np.argmax(sk.W[self.F].sum(axis=1), axis=1)
        self.n0 = face_normals(sk.V0, self.F)
        lo = self.Vg.min(0) - 14.0
        hi = self.Vg.max(0) + 14.0
        self.glo, self.gstep = lo, GRID_MM
        self.sdf = grip_sdf_grid(self.Vg, self.Fg, lo, hi, GRID_MM, os.path.join(WORK, "grip_sdf.npz"))
        pal = np.isin(self.lab, ["palm", "index_metacarpal", "middle_metacarpal", "ring_metacarpal",
                                 "pinky_metacarpal", "thumb_metacarpal"])
        self.sel = {"palm": np.flatnonzero(pal)}
        for f in ("index", "middle", "ring", "pinky", "thumb"):
            self.sel[f + "_distal"] = np.flatnonzero(self.lab == f + "_distal")
        self.names = list(skin.NAMES)

    # -- grip distance
    def grip_sd(self, X):
        from scipy.ndimage import map_coordinates
        c = ((X - self.glo) / self.gstep).T
        return map_coordinates(self.sdf, c, order=1, mode="nearest")

    # -- forward model
    def forward(self, z, verts=True):
        """z = [22 rig DOFs, 2 thumb twists, rotvec(3), t(3)] -> dict(V world, P landmarks world, M, R, t)."""
        from scipy.spatial.transform import Rotation
        x, tw = z[:22], {"thumb_metacarpal": z[22], "thumb_proximal": z[23]}
        R = Rotation.from_rotvec(z[24:27]).as_matrix() @ self.R0
        t = z[27:30]
        M, P = self.rig.fk(x, tw)
        Pw = {k: R @ v + t for k, v in P.items()}
        out = dict(P=Pw, M=M, R=R, t=t)
        if verts:
            bv = self.sk.bulge_values(M=M)
            out["Vl"] = self.sk._pose_numpy(M, bv)
            out["V"] = out["Vl"] @ R.T + t
            out["bulge"] = bv
        return out

    def flips(self, o, thresh=0.0):
        """Faces whose posed normal points against the rest normal turned by the face's bone."""
        n1 = face_normals(o["Vl"], self.F)
        import skin
        Rb = np.array([o["M"][b][0] for b in skin.BONES])
        n0r = np.einsum("fij,fj->fi", Rb[self.face_bone], self.n0)
        return np.einsum("ij,ij->i", n0r, n1) < thresh, np.einsum("ij,ij->i", n0r, n1)


def face_normals(V, F):
    t = V[F]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)


def softmin(d, tau=0.15, k=4):
    """Smooth minimum of the k smallest values (no log(N) bias: equals the min when one value
    is far below the rest, at most tau*log(k) below the k-th smallest otherwise)."""
    s = np.partition(d, k)[:k] if len(d) > k else np.asarray(d)
    m = s.min()
    return float(m - tau * np.log(np.exp(-(s - m) / tau).mean()))


def old_placement(S, angle_prior=0.05):
    """run_all.py's hold placement ("before"): hold angles, rigid fit, joint least squares."""
    from run_all import kabsch, refine_on_joints
    from skin import MEAS
    hold = S.ref
    pkeys = list(MEAS) + ["thumb_cmc_azimuth"]
    pose = {"angles": {k: float(hold[k]) for k in pkeys}}
    Hs = np.array([S.H[n] for n in S.names])
    M0, P0, info = S.rig.solve(pose)
    R0, t0 = kabsch(np.array([P0[n] for n in S.names]), Hs)
    x, R, t = refine_on_joints(S.rig, info["x"], R0, t0, Hs, S.names, angle_prior)
    return np.r_[x, np.zeros(3), t], R


# ------------------------------------------------------------------ objective

R1_OVERLAP_MM = 24.0       # Olex's case rig: the R1 centre sits this far from the index fingertip end
R1_OVERLAP_SIG = 6.0


class Objective:
    """Residual vector of the hold solve (see the module docstring)."""

    def __init__(self, S, z_ref, w=None):
        import skin
        self.S, self.z_ref = S, np.asarray(z_ref, float)
        self.w = dict(pen=4.0, swell=4.0, button=4.0, r1_along=1.0, pinky=2.0, thumb=1.0,
                      joints=1.0, angles=1.0, limits=2.0, flips=3.0, twist=1.0)
        self.w.update(w or {})
        ref = S.ref
        # angle keys of the prior: SPEC angles (spreads signed), thumb by azimuth + elevation
        self.akeys = [k for k in skin.MEAS if k != "thumb_cmc"] + ["thumb_cmc_azimuth"]
        self.aref = np.array([ref["_s" + k] if k.startswith("spread") else ref[k] for k in self.akeys])
        self.asig = np.array([SIG.get(k, SIG_DEFAULT) for k in self.akeys])
        self.jw = np.array([w_ if n.startswith("thumb") else 1.0 for n in S.names
                            for w_ in [self.w["thumb"]]])
        self.lkeys = [k for k in skin.MEAS if limit_of(k) is not None and k != "thumb_cmc"]
        self.Href = np.array([S.H[n] for n in S.names])
        self.jsig = np.array([6.0 if n == "wrist" else 3.0 if n in ("thumb_mcp", "thumb_ip", "thumb_tip")
                              else 12.0 for n in S.names])
        # the R1 contact point: R1's centre projected onto the grip surface (the marker sits 4.6 mm
        # inside the plastic); the index skin must touch it
        p = S.markers["R1 target"]
        g = np.array(np.gradient(S.sdf, S.gstep))     # not used at run time; projection by bisection below
        del g
        self.r1_surf = project_to_surface(S, p)
        idx = np.flatnonzero(np.char.startswith(S.lab.astype(str), "index_")
                             & (S.lab != "index_metacarpal"))
        self.idx_finger = idx
        self.idx_s_from_tip = s_from_tip(S, idx, "index")
        self.targets = {"R1 target": (self.r1_surf, self.idx_finger)}
        for m in ("R2 press", "R4 press"):
            self.targets[m] = (S.markers[m], S.sel[BUTTON[m] + "_distal"])

    def angles(self, P):
        import skin
        m = skin.measurements(P)
        return m, np.array([m["_s" + k] if k.startswith("spread") else m[k] for k in self.akeys])

    def __call__(self, z, detail=False):
        S, w = self.S, self.w
        o = S.forward(z)
        V = o["V"]
        r = []
        sd = S.grip_sd(V)
        r.append(w["pen"] * S.vw * np.maximum(0.0, -sd - FREE_PRESS_MM))
        # palm on the swell
        ps = S.markers["palm swell"]
        d_sw = softmin(np.linalg.norm(V[S.sel["palm"]] - ps, axis=1))
        r.append([w["swell"] * (d_sw - TOUCH_MM)])
        # buttons
        dist = {}
        for m, (p, sel) in self.targets.items():
            d = np.linalg.norm(V[sel] - p, axis=1)
            dist[m] = softmin(d)
            r.append([w["button"] * (dist[m] - TOUCH_MM)])
            if m == "R1 target":
                a = np.exp(-(d - d.min()) / 0.5)
                s_eff = float((a * self.idx_s_from_tip).sum() / a.sum())
                r.append([w["r1_along"] * (s_eff - R1_OVERLAP_MM) / R1_OVERLAP_SIG])
        # pinky pad on the grip
        pk = softmin(sd[S.sel["pinky_distal"]])
        r.append([w["pinky"] * (pk - TOUCH_MM)])
        # reference hold: joints, angles, limits, twists
        P = o["P"]
        Q = np.array([P[n] for n in S.names])
        r.append(w["joints"] * (self.jw[:, None] * (Q - self.Href) / self.jsig[:, None]).ravel())
        m, a = self.angles(P)
        r.append(w["angles"] * (a - self.aref) / self.asig)
        lim = []
        for k in self.lkeys:
            lo, hi = limit_of(k)
            v = m["_s" + k] if k.startswith("spread") else m[k]
            lim.append(max(0.0, lo - v) + max(0.0, v - hi))
        r.append(w["limits"] * np.array(lim))
        r.append(w["twist"] * (z[21:24] - self.z_ref[21:24]) / 10.0)
        fl, dot = S.flips(o)
        r.append(w["flips"] * np.maximum(0.0, 0.2 - dot))
        res = np.concatenate([np.atleast_1d(np.asarray(x, float)) for x in r])
        if detail:
            return res, dict(o=o, sd=sd, swell=d_sw, buttons=dist, pinky_sd=pk, angles=m, flips=int(fl.sum()),
                             pen_max=float(max(0.0, -sd.min())))
        return res


def project_to_surface(S, p, iters=40):
    """Nearest grip surface point to p (exact BVH)."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    bvh = BVHTree.FromPolygons(S.Vg.tolist(), S.Fg.tolist())
    loc, nrm, _, _ = bvh.find_nearest(Vector(p))
    del iters
    return np.array(loc)


def s_from_tip(S, idx, finger):
    """Rest-pose distance (mm, along the finger) of each skin vertex from the fingertip END of the skin."""
    L = S.rig.L
    pts = [L[f"{finger}_{j}"] for j in ("mcp", "pip", "dip", "tip")]
    V = S.sk.V0[idx]
    seg_len = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(3)]
    cum = np.r_[0, np.cumsum(seg_len)]
    best = np.full(len(V), np.inf)
    s = np.zeros(len(V))
    for i in range(3):
        a, b = pts[i], pts[i + 1]
        d = b - a
        t = np.clip((V - a) @ d / (d @ d), 0, 1)
        dist = np.linalg.norm(V - (a + t[:, None] * d), axis=1)
        u = (V - a) @ d / np.linalg.norm(d)
        upd = dist < best
        best[upd] = dist[upd]
        s[upd] = cum[i] + u[upd]
    end = s.max()                     # the skin's tip end along the finger
    return end - s


# ------------------------------------------------------------------ solve

_OBJ = None


def _col(args):
    j, z, h, f0 = args
    zz = z.copy()
    zz[j] += h
    return (_OBJ(zz) - f0) / h


def solve(obj, z0, max_nfev=40, workers=4, log_every=1):
    """least_squares (trf) with a parallel forward-difference Jacobian (absolute steps)."""
    import multiprocessing as mp
    from scipy.optimize import least_squares
    global _OBJ
    _OBJ = obj
    steps = np.r_[np.full(21, 0.05), 0.05, 0.05, 0.05, np.full(3, 1e-3), np.full(3, 0.02)]
    pool = mp.get_context("fork").Pool(workers)
    cache = {}

    def fun(z):
        k = z.tobytes()
        if k not in cache:
            cache.clear()
            cache[k] = obj(z)
        return cache[k]

    it = [0]

    def jac(z):
        f0 = fun(z)
        cols = pool.map(_col, [(j, z, steps[j], f0) for j in range(len(z))])
        it[0] += 1
        if it[0] % log_every == 0:
            log(f"  jac {it[0]}: cost {0.5 * float(f0 @ f0):.3f}")
        return np.stack(cols, 1)
    x_scale = np.r_[np.full(24, 5.0), np.full(3, 0.05), np.full(3, 2.0)]
    try:
        sol = least_squares(fun, z0, jac=jac, method="trf", x_scale=x_scale, max_nfev=max_nfev,
                            xtol=1e-6, ftol=1e-6)
    finally:
        pool.close()
        pool.join()
    return sol


def leak(S, z, deg=20.0):
    """Index -> thumb weight leak: in the hold pose z, bend the index MCP by +deg and measure how far
    the thumb phalanx skin (thumb_proximal / thumb_distal labels) moves; plus the largest
    index-phalanx weight on that skin."""
    import skin
    o0 = S.forward(z)
    z1 = np.array(z, float).copy()
    z1[5] += deg                       # rig DOF 5 = index MCP flexion
    o1 = S.forward(z1)
    th = np.isin(S.lab, ["thumb_proximal", "thumb_distal"])
    mv = np.linalg.norm(o1["V"] - o0["V"], axis=1)
    ib = [skin.BI[b] for b in ("index_proximal", "index_middle", "index_distal")]
    return {"thumb_moves_mm_when_index_mcp_bends_20deg": float(mv[th].max()),
            "index_weight_on_thumb_skin_max": float(S.sk.W[th][:, ib].sum(1).max()),
            "index_tip_moves_mm": float(mv[S.lab == "index_distal"].max())}


def flip_report(S, z):
    from collections import Counter
    o = S.forward(z)
    fl, dot = S.flips(o)
    lab = S.lab[S.F[:, 0]]
    return {"flipped_faces": int(fl.sum()), "faces_turned_over_60deg": int((dot < 0.5).sum()),
            "flipped_by_part": dict(Counter(lab[fl].tolist()).most_common(8))}


def pose_report(obj, z):
    """Angles vs the reference hold, joint shifts from the reference, placement change."""
    S = obj.S
    o = S.forward(z, verts=False)
    m, a = obj.angles(o["P"])
    ang = {k: {"ref": round(float(r), 1), "now": round(float(v), 1), "diff": round(float(v - r), 1)}
           for k, r, v in zip(obj.akeys, obj.aref, a)}
    jd = {n: round(float(np.linalg.norm(o["P"][n] - S.H[n])), 1) for n in S.names}
    return {"angles_deg": ang, "joint_shift_from_reference_mm": jd}


def cmd_skincheck(a):
    S, z0 = setup(a.params)
    zs = {"before": z0}
    if a.solve:
        zs["solved"] = np.array(json.load(open(a.solve))["z"])
    for k, z in zs.items():
        print(k, json.dumps({**flip_report(S, z), **leak(S, z)}))
        if a.verbose:
            print(json.dumps(pose_report(Objective(S, z0), z)))


def setup(params=None):
    """Scene + the "before" placement (run_all's) as the start and the angle/twist reference."""
    S = Scene(params)
    zb_path, R0_path = os.path.join(WORK, "z_before.npy"), os.path.join(WORK, "R0.npy")
    if os.path.exists(zb_path) and os.path.exists(R0_path):
        z0, R0 = np.load(zb_path), np.load(R0_path)
    else:
        z0, R0 = old_placement(S)
        np.save(zb_path, z0)
        np.save(R0_path, R0)
    S.R0 = R0
    return S, z0


def summary(obj, z):
    r, d = obj(z, detail=True)
    return {"cost": 0.5 * float(r @ r), "palm_swell_gap_mm": d["swell"], "pen_max_vertex_mm": d["pen_max"],
            "buttons_mm": d["buttons"], "pinky_pad_sd_mm": d["pinky_sd"], "flipped_faces": d["flips"]}


def cmd_solve(a):
    S, z0 = setup(a.params)
    obj = Objective(S, z0, json.loads(a.w) if a.w else None)
    start = z0
    if a.start:
        start = np.array(json.load(open(a.start))["z"])
    log("start", json.dumps(summary(obj, start)))
    t0 = time.time()
    sol = solve(obj, start, max_nfev=a.nfev, workers=a.workers)
    s = summary(obj, sol.x)
    log(f"done {time.time() - t0:.0f} s, nfev {sol.nfev}:", json.dumps(s))
    jdump({"z": sol.x, "R0": S.R0, "summary": s, "params": S.params_path, "nfev": sol.nfev,
           "status": sol.status, "message": sol.message}, os.path.join(WORK, f"solve_{a.tag}.json"))


# ------------------------------------------------------------------ export

BEFORE_BLEND = "/mnt/project-files/hand_scan_rig/fitted/hand_on_grip_fitted.blend"
BEFORE_CONTACTS = "/mnt/project-files/hand_scan_rig/fitted/contacts.json"


def unpack(z, R0):
    from scipy.spatial.transform import Rotation
    z = np.asarray(z, float)
    return z[:22], {"thumb_metacarpal": float(z[22]), "thumb_proximal": float(z[23])}, \
        Rotation.from_rotvec(z[24:27]).as_matrix() @ R0, z[27:30]


def pose_json(rig, z, R0, source):
    """SPEC pose.json: angles (relaxed_pose definitions + the extra DOFs) and the root placement
    (rotation about the wrist, then translation, armature space), plus the exact solver state."""
    import skin
    from scipy.spatial.transform import Rotation
    x, tw, R, t = unpack(z, R0)
    M, P = rig.fk(x, tw)
    m = skin.measurements(P)
    keys = list(skin.MEAS) + ["thumb_cmc_azimuth"]
    w = rig.L["wrist"]
    q = Rotation.from_matrix(R).as_quat()          # x, y, z, w
    angles = {k: round(float(m[k]), 4) for k in keys}
    angles["thumb_pronation"] = round(float(x[21]), 4)   # rig DOF (axial, + about wrist->tip of the metacarpal)
    pose = {"angles": angles,
            "twist": {b: round(v, 4) for b, v in tw.items()},
            "root": {"location": (R @ w + t - w).tolist(), "rotation_quat": [q[3], q[0], q[1], q[2]]},
            "rig_dofs_deg": [round(float(v), 6) for v in x],
            "placement": {"R": R.tolist(), "t_mm": t.tolist(), "how": "world = R @ armature-space point + t"},
            "source": source}
    # check: the SPEC solver reproduces the joints from angles + twist + root
    Ms, Ps, info = rig.solve({"angles": pose["angles"], "twist": tw, "root": pose["root"]})
    pose["check_resolve_joint_err_mm"] = round(float(max(np.linalg.norm(Ps[n] - (R @ P[n] + t))
                                                         for n in skin.NAMES)), 3)
    return pose


def cmd_blend(a):
    """(own process) write the .blend for solve a.solve with params a.params."""
    import skin
    from run_all import write_hold_blend
    d = json.load(open(a.solve))
    sk = skin.Skinner(FIT_BLEND, d["params"])
    sk.arm.data.bones["palm"].use_deform = True
    sk.apply_to_blend()
    x, tw, R, t = unpack(d["z"], np.array(d["R0"]))
    M, P = sk.rig.fk(x, tw)
    pose = pose_json(sk.rig, d["z"], np.array(d["R0"]), "hold_solve.py")
    Vb, Vn, check = write_hold_blend(sk, M, R, t, pose["angles"], HOLD_RIG, a.out_blend)
    np.save(os.path.join(WORK, "after_verts_blender.npy"), Vb)
    jdump({"live_check": check, "pose": pose}, os.path.join(WORK, "blend_check.json"))
    log("blend written", a.out_blend, "blender vs numpy", check["blender_vs_numpy_same_bulge_mm"])


def cmd_contacts(a):
    from run_all import contacts_of_blend
    res = contacts_of_blend(a.blend, 0.5, old=False)
    jdump(res, a.out_json)


def r1_surface_gap(S, V):
    """R1: the marker centre sits 4.6 mm inside the plastic, so contacts.json's R1 number cannot
    go below ~4.6 without the finger going into the grip.  This is the distance from R1's point
    ON the grip surface to the index skin (distal / whole finger) and the touching segment."""
    p = project_to_surface(S, S.markers["R1 target"])
    out = {}
    for key, sel in (("fingertip", S.lab == "index_distal"),
                     ("finger", np.char.startswith(S.lab.astype(str), "index_") & (S.lab != "index_metacarpal"))):
        d = np.linalg.norm(V[sel] - p, axis=1)
        out[f"R1_surface_point_to_index_{key}_mm"] = round(float(d.min()), 2)
        if key == "finger":
            out["R1_touching_segment"] = str(S.lab[np.flatnonzero(sel)[np.argmin(d)]])
    return out


def cmd_export(a):
    import subprocess
    os.makedirs(OUT, exist_ok=True)
    d = json.load(open(os.path.join(WORK, f"solve_{a.tag}.json")))
    blend = os.path.join(OUT, "hand_on_grip_hold_v2.blend")
    py, me = sys.executable, os.path.abspath(__file__)
    sol = os.path.join(WORK, f"solve_{a.tag}.json")
    subprocess.run([py, me, "blend", "--solve", sol, "--out-blend", blend], check=True)
    subprocess.run([py, me, "contacts", "--blend", blend, "--out-json", os.path.join(OUT, "contacts.json")], check=True)
    subprocess.run([py, me, "metrics", "--solve", sol], check=True)


def cmd_metrics(a):
    d = json.load(open(a.solve))
    before_c = json.load(open(BEFORE_CONTACTS))["new_scan_hand"]
    after_c = json.load(open(os.path.join(OUT, "contacts.json")))["new_scan_hand"]
    rows = {}
    for which, params, z, con in (("before", PARAMS, None, before_c), ("after", d["params"], d["z"], after_c)):
        S, z0 = setup(params)
        z = z0 if z is None else np.array(z)
        o = S.forward(z)
        fl = flip_report(S, z)
        lk = leak(S, z)
        obj = Objective(S, z0)
        pr = pose_report(obj, z)
        from skin import contact_labels, resolve_contacts
        _, cinf = resolve_contacts(o["V"], S.F, S.sk.V0, S.sk.cs.A, labels=contact_labels(S.sk.W), iters=0)
        rows[which] = {
            "palm_swell_gap_mm": con["palm_swell_gap_mm"],
            "max_penetration_into_grip_mm": round(con["penetration_max_mm"], 2),
            "penetration_p95_mm": round(con["penetration_p95_mm"], 2),
            "penetrating_area_cm2": round(con["penetrating_area_cm2"], 2),
            "deepest_at": con["deepest_at"],
            "penetration_max_by_part_mm": con["penetration_max_by_part_mm"],
            "fingertip_to_button_mm": {"R1_index (marker centre, as contacts.json)": con["R1_index_fingertip_mm"],
                                       "R1_index_whole_finger (marker centre)": con["R1_index_finger_mm"],
                                       **r1_surface_gap(S, o["V"]),
                                       "R2_middle": con["R2_middle_fingertip_mm"],
                                       "R4_ring": con["R4_ring_fingertip_mm"]},
            "flipped_triangles": fl["flipped_faces"],
            "flipped_by_part": fl["flipped_by_part"],
            "faces_turned_over_60deg": fl["faces_turned_over_60deg"],
            "index_to_thumb_leak": lk,
            "skin_into_skin_max_mm": round(cinf["contact_depth_max_mm"], 2),
            "params": params,
            "angles_vs_reference_deg": pr["angles_deg"],
            "joint_shift_from_reference_mm": pr["joint_shift_from_reference_mm"],
        }
        if which == "after":
            pj = pose_json(S.rig, z, S.R0, "hold_solve.py solve (fitted hand, re-placed on the grip)")
            jdump(pj, os.path.join(OUT, "pose.json"))
            rows["after"]["wrist_moved_mm"] = round(float(np.linalg.norm(o["P"]["wrist"] - S.forward(z0, False)["P"]["wrist"])), 2)
    bc = json.load(open(os.path.join(WORK, "blend_check.json")))
    met = {"before": rows["before"], "after": rows["after"],
           "definitions": {
               "palm_swell_gap_mm": "distance from the 'palm swell' marker (on the grip surface) to the nearest skin; - = skin over it",
               "max_penetration_into_grip_mm": "deepest skin point inside the grip (0.5 mm skin samples, contacts.json)",
               "fingertip_to_button_mm": ("marker centre to the nearest fingertip (distal segment) skin, - = marker inside the "
                                          "finger; R1's marker centre is 4.6 mm inside the plastic, so the R1 surface point is "
                                          "given too (index skin touching the grip at R1)"),
               "flipped_triangles": ("skin triangles whose normal points against the rest normal turned by their bone "
                                     "(fold-overs / inside-out skin) in the hold pose"),
               "skin_into_skin_max_mm": "deepest skin of one part (finger / palm) inside another (skin.resolve_contacts test, not corrected)",
               "index_to_thumb_leak": ("bend the index MCP 20 deg in the hold pose: how far the thumb phalanx skin moves "
                                       "(should be ~0), and the largest index weight on thumb phalanx skin"),
           },
           "blender_vs_numpy_after_mm": bc["live_check"]["blender_vs_numpy_same_bulge_mm"],
           "solve": {"file": a.solve, "summary": d["summary"], "nfev": d["nfev"]}}
    jdump(met, os.path.join(OUT, "metrics.json"))
    log("metrics written")


# ------------------------------------------------------------------ picture

FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FONT_B = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
# title, direction the camera looks FROM, up
SHEET_VIEWS = [("side (back of the hand)", (1.0, -0.15, 0.25), (0, 0, 1), "glass"),
               ("front (trigger side)", (0.25, 1.0, 0.2), (0, 0, 1), "solid"),
               ("top, cut at palm height (close-up)", (0.0, 0.0, 1.0), (0, 1, 0), "slice")]
RED_MM = 1.0       # skin deeper than this inside the plastic is painted red
SHEET_RES = (960, 800)


def inside_depth(Vg, Fg, V):
    """Depth (mm) of each point inside the closed grip (0 outside), nearest-face-normal sign."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    bvh = BVHTree.FromPolygons(Vg.tolist(), Fg.tolist())
    d = np.zeros(len(V))
    for i, p in enumerate(V):
        loc, nrm, _, dist = bvh.find_nearest(Vector(p), 20.0)
        if loc is not None and (p - np.array(loc)) @ np.array(nrm) < 0:
            d[i] = dist
    return d


def problem_spots(S, V, fl, dep, met):
    """3D points to ring: (key, point, is_problem)."""
    sp = []
    i = int(np.argmax(dep))
    sp.append(("pen", V[i], dep[i] > 1.0))
    sp.append(("swell", S.markers["palm swell"], met["palm_swell_gap_mm"] > 1.0))
    p1 = project_to_surface(S, S.markers["R1 target"])
    fb = met["fingertip_to_button_mm"]
    sp.append(("R1", p1, fb["R1_surface_point_to_index_finger_mm"] > 1.0))
    sp.append(("R2", S.markers["R2 press"], abs(fb["R2_middle"]) > 1.0))
    sp.append(("R4", S.markers["R4 press"], abs(fb["R4_ring"]) > 1.0))
    idx = np.flatnonzero(fl)
    if len(idx):
        from scipy.cluster.hierarchy import fcluster, linkage
        C = V[S.F[idx]].mean(axis=1)
        cl = fcluster(linkage(C, "single"), 4.0, "distance") if len(C) > 1 else np.array([1])
        big = np.bincount(cl).argmax()
        n = int((cl == big).sum())
        sp.append(("flip", C[cl == big].mean(axis=0), n >= 5))
    return sp


def cmd_render(a):
    """Render before / after on the whole grip (3 views each) + ring positions -> WORK/sheet/."""
    import bpy
    from mathutils import Matrix, Vector
    from bpy_extras.object_utils import world_to_camera_view
    met = json.load(open(os.path.join(OUT, "metrics.json")))
    d = json.load(open(a.solve))
    out = os.path.join(WORK, "sheet")
    os.makedirs(out, exist_ok=True)
    hands = {}
    for which, params, z in (("before", PARAMS, None), ("after", d["params"], d["z"])):
        S, z0 = setup(params)
        z = z0 if z is None else np.array(z)
        o = S.forward(z)
        fl, _ = S.flips(o)
        dep = inside_depth(S.Vg, S.Fg, o["V"])
        hands[which] = dict(V=o["V"], dep=dep, spots=problem_spots(S, o["V"], fl, dep, met[which]))
        log(which, "max depth (vertices)", round(float(dep.max()), 2))
    F, Vg, Fg = S.F, S.Vg, S.Fg

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene

    def mk(name, V, Ff, col=None):
        me = bpy.data.meshes.new(name)
        me.from_pydata(V.tolist(), [], Ff.tolist())
        for p in me.polygons:
            p.use_smooth = True
        if col is not None:
            ca = me.color_attributes.new("col", "FLOAT_COLOR", "POINT")
            ca.data.foreach_set("color", col.ravel().tolist())
        ob = bpy.data.objects.new(name, me)
        sc.collection.objects.link(ob)
        return ob

    def mat(name, rgb, alpha=1.0, attr=False, rough=0.5):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (*rgb, 1)
        b.inputs["Roughness"].default_value = rough
        b.inputs["Alpha"].default_value = alpha
        if attr:
            at = m.node_tree.nodes.new("ShaderNodeAttribute")
            at.attribute_name = "col"
            m.node_tree.links.new(at.outputs["Color"], b.inputs["Base Color"])
            m.node_tree.links.new(at.outputs["Alpha"], b.inputs["Alpha"])
        return m
    # per view: 'glass' = see-through skin (skin inside the plastic shows red through it),
    # 'solid' = plain opaque skin, 'slice' = everything cut away above the palm swell's height
    zcut = float(S.markers["palm swell"][2]) + 2.0

    def cut(V, Ff):
        keep = (V[Ff][:, :, 2] < zcut).all(axis=1)
        return Ff[keep]
    objs = {}       # (mode, which) -> objects; which = 'grip' | 'before' | 'after'
    gm = mat("grip", (0.16, 0.18, 0.22), 0.5)
    for mode in ("glass", "solid", "slice"):
        g = mk(f"grip {mode}", Vg, cut(Vg, Fg) if mode == "slice" else Fg)
        g.data.materials.append(gm)
        objs[(mode, "grip")] = g
        for which, h in hands.items():
            red = h["dep"] > RED_MM
            col = np.where(red[:, None], [[0.9, 0.02, 0.02, 1.0]], [[0.80, 0.55, 0.42, 0.42 if mode == "glass" else 1.0]])
            ob = mk(f"hand {which} {mode}", h["V"], cut(h["V"], F) if mode == "slice" else F, col)
            ob.data.materials.append(mat(f"skin {which} {mode}", (0.8, 0.55, 0.42), 1.0, attr=True))
            objs[(mode, which)] = ob
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = a.samples
    sc.cycles.transparent_max_bounces = 16
    try:
        sc.cycles.use_denoising = True
    except Exception:
        pass
    sc.render.resolution_x, sc.render.resolution_y = SHEET_RES
    sc.view_settings.view_transform = "Standard"
    w = bpy.data.worlds.new("w")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.97, 0.97, 0.98, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.8
    for name, e, en in (("key", (0.6, -0.3, 0.9), 2.6), ("fill", (-0.7, -0.5, 0.4), 1.0), ("rim", (0.2, 0.9, -0.2), 0.8)):
        ld = bpy.data.lights.new(name, "SUN")
        ld.energy = en
        lo = bpy.data.objects.new(name, ld)
        sc.collection.objects.link(lo)
        lo.rotation_euler = Vector(e).to_track_quat("Z", "Y").to_euler()
    cam_d = bpy.data.cameras.new("cam")
    cam_d.type = "ORTHO"
    cam_d.clip_start, cam_d.clip_end = 1, 5000
    cam = bpy.data.objects.new("cam", cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam
    allp = np.concatenate([Vg, hands["before"]["V"], hands["after"]["V"]])
    ctr = (allp.min(0) + allp.max(0)) / 2
    aspect = SHEET_RES[0] / SHEET_RES[1]
    meta = {}
    for vi, (title, dv, up, mode) in enumerate(SHEET_VIEWS):
        dv = np.array(dv, float) / np.linalg.norm(dv)
        up = np.array(up, float)
        up = up - (up @ dv) * dv
        up /= np.linalg.norm(up)
        right = np.cross(up, dv)
        pts, cc = allp, ctr
        if mode == "slice":      # zoom on the palm swell: the gap is a few mm
            sw = S.markers["palm swell"]
            pts = allp[(np.linalg.norm((allp - sw)[:, :2], axis=1) < 38.0) & (allp[:, 2] < zcut)]
            cc = (pts.min(0) + pts.max(0)) / 2
        Q = pts - cc
        qr, qu = Q @ right, Q @ up
        cam_d.ortho_scale = 1.08 * max(np.ptp(qr), np.ptp(qu) * aspect)
        c2 = cc + right * 0.5 * (qr.max() + qr.min()) + up * 0.5 * (qu.max() + qu.min())
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = np.column_stack([right, up, dv]), c2 + dv * 800
        cam.matrix_world = Matrix(M.tolist())
        bpy.context.view_layer.update()
        for which in hands:
            for (md, k), ob in objs.items():
                ob.hide_render = not (md == mode and k in ("grip", which))
            p = os.path.join(out, f"{which}_{vi}.png")
            sc.render.filepath = p
            t0 = time.time()
            bpy.ops.render.render(write_still=True)
            log("rendered", p, f"{time.time() - t0:.0f} s")

            def px(q):
                c = world_to_camera_view(sc, cam, Vector(q))
                return [float(c.x * SHEET_RES[0]), float((1 - c.y) * SHEET_RES[1])]
            meta[f"{which}_{vi}"] = {"view": title, "mm_per_px": cam_d.ortho_scale / SHEET_RES[0],
                                     "spots": {k: {"px": px(q), "problem": bool(bad)}
                                               for k, q, bad in hands[which]["spots"]}}
    jdump(meta, os.path.join(out, "meta.json"))


# which spots are ringed in which view (index into SHEET_VIEWS)
SPOT_VIEW = {"pen": 0, "swell": 2, "flip": 0, "R1": 1, "R2": 1, "R4": 1}


def captions(met):
    b, a = met["before"], met["after"]
    fb, fa = b["fingertip_to_button_mm"], a["fingertip_to_button_mm"]

    def btn(f):
        return (f"index {f['R1_surface_point_to_index_finger_mm']:.1f} mm from R1, middle "
                f"{abs(f['R2_middle']):.1f} mm from R2, ring {abs(f['R4_ring']):.1f} mm from R4")
    return {
        "before": [f"Skin goes {b['max_penetration_into_grip_mm']:.1f} mm into the plastic (red); "
                   f"{b['flipped_triangles']} skin triangles flipped",
                   f"Fingertips off the buttons: {btn(fb)}",
                   f"Palm stands {b['palm_swell_gap_mm']:.1f} mm off the grip's palm swell"],
        "after": [f"Deepest skin in the plastic {a['max_penetration_into_grip_mm']:.1f} mm; "
                  f"{a['flipped_triangles']} tiny flipped triangles left",
                  f"On the buttons: {btn(fa)}",
                  f"Palm rests on the palm swell: gap {a['palm_swell_gap_mm']:.1f} mm"],
    }


def cmd_sheet(a):
    from PIL import Image, ImageDraw, ImageFont
    met = json.load(open(os.path.join(OUT, "metrics.json")))
    wd = os.path.join(WORK, "sheet")
    meta = json.load(open(os.path.join(wd, "meta.json")))
    cap = captions(met)
    f_title = ImageFont.truetype(FONT_B, 54)
    f_row = ImageFont.truetype(FONT_B, 46)
    f_view = ImageFont.truetype(FONT_B, 34)
    f_cap = ImageFont.truetype(FONT, 30)
    f_leg = ImageFont.truetype(FONT, 30)
    pw = 800
    sc_ = pw / SHEET_RES[0]
    ph = int(SHEET_RES[1] * sc_)
    pad, top, rowh, viewh, caph = 24, 100, 70, 50, 84
    W = 3 * pw + 4 * pad
    H = top + 2 * (rowh + viewh + ph + caph) + 150
    sh = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(sh)
    dr.text((pad, 22), "Your hand on the grip: before and after re-placing it", fill=(15, 15, 15), font=f_title)
    rows = [("before", "BEFORE: old placement (copied from the cylinder hand)", (170, 20, 20)),
            ("after", "AFTER: re-placed so the real skin sits on the grip", (20, 120, 40))]
    y = top
    for which, label, colr in rows:
        dr.text((pad, y + 8), label, fill=colr, font=f_row)
        y += rowh
        for vi, (title, _, _, _) in enumerate(SHEET_VIEWS):
            x = pad + vi * (pw + pad)
            dr.text((x, y + 6), "seen from the " + title, fill=(60, 60, 60), font=f_view)
            im = Image.open(os.path.join(wd, f"{which}_{vi}.png")).convert("RGB").resize((pw, ph), Image.LANCZOS)
            di = ImageDraw.Draw(im)
            for k, sp in meta[f"{which}_{vi}"]["spots"].items():
                if SPOT_VIEW.get(k) != vi:
                    continue
                if which == "after" and not sp["problem"] and not meta[f"before_{vi}"]["spots"].get(k, {}).get("problem"):
                    continue
                if which == "before" and not sp["problem"]:
                    continue
                cx, cy = sp["px"][0] * sc_, sp["px"][1] * sc_
                r = 38
                col = (225, 0, 0) if sp["problem"] else (0, 170, 60)
                di.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=7)
            sh.paste(im, (x, y + viewh))
            dr.rectangle([x, y + viewh, x + pw - 1, y + viewh + ph - 1], outline=(200, 200, 200), width=2)
            txt = cap[which][vi]
            ft = f_cap
            while dr.textlength(txt, font=ft) > pw and ft.size > 28:
                ft = ImageFont.truetype(FONT, ft.size - 1)
            lines = [txt]
            if dr.textlength(txt, font=ft) > pw:      # wrap onto two lines
                words, lines = txt.split(), [""]
                for wd_ in words:
                    t = (lines[-1] + " " + wd_).strip()
                    if dr.textlength(t, font=ft) <= pw:
                        lines[-1] = t
                    else:
                        lines.append(wd_)
            for li, t in enumerate(lines[:2]):
                dr.text((x, y + viewh + ph + 8 + li * 36), t, fill=(30, 30, 30), font=ft)
        y += viewh + ph + caph
    leg = ["Red ring = problem spot.  Green ring = the same spot after re-placing (now fine).  "
           "Red skin = more than 1 mm inside the plastic.",
           "Grey = the grip (see-through).  Side view: the skin is see-through too.  Top view: hand and grip "
           "cut flat at palm height like a slice, close-up, so the palm-to-grip gap shows."]
    for i, t in enumerate(leg):
        dr.text((pad, H - 130 + i * 44), t, fill=(40, 40, 40), font=f_leg)
    out = os.path.join(OUT, "hold_v2_sheet.jpg")
    sh.save(out, quality=90)
    log("WROTE", out, sh.size)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("solve")
    s.add_argument("--params", default=None)
    s.add_argument("--tag", default="v2")
    s.add_argument("--start", default=None, help="a previous solve_<tag>.json to continue from")
    s.add_argument("--nfev", type=int, default=40)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--w", default=None, help="objective weights, json (see Objective)")
    e = sp.add_parser("export")
    e.add_argument("--tag", default="v2")
    b = sp.add_parser("blend")
    b.add_argument("--solve", required=True)
    b.add_argument("--out-blend", required=True)
    c2 = sp.add_parser("contacts")
    c2.add_argument("--blend", required=True)
    c2.add_argument("--out-json", required=True)
    m = sp.add_parser("metrics")
    m.add_argument("--solve", required=True)
    r = sp.add_parser("render")
    r.add_argument("--solve", required=True)
    r.add_argument("--samples", type=int, default=24)
    sp.add_parser("sheet")
    c = sp.add_parser("skincheck")
    c.add_argument("--params", default=None)
    c.add_argument("--solve", default=None)
    c.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
