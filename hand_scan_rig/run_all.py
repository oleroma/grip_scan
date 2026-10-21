"""One command from a hand scan to the hand holding the grip, with pictures.

    P=/tmp/claude-0/hs/venv/bin/python
    $P run_all.py SCAN.ply|obj|glb [--out DIR] [--units auto|mm|cm|m|in]
    $P run_all.py --demo            # the stand-in hand, rotated/moved/rescaled first (no cheating)

Steps (each writes its files into --out; a step is skipped when its output already exists,
unless --force; --steps picks a subset):
  fit       fit_rig.py: clean the scan, find the joints, build the armature   -> fit/
  skin      skin.py: weight maps, fold shape keys, Armature + Corrective Smooth modifiers
  hold      pose the scan rig into Olex's HOLD pose (the rest pose of 'hand rig' in
            hold_rig.blend): same joint angles, then the whole hand moved/rotated so its
            joints land on the cylinder rig's joints (least squares over the 21 joints);
            builds hand_on_grip_scan.blend (grip + markers, old cylinder hand hidden in
            collection 'old cylinder hand', new skinned hand live in Blender modifiers)
  contacts  skin vs grip for both hands: penetration, palm swell gap, R1/R2/R4 distances
  sheet     demo_sheet.jpg: outside / front / top views, old hand row vs new hand row

Conventions: 1 unit = 1 mm, right hand.  Angles as skin.py / SPEC (flexion + = toward the
palm, thumb too).  The posed hand stays live: shape-key drivers read each bone's local X
rotation (the flexion axis, + toward the palm) so the pad bulges follow when Olex rotates
bones in Pose Mode (bones are XYZ Euler, X = flexion); the armature object carries the
placement on the grip, the pose bones carry only the joint angles (Alt+R = open scan pose).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PY = sys.executable
HOLD_RIG = "/tmp/claude-0/hs/hold_rig.blend"
STANDIN = "/mnt/project-files/hand_scan_rig/standin"
DEMO_OUT = "/mnt/project-files/hand_scan_rig/demo"
PARAMS = "/mnt/project-files/hand_scan_rig/skin/skin_params.json"
FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FONT_B = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
GRIP = "grip (current GripV)"
MARKERS = ["R1 target", "R2 press", "R4 press", "palm swell"]
OLD_COLL = "old cylinder hand"
NEW_COLL = "new hand from scan"
GRIP_COLL = "grip"
TIP_TARGET = {"R1 target": "index", "R2 press": "middle", "R4 press": "ring"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))


# ------------------------------------------------------------------ demo input

def make_demo_scan(out):
    """The stand-in, moved like a real scan would come in: random rotation, offset, metres."""
    import trimesh
    from scipy.spatial.transform import Rotation
    d = os.path.join(out, "input")
    os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "scan_demo.ply")) and os.path.exists(os.path.join(d, "disguise.json")):
        return os.path.join(d, "scan_demo.ply")
    m = trimesh.load(os.path.join(STANDIN, "standin_open.ply"), process=False)
    R = Rotation.random(random_state=2026).as_matrix()
    t = np.array([123.0, -48.0, 310.0])
    s = 0.001                                   # the "scan" is in metres
    V = (np.asarray(m.vertices) @ R.T + t) * s
    path = os.path.join(d, "scan_demo.ply")
    trimesh.Trimesh(V, m.faces, process=False).export(path)
    jdump({"R": R, "t_mm": t, "scale": s, "note": "scan = (standin @ R.T + t) * scale; used only to "
           "grade the fitted joints afterwards, never by the pipeline"}, os.path.join(d, "disguise.json"))
    log("demo scan", path)
    return path


# ------------------------------------------------------------------ step: fit

def step_fit(a):
    fit_dir = os.path.join(a.out, "fit")
    if os.path.exists(os.path.join(fit_dir, "hand_rigged.blend")) and not a.force:
        log("fit: exists, skipped")
        return
    cmd = [PY, os.path.join(HERE, "fit_rig.py"), a.scan, "--out", fit_dir, "--units", a.units, "--name", "scan"]
    log("fit:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ------------------------------------------------------------------ hold rig reading

def read_hold(hold_rig):
    """Landmarks of the cylinder rig's rest (= hold) pose, grip-frame mm."""
    import bpy
    from skin import _landmarks_from_armature
    bpy.ops.wm.open_mainfile(filepath=hold_rig)
    L = _landmarks_from_armature(bpy.data.objects["hand rig"])
    return {k: np.asarray(v, float) for k, v in L.items()}


def kabsch(P, Q, w=None):
    """R, t minimising sum w |R P + t - Q|^2."""
    w = np.ones(len(P)) if w is None else np.asarray(w, float)
    pc, qc = (w[:, None] * P).sum(0) / w.sum(), (w[:, None] * Q).sum(0) / w.sum()
    H = ((P - pc) * w[:, None]).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def refine_on_joints(rig, x0, R0, t0, Hs, names, prior):
    """Pose DOFs (+ thumb metacarpal/proximal twist) and rigid placement so the posed joints land
    on the cylinder rig's joints.  prior: mm of residual per degree away from the angle solution."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    w0 = rig.L["wrist"]

    def unpack(z):
        x = z[:len(x0) + 2]
        Rr = Rotation.from_rotvec(z[-6:-3]).as_matrix() @ R0
        return x, Rr, z[-3:]

    def res(z):
        x, Rr, tt = unpack(z)
        _, P = rig.fk(x[:-2], {"thumb_metacarpal": x[-2], "thumb_proximal": x[-1]})
        Q = np.array([P[n] for n in names]) @ Rr.T + tt
        return np.r_[(Q - Hs).ravel(), prior * (x[:len(x0)] - x0), prior * x[-2:]]
    z0 = np.r_[x0, 0.0, 0.0, 0.0, 0.0, 0.0, t0]
    sol = least_squares(res, z0, method="trf", x_scale="jac")
    del w0
    return unpack(sol.x)


# ------------------------------------------------------------------ step: skin + hold scene

def append_from_hold(hold_rig):
    """Grip, markers and the old cylinder hand (with its armature) from hold_rig.blend."""
    import bpy
    with bpy.data.libraries.load(hold_rig, link=False) as (src, dst):
        dst.objects = list(src.objects)
    sc = bpy.context.scene

    def coll(name):
        c = bpy.data.collections.get(name) or bpy.data.collections.new(name)
        if c.name not in sc.collection.children:
            sc.collection.children.link(c)
        return c
    cg, co = coll(GRIP_COLL), coll(OLD_COLL)
    for o in dst.objects:
        if o is None:
            continue
        if o.name == GRIP or o.name in MARKERS:
            cg.objects.link(o)
        else:
            co.objects.link(o)
            if o.type == "ARMATURE":
                o.name = "old cylinder hand rig"
                o.data.name = "old cylinder hand rig"
    # hidden but one click away (eye icon of the collection)
    co.hide_render = True
    lc = bpy.context.view_layer.layer_collection.children[OLD_COLL]
    lc.hide_viewport = True
    for o in co.objects:
        o.hide_set(True)
    return cg, co


def write_hold_blend(sk, M, R, t, pose_angles, hold_rig, out_blend):
    """Pose the open Skinner's rig in Blender (bone transforms M, armature-local), place the
    armature object on the grip (R, t), add grip + markers + the hidden old cylinder hand from
    hold_rig, save out_blend.  Returns (Blender-evaluated skin verts, numpy reference with the
    same live bulge values, live-vs-numpy check)."""
    import bpy
    from mathutils import Matrix
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    # bones (relative to the scan's rest), then the armature object carries the placement
    bv_meas = sk.bulge_values(M=M)       # same hinge measure the live drivers read
    sk._pose_blender(M, bv_meas)
    arm, ob = sk.arm, sk.ob
    arm.matrix_world = Matrix(T.tolist())
    drivers = list(getattr(sk, "drivers", []))   # added by Skinner.apply_to_blend

    # material for the new skin
    mat = bpy.data.materials.new("scan skin")
    mat.diffuse_color = (0.86, 0.62, 0.50, 1)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.80, 0.52, 0.40, 1)
        bsdf.inputs["Roughness"].default_value = 0.55
    ob.data.materials.clear()
    ob.data.materials.append(mat)

    # collections: new hand, grip, old hand
    sc = bpy.context.scene
    cn = bpy.data.collections.new(NEW_COLL)
    sc.collection.children.link(cn)
    for o in (ob, arm):
        for c in list(o.users_collection):
            c.objects.unlink(o)
        cn.objects.link(o)
    append_from_hold(hold_rig)
    sc.unit_settings.system, sc.unit_settings.scale_length = "METRIC", 0.001
    sc.unit_settings.length_unit = "MILLIMETERS"
    arm.data.display_type = "STICK"
    arm.show_in_front = True
    arm["hold_pose"] = json.dumps(pose_angles)
    arm["how_to"] = ("Select 'hand rig', Ctrl+Tab for Pose Mode, rotate bones (R, then X X for "
                     "flexion: + curls toward the palm). Skin follows through the Armature + "
                     "CorrectiveSmooth modifiers and the bulge shape keys (drivers).")

    # --- evaluate the live Blender result and check it against the numpy backend
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    bv_live = {kb.name[6:]: float(kb.value) for kb in ob.evaluated_get(dg).data.shape_keys.key_blocks[1:]} \
        if ob.data.shape_keys else {}
    me = ob.evaluated_get(dg).to_mesh()
    Vb = np.zeros(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", Vb)
    ob.evaluated_get(dg).to_mesh_clear()
    mw = np.array(ob.matrix_world)
    Vb = Vb.reshape(-1, 3).astype(float) @ mw[:3, :3].T + mw[:3, 3]
    # numpy reference in the armature's own space (as Blender deforms), then placed
    Vn_meas = sk._pose_numpy(M, bv_meas) @ R.T + t
    Vn_live = sk._pose_numpy(M, bv_live) @ R.T + t
    check = {
        "blender_vs_numpy_same_bulge_mm": {"max": float(np.abs(Vb - Vn_live).max()),
                                           "mean": float(np.linalg.norm(Vb - Vn_live, axis=1).mean())},
        "driver_bulge_vs_measured_bulge_mm": {"max": float(np.linalg.norm(Vn_live - Vn_meas, axis=1).max())},
        "bulge_live": bv_live, "bulge_measured": bv_meas,
        "drivers": [{"key": k, "bone": b, "simple_expression": s} for k, b, s in drivers],
    }
    # make the rig the active object, in Pose Mode
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    try:
        bpy.ops.object.mode_set(mode="POSE")
    except Exception as e:  # headless context; Olex presses Ctrl+Tab
        log("pose mode not set:", e)
    os.makedirs(os.path.dirname(os.path.abspath(out_blend)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    return Vb, Vn_live, check


def step_hold(a):
    import skin
    from skin import MEAS, NAMES, Skinner, measurements, load_params
    out_blend = os.path.join(a.out, "hand_on_grip_scan.blend")
    rep_path = os.path.join(a.out, "hold_placement.json")
    if os.path.exists(out_blend) and os.path.exists(rep_path) and not a.force:
        log("hold: exists, skipped")
        return
    fit_blend = os.path.join(a.out, "fit", "hand_rigged.blend")
    t0 = time.time()
    H = read_hold(a.hold_rig)
    hold_angles = measurements(H)
    # thumb_cmc_azimuth (signed, palm plane) makes the thumb branch unique; the SPEC's unsigned
    # thumb_cmc alone has a mirror solution (see skin.Rig.solve)
    pkeys = list(MEAS) + [k for k in ("thumb_cmc_azimuth",) if k in hold_angles]
    pose = {"angles": {k: float(hold_angles[k]) for k in pkeys}}
    jdump({**pose, "source": f"rest pose of 'hand rig' in {a.hold_rig}"}, os.path.join(a.out, "hold_pose.json"))

    # --- skin: parametric weights, bulge keys, modifiers on the fitted rig
    sk = Skinner(fit_blend, a.params)
    # the skin weights give the palm flesh to the 'palm' bone (fit_rig makes it a non-deforming
    # handle); it must deform or Blender renormalises those vertices onto the metacarpals
    sk.arm.data.bones["palm"].use_deform = True
    sk.apply_to_blend()
    log(f"skin applied ({time.time() - t0:.1f} s): {len(sk.V0)} verts, params {a.params}")
    params = sk.params

    # --- hold angles on the scan rig, then the placement on the grip
    names = list(NAMES)
    Hs = np.array([H[n] for n in names])
    M0, P0, info = sk.rig.solve(pose)
    R0, t0_ = kabsch(np.array([P0[n] for n in names]), Hs)
    diff0 = {n: float(np.linalg.norm(R0 @ P0[n] + t0_ - H[n])) for n in names}
    x, R, t = refine_on_joints(sk.rig, info["x"], R0, t0_, Hs, names, a.angle_prior)
    twist = {"thumb_metacarpal": float(x[-2]), "thumb_proximal": float(x[-1])}
    M, P = sk.rig.fk(x[:-2], twist)
    diff = {n: float(np.linalg.norm(R @ P[n] + t - H[n])) for n in names}
    got = measurements({n: R @ P[n] + t for n in names})
    ang_err = {k: float(got[k] - hold_angles[k]) for k in pkeys}

    Vb, Vn_live, check = write_hold_blend(sk, M, R, t, pose["angles"], a.hold_rig, out_blend)
    np.save(os.path.join(a.out, "hand_hold_verts.npy"), Vb)
    # skin-on-skin contact in the hold pose (fingers on fingers / palm).  The live .blend has no
    # contact pass (Blender modifiers only); measure what it would correct.
    contact_rep = {}
    try:
        from skin import contact_labels, resolve_contacts
        Vc, cinf = resolve_contacts(Vn_live, sk.F, sk.V0, sk.cs.A, labels=contact_labels(sk.W),
                                    rest_sep_mm=params["global"].get("contact_rest_sep_mm", 8.0),
                                    bulge=params["global"].get("contact_bulge", 0.3))
        contact_rep = {"penetration_depth_max_mm": cinf["contact_depth_max_mm"],
                       "vertices_in_contact": cinf["contact_vertices"],
                       "max_correction_mm": float(np.linalg.norm(Vc - Vn_live, axis=1).max()),
                       "note": "not applied to the live .blend (Pose Mode has no contact pass)"}
    except Exception as e:  # report only
        contact_rep = {"error": repr(e)}
    rep = {
        "blend": out_blend,
        "placement": {"R": R, "t_mm": t, "how": "1) hold angles set on the scan rig (relaxed_pose "
                      "definitions + palm arch + thumb CMC elevation), 2) rigid fit over the 21 joints, "
                      "3) joint-position least squares over the 22 pose DOFs + thumb axial twists + "
                      "the rigid placement, with a weak pull to the hold angles"},
        "angles_only_then_rigid_joint_diff_mm": {"median": float(np.median(list(diff0.values()))),
                                                  "max": float(max(diff0.values())),
                                                  "worst": max(diff0, key=diff0.get),
                                                  "per_joint": {k: round(v, 2) for k, v in diff0.items()}},
        "thumb_twist_deg": twist,
        "joint_diff_mm": {k: round(v, 2) for k, v in diff.items()},
        "joint_diff_summary_mm": {"median": float(np.median(list(diff.values()))),
                                  "max": float(max(diff.values())),
                                  "worst": max(diff, key=diff.get),
                                  "mcp_mean": float(np.mean([diff[f + "_mcp"] for f in skin.F4])),
                                  "tips_mean": float(np.mean([diff[f + "_tip"] for f in ["thumb"] + skin.F4]))},
        "angle_err_deg_max": float(max(abs(v) for v in ang_err.values())),
        "angle_err_deg": {k: round(v, 3) for k, v in ang_err.items()},
        "angle_prior_mm_per_deg": a.angle_prior,
        "segment_lengths_scan_vs_cylinder_mm": {
            b: [round(float(np.linalg.norm(sk.rig.tail[b] - sk.rig.head[b])), 2),
                round(float(np.linalg.norm(H[skin.HEADTAIL[b][1]] - H[skin.HEADTAIL[b][0]])), 2)]
            for b in skin.BONES[1:]},
        "live_check": check,
        "solve_warnings": info.get("warnings", []),
        "skin_contact_in_hold": contact_rep,
        "seconds": time.time() - t0,
    }
    jdump(rep, rep_path)
    log(f"hold: joints median {rep['joint_diff_summary_mm']['median']:.2f} max "
        f"{rep['joint_diff_summary_mm']['max']:.2f} mm ({rep['joint_diff_summary_mm']['worst']}); "
        f"blender vs numpy {check['blender_vs_numpy_same_bulge_mm']['max']:.4f} mm; "
        f"driver bulge effect {check['driver_bulge_vs_measured_bulge_mm']['max']:.3f} mm")


# ------------------------------------------------------------------ step: contacts

def winding(Q, V, F, chunk=400):
    """Generalised winding number of points Q w.r.t. a closed triangle mesh (1 inside, 0 out)."""
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    w = np.zeros(len(Q))
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk, None, :]
        a, b, c = A[None] - q, B[None] - q, C[None] - q
        la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
        det = np.einsum("ijk,ijk->ij", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("ijk,ijk->ij", a, b) * lc + np.einsum("ijk,ijk->ij", b, c) * la
               + np.einsum("ijk,ijk->ij", c, a) * lb)
        w[s:s + chunk] = np.arctan2(det, den).sum(1) / (2 * np.pi)
    return w


class Surface:
    """Closed triangle mesh: exact nearest distance (Blender BVH) + winding-number sign."""

    def __init__(self, V, F):
        from mathutils.bvhtree import BVHTree
        self.V, self.F = np.asarray(V, float), np.asarray(F, np.int64)
        self.bvh = BVHTree.FromPolygons(self.V.tolist(), self.F.tolist())

    def dist(self, Q):
        d = np.empty(len(Q))
        for i, q in enumerate(Q):
            r = self.bvh.find_nearest(q)
            d[i] = r[3] if r[0] is not None else np.inf
        return d

    def signed(self, Q, near=10.0):
        d = self.dist(Q)
        s = np.ones(len(Q))
        idx = np.nonzero(d < near)[0]
        if len(idx):
            s[idx] = np.where(winding(Q[idx], self.V, self.F) > 0.5, -1.0, 1.0)
        return s * d


def sample_surface(V, F, spacing, seed=0):
    rng = np.random.default_rng(seed)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = np.linalg.norm(np.cross(B - A, C - A), axis=1) / 2
    n = int(area.sum() / spacing ** 2)
    fi = rng.choice(len(F), n, p=area / area.sum())
    u, v = rng.random((2, n))
    fl = u + v > 1
    u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
    return A[fi] + u[:, None] * (B - A)[fi] + v[:, None] * (C - A)[fi], fi, float(area.sum()) / n


def world_mesh(o, dg=None):
    me = (o.evaluated_get(dg).to_mesh() if dg else o.data)
    me.calc_loop_triangles()
    V = np.array([o.matrix_world @ v.co for v in me.vertices], float)
    F = np.array([t.vertices[:] for t in me.loop_triangles], np.int64)
    if dg:
        o.evaluated_get(dg).to_mesh_clear()
    return V, F


def hand_contacts(parts, grip, markers, spacing):
    """parts: list of (bone, V, F); a closed skin made of one or more closed pieces.
    Skin samples hidden inside another piece are dropped (only the outer skin counts)."""
    S, lab, cell = [], [], None
    for i, (b, V, F) in enumerate(parts):
        P, fi, cell_i = sample_surface(V, F, spacing, seed=i)
        cell = cell_i if cell is None else cell
        if isinstance(b, np.ndarray):
            labels = b[F[fi, 0]]
        else:
            labels = np.full(len(P), b, object)
        keep = np.ones(len(P), bool)
        if len(parts) > 1:
            for j, (_, V2, F2) in enumerate(parts):
                if j != i:
                    lo, hi = V2.min(0) - 1, V2.max(0) + 1
                    box = np.all((P > lo) & (P < hi), axis=1) & keep
                    if box.any():
                        keep[np.nonzero(box)[0][winding(P[box], V2, F2) > 0.5]] = False
        S.append(P[keep])
        lab.append(labels[keep])
    S, lab = np.concatenate(S), np.concatenate(lab)
    t0 = time.time()
    sd = grip.signed(S)
    log(f"  {len(S)} skin samples, grip distance {time.time() - t0:.0f} s")
    pen = -sd[sd < 0]
    res = {
        "skin_samples": int(len(S)), "sample_spacing_mm": spacing,
        "penetration_max_mm": float(pen.max()) if len(pen) else 0.0,
        "penetration_p95_mm": float(np.percentile(pen, 95)) if len(pen) else 0.0,
        "penetration_median_mm": float(np.median(pen)) if len(pen) else 0.0,
        "penetrating_area_cm2": float(len(pen) * spacing ** 2 / 100.0),
        "deepest_at": str(lab[sd < 0][np.argmax(pen)]) if len(pen) else None,
    }
    by = {}
    for b in np.unique(lab[sd < 0]) if len(pen) else []:
        m = (lab == b) & (sd < 0)
        by[str(b)] = round(float((-sd[m]).max()), 2)
    res["penetration_max_by_part_mm"] = dict(sorted(by.items(), key=lambda kv: -kv[1]))

    def inside_hand(q):
        return any(winding(q[None], V, F)[0] > 0.5 for _, V, F in parts)

    from scipy.spatial import cKDTree
    tree = cKDTree(S)
    for name, p in markers.items():
        if name == "palm swell":
            d, _ = tree.query(p)
            sgn = -1 if inside_hand(p) else 1
            res["palm_swell_gap_mm"] = round(float(sgn * d), 2)
            res["palm_swell_on_grip_surface_mm"] = round(float(grip.signed(p[None])[0]), 2)
        else:
            f = TIP_TARGET[name]
            for seg, key in ((f + "_distal", "fingertip"), (f + "_", "finger")):
                m = np.array([str(x).startswith(seg) for x in lab])
                d = float(np.linalg.norm(S[m] - p, axis=1).min())
                sgn = -1 if inside_hand(p) else 1
                res[f"{name.split()[0]}_{f}_{key}_mm"] = round(sgn * d, 2)
    return res


def step_contacts(a):
    out = os.path.join(a.out, "contacts.json")
    if os.path.exists(out) and not a.force:
        log("contacts: exists, skipped")
        return
    res = contacts_of_blend(os.path.join(a.out, "hand_on_grip_scan.blend"), a.spacing)
    jdump(res, out)
    for k in ("old_cylinder_hand", "new_scan_hand"):
        r = res[k]
        log(f"{k}: pen max {r['penetration_max_mm']:.2f} p95 {r['penetration_p95_mm']:.2f} mm "
            f"area {r['penetrating_area_cm2']:.1f} cm2; swell gap {r['palm_swell_gap_mm']}; "
            f"R1 {r['R1_index_fingertip_mm']} R2 {r['R2_middle_fingertip_mm']} R4 {r['R4_ring_fingertip_mm']}")


def contacts_of_blend(blend, spacing, old=True):
    """contacts.json content for a hand_on_grip .blend (old cylinder hand too unless old=False)."""
    import bpy
    t0 = time.time()
    bpy.ops.wm.open_mainfile(filepath=blend)
    dg = bpy.context.evaluated_depsgraph_get()
    grip = Surface(*world_mesh(bpy.data.objects[GRIP]))
    markers = {n: np.array(bpy.data.objects[n].matrix_world.translation) for n in MARKERS}
    # old: the cylinder pieces (each bone-parented, closed)
    old_parts = []
    for o in (bpy.data.collections[OLD_COLL].objects if old else []):
        if o.type == "MESH":
            V, F = world_mesh(o, dg)
            old_parts.append((o.parent_bone or o.name, V, F))
    # new: the evaluated scan skin; label each vertex by its strongest bone
    ob = bpy.data.objects["hand scan"]
    V, F = world_mesh(ob, dg)
    names = {vg.index: vg.name for vg in ob.vertex_groups}
    lab = np.empty(len(V), object)
    for v in ob.data.vertices:
        best = max(((g.weight, names[g.group]) for g in v.groups if names[g.group] != "cs_mask"),
                   default=(0, "palm"))
        lab[v.index] = best[1]
    res = {**({"old_cylinder_hand": hand_contacts(old_parts, grip, markers, spacing)} if old else {}),
           "new_scan_hand": hand_contacts([(lab, V, F)], grip, markers, spacing),
           "definitions": {
               "penetration": "skin sampled evenly (spacing^2 per sample); depth of each sample inside "
                              "the closed grip mesh; p95/median over the penetrating samples; area = "
                              "penetrating samples x spacing^2",
               "palm_swell_gap": "distance from the 'palm swell' marker (on the grip surface) to the "
                                 "nearest skin; negative = the marker is inside the hand (skin overlaps)",
               "R1/R2/R4": "distance from the marker centre to the nearest skin of the index / middle / "
                           "ring fingertip (distal segment) and of the whole finger; negative = inside",
           },
           "seconds": time.time() - t0}
    return res


# ------------------------------------------------------------------ step: sheet

VIEWS = [  # title, direction the camera looks FROM (unit), up vector
    ("outside (right side)", (1, 0, 0), (0, 0, 1)),
    ("front (trigger side)", (0, 1, 0), (0, 0, 1)),
    ("top", (0, 0, 1), (0, 1, 0)),
]


def render_views(a, wdir):
    import bpy
    from mathutils import Matrix, Vector
    bpy.ops.wm.open_mainfile(filepath=os.path.join(a.out, "hand_on_grip_scan.blend"))
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = a.samples
    try:
        sc.cycles.use_denoising = True
    except Exception:
        pass
    sc.render.resolution_x, sc.render.resolution_y = 900, 700
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    w = sc.world or bpy.data.worlds.new("w")
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.93, 0.93, 0.94, 1)
    bg.inputs[1].default_value = 0.6

    def mat(name, rgb, rough=0.6):
        m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        m.use_nodes = True
        b = m.node_tree.nodes.get("Principled BSDF")
        b.inputs["Base Color"].default_value = (*rgb, 1)
        b.inputs["Roughness"].default_value = rough
        return m
    skin_m = mat("sheet skin", (0.72, 0.42, 0.30), 0.5)
    grip_m = mat("sheet grip", (0.32, 0.34, 0.37), 0.45)
    cols = {"R1 target": (0.95, 0.75, 0.0), "R2 press": (0.9, 0.08, 0.08), "R4 press": (0.05, 0.65, 0.15),
            "palm swell": (0.1, 0.35, 1.0)}
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        if o.name == GRIP:
            m = grip_m
        elif o.name in cols:
            # the markers sit under the fingertips / inside the grip and would only show as
            # stray specks; the contact numbers are in contacts.json
            m = mat("sheet " + o.name, cols[o.name], 0.3)
            o.hide_render = True
        else:
            m = skin_m
        o.data.materials.clear()
        o.data.materials.append(m)
    for name, e in (("key", (0.6, 0.3, 0.9)), ("fill", (-0.7, -0.5, 0.4)), ("rim", (0.2, 0.9, -0.2))):
        ld = bpy.data.lights.new(name, "SUN")
        ld.energy = {"key": 2.4, "fill": 0.9, "rim": 0.8}[name]
        lo = bpy.data.objects.new(name, ld)
        sc.collection.objects.link(lo)
        lo.rotation_euler = Vector(e).to_track_quat("Z", "Y").to_euler()

    cam_d = bpy.data.cameras.new("sheet cam")
    cam_d.type = "ORTHO"
    cam_d.clip_start, cam_d.clip_end = 1, 5000
    cam = bpy.data.objects.new("sheet cam", cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam

    # framing: grip + both hands, the same for both rows
    dg = bpy.context.evaluated_depsgraph_get()
    pts = [world_mesh(bpy.data.objects[GRIP])[0]]
    for o in bpy.data.objects:
        if o.type == "MESH" and (o.name == "hand scan" or o.users_collection[0].name == OLD_COLL):
            pts.append(world_mesh(o, dg)[0])
    allp = np.concatenate(pts)
    lo, hi = allp.min(0), allp.max(0)
    ctr = (lo + hi) / 2
    aspect = sc.render.resolution_x / sc.render.resolution_y

    lc_old = bpy.context.view_layer.layer_collection.children[OLD_COLL]
    lc_new = bpy.context.view_layer.layer_collection.children[NEW_COLL]
    out = {}
    for row, show_old in (("old", True), ("new", False)):
        lc_old.hide_viewport, lc_new.hide_viewport = not show_old, show_old
        bpy.data.collections[OLD_COLL].hide_render = not show_old
        bpy.data.collections[NEW_COLL].hide_render = show_old
        for o in bpy.data.collections[OLD_COLL].objects:
            o.hide_set(not show_old)
            o.hide_render = not show_old
        for o in bpy.data.collections[NEW_COLL].objects:
            o.hide_render = show_old
        for vi, (title, d, up) in enumerate(VIEWS):
            d, up = np.array(d, float), np.array(up, float)
            right = np.cross(up, d)
            ext = hi - lo
            wspan, hspan = abs(ext @ right), abs(ext @ up)
            cam_d.ortho_scale = 1.08 * max(wspan, hspan * aspect)
            loc = ctr + d * 600
            Rm = np.column_stack([right, up, d])       # camera looks along -Z
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = Rm, loc
            cam.matrix_world = Matrix(M.tolist())
            p = os.path.join(wdir, f"{row}_{vi}.png")
            sc.render.filepath = p
            t0 = time.time()
            bpy.ops.render.render(write_still=True)
            log(f"render {row} {title}: {time.time() - t0:.1f} s")
            out[(row, vi)] = p
    return out


def step_sheet(a):
    out_jpg = os.path.join(a.out, "demo_sheet.jpg")
    if os.path.exists(out_jpg) and not a.force:
        log("sheet: exists, skipped")
        return
    wdir = os.path.join(a.out, "renders")
    os.makedirs(wdir, exist_ok=True)
    imgs = render_views(a, wdir)
    from PIL import Image, ImageDraw, ImageFont
    f_row, f_col, f_cap = ImageFont.truetype(FONT_B, 40), ImageFont.truetype(FONT, 32), ImageFont.truetype(FONT, 30)
    pw, ph = Image.open(imgs[("old", 0)]).size
    pad, head, rowt, cap = 16, 56, 62, 70
    W = 3 * pw + 4 * pad
    Hh = head + 2 * (rowt + ph) + pad + cap
    sheet = Image.new("RGB", (W, Hh), "white")
    dr = ImageDraw.Draw(sheet)
    for vi, (title, _, _) in enumerate(VIEWS):
        x = pad + vi * (pw + pad)
        tw = dr.textlength(title, font=f_col)
        dr.text((x + (pw - tw) / 2, 12), title, fill=(60, 60, 60), font=f_col)
    src = "stand-in" if a.demo else os.path.basename(a.scan or "scan")
    rows = [("old", "old: cylinder hand"), ("new", f"new: smooth hand from scan ({src})")]
    for ri, (row, label) in enumerate(rows):
        y = head + ri * (rowt + ph)
        dr.text((pad, y + 10), label, fill=(20, 20, 20), font=f_row)
        for vi in range(3):
            sheet.paste(Image.open(imgs[(row, vi)]).convert("RGB"), (pad + vi * (pw + pad), y + rowt))
    caption = ("Right hand holding the controller in Olex's hold pose, seen from three sides; "
               "both rows use the same pose and place.")
    dr.text((pad, Hh - cap + 16), caption, fill=(40, 40, 40), font=f_cap)
    sheet.save(out_jpg, quality=90)
    log("WROTE", out_jpg, sheet.size)


# ------------------------------------------------------------------ demo grading

def grade_demo(a):
    """Fitted joints vs the stand-in's true joints (undoing the disguise) -- demo only."""
    dj = os.path.join(a.out, "input", "disguise.json")
    fj = os.path.join(a.out, "fit", "joints.json")
    if not (os.path.exists(dj) and os.path.exists(fj)):
        return
    d = json.load(open(dj))
    R, t, s = np.array(d["R"]), np.array(d["t_mm"]), d["scale"]
    truth = json.load(open(os.path.join(STANDIN, "joints.json")))["landmarks"]
    fit = json.load(open(fj))
    # the fit keeps the scan frame and rescales to mm: x_fit = k * scan, scan = (truth R^T + t) s
    k = fit.get("fit", {}).get("scale_raw_to_mm")
    T = np.array([truth[n] for n in truth])
    Fp = np.array([fit["landmarks"][n] for n in truth])
    Tt = T @ R.T + t                       # truth in the disguised frame, mm
    # undo the fitted uniform scale/frame: best similarity-free compare = rigid in mm
    Rr, tr = kabsch(Fp, Tt)
    e_rigid = np.linalg.norm(Fp @ Rr.T + tr - Tt, axis=1)
    # direct: fitted scale estimate vs truth, joint errors in the true frame
    sc_est = None
    if k is not None:
        sc_est = k * s                       # mm per input unit * input unit per mm  -> ~1
    direct = np.linalg.norm(Fp - Tt * (sc_est if sc_est else 1.0), axis=1)
    res = {"names": list(truth), "direct_err_mm": {"median": float(np.median(direct)), "max": float(direct.max()),
                                                   "worst": list(truth)[int(direct.argmax())]},
           "after_rigid_align_mm": {"median": float(np.median(e_rigid)), "max": float(e_rigid.max())},
           "scale_ratio_fit_vs_true": sc_est}
    jdump(res, os.path.join(a.out, "demo_fit_grade.json"))
    log("demo fit grade:", json.dumps(res["direct_err_mm"]), "scale ratio", sc_est)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", nargs="?")
    ap.add_argument("--demo", action="store_true", help="use the stand-in (rotated, moved, in metres)")
    ap.add_argument("--out")
    ap.add_argument("--units", default="auto")
    ap.add_argument("--hold-rig", default=HOLD_RIG)
    ap.add_argument("--params", default=PARAMS if os.path.exists(PARAMS) else None)
    ap.add_argument("--steps", default="fit,hold,contacts,sheet")
    ap.add_argument("--spacing", type=float, default=0.5, help="skin sample spacing for contacts (mm)")
    ap.add_argument("--samples", type=int, default=32, help="render samples")
    ap.add_argument("--angle-prior", type=float, default=0.05,
                    help="hold placement: mm of joint residual one degree of angle change costs")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.demo:
        a.out = a.out or DEMO_OUT
        os.makedirs(a.out, exist_ok=True)
        a.scan = a.scan or make_demo_scan(a.out)
    if not a.scan and "fit" in a.steps:
        ap.error("give a scan file (or --demo)")
    a.out = os.path.abspath(a.out or os.path.join(os.path.dirname(os.path.abspath(a.scan)), "hand_on_grip"))
    os.makedirs(a.out, exist_ok=True)
    steps = a.steps.replace("skin", "hold").split(",")
    t0 = time.time()
    if "fit" in steps:
        step_fit(a)
        if a.demo:
            grade_demo(a)
    for s in ("hold", "contacts", "sheet"):
        if s in steps:
            # each Blender-heavy step in its own process: a clean bpy state per step
            if os.environ.get("RUN_ALL_CHILD") == s:
                globals()["step_" + s](a)
                return
            env = dict(os.environ, RUN_ALL_CHILD=s)
            cmd = [PY, os.path.abspath(__file__), *([a.scan] if a.scan else []), "--out", a.out,
                   "--steps", s, "--hold-rig", a.hold_rig, "--spacing", str(a.spacing),
                   "--samples", str(a.samples), "--angle-prior", str(a.angle_prior)] + (["--params", a.params] if a.params else []) + \
                  (["--force"] if a.force else []) + (["--demo"] if a.demo else [])
            subprocess.run(cmd, check=True, env=env)
    log(f"done in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
