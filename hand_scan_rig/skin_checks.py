"""Checks for skin.py: numpy-vs-Blender verification, quality metrics, comparison picture."""
import json
import os
import time

import numpy as np

import skin
from skin import (AUTO_BLEND, BI, BONES, CHAINS, FLEX_JOINTS, JOINTS, OUT, TEST_BLEND, WORK,
                  Skinner, load_params, unit, vertex_normals)

HOLD_JSON = os.path.join(WORK, "hold_pose.json")

# configurations compared
CONFIGS = {
    # Blender bone-heat (automatic) weights + plain linear blend skinning, no correctives
    "plain": dict(blend=AUTO_BLEND, params={"global": {"weights": "blend", "dqs": False, "cs_factor": 0.0}}),
    # parametric weights, plain LBS (no corrective smooth, no bulge)
    "param_lbs": dict(blend=TEST_BLEND, params={"global": {"dqs": False, "cs_factor": 0.0},
                                                "joints": {j: {"bulge_mm": 0.0} for j in FLEX_JOINTS}}),
    # shipped defaults (skin_params.json)
    "default": dict(blend=TEST_BLEND, params=None),
}


def config_params(name, base=None):
    c = CONFIGS[name]
    if c["params"] is None:
        return load_params(base)
    p = load_params(base)
    for k, v in c["params"].get("global", {}).items():
        p["global"][k] = v
    for j, d in c["params"].get("joints", {}).items():
        p["joints"][j].update(d)
    return p


# ------------------------------------------------------------------ hold pose

def hold_pose():
    """Angles of the 'hand rig' rest pose in hold_rig.blend (relaxed_pose definitions, thumb
    sign per SPEC) + palm arch and thumb elevation."""
    if os.path.exists(HOLD_JSON):
        return json.load(open(HOLD_JSON))
    import bpy
    bpy.ops.wm.open_mainfile(filepath=skin.HOLD_RIG)
    L = skin._landmarks_from_armature(bpy.data.objects["hand rig"])
    m = skin.measurements(L)
    pose = {"angles": {k: round(float(v), 3) for k, v in m.items() if not k.startswith("_")},
            "source": "rest pose of 'hand rig' in " + skin.HOLD_RIG}
    os.makedirs(WORK, exist_ok=True)
    json.dump(pose, open(HOLD_JSON, "w"), indent=1)
    return pose


# ------------------------------------------------------------------ verify

def cmd_verify(a):
    hold = hold_pose()
    poses = {"index 90/90/60": {"angles": {"index_mcp": 90, "index_pip": 90, "index_dip": 60}},
             "hold": hold,
             "thumb twist 45": {"twist": {"thumb_proximal": 45}, "angles": {"thumb_mcp": 40}}}
    out = {}
    for cname, dqs in [("default", True), ("default", False), ("plain", False), ("plain", True)]:
        c = CONFIGS[cname]
        p = config_params(cname)
        p["global"]["dqs"] = dqs
        t0 = time.time()
        sk = Skinner(c["blend"], p)
        t_load = time.time() - t0
        for pn, pose in poses.items():
            t0 = time.time()
            Vb, info = sk.pose(pose, "blender", return_info=True)
            tb = time.time() - t0
            t0 = time.time()
            Vn = sk.pose(pose, "numpy")
            tn = time.time() - t0
            d = np.linalg.norm(Vb - Vn, axis=1)
            key = f"{cname} ({'DQS' if dqs else 'LBS'}) | {pn}"
            out[key] = dict(max_diff_mm=float(d.max()), mean_diff_mm=float(d.mean()), blender_s=round(tb, 3),
                            numpy_s=round(tn, 3), load_and_weights_s=round(t_load, 2),
                            solve_err_deg=info["max_angle_err_deg"], moved_mm=float(np.linalg.norm(Vb - sk.V0, axis=1).max()))
            print(key, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out[key].items()}, flush=True)
    # pure LBS/DQS (no CS) to separate the stages
    for dqs in (False, True):
        p = config_params("param_lbs")
        p["global"]["dqs"] = dqs
        sk = Skinner(TEST_BLEND, p)
        Vb = sk.pose(poses["hold"], "blender")
        Vn = sk.pose(poses["hold"], "numpy")
        d = np.linalg.norm(Vb - Vn, axis=1)
        out[f"armature only {'dqs' if dqs else 'lbs'} | hold"] = dict(max_diff_mm=float(d.max()), mean_diff_mm=float(d.mean()))
        print("armature only", dqs, d.max(), d.mean())
    json.dump(out, open(os.path.join(OUT, "verify.json"), "w"), indent=1)


# ------------------------------------------------------------------ metrics

class Regions:
    """Per-joint vertex/face/edge sets (from the default parametric fields; geometric only,
    so every configuration is measured on the same skin)."""
    RADIUS = {"mcp": 18.0, "pip": 12.0, "dip": 9.0, "thumb_mcp": 14.0, "thumb_ip": 10.0}

    def __init__(self, sk_default):
        sk = sk_default
        f = sk.fields
        self.V0, self.F = sk.V0, sk.F
        self.E = f["edges"]
        self.reg = {}
        for key, pk, c, lm, pb, cb in JOINTS:
            if key not in FLEX_JOINTS:
                continue
            g = f["geo"][key]
            R = self.RADIUS[key] if key in self.RADIUS else self.RADIUS[key.split("_")[-1]]
            ci = CHAINS.index(c)
            vin = (f["mu"][:, ci] > 0.5) & (np.abs(g["s"]) < R)
            fin = vin[self.F].all(axis=1)
            ein = vin[self.E].all(axis=1)
            dors = ein & (g["p"][self.E] < -0.3).all(axis=1)
            near = np.linalg.norm(self.V0 - g["J"], axis=1) < 40.0
            core = vin & (np.abs(g["s"]) < 0.5 * sk.params["joints"][pk]["width_mm"] + 1.0) & \
                (np.linalg.norm(self.V0 - g["J"], axis=1) > 3.0)
            mid = vin & (g["p"] < -0.85)
            # rest volume of the joint region for normalising: cylinder of the mean skin radius
            rr = np.linalg.norm((self.V0[vin] - g["J"]) - np.outer((self.V0[vin] - g["J"]) @ g["nj"], g["nj"]), axis=1)
            cyl = float(np.pi * np.mean(rr) ** 2 * 2 * R)
            self.reg[key] = dict(v=vin, f=fin, e=ein, dors=dors, near=near, core=core, J=g["J"], lm=lm,
                                 mid_v=np.nonzero(mid)[0], mid_s=g["s"][mid], cyl=cyl)
        # face adjacency for dihedral angles
        F = self.F
        e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        fid = np.tile(np.arange(len(F)), 3)
        es = np.sort(e, axis=1)
        order = np.lexsort((es[:, 1], es[:, 0]))
        es, fid = es[order], fid[order]
        same = (es[1:] == es[:-1]).all(axis=1)
        self.fpair = np.stack([fid[:-1][same], fid[1:][same]], 1)
        self.fpair_edge = es[:-1][same]
        # face shape quality (1 = equilateral); slivers (< 0.2) flip under any non-rigid motion
        # and are left out of the crease/fold measures
        t = self.V0[F]
        area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
        e2 = sum(np.sum((t[:, (k + 1) % 3] - t[:, k]) ** 2, 1) for k in range(3))
        self.quality = 4 * np.sqrt(3) * area / e2
        self.good_pair = (self.quality[self.fpair] >= 0.2).all(axis=1)
        self.vol0 = mesh_volume(self.V0, F)


def face_normals(V, F):
    t = V[F]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)


def mesh_volume(V, F, o=None):
    o = np.zeros(3) if o is None else o
    t = V[F] - o
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


def penetration(V, F, sel, V0=None, local_mm=6.0, search_mm=25.0):
    """Self-intersection depth (mm) per vertex.  Inside test: cast a ray from the vertex along
    its outward normal; if the first surface hit faces away from the ray, the vertex lies inside
    the hand (skin pushed through skin).  Depth: distance from the vertex to the nearest posed
    surface that is not its own neighbourhood (faces whose rest position is > local_mm away)."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    bvh = BVHTree.FromPolygons([tuple(v) for v in V.tolist()], F.tolist(), epsilon=0.0)
    N = vertex_normals(V, F)
    fn = face_normals(V, F)
    V0 = V if V0 is None else V0
    C0 = V0[F].mean(axis=1)
    depth = np.zeros(len(V))
    for i in np.nonzero(sel)[0]:
        loc, nrm, idx, dist = bvh.ray_cast(Vector(V[i] + 1e-3 * N[i]), Vector(N[i]), 60.0)
        if idx is None or fn[idx] @ N[i] <= 0:
            continue
        best = search_mm
        for loc, nrm, fi, d in bvh.find_nearest_range(Vector(V[i]), search_mm):
            if d < best and np.linalg.norm(C0[fi] - V0[i]) > local_mm:
                best = d
        depth[i] = best
    return depth


def dihedral_stats(V, F, reg, R):
    """max and 99th percentile of the dihedral angle between adjacent well-shaped faces in the
    joint region, and the number of fold-overs (> 90 deg)."""
    fn = face_normals(V, F)
    sel = R.fpair[reg["f"][R.fpair].all(axis=1) & R.good_pair]
    if not len(sel):
        return 0.0, 0.0, 0
    c = np.sum(fn[sel[:, 0]] * fn[sel[:, 1]], 1)
    a = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return float(a.max()), float(np.percentile(a, 99)), int((a > 90).sum())


def midline_length(V, reg):
    s, idx = reg["mid_s"], reg["mid_v"]
    bins = np.floor(s).astype(int)
    pts = []
    for b in np.unique(bins):
        m = bins == b
        if m.sum() >= 2:
            pts.append(V[idx[m]].mean(axis=0))
    pts = np.array(pts)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def joint_metrics(V, info, R, key, rest_cache, single=True):
    """Metrics of one joint region.  single=True: only this joint moved, so the exact change of
    the closed mesh volume is this joint's; it is given relative to the joint region's rest
    volume (cone volume about the joint centre of the faces within the region radius)."""
    reg = R.reg[key]
    V0, F = R.V0, R.F
    Jr = reg["J"]
    Jp = info["landmarks"][reg["lm"]]
    if key not in rest_cache:
        rest_cache[key] = dict(vol=mesh_volume(V0, F[reg["f"]], Jr), dih=dihedral_stats(V0, F, reg, R),
                               mid=midline_length(V0, reg))
    rc = rest_cache[key]
    e = R.E[reg["dors"]]
    l0 = np.linalg.norm(V0[e[:, 0]] - V0[e[:, 1]], axis=1)
    l1 = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    strain = (l1 / l0 - 1)[l0 > 0.5]
    core = reg["core"]
    r0 = np.linalg.norm(V0[core] - Jr, axis=1)
    r1 = np.linalg.norm(V[core] - Jp, axis=1)
    shrink = 1 - r1 / r0
    pen = penetration(V, F, reg["near"], V0)
    dmax, d99, folds = dihedral_stats(V, F, reg, R)
    out = dict(intersect_mm=float(pen.max()),
               dorsal_stretch_pct=100 * (midline_length(V, reg) / rc["mid"] - 1),
               dorsal_strain_p95_pct=float(100 * np.percentile(strain, 95)),
               collapse_max_pct=float(100 * shrink.max()),
               crease_max_deg=dmax, crease_p99_deg=d99, folds=folds,
               crease_rest_max_deg=rc["dih"][0], crease_rest_p99_deg=rc["dih"][1])
    if single:
        out["vol_change_pct"] = 100 * (mesh_volume(V, F) - R.vol0) / reg["cyl"]
    return out


def cmd_check(a):
    names = a.configs.split(",")
    hold = hold_pose()
    angles = [30, 60, 90] if not a.quick else [90]
    joints = FLEX_JOINTS if not a.quick else ["index_mcp", "index_pip"]
    path = os.path.join(OUT, "quality.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    sk_def = Skinner(TEST_BLEND, config_params("default", a.params))
    R = Regions(sk_def)
    rest_cache = {}
    V0 = R.V0
    vol0 = mesh_volume(V0, R.F)
    for cname in names:
        t0 = time.time()
        sk = Skinner(CONFIGS[cname]["blend"], config_params(cname, a.params))
        if not np.allclose(sk.V0, V0):
            raise SystemExit("rigs do not share the mesh")
        out = {"joints": {}, "hold": {}, "twist": {}}
        for j in joints:
            for ang in angles:
                V, info = sk.pose({"angles": {j: ang}}, "numpy", return_info=True)
                m = joint_metrics(V, info, R, j, rest_cache)
                m["total_vol_change_pct"] = 100 * (mesh_volume(V, R.F) - vol0) / vol0
                out["joints"][f"{j}@{ang}"] = m
        # hold pose: every flex joint region at once, intersections over the whole hand
        V, info = sk.pose(hold, "numpy", return_info=True)
        for j in FLEX_JOINTS if not a.quick else joints:
            m = joint_metrics(V, info, R, j, rest_cache, single=False)
            out["hold"][j] = m
        out["hold_total_vol_change_pct"] = 100 * (mesh_volume(V, R.F) - vol0) / vol0
        out["hold_intersect_all_mm"] = float(penetration(V, R.F, np.ones(len(V), bool), V0).max())
        # candy-wrapper test: 45 deg axial twist of each proximal phalanx (shrink of the MCP zone)
        for f in ("index", "middle", "ring", "pinky", "thumb"):
            b = f"{f}_proximal"
            j = f"{f}_mcp"
            V, info = sk.pose({"twist": {b: 45.0}}, "numpy", return_info=True)
            m = joint_metrics(V, info, R, j, rest_cache)
            out["twist"][f"{b}@45"] = {k: m[k] for k in ("collapse_max_pct", "vol_change_pct", "intersect_mm", "folds")}
        out["seconds"] = round(time.time() - t0, 1)
        res[cname] = out
        json.dump(res, open(path, "w"), indent=1)
        print(cname, "done", out["seconds"], "s", flush=True)
    summarize(res)


def summarize(res, path=None):
    keys = ["vol_change_pct", "intersect_mm", "dorsal_stretch_pct", "dorsal_strain_p95_pct", "collapse_max_pct",
            "crease_p99_deg", "crease_max_deg", "folds"]
    short = ["local vol %", "intersect mm", "back stretch %", "back strain p95 %", "collapse %",
             "crease p99 deg", "crease max deg", "fold-overs"]
    lines = ["Quality of the skin deformation on the stand-in (numpy backend, identical to Blender).",
             "Cells: worst joint (median over joints).  14 joints: index/middle/ring/pinky MCP PIP DIP, thumb MCP IP.",
             "plain = Blender automatic (bone heat) weights + linear blend skinning; param_lbs = our weights +",
             "linear blend only; default = our weights + dual quaternion + corrective smooth + pad bulge.",
             "local vol: exact closed-mesh volume change / rest volume of the joint region (cylinder of its mean",
             "skin radius, length 2 x region radius: MCP 36, PIP 24, DIP 18, thumb MCP 28, IP 20 mm); intersect: depth",
             "of skin pushed through skin; back stretch: length change of the back-of-finger line over the",
             "joint (real skin stretches ~20-40% at 90); back strain p95: local edge stretch (concentration);",
             "collapse: shrink of the joint zone towards the joint centre (candy-wrapper/LBS loss);",
             "crease: dihedral angle between neighbouring faces (rest p99 ~5-10 deg); fold-overs: face pairs",
             "> 90 deg (slivers excluded)."]
    cn = [c for c in ("plain", "param_lbs", "default") if c in res] + [c for c in res if c not in ("plain", "param_lbs", "default")]
    head = "config      " + "".join(f"{k:>19s}" for k in short)
    for ang in (30, 60, 90):
        lines.append(f"\n== one joint bent to {ang} deg")
        lines.append(head)
        for c in cn:
            rows = [v for k, v in res[c]["joints"].items() if k.endswith(f"@{ang}")]
            if not rows:
                continue
            cells = []
            for k in keys:
                vals = np.array([r[k] for r in rows], float)
                worst = vals[np.argmax(np.abs(vals))]
                cells.append(f"{worst:7.1f} ({np.median(vals):6.1f})")
            lines.append(f"{c:12s}" + "".join(f"{s:>19s}" for s in cells))
    lines.append("\n== hold pose (hold_rig.blend rest pose angles): per-joint regions, worst (median)")
    lines.append(head)
    for c in cn:
        rows = list(res[c]["hold"].values())
        cells = []
        for k in keys:
            if k == "vol_change_pct":
                cells.append("see total")
                continue
            vals = np.array([r[k] for r in rows], float)
            cells.append(f"{vals[np.argmax(np.abs(vals))]:7.1f} ({np.median(vals):6.1f})")
        lines.append(f"{c:12s}" + "".join(f"{s:>19s}" for s in cells))
        lines.append(f"{'':12s}whole hand: volume {res[c]['hold_total_vol_change_pct']:+.2f}%, deepest skin-through-skin "
                     f"{res[c]['hold_intersect_all_mm']:.2f} mm")
    lines.append("\n== candy-wrapper test: proximal phalanx twisted 45 deg about its length, collapse % of the MCP zone")
    for c in cn:
        lines.append(f"{c:12s}" + "  ".join(f"{k.split('_')[0]} {v['collapse_max_pct']:5.1f}" for k, v in res[c]["twist"].items()))
    txt = "\n".join(lines)
    print(txt)
    open(path or os.path.join(OUT, "quality_summary.txt"), "w").write(txt + "\n")
    return txt


def cmd_render(a):
    import skin_render
    skin_render.render(a)
