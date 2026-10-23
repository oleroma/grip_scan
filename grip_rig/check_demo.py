"""check_demo.py: prove the rest pose is the original GripV, then pose each bone once (demo).

    $PY check_demo.py [--rig grip_rig.blend] [--grip Grip.blend] [--work /tmp/claude-0/hs/work/grip_rig]
                      [--parts case_top,case_bottom,...] [--no-parts]

Reads rig_state.npz / rig.json written by build_grip_rig.py (same --work).
Writes <work>/check.json and <work>/demo_meshes.npz (rest + every demo pose, for the sheet).

Checks (all on the evaluated meshes in Blender, nothing simulated):
  rest     cage after the rig modifiers == original cage; MirrorX + subdivision output == original;
           every case part that reads the basemesh: same vertex count, volume, bounding box
  demo     each bone moved alone by a moderate amount:
           exception zone: largest move of any output vertex inside the zone (and of the surface:
           distance of those vertices to the original surface); flat top: largest Z change of the
           top outputs; smoothness: largest change of the angle between neighbouring faces and the
           roughness of the displacement field; how far the bone's tip region really moved; what the
           move does to the basemesh bounding box (screws, R1/L1 read it); case parts still build
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rig_core as rc  # noqa: E402

import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

CAGE = "basemesh GripV"
RIG = "grip rig"
# case parts Olex's GripV export writes that read 'basemesh variant' (from smooth_fit eval_all.py)
PARTS = ("case_top", "case_bottom", "case_middle_plate", "R1", "L1", "R2", "R4", "L2", "L4",
         "cable holder", "capacitive", "nut_lock", "StickL", "stick", "home cap")
EXTRA = ("screw bools",)

DEMO = [  # bone, amount mm, how
    (rc.PALM, 4.0, "moved 4 mm outward (along the surface normal)"),
    ("fin middle", 4.0, "tip swung 4 mm outward"),
    ("fin ring", 4.0, "tip swung 4 mm outward"),
    ("fin pinky", 4.0, "tip swung 4 mm outward"),
    (rc.STAND, 4.0, "tip swung 4 mm outward"),
    (rc.THENAR, 3.0, "moved 3 mm outward"),
    (rc.TOP_EDGE, 3.0, "top back edge slid 3 mm back (-Y)"),
]


def log(*a):
    print("[check]", *a, flush=True)


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", default="/mnt/project-files/hand_scan_rig/grip_rig/grip_rig.blend")
    ap.add_argument("--grip", default="/home/claude/Grip/blender/Grip.blend")
    ap.add_argument("--work", default="/tmp/claude-0/hs/work/grip_rig")
    ap.add_argument("--parts", default=",".join(PARTS))
    ap.add_argument("--no-parts", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return ap.parse_args(argv)


def eval_mesh(ob, dg, world=True):
    e = ob.evaluated_get(dg)
    me = e.to_mesh()
    if me is None or len(me.vertices) == 0:
        e.to_mesh_clear()
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), {"verts": 0}
    me.calc_loop_triangles()
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    T = np.empty(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", T)
    T = T.reshape(-1, 3)
    le = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("edge_index", le)
    cnt = np.bincount(le, minlength=len(me.edges))
    info = {"verts": int(len(V)), "faces": int(len(me.polygons)),
            "closed_manifold": bool((cnt == 2).all())}
    e.to_mesh_clear()
    if world:
        M = np.array(ob.matrix_world)
        V = V @ M[:3, :3].T + M[:3, 3]
    if len(T):
        P = V[T]
        info["volume_mm3"] = round(abs(float(np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 1], P[:, 2])).sum() / 6.0)), 1)
        info["bbox"] = [np.round(V.min(0), 3).tolist(), np.round(V.max(0), 3).tolist()]
    return V, T, info


def open_grip(path):
    bpy.ops.wm.open_mainfile(filepath=path)
    bpy.data.node_groups["global"].nodes["global parameters"].inputs["variant"].default_value = "GripV"


def scratch(names):
    sc = bpy.data.scenes.new("check scratch")
    for n in names:
        if n in bpy.data.objects and n not in sc.collection.objects:
            sc.collection.objects.link(bpy.data.objects[n])
    bpy.context.window.scene = sc
    return sc


def eval_parts(names):
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    out = {}
    for n in names:
        if n not in bpy.data.objects:
            continue
        t = time.time()
        _, _, info = eval_mesh(bpy.data.objects[n], dg)
        info["seconds"] = round(time.time() - t, 2)
        out[n] = info
    return out


def surface_distance(P, V, T, k=24):
    """Exact distance from points P to the triangle mesh (V, T), checking the k triangles whose
    centroids are nearest."""
    from scipy.spatial import cKDTree
    C = V[T].mean(1)
    _, idx = cKDTree(C).query(P, k=min(k, len(T)))
    best = np.full(len(P), np.inf)
    for j in range(idx.shape[1]):
        tri = V[T[idx[:, j]]]
        best = np.minimum(best, _pt_tri(P, tri[:, 0], tri[:, 1], tri[:, 2]))
    return best


def _pt_tri(p, a, b, c):
    """Point-triangle distance (Ericson), vectorised."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = np.where(np.abs(va + vb + vc) < 1e-18, 1e-18, va + vb + vc)
    v, w = vb / den, vc / den
    q = a + ab * v[:, None] + ac * w[:, None]
    # regions
    m = (d1 <= 0) & (d2 <= 0)
    q[m] = a[m]
    m = (d3 >= 0) & (d4 <= d3)
    q[m] = b[m]
    m = (d6 >= 0) & (d5 <= d6)
    q[m] = c[m]
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / np.where(np.abs(d1 - d3) < 1e-18, 1e-18, d1 - d3)
    q[m] = (a + ab * t[:, None])[m]
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / np.where(np.abs(d2 - d6) < 1e-18, 1e-18, d2 - d6)
    q[m] = (a + ac * t[:, None])[m]
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) < 1e-18, 1e-18, (d4 - d3) + (d5 - d6))
    q[m] = (b + (c - b) * t[:, None])[m]
    return np.linalg.norm(p - q, axis=1)


def pose_bone(rig, name, amt):
    for pb in rig.pose.bones:
        pb.location = (0, 0, 0)
        pb.rotation_euler = (0, 0, 0)
    if name is None:
        bpy.context.view_layer.update()
        return
    pb = rig.pose.bones[name]
    b = pb.bone
    R = np.array(b.matrix_local.to_3x3())          # rest orientation (armature space = world, rig at origin)
    if name == rc.TOP_EDGE:
        pb.location = (0.0, amt, 0.0)
    elif name in (rc.PALM, rc.THENAR):
        if name == rc.PALM:
            n = np.array(b.matrix_local.col[1][:3])     # the palm swell bone points along the surface normal
        else:
            n = np.array(json.loads(rig["grip_rig_info"])["landmarks"][rc.THENAR]["n"])
        pb.location = Vector((R.T @ (amt * n / np.linalg.norm(n))).tolist())
    else:
        L = b.length
        pb.rotation_euler = (2.0 * np.arcsin(amt / (2.0 * L)), 0.0, 0.0)   # about local X: tip -> local +Z (outward)
    bpy.context.view_layer.update()


def main():
    a = args()
    t0 = time.time()
    st = np.load(os.path.join(a.work, "rig_state.npz"))
    info = json.load(open(os.path.join(a.work, "rig.json")))
    oz = st["out_in_zone"]
    parts = [] if a.no_parts else [p for p in a.parts.split(",") if p]
    rep = {"rest": {}, "demo": {}}

    # ---- original
    open_grip(a.grip)
    scratch([CAGE] + parts + list(EXTRA))
    dg = bpy.context.evaluated_depsgraph_get()
    S_orig, T, _ = eval_mesh(bpy.data.objects[CAGE], dg)
    cage_orig = np.array([v.co[:] for v in bpy.data.objects[CAGE].data.vertices])
    parts_orig = eval_parts(parts + list(EXTRA))
    log("original evaluated", f"{time.time() - t0:.0f} s")

    # ---- rig file at rest
    open_grip(a.rig)
    rig = bpy.data.objects[RIG]
    cage = bpy.data.objects[CAGE]
    scratch([CAGE, RIG] + parts + list(EXTRA))
    pose_bone(rig, None, 0)
    dg = bpy.context.evaluated_depsgraph_get()
    S_rest, T2, _ = eval_mesh(cage, dg)
    assert len(S_rest) == len(S_orig) and (T2 == T).all(), "topology changed"
    # the cage right after the rig modifiers (MirrorX and subdivision switched off for this read)
    keep = {}
    for m in cage.modifiers:
        if m.name in ("MirrorX", "subdivision surface"):
            keep[m.name] = m.show_viewport
            m.show_viewport = False
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    cage_rest, _, _ = eval_mesh(cage, dg, world=False)
    for n, v in keep.items():
        cage.modifiers[n].show_viewport = v
    parts_rest = eval_parts(parts + list(EXTRA))
    rep["rest"] = {
        "cage_max_deviation_mm": float(np.abs(cage_rest - cage_orig).max()),
        "basemesh_output_max_deviation_mm": float(np.abs(S_rest - S_orig).max()),
        "parts": {},
    }
    for n, i0 in parts_orig.items():
        i1 = parts_rest.get(n, {})
        rep["rest"]["parts"][n] = {
            "verts": [i0.get("verts"), i1.get("verts")],
            "volume_mm3": [i0.get("volume_mm3"), i1.get("volume_mm3")],
            "bbox_max_change_mm": (round(float(np.abs(np.array(i0["bbox"]) - np.array(i1["bbox"])).max()), 4)
                                   if "bbox" in i0 and "bbox" in i1 else None),
            "closed": [i0.get("closed_manifold"), i1.get("closed_manifold")]}
    log("rest", json.dumps({k: v for k, v in rep["rest"].items() if k != "parts"}), f"{time.time() - t0:.0f} s")

    # ---- demo poses
    meshes = {"rest|V": S_rest, "T": T}
    S0 = S_rest
    top = S0[:, 2] > rc.TOP_Z - 0.001
    pinned_out = st["out_free"] < 0.05            # outputs that are (almost) fully pinned
    for name, amt, how in DEMO:
        pose_bone(rig, name, amt)
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        S1, _, _ = eval_mesh(cage, dg)
        mv = np.linalg.norm(S1 - S0, axis=1)
        r = {"how": how, "amount_mm": amt}
        if name in info["landmarks"]:
            p = np.array(info["landmarks"][name]["p"])
            near = np.linalg.norm(S0 - p, axis=1) < 6.0
            r["tip_region_moved_mm"] = round(float(mv[near].max()), 2)
        r["zone_vertex_max_move_mm"] = round(float(mv[oz].max()), 4)
        zm = oz & (mv > 1e-6)
        r["zone_surface_max_deviation_mm"] = round(float(surface_distance(S1[zm], S0, T).max()), 4) if zm.any() else 0.0
        r["zone_vertices_moved"] = int(zm.sum())
        r["top_max_z_change_mm"] = round(float(np.abs(S1[top, 2] - S0[top, 2]).max()), 5)
        r["top_max_z"] = round(float(S1[:, 2].max()), 4)
        r["pinned_outputs_max_move_mm"] = round(float(mv[pinned_out].max()), 4)
        r["smooth"] = rc.smooth_report(S0, S1, T)
        r["basemesh_bbox_change_mm"] = {"min": np.round(S1.min(0) - S0.min(0), 3).tolist(),
                                        "max": np.round(S1.max(0) - S0.max(0), 3).tolist()}
        if parts:
            pp = eval_parts(parts + list(EXTRA))
            r["parts"] = {n: {"verts": v.get("verts"), "closed": v.get("closed_manifold"),
                              "volume_change_mm3": (round(v["volume_mm3"] - parts_rest[n]["volume_mm3"], 1)
                                                    if "volume_mm3" in v and "volume_mm3" in parts_rest.get(n, {}) else None),
                              "bbox_max_change_mm": (round(float(np.abs(np.array(v["bbox"]) - np.array(parts_rest[n]["bbox"])).max()), 3)
                                                     if "bbox" in v and "bbox" in parts_rest.get(n, {}) else None)}
                          for n, v in pp.items()}
        rep["demo"][name] = r
        meshes[name + "|V"] = S1
        log(name, json.dumps({k: v for k, v in r.items() if k != "parts"}), f"{time.time() - t0:.0f} s")
    pose_bone(rig, None, 0)
    np.savez_compressed(os.path.join(a.work, "demo_meshes.npz"), **meshes)
    rep["seconds"] = round(time.time() - t0, 1)
    json.dump(rep, open(os.path.join(a.work, "check.json"), "w"), indent=1)
    log("done", f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
