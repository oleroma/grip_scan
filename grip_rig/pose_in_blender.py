"""pose_in_blender.py: put the solved pose on the grip rig, check it in Blender, save the posed file.

    $PY pose_in_blender.py [--rig grip_rig.blend] [--solve solve.json] [--out grip_rig_posed.blend]
                           [--work /tmp/claude-0/hs/work/grip_rig] [--no-parts]

Checks on the evaluated meshes (nothing simulated):
  numpy model vs Blender basemesh; exception zone (largest move of any surface point inside it);
  flat top (largest Z change, max Z); every case part that reads the basemesh still builds closed
  (vertex count, volume and bounding-box change vs the rest pose); smoothness of the change.
The posed rig is saved as an action-free pose (the bones keep the solved values), so opening the file
shows the posed grip; 'Pose > Clear Transform' brings back the original GripV.
Writes <work>/pose_check.json and <work>/posed_meshes.npz (rest and posed basemesh, for the sheet).
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
import solve_core as sc  # noqa: E402
from check_demo import EXTRA, PARTS, eval_mesh, eval_parts, open_grip, scratch, surface_distance  # noqa: E402

import bpy  # noqa: E402

PROTECTED = {"/home/claude/Grip/blender/Grip.blend"}


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", default="/tmp/claude-0/hs/work/grip_rig/grip_rig.blend")
    ap.add_argument("--work", default="/tmp/claude-0/hs/work/grip_rig")
    ap.add_argument("--solve", default=None)
    ap.add_argument("--out", default="/tmp/claude-0/hs/work/grip_rig/grip_rig_posed.blend")
    ap.add_argument("--no-parts", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return ap.parse_args(argv)


def main():
    a = args()
    t0 = time.time()
    assert os.path.abspath(a.out) not in PROTECTED
    sol = json.load(open(a.solve or os.path.join(a.work, "solve.json")))
    x = np.array(sol["x"])
    model = sc.Model(a.work)
    open_grip(a.rig)
    rig = bpy.data.objects[sc.RIG]
    cage = bpy.data.objects[sc.CAGE]
    win_scene = bpy.context.window.scene
    parts = [] if a.no_parts else list(PARTS)
    scratch([sc.CAGE, sc.RIG] + parts + list(EXTRA))
    model.apply_to_rig(rig, np.zeros(model.n))
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    S_rest, T, _ = eval_mesh(cage, dg)
    parts_rest = eval_parts(parts + list(EXTRA))
    model.apply_to_rig(rig, x)
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    S1, _, _ = eval_mesh(cage, dg)
    parts_posed = eval_parts(parts + list(EXTRA))
    oz = model.out_in_zone
    mv = np.linalg.norm(S1 - S_rest, axis=1)
    zm = oz & (mv > 1e-6)
    top = S_rest[:, 2] > rc.TOP_Z - 0.001
    rep = {
        "numpy_vs_blender_max_mm": round(float(np.abs(S1 - model.surface(x)).max()), 4),
        "rest_vs_original_max_mm": round(float(np.abs(S_rest - model.S0).max()), 5),
        "largest_move_mm": round(float(mv.max()), 2),
        "zone_max_move_mm": round(float(mv[oz].max()), 5),
        "zone_surface_max_deviation_mm": round(float(surface_distance(S1[zm], S_rest, T).max()), 5) if zm.any() else 0.0,
        "top_max_z_change_mm": round(float(np.abs(S1[top, 2] - S_rest[top, 2]).max()), 5),
        "top_max_z": round(float(S1[:, 2].max()), 4),
        "basemesh_bbox_change_mm": {"min": np.round(S1.min(0) - S_rest.min(0), 3).tolist(),
                                    "max": np.round(S1.max(0) - S_rest.max(0), 3).tolist()},
        "smooth": rc.smooth_report(S_rest, S1, T),
        "parts": {},
    }
    for n, i0 in parts_rest.items():
        i1 = parts_posed.get(n, {})
        rep["parts"][n] = {
            "verts": [i0.get("verts"), i1.get("verts")],
            "closed": [i0.get("closed_manifold"), i1.get("closed_manifold")],
            "volume_change_mm3": (round(i1["volume_mm3"] - i0["volume_mm3"], 1)
                                  if "volume_mm3" in i0 and "volume_mm3" in i1 else None),
            "bbox_max_change_mm": (round(float(np.abs(np.array(i1["bbox"]) - np.array(i0["bbox"])).max()), 3)
                                   if "bbox" in i0 and "bbox" in i1 else None)}
    print("[pose]", json.dumps({k: v for k, v in rep.items() if k != "parts"}), flush=True)
    for n, v in rep["parts"].items():
        print("[pose] part", n, json.dumps(v), flush=True)
    bpy.context.window.scene = win_scene
    for s in list(bpy.data.scenes):
        if s.name.startswith("check scratch"):
            bpy.data.scenes.remove(s)
    txt = bpy.data.texts.get("grip rig solve") or bpy.data.texts.new("grip rig solve")
    txt.clear()
    txt.write(json.dumps({"solve": {k: v for k, v in sol.items() if k != "rest"}, "blender_check": rep}, indent=1))
    rig["solved_pose"] = json.dumps(sol["channels"])
    model.apply_to_rig(rig, x)
    bpy.ops.wm.save_as_mainfile(filepath=a.out, compress=True)
    rep["seconds"] = round(time.time() - t0, 1)
    json.dump(rep, open(os.path.join(a.work, "pose_check.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.work, "posed_meshes.npz"), rest=S_rest, posed=S1, T=T)
    print("[pose] saved", a.out, f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
