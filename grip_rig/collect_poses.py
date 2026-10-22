"""collect_poses.py: one Blender file with every solved grip pose, plus a render of each.

    $PY collect_poses.py --out grip_rig_all_poses.blend --renders DIR  LABEL=WORKDIR [LABEL=WORKDIR ...]

Each WORKDIR holds the solve (solve.json, or solve_s5.json) and hand_hold.npz of one hand.
In the file:
  * armature 'grip rig' on basemesh GripV (rest pose = original GripV)
  * one action per hand, 'pose: <label>' (open the Action editor and pick one, or scrub the timeline)
  * the action 'all poses' keys every pose on its own frame: frame 1 = original, frame 2.. = the hands;
    each hand's skin ('hand: <label>', a static mesh in the solved hold) shows only on its frame
Renders: DIR/<n>.png per pose (grip + see-through hand, palm side) and DIR/poses_sheet.jpg.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import solve_core as sc  # noqa: E402
from render_sheet import FONT, FONT_B, camera, material, mesh_obj  # noqa: E402

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

HAND_COLL = "hands (one per pose)"
VIEWS = [("palm", (1.0, -0.15, 0.15), False, 130), ("front", (0.35, 1.0, -0.1), False, 130),
         ("hand", (1.0, -0.15, 0.15), True, 190)]
TITLES = {"palm": "grip, palm side", "front": "grip, finger side (front)", "hand": "with the hand (see-through)"}


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", default="/tmp/claude-0/hs/work/grip_rig/grip_rig.blend")
    ap.add_argument("--work", default="/tmp/claude-0/hs/work/grip_rig", help="rig_state / bones of the rig")
    ap.add_argument("--out", required=True)
    ap.add_argument("--renders", required=True)
    ap.add_argument("runs", nargs="+")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return ap.parse_args(argv)


def load_x(w):
    for n in ("solve.json", "solve_s5.json"):
        p = os.path.join(w, n)
        if os.path.exists(p):
            return np.array(json.load(open(p))["x"])
    raise FileNotFoundError(w)


def key_pose(rig, model, x, frame, action):
    rig.animation_data.action = action
    model.apply_to_rig(rig, x)
    for pb in rig.pose.bones:
        pb.keyframe_insert("location", frame=frame)
        pb.keyframe_insert("rotation_euler", frame=frame)


def set_constant(action):
    """Constant interpolation: each frame shows exactly one solved pose (no blends in between)."""
    try:
        curves = action.fcurves
    except AttributeError:                      # Blender 5: layered actions
        curves = [fc for layer in action.layers for strip in layer.strips
                  for bag in strip.channelbags for fc in bag.fcurves]
    for fc in curves:
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"


def main():
    a = args()
    runs = [r.split("=", 1) for r in a.runs]
    model = sc.Model(a.work)
    bpy.ops.wm.open_mainfile(filepath=a.rig)
    bpy.data.node_groups["global"].nodes["global parameters"].inputs["variant"].default_value = "GripV"
    rig = bpy.data.objects[sc.RIG]
    # the build appended one rigged hand for reference; replace it by one static hand per pose
    old = bpy.data.collections.get("hand reference (hold_upright)")
    if old is not None:
        for o in list(old.objects):
            bpy.data.objects.remove(o)
        bpy.data.collections.remove(old)
    coll = bpy.data.collections.new(HAND_COLL)
    bpy.context.scene.collection.children.link(coll)
    skin = bpy.data.materials.new("hand skin")
    skin.diffuse_color = (0.86, 0.62, 0.5, 0.5)
    rig.animation_data_create()
    allp = bpy.data.actions.new("all poses")
    allp.use_fake_user = True
    key_pose(rig, model, np.zeros(model.n), 1, allp)
    hands = []
    for k, (lab, w) in enumerate(runs):
        x = load_x(w)
        act = bpy.data.actions.new(f"pose: {lab}")
        act.use_fake_user = True
        key_pose(rig, model, x, 1, act)
        set_constant(act)
        key_pose(rig, model, x, k + 2, allp)
        h = np.load(os.path.join(w, "hand_hold.npz"))
        me = bpy.data.meshes.new(f"hand: {lab}")
        me.from_pydata(h["V"].tolist(), [], h["T"].tolist())
        for p in me.polygons:
            p.use_smooth = True
        me.materials.append(skin)
        ob = bpy.data.objects.new(f"hand: {lab}", me)
        coll.objects.link(ob)
        ob.display_type = "SOLID"
        ob.show_transparent = True
        ob.color = (0.86, 0.62, 0.5, 0.45)
        for f in range(1, len(runs) + 2):        # visible on its own frame only
            ob.hide_viewport = ob.hide_render = (f != k + 2)
            ob.keyframe_insert("hide_viewport", frame=f)
            ob.keyframe_insert("hide_render", frame=f)
        hands.append((lab, ob, x))
    set_constant(allp)
    rig.animation_data.action = allp
    sc_ = bpy.context.scene
    sc_.frame_start, sc_.frame_end = 1, len(runs) + 1
    txt = bpy.data.texts.get("grip rig poses") or bpy.data.texts.new("grip rig poses")
    txt.clear()
    txt.write("Solved grip poses\n\nTimeline: frame 1 = original GripV, then one frame per hand:\n" +
              "".join(f"  frame {k + 2}: {lab}\n" for k, (lab, _) in enumerate(runs)) +
              "\nOr pick one action 'pose: <hand>' on the 'grip rig' armature (Action editor).\n"
              "Clear the action and Pose > Clear Transform to get the original GripV back.\n")
    sc_.frame_set(2)
    bpy.ops.wm.save_as_mainfile(filepath=a.out, compress=True)
    print("[poses] saved", a.out, flush=True)

    # ---- renders: evaluated basemesh of each pose with its hand (separate empty scene)
    os.makedirs(a.renders, exist_ok=True)
    cage = bpy.data.objects[sc.CAGE]
    meshes = [("Original GripV", np.zeros(model.n), None)] + [(lab, x, ob) for lab, ob, x in hands]
    Vs = []
    rig.animation_data.action = None
    for lab, x, ob in meshes:
        model.apply_to_rig(rig, x)
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        e = cage.evaluated_get(dg)
        me = e.to_mesh()
        me.calc_loop_triangles()
        V = np.array([v.co[:] for v in me.vertices])
        T = np.array([t.vertices[:] for t in me.loop_triangles])
        e.to_mesh_clear()
        hV = None if ob is None else np.array([v.co[:] for v in ob.data.vertices])
        hT = None if ob is None else np.array([p.vertices[:] for p in ob.data.polygons])
        Vs.append((lab, V, T, hV, hT))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.render.engine = "CYCLES"
    s.cycles.device = "CPU"
    s.cycles.samples = 24
    s.render.resolution_x = s.render.resolution_y = 800
    s.view_settings.view_transform = "Standard"
    wd = bpy.data.worlds.new("w")
    wd.use_nodes = True
    wd.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    wd.node_tree.nodes["Background"].inputs[1].default_value = 0.9
    s.world = wd
    for i, (loc, en) in enumerate([((300, -200, 300), 2.5), ((-200, 300, -100), 1.0)]):
        L = bpy.data.objects.new(f"sun{i}", bpy.data.lights.new(f"sun{i}", "SUN"))
        L.data.energy = en
        L.rotation_euler = (-Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        s.collection.objects.link(L)
    mg = material("grip", (0.62, 0.64, 0.68))
    mh = material("hand", (0.95, 0.62, 0.48), alpha=0.18)
    tiles = []
    for i, (lab, V, T, hV, hT) in enumerate(Vs):
        for o in list(s.collection.objects):
            if o.type == "MESH":
                bpy.data.objects.remove(o)
        g = mesh_obj("grip", V, T)
        g.data.materials.append(mg)
        hh = None
        if hV is not None:
            hh = mesh_obj("hand", hV, hT)
            hh.data.materials.append(mh)
        for view, d, with_hand, scale in VIEWS:
            if hh is not None:
                hh.hide_render = not with_hand
            camera(np.array([62.0, -50.0, 18.0]), d, (0, 0, 1), scale)
            p = os.path.join(a.renders, f"{i}_{view}.png")
            s.render.filepath = p
            bpy.ops.render.render(write_still=True)
            tiles.append((i, view, Image.open(p).convert("RGB")))
    # sheet: one row per pose, palm side + finger side
    R, pad, head = 520, 16, 70
    n = len(Vs)
    W = 3 * R + 4 * pad + 520
    H = head + n * (R + pad)
    sheet = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(sheet)
    fb, f = ImageFont.truetype(FONT_B, 34), ImageFont.truetype(FONT, 30)
    order = [v[0] for v in VIEWS]
    for c, v in enumerate(order):
        dr.text((pad + c * (R + pad) + R // 2, 35), TITLES[v], font=f, fill="black", anchor="mm")
    for i, view, im in tiles:
        c = order.index(view)
        sheet.paste(im.resize((R, R)), (pad + c * (R + pad), head + i * (R + pad)))
    for i, (lab, *_r) in enumerate(Vs):
        y = head + i * (R + pad) + R // 2
        words, line, lines = lab.split(), "", []
        for wd_ in words:
            if len(line) + len(wd_) > 22:
                lines.append(line)
                line = ""
            line += ("" if not line else " ") + wd_
        lines.append(line)
        for j, ln in enumerate(lines):
            dr.text((3 * R + 4 * pad + 10, y - 20 * len(lines) + 42 * j), ln, font=fb if j == 0 else f, fill="black")
    sheet.save(os.path.join(a.renders, "poses_sheet.jpg"), quality=88)
    print("[poses] sheet", os.path.join(a.renders, "poses_sheet.jpg"), flush=True)


if __name__ == "__main__":
    main()
