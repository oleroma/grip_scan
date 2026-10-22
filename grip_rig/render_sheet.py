"""render_sheet.py: a picture of the solved grip that a non-expert can read.

    $PY render_sheet.py [--work /tmp/claude-0/hs/work/grip_rig] [--out grip_rig_sheet.jpg]

Columns: original GripV | posed by the solver.  Rows: two familiar views with the hand see-through,
and the same views coloured by how much the surface moved (red = grew out towards the hand,
blue = pulled in). Reads posed_meshes.npz (pose_in_blender.py) and hand_hold.npz (extract_hand.py).
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import bpy
from mathutils import Vector

FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FONT_B = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
VIEWS = [  # name, camera direction (from target towards camera), up
    ("right handle, palm side (outside)", (1.0, -0.15, 0.15), (0, 0, 1)),
    ("right handle, finger side (front)", (0.35, 1.0, -0.1), (0, 0, 1)),
    ("right handle, from below", (0.35, 0.1, -1.0), (0, 1, 0)),
]
HAND_VIEW = 0      # the row with the hand uses this view
RES = 700


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/claude-0/hs/work/grip_rig")
    ap.add_argument("--out", default="/tmp/claude-0/hs/work/grip_rig/grip_rig_sheet.jpg")
    ap.add_argument("--max-mm", type=float, default=4.0)
    ap.add_argument("--compare", default=None,
                    help="second work dir: left = this work's posed grip + hand, middle = the other's, right = other vs this")
    ap.add_argument("--titles", default=None, help="three column titles separated by '|'")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return ap.parse_args(argv)


def mesh_obj(name, V, T, col=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], T.tolist())
    me.update()
    for p in me.polygons:
        p.use_smooth = True
    if col is not None:
        a = me.color_attributes.new("chg", "FLOAT_COLOR", "POINT")
        rgba = np.concatenate([col, np.ones((len(col), 1))], 1).astype(np.float32)
        a.data.foreach_set("color", rgba.ravel())
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def material(name, rgb, alpha=1.0, attr=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.55
    bsdf.inputs["Alpha"].default_value = alpha
    if attr:
        an = nt.nodes.new("ShaderNodeVertexColor")
        an.layer_name = attr
        nt.links.new(an.outputs["Color"], bsdf.inputs["Base Color"])
    return m


def change_colours(S0, S1, T, max_mm):
    fn = np.cross(S0[T[:, 1]] - S0[T[:, 0]], S0[T[:, 2]] - S0[T[:, 0]])
    n = np.zeros_like(S0)
    for k in range(3):
        np.add.at(n, T[:, k], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    s = ((S1 - S0) * n).sum(1)
    t = np.clip(s / max_mm, -1, 1)
    grey = np.array([0.8, 0.8, 0.8])
    red, blue = np.array([0.85, 0.08, 0.05]), np.array([0.05, 0.25, 0.9])
    c = np.where(t[:, None] > 0, grey + t[:, None] * (red - grey), grey + (-t[:, None]) * (blue - grey))
    return c ** 2.2, s          # linear colour space


def camera(target, d, up, scale):
    cam = bpy.data.objects.get("cam") or bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    if cam.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = scale
    d = Vector(d).normalized()
    cam.location = Vector(target) + d * 400
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    cam.data.clip_end = 2000
    bpy.context.scene.camera = cam


def render(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return Image.open(path).convert("RGB")


def main():
    a = args()
    pm = np.load(os.path.join(a.work, "posed_meshes.npz"))
    h = np.load(os.path.join(a.work, "hand_hold.npz"))
    S0, S1, T = pm["rest"], pm["posed"], pm["T"]
    h1 = h
    if a.compare:
        S0 = S1
        S1 = np.load(os.path.join(a.compare, "posed_meshes.npz"))["posed"]
        h1 = np.load(os.path.join(a.compare, "hand_hold.npz"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 24
    sc.render.resolution_x = sc.render.resolution_y = RES
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.9
    sc.world = w
    for i, (loc, e) in enumerate([((300, -200, 300), 2.5), ((-200, 300, -100), 1.0)]):
        L = bpy.data.objects.new(f"sun{i}", bpy.data.lights.new(f"sun{i}", "SUN"))
        L.data.energy = e
        L.rotation_euler = (-Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        sc.collection.objects.link(L)
    col, s = change_colours(S0, S1, T, a.max_mm)
    m_grip = material("grip", (0.62, 0.64, 0.68))
    m_chg = material("change", (1, 1, 1), attr="chg")
    m_hand = material("hand", (0.95, 0.62, 0.48), alpha=0.3)
    g0 = mesh_obj("original", S0, T)
    g1 = mesh_obj("posed", S1, T, col)
    g1c = mesh_obj("posed change", S1, T, col)
    g0.data.materials.append(m_grip)
    g1.data.materials.append(m_grip)
    g1c.data.materials.append(m_chg)
    hand = mesh_obj("hand", h["V"], h["T"])
    hand.data.materials.append(m_hand)
    hand1 = mesh_obj("hand 2", h1["V"], h1["T"])
    hand1.data.materials.append(m_hand)
    target = np.array([62.0, -50.0, 18.0])
    tiles = {}
    jobs = [(vi, key, objs) for vi in range(len(VIEWS))
            for key, objs in (("orig", [g0]), ("posed", [g1]), ("posed_chg", [g1c]))]
    jobs += [(len(VIEWS), "orig", [g0, hand]), (len(VIEWS), "posed", [g1, hand1])]
    for vi, key, objs in jobs:
        vname, d, up = VIEWS[min(vi, HAND_VIEW if vi == len(VIEWS) else vi)]
        for o in (g0, g1, g1c, hand, hand1):
            o.hide_render = o not in objs
        camera(target, d, up, 140 if vi < len(VIEWS) else 190)
        tiles[(vi, key)] = render(os.path.join(a.work, f"tile_{vi}_{key}.png"))
    # sheet: rows = views, columns = original + hand | posed + hand | what changed
    pad, head = 20, 120
    W = 3 * RES + 4 * pad
    rows = [v[0] for v in VIEWS] + [("with each hand" if a.compare else "with your hand") + " (see-through), palm side"]
    H = head + len(rows) * (RES + 60 + pad) + 150
    sheet = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(sheet)
    fb, f, fs = ImageFont.truetype(FONT_B, 44), ImageFont.truetype(FONT, 34), ImageFont.truetype(FONT, 28)
    titles = (a.titles.split("|") if a.titles else
              ["Original GripV", "After the solver", "What changed (red = grew, blue = pulled in)"])
    for c, t in enumerate(titles):
        dr.text((pad + c * (RES + pad) + RES // 2, 60), t, font=f, fill="black", anchor="mm")
    for r, vname in enumerate(rows):
        y = head + r * (RES + 60 + pad)
        dr.text((pad, y + 10), vname, font=fb, fill="black")
        for c, key in enumerate(("orig", "posed", "posed_chg")):
            if (r, key) in tiles:
                sheet.paste(tiles[(r, key)], (pad + c * (RES + pad), y + 60))
    # legend
    y = H - 120
    for i in range(400):
        t = i / 399 * 2 - 1
        c = np.array([0.8, 0.8, 0.8]) + (t if t > 0 else -t) * ((np.array([0.85, 0.08, 0.05]) if t > 0 else np.array([0.05, 0.25, 0.9])) - 0.8)
        dr.line([(W // 2 - 200 + i, y), (W // 2 - 200 + i, y + 40)], fill=tuple(int(255 * v) for v in c))
    dr.text((W // 2 - 220, y + 20), f"-{a.max_mm:.0f} mm", font=fs, fill="black", anchor="rm")
    dr.text((W // 2 + 220, y + 20), f"+{a.max_mm:.0f} mm", font=fs, fill="black", anchor="lm")
    dr.text((W // 2, y + 70), f"largest change {np.abs(s).max():.1f} mm; grey = unchanged (exception zone, flat top, triggers, battery)",
            font=fs, fill="black", anchor="mm")
    sheet.save(a.out, quality=88)
    print("[sheet] wrote", a.out)


if __name__ == "__main__":
    main()
