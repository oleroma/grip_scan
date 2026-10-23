"""Render board.json targets on top of the GripV basemesh, to check them by eye.

    blender -b D:/Grip/blender/Grip.blend --python blender/preview_board.py -- \
        --board data/board_gripv.json --out data/board_preview

Writes <out>_side.png, <out>_front.png and <out>_top.png. The .blend is not saved.
"""
import argparse
import json
import os
import sys

import bpy
from mathutils import Vector

COLOURS = {"thumb": (0.1, 0.4, 1.0, 1), "r1": (1.0, 0.8, 0.0, 1),
           "r2": (1.0, 0.1, 0.1, 1), "r4": (0.1, 0.9, 0.2, 1)}
VIEWS = {  # name: (camera location, look-at)
    "side": ((260, -35, 35), (0, -35, 35)),
    "front": ((30, 260, 35), (30, 0, 35)),
    "top": ((30, -35, 300), (30, -35, 0)),
}


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--board", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def material(name, rgba):
    m = bpy.data.materials.new(name)
    m.diffuse_color = rgba
    return m


def marker(coll, name, loc, rgba, radius=2.5, normal=None):
    me = bpy.data.meshes.new(name)
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=radius)
    if normal is not None:  # a short stick showing the press direction
        tip = Vector(normal) * 10
        for v in bmesh.ops.create_cone(bm, cap_ends=True, segments=8, radius1=0.6,
                                       radius2=0.6, depth=10)["verts"]:
            v.co = v.co.z * Vector(normal) + tip / 2 + (v.co - Vector((0, 0, v.co.z)))
    bm.to_mesh(me)
    o.location = loc
    o.data.materials.append(material(name, rgba))
    o.show_in_front = True


def main():
    a = args()
    a.out = os.path.abspath(a.out)
    board = json.load(open(a.board))
    bpy.data.node_groups["global"].nodes["global parameters"].inputs["variant"].default_value = "GripV"
    sc = bpy.data.scenes["GripV"]
    root = sc.collection
    # hide everything except the triggers and the bumper; show the basemesh instead of the slow case
    for lc in sc.view_layers[0].layer_collection.children:
        lc.exclude = True
    coll = bpy.data.collections.new("preview")
    root.children.link(coll)
    for nm in ("basemesh GripV", "R2", "R4", "R1"):
        coll.objects.link(bpy.data.objects[nm])
    for c in ("abxy caps", "select caps", "home cap"):
        for o in bpy.data.collections[c].objects:
            coll.objects.link(o)
    coll.objects.link(bpy.data.objects["StickR"])

    for t in board["thumb_targets"]:
        marker(coll, "thumb " + t["name"], t["top_centre"], COLOURS["thumb"], 1.8)
    marker(coll, "R1 target", board["r1"]["target"], COLOURS["r1"], normal=board["r1"]["normal"])
    for k in ("r2", "r4"):
        marker(coll, k + " press", board[k]["press_centre"], COLOURS[k], normal=board[k]["press_normal"])

    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light = "STUDIO"
    sh.color_type = "MATERIAL"
    sh.show_xray = True
    sh.xray_alpha = 0.6
    sc.render.resolution_x, sc.render.resolution_y = 1200, 1000
    sc.render.film_transparent = False

    cam_data = bpy.data.cameras.new("preview cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 170
    cam = bpy.data.objects.new("preview cam", cam_data)
    root.objects.link(cam)
    sc.camera = cam
    for name, (loc, target) in VIEWS.items():
        cam.location = loc
        d = Vector(target) - Vector(loc)
        cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = f"{a.out}_{name}.png"
        with bpy.context.temp_override(scene=sc):
            bpy.ops.render.render(write_still=True, scene=sc.name)
        print("WROTE", sc.render.filepath)


main()
