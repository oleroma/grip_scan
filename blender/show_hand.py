"""Build a scene with the grip and the placed hand, render views, save a .blend.

    blender -b --factory-startup --python blender/show_hand.py -- \
        --grip data/reference_basemesh_gripv.obj --hand data/hand.json \
        --placed data/hand_on_grip.json --board data/board_gripv.json --out out/hand_on_grip

Writes <out>.blend and <out>_<view>.png. Fingers are capsules (spheres at the
joints joined by cylinders) using the hand's radii; the palm is a hull of the
wrist, CMC and MCP points thickened by the palm thickness.
"""
import argparse
import json
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

FINGERS = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
           "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20]}
NAMES = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_mcp", "index_pip",
         "index_dip", "index_tip", "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
         "ring_mcp", "ring_pip", "ring_dip", "ring_tip", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]
VIEWS = {"outside": (Vector((260, -40, 20)), Vector((40, -40, 20))),
         "front": (Vector((40, 240, 20)), Vector((40, -40, 20))),
         "inside": (Vector((-180, -40, 20)), Vector((40, -40, 20))),
         "below": (Vector((60, -40, -220)), Vector((40, -40, 20)))}


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    for k in ("grip", "hand", "placed", "board", "out"):
        p.add_argument("--" + k, required=k != "board")
    return p.parse_args(argv)


def material(name, rgba):
    m = bpy.data.materials.new(name)
    m.diffuse_color = rgba
    return m


def capsule(bm, a, b, r):
    for c in (a, b):
        res = bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=r)
        bmesh.ops.translate(bm, vec=c, verts=res["verts"])
    d = b - a
    if d.length < 1e-6:
        return
    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=r, radius2=r, depth=d.length)
    rot = d.to_track_quat("Z", "Y").to_matrix().to_4x4()
    bmesh.ops.transform(bm, matrix=Matrix.Translation((a + b) / 2) @ rot, verts=res["verts"])


def main():
    a = args()
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    hand, placed = json.load(open(a.hand)), json.load(open(a.placed))
    J = [Vector(placed["joints_grip_mm"][n]) for n in NAMES]
    radii = hand["radii_mm"]
    sc = bpy.context.scene
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

    bpy.ops.wm.obj_import(filepath=os.path.abspath(a.grip), forward_axis="Y", up_axis="Z")
    grip = bpy.context.selected_objects[0]
    grip.name = "grip (current GripV)"
    grip.data.materials.append(material("grip", (0.75, 0.75, 0.78, 1)))

    me = bpy.data.meshes.new("hand")
    bm = bmesh.new()
    for f, ids in FINGERS.items():
        for s in range(3):
            r = radii[f][s] if f != "thumb" else radii[f][min(s, 2)]
            capsule(bm, J[ids[s]], J[ids[s + 1]], r)
    # palm: hull of the palm points pushed half the thickness both ways along the palm normal
    pal = [J[i] for i in (0, 1, 5, 9, 13, 17)]
    normal = (J[9] - J[0]).cross(J[5] - J[17]).normalized()
    t = hand["palm_thickness_mm"] / 2
    verts = [bm.verts.new(p + s * t * normal) for p in pal for s in (-1, 1)]
    bmesh.ops.convex_hull(bm, input=verts)
    bm.to_mesh(me)
    bm.free()
    ho = bpy.data.objects.new("relaxed hand", me)
    sc.collection.objects.link(ho)
    ho.data.materials.append(material("hand", (0.95, 0.65, 0.5, 1)))
    rem = ho.modifiers.new("unite", "REMESH")
    rem.mode, rem.voxel_size = "VOXEL", 1.0

    if a.board:
        board = json.load(open(a.board))
        for key, col in (("r2", (1, 0.1, 0.1, 1)), ("r4", (0.1, 0.8, 0.2, 1))):
            bpy.ops.mesh.primitive_uv_sphere_add(radius=2.5, location=board[key]["press_centre"])
            m = bpy.context.object
            m.name = key + " press"
            m.data.materials.append(material(key, col))
            m.show_in_front = True

    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light, sh.color_type = "STUDIO", "MATERIAL"
    sc.render.resolution_x = sc.render.resolution_y = 900
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type, cam_data.ortho_scale = "ORTHO", 230
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    for name, (loc, target) in VIEWS.items():
        cam.location = loc
        cam.rotation_euler = (target - loc).to_track_quat("-Z", "Y" if name != "below" else "X").to_euler()
        sc.render.filepath = f"{out}_{name}.png"
        bpy.ops.render.render(write_still=True)
        print("WROTE", sc.render.filepath)
    bpy.ops.wm.save_as_mainfile(filepath=out + ".blend")
    print("WROTE", out + ".blend")


main()
