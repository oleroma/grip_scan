"""Build a posable hand (armature + capsule meshes) on the grip, for posing by hand.

    blender -b --factory-startup --python blender/make_rig.py -- \
        --grip data/reference_basemesh_gripv.obj --hand data/hand.json \
        --placed build/gen/data/hand_posed_on_grip.json --board data/board_gripv.json --out build/gen/out/hand_rig.blend

Open the .blend, select "hand rig", Pose Mode: rotate "palm" (moves the whole
hand; G to move it too) and rotate the finger bones (R). Bone lengths are your
measured segment lengths, so only angles and the hand placement change. Save,
then read the pose back with blender/read_rig.py.
"""
import argparse
import json
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

NAMES = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_mcp", "index_pip",
         "index_dip", "index_tip", "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
         "ring_mcp", "ring_pip", "ring_dip", "ring_tip", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]
FINGERS = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
           "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20]}
SEG = {"thumb": ["metacarpal", "proximal", "distal"]}
SEG_DEFAULT = ["proximal", "middle", "distal"]


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    for k in ("grip", "hand", "placed", "board", "out"):
        p.add_argument("--" + k, required=True)
    return p.parse_args(argv)


def material(name, rgba):
    m = bpy.data.materials.new(name)
    m.diffuse_color = rgba
    return m


def capsule_mesh(name, a, b, r):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    for c in (a, b):
        res = bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=r)
        bmesh.ops.translate(bm, vec=c, verts=res["verts"])
    d = b - a
    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=r, radius2=r, depth=d.length)
    rot = d.to_track_quat("Z", "Y").to_matrix().to_4x4()
    bmesh.ops.transform(bm, matrix=Matrix.Translation((a + b) / 2) @ rot, verts=res["verts"])
    bm.to_mesh(me)
    bm.free()
    return me


def parent_to_bone(obj, arm, bone_name):
    obj.parent = arm
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    b = arm.data.bones[bone_name]
    # bone parenting attaches at the bone's tail; cancel that so the mesh stays where it was built
    obj.matrix_parent_inverse = (arm.matrix_world @ b.matrix_local @ Matrix.Translation((0, b.length, 0))).inverted()


def marker(name, loc, rgba, r=2.5):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=loc)
    m = bpy.context.object
    m.name = name
    m.data.materials.append(material(name, rgba))
    m.show_in_front = True
    m.hide_select = True


def main():
    a = args()
    hand, placed, board = json.load(open(a.hand)), json.load(open(a.placed)), json.load(open(a.board))
    J = {n: Vector(placed["joints_grip_mm"][n]) for n in NAMES}
    radii = hand["radii_mm"]
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 0.001
    sc.unit_settings.length_unit = "MILLIMETERS"

    bpy.ops.wm.obj_import(filepath=os.path.abspath(a.grip), forward_axis="Y", up_axis="Z")
    grip = bpy.context.selected_objects[0]
    grip.name = "grip (current GripV)"
    grip.data.materials.append(material("grip", (0.75, 0.75, 0.78, 1)))
    grip.hide_select = True

    marker("R1 target", board["r1"]["target"], (1.0, 0.8, 0.0, 1))
    marker("R2 press", board["r2"]["press_centre"], (1, 0.1, 0.1, 1))
    marker("R4 press", board["r4"]["press_centre"], (0.1, 0.8, 0.2, 1))
    if placed.get("palm_swell_point"):
        marker("palm swell", placed["palm_swell_point"], (0.2, 0.4, 1.0, 1), 3.0)

    # armature: palm (wrist -> knuckle-row centre) is the root; each finger chain hangs off it
    arm_data = bpy.data.armatures.new("hand rig")
    arm = bpy.data.objects.new("hand rig", arm_data)
    sc.collection.objects.link(arm)
    arm.show_in_front = True
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones
    knuckles = (J["index_mcp"] + J["middle_mcp"] + J["ring_mcp"] + J["pinky_mcp"]) / 4
    palm = eb.new("palm")
    palm.head, palm.tail = J["wrist"], knuckles
    palm_normal = (J["middle_mcp"] - J["wrist"]).cross(J["index_mcp"] - J["pinky_mcp"]).normalized()
    palm.align_roll(palm_normal)
    # metacarpals: the palm arches (ring and pinky fold towards the palm, index and middle
    # a little), so every finger hangs off its own metacarpal bone rooted at the wrist
    metas = {}
    for f in ("index", "middle", "ring", "pinky"):
        m = eb.new(f"{f}_metacarpal")
        m.head, m.tail = J["wrist"], J[f + "_mcp"]
        m.parent = palm
        m.align_roll(palm_normal)
        metas[f] = m
    for f, ids in FINGERS.items():
        parent = metas.get(f, palm)
        for s in range(3):
            b = eb.new(f"{f}_{(SEG.get(f, SEG_DEFAULT))[s]}")
            b.head, b.tail = J[NAMES[ids[s]]], J[NAMES[ids[s + 1]]]
            b.parent = parent
            b.use_connect = s > 0 or f in metas
            b.align_roll(palm_normal)
            parent = b
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in arm.pose.bones:
        if pb.name != "palm":
            pb.lock_location = (True, True, True)  # fingers rotate only; lengths stay measured
        pb.lock_scale = (True, True, True)
        pb.rotation_mode = "XYZ"

    skin = material("hand", (0.95, 0.65, 0.5, 1))
    for f, ids in FINGERS.items():
        for s in range(3):
            bn = f"{f}_{(SEG.get(f, SEG_DEFAULT))[s]}"
            r = radii[f][min(s, 2)]
            o = bpy.data.objects.new(bn + " skin", capsule_mesh(bn, J[NAMES[ids[s]]], J[NAMES[ids[s + 1]]], r))
            sc.collection.objects.link(o)
            o.data.materials.append(skin)
            o.hide_select = True
            parent_to_bone(o, arm, bn)
    # palm skin: a thick capsule along every metacarpal, so the palm follows the arch
    t = hand["palm_thickness_mm"] / 2
    for f in ("index", "middle", "ring", "pinky"):
        a_ = J["wrist"] + 0.25 * (J[f + "_mcp"] - J["wrist"])
        o = bpy.data.objects.new(f"{f}_metacarpal skin", capsule_mesh(f + "_meta", a_, J[f + "_mcp"], t))
        sc.collection.objects.link(o)
        o.data.materials.append(skin)
        o.hide_select = True
        parent_to_bone(o, arm, f"{f}_metacarpal")
    o = bpy.data.objects.new("palm heel skin", capsule_mesh("heel", J["wrist"], J["thumb_cmc"], t))
    sc.collection.objects.link(o)
    o.data.materials.append(skin)
    o.hide_select = True
    parent_to_bone(o, arm, "palm")

    arm["grip_scan"] = json.dumps({"names": NAMES, "fingers": FINGERS})
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out))
    print("WROTE", os.path.abspath(a.out))


main()
