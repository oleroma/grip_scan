"""Export the minimal board description the grip solver needs from Grip.blend.

Run headless; the .blend is never saved:

    blender -b D:/Grip/blender/Grip.blend --python blender/export_board.py -- \
        --out data/board_gripv.json --ref-mesh data/reference_basemesh_gripv.obj

What is exported (right half, X >= 0, Blender units = mm):
- top_face: the fixed flat top face of `basemesh GripV` (hard constraint)
- thumb_targets: top centres of the right-side controls on that face
- r1: a target point + normal on the R1 bumper (index finger)
- r2 / r4: current press centre + normal of each arch trigger (middle / ring
  fingertip), found by comparing the released and pressed (`action`) meshes,
  plus a default allowed region around it
- keep_out: bounding box of the generated PCB pocket (`pcb cutout`)
- the evaluated basemesh (mirrored + subdivided) as a reference OBJ
"""
import argparse
import json
import sys

import bpy
from mathutils import Vector

SCENE = "GripV"
VARIANT = "GripV"
BASEMESH = "basemesh GripV"
TRIGGERS = {
    # name: (parameter group, active parameter node, finger)
    "R2": ("GripV R2 parameters", "trigger parameters.001", "middle"),
    "R4": ("GripV R4 parameters", "trigger parameters.001", "ring"),
}
THUMB_CONTROLS = [
    "abxy cap up", "abxy cap down", "abxy cap left", "abxy cap right",
    "StickR", "select cap top right", "select cap down right", "home cap",
    "scrollwheel",
]
PCB_CUTOUT = "pcb cutout"  # generated board pocket; lives in the "source" scene
# (the PCB/FRONTPCB/BACKPCB models in the file are not placed at the board position)
REGION_RADIUS = 15.0  # default allowed radius around the current R2/R4 press centre


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--ref-mesh", default=None)
    return p.parse_args(argv)


def vec(v, nd=3):
    return [round(x, nd) for x in v]


def depsgraph(scene):
    for o in bpy.data.objects:
        o.update_tag()
    with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
        dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    return dg


def evaluated(obj, dg):
    """World-space vertices and faces (centre, normal, area, vertex ids)."""
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    mw = ev.matrix_world
    rot = mw.to_3x3()
    verts = [mw @ v.co for v in me.vertices]
    faces = [(mw @ p.center, (rot @ p.normal).normalized(), p.area, list(p.vertices))
             for p in me.polygons]
    tris = [tuple(t.vertices) for t in me.loop_triangles] if me.loop_triangles else None
    ev.to_mesh_clear()
    return verts, faces, tris


def weighted(faces):
    area = sum(f[2] for f in faces)
    c = sum((f[0] * f[2] for f in faces), Vector()) / area
    n = sum((f[1] * f[2] for f in faces), Vector()).normalized()
    return c, n, area


def top_face():
    me = bpy.data.objects[BASEMESH].data  # raw half mesh, before modifiers
    up = [p for p in me.polygons if p.normal.z > 0.99]
    top = max(up, key=lambda p: p.center.z)
    loop = [vec(me.vertices[i].co) for i in top.vertices]
    return {"z": round(top.center.z, 3), "polygon": loop, "area": round(top.area, 1),
            "note": "raw half-mesh n-gon of basemesh GripV; fixed, never moved by the solver"}


def control_top(obj, dg):
    """Centre of the highest region of a control (its top surface)."""
    verts, _, _ = evaluated(obj, dg)
    zmax = max(v.z for v in verts)
    top = [v for v in verts if v.z > zmax - 0.3]
    c = sum(top, Vector()) / len(top)
    return {"name": obj.name, "top_centre": vec(c)}


def trigger_press(name, group, node, dg_fn):
    sock = bpy.data.node_groups[group].nodes[node].inputs["action"]
    original = sock.default_value
    sock.default_value = False
    v0, f0, _ = evaluated(bpy.data.objects[name], dg_fn())
    sock.default_value = True
    v1, _, _ = evaluated(bpy.data.objects[name], dg_fn())
    sock.default_value = original
    if len(v0) != len(v1):
        raise RuntimeError(f"{name}: pressed mesh has different topology")
    move = [b - a for a, b in zip(v0, v1)]
    switch = bpy.data.objects[name + "_pcb_button"].matrix_world.translation
    # faces the finger pushes on: their normal points against the press motion
    push = []
    for c, n, a, ids in f0:
        m = sum((move[i] for i in ids), Vector()) / len(ids)
        if m.length > 1e-6 and n.dot(-m.normalized()) > 0.5:
            push.append((c, n, a, ids, (c - switch).length))
    if not push:
        raise RuntimeError(f"{name}: no pressing faces found")
    dmin = min(p[4] for p in push)
    dmax = max(p[4] for p in push)
    tip = [p for p in push if p[4] >= dmin + 0.6 * (dmax - dmin)]  # finger end of the bar
    c, n, area = weighted(tip)
    travel = max(m.length for m in move)
    return {"press_centre": vec(c), "press_normal": vec(n), "press_area": round(area, 1),
            "switch": vec(switch), "preview_travel": round(travel, 3),
            "allowed_region": {"type": "sphere", "centre": vec(c), "radius": REGION_RADIUS}}


def bumper_target(dg):
    _, faces, _ = evaluated(bpy.data.objects["R1"], dg)
    top = [f for f in faces if f[1].z > 0.5 and f[0].x > 0]
    c, n, area = weighted(top)
    return {"target": vec(c), "normal": vec(n), "area": round(area, 1),
            "note": "upward-facing surface of R1; the index finger may wrap it, not only the tip"}


def bbox(name, dg):
    pts = evaluated(bpy.data.objects[name], dg)[0]
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    return {"type": "box", "min": vec(lo), "max": vec(hi), "object": name}


def write_obj(path, verts, tris):
    with open(path, "w") as f:
        f.write("# basemesh GripV, evaluated (MirrorX + subdivision), mm\n")
        for v in verts:
            f.write(f"v {v.x:.4f} {v.y:.4f} {v.z:.4f}\n")
        for t in tris:
            f.write("f " + " ".join(str(i + 1) for i in t) + "\n")


def main():
    args = parse_args()
    scene = bpy.data.scenes[SCENE]
    # Grip.blend may be saved on GripX; the triggers follow the global variant.
    variant = bpy.data.node_groups["global"].nodes["global parameters"].inputs["variant"]
    saved_variant = variant.default_value
    variant.default_value = VARIANT
    dg_fn = lambda: depsgraph(scene)
    dg = dg_fn()

    board = {
        "source": {"blend": bpy.data.filepath, "scene": SCENE, "variant": VARIANT,
                   "saved_variant": saved_variant, "blender": bpy.app.version_string},
        "units": "mm",
        "side": "right half, X >= 0; mirrored at X = 0",
        "parting_plane_z": 40.0,
        "top_face": top_face(),
        "thumb_targets": [control_top(bpy.data.objects[n], dg) for n in THUMB_CONTROLS
                          if n in bpy.data.objects],
        "r1": {"finger": "index", **bumper_target(dg)},
        "keep_out": [bbox(PCB_CUTOUT, depsgraph(bpy.data.scenes["source"]))],
    }
    for name, (group, node, finger) in TRIGGERS.items():
        board[name.lower()] = {"finger": finger, **trigger_press(name, group, node, dg_fn)}

    with open(args.out, "w") as f:
        json.dump(board, f, indent=1)
    print("WROTE", args.out)

    if args.ref_mesh:
        ev = bpy.data.objects[BASEMESH].evaluated_get(dg_fn())
        me = ev.to_mesh()
        me.calc_loop_triangles()
        mw = ev.matrix_world
        write_obj(args.ref_mesh, [mw @ v.co for v in me.vertices],
                  [tuple(t.vertices) for t in me.loop_triangles])
        ev.to_mesh_clear()
        print("WROTE", args.ref_mesh)


main()
