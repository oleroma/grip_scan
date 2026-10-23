"""build_grip_rig.py: put the 'grip rig' armature inside the GripV basemesh cage.

    PY=/tmp/claude-0/hs/venv/bin/python      # bpy 5.2 as a module
    $PY build_grip_rig.py [--grip /home/claude/Grip/blender/Grip.blend]
                          [--hold /mnt/project-files/hand_scan_rig/hold_upright/hand_on_grip_upright.blend]
                          [--out /mnt/project-files/hand_scan_rig/grip_rig/grip_rig.blend]
                          [--work /tmp/claude-0/hs/work/grip_rig]

What it does (phase 1: rig only, rest pose = the original mesh):
  1. opens Olex's Grip.blend, selects the GripV variant
  2. measures the subdivision of the cage (it is linear: outputs = A @ cage), so it knows exactly
     which cage vertices reach into the exception zone
  3. appends the scanned hand in the approved hold (hold_upright) into its own collection
  4. finds the bone ends (palm swell, fins, stand, thenar surface) on the grip, pins the frozen
     parts, computes smooth weights
  5. adds the armature 'grip rig' and three modifiers at the top of the cage's stack:
       'grip rig rest'  (GN)       remembers the cage as it is (attribute rig_rest)
       'grip rig'       (Armature) the bones deform the cage
       'grip rig pins'  (GN)       puts frozen vertices back exactly; per axis: mirror seam X,
                                   flat top Z (and the top back edge may only slide in Y)
  6. saves a copy (never over Grip.blend) and writes rig.json + rig_state.npz to --work
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
from mathutils import Vector  # noqa: E402

CAGE = "basemesh GripV"
RIG = "grip rig"
RIG_COLL = "grip rig"
HAND_COLL = "hand reference (hold_upright)"
ZONE_OBJ = "exception zone (never changes)"
MOD_REST, MOD_ARM, MOD_PINS = "grip rig rest", "grip rig", "grip rig pins"
PROTECTED = {"/home/claude/Grip/blender/Grip.blend"}


def log(*a):
    print("[grip_rig]", *a, flush=True)


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grip", default="/home/claude/Grip/blender/Grip.blend")
    ap.add_argument("--hold", default="/mnt/project-files/hand_scan_rig/hold_upright/hand_on_grip_upright.blend")
    ap.add_argument("--zone", default=rc.ZONE_STL)
    ap.add_argument("--out", default="/mnt/project-files/hand_scan_rig/grip_rig/grip_rig.blend")
    ap.add_argument("--work", default="/tmp/claude-0/hs/work/grip_rig")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return ap.parse_args(argv)


# ---------------------------------------------------------------- evaluation helpers
def eval_mesh(ob, dg=None, world=True):
    dg = dg or bpy.context.evaluated_depsgraph_get()
    e = ob.evaluated_get(dg)
    me = e.to_mesh()
    me.calc_loop_triangles()
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    T = np.empty(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", T)
    T = T.reshape(-1, 3)
    e.to_mesh_clear()
    if world:
        M = np.array(ob.matrix_world)
        V = V @ M[:3, :3].T + M[:3, 3]
    return V, T


def scratch_scene(names):
    """A scene holding only the named objects, so evaluating never touches the whole Grip scene."""
    sc = bpy.data.scenes.new("grip rig scratch")
    for n in names:
        sc.collection.objects.link(bpy.data.objects[n])
    return sc


def subdivision_matrix(cage):
    """A (n_out, n_cage): how much each cage vertex moves each output vertex (Z test moves;
    MirrorX + subdivision surface are linear in the cage positions)."""
    me = cage.data
    co0 = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co0)
    co0 = co0.reshape(-1, 3)
    base, T = eval_mesh(cage, world=False)
    A = np.zeros((len(base), len(co0)))
    for i in range(len(co0)):
        c = co0.copy()
        c[i, 2] += 1.0
        me.vertices.foreach_set("co", c.ravel())
        me.update()
        V, _ = eval_mesh(cage, world=False)
        A[:, i] = V[:, 2] - base[:, 2]
    me.vertices.foreach_set("co", co0.ravel())
    me.update()
    A[np.abs(A) < 1e-5] = 0.0                       # float32 noise of the evaluation
    return A, base, T, co0


# ---------------------------------------------------------------- hand reference
def append_hand(path):
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        want = ["hand scan", "hand rig", "palm swell", "R1 target", "R2 press", "R4 press"]
        dst.objects = [n for n in src.objects if n in want]
    coll = bpy.data.collections.get(HAND_COLL) or bpy.data.collections.new(HAND_COLL)
    if coll.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(coll)
    objs = {}
    for o in dst.objects:
        if o is None:
            continue
        coll.objects.link(o)
        objs[o.name.split(".")[0]] = o
    for n, o in objs.items():
        if n in ("palm swell", "R1 target", "R2 press", "R4 press"):
            o.name = "hold marker: " + n
            o.hide_render = True
    hs = objs["hand scan"]
    hs.display_type = "SOLID"
    hs.show_transparent = True
    hs.color = (0.85, 0.62, 0.5, 0.45)
    return objs


def hand_bones(rig):
    bpy.context.view_layer.update()
    M = rig.matrix_world
    return {pb.name: [list(M @ pb.head), list(M @ pb.tail)] for pb in rig.pose.bones}


# ---------------------------------------------------------------- GN node groups
def _node(nt, kind, loc, **props):
    n = nt.nodes.new(kind)
    n.location = loc
    for k, v in props.items():
        setattr(n, k, v)
    return n


def _sock(sockets, name):
    for s in sockets:
        if s.name == name and s.enabled:
            return s
    for s in sockets:
        if s.name == name:
            return s
    raise KeyError(name)


def make_rest_group():
    ng = bpy.data.node_groups.get(MOD_REST) or bpy.data.node_groups.new(MOD_REST, "GeometryNodeTree")
    ng.nodes.clear()
    ng.interface.clear()
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi = _node(ng, "NodeGroupInput", (-400, 0))
    go = _node(ng, "NodeGroupOutput", (300, 0))
    pos = _node(ng, "GeometryNodeInputPosition", (-400, -150))
    st = _node(ng, "GeometryNodeStoreNamedAttribute", (0, 0), data_type="FLOAT_VECTOR", domain="POINT")
    _sock(st.inputs, "Name").default_value = "rig_rest"
    L = ng.links
    L.new(gi.outputs[0], _sock(st.inputs, "Geometry"))
    L.new(pos.outputs[0], _sock(st.inputs, "Value"))
    L.new(_sock(st.outputs, "Geometry"), go.inputs[0])
    return ng


def make_pins_group():
    """position = rest + (position - rest) * rig_axis_mask; moves below 1e-4 mm (armature float noise) snap to rest
    (so the rest pose is bit-exact); then the helper attributes are removed."""
    ng = bpy.data.node_groups.get(MOD_PINS) or bpy.data.node_groups.new(MOD_PINS, "GeometryNodeTree")
    ng.nodes.clear()
    ng.interface.clear()
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    L = ng.links
    gi = _node(ng, "NodeGroupInput", (-900, 0))
    go = _node(ng, "NodeGroupOutput", (700, 0))
    pos = _node(ng, "GeometryNodeInputPosition", (-900, -200))
    rest = _node(ng, "GeometryNodeInputNamedAttribute", (-900, -350), data_type="FLOAT_VECTOR")
    _sock(rest.inputs, "Name").default_value = "rig_rest"
    msk = _node(ng, "GeometryNodeInputNamedAttribute", (-900, -500), data_type="FLOAT_VECTOR")
    _sock(msk.inputs, "Name").default_value = "rig_axis_mask"
    sub = _node(ng, "ShaderNodeVectorMath", (-650, -250), operation="SUBTRACT")
    L.new(pos.outputs[0], sub.inputs[0])
    L.new(_sock(rest.outputs, "Attribute"), sub.inputs[1])
    mul = _node(ng, "ShaderNodeVectorMath", (-450, -300), operation="MULTIPLY")
    L.new(sub.outputs[0], mul.inputs[0])
    L.new(_sock(msk.outputs, "Attribute"), mul.inputs[1])
    ln = _node(ng, "ShaderNodeVectorMath", (-250, -450), operation="LENGTH")
    L.new(mul.outputs[0], ln.inputs[0])
    cmp_ = _node(ng, "FunctionNodeCompare", (-50, -450), data_type="FLOAT", operation="LESS_THAN")
    L.new(_sock(ln.outputs, "Value"), cmp_.inputs[0])
    cmp_.inputs[1].default_value = 1e-4
    add = _node(ng, "ShaderNodeVectorMath", (-250, -250), operation="ADD")
    L.new(_sock(rest.outputs, "Attribute"), add.inputs[0])
    L.new(mul.outputs[0], add.inputs[1])
    sw = _node(ng, "GeometryNodeSwitch", (100, -250), input_type="VECTOR")
    L.new(_sock(cmp_.outputs, "Result"), _sock(sw.inputs, "Switch"))
    L.new(add.outputs[0], _sock(sw.inputs, "False"))
    L.new(_sock(rest.outputs, "Attribute"), _sock(sw.inputs, "True"))
    sp = _node(ng, "GeometryNodeSetPosition", (300, 0))
    L.new(gi.outputs[0], _sock(sp.inputs, "Geometry"))
    L.new(sw.outputs[0], _sock(sp.inputs, "Position"))
    rm = _node(ng, "GeometryNodeRemoveAttribute", (500, 0))
    L.new(_sock(sp.outputs, "Geometry"), _sock(rm.inputs, "Geometry"))
    try:
        rm.pattern_mode = "WILDCARD"
        _sock(rm.inputs, "Name").default_value = "rig_*"
    except Exception:
        _sock(rm.inputs, "Name").default_value = "rig_rest"
    L.new(_sock(rm.outputs, "Geometry"), go.inputs[0])
    return ng


# ---------------------------------------------------------------- armature
def build_armature(marks, top_edge_pts, coll):
    old = bpy.data.objects.get(RIG)
    if old is not None:
        bpy.data.objects.remove(old)
    data = bpy.data.armatures.new(RIG)
    rig = bpy.data.objects.new(RIG, data)
    coll.objects.link(rig)
    data.display_type = "OCTAHEDRAL"
    rig.show_in_front = True
    data.show_names = True
    view = bpy.context.view_layer
    for o in view.objects:
        o.select_set(False)
    view.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = data.edit_bones
    P0 = Vector(marks[rc.PALM]["p"])
    root = eb.new(rc.ROOT)
    root.head, root.tail = Vector((0.0, -50.0, 30.0)), Vector((0.0, -50.0, 45.0))
    ps = eb.new(rc.PALM)
    ps.head = P0
    ps.tail = P0 + Vector(marks[rc.PALM]["n"]) * rc.PARAMS["palm_tail_mm"]
    ps.parent = root
    ps.align_roll(Vector((0.0, 0.0, 1.0)))
    for name in rc.FINS + (rc.STAND, rc.THENAR):
        b = eb.new(name)
        b.head = P0
        b.tail = Vector(marks[name]["p"])
        b.parent = ps
        b.use_connect = False
        b.align_roll(Vector(marks[name]["n"]))          # local Z = outward surface normal at the tip
    te = eb.new(rc.TOP_EDGE)
    c = Vector(top_edge_pts.mean(0))
    te.head, te.tail = c, c + Vector((0.0, -12.0, 0.0))   # local Y = world -Y (backwards, towards the palm)
    te.parent = root
    te.align_roll(Vector((0.0, 0.0, 1.0)))
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in rig.pose.bones:
        pb.rotation_mode = "XYZ"
        pb.lock_scale = (True, True, True)
    pb = rig.pose.bones[rc.TOP_EDGE]
    pb.lock_location = (True, False, True)
    pb.lock_rotation = (True, True, True)
    rig.pose.bones[rc.ROOT].lock_location = (True, True, True)
    rig.pose.bones[rc.ROOT].lock_rotation = (True, True, True)
    # colours: one bold colour per bone (Blender 4+ bone colours)
    cols = {rc.PALM: "THEME01", "fin middle": "THEME03", "fin ring": "THEME04", "fin pinky": "THEME09",
            rc.STAND: "THEME07", rc.THENAR: "THEME06", rc.TOP_EDGE: "THEME10", rc.ROOT: "DEFAULT"}
    for b in data.bones:
        try:
            b.color.palette = cols.get(b.name, "DEFAULT")
        except Exception:
            pass
    return rig


def write_groups(cage, W):
    for name, w in W.items():
        vg = cage.vertex_groups.get(name) or cage.vertex_groups.new(name=name)
        vg.remove(list(range(len(w))))
        for i in np.nonzero(w > 1e-5)[0]:
            vg.add([int(i)], float(w[i]), "REPLACE")


def write_mask(cage, mask):
    me = cage.data
    a = me.attributes.get("rig_axis_mask")
    if a is not None:
        me.attributes.remove(a)
    a = me.attributes.new("rig_axis_mask", "FLOAT_VECTOR", "POINT")
    a.data.foreach_set("vector", mask.astype(np.float32).ravel())


def install_modifiers(cage, rig):
    for n in (MOD_PINS, MOD_ARM, MOD_REST):
        m = cage.modifiers.get(n)
        if m is not None:
            cage.modifiers.remove(m)
    mr = cage.modifiers.new(MOD_REST, "NODES")
    mr.node_group = make_rest_group()
    ma = cage.modifiers.new(MOD_ARM, "ARMATURE")
    ma.object = rig
    ma.use_vertex_groups = True
    ma.use_bone_envelopes = False
    ma.use_deform_preserve_volume = False
    mp = cage.modifiers.new(MOD_PINS, "NODES")
    mp.node_group = make_pins_group()
    for i, n in enumerate((MOD_REST, MOD_ARM, MOD_PINS)):
        cage.modifiers.move(cage.modifiers.find(n), i)
    cage.data.update()
    return [m.name for m in cage.modifiers]


def add_zone_object(ZV, ZF, coll):
    me = bpy.data.meshes.new(ZONE_OBJ)
    me.from_pydata(ZV.tolist(), [], ZF.tolist())
    o = bpy.data.objects.get(ZONE_OBJ)
    if o is not None:
        bpy.data.objects.remove(o)
    o = bpy.data.objects.new(ZONE_OBJ, me)
    coll.objects.link(o)
    o.display_type = "WIRE"
    o.hide_render = True
    o.hide_select = True
    return o


# ---------------------------------------------------------------- main
def main():
    a = args()
    t0 = time.time()
    assert os.path.abspath(a.out) not in PROTECTED, "refusing to overwrite Grip.blend"
    os.makedirs(a.work, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=a.grip)
    gp = bpy.data.node_groups["global"].nodes["global parameters"]
    gp.inputs["variant"].default_value = "GripV"
    cage = bpy.data.objects[CAGE]
    assert not any(m.name in (MOD_REST, MOD_ARM, MOD_PINS) for m in cage.modifiers), "cage is already rigged"
    win_scene = bpy.context.window.scene
    sc = scratch_scene([CAGE, "R2", "R4", "battery bool"])
    bpy.context.window.scene = sc
    A, S0, T, co = subdivision_matrix(cage)
    log("subdivision matrix", A.shape, f"{time.time() - t0:.1f} s")
    dg = bpy.context.evaluated_depsgraph_get()
    trig = np.vstack([eval_mesh(bpy.data.objects[n], dg)[0] for n in ("R2", "R4")])
    bat, _ = eval_mesh(bpy.data.objects["battery bool"], dg)
    bpy.context.window.scene = win_scene
    bpy.data.scenes.remove(sc)

    # hand reference + bone ends
    hand = append_hand(a.hold)
    hb = hand_bones(hand["hand rig"])
    ps_obj = bpy.data.objects["hold marker: palm swell"]
    palm = np.array(ps_obj.matrix_world.translation)

    zone = rc.zone_mesh(a.zone)
    cz, infl, oz = rc.zone_pins(co, A, S0, zone)
    top = rc.top_face(co)
    top_edge = top & ~cz                                   # top-face vertices outside the zone
    bat_lo = np.array([0.0, bat[:, 1].min(), bat[:, 2].min()])          # box in (|x|, y, z)
    bat_hi = np.array([np.abs(bat[:, 0]).max(), bat[:, 1].max(), bat[:, 2].max()])
    bat_pin = rc.battery_pins(co, bat_lo, bat_hi, rc.PARAMS["battery_margin_mm"])
    pinned = cz | infl | bat_pin                           # never move (top edge: Y only, see axis mask)
    ramp_src = pinned | top_edge
    marks0 = rc.landmarks(S0, T, hb, palm)
    W, free = rc.compute_weights(co, marks0, ramp_src, trig, top_edge)
    out_free = rc.subdivide(A, S0, np.repeat(free[:, None], 3, 1))[:, 2]
    marks = rc.landmarks(S0, T, hb, palm, out_free=out_free)
    W, free = rc.compute_weights(co, marks, ramp_src, trig, top_edge)
    mask = rc.axis_mask(co, pinned & ~top_edge, top_edge)
    log("pins: zone", int(cz.sum()), "reach into zone", int((infl & ~top_edge).sum()), "top back edge (Y only)",
        int(top_edge.sum()), "battery guard extra", int((bat_pin & ~cz & ~infl).sum()), "free", int((~pinned).sum()))

    rc_coll = bpy.data.collections.get(RIG_COLL) or bpy.data.collections.new(RIG_COLL)
    if rc_coll.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(rc_coll)
    rig = build_armature(marks, co[top_edge], rc_coll)
    write_groups(cage, W)
    write_mask(cage, mask)
    stack = install_modifiers(cage, rig)
    add_zone_object(zone[0], zone[1], rc_coll)
    log("modifier stack", stack)

    bones = {b.name: {"head": np.round(np.array(rig.matrix_world @ b.head_local), 2).tolist(),
                      "tail": np.round(np.array(rig.matrix_world @ b.tail_local), 2).tolist(),
                      "parent": b.parent.name if b.parent else None} for b in rig.data.bones}
    info = {
        "bones": bones,
        "landmarks": {k: {"p": np.round(v["p"], 2).tolist(), "n": np.round(v["n"], 3).tolist(), "how": v["how"],
                          **({"crease_hit": np.round(v["crease_hit"], 2).tolist()} if "crease_hit" in v else {})}
                      for k, v in marks.items()},
        "pins": {"cage_vertices": int(len(co)), "in_zone": int(cz.sum()),
                 "reach_into_zone_through_subdivision": int((infl & ~top_edge).sum()),
                 "top_back_edge_Y_only": int(top_edge.sum()),
                 "battery_guard_extra": int((bat_pin & ~cz & ~infl).sum()),
                 "free": int((~pinned).sum()),
                 "subdiv_outputs": int(len(S0)), "subdiv_outputs_in_zone": int(oz.sum())},
        "weights": {k: {"max": round(float(w.max()), 3), "vertices": int((w > 1e-3).sum())} for k, w in W.items()},
        "params": rc.PARAMS,
        "modifier_stack": stack,
        "trigger_points": int(len(trig)),
        "battery_box": [np.round(bat.min(0), 1).tolist(), np.round(bat.max(0), 1).tolist()],
    }
    rig["grip_rig_info"] = json.dumps(info)
    txt = bpy.data.texts.get("grip rig info") or bpy.data.texts.new("grip rig info")
    txt.clear()
    txt.write(json.dumps(info, indent=1))
    json.dump(info, open(os.path.join(a.work, "rig.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.work, "rig_state.npz"), A=A, S0=S0, T=T, co=co, mask=mask, free=free,
                        pinned=pinned, top_edge=top_edge, out_in_zone=oz, out_free=out_free,
                        **{"W|" + k: v for k, v in W.items()})
    bpy.context.view_layer.objects.active = rig
    bpy.ops.wm.save_as_mainfile(filepath=a.out, compress=True)
    log("saved", a.out, f"{time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
