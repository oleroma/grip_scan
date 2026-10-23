"""add_rig_handles.py: user-friendly handles on the grip rig + a printable mock-up.

    PY=/path/to/venv/bin/python        # pip install bpy==5.2.2 numpy
    $PY trigger_controls/build_trigger_controls.py grip_rig_posed.blend build/combined.blend
    $PY rig_handles/add_rig_handles.py build/combined.blend OUT.blend [--stl-dir DIR]

IN must be a rigged grip file (grip_rig/ pipeline) that already has the R2/R4 'button top control'
nodes (trigger_controls/). The current pose of IN is kept and stored as the "scan result".

What it adds:
  handles   ball/target shapes the user drags with G (R2/R4 also rotate with R):
              palm swell, fin middle/ring/pinky handle, stand handle, thenar surface  (grip shape)
              R2 button top, R4 button top (+ 'arch radius' slider on each)          (buttons)
            The fin and stand bones now aim at their handle (Damped Track), so a handle drag swings
            the fin; R2/R4 handles drive 'button top' / 'arch radius' of 'button top control', and
            their rotation (local X = tip angle offset, local Z = tip shear angle) points the press face.
  panel     3D view sidebar > Grip: named sliders for every handle, reset to scan result / to
            original GripV, show mock-up, export mock-up STLs (text block 'grip_controls_ui.py').
  mock-up   'mock-up whole' = posed basemesh + caps, triggers, bumpers, sticks, scroll wheel (no
            shafts), booleaned into one solid; 'mock-up top' / 'mock-up bottom' split it with the
            case's own node group 'cut basemesh' (same Z=40 cut as case_top / case_bottom).
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

RIG = "grip rig"
CAGE = "basemesh GripV"
SCENE = "GripV"
WGT_COLL = "grip rig widgets"
MOCK_COLL = "mock-up print"
UI_TEXT = "grip_controls_ui.py"
FINS = ("fin middle", "fin ring", "fin pinky", "stand")
BUTTONS = {"R2": ("GripV R2 parameters", (15.0, 80.0)), "R4": ("GripV R4 parameters", (40.0, 200.0))}
BUTTON_RANGE_MM = 10.0
R1_SLIDE_MIN_MM = -15.0   # R1 hand handle: contact can move this far toward the knuckles
R1_TURN_DEG = 15.0        # ... and turn about X by at most this much either way
HAND_OBJECTS = ("hand rig", "hold marker: R2 press", "hold marker: R4 press", "hold marker: palm swell")
MAX_PRESS_DEG = 30.0      # warn when the press face is tilted more than this from the travel direction
SEAT_FULL_MM, SEAT_ZERO_MM = 6.0, 20.0   # skin this close to a trigger follows it fully, fading to 0 here
# objects booleaned into the mock-up: everything the hand touches, never the shafts
MOCK_PARTS = ["abxy cap up", "abxy cap down", "abxy cap left", "abxy cap right",
              "dpad cap up", "dpad cap down", "dpad cap left", "dpad cap right",
              "select cap top left", "select cap top right", "select cap down left", "select cap down right",
              "home cap", "R2", "R4", "L2", "L4", "R1", "L1", "StickR", "StickL", "scrollwheel"]
PROTECTED = {"/home/claude/Grip/blender/Grip.blend"}
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "trigger_controls"))
import trigger_model as tm  # noqa: E402


def log(*a):
    print("[rig_handles]", *a, flush=True)


# ---------------------------------------------------------------- widgets (wire meshes, mm)
def circle(r, axis, n=32, c=(0.0, 0.0, 0.0)):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        u, v = r * math.cos(a), r * math.sin(a)
        p = {"x": (0, u, v), "y": (u, 0, v), "z": (u, v, 0)}[axis]
        pts.append(tuple(p[k] + c[k] for k in range(3)))
    return pts, [(i, (i + 1) % n) for i in range(n)]


def widget(name, parts, coll):
    V, E = [], []
    for pts, edges in parts:
        o = len(V)
        V += pts
        E += [(a + o, b + o) for a, b in edges]
    me = bpy.data.meshes.get(name) or bpy.data.meshes.new(name)
    me.clear_geometry()
    me.from_pydata(V, E, [])
    ob = bpy.data.objects.get(name) or bpy.data.objects.new(name, me)
    if ob.name not in coll.objects:
        coll.objects.link(ob)
    return ob


def ball(r, c=(0.0, 0.0, 0.0)):
    return [circle(r, a, c=c) for a in "xyz"]


def arrow(length, axis=1, head=2.0):
    tip = [0.0, 0.0, 0.0]
    tip[axis] = length
    side = [0.0, 0.0, 0.0]
    side[(axis + 1) % 3] = head * 0.6
    back = list(tip)
    back[axis] -= head
    b1 = [back[k] + side[k] for k in range(3)]
    b2 = [back[k] - side[k] for k in range(3)]
    return ([(0, 0, 0), tuple(tip), tuple(b1), tuple(b2)], [(0, 1), (1, 2), (1, 3)])


def square(s, axis="z"):
    h = s / 2
    q = {"z": [(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)],
         "y": [(-h, 0, -h), (h, 0, -h), (h, 0, h), (-h, 0, h)]}[axis]
    return q, [(0, 1), (1, 2), (2, 3), (3, 0)]


def make_widgets(scene):
    coll = bpy.data.collections.get(WGT_COLL) or bpy.data.collections.new(WGT_COLL)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    coll.hide_viewport = True
    coll.hide_render = True
    w = {
        "palm": widget("WGT palm swell", ball(7.0) + [arrow(16.0)], coll),
        "fin": widget("WGT fin handle", ball(4.0) + [arrow(9.0, head=1.6)], coll),
        "stand": widget("WGT stand handle", ball(5.0) + [square(12.0, "z")], coll),
        # R1 hand handle: double arrow along X (slide) + ring around X (turn)
        "r1": widget("WGT R1 hand", [circle(6.0, "x"), circle(2.5, "x"),
                                     ([(-12, 0, 0), (12, 0, 0), (9, 2, 0), (9, -2, 0), (-9, 2, 0), (-9, -2, 0)],
                                      [(0, 1), (1, 2), (1, 3), (0, 4), (0, 5)])], coll),
        "thenar": widget("WGT thenar", [square(14.0, "z"), circle(3.0, "z"), arrow(10.0, axis=2)], coll),
        # press face (square in local XZ) + arrow along local Y = the face normal
        "button": widget("WGT button top", [square(10.0, "y"), circle(2.0, "y"), arrow(11.0, head=2.0),
                                             ([(-7, 0, 0), (7, 0, 0)], [(0, 1)]),
                                             ([(0, 0, -7), (0, 0, 7)], [(0, 1)])], coll),
    }
    return w


# ---------------------------------------------------------------- rig handles
def bone_axes(b):
    M = b.matrix_local
    return Vector(M.col[0][:3]), Vector(M.col[1][:3]), Vector(M.col[2][:3])


def face_params(part):
    """Original trigger values -> rot0, tip angle offset o0, tip shear sh0, arch angle a0 and the
    handle's shear angle psi0. The press-face normal is exactly rz(rot) @ rx(o) @ rz(psi) @ +Y with
    tan(psi) = sh * cos(a - o) (see trigger_model.face_normal), so a bone with rest frame rz(rot0) and
    euler (x = o, z = psi), mode ZYX, points its Y axis along the face normal."""
    node = bpy.data.node_groups[BUTTONS[part][0]].nodes["button top control"]
    g = lambda n: float(node.inputs["original " + n].default_value)  # noqa: E731
    p = dict(R=g("arch radius"), z=g("shift z"), rot=g("button rotation"), L=g("trigger length"),
             tb=g("top bevel"), h=g("height"), bro=g("button radius offset"), ph=g("button pressed height"),
             xoff=g("x offset"), o=g("tip angle offset"), sh=g("tip shear"))
    _, _, a0 = tm.forward(p, np.zeros(3))
    o0 = float(node.inputs["tip angle offset"].default_value)
    sh0 = float(node.inputs["tip shear"].default_value)
    psi0 = math.atan(sh0 * math.cos(a0 - o0))
    return dict(rot0=p["rot"], o0=o0, sh0=sh0, a0=float(a0), psi0=psi0)


def add_handle_bones(rig):
    """Edit mode: fin/stand handles at the bone tips (local Y = outward normal, X = the fin's X),
    R2/R4 button-top bones at the trigger tops (rest frame = world turned by the trigger's rotation)."""
    tops = {}
    for part, (grp, _) in BUTTONS.items():
        tops[part] = Vector(bpy.data.node_groups[grp].nodes["button top control"].inputs["button top"].default_value)
    rest = {}
    for n in FINS:
        x, y, z = bone_axes(rig.data.bones[n])
        rest[n] = (rig.data.bones[n].tail_local.copy(), x, y, z)
    view = bpy.context.view_layer
    view.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    eb = rig.data.edit_bones
    for n in FINS:
        tail, x, y, z = rest[n]
        h = eb.get(n + " handle") or eb.new(n + " handle")
        h.head = tail
        h.tail = tail + z * 8.0              # local Y = the fin's outward normal
        h.align_roll(-y)                     # local X = the fin's X (sideways), local Z = back to palm swell
        h.parent = eb["palm swell"]
        h.use_deform = False
    for part, t in tops.items():
        b = eb.get(part + " button top") or eb.new(part + " button top")
        b.head = t
        r0 = face_params(part)["rot0"]
        b.tail = t + Vector((-math.sin(r0), math.cos(r0), 0.0)) * 8.0   # local Y = rz(rot0) @ +Y
        b.align_roll(Vector((0.0, 0.0, 1.0)))                            # local Z = world Z
        b.parent = eb["grip root"]
        b.use_deform = False
    bpy.ops.object.mode_set(mode="OBJECT")
    return tops


def add_trigger_seats(rig):
    """Skin under R2/R4 follows the button handle's move: a deform bone '<part> seat' per trigger
    (Copy Location from the handle) with weights 1 within SEAT_FULL_MM of the trigger at rest, fading to 0
    at SEAT_ZERO_MM. The other weights are scaled by (1 - seat) so the sum stays the same. Exception zone,
    battery, flat top and mirror seam stay pinned by 'grip rig pins' (applied after the armature)."""
    cage = bpy.data.objects[CAGE]
    me = cage.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    dg = bpy.context.evaluated_depsgraph_get()
    seat = {}
    for part in BUTTONS:
        e = bpy.data.objects[part].evaluated_get(dg)
        m = e.to_mesh()
        T = np.empty(len(m.vertices) * 3)
        m.vertices.foreach_get("co", T)
        e.to_mesh_clear()
        T = T.reshape(-1, 3) @ np.array(bpy.data.objects[part].matrix_world)[:3, :3].T
        d = np.min(np.linalg.norm(co[:, None, :] - T[None, ::2, :], axis=2), axis=1)
        t = np.clip((SEAT_ZERO_MM - d) / (SEAT_ZERO_MM - SEAT_FULL_MM), 0.0, 1.0)
        seat[part] = t * t * (3 - 2 * t)
    # the seat only takes weight from 'grip root' (the fixed part), so the scan pose is unchanged
    root = cage.vertex_groups["grip root"]
    wsum = np.zeros(len(co))
    wroot = np.zeros(len(co))
    for v in me.vertices:
        for g in v.groups:
            wsum[v.index] += g.weight
            if g.group == root.index:
                wroot[v.index] = g.weight
    tot = sum(seat.values())
    share = np.where(tot > 0, np.minimum(tot * wsum, wroot) / np.maximum(tot, 1e-9), 0.0)
    for part in seat:
        seat[part] = seat[part] * share                 # absolute weights (same scale as the others)
    taken = sum(seat.values())
    for i in np.nonzero(taken > 0)[0]:
        root.add([int(i)], float(wroot[i] - taken[i]), "REPLACE")
    for part, w in seat.items():
        vg = cage.vertex_groups.get(part + " seat") or cage.vertex_groups.new(name=part + " seat")
        for i in np.nonzero(w > 1e-5)[0]:
            vg.add([int(i)], float(w[i]), "REPLACE")
    view = bpy.context.view_layer
    view.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    eb = rig.data.edit_bones
    for part in BUTTONS:
        h = eb[part + " button top"]
        b = eb.get(part + " seat") or eb.new(part + " seat")
        b.head, b.tail, b.roll = h.head.copy(), h.tail.copy(), h.roll
        b.parent = eb["grip root"]
        b.use_deform = True
    bpy.ops.object.mode_set(mode="OBJECT")
    for part in BUTTONS:
        pb = rig.pose.bones[part + " seat"]
        c = pb.constraints.new("COPY_LOCATION")
        c.name = "follow the button"
        c.target = rig
        c.subtarget = part + " button top"
        pb.lock_location = pb.lock_rotation = pb.lock_scale = (True, True, True)
    return {p: int((w > 1e-3).sum()) for p, w in seat.items()}


def r1_top_centre():
    """Centre of the outer face of the R1 tab (the part that sticks out of the shoulder, its outermost
    +Y face): middle of that face's outline, on the face."""
    dg = bpy.context.evaluated_depsgraph_get()
    ob = bpy.data.objects["R1"]
    e = ob.evaluated_get(dg)
    me = e.to_mesh()
    M = np.array(ob.matrix_world)
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", V)
    e.to_mesh_clear()
    V = V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    face = V[V[:, 1] > V[:, 1].max() - 0.05]
    c = (face.min(0) + face.max(0)) / 2
    c[1] = V[:, 1].max()
    return Vector(c.tolist())


def finger_curve(hand):
    """Smooth curve through the index joints (metacarpal head .. fingertip), world mm, by arc length."""
    from scipy.interpolate import CubicSpline
    names = ("index_metacarpal", "index_proximal", "index_middle", "index_distal")
    M = hand.matrix_world
    pts = [M @ hand.pose.bones[names[0]].head] + [M @ hand.pose.bones[n].tail for n in names]
    P = np.array([list(p) for p in pts])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return CubicSpline(s, P, bc_type="natural"), s


def rot_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def add_r1_handle(rig):
    """'R1 hand' handle on the centre of the R1 tab's outer face: moves only in X, turns only about X.

    X = where along the index finger R1 touches (+ = toward the fingertip; max = fingertip on R1). The
    hand slides along the finger's own curve (like pulling the finger through a ring at R1): for each
    contact s the rigid move that puts curve point s where the scan contact s0 was, with the curve's
    direction there turned onto the old one. Those moves are keyed into the action 'R1 finger slide'
    (one key per 0.5 mm) and played by an Action constraint on 'hand mover' driven by the handle's X.
    The turn rotates 'R1 turn' (parent of 'hand mover') about X through the tab centre.
    The palm swell is NOT rigid to it: 'palm follow' copies only the position of the palm point carried
    by the hand ('palm target'), with influence = palm follow (0..1); palm swell hangs off 'palm follow'."""
    c = r1_top_centre()
    mk = bpy.data.objects.get("hold marker: R1 target")
    if mk is not None:
        mk.matrix_world.translation = c
    hand = bpy.data.objects["hand rig"]
    bpy.context.view_layer.update()
    curve, joints = finger_curve(hand)
    L = float(joints[-1])
    ss = np.linspace(0.0, L, 4001)
    cc = np.array(list(c))
    s0 = float(ss[np.argmin(np.linalg.norm(curve(ss) - cc, axis=1))])
    s_min = max(0.0, s0 + R1_SLIDE_MIN_MM)
    lo, hi = s_min - s0, L - s0                       # handle X range (mm along the finger)
    ps_rest = rig.data.bones["palm swell"].head_local.copy()

    view = bpy.context.view_layer
    view.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    eb = rig.data.edit_bones

    def bone(name, head, parent, length=10.0, deform=False):
        b = eb.get(name) or eb.new(name)
        b.head = head
        b.tail = head + Vector((0.0, length, 0.0))
        b.roll = 0.0
        b.parent = eb[parent] if parent else None
        b.use_deform = deform
        return b

    bone("R1 hand", c, "grip root")
    bone("R1 turn", c, "grip root")
    bone("hand mover", c, "R1 turn")
    bone("palm target", ps_rest, "hand mover", 6.0)
    bone("palm follow", ps_rest, "grip root", 6.0)
    link = bone("R1 palm link", c, "grip root")
    link.tail = ps_rest.copy()
    eb["palm swell"].parent = eb["palm follow"]       # rest unchanged, so the pose is unchanged
    bpy.ops.object.mode_set(mode="OBJECT")
    pbs = rig.pose.bones

    pb = pbs["R1 hand"]
    pb.rotation_mode = "XYZ"
    pb.lock_location = (False, True, True)
    pb.lock_rotation = (False, True, True)
    pb.lock_scale = (True, True, True)
    cl = pb.constraints.new("LIMIT_LOCATION")
    cl.name = "slide range"
    cl.use_min_x = cl.use_max_x = True
    cl.min_x, cl.max_x = lo, hi
    cl.owner_space = "LOCAL"
    cl.use_transform_limit = True
    cr = pb.constraints.new("LIMIT_ROTATION")
    cr.name = "turn range"
    cr.use_limit_x = True
    cr.min_x, cr.max_x = -math.radians(R1_TURN_DEG), math.radians(R1_TURN_DEG)
    cr.owner_space = "LOCAL"
    cr.use_transform_limit = True

    t = pbs["R1 turn"]
    ct = t.constraints.new("COPY_ROTATION")
    ct.name = "turn with the handle"
    ct.target, ct.subtarget = rig, "R1 hand"
    ct.use_y = ct.use_z = False
    ct.target_space = ct.owner_space = "LOCAL"

    # slide action: hand mover pose per contact offset d = s - s0
    T0 = curve(s0, 1)
    T0 = T0 / np.linalg.norm(T0)
    P0 = curve(s0)
    act = bpy.data.actions.get("R1 finger slide") or bpy.data.actions.new("R1 finger slide")
    hm = pbs["hand mover"]
    hm.rotation_mode = "QUATERNION"
    try:
        slot = act.slots.new(id_type="OBJECT", name="grip rig")
    except Exception:
        slot = None
    ad = rig.animation_data or rig.animation_data_create()
    keep_act = ad.action
    ad.action = act
    if slot is not None:
        ad.action_slot = slot
    from mathutils import Matrix as Mx
    step = 0.5                                        # keys every 0.5 mm, one exactly at 0 (the scan pose)
    a_lo, a_hi = math.floor(lo / step) * step, math.ceil(hi / step) * step
    steps = int(round((a_hi - a_lo) / step))
    ds = a_lo + step * np.arange(steps + 1)
    k0 = int(round(-a_lo / step))
    ds[k0] = 0.0
    for k, d in enumerate(ds):
        s = s0 + d
        Ts = curve(s, 1)
        R = rot_between(Ts, T0)
        l = R @ (cc - curve(s)) + P0 - cc
        if k == k0:
            R, l = np.eye(3), np.zeros(3)
        hm.location = l.tolist()
        hm.rotation_quaternion = Mx(R.tolist()).to_quaternion()
        hm.keyframe_insert("location", frame=k + 1, group="hand mover")
        hm.keyframe_insert("rotation_quaternion", frame=k + 1, group="hand mover")
    try:
        fcs = act.fcurves
    except AttributeError:
        fcs = [fc for lay in act.layers for st in lay.strips for cb in st.channelbags for fc in cb.fcurves]
    for fc in fcs:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    ad.action = keep_act
    hm.location = (0.0, 0.0, 0.0)
    hm.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    ca = hm.constraints.new("ACTION")
    ca.name = "slide along the finger"
    ca.target, ca.subtarget = rig, "R1 hand"
    ca.transform_channel = "LOCATION_X"
    ca.target_space = "LOCAL"
    ca.action = act
    if slot is not None:
        try:
            ca.action_slot = slot
        except Exception:
            pass
    ca.min, ca.max = a_lo, a_hi
    ca.frame_start, ca.frame_end = 1, steps + 1
    ca.mix_mode = "BEFORE_FULL"

    pf = pbs["palm follow"]
    cf = pf.constraints.new("COPY_LOCATION")
    cf.name = "palm follow"
    cf.target, cf.subtarget = rig, "palm target"
    cf.influence = 1.0
    st = pbs["R1 palm link"].constraints.new("STRETCH_TO")
    st.name = "stretch to the palm"
    st.target, st.subtarget = rig, "palm follow"
    for n in ("R1 turn", "hand mover", "palm target", "palm follow", "R1 palm link"):
        p = pbs[n]
        p.lock_location = p.lock_rotation = p.lock_scale = (True, True, True)
        p.lock_rotation_w = True

    view.update()
    for n in HAND_OBJECTS:                              # the hand rides on 'hand mover'
        ob = bpy.data.objects.get(n)
        if ob is None:
            continue
        M = ob.matrix_world.copy()
        ob.parent = rig
        ob.parent_type = "BONE"
        ob.parent_bone = "hand mover"
        view.update()
        ob.matrix_world = M
    view.update()
    seg = {n: round(float(j), 2) for n, j in zip(("knuckle", "proximal", "middle", "distal", "tip"), joints)}
    rig["index_curve"] = json.dumps({"joints_mm": seg, "s0": s0, "length": L})
    return {"centre": [round(x, 2) for x in c], "contact_from_tip_mm": round(L - s0, 2),
            "slide_mm": [round(lo, 2), round(hi, 2)], "turn_deg": R1_TURN_DEG, "keys": steps + 1}


def eval_cage():
    dg = bpy.context.evaluated_depsgraph_get()
    e = bpy.data.objects[CAGE].evaluated_get(dg)
    me = e.to_mesh()
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", V)
    e.to_mesh_clear()
    return V.reshape(-1, 3)


def wire_handles(rig, w):
    """Move the fin/stand pose onto the handles (same shape), add Damped Track, locks, shapes, limits."""
    pbs = rig.pose.bones
    bpy.context.view_layer.update()
    tips = {n: (pbs[n].tail).copy() for n in FINS}                # armature space, current pose
    for n in FINS:
        h = pbs[n + " handle"]
        h.rotation_mode = "XYZ"
        par = pbs["palm swell"]
        rel = par.bone.matrix_local.inverted() @ h.bone.matrix_local
        M = par.matrix @ rel
        M.translation = tips[n]
        h.matrix = M
        bpy.context.view_layer.update()
        h.rotation_euler = (0.0, 0.0, 0.0)
        pb = pbs[n]
        pb.rotation_euler = (0.0, 0.0, 0.0)
        pb.location = (0.0, 0.0, 0.0)
        for c in list(pb.constraints):
            pb.constraints.remove(c)
        c = pb.constraints.new("DAMPED_TRACK")
        c.name = "aim at handle"
        c.target = rig
        c.subtarget = h.name
        c.track_axis = "TRACK_Y"
        pb.lock_location = (True, True, True)
        pb.lock_rotation = (True, True, True)
        h.lock_rotation = (True, True, True)
        h.lock_scale = (True, True, True)
        h.lock_location = (False, False, True)                 # along the fin does nothing: locked
        h.custom_shape = w["stand" if n == "stand" else "fin"]
        bpy.context.view_layer.update()
    for n, key in (("palm swell", "palm"), ("thenar surface", "thenar")):
        pb = pbs[n]
        pb.lock_rotation = (True, True, True)
        pb.lock_scale = (True, True, True)
        pb.custom_shape = w[key]
        if n == "thenar surface":
            pb.custom_shape_translation = (0.0, pb.bone.length, 0.0)   # drawn at the thenar end
    for part in BUTTONS:
        pb = pbs[part + " button top"]
        pb.custom_shape = w["button"]
        f = face_params(part)
        pb.rotation_mode = "ZYX"                      # matrix = Rx(tip angle) @ Rz(shear angle)
        pb.rotation_euler = (f["o0"], 0.0, f["psi0"])
        pb.lock_rotation = (False, True, False)
        pb.lock_scale = (True, True, True)
        c = pb.constraints.new("LIMIT_LOCATION")
        c.name = "stay near the switch"
        for ax in "xyz":
            setattr(c, f"use_min_{ax}", True)
            setattr(c, f"use_max_{ax}", True)
            setattr(c, f"min_{ax}", -BUTTON_RANGE_MM)
            setattr(c, f"max_{ax}", BUTTON_RANGE_MM)
        c.owner_space = "LOCAL"
        c.use_transform_limit = True
    for pb in pbs:
        pb.use_custom_shape_bone_size = False
        if pb.custom_shape is not None:
            pb.custom_shape_wire_width = 2.5


def add_button_drivers(rig):
    pbs = rig.pose.bones
    for part, (grp, (lo, hi)) in BUTTONS.items():
        ng = bpy.data.node_groups[grp]
        node = ng.nodes["button top control"]
        pb = pbs[part + " button top"]
        r0 = float(node.inputs["arch radius"].default_value)
        pb["arch radius"] = r0
        ui = pb.id_properties_ui("arch radius")
        ui.update(min=lo, max=hi, soft_min=lo, soft_max=hi, step=50, precision=2, default=r0,
                  description=f"radius of the {part} trigger arch, mm (default {r0:g})")
        sock = node.inputs["button top"]
        T0 = [float(x) for x in sock.default_value]
        bpy.context.view_layer.update()
        H0 = rig.matrix_world @ pb.head                # what the driver reads at rest (float32)
        for i, ax in enumerate("XYZ"):
            sock.driver_remove("default_value", i)
            fc = sock.driver_add("default_value", i)
            d = fc.driver
            d.type = "SCRIPTED"                        # original + move: bit-exact at rest
            d.expression = f"{T0[i]!r} + (top - {float(H0[i])!r})"
            v = d.variables.new()
            v.name = "top"
            v.type = "TRANSFORMS"
            t = v.targets[0]
            t.id = rig
            t.bone_target = pb.name
            t.transform_type = "LOC_" + ax
            t.transform_space = "WORLD_SPACE"
        f = face_params(part)
        x0 = float(pb.rotation_euler[0])              # float32 as stored: default gives o0 exactly
        z0 = float(pb.rotation_euler[2])
        t0 = math.tan(z0) / math.cos(f["a0"] - x0)
        exprs = {"tip angle offset": f"{f['o0']!r} + (x - {x0!r})",
                 "tip shear": f"{f['sh0']!r} + tan(z) / cos({f['a0']!r} - x) - {t0!r}"}
        for sname, expr in exprs.items():
            so = node.inputs[sname]
            so.driver_remove("default_value")
            fc = so.driver_add("default_value")
            d = fc.driver
            d.type = "SCRIPTED"
            d.expression = expr                       # simple expression: runs without Python
            for vn, i in (("x", 0), ("z", 2)):
                v = d.variables.new()
                v.name = vn
                v.type = "SINGLE_PROP"
                v.targets[0].id = rig
                v.targets[0].data_path = f'pose.bones["{pb.name}"].rotation_euler[{i}]'
        rs = node.inputs["arch radius"]
        rs.driver_remove("default_value")
        fc = rs.driver_add("default_value")
        d = fc.driver
        d.type = "AVERAGE"
        v = d.variables.new()
        v.name = "radius"
        v.type = "SINGLE_PROP"
        v.targets[0].id = rig
        v.targets[0].data_path = f'pose.bones["{pb.name}"]["arch radius"]'


def bone_collections(rig):
    arm = rig.data
    for c in list(arm.collections_all):
        arm.collections.remove(c)
    groups = {
        "grip shape (drag these)": ["palm swell", "fin middle handle", "fin ring handle", "fin pinky handle",
                                    "stand handle", "thenar surface"],
        "buttons (drag these)": ["R2 button top", "R4 button top"],
        "hand on R1 (drag this)": ["R1 hand", "R1 palm link"],
        "mechanics (do not touch)": ["R1 turn", "hand mover", "palm target", "palm follow", "grip root", "fin middle", "fin ring", "fin pinky", "stand", "top back edge",
                                     "R2 seat", "R4 seat"],
    }
    for name, bones in groups.items():
        c = arm.collections.new(name)
        for b in bones:
            c.assign(arm.bones[b])
        c.is_visible = not name.startswith("mechanics")
    cols = {"fin middle handle": "THEME03", "fin ring handle": "THEME04", "fin pinky handle": "THEME09",
            "stand handle": "THEME07", "R2 button top": "THEME05", "R4 button top": "THEME05"}
    for b, p in cols.items():
        arm.bones[b].color.palette = p
    arm.show_names = True
    arm.display_type = "STICK"
    arm.relation_line_position = "HEAD"


def store_scan_pose(rig):
    P = {}
    for pb in rig.pose.bones:
        e = {"loc": list(pb.location), "rot": list(pb.rotation_euler)}
        if "arch radius" in pb:
            e["arch radius"] = float(pb["arch radius"])
        P[pb.name] = e
    rig["scan_pose"] = json.dumps(P)
    orig = {}
    for part, (grp, _) in BUTTONS.items():
        node = bpy.data.node_groups[grp].nodes["button top control"]
        # defaults of the original trigger = the 'original (do not edit)' panel values solved forward;
        # the scan file had no trigger moves, so the current values are the original ones
        pb = rig.pose.bones[part + " button top"]
        orig[part] = {"arch radius": float(node.inputs["arch radius"].default_value),
                      "rot": list(pb.rotation_euler)}
    rig["original_buttons"] = json.dumps(orig)
    rig["max press angle"] = MAX_PRESS_DEG
    rig.id_properties_ui("max press angle").update(min=5.0, max=60.0, default=MAX_PRESS_DEG,
                                                    description="press angle above this shows TOO STEEP")


# ---------------------------------------------------------------- mock-up
def _sock(sockets, name):
    for s in sockets:
        if s.name == name and s.enabled:
            return s
    raise KeyError(name)


def mock_whole_group():
    ng = bpy.data.node_groups.get("mock-up whole") or bpy.data.node_groups.new("mock-up whole", "GeometryNodeTree")
    ng.nodes.clear()
    ng.interface.clear()
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    L = ng.links
    go = ng.nodes.new("NodeGroupOutput")
    go.location = (900, 0)
    bv = ng.nodes.new("GeometryNodeGroup")
    bv.node_tree = bpy.data.node_groups["basemesh variant"]
    bv.location = (-400, 200)
    bv.label = "posed basemesh"
    join = ng.nodes.new("GeometryNodeJoinGeometry")
    join.location = (0, 0)
    for i, name in enumerate(MOCK_PARTS):
        oi = ng.nodes.new("GeometryNodeObjectInfo")
        oi.location = (-400, -120 * i)
        oi.transform_space = "RELATIVE"
        oi.inputs["Object"].default_value = bpy.data.objects[name]
        oi.label = name
        L.new(_sock(oi.outputs, "Geometry"), join.inputs[0])
    real = ng.nodes.new("GeometryNodeRealizeInstances")
    real.location = (200, 0)
    L.new(join.outputs[0], real.inputs[0])
    boo = ng.nodes.new("GeometryNodeMeshBoolean")
    boo.location = (450, 100)
    boo.operation = "UNION"
    boo.solver = "MANIFOLD"           # fast (0.2 s) and always manifold; EXACT left non-manifold seams at the sticks
    L.new(bv.outputs[0], boo.inputs[1])        # UNION: one multi-input ("Mesh 2")
    L.new(real.outputs[0], boo.inputs[1])
    L.new(boo.outputs[0], go.inputs[0])
    return ng


def mock_half_group():
    ng = bpy.data.node_groups.get("mock-up half") or bpy.data.node_groups.new("mock-up half", "GeometryNodeTree")
    ng.nodes.clear()
    ng.interface.clear()
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    s = ng.interface.new_socket("bottom", in_out="INPUT", socket_type="NodeSocketBool")
    s.default_value = False
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    L = ng.links
    gi = ng.nodes.new("NodeGroupInput")
    gi.location = (-600, 0)
    go = ng.nodes.new("NodeGroupOutput")
    go.location = (400, 0)
    oi = ng.nodes.new("GeometryNodeObjectInfo")
    oi.location = (-400, 100)
    oi.transform_space = "RELATIVE"
    oi.inputs["Object"].default_value = bpy.data.objects["mock-up whole"]
    cut = ng.nodes.new("GeometryNodeGroup")
    cut.node_tree = bpy.data.node_groups["cut basemesh"]   # the case's own top/bottom split (Z = 40)
    cut.location = (0, 0)
    L.new(_sock(oi.outputs, "Geometry"), _sock(cut.inputs, "Mesh 1"))
    L.new(_sock(gi.outputs, "bottom"), _sock(cut.inputs, "bottom"))
    L.new(_sock(cut.outputs, "Geometry"), go.inputs[0])
    return ng


def mock_object(name, ng, coll, bottom=None):
    old = bpy.data.objects.get(name)
    if old is not None:
        bpy.data.objects.remove(old)
    me = bpy.data.meshes.new(name)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    m = ob.modifiers.new(ng.name, "NODES")
    m.node_group = ng
    if bottom is not None:
        for it in ng.interface.items_tree:
            if it.item_type == "SOCKET" and it.in_out == "INPUT" and it.name == "bottom":
                getattr(m.properties.inputs, it.identifier).value = bool(bottom)
    return ob


def build_mockup(scene):
    coll = bpy.data.collections.get(MOCK_COLL) or bpy.data.collections.new(MOCK_COLL)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    whole = mock_object("mock-up whole", mock_whole_group(), coll)
    whole.hide_render = True
    half = mock_half_group()
    top = mock_object("mock-up top", half, coll, bottom=False)
    bot = mock_object("mock-up bottom", half, coll, bottom=True)
    return coll, whole, top, bot


def mesh_report(ob):
    import bmesh
    dg = bpy.context.evaluated_depsgraph_get()
    e = ob.evaluated_get(dg)
    me = e.to_mesh()
    bm = bmesh.new()
    bm.from_mesh(me)
    nm = sum(1 for ed in bm.edges if not ed.is_manifold)
    vol = bm.calc_volume(signed=True)
    co = np.array([v.co[:] for v in bm.verts]) if len(bm.verts) else np.zeros((1, 3))
    r = {"verts": len(bm.verts), "faces": len(bm.faces), "non_manifold_edges": nm,
         "volume_cm3": round(vol / 1000.0, 2), "bbox_min": np.round(co.min(0), 2).tolist(),
         "bbox_max": np.round(co.max(0), 2).tolist()}
    bm.free()
    e.to_mesh_clear()
    return r


def export_stl(ob, path):
    view = bpy.context.view_layer
    for o in view.objects:
        o.select_set(False)
    ob.hide_set(False)
    ob.select_set(True)
    view.objects.active = ob
    bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True, apply_modifiers=True, ascii_format=False)


# ---------------------------------------------------------------- UI text block
def install_ui():
    src = open(os.path.join(HERE, "grip_controls_ui.py")).read()
    t = bpy.data.texts.get(UI_TEXT) or bpy.data.texts.new(UI_TEXT)
    t.clear()
    t.write(src)
    t.use_module = True                      # registers on load when the file is trusted
    exec(compile(src, UI_TEXT, "exec"), {"__name__": "__main__"})


# ---------------------------------------------------------------- main
HEAVY_COLLS = ("case", "case middle plate",       # case top/bottom/plate: heavy booleans, never needed here
               "buttons", "capacitive", "dpad stick", "battery cover", "cable holder", "pins", "nut lock")
LIGHT_COLLS = ("basemesh GripV", "parts", "caps", "triggers", "bumpers", "sticks", "scrollwheel", "grip rig", "mock-up print")


def light_scene(view_layer):
    """Exclude the heavy case parts; keep the basemesh and the buttons."""
    def walk(lc):
        yield lc
        for c in lc.children:
            yield from walk(c)
    for lc in walk(view_layer.layer_collection):
        n = lc.collection.name
        if n in HEAVY_COLLS:
            lc.exclude = True
        elif n in LIGHT_COLLS:
            lc.exclude = False


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    stl_dir = None
    if "--stl-dir" in argv:
        i = argv.index("--stl-dir")
        stl_dir = argv[i + 1]
        del argv[i:i + 2]
    src, dst = argv[0], argv[1]
    assert os.path.abspath(dst) not in PROTECTED, "refusing to overwrite Grip.blend"
    bpy.ops.wm.open_mainfile(filepath=src)
    bpy.data.node_groups["global"].nodes["global parameters"].inputs["variant"].default_value = "GripV"
    scene = bpy.data.scenes[SCENE]
    bpy.context.window.scene = scene
    for cn in ("grip rig", "hand reference (hold_upright)"):
        c = bpy.data.collections.get(cn)
        if c is not None and c.name not in scene.collection.children:
            scene.collection.children.link(c)
    light_scene(bpy.context.view_layer)
    rig = bpy.data.objects[RIG]
    assert "fin middle handle" not in rig.data.bones, "handles already added"
    for part, (grp, _) in BUTTONS.items():
        assert "button top control" in bpy.data.node_groups[grp].nodes, "run trigger_controls first"
    bpy.context.view_layer.update()
    before = eval_cage()

    w = make_widgets(scene)
    add_handle_bones(rig)
    wire_handles(rig, w)
    log("trigger seats (cage vertices that follow):", add_trigger_seats(rig))
    log("R1 hand handle:", add_r1_handle(rig))
    rig.pose.bones["R1 hand"].custom_shape = w["r1"]
    rig.pose.bones["R1 hand"].use_custom_shape_bone_size = False
    rig.pose.bones["R1 hand"].custom_shape_wire_width = 2.5
    rig.data.bones["R1 hand"].color.palette = "THEME01"
    add_button_drivers(rig)
    bone_collections(rig)
    bpy.context.view_layer.update()
    after = eval_cage()
    shape_change = float(np.abs(after - before).max())
    log("cage change from moving the pose onto handles: %.5f mm" % shape_change)
    store_scan_pose(rig)

    coll, whole, top, bot = build_mockup(scene)
    install_ui()

    rep = {"cage_change_after_handles_mm": round(shape_change, 5)}
    bpy.context.view_layer.update()
    for ob in (whole, top, bot):
        rep[ob.name] = mesh_report(ob)
        log(ob.name, rep[ob.name])
    if stl_dir:
        os.makedirs(stl_dir, exist_ok=True)
        for ob in (top, bot):
            p = os.path.join(stl_dir, ob.name.replace(" ", "_").replace("-", "") + ".stl")
            export_stl(ob, p)
            log("wrote", p)
    coll.hide_viewport = True                 # booleans only run when the user shows the mock-up
    whole.hide_viewport = True                # only the two halves show
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=dst, compress=True)
    log("saved", dst)
    print("REPORT " + json.dumps(rep))


if __name__ == "__main__":
    main()
