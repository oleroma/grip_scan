"""Make the GripV R2/R4 arched triggers driven by their button top centre.

Adds two node groups to Grip.blend and wires them into 'GripV R2 parameters'
and 'GripV R4 parameters':

  trigger top solve    math only: button top + arch radius -> button rotation,
                       shift z, trigger length (the original trigger inputs)
  trigger top control  the node you edit: 'button top' (x, y, z), 'arch radius'
                       and the offsets the solve needs; feeds 'trigger parameters.001'

The trigger graph itself is not touched, so every part that reads the R2/R4
bundle (trigger, bool cutter, tip, marks) follows. The defaults are computed
from the original values, so the default result is the original trigger.

Run:  python build_trigger_controls.py IN.blend OUT.blend [--lab]
      --lab keeps only the trigger parts and the switches (a small work file).
Needs bpy 5.2 as a Python module (pip install bpy==5.2.2 numpy).
"""
import sys
import math
import bpy
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import trigger_model as tm  # noqa: E402

# Constant in 'trigger align': the trigger is lowered by 1.6 + pressed height.
SMD_DROP = 1.6
ITERATIONS = 8

PASS = ["top bevel", "height", "button radius offset", "button pressed height",
        "x offset", "tip angle offset", "tip shear"]


class Builder:
    def __init__(self, ng):
        self.ng = ng
        self.x = 0

    def node(self, kind, **props):
        n = self.ng.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        n.location = (self.x, 0)
        self.x += 160
        return n

    def feed(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.ng.links.new(v, sock)
        else:
            sock.default_value = v

    def math(self, op, a, b=None):
        n = self.node("ShaderNodeMath", operation=op)
        self.feed(n.inputs[0], a)
        if b is not None:
            self.feed(n.inputs[1], b)
        return n.outputs[0]

    def add(self, a, b): return self.math("ADD", a, b)
    def sub(self, a, b): return self.math("SUBTRACT", a, b)
    def mul(self, a, b): return self.math("MULTIPLY", a, b)
    def div(self, a, b): return self.math("DIVIDE", a, b)
    def atan2(self, y, x): return self.math("ARCTAN2", y, x)

    def sqrt_pos(self, a):
        return self.math("SQRT", self.math("MAXIMUM", a, 0.0))

    def sq(self, a): return self.mul(a, a)

    def xyz(self, v):
        n = self.node("ShaderNodeSeparateXYZ")
        self.feed(n.inputs[0], v)
        return n.outputs

    def combine(self, x, y, z):
        n = self.node("ShaderNodeCombineXYZ")
        for s, v in zip(n.inputs, (x, y, z)):
            self.feed(s, v)
        return n.outputs[0]

    def vmath(self, op, a, b=None):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self.feed(n.inputs[0], a)
        if b is not None:
            self.feed(n.inputs[1], b)
        return n.outputs[0]


def new_group(name):
    old = bpy.data.node_groups.get(name)
    if old:
        bpy.data.node_groups.remove(old)
    return bpy.data.node_groups.new(name, "GeometryNodeTree")


def sock(ng, name, io, kind, default=None, desc=""):
    s = ng.interface.new_socket(name, in_out=io, socket_type=kind)
    if default is not None:
        s.default_value = default
    s.description = desc
    return s


def build_solve():
    """button top + arch radius (+ offsets) -> rotation, shift z, length.

    Same math as trigger_model.inverse(): a fixed-point loop over the tip
    angle a, because the top bevel follows the top face normal, which tilts
    with a. Converges to float precision in 3-4 rounds.
    """
    ng = new_group("trigger top solve")
    for n, k in [("button top", "NodeSocketVector"), ("arch radius", "NodeSocketFloat"),
                 ("SMD", "NodeSocketVector")] + [(p, "NodeSocketFloat") for p in PASS]:
        sock(ng, n, "INPUT", k)
    for n in ["button rotation", "shift z", "trigger length", "tip angle"]:
        sock(ng, n, "OUTPUT", "NodeSocketFloat")
    b = Builder(ng)
    gi = b.node("NodeGroupInput")
    go = b.node("NodeGroupOutput")
    I = {s.name: s for s in gi.outputs}
    R = I["arch radius"]
    tb, h, bro = I["top bevel"], I["height"], I["button radius offset"]
    o, sh, xo = I["tip angle offset"], I["tip shear"], I["x offset"]

    drop = b.add(I["button pressed height"], SMD_DROP)
    d = b.vmath("SUBTRACT", I["button top"], b.combine(0.0, 0.0, b.mul(drop, -1.0)))
    d = b.vmath("SUBTRACT", d, I["SMD"])
    dx, dy, dz = b.xyz(d)
    dxy2 = b.add(b.sq(dx), b.sq(dy))
    phi_t = b.atan2(dy, dx)
    rho = b.sub(R, b.mul(h, 0.5))
    rr = b.sub(b.sq(R), b.sq(rho))

    ri = b.node("GeometryNodeRepeatInput")
    ro = b.node("GeometryNodeRepeatOutput")
    ri.pair_with_output(ro)
    ro.repeat_items.clear()
    for n in ("a", "rot", "cz"):
        ro.repeat_items.new("FLOAT", n)
    ri.inputs["Iterations"].default_value = ITERATIONS
    a = ri.outputs["a"]

    # top face normal at tip angle a, trigger-local frame
    nv = b.vmath("NORMALIZE", b.combine(b.mul(sh, -1.0), 1.0,
                                        b.mul(b.math("TANGENT", b.sub(a, o)), -1.0)))
    vr = b.node("ShaderNodeVectorRotate", rotation_type="X_AXIS")
    b.feed(vr.inputs["Vector"], nv)
    b.feed(vr.inputs["Angle"], a)
    nx, ny, nz = b.xyz(vr.outputs[0])

    qx = b.add(xo, b.mul(tb, nx))
    qy = b.sqrt_pos(b.sub(dxy2, b.sq(qx)))
    rot = b.sub(phi_t, b.atan2(qy, qx))
    U = b.add(b.sub(qy, b.mul(tb, ny)), bro)
    V = b.sub(dz, b.mul(tb, nz))
    W = b.math("MAXIMUM", b.sqrt_pos(b.add(b.sq(U), b.sq(V))), 1e-6)
    xx = b.div(b.add(rr, b.sq(W)), b.mul(W, 2.0))
    yy = b.sqrt_pos(b.sub(b.sq(R), b.sq(xx)))
    cy = b.div(b.sub(b.mul(xx, U), b.mul(yy, V)), W)
    cz = b.div(b.add(b.mul(xx, V), b.mul(yy, U)), W)
    a_new = b.atan2(b.sub(U, cy), b.sub(cz, V))
    b.feed(ro.inputs["a"], a_new)
    b.feed(ro.inputs["rot"], rot)
    b.feed(ro.inputs["cz"], cz)

    A, ROT, CZ = ro.outputs["a"], ro.outputs["rot"], ro.outputs["cz"]
    asin = b.math("ARCSINE", b.math("MINIMUM", b.math("MAXIMUM", b.div(CZ, R), -1.0), 1.0))
    tal = b.mul(b.sub(math.pi / 2, asin), R)
    L = b.add(b.add(tal, b.mul(R, A)), tb)
    b.feed(go.inputs["button rotation"], ROT)
    b.feed(go.inputs["shift z"], CZ)
    b.feed(go.inputs["trigger length"], L)
    b.feed(go.inputs["tip angle"], A)
    return ng


def build_control(solve):
    """The node you edit. Its outputs feed 'trigger parameters.001'.

    The panel 'original (do not edit)' holds the values the trigger had before.
    The outputs are original + solve(new) - solve(original), so with the
    default inputs they equal the original values bit for bit (float rounding
    in the solve cancels), and anywhere else they differ from a plain solve by
    float noise only (~1e-5 mm).
    """
    ng = new_group("trigger top control")
    sock(ng, "button top", "INPUT", "NodeSocketVector", (0, 0, 0),
         "Centre of the button's top (press) face, on its outer surface, mm. "
         "The arch is fitted between the SMD switch and this point")
    ng.interface.items_tree["button top"].subtype = "TRANSLATION"
    sock(ng, "arch radius", "INPUT", "NodeSocketFloat", 30.0,
         "Radius of the arch (the old 'trigger radius'), mm")
    sock(ng, "SMD button", "INPUT", "NodeSocketObject", None,
         "The switch the trigger presses (R2_pcb_button / R4_pcb_button)")
    for p in PASS:
        sock(ng, p, "INPUT", "NodeSocketFloat")
    panel = ng.interface.new_panel("original (do not edit)", default_closed=True)
    refs = ["button top", "arch radius", "button rotation", "shift z", "trigger length"] + PASS
    for r in refs:
        kind = "NodeSocketVector" if r == "button top" else "NodeSocketFloat"
        s = sock(ng, "original " + r, "INPUT", kind)
        ng.interface.move_to_parent(s, panel, len(panel.interface_items))
    for n in ["trigger radius", "button rotation", "shift z", "trigger length"] + PASS:
        sock(ng, n, "OUTPUT", "NodeSocketFloat")
    b = Builder(ng)
    gi = b.node("NodeGroupInput")
    go = b.node("NodeGroupOutput")
    oi = b.node("GeometryNodeObjectInfo", transform_space="ORIGINAL")
    b.feed(oi.inputs["Object"], gi.outputs["SMD button"])
    solves = []
    for pre in ("", "original "):
        s = b.node("GeometryNodeGroup")
        s.node_tree = solve
        b.feed(s.inputs["button top"], gi.outputs[pre + "button top"])
        b.feed(s.inputs["arch radius"], gi.outputs[pre + "arch radius"])
        b.feed(s.inputs["SMD"], oi.outputs["Location"])
        for p in PASS:
            b.feed(s.inputs[p], gi.outputs[pre + p])
        solves.append(s)
    for p in PASS:
        b.feed(go.inputs[p], gi.outputs[p])
    b.feed(go.inputs["trigger radius"], gi.outputs["arch radius"])
    for n in ["button rotation", "shift z", "trigger length"]:
        delta = b.sub(solves[0].outputs[n], solves[1].outputs[n])
        b.feed(go.inputs[n], b.add(gi.outputs["original " + n], delta))
    return ng


def wire(group_name, smd_name, control):
    ng = bpy.data.node_groups[group_name]
    tp = ng.nodes["trigger parameters.001"]
    for old in [n for n in ng.nodes if n.name == "button top control"]:
        ng.nodes.remove(old)
    val = {s.name: s.default_value for s in tp.inputs if hasattr(s, "default_value")}
    p = dict(R=val["trigger radius"], z=val["shift z"], rot=val["button rotation"],
             L=val["trigger length"], tb=val["top bevel"], h=val["height"],
             bro=val["button radius offset"], ph=val["button pressed height"],
             xoff=val["x offset"], o=val["tip angle offset"], sh=val["tip shear"])
    smd = bpy.data.objects[smd_name]
    T, _, _ = tm.forward(p, tuple(smd.location), drop=SMD_DROP)
    c = ng.nodes.new("GeometryNodeGroup")
    c.node_tree = control
    c.name = c.label = "button top control"
    c.location = (tp.location.x - 300, tp.location.y)
    c.width = 220
    c.inputs["button top"].default_value = tuple(T)
    c.inputs["arch radius"].default_value = p["R"]
    c.inputs["SMD button"].default_value = smd
    for k in PASS:
        c.inputs[k].default_value = val[k]
        c.inputs["original " + k].default_value = val[k]
    c.inputs["original button top"].default_value = tuple(T)
    c.inputs["original arch radius"].default_value = p["R"]
    for k in ["button rotation", "shift z", "trigger length"]:
        c.inputs["original " + k].default_value = val[k]
    for out in c.outputs:
        if out.name in tp.inputs:
            ng.links.new(out, tp.inputs[out.name])
    return p, T


def keep_only_triggers():
    keep_cols = {"triggers", "R2R4 cutouts", "R2R4 tip", "R2R4 control smd", "R2R4 smds"}
    for ob in list(bpy.data.objects):
        if not ({c.name for c in ob.users_collection} & keep_cols):
            bpy.data.objects.remove(ob)
    for _ in range(3):
        bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src, dst = args[0], args[1]
    bpy.ops.wm.open_mainfile(filepath=src)
    solve = build_solve()
    control = build_control(solve)
    rep = {}
    for part in ("R2", "R4"):
        p, T = wire(f"GripV {part} parameters", f"{part}_pcb_button", control)
        rep[part] = dict(original=p, button_top=[float(x) for x in T])
    if "--lab" in sys.argv:
        keep_only_triggers()
        g = bpy.data.node_groups["global"]
        n = [n for n in g.nodes if n.name.startswith("global parameters")][0]
        n.inputs["variant"].default_value = "GripV"
    bpy.ops.wm.save_as_mainfile(filepath=dst, compress=True)
    import json
    print("REPORT " + json.dumps(rep))


if __name__ == "__main__":
    main()
