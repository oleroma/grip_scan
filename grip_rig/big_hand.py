"""big_hand.py (needs hand_scan_rig/ from branch claude/project-thread-i3yp76 and gripscan/ from claude/project-thread-pgewvn on PYTHONPATH; copy it next to hold_upright.py):
the scan hand scaled up, curled round GripV again (hold_upright's solve), for the grip rig.

    $PY big_hand.py scale   --s 1.08     # hand_rigged.blend scaled about the wrist -> WORK/hand_rigged_s.blend
    $PY big_hand.py holdrig              # hold scene (grip, markers, cylinder reference rig) -> WORK/hold_rig.blend
    $PY big_hand.py ref | start | solve [--nfev N] | blend
                                         # hold_upright.py steps on the scaled hand; never writes the originals

Same objective as hold_upright.py (upright palm, palm on the swell, index on R1, middle on R2, ring on
R4, pinky pad on the grip, <= 0.4 mm free squash): only the hand is bigger, so the joints curl
differently to reach the same buttons.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hold_solve as hs  # noqa: E402
import hold_upright as hu  # noqa: E402

UPRIGHT_BLEND = "/mnt/project-files/hand_scan_rig/hold_upright/hand_on_grip_upright.blend"
WORK = os.environ.get("BIG_HAND_WORK", "/tmp/claude-0/hs/work/big_hand")
OUT_BLEND = os.path.join(WORK, "hand_on_grip_big.blend")

# every path the hold code writes goes to WORK; inputs: the scaled hand and the rebuilt hold scene
hs.FIT_BLEND = os.path.join(WORK, "hand_rigged_s.blend")
hs.HOLD_RIG = os.path.join(WORK, "hold_rig.blend")
hs.WORK = WORK
hs.OUT = WORK
hu.WORK = WORK
hu.OUT = WORK
hs.Scene.__init__.__defaults__ = (None, hs.FIT_BLEND, hs.HOLD_RIG)   # bound at import time


def cmd_scale(a):
    """Uniform scale of the rigged hand (skin, shape keys, bones) about the wrist, applied to the data."""
    import bpy
    from mathutils import Matrix
    bpy.ops.wm.open_mainfile(filepath=os.path.join(hu.hs.FIT_RUN, "fit", "hand_rigged.blend"))
    arm, ob = bpy.data.objects["hand rig"], bpy.data.objects["hand scan"]
    w = arm.matrix_world @ arm.data.bones["palm"].head_local
    S = Matrix.Translation(w) @ Matrix.Scale(a.s, 4) @ Matrix.Translation(-w)
    for o in (arm, ob):
        o.matrix_world = S @ o.matrix_world
    for o in bpy.context.view_layer.objects:
        o.select_set(o in (arm, ob))
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    os.makedirs(WORK, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=hs.FIT_BLEND)
    b = arm.data.bones
    print("[big] scaled", a.s, "palm length", round((b["palm"].tail_local - b["palm"].head_local).length, 2),
          "middle finger", round(sum(b[n].length for n in ("middle_proximal", "middle_middle", "middle_distal")), 2),
          "->", hs.FIT_BLEND, flush=True)


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
RANGES = {"size": (0.92, 1.12), "palm_length": (0.94, 1.06), "palm_width": (0.94, 1.06), "finger": (0.93, 1.07)}


def random_factors(seed):
    """Reproducible random proportions: overall size, palm length (metacarpals), hand width, one
    length factor per finger (its three phalanges; the thumb's metacarpal too)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    u = lambda k: float(rng.uniform(*RANGES[k]))  # noqa: E731
    return {"seed": seed, "size": u("size"), "palm_length": u("palm_length"), "palm_width": u("palm_width"),
            "finger": {f: u("finger") for f in FINGERS}}


def cmd_random(a):
    """Reshape the rigged hand in its rest pose: every bone gets a length factor, every skin vertex follows
    its bones by its skin weights (each bone: move to the bone's new head, stretch along the bone);
    then the hand is stretched across the palm (width) and scaled about the wrist (size)."""
    import bpy
    import numpy as np
    fac = random_factors(a.seed) if a.factors is None else json.load(open(a.factors))
    bpy.ops.wm.open_mainfile(filepath=os.path.join(hu.hs.FIT_RUN, "fit", "hand_rigged.blend"))
    arm, ob = bpy.data.objects["hand rig"], bpy.data.objects["hand scan"]
    assert np.allclose(np.array(arm.matrix_world), np.eye(4)) and np.allclose(np.array(ob.matrix_world), np.eye(4))
    bones = arm.data.bones
    names = [b.name for b in bones]
    H0 = {b.name: np.array(b.head_local) for b in bones}
    T0 = {b.name: np.array(b.tail_local) for b in bones}

    def length_factor(n):
        if n == "palm" or (n.endswith("metacarpal") and not n.startswith("thumb")):
            return fac["palm_length"]
        return fac["finger"][n.split("_")[0]]
    # new rest joints: parents first, a bone keeps its direction, its head follows the parent's
    # tail when connected, or the parent's motion at the head point otherwise
    H1, T1, A = {}, {}, {}
    for b in bones:
        n, f = b.name, length_factor(b.name)
        if b.parent is None:
            h = H0[n]
        elif b.use_connect:
            h = T1[b.parent.name]
        else:
            h = A[b.parent.name](H0[n][None])[0]
        d = (T0[n] - H0[n]) / np.linalg.norm(T0[n] - H0[n])
        H1[n], T1[n] = h, h + (T0[n] - H0[n]) * f

        def make(h0=H0[n], h1=h, d=d, f=f):
            def m(P):
                q = P - h0
                along = q @ d
                return h1 + q + ((f - 1.0) * along)[:, None] * d
            return m
        A[n] = make()
    # skin: blend of the bones' maps by the vertex-group weights
    co = np.empty(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    gi = {g.index: g.name for g in ob.vertex_groups}
    W = np.zeros((len(co), len(names)))
    for v in ob.data.vertices:
        for g in v.groups:
            if gi.get(g.group) in names:
                W[v.index, names.index(gi[g.group])] = g.weight
    W[W.sum(1) == 0, names.index("palm")] = 1.0
    W /= W.sum(1, keepdims=True)
    new = sum(W[:, i:i + 1] * A[n](co) for i, n in enumerate(names))
    # width (across the palm, index MCP -> pinky MCP direction) and size, about the wrist
    w0 = H1["palm"]
    across = T1["index_metacarpal"] - T1["pinky_metacarpal"]
    across /= np.linalg.norm(across)
    G = fac["size"] * (np.eye(3) + (fac["palm_width"] - 1.0) * np.outer(across, across))
    g = lambda P: w0 + (P - w0) @ G.T  # noqa: E731
    new = g(new)
    ob.data.vertices.foreach_set("co", new.ravel())
    ob.data.update()
    for o in bpy.context.view_layer.objects:
        o.select_set(o == arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.data.edit_bones
    for n in names:                      # parents first: connected children follow their parent's tail
        eb[n].head = g(H1[n][None])[0].tolist()
        eb[n].tail = g(T1[n][None])[0].tolist()
    bpy.ops.object.mode_set(mode="OBJECT")
    os.makedirs(WORK, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=hs.FIT_BLEND)
    b = arm.data.bones
    fac["result_mm"] = {"palm_length": round(b["middle_metacarpal"].length, 1),
                        **{f + "_finger": round(sum(b[f"{f}_{s}"].length for s in (("metacarpal", "proximal", "distal") if f == "thumb" else ("proximal", "middle", "distal"))), 1) for f in FINGERS},
                        "hand_width_at_knuckles": round((b["index_metacarpal"].tail_local - b["pinky_metacarpal"].tail_local).length, 1)}
    json.dump(fac, open(os.path.join(WORK, "hand_factors.json"), "w"), indent=1)
    print("[big] random hand", json.dumps(fac), "->", hs.FIT_BLEND, flush=True)


def cmd_holdrig(a):
    """The approved hold scene without the scan hand; the cylinder rig is the 'hand rig' again."""
    import bpy
    bpy.ops.wm.open_mainfile(filepath=UPRIGHT_BLEND)
    for n in ("hand scan", "hand rig"):
        bpy.data.objects.remove(bpy.data.objects[n])
    o = bpy.data.objects["old cylinder hand rig"]
    o.name = o.data.name = "hand rig"
    os.makedirs(WORK, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=hs.HOLD_RIG)
    print("[big] hold scene ->", hs.HOLD_RIG, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scale").add_argument("--s", type=float, default=1.08)
    r = sub.add_parser("random")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--factors", default=None, help="json with the factors instead of a seed")
    sub.add_parser("holdrig")
    sub.add_parser("ref")
    sub.add_parser("start")
    s = sub.add_parser("solve")
    s.add_argument("--tag", default="big")
    s.add_argument("--start", default=None)
    s.add_argument("--nfev", type=int, default=60)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--w", default=None)
    b = sub.add_parser("blend")
    b.add_argument("--tag", default="big")
    a = ap.parse_args()
    if a.cmd == "scale":
        cmd_scale(a)
    elif a.cmd == "random":
        cmd_random(a)
    elif a.cmd == "holdrig":
        cmd_holdrig(a)
    elif a.cmd == "ref":
        hu.cmd_ref(a)
    elif a.cmd == "start":
        hu.cmd_start(a)
    elif a.cmd == "solve":
        hu.cmd_solve(a)
    elif a.cmd == "blend":
        a.solve = os.path.join(WORK, f"solve_{a.tag}.json")
        a.out_blend = OUT_BLEND
        hu.cmd_blend(a)
        print("[big] blend ->", OUT_BLEND, json.dumps(json.load(open(a.solve))["summary"]), flush=True)


if __name__ == "__main__":
    main()
