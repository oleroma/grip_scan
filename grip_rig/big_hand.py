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
