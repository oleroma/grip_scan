"""Read the posed hand rig back as joint positions (grip coordinates, mm).

    blender -b out/hand_rig.blend --python blender/read_rig.py -- --out data/hand_hold_reference.json

Then: python -m gripscan.check_hold data/hand_hold_reference.json
reports the contacts and joint angles of your pose.
"""
import argparse
import json
import sys

import bpy

FINGERS = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
           "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20]}
SEG = {"thumb": ["metacarpal", "proximal", "distal"]}
SEG_DEFAULT = ["proximal", "middle", "distal"]
NAMES = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_mcp", "index_pip",
         "index_dip", "index_tip", "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
         "ring_mcp", "ring_pip", "ring_dip", "ring_tip", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    arm = bpy.data.objects["hand rig"]
    mw = arm.matrix_world
    pb = arm.pose.bones
    J = {"wrist": mw @ pb["palm"].head}
    for f, ids in FINGERS.items():
        names = [f"{f}_{s}" for s in SEG.get(f, SEG_DEFAULT)]
        J[NAMES[ids[0]]] = mw @ pb[names[0]].head
        for s, bn in enumerate(names):
            J[NAMES[ids[s + 1]]] = mw @ pb[bn].tail
    out = {"source": bpy.data.filepath, "joints_grip_mm": {n: [round(c, 2) for c in J[n]] for n in NAMES}}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE", a.out)


main()
