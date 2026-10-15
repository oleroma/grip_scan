"""Combine the flat and relaxed measurements into data/hand.json (the solver's hand).

    python -m gripscan.build_hand --flat data/hand_flat.json --relaxed data/hand_relaxed.json \
        [--widths data/finger_widths.json] --out data/hand.json

Finger widths: caliper values (mm, side to side, middle of each segment) if
given, otherwise typical adult values. Capsule radius = 0.48 x width (fingers
are a little wider than deep).
"""
import argparse
import json

import numpy as np

from .handmodel import ANGLE_PARAMS, OFFSETS, PARAMS, HandModel
from .landmarks import NAMES

DEFAULT_WIDTHS_MM = {  # middle of proximal, middle, distal segment (thumb: metacarpal is palm)
    "thumb": [24.0, 21.0, 19.5],
    "index": [19.0, 17.0, 15.0],
    "middle": [19.5, 17.5, 15.5],
    "ring": [18.0, 16.5, 14.5],
    "pinky": [16.0, 14.5, 13.0],
}
DEFAULT_PALM_THICKNESS_MM = 30.0
RADIUS_PER_WIDTH = 0.48


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flat", default="data/hand_flat.json")
    p.add_argument("--relaxed", default="data/hand_relaxed.json")
    p.add_argument("--widths", default=None, help='{"index": [w1, w2, w3], ..., "palm_thickness": t}')
    p.add_argument("--out", default="data/hand.json")
    a = p.parse_args()

    flat = json.load(open(a.flat))
    relaxed = json.load(open(a.relaxed))
    model = HandModel(flat["per_photo"][0]["landmarks_board_mm"])
    widths = dict(DEFAULT_WIDTHS_MM)
    palm_t = DEFAULT_PALM_THICKNESS_MM
    source = "typical adult values"
    if a.widths:
        w = json.load(open(a.widths))
        palm_t = w.pop("palm_thickness", palm_t)
        widths.update(w)
        source = a.widths

    # relaxed pose: median angle per joint over the takes, and median wrist/CMC offsets
    q = {n: 0.0 for n in PARAMS[6:]}
    tol = {}
    for n, v in relaxed["joints"].items():
        q[n] = np.radians(v["deg"])
        tol[n] = max(5.0, v["range_between_takes"])  # degrees; at least 5
    for n in OFFSETS:
        q[n] = float(np.median([t["model_offsets_mm"][n] for t in relaxed["takes"]]))
    # the wrist offset only absorbs where MediaPipe puts its wrist point (forearm side,
    # moves with wrist bend); the hand's own geometry keeps the flat-photo wrist
    for n in ("wrist_dx", "wrist_dy", "wrist_dz"):
        q[n] = 0.0
    X = model.local(q)

    hand = {
        "units": "mm",
        "handedness": "right",
        "frame": "palm frame: origin wrist landmark, x towards middle MCP, z towards the palm",
        "segments_mm": {f: [round(L, 2) for L in model.len[f]] for f in model.len},
        "radii_mm": {f: [round(RADIUS_PER_WIDTH * w, 2) for w in ws] for f, ws in widths.items()},
        "palm_thickness_mm": palm_t,
        "widths_source": source,
        "template_palm_mm": {n: [round(float(c), 2) for c in model.tpl[i]] for i, n in enumerate(NAMES)},
        "relaxed": {
            "angles_deg": {n: round(float(np.degrees(q[n])), 1) for n in ANGLE_PARAMS},
            "tolerance_deg": tol,
            "offsets_mm": {n: round(q[n], 1) for n in OFFSETS},
            "joints_palm_mm": {n: [round(float(c), 2) for c in X[i]] for i, n in enumerate(NAMES)},
        },
        "sources": {"flat": a.flat, "relaxed": a.relaxed},
    }
    with open(a.out, "w") as f:
        json.dump(hand, f, indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
