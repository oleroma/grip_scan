"""Report how a given hand pose sits on the grip (e.g. the pose you made with the rig).

    python -m gripscan.check_hold data/hand_hold_reference.json

Prints the contacts (R1/R2/R4, palm swell, index knuckle position, thumb to
ABXY), how deep any finger sinks into the grip, and the joint angles compared
with the measured relaxed hand.
"""
import argparse
import json

import numpy as np

from .landmarks import FINGERS, NAMES
from .place_hand import MeshSDF, finger_samples, load_obj, pad_point
from .relaxed_pose import angles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pose")
    p.add_argument("--hand", default="data/hand.json")
    p.add_argument("--board", default="data/board_gripv.json")
    p.add_argument("--grip", default="data/reference_basemesh_gripv.obj")
    p.add_argument("--swell", default="82.9,-71.0,25.4")
    a = p.parse_args()
    hand, board = json.load(open(a.hand)), json.load(open(a.board))
    J = np.array([json.load(open(a.pose))["joints_grip_mm"][n] for n in NAMES])
    sdf = MeshSDF(*load_obj(a.grip))
    radii = hand["radii_mm"]
    swell = np.array([float(c) for c in a.swell.split(",")])

    ids = FINGERS["index"]
    idx = np.array([J[ids[0]] + t * (J[ids[2]] - J[ids[0]]) for t in np.linspace(0, 1, 30)])
    r1 = np.array(board["r1"]["target"])
    abxy = {t["name"]: np.array(t["top_centre"]) for t in board["thumb_targets"]}
    fs, fr, _ = finger_samples(J, radii)
    # palm centre line on the palm skin side (towards the grip)
    normal = np.cross(J[9] - J[0], J[5] - J[17])
    normal /= np.linalg.norm(normal)
    k = J[[5, 9, 13, 17]].mean(0)
    line = np.array([J[0] + t * (k - J[0]) for t in np.linspace(0, 1, 41)])
    for s in (1, -1):  # pick the side of the palm that faces the grip
        cand = line + s * normal * hand["palm_thickness_mm"] / 2
        if np.mean(sdf(cand)) < np.mean(sdf(line - s * normal * hand["palm_thickness_mm"] / 2)):
            break
    i_sw = int(np.argmin(np.linalg.norm(cand - swell, axis=1)))
    report = {
        "index_on_R1_mm": round(float(np.linalg.norm(idx - r1, axis=1).min() - radii["index"][0]), 1),
        "middle_tip_to_R2_mm": round(float(np.linalg.norm(pad_point(J, radii, "middle") - board["r2"]["press_centre"])), 1),
        "ring_tip_to_R4_mm": round(float(np.linalg.norm(pad_point(J, radii, "ring") - board["r4"]["press_centre"])), 1),
        "thumb_tip_to_abxy_centre_mm": round(float(np.linalg.norm(J[4] - np.mean([v for n, v in abxy.items() if n.startswith("abxy")], axis=0))), 1),
        "index_knuckle_xyz_mm": [round(float(c), 1) for c in J[5]],
        "swell_to_palm_line_mm": round(float(np.linalg.norm(cand[i_sw] - swell)), 1),
        "swell_lands_on_palm_at": round(i_sw / 40, 2),
        "deepest_finger_sink_mm": round(float((sdf(fs) - fr).min()), 1),
    }
    print(json.dumps(report, indent=1))
    ang = angles(J)
    relaxed = hand["relaxed"]["angles_deg"]
    print("joint angles (deg): pose / relaxed")
    for kname, v in ang.items():
        r = relaxed.get(kname)
        print(f"  {kname:20s} {v:6.1f}" + (f" / {r:6.1f}" if r is not None else ""))


if __name__ == "__main__":
    main()
