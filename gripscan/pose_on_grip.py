"""Pose the hand on an existing grip: placement plus joint changes away from relaxed.

    python -m gripscan.pose_on_grip --hand data/hand.json --board data/board_gripv.json \
        --grip data/reference_basemesh_gripv.obj --init data/hand_on_grip.json \
        --out data/hand_posed_on_grip.json

This is the posture half of the solver, run against a fixed grip. The hand may
leave its relaxed pose, at a cost of (change / tolerance)^2 per joint, to:
- lay the index finger on R1 (not necessarily the tip),
- put the middle and ring fingertip pads on the R2 and R4 press centres,
- keep the thumb over the face buttons,
- nest the grip's palm swell (the bump on the outer side) in the palm's central
  crease, which the wrapping fingers press against it,
- never sink into the grip.
The report lists how far every joint had to move from relaxed: the joints
the current grip forces out of the relaxed hand.
"""
import argparse
import json

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .handmodel import ANGLE_PARAMS, OFFSETS, HandModel
from .landmarks import FINGERS, NAMES
from .place_hand import MeshSDF, finger_samples, load_obj, pad_point, palm_samples, transform

W_R1, W_TRIG, W_THUMB, W_PALM, W_PEN, W_POSE = 3.0, 1.0, 0.3, 0.0, 2.0, 1.0
W_SWELL = 2.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hand", default="data/hand.json")
    p.add_argument("--board", default="data/board_gripv.json")
    p.add_argument("--grip", default="data/reference_basemesh_gripv.obj")
    p.add_argument("--init", default="data/hand_on_grip.json", help="rigid placement to start from")
    p.add_argument("--out", default="data/hand_posed_on_grip.json")
    p.add_argument("--starts", type=int, default=30)
    p.add_argument("--swell", default=None, help="x,y,z of the palm swell apex (default: outermost "
                   "point of the right handle)")
    a = p.parse_args()
    hand, board = json.load(open(a.hand)), json.load(open(a.board))
    sdf = MeshSDF(*load_obj(a.grip))
    model = HandModel(hand["template_palm_mm"])
    radii = hand["radii_mm"]
    relaxed = {n: np.radians(v) for n, v in hand["relaxed"]["angles_deg"].items()}
    tol = {n: np.radians(max(8.0, v)) for n, v in hand["relaxed"]["tolerance_deg"].items()}
    offs = dict(hand["relaxed"]["offsets_mm"])
    offs.update({"wrist_dx": 0.0, "wrist_dy": 0.0, "wrist_dz": 0.0})
    names = list(ANGLE_PARAMS)
    V, F = load_obj(a.grip)
    if a.swell:
        swell = np.array([float(c) for c in a.swell.split(",")])
    else:
        right = V[(V[:, 0] > 0) & (V[:, 2] < board["top_face"]["z"] - 10)]
        swell = right[np.argmax(right[:, 0])]

    r1 = np.array(board["r1"]["target"])
    # the bumper sits in an opening below the uncut basemesh surface; the finger can only
    # reach the surface, so lift the target along the bumper normal onto it
    r1 = r1 + np.array(board["r1"]["normal"]) * max(0.0, -float(sdf(r1[None])[0]))
    r2 = np.array(board["r2"]["press_centre"])
    r4 = np.array(board["r4"]["press_centre"])
    abxy = np.mean([t["top_centre"] for t in board["thumb_targets"] if t["name"].startswith("abxy")], axis=0)

    def joints(delta):
        q = {n: relaxed[n] + d for n, d in zip(names, delta)}
        q.update(offs)
        return model.local(q)

    def hollow(J):
        """Centre of the palm's crease: between wrist and middle MCP, on the palm skin."""
        c = J[0] + 0.55 * (J[9] - J[0])
        return c + np.array([0.0, 0.0, hand["palm_thickness_mm"] / 2])

    def resid(v):
        x, delta = v[:6], v[6:]
        J = joints(delta)
        fs, fr, _ = finger_samples(J, radii)
        ps = palm_samples(J, hand["palm_thickness_mm"])
        ids = FINGERS["index"]
        idx = np.array([J[ids[0]] + t * (J[ids[2]] - J[ids[0]]) for t in np.linspace(0, 1, 12)])
        out = [W_R1 * (np.linalg.norm(transform(x, idx) - r1, axis=1).min() - radii["index"][0])]
        out += list(W_TRIG * (transform(x, pad_point(J, radii, "middle")[None])[0] - r2))
        out += list(W_TRIG * (transform(x, pad_point(J, radii, "ring")[None])[0] - r4))
        out.append(W_THUMB * max(0.0, np.linalg.norm(transform(x, J[4:5])[0] - abxy) - 15.0))
        out += list(W_SWELL * (transform(x, hollow(J)[None])[0] - swell))
        gp = sdf(transform(x, ps))
        out += list(W_PALM * np.clip(gp, 0, None))
        out += list(W_PEN * np.minimum(0, sdf(transform(x, fs)) - fr)) + list(W_PEN * np.minimum(0, gp))
        out += [W_POSE * d / tol[n] for n, d in zip(names, delta)]
        return np.array(out)

    # starts: the given rigid placement plus random orientations with the index on R1;
    # a short run from each, then the best few are refined
    J0 = joints(np.zeros(len(names)))
    ic = J0[FINGERS["index"][0]] + 0.6 * (J0[FINGERS["index"][2]] - J0[FINGERS["index"][0]])
    rng = np.random.default_rng(0)
    init = json.load(open(a.init))["placement"]
    starts = [np.concatenate([init["rotvec"], init["translation"]])]
    for _ in range(a.starts):
        rv = Rotation.random(random_state=rng.integers(1 << 31)).as_rotvec()
        starts.append(np.concatenate([rv, r1 - Rotation.from_rotvec(rv).apply(ic)]))
    coarse = []
    for x0 in starts:
        sol = least_squares(resid, np.concatenate([x0, np.zeros(len(names))]), loss="soft_l1",
                            f_scale=3.0, max_nfev=40)
        coarse.append(sol)
    coarse.sort(key=lambda s: s.cost)
    sol = min((least_squares(resid, c.x, loss="soft_l1", f_scale=3.0, max_nfev=600) for c in coarse[:4]),
              key=lambda s: s.cost)
    x, delta = sol.x[:6], sol.x[6:]
    J = joints(delta)
    Jw = transform(x, J)
    fs, fr, tags = finger_samples(J, radii)
    gap = sdf(transform(x, fs)) - fr
    ps = palm_samples(J, hand["palm_thickness_mm"])
    ids = FINGERS["index"]
    idx = np.array([J[ids[0]] + t * (J[ids[2]] - J[ids[0]]) for t in np.linspace(0, 1, 30)])
    report = {
        "index_on_R1_mm": round(float(np.linalg.norm(transform(x, idx) - r1, axis=1).min() - radii["index"][0]), 1),
        "middle_on_R2_mm": round(float(np.linalg.norm(transform(x, pad_point(J, radii, "middle")[None])[0] - r2)), 1),
        "ring_on_R4_mm": round(float(np.linalg.norm(transform(x, pad_point(J, radii, "ring")[None])[0] - r4)), 1),
        "thumb_tip_to_abxy_centre_mm": round(float(np.linalg.norm(Jw[4] - abxy)), 1),
        "palm_crease_to_swell_mm": round(float(np.linalg.norm(transform(x, hollow(J)[None])[0] - swell)), 1),
        "deepest_sink_mm": round(float(min(gap.min(), sdf(transform(x, ps)).min())), 1),
        "palm_gap_median_mm": round(float(np.median(sdf(transform(x, ps)))), 1),
    }
    change = {n: round(float(np.degrees(d)), 1) for n, d in zip(names, delta)}
    result = {
        "units": "mm, degrees",
        "fit": report,
        "change_from_relaxed_deg": change,
        "palm_swell_point": [round(float(c), 2) for c in swell],
        "placement": {"rotvec": [float(v) for v in x[:3]], "translation": [float(v) for v in x[3:]]},
        "joints_grip_mm": {n: [round(float(c), 2) for c in Jw[i]] for i, n in enumerate(NAMES)},
    }
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(report, indent=1))
    for n, d in sorted(change.items(), key=lambda kv: -abs(kv[1])):
        if abs(d) >= 3:
            print(f"{n:16s} {d:+6.1f} deg   (tolerance {np.degrees(tol[n]):.0f})")
    print("->", a.out)


if __name__ == "__main__":
    main()
