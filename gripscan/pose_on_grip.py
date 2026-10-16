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
CREASE_FROM_WRIST = 0.33
SQUISH_MM, SQUISH_RADIUS_MM = 6.0, 25.0
R1_SLACK_MM = 8.0
# the user's hold: the index leaves the palm in a straight line, then wraps the grip
HOLD_OVERRIDES_DEG = {"index_mcp": 5.0}
INDEX_STIFF = 3.0
TRIGGER_REACH_MM = 10.0
SWELL_ON_PALM = (0.2, 0.6)  # where along wrist -> knuckles the swell may land
W_KNUCKLE, KNUCKLE_SLACK_MM = 3.0, 5.0
KNUCKLE_ABOVE_TOP_MM = 5.0  # joint centre; the skin of the knuckle is then just above the top
R1_OVERLAP_MM = 15.0  # the index keeps its relaxed shape more firmly than the others


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hand", default="data/hand.json")
    p.add_argument("--board", default="data/board_gripv.json")
    p.add_argument("--grip", default="data/reference_basemesh_gripv.obj")
    p.add_argument("--init", default="data/hand_on_grip.json", help="rigid placement to start from")
    p.add_argument("--out", default="data/hand_posed_on_grip.json")
    p.add_argument("--starts", type=int, default=30)
    p.add_argument("--swell", default=None, help="x,y,z of the palm swell apex (default: the bump on "
                   "the outer side of the right handle, its outermost point in X)")
    a = p.parse_args()
    hand, board = json.load(open(a.hand)), json.load(open(a.board))
    sdf = MeshSDF(*load_obj(a.grip))
    model = HandModel(hand["template_palm_mm"])
    radii = hand["radii_mm"]
    relaxed = {n: np.radians(v) for n, v in hand["relaxed"]["angles_deg"].items()}
    # holding-posture targets that differ from the measured relaxed hand (degrees)
    for n, v in HOLD_OVERRIDES_DEG.items():
        relaxed[n] = np.radians(v)
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

    # anatomical limits on the absolute angles (degrees); the search may not leave them
    def limits(n):
        if n.endswith("_spread"):
            return -15.0, 15.0
        if n == "palm_arch":
            return 0.0, 25.0
        if n.startswith("thumb"):
            return -180.0, 180.0
        return -10.0, 110.0
    lo = np.array([np.radians(limits(n)[0]) - relaxed[n] for n in names])
    hi = np.array([np.radians(limits(n)[1]) - relaxed[n] for n in names])
    # relaxed values already outside a limit (measurement noise) keep a little room
    lo, hi = np.minimum(lo, -1e-3), np.maximum(hi, 1e-3)
    bounds = (np.concatenate([np.full(6, -np.inf), lo]), np.concatenate([np.full(6, np.inf), hi]))

    def hollow(J):
        """Where the swell nests: a third of the way from the wrist (where the palm meets the
        arm) to the centre of the knuckle row, on the palm skin."""
        c = J[0] + CREASE_FROM_WRIST * (J[[5, 9, 13, 17]].mean(0) - J[0])
        return c + np.array([0.0, 0.0, hand["palm_thickness_mm"] / 2])

    # the palm is soft: around the crease the swell may press in up to SQUISH_MM,
    # fading to nothing SQUISH_RADIUS_MM away; the crease itself is pressed in fully
    _, swell_idx = sdf.tree.query(swell)
    swell_n = sdf.N[swell_idx] / np.linalg.norm(sdf.N[swell_idx])
    crease_target = swell - SQUISH_MM * swell_n

    def palm_allowance(J, ps):
        d = np.linalg.norm(ps - hollow(J), axis=1)
        return SQUISH_MM * np.clip(1 - d / SQUISH_RADIUS_MM, 0, 1)

    def swell_contact(x, J):
        """Nearest point to the swell on the palm's centre line (wrist -> knuckle-row centre,
        between SWELL_ON_PALM fractions), on the palm skin, in grip coordinates. The swell may
        land anywhere along that line; where the hand sits is set by the knuckle rule."""
        lift = np.array([0.0, 0.0, hand["palm_thickness_mm"] / 2])
        k = J[[5, 9, 13, 17]].mean(0)
        a_ = transform(x, (J[0] + SWELL_ON_PALM[0] * (k - J[0]) + lift)[None])[0]
        b_ = transform(x, (J[0] + SWELL_ON_PALM[1] * (k - J[0]) + lift)[None])[0]
        t_ = np.clip((crease_target - a_) @ (b_ - a_) / ((b_ - a_) @ (b_ - a_)), 0, 1)
        return a_ + t_ * (b_ - a_)

    # where the index knuckle sits seen from above: about the Y of the B / Cross buttons
    abxy_t = {t["name"]: t["top_centre"] for t in board["thumb_targets"]}
    knuckle_y = 0.5 * (abxy_t["abxy cap right"][1] + abxy_t["abxy cap down"][1])
    knuckle_z = board["top_face"]["z"] + KNUCKLE_ABOVE_TOP_MM

    def resid(v):
        x, delta = v[:6], v[6:]
        J = joints(delta)
        fs, fr, _ = finger_samples(J, radii)
        ps = palm_samples(J, hand["palm_thickness_mm"])
        ids = FINGERS["index"]
        idx = np.array([J[ids[0]] + t * (J[ids[2]] - J[ids[0]]) for t in np.linspace(0, 1, 12)])
        # R1: the index lies on or near the bumper (within R1_SLACK_MM counts as on it),
        # it should not hook its tip to touch it
        d_r1 = np.linalg.norm(transform(x, idx) - r1, axis=1).min() - radii["index"][0]
        out = [W_R1 * max(0.0, d_r1 - R1_SLACK_MM)]
        # R2/R4: the fingertips need not rest on the triggers, only arrive at them:
        # no cost within TRIGGER_REACH_MM
        for f_, r_ in (("middle", r2), ("ring", r4)):
            d_ = np.linalg.norm(transform(x, pad_point(J, radii, f_)[None])[0] - r_)
            out.append(W_TRIG * max(0.0, d_ - TRIGGER_REACH_MM))
        out.append(W_THUMB * max(0.0, np.linalg.norm(transform(x, J[4:5])[0] - abxy) - 15.0))
        out += list(W_SWELL * (swell_contact(x, J) - crease_target))
        k_y = transform(x, J[5:6])[0][1]
        out.append(W_KNUCKLE * max(0.0, abs(k_y - knuckle_y) - KNUCKLE_SLACK_MM))
        # ... and its height: the knuckle slightly above the top face
        k_z = transform(x, J[5:6])[0][2]
        out.append(W_KNUCKLE * max(0.0, abs(k_z - knuckle_z) - KNUCKLE_SLACK_MM))
        gp = sdf(transform(x, ps))
        out += list(W_PALM * np.clip(gp, 0, None))
        # the index may overlap the bumper: no sinking cost for finger samples near R1
        fw = transform(x, fs)
        near_r1 = np.linalg.norm(fw - r1, axis=1) < R1_OVERLAP_MM
        out += list(W_PEN * np.where(near_r1, 0.0, np.minimum(0, sdf(fw) - fr)))
        sc_ = swell_contact(x, J)
        allow = SQUISH_MM * np.clip(1 - np.linalg.norm(transform(x, ps) - sc_, axis=1) / SQUISH_RADIUS_MM, 0, 1)
        out += list(W_PEN * np.minimum(0, gp + allow))
        out += [W_POSE * (INDEX_STIFF if n.startswith("index") else 1.0) * d / tol[n]
                for n, d in zip(names, delta)]
        return np.array(out)

    # start from the placement that best matches the four contacts of a real grip
    # (palm crease on the swell, index base on R1, middle pad on R2, ring pad on R4),
    # then refine with the joints free. Random orientations found non-grasps.
    J0 = joints(np.zeros(len(names)))
    ids = FINGERS["index"]
    A = np.array([hollow(J0), J0[ids[0]] + 0.5 * (J0[ids[2]] - J0[ids[0]]),
                  pad_point(J0, radii, "middle"), pad_point(J0, radii, "ring")])
    B = np.array([swell, r1, r2, r4])
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    x0 = np.concatenate([Rotation.from_matrix(R).as_rotvec(), cb - R @ ca])
    print("contact alignment of the relaxed hand (mm):",
          np.round(np.linalg.norm(A @ R.T + x0[3:] - B, axis=1), 1))
    # slide the start back so the index knuckle sits at the B / Cross button Y
    x0[4] += knuckle_y - transform(x0, J0[5:6])[0][1]
    x0[5] += knuckle_z - transform(x0, J0[5:6])[0][2]
    sol = least_squares(resid, np.concatenate([x0, np.zeros(len(names))]), loss="soft_l1",
                        f_scale=3.0, max_nfev=800, bounds=bounds)
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
        "index_knuckle_y_mm": round(float(Jw[5][1]), 1),
        "knuckle_target_y_mm": round(float(knuckle_y), 1),
        "index_knuckle_z_mm": round(float(Jw[5][2]), 1),
        "knuckle_target_z_mm": round(float(knuckle_z), 1),
        "swell_lands_on_palm_at": round(float(np.linalg.norm(np.linalg.solve(Rotation.from_rotvec(x[:3]).as_matrix(), swell_contact(x, J) - x[3:]) - J[0]) / np.linalg.norm(J[[5, 9, 13, 17]].mean(0) - J[0])), 2),
        "palm_crease_to_swell_mm": round(float(np.linalg.norm(transform(x, hollow(J)[None])[0] - swell)), 1),
        "swell_pressed_into_palm_mm": round(float(-sdf(swell_contact(x, J)[None])[0]), 1),
        "deepest_finger_sink_mm": round(float(gap.min()), 1),
        "deepest_palm_sink_beyond_soft_tissue_mm": round(float(np.min(sdf(transform(x, ps)) + SQUISH_MM * np.clip(
            1 - np.linalg.norm(transform(x, ps) - swell_contact(x, J), axis=1) / SQUISH_RADIUS_MM, 0, 1))), 1),
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
