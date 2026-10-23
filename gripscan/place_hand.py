"""Place the relaxed hand (fixed pose) on an existing grip and report the fit.

    python -m gripscan.place_hand --hand data/hand.json --board data/board_gripv.json \
        --grip data/reference_basemesh_gripv.obj --out data/hand_on_grip.json

A rigid placement (6 DOF) only: the hand keeps its measured relaxed pose. It
looks for the placement where the middle fingertip rests on R2, the ring
fingertip on R4, the index finger lies on R1, and no part of the hand sinks
into the grip. The report then says, per finger segment, how far it floats
above or sinks into the surface: how well the grip already fits the relaxed hand.
"""
import argparse
import json

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .landmarks import FINGERS, NAMES


# ------------------------------------------------------------- grip distance field

def load_obj(path):
    V, F = [], []
    for line in open(path):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    return np.array(V), np.array(F)


class MeshSDF:
    """Approximate signed distance to a closed mesh from dense surface samples
    (about 0.7 mm apart); negative inside."""

    def __init__(self, V, F, spacing=0.7, seed=0):
        rng = np.random.default_rng(seed)
        A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        n = np.cross(B - A, C - A)
        area = np.linalg.norm(n, axis=1) / 2
        n = n / np.maximum(2 * area[:, None], 1e-12)
        count = np.maximum(1, (area / spacing ** 2).astype(int))
        idx = np.repeat(np.arange(len(F)), count)
        u, v = rng.random((2, len(idx)))
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        self.P = A[idx] + u[:, None] * (B - A)[idx] + v[:, None] * (C - A)[idx]
        self.N = n[idx]
        self.tree = cKDTree(self.P)

    def __call__(self, Q, k=3):
        d, i = self.tree.query(Q, k=k)
        nearest = self.P[i[:, 0]]
        # sign from the average normal of the nearest samples
        nrm = self.N[i].mean(axis=1)
        s = np.sign(np.einsum("ij,ij->i", Q - nearest, nrm))
        s[s == 0] = 1
        return s * d[:, 0]


# ------------------------------------------------------------- hand samples

def finger_samples(J, radii, step=3.0):
    """Points along every finger bone axis, with the capsule radius at each."""
    pts, rad, tag = [], [], []
    for f, ids in FINGERS.items():
        for s in range(3):
            a, b = J[ids[s]], J[ids[s + 1]]
            n = max(2, int(np.linalg.norm(b - a) / step) + 1)
            for t in np.linspace(0, 1, n):
                pts.append(a + t * (b - a))
                rad.append(radii[f][s])
                tag.append(f"{f}_{s}")
    return np.array(pts), np.array(rad), tag


def palm_samples(J, thickness, step=4.0):
    """Points on the palmar skin of the palm: the polygon wrist-CMC-MCPs, pushed
    half the palm thickness towards the palm (+z in the palm frame)."""
    poly = J[[0, 1, 5, 9, 13, 17]]
    c = poly.mean(0)
    pts = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        for s in np.linspace(0, 1, 5):
            for t in np.linspace(0, 1, 6):
                pts.append(c + t * (a + s * (b - a) - c))
    P = np.array(pts)
    P[:, 2] += thickness / 2
    return P


def pad_point(J, radii, finger):
    """Fingertip pad: on the distal segment, 70% out, one radius towards the palm side."""
    ids = FINGERS[finger]
    a, b = J[ids[2]], J[ids[3]]
    axis = (b - a) / np.linalg.norm(b - a)
    prev = J[ids[2]] - J[ids[1]]
    side = np.cross(np.cross(prev, axis), axis)  # in the bending plane, towards the palm
    side /= np.linalg.norm(side) + 1e-9
    return a + 0.7 * (b - a) + radii[finger][2] * side


# ------------------------------------------------------------- placement

def transform(x, P):
    return P @ Rotation.from_rotvec(x[:3]).as_matrix().T + x[3:]


def place(hand, board, sdf, starts=60, seed=0):
    J = np.array([hand["relaxed"]["joints_palm_mm"][n] for n in NAMES])
    radii = hand["radii_mm"]
    fs, fr, _ = finger_samples(J, radii)
    ps = palm_samples(J, hand["palm_thickness_mm"])
    pads = {f: pad_point(J, radii, f) for f in ("middle", "ring")}
    idx_axis = np.array([J[FINGERS["index"][0]] + t * (J[FINGERS["index"][2]] - J[FINGERS["index"][0]])
                         for t in np.linspace(0, 1, 12)])
    r2 = np.array(board["r2"]["press_centre"])
    r4 = np.array(board["r4"]["press_centre"])
    r1 = np.array(board["r1"]["target"])
    # the bumper sits in an opening below the uncut basemesh surface; the finger can only
    # reach the surface, so lift the target along the bumper normal onto it
    r1 = r1 + np.array(board["r1"]["normal"]) * max(0.0, -float(sdf(r1[None])[0]))
    r_idx = radii["index"][0]
    abxy = np.mean([t["top_centre"] for t in board["thumb_targets"] if t["name"].startswith("abxy")], axis=0)
    top_z = board["top_face"]["z"]
    thumb_tip = J[4]
    thumb_mcp = J[2]
    # the relaxed thumb rests near the face buttons and only has to move a little to
    # press them; how far it may be is the thumb's own reach from its MCP joint
    thumb_slack = 15.0

    def resid(x):
        out = []
        out += list(transform(x, pads["middle"][None])[0] - r2)
        out += list(transform(x, pads["ring"][None])[0] - r4)
        d_r1 = np.linalg.norm(transform(x, idx_axis) - r1, axis=1).min() - r_idx
        out.append(3.0 * d_r1)  # the index finger must lie on R1
        pen_f = np.minimum(0, sdf(transform(x, fs)) - fr)       # finger capsules sinking in
        pen_p = np.minimum(0, sdf(transform(x, ps)))            # palm skin sinking in
        out += list(1.5 * pen_f) + list(1.5 * pen_p)
        # thumb over the top, near the face buttons
        d_thumb = np.linalg.norm(transform(x, thumb_tip[None])[0] - abxy)
        out.append(0.7 * max(0.0, d_thumb - thumb_slack))
        out.append(0.5 * max(0.0, top_z - transform(x, thumb_mcp[None])[0][2] - 15.0))
        # the palm rests on the handle: palm skin close to the surface (weak, robust)
        out += list(0.3 * np.clip(sdf(transform(x, ps)), 0, None))
        return np.array(out)

    # coarse pass from many random orientations, then refine the best few
    rng = np.random.default_rng(seed)
    coarse = []
    for k in range(starts):
        rv = Rotation.random(random_state=rng.integers(1 << 31)).as_rotvec()
        t = r2 - Rotation.from_rotvec(rv).apply(pads["middle"])
        sol = least_squares(resid, np.concatenate([rv, t]), loss="soft_l1", f_scale=3.0, max_nfev=25)
        coarse.append(sol)
    coarse.sort(key=lambda s: s.cost)
    best = min((least_squares(resid, c.x, loss="soft_l1", f_scale=3.0, max_nfev=300) for c in coarse[:5]),
               key=lambda s: s.cost)
    return best.x, J, radii, fs, fr, ps, pads


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hand", default="data/hand.json")
    p.add_argument("--board", default="data/board_gripv.json")
    p.add_argument("--grip", default="data/reference_basemesh_gripv.obj")
    p.add_argument("--out", default="data/hand_on_grip.json")
    p.add_argument("--starts", type=int, default=60)
    a = p.parse_args()
    hand, board = json.load(open(a.hand)), json.load(open(a.board))
    sdf = MeshSDF(*load_obj(a.grip))
    x, J, radii, fs, fr, ps, pads = place(hand, board, sdf, a.starts)

    Jw = transform(x, J)
    _, _, tags = finger_samples(J, radii)
    gap = sdf(transform(x, fs)) - fr   # >0 floating above the surface, <0 sinking in
    per_seg = {}
    for tag in dict.fromkeys(tags):
        g = gap[[i for i, t in enumerate(tags) if t == tag]]
        f, s = tag.split("_")
        per_seg[f"{f}_{['proximal', 'middle', 'distal'][int(s)] if f != 'thumb' else ['metacarpal', 'proximal', 'distal'][int(s)]}"] = {
            "min_gap_mm": round(float(g.min()), 1), "median_gap_mm": round(float(np.median(g)), 1)}
    palm_gap = sdf(transform(x, ps))
    pad_err = {
        "middle_on_R2_mm": round(float(np.linalg.norm(transform(x, pads["middle"][None])[0] - board["r2"]["press_centre"])), 1),
        "ring_on_R4_mm": round(float(np.linalg.norm(transform(x, pads["ring"][None])[0] - board["r4"]["press_centre"])), 1),
    }
    thumb_tip = Jw[4]
    reach = {t["name"]: round(float(np.linalg.norm(thumb_tip - t["top_centre"])), 1) for t in board["thumb_targets"]}
    result = {
        "units": "mm",
        "placement": {"rotvec": [round(float(v), 5) for v in x[:3]], "translation": [round(float(v), 2) for v in x[3:]]},
        "triggers": pad_err,
        "finger_gap_to_grip": per_seg,
        "palm_gap_to_grip": {"min": round(float(palm_gap.min()), 1), "median": round(float(np.median(palm_gap)), 1),
                             "share_within_3mm": round(float(np.mean(np.abs(palm_gap) < 3)), 2)},
        "relaxed_thumb_tip_to_controls": reach,
        "joints_grip_mm": {n: [round(float(c), 2) for c in Jw[i]] for i, n in enumerate(NAMES)},
        "note": ("gap > 0: the segment floats above the grip surface; gap < 0: it would sink in. "
                 "Thumb distances are for the relaxed thumb; reaching the buttons is a thumb "
                 "movement the solver checks separately."),
    }
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: result[k] for k in ("triggers", "palm_gap_to_grip")}, indent=1))
    for k, v in per_seg.items():
        print(f"{k:18s} min {v['min_gap_mm']:6.1f}  median {v['median_gap_mm']:6.1f}")
    print("relaxed thumb tip to controls:", reach)
    print("->", a.out)


if __name__ == "__main__":
    main()
