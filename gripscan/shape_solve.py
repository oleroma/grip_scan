"""Shape solver v1: a new right handle that fills the hollow of the held hand.

    python -m gripscan.shape_solve --pose data/hand_hold_reference.json --out out/handle_v1

1. The hand in its hold is turned into a signed distance field (capsules for the
   finger bones, thick capsules along the metacarpals for the palm).
2. The space the hand wraps around is the convex envelope of the palm and the
   four finger bones (a power grasp holds its object inside it), minus the hand.
3. New grip = the current basemesh kept as it is above the handle (top face,
   body, board pocket) + the hollow inside the handle region, smoothly joined.
   Where the palm swell presses in, the soft palm is allowed (the hand is shrunk
   by SQUISH_MM around the swell contact).
4. Meshed with marching cubes as the right half (X >= 0), ready to go in place
   of the `basemesh GripV` half; plus a comparison with the current handle.

This is a constructive first version (no optimisation yet): it shows the shape
the hold implies. Trigger placement and smoothing for printing come next.
"""
import argparse
import json

import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay
from skimage import measure

from .landmarks import FINGERS, NAMES
from .place_hand import MeshSDF, load_obj

VOXEL = 1.0
BOUNDS = np.array([[0.0, 110.0], [-115.0, 10.0], [-45.0, 75.0]])  # right half, mm
HANDLE_X_MIN, HANDLE_Z_MAX, BLEND_MM = 28.0, 36.0, 6.0             # where the handle may change
SQUISH_MM = 4.0       # soft palm over the swell


def capsule_sdf(P, a, b, r):
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1) - r


def hand_sdf(P, J, hand, swell=None):
    radii = hand["radii_mm"]
    d = np.full(len(P), np.inf)
    for f, ids in FINGERS.items():
        for s in range(3):
            d = np.minimum(d, capsule_sdf(P, J[ids[s]], J[ids[s + 1]], radii[f][min(s, 2)]))
    t = hand["palm_thickness_mm"] / 2
    for i in (5, 9, 13, 17):
        d = np.minimum(d, capsule_sdf(P, J[0] + 0.25 * (J[i] - J[0]), J[i], t))
    d = np.minimum(d, capsule_sdf(P, J[0], J[1], t))
    if swell is not None:  # soft palm: let the swell press in
        d = d + SQUISH_MM * np.exp(-np.sum((P - swell) ** 2, axis=1) / (2 * 15.0 ** 2))
    return d


def smooth_union(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose", default="data/hand_hold_reference.json")
    p.add_argument("--hand", default="data/hand.json")
    p.add_argument("--board", default="data/board_gripv.json")
    p.add_argument("--grip", default="data/reference_basemesh_gripv.obj")
    p.add_argument("--out", default="out/handle_v1")
    a = p.parse_args()
    pose, hand, board = json.load(open(a.pose)), json.load(open(a.hand)), json.load(open(a.board))
    J = np.array([pose["joints_grip_mm"][n] for n in NAMES])
    swell = np.array(pose.get("palm_swell_point", [82.9, -71.0, 25.4]))

    axes = [np.arange(lo, hi + VOXEL, VOXEL) for lo, hi in BOUNDS]
    shape = tuple(len(x) for x in axes)
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    print("grid", shape, len(G), "voxels")

    # current grip as a distance field (negative inside)
    V, F = load_obj(a.grip)
    old = MeshSDF(V, F)(G, k=3).reshape(shape)
    # hand in its hold
    H = hand_sdf(G, J, hand, swell).reshape(shape)
    hand_in = H <= 0

    # the grasped volume: what the palm and the four fingers enclose. A power grasp
    # holds the object inside the convex envelope of the palm and finger bones; the
    # thumb rides on top and is left out. The hand itself is then cut away.
    X, Y, Z = (G[:, i].reshape(shape) for i in range(3))
    pts = []
    for f in ("index", "middle", "ring", "pinky"):
        ids = FINGERS[f]
        for s_ in range(3):
            for t in np.linspace(0, 1, 6):
                pts.append(J[ids[s_]] + t * (J[ids[s_ + 1]] - J[ids[s_]]))
        for t in np.linspace(0.25, 1, 5):  # metacarpal
            pts.append(J[0] + t * (J[ids[0]] - J[0]))
    pts.append(J[0])
    hull = Delaunay(np.array(pts))
    enclosed = (hull.find_simplex(G) >= 0).reshape(shape)

    region = (X >= HANDLE_X_MIN) & (Z <= HANDLE_Z_MAX)
    old_in = old <= 0
    # keep the body; inside the handle region the handle becomes the enclosed volume,
    # bridged to the body by the old handle's top band so they stay connected
    band = region & (Z > HANDLE_Z_MAX - BLEND_MM) & old_in
    solid = (old_in & ~region) | (enclosed & region) | band
    solid &= ~hand_in              # contact with the hand, never inside it
    grip = (ndimage.distance_transform_edt(~solid, sampling=VOXEL)
            - ndimage.distance_transform_edt(solid, sampling=VOXEL))
    grip = ndimage.gaussian_filter(grip, 1.5)

    verts, faces, _, _ = measure.marching_cubes(grip, 0.0, spacing=(VOXEL,) * 3)
    verts += BOUNDS[:, 0]
    with open(a.out + ".obj", "w") as fh:
        fh.write("# grip_scan handle v1, right half (X >= 0), mm\n")
        for v in verts:
            fh.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for f in faces:
            fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")

    # comparison with the current handle, in the handle region
    added = np.sum((grip <= 0) & (old > 0) & region) * VOXEL ** 3 / 1000
    removed = np.sum((grip > 0) & (old <= 0) & region) * VOXEL ** 3 / 1000
    surf = np.abs(H) < 1.0
    contact = np.mean(np.abs(grip[surf & region]) < 1.5) if np.any(surf & region) else 0.0
    report = {
        "method": "grasp envelope (convex hull of palm + finger bones) minus the hand",
        "handle_volume_cm3": {"current": round(float(np.sum((old <= 0) & region) / 1000), 1),
                              "new": round(float(np.sum((grip <= 0) & region) / 1000), 1),
                              "added": round(float(added), 1), "removed": round(float(removed), 1)},
        "hand_skin_touching_grip_share": round(float(contact), 2),
        "mesh": {"vertices": len(verts), "faces": len(faces), "file": a.out + ".obj"},
    }
    json.dump(report, open(a.out + ".json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
