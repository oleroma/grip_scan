"""Synthetic check of triangulation and joint angles.

    .venv/Scripts/python -m tests.test_triangulate

A curled hand (known angles) floats 120 mm in front of the board; cameras on
an arc look at it; joints are projected with 1.5 px noise, triangulated, and
the angles compared with the truth. A flat hand must come back flat.
"""
import cv2
import numpy as np

from gripscan.board import Camera, Pose
from gripscan.relaxed_pose import angles
from gripscan.triangulate import normalize, projection, robust_point


def hand(curl):
    """21 joints (mm) in board frame; curl = (mcp, pip, dip) degrees per finger."""
    W = np.zeros((21, 3))
    W[0] = [0, 0, -120]
    bases = {"thumb": None, "index": 30, "middle": 10, "ring": -10, "pinky": -28}
    lengths = {"index": (42, 24, 20), "middle": (46, 26, 20), "ring": (43, 26, 20), "pinky": (33, 20, 18)}
    starts = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    for f, y in bases.items():
        if f == "thumb":
            continue
        i0 = starts[f]
        W[i0] = W[0] + [88 if f == "middle" else 82, y, 0]
        d = np.array([1.0, 0, 0])
        ang = 0.0
        for k, (L, c) in enumerate(zip(lengths[f], curl[f])):
            ang += np.radians(c)
            d = np.array([np.cos(ang), 0, np.sin(ang)])  # curl towards +z (towards the board)
            W[i0 + k + 1] = W[i0 + k] + L * d
    # thumb: a simple chain out to the side
    W[1] = W[0] + [25, 25, 0]
    W[2] = W[1] + [25, 20, 0]
    W[3] = W[2] + [25, 10, 5]
    W[4] = W[3] + [20, 5, 8]
    return W


def cameras(n=40):
    K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    cam = Camera(K, np.zeros(5), (1920, 1080))
    poses = []
    target = np.array([60.0, 0, -120])
    for i in range(n):
        az = np.radians(-40 + 80 * i / (n - 1))
        el = np.radians(40 * np.sin(4 * np.pi * i / n))
        c = target + 350 * np.array([np.sin(az) * np.cos(el), np.sin(el), -np.cos(az) * np.cos(el)])
        z = (target - c) / np.linalg.norm(target - c)
        x = np.cross([0, 1.0, 0], z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.vstack([x, y, z])
        t = -R @ c
        poses.append(Pose(cv2.Rodrigues(R)[0], t.reshape(3, 1), 0.0, 0))
    return cam, poses


def roundtrip(W, cam, poses, noise=1.5, seed=0):
    rng = np.random.default_rng(seed)
    Ps, xs = [], []
    for p in poses:
        px, _ = cv2.projectPoints(W, p.rvec, p.tvec, cam.K, cam.dist)
        px = px.reshape(-1, 2) + rng.normal(0, noise, (21, 2))
        Ps.append(projection(cam, p))
        xs.append(normalize(px, cam))
    return np.array([robust_point(Ps, [x[j] for x in xs], cam.K[0, 0])[0] for j in range(21)])


def main():
    cam, poses = cameras()
    for name, curl in [("flat", {f: (0, 0, 0) for f in ("index", "middle", "ring", "pinky")}),
                       ("relaxed", {"index": (20, 30, 15), "middle": (25, 40, 20),
                                    "ring": (30, 45, 25), "pinky": (35, 55, 30)})]:
        W = hand(curl)
        est = roundtrip(W, cam, poses)
        pos_err = np.linalg.norm(est - W, axis=1).max()
        true_a, est_a = angles(W), angles(est)
        keys = [k for k in true_a if not k.startswith("thumb")]
        ang_err = max(abs(abs(est_a[k]) - abs(true_a[k])) for k in keys)
        print(f"[{name}] worst joint position error {pos_err:.2f} mm, worst angle error {ang_err:.2f} deg")
        print("   true", {k: round(abs(true_a[k]), 1) for k in keys if k.endswith(("_pip", "_mcp"))})
        print("   est ", {k: round(abs(est_a[k]), 1) for k in keys if k.endswith(("_pip", "_mcp"))})
        assert pos_err < 1.0 and ang_err < 3.0
    print("OK")


if __name__ == "__main__":
    main()
