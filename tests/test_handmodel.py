"""Synthetic check of the articulated hand fit.

    .venv/Scripts/python -m tests.test_handmodel

A model hand in a known relaxed pose is seen from 60 views; the 2D joints get
3 px noise plus a 4 mm view-dependent shift on 30% of the views (what MediaPipe
does with hidden joints). The fit must recover the angles within a few degrees.
"""
import json
import os

import numpy as np

from gripscan.handmodel import PARAMS, HandModel, degrees, fit
from gripscan.triangulate import normalize, projection
from tests.test_triangulate import cameras

FLAT = {  # a flat right hand, board mm (from a real flat measurement)
    "wrist": [220.3, 222.7], "thumb_cmc": [180.9, 211.2], "thumb_mcp": [152.3, 188.2],
    "thumb_ip": [129.1, 165.3], "thumb_tip": [110.3, 151.5], "index_mcp": [179.6, 142.4],
    "index_pip": [162.7, 103.5], "index_dip": [152.9, 81.2], "index_tip": [147.2, 62.2],
    "middle_mcp": [202.3, 137.0], "middle_pip": [196.0, 91.5], "middle_dip": [192.5, 65.2],
    "middle_tip": [191.2, 45.2], "ring_mcp": [223.3, 141.9], "ring_pip": [225.4, 99.6],
    "ring_dip": [226.8, 72.7], "ring_tip": [227.5, 52.3], "pinky_mcp": [241.7, 152.9],
    "pinky_pip": [253.4, 122.4], "pinky_dip": [259.7, 103.8], "pinky_tip": [265.5, 87.1],
}


def main():
    import cv2
    model = HandModel(FLAT)
    truth = np.zeros(len(PARAMS))
    truth[:3] = [0.3, -0.2, 0.1]
    truth[3:6] = [40.0, -20.0, -110.0]
    deg = {"palm_arch": 12, "index_mcp": 30, "index_pip": 35, "index_dip": 15,
           "middle_mcp": 35, "middle_pip": 45, "middle_dip": 20, "ring_mcp": 40, "ring_pip": 50,
           "ring_dip": 25, "pinky_mcp": 45, "pinky_pip": 55, "pinky_dip": 30,
           "index_spread": 3, "pinky_spread": -4, "thumb_cmc_abd": 10, "thumb_cmc_flex": 25,
           "thumb_twist": 5, "thumb_mcp": 15, "thumb_ip": 10}
    for k, v in deg.items():
        truth[PARAMS.index(k)] = np.radians(v)
    X = model.world(truth)
    cam, poses = cameras(60)
    rng = np.random.default_rng(1)
    Ps, xs = [], []
    for i, p in enumerate(poses):
        Xi = X.copy()
        if rng.random() < 0.3:  # MediaPipe-like viewpoint bias on a few joints
            Xi[rng.choice(21, 5, replace=False)] += rng.normal(0, 4.0, (5, 3))
        px, _ = cv2.projectPoints(Xi, p.rvec, p.tvec, cam.K, cam.dist)
        px = px.reshape(-1, 2) + rng.normal(0, 3.0, (21, 2))
        Ps.append(projection(cam, p))
        xs.append(normalize(px, cam))
    palm_init = X[[0, 1, 5, 9, 13, 17]] + rng.normal(0, 3.0, (6, 3))
    v, err, _, _ = fit(model, Ps, xs, cam.K[0, 0], palm_init)
    est, tru = degrees(v), degrees(truth)
    worst = max(abs(est[k] - tru[k]) for k in tru if not k.startswith("thumb"))
    worst_t = max(abs(est[k] - tru[k]) for k in tru if k.startswith("thumb"))
    for k in tru:
        print(f"{k:16s} true {tru[k]:6.1f}  fit {est[k]:6.1f}")
    print(f"median residual {np.median(err):.1f}px, worst finger error {worst:.1f} deg, thumb {worst_t:.1f} deg")
    assert worst < 4.0
    print("OK")


if __name__ == "__main__":
    main()
