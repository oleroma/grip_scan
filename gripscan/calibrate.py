"""Calibrate a phone camera from 10-20 photos of the printed sheet (once per phone).

    python -m gripscan.calibrate build/gen/photos/calib/*.jpg --out data/camera.json

Shoot the sheet lying flat from different angles and distances, filling
different parts of the frame. Use the same camera, zoom (1x) and resolution
as for the hand photos; turn off any "auto lens correction" if the phone has one.
"""
import argparse
import glob
import json

import cv2
import numpy as np

from .board import BoardSpec, Camera, detect, make_board


def calibrate(paths, spec=None):
    board = make_board(spec)
    objs, imgs, size, used = [], [], None, []
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            continue
        s = (im.shape[1], im.shape[0])
        if size is None:
            size = s
        elif s != size:
            raise ValueError(f"{p}: size {s} differs from {size}; use one resolution")
        d = detect(im, board)
        if d is None or len(d[0]) < 12:
            print("skip (board not found):", p)
            continue
        objs.append(d[0].astype(np.float32))
        imgs.append(d[1].astype(np.float32))
        used.append(p)
    if len(objs) < 5:
        raise RuntimeError(f"only {len(objs)} usable photos; need at least 5")
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None)
    return Camera(K, dist, size), float(rms), used


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images", nargs="+")
    p.add_argument("--out", default="data/camera.json")
    p.add_argument("--board-spec", default=None, help="board json (default: printed sheet)")
    a = p.parse_args()
    paths = [f for g in a.images for f in (glob.glob(g) or [g])]
    cam, rms, used = calibrate(paths, BoardSpec.load(a.board_spec))
    with open(a.out, "w") as f:
        json.dump({**cam.to_dict(), "rms_px": rms, "images": used}, f, indent=1)
    print(f"calibrated from {len(used)} photos, RMS {rms:.3f} px -> {a.out}")
    if rms > 1.0:
        print("warning: RMS above 1 px; retake blurry photos or check the print is flat")


if __name__ == "__main__":
    main()
