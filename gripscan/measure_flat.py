"""Measure finger segment lengths from a photo of the hand lying flat on the sheet.

    python -m gripscan.measure_flat photos/flat_top.jpg --camera data/camera.json \
        --out data/hand_flat.json --annotated out/flat_top_check.jpg

The hand lies palm down on the printed sheet, fingers straight and slightly
spread, with as much of the board visible around it as possible. Shoot from
straight above, about 40-50 cm away.

Landmarks are back-projected onto planes above the paper (a joint centre sits
roughly half a finger thickness above it), so the camera's perspective does
not inflate the lengths. The default heights can be overridden once the side
photo gives real finger thicknesses.
"""
import argparse
import json

import cv2
import numpy as np

from .board import Camera, detect, estimate_pose, pixels_to_board
from .landmarks import FINGERS, NAMES, SEGMENTS, Landmarker

# joint-centre height above the paper, mm (typical adult hand; refined later)
DEFAULT_HEIGHT_MM = {"wrist": 18, "cmc": 14, "mcp": 11, "pip": 9, "ip": 9, "dip": 7.5, "tip": 6}


def heights(overrides=None):
    table = {**DEFAULT_HEIGHT_MM, **(overrides or {})}
    return np.array([table[n.split("_")[-1]] for n in NAMES], float)


def measure(px, cam, pose, h):
    pts = np.vstack([pixels_to_board(px[i:i + 1], cam, pose, h[i]) for i in range(21)])
    fingers = {}
    for f, ids in FINGERS.items():
        seg = np.linalg.norm(np.diff(pts[ids], axis=0), axis=1)
        fingers[f] = {name: round(float(v), 2) for name, v in zip(SEGMENTS[f], seg)}
    palm = {
        "width_index_to_pinky_mcp": round(float(np.linalg.norm(pts[5] - pts[17])), 2),
        "length_wrist_to_middle_mcp": round(float(np.linalg.norm(pts[0] - pts[9])), 2),
    }
    return pts, fingers, palm


def annotate(image, px, pose, cam, obj, img_pts, fingers):
    out = image.copy()
    for a, b in [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)] + \
            [(ids[i], ids[i + 1]) for ids in FINGERS.values() for i in range(3)]:
        cv2.line(out, tuple(map(int, px[a])), tuple(map(int, px[b])), (0, 220, 0), 2)
    for p in px:
        cv2.circle(out, tuple(map(int, p)), 5, (0, 0, 255), -1)
    for p in img_pts:
        cv2.circle(out, tuple(map(int, p)), 3, (255, 128, 0), -1)
    y = 40
    for f, segs in fingers.items():
        txt = f"{f}: " + "  ".join(f"{k} {v:.1f}" for k, v in segs.items())
        cv2.putText(out, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        y += 38
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image")
    p.add_argument("--camera", required=True)
    p.add_argument("--model", default=None, help="hand_landmarker.task (default: models/)")
    p.add_argument("--out", default="data/hand_flat.json")
    p.add_argument("--annotated", default=None)
    a = p.parse_args()

    image = cv2.imread(a.image)
    if image is None:
        raise SystemExit(f"cannot read {a.image}")
    cam = Camera.from_dict(json.load(open(a.camera)))
    size = (image.shape[1], image.shape[0])
    if size != cam.size:
        cam = cam.scaled_to(size)
    d = detect(image)
    if d is None:
        raise SystemExit("board not found; keep more of the sheet visible around the hand")
    pose = estimate_pose(*d, cam)
    lm = Landmarker(a.model).detect(image)
    if lm is None:
        raise SystemExit("no hand found in the photo")
    px, handed, score = lm
    pts, fingers, palm = measure(px, cam, pose, heights())

    result = {
        "units": "mm",
        "method": "flat photo on ChArUco sheet, MediaPipe landmarks back-projected to joint heights",
        "image": a.image,
        "handedness_mediapipe": handed, "handedness_score": round(score, 3),
        "board": {"corners_used": pose.n_corners, "reprojection_rms_px": round(pose.reproj_px, 3),
                  "camera_height_mm": round(float(-pose.camera_centre()[2]), 1)},
        "fingers": fingers,
        "palm": palm,
        "landmarks_board_mm": {n: [round(float(v), 2) for v in pts[i]] for i, n in enumerate(NAMES)},
        "landmark_heights_mm": dict(DEFAULT_HEIGHT_MM),
    }
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({"fingers": fingers, "palm": palm, "board": result["board"]}, indent=1))
    if a.annotated:
        cv2.imwrite(a.annotated, annotate(image, px, pose, cam, *d, fingers))


if __name__ == "__main__":
    main()
