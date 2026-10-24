"""Measure finger segment lengths from a photo of the hand lying flat on the sheet.

    python -m gripscan.measure_flat build/gen/photos/flat_top.jpg --camera data/camera.json \
        --out build/gen/data/hand_flat.json --annotated build/gen/out/flat_top_check.jpg

The hand lies palm down on the printed sheet, fingers straight and slightly
spread, with as much of the board visible around it as possible. Shoot from
straight above, about 40-50 cm away.

Landmarks are back-projected onto planes above the paper (a joint centre sits
roughly half a finger thickness above it), so the camera's perspective does
not inflate the lengths. The default heights can be overridden once the side
photo gives real finger thicknesses.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from .board import BoardSpec, Camera, detect, estimate_pose, make_board, pixels_to_board
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


def match_calibration(image, cam, board):
    """Bring the photo into the orientation the camera was calibrated in.

    Some phones rotate the pixels of portrait photos instead of setting an EXIF
    flag, so a portrait shot is the calibrated landscape image turned by 90
    degrees one way or the other. Both turns are tried and the one whose board
    pose fits best is kept."""
    w, h = image.shape[1], image.shape[0]
    if (w, h) == cam.size or (w / h - cam.size[0] / cam.size[1]) ** 2 < 1e-6:
        candidates = [(image, 0)]
    else:
        candidates = [(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 90),
                      (cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), -90)]
    best = None
    for img, rot in candidates:
        c = cam if (img.shape[1], img.shape[0]) == cam.size else cam.scaled_to((img.shape[1], img.shape[0]))
        d = detect(img, board)
        if d is None:
            continue
        pose = estimate_pose(*d, c)
        if best is None or pose.reproj_px < best[2].reproj_px:
            best = (img, d, pose, rot)
    if best is None:
        raise SystemExit("board not found; keep more of the board visible around the hand")
    if best[2].reproj_px > 3.0:
        print(f"warning: board pose fits poorly ({best[2].reproj_px:.1f} px); check the photo")
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images", nargs="+", help="one or more photos; several are averaged")
    p.add_argument("--camera", required=True)
    p.add_argument("--model", default=None, help="hand_landmarker.task (default: build/gen/models/)")
    p.add_argument("--out", default="build/gen/data/hand_flat.json")
    p.add_argument("--annotated-dir", default=None, help="write a checked copy of each photo here")
    p.add_argument("--board-spec", default=None, help="board json (default: printed sheet)")
    a = p.parse_args()
    spec = BoardSpec.load(a.board_spec)
    cam0 = Camera.from_dict(json.load(open(a.camera)))
    board = make_board(spec)
    landmarker = Landmarker(a.model)
    paths = [f for g in a.images for f in (glob.glob(g) or [g])]

    per_photo = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print("skip (cannot read):", path)
            continue
        image, d, pose, rotation = match_calibration(image, cam0, board)
        cam = cam0 if cam0.size == (image.shape[1], image.shape[0]) else cam0.scaled_to((image.shape[1], image.shape[0]))
        lm = landmarker.detect(image)
        if lm is None:
            print("skip (no hand found):", path)
            continue
        px, handed, score = lm
        pts, fingers, palm = measure(px, cam, pose, heights() + spec.surface_offset_mm)
        per_photo.append({
            "image": path,
            "rotation_applied_deg": rotation,
            "handedness_mediapipe": handed, "handedness_score": round(score, 3),
            "board": {"corners_used": pose.n_corners, "reprojection_rms_px": round(pose.reproj_px, 3),
                      "camera_distance_mm": round(float(-pose.camera_centre()[2]), 1)},
            "fingers": fingers,
            "palm": palm,
            "landmarks_board_mm": {n: [round(float(v), 2) for v in pts[i]] for i, n in enumerate(NAMES)},
        })
        if a.annotated_dir:
            os.makedirs(a.annotated_dir, exist_ok=True)
            out = os.path.join(a.annotated_dir, os.path.splitext(os.path.basename(path))[0] + "_check.jpg")
            cv2.imwrite(out, annotate(image, px, pose, cam, *d, fingers))
    if not per_photo:
        raise SystemExit("no usable photos")

    def combine(key):
        out = {}
        for group, vals in per_photo[0][key].items():
            if isinstance(vals, dict):
                out[group] = {k: {"mean": round(float(np.mean([r[key][group][k] for r in per_photo])), 2),
                                  "range": round(float(np.ptp([r[key][group][k] for r in per_photo])), 2)}
                              for k in vals}
            else:
                v = [r[key][group] for r in per_photo]
                out[group] = {"mean": round(float(np.mean(v)), 2), "range": round(float(np.ptp(v)), 2)}
        return out

    result = {
        "units": "mm",
        "method": "flat photo on ChArUco board, MediaPipe landmarks back-projected to joint heights",
        "note": ("segment = distance between MediaPipe landmarks (joint estimates from the skin), "
                 "not bone length; thumb 'metacarpal' starts at MediaPipe's CMC estimate, which is "
                 "the least reliable point. Handedness from MediaPipe is unreliable for "
                 "back-of-hand views."),
        "photos": len(per_photo),
        "fingers": combine("fingers"),
        "palm": combine("palm"),
        "landmark_heights_mm": dict(DEFAULT_HEIGHT_MM),
        "board_spec": spec.source, "surface_offset_mm": spec.surface_offset_mm,
        "per_photo": per_photo,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    for f, segs in result["fingers"].items():
        print(f"{f:7s}" + "  ".join(f"{k} {v['mean']:.1f} (+/-{v['range'] / 2:.1f})" for k, v in segs.items()))
    print("palm   " + "  ".join(f"{k} {v['mean']:.1f}" for k, v in result["palm"].items()))
    print(f"{len(per_photo)} photos -> {a.out}")

if __name__ == "__main__":
    main()
