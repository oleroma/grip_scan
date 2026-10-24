"""Click two points in a board photo and get their distance in mm (scale check).

    python -m gripscan.measure_points build/gen/photos/check/caliper.jpg --camera data/camera.json \
        --board-spec data/board_screen.json --height-mm 1.5

Click near the first point, then click it precisely in the zoomed view; same
for the second point. --height-mm is how far the clicked points sit above the
surface the board is on (caliper jaw tips lying flat: about 1-2 mm).
"""
import argparse
import json

import cv2
import numpy as np

from .board import BoardSpec, Camera, make_board, pixels_to_board
from .measure_flat import match_calibration

VIEW_H = 1000
ZOOM_HALF = 60   # half size of the zoom crop, image pixels
ZOOM = 8


def click(window, img):
    pt = []
    cv2.imshow(window, img)
    cv2.setMouseCallback(window, lambda e, x, y, *_: pt.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    while not pt:
        if cv2.waitKey(30) & 0xFF == 27:
            raise SystemExit("cancelled")
    return pt[0]


def pick(image, label):
    s = VIEW_H / image.shape[0]
    view = cv2.resize(image, (int(image.shape[1] * s), VIEW_H))
    cv2.putText(view, f"click near {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    x, y = click("photo", view)
    cx, cy = int(x / s), int(y / s)
    x0, y0 = max(cx - ZOOM_HALF, 0), max(cy - ZOOM_HALF, 0)
    crop = image[y0:y0 + 2 * ZOOM_HALF, x0:x0 + 2 * ZOOM_HALF]
    big = cv2.resize(crop, (crop.shape[1] * ZOOM, crop.shape[0] * ZOOM), interpolation=cv2.INTER_NEAREST)
    zx, zy = click("zoom: click exactly", big)
    cv2.destroyWindow("zoom: click exactly")
    return x0 + (zx + 0.5) / ZOOM - 0.5, y0 + (zy + 0.5) / ZOOM - 0.5


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image")
    p.add_argument("--camera", required=True)
    p.add_argument("--board-spec", default=None)
    p.add_argument("--height-mm", type=float, default=1.5)
    a = p.parse_args()
    spec = BoardSpec.load(a.board_spec)
    cam = Camera.from_dict(json.load(open(a.camera)))
    image, _, pose, _ = match_calibration(cv2.imread(a.image), cam, make_board(spec))
    if cam.size != (image.shape[1], image.shape[0]):
        cam = cam.scaled_to((image.shape[1], image.shape[0]))
    cv2.namedWindow("photo")
    px = np.array([pick(image, "point 1"), pick(image, "point 2")])
    cv2.destroyAllWindows()
    pts = pixels_to_board(px, cam, pose, a.height_mm + spec.surface_offset_mm)
    print(f"board fit: {pose.n_corners} corners, {pose.reproj_px:.2f} px RMS")
    print(f"distance: {np.linalg.norm(pts[1] - pts[0]):.2f} mm")


if __name__ == "__main__":
    main()
