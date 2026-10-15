"""Show the ChArUco board on a monitor at exact pixel size (instead of printing).

    python -m gripscan.screen_board                  # show it, using the spec pitch
    python -m gripscan.screen_board --bar-mm 298.6   # after measuring the bar with a ruler

The board is drawn 1:1 in screen pixels, so its size in mm is fixed by the
monitor's pixel pitch. Measure the long bar under the board with a ruler and
pass the result as --bar-mm: that sets the pitch from your own measurement.
Writes the board spec (data/board_screen.json) that calibrate and
measure_flat read with --board-spec. Press Esc or Q to close.
"""
import argparse
import ctypes
import math
import os
import sys

import cv2
import numpy as np

from .board import DICTIONARY, BoardSpec

# ASUS VG27AQL5A: 27" 16:9, 2560x1440
DEFAULT_DIAGONAL_IN = 27.0
SQUARES_X, SQUARES_Y = 13, 9          # landscape
SQUARE_PX, MARKER_PX = 128, 96
BAR_PX = 1280                         # 10 squares
GLASS_MM = 1.0                        # pattern lies ~1 mm under the screen surface


def screen_size():
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # real pixels, not 125%-scaled ones
        except OSError:
            ctypes.windll.user32.SetProcessDPIAware()
        u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    raise SystemExit("screen mode is only implemented for Windows")


def pitch_from_diagonal(w, h, diagonal_in):
    return diagonal_in * 25.4 / math.hypot(w, h)


def render(w, h, pitch):
    img = np.full((h, w), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_PX, MARKER_PX, d)
    bw, bh = SQUARES_X * SQUARE_PX, SQUARES_Y * SQUARE_PX
    board_img = board.generateImage((bw, bh), marginSize=0, borderBits=1)
    x0, y0 = (w - bw) // 2, 40
    img[y0:y0 + bh, x0:x0 + bw] = board_img
    by = y0 + bh + 70
    cv2.rectangle(img, (x0, by - 3), (x0 + BAR_PX, by + 3), 0, -1)
    for i in range(BAR_PX // SQUARE_PX + 1):
        cv2.line(img, (x0 + i * SQUARE_PX, by - 25), (x0 + i * SQUARE_PX, by), 0, 3)
    txt = (f"Measure this bar end to end with a ruler (expected about {BAR_PX * pitch:.1f} mm), "
           f"then run with --bar-mm <value>.   pitch {pitch:.5f} mm/px   Esc/Q closes")
    cv2.putText(img, txt, (x0, by + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagonal-in", type=float, default=DEFAULT_DIAGONAL_IN)
    p.add_argument("--bar-mm", type=float, default=None, help="ruler length of the bar")
    p.add_argument("--glass-mm", type=float, default=GLASS_MM)
    p.add_argument("--out-spec", default="data/board_screen.json")
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()

    w, h = screen_size()
    pitch = a.bar_mm / BAR_PX if a.bar_mm else pitch_from_diagonal(w, h, a.diagonal_in)
    spec = BoardSpec(SQUARES_X, SQUARES_Y, SQUARE_PX * pitch, MARKER_PX * pitch, DICTIONARY,
                     a.glass_mm, f"screen {w}x{h}, pitch {pitch:.5f} mm/px "
                     f"({'ruler ' + str(a.bar_mm) + ' mm' if a.bar_mm else 'from diagonal, NOT ruler-checked'})")
    os.makedirs(os.path.dirname(a.out_spec) or ".", exist_ok=True)
    spec.save(a.out_spec)
    print(f"screen {w}x{h}, pitch {pitch:.5f} mm/px, square {spec.square_mm:.3f} mm -> {a.out_spec}")
    if not a.bar_mm:
        print("note: pitch is from the nominal diagonal; measure the bar and rerun with --bar-mm")
    if a.no_show:
        return

    img = render(w, h, pitch)
    name = "grip_scan board"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(name, img)
    cv2.waitKey(200)
    rect = cv2.getWindowImageRect(name)
    if (rect[2], rect[3]) != (w, h):
        print(f"WARNING: window is {rect[2]}x{rect[3]}, not {w}x{h}; the board is being scaled")
    while True:
        k = cv2.waitKey(100) & 0xFF
        if k in (27, ord("q")) or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
