"""Write the printable ChArUco sheet (A4, print at 100% / "actual size").

    python -m gripscan.marker_sheet --out build/gen/out/marker_sheet

Writes <out>.png (300 dpi) and <out>.pdf. After printing, check the 100 mm
bar with a ruler: if it is off, the printer scaled the page and the
measurements will be off by the same factor.
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image

from .board import MARKER_MM, SQUARE_MM, SQUARES_X, SQUARES_Y, make_board

DPI = 300
A4_MM = (210.0, 297.0)


def mm(v):
    return int(round(v / 25.4 * DPI))


def render():
    board = make_board()
    bw, bh = SQUARES_X * SQUARE_MM, SQUARES_Y * SQUARE_MM
    board_img = board.generateImage((mm(bw), mm(bh)), marginSize=0, borderBits=1)
    page = np.full((mm(A4_MM[1]), mm(A4_MM[0])), 255, np.uint8)
    x0 = (page.shape[1] - board_img.shape[1]) // 2
    y0 = mm(8)
    page[y0:y0 + board_img.shape[0], x0:x0 + board_img.shape[1]] = board_img

    # 100 mm scale check bar with 10 mm ticks
    by = y0 + board_img.shape[0] + mm(7)
    bx = x0
    cv2.line(page, (bx, by), (bx + mm(100), by), 0, 3)
    for i in range(11):
        h = mm(3 if i % 5 == 0 else 1.5)
        cv2.line(page, (bx + mm(10 * i), by - h), (bx + mm(10 * i), by), 0, 3)
    label = (f"grip_scan ChArUco {SQUARES_X}x{SQUARES_Y}, square {SQUARE_MM:g} mm, "
             f"marker {MARKER_MM:g} mm. Print at 100%. Bar = 100 mm.")
    cv2.putText(page, label, (bx, by + mm(6)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2, cv2.LINE_AA)
    return page


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="build/gen/out/marker_sheet")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    page = render()
    cv2.imwrite(a.out + ".png", page)
    Image.fromarray(page).save(a.out + ".pdf", resolution=DPI)
    print("wrote", a.out + ".png", a.out + ".pdf")


if __name__ == "__main__":
    main()
