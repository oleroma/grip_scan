"""Synthetic check of the sheet -> pose -> back-projection chain.

    .venv/Scripts/python -m tests.test_geometry

Renders the printed sheet into a virtual phone camera with a known pose,
detects the board, and checks that points above the paper come back at
the right positions and distances.
"""
import cv2
import numpy as np

from gripscan.board import BoardSpec, Camera, detect, estimate_pose, make_board, pixels_to_board
from gripscan.marker_sheet import DPI, mm, render

PX_PER_MM = DPI / 25.4


def printed_sheet():
    from gripscan.board import SQUARES_X, SQUARE_MM
    page = render()
    x0 = (page.shape[1] - mm(SQUARES_X * SQUARE_MM)) // 2
    # board mm -> page px
    return page, np.array([[PX_PER_MM, 0, x0], [0, PX_PER_MM, mm(8)], [0, 0, 1]]), BoardSpec()


def screen_board(pitch=0.23310):
    from gripscan import screen_board as sb
    w, h = 2560, 1440
    img = sb.render(w, h, pitch)
    x0 = (w - sb.SQUARES_X * sb.SQUARE_PX) // 2
    spec = BoardSpec(sb.SQUARES_X, sb.SQUARES_Y, sb.SQUARE_PX * pitch, sb.MARKER_PX * pitch)
    return img, np.array([[1 / pitch, 0, x0], [0, 1 / pitch, 40], [0, 0, 1]]), spec


def synthetic_view(rvec, tvec, K, size, page, A):
    R = cv2.Rodrigues(rvec)[0]
    H = K @ np.column_stack([R[:, 0], R[:, 1], tvec.ravel()])  # board mm (z=0) -> image px
    M = H @ np.linalg.inv(A)
    img = cv2.warpPerspective(page, M, size, flags=cv2.INTER_AREA, borderValue=200)
    noise = np.random.default_rng(0).normal(0, 2, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def check(name, page, A, spec):
    size = (3000, 4000)
    K = np.array([[3100.0, 0, 1500], [0, 3100.0, 2000], [0, 0, 1]])
    cam = Camera(K, np.zeros(5), size)
    # camera ~450 mm above the sheet centre, tilted ~12 degrees
    rvec = np.array([0.18, -0.12, 0.05])
    R = cv2.Rodrigues(rvec)[0]
    centre_board = np.array([90.0, 130.0, -450.0])  # camera centre in board frame
    tvec = (-R @ centre_board).reshape(3, 1)

    img = synthetic_view(rvec, tvec, K, size, page, A)
    d = detect(img, make_board(spec))
    assert d is not None, "board not detected"
    pose = estimate_pose(*d, cam)
    c = pose.camera_centre()
    print(f"[{name}] corners {pose.n_corners}, reproj {pose.reproj_px:.3f} px, camera centre {np.round(c, 1)}")
    assert np.linalg.norm(c - centre_board) < 3.0, "camera position off by more than 3 mm"

    # two "joints" 10 mm above the paper, 45 mm apart, near the sheet edge (worst parallax)
    h = 10.0
    P = np.array([[20.0, 30.0, -h], [20.0 + 45.0, 30.0, -h]])  # OpenCV board frame: up = -Z
    px, _ = cv2.projectPoints(P, rvec, tvec, K, np.zeros(5))
    px = px.reshape(-1, 2)
    back = pixels_to_board(px, cam, pose, h)
    flat = pixels_to_board(px, cam, pose, 0.0)
    err = np.linalg.norm(back[:, :2] - P[:, :2], axis=1).max()
    L = np.linalg.norm(back[1] - back[0])
    L_flat = np.linalg.norm(flat[1] - flat[0])
    print(f"position error {err:.2f} mm, length {L:.2f} mm (true 45), ignoring height {L_flat:.2f} mm")
    assert err < 0.5, "back-projected position off by more than 0.5 mm"
    assert abs(L - 45.0) < 0.3, "segment length off by more than 0.3 mm"
    print(f"[{name}] OK")


def main():
    check("printed sheet", *printed_sheet())
    check("screen board", *screen_board())


if __name__ == "__main__":
    main()
