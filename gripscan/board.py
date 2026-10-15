"""The printed ChArUco board and the geometry around it.

Board frame: millimetres, origin at the board's top-left chessboard corner,
X right, Y down the sheet, Z = -(board normal) as OpenCV defines it. Points
above the paper (towards the camera) have negative Z in OpenCV's board frame;
`height` arguments below are positive-up and converted internally.
"""
from dataclasses import dataclass

import cv2
import numpy as np

SQUARES_X = 9
SQUARES_Y = 13
SQUARE_MM = 20.0
MARKER_MM = 15.0
DICTIONARY = cv2.aruco.DICT_5X5_100


def make_board():
    d = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    return cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_MM, MARKER_MM, d)


@dataclass
class Camera:
    K: np.ndarray       # 3x3 intrinsics, pixels
    dist: np.ndarray    # distortion coefficients
    size: tuple         # (width, height) the calibration was made at

    def to_dict(self):
        return {"K": self.K.tolist(), "dist": self.dist.ravel().tolist(), "size": list(self.size)}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["K"], float), np.array(d["dist"], float), tuple(d["size"]))

    def scaled_to(self, size):
        """Same lens at another image resolution (e.g. photo vs. video frame)."""
        sx, sy = size[0] / self.size[0], size[1] / self.size[1]
        K = self.K.copy()
        K[0] *= sx
        K[1] *= sy
        return Camera(K, self.dist, tuple(size))


def detect(image, board=None):
    """ChArUco corners in an image: (object points Nx3 mm, image points Nx2 px) or None."""
    board = board or make_board()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ids is None or len(ids) < 6:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    return obj.reshape(-1, 3).astype(np.float64), img.reshape(-1, 2).astype(np.float64)


@dataclass
class Pose:
    rvec: np.ndarray
    tvec: np.ndarray
    reproj_px: float     # RMS reprojection error of the board corners
    n_corners: int

    @property
    def R(self):
        return cv2.Rodrigues(self.rvec)[0]

    def camera_centre(self):
        """Camera position in board coordinates (mm)."""
        return (-self.R.T @ self.tvec).ravel()


def estimate_pose(obj, img, cam):
    ok, rvec, tvec = cv2.solvePnP(obj, img, cam.K, cam.dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("board pose could not be solved")
    proj, _ = cv2.projectPoints(obj, rvec, tvec, cam.K, cam.dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    return Pose(rvec, tvec, err, len(obj))


def pixels_to_board(px, cam, pose, height_mm=0.0):
    """Intersect camera rays through pixels with the plane `height_mm` above the paper.

    Returns Nx3 board coordinates in mm (Z positive up, towards the camera)."""
    px = np.asarray(px, float).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(px, cam.K, cam.dist).reshape(-1, 2)
    rays_cam = np.column_stack([norm, np.ones(len(norm))])
    R = pose.R
    rays = rays_cam @ R  # == (R.T @ ray).T, rays in board frame
    c = pose.camera_centre()
    z_plane = -height_mm  # OpenCV board frame: towards the camera is -Z
    t = (z_plane - c[2]) / rays[:, 2]
    pts = c + rays * t[:, None]
    pts[:, 2] = height_mm
    return pts
