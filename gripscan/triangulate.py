"""Multi-view triangulation of 2D points seen from known camera poses."""
import cv2
import numpy as np


def projection(cam, pose):
    """3x4 matrix mapping board-frame points to normalized (undistorted) image coords."""
    return np.hstack([pose.R, pose.tvec.reshape(3, 1)])


def normalize(px, cam):
    return cv2.undistortPoints(np.asarray(px, float).reshape(-1, 1, 2), cam.K, cam.dist).reshape(-1, 2)


def dlt(Ps, xs):
    """Linear triangulation of one point from N views (normalized coords)."""
    A = []
    for P, (x, y) in zip(Ps, xs):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.array(A))
    X = vt[-1]
    return X[:3] / X[3]


def reproj_px(X, P, x, f):
    p = P @ np.append(X, 1.0)
    return float(np.linalg.norm(p[:2] / p[2] - x)) * f


def robust_point(Ps, xs, f, thresh_px=12.0, iters=3):
    """Triangulate, then drop views whose reprojection error is above threshold
    and re-solve. Returns (X, inlier mask, RMS px over inliers)."""
    keep = np.ones(len(Ps), bool)
    X = dlt(Ps, xs)
    for _ in range(iters):
        err = np.array([reproj_px(X, P, x, f) for P, x in zip(Ps, xs)])
        # threshold relative to the median, so one bad cluster cannot dominate
        new = err < max(thresh_px, 2.5 * np.median(err))
        if new.sum() < 2:
            break
        keep = new
        X = dlt([P for P, k in zip(Ps, keep) if k], [x for x, k in zip(xs, keep) if k])
    err = np.array([reproj_px(X, P, x, f) for P, x, k in zip(Ps, xs, keep) if k])
    return X, keep, float(np.sqrt(np.mean(err ** 2)))


def baseline_deg(poses, keep):
    """Largest angle between viewing directions used (triangulation strength)."""
    dirs = [p.R[2] for p, k in zip(poses, keep) if k]  # camera z axis in board frame
    best = 0.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            best = max(best, float(np.degrees(np.arccos(np.clip(dirs[i] @ dirs[j], -1, 1)))))
    return best
