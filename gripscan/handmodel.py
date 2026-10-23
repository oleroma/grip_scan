"""Articulated hand model built from the flat measurement, and its fit to 2D views.

The model is the solver's hand: segment lengths and palm layout come from
hand_flat.json; the pose is a small set of joint angles. Fitting this model to
MediaPipe's 2D joints across many board-posed views is much better constrained
than triangulating each joint on its own, because MediaPipe's joint placement
shifts with the viewpoint (several mm) while the bone lengths cannot.

Palm frame: origin at the wrist landmark, x along wrist -> middle MCP, z towards
the palm (palmar side), y = z cross x. Angles in radians internally, degrees out.
Flexion is positive towards the palm.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .landmarks import FINGERS, NAMES

LONG = ["index", "middle", "ring", "pinky"]
PALM_IDS = [0, 1, 5, 9, 13, 17]
PARAMS = (["rx", "ry", "rz", "tx", "ty", "tz", "palm_arch"]
          + [f"{f}_{j}" for f in LONG for j in ("spread", "mcp", "pip", "dip")]
          + ["thumb_cmc_abd", "thumb_cmc_flex", "thumb_twist", "thumb_mcp", "thumb_ip"]
          + ["wrist_dx", "wrist_dy", "wrist_dz", "cmc_dx", "cmc_dy", "cmc_dz"])
# MediaPipe's wrist and thumb-CMC points are the least anatomical: the wrist point
# sits on the forearm side and moves with wrist bend, the CMC point is guessed inside
# the palm. Both get a free offset (mm) from the flat-photo template.
OFFSETS = PARAMS[-6:]
ANGLE_PARAMS = [n for n in PARAMS[6:] if n not in OFFSETS]


def rot(axis, angle):
    return Rotation.from_rotvec(np.asarray(axis, float) / np.linalg.norm(axis) * angle).as_matrix()


class HandModel:
    def __init__(self, flat_landmarks_mm):
        P = np.array([flat_landmarks_mm[n][:2] + [0.0] for n in NAMES])  # flat on the board
        x = P[9] - P[0]
        x /= np.linalg.norm(x)
        z = np.array([0.0, 0.0, 1.0])  # palm down on the board: palmar side is +Z (into the board)
        y = np.cross(z, x)
        B = np.vstack([x, y, z])
        self.tpl = (P - P[0]) @ B.T  # template points in palm frame
        self.len = {f: [float(np.linalg.norm(self.tpl[ids[i + 1]] - self.tpl[ids[i]])) for i in range(3)]
                    for f, ids in FINGERS.items()}
        self.dir0 = {f: self._unit(self.tpl[ids[1]] - self.tpl[ids[0]]) for f, ids in FINGERS.items()}

    @staticmethod
    def _unit(v):
        return v / np.linalg.norm(v)

    def local(self, q):
        """21x3 joint positions in the palm frame for angle parameters q (dict)."""
        X = np.zeros((21, 3))
        ez = np.array([0.0, 0.0, 1.0])
        ex = np.array([1.0, 0.0, 0.0])
        X[PALM_IDS] = self.tpl[PALM_IDS]
        X[0] = self.tpl[0] + [q["wrist_dx"], q["wrist_dy"], q["wrist_dz"]]
        cmc = self.tpl[1] + [q["cmc_dx"], q["cmc_dy"], q["cmc_dz"]]
        arch = {"ring": 0.6 * q["palm_arch"], "pinky": q["palm_arch"]}
        for f in LONG:
            a, b, c, d = FINGERS[f]
            R_arch = rot(ex, arch.get(f, 0.0))      # ring/pinky metacarpals fold towards the palm
            base = R_arch @ self.tpl[a]
            zf = R_arch @ ez
            dirf = R_arch @ (rot(ez, q[f"{f}_spread"]) @ self.dir0[f])
            k = np.cross(dirf, zf)                  # flexion axis: rotates dirf towards the palm
            X[a] = base
            ang = 0.0
            for i, (j, L) in enumerate(zip((b, c, d), self.len[f])):
                ang += q[f"{f}_{('mcp', 'pip', 'dip')[i]}"]
                X[j] = X[[a, b, c][i]] + L * (rot(k, ang) @ dirf)
        # thumb
        a, b, c, d = FINGERS["thumb"]
        t = rot(ez, q["thumb_cmc_abd"]) @ self.dir0["thumb"]
        t = rot(np.cross(t, ez), q["thumb_cmc_flex"]) @ t
        k = rot(t, q["thumb_twist"]) @ self._unit(np.cross(t, ez))
        X[a] = cmc
        L = self.len["thumb"]
        X[b] = X[a] + L[0] * t
        X[c] = X[b] + L[1] * (rot(k, q["thumb_mcp"]) @ t)
        X[d] = X[c] + L[2] * (rot(k, q["thumb_mcp"] + q["thumb_ip"]) @ t)
        return X

    def world(self, v):
        q = dict(zip(PARAMS, v))
        R = Rotation.from_rotvec(v[:3]).as_matrix()
        return self.local(q) @ R.T + v[3:6]


def init_root(model, palm_pts_board):
    """Rigid transform (rotvec, t) taking the palm template onto triangulated palm points."""
    A = model.tpl[PALM_IDS]
    Bp = palm_pts_board
    ca, cb = A.mean(0), Bp.mean(0)
    H = (A - ca).T @ (Bp - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return Rotation.from_matrix(R).as_rotvec(), cb - R @ ca


def fit(model, Ps, xs, f_px, palm_init, sway=True):
    """Fit model pose to normalized 2D joints xs (list of 21x2) seen through Ps (list of 3x4).

    Robust (soft L1, ~8 px scale) so views where MediaPipe guessed a hidden joint
    badly do not dominate. With `sway`, every frame also gets a small translation
    of the whole hand (a floating hand drifts a few mm while the phone moves),
    kept small and smooth by priors. Returns (params, per-frame/per-joint
    residuals in px, 21x3 joints in board mm, per-frame sway in mm)."""
    rv, t = init_root(model, palm_init)
    n_glob = len(PARAMS)
    N = len(Ps)
    v0 = np.zeros(n_glob + (3 * N if sway else 0))
    v0[:3], v0[3:6] = rv, t
    for f in LONG:
        v0[PARAMS.index(f"{f}_mcp")] = np.radians(30)
        v0[PARAMS.index(f"{f}_pip")] = np.radians(40)
        v0[PARAMS.index(f"{f}_dip")] = np.radians(20)
    v0[PARAMS.index("thumb_mcp")] = np.radians(10)
    Ps = np.array(Ps)
    obs = np.array(xs)  # N x 21 x 2
    # weak priors: spreads, arch and thumb twist stay near neutral unless the views insist
    prior_idx = [PARAMS.index(n) for n in PARAMS if n.endswith("_spread") or n in ("palm_arch", "thumb_twist")]
    w_prior = 2.0 / np.radians(10)  # ~2 px of residual per 10 degrees
    off_idx = [PARAMS.index(n) for n in OFFSETS]
    w_off = 1.0                     # ~1 px per mm of wrist/CMC offset
    w_sway = 0.5                    # ~1 px per 2 mm of sway
    w_smooth = 2.0                  # ~2 px per mm of frame-to-frame jump

    def project(v):
        X = model.world(v[:n_glob])
        T = v[n_glob:].reshape(N, 3) if sway else np.zeros((N, 3))
        Xn = X[None] + T[:, None, :]                     # N x 21 x 3
        Xh = np.concatenate([Xn, np.ones((N, 21, 1))], axis=2)
        proj = np.einsum("nij,nkj->nki", Ps, Xh)
        return proj[..., :2] / proj[..., 2:3], X, T

    def resid(v):
        uv, _, T = project(v)
        parts = [((uv - obs) * f_px).ravel(), w_prior * v[prior_idx], w_off * v[off_idx]]
        if sway:
            parts += [w_sway * T.ravel(), w_smooth * np.diff(T, axis=0).ravel()]
        return np.concatenate(parts)

    sparsity = None
    if sway:
        from scipy.sparse import lil_matrix
        n_res = len(resid(v0))
        S = lil_matrix((n_res, len(v0)), dtype=int)
        S[: N * 42, :n_glob] = 1
        for n in range(N):
            S[n * 42:(n + 1) * 42, n_glob + 3 * n:n_glob + 3 * n + 3] = 1
        r0 = N * 42
        S[r0:r0 + len(prior_idx) + len(off_idx), :n_glob] = 1
        r0 += len(prior_idx) + len(off_idx)
        for i in range(3 * N):
            S[r0 + i, n_glob + i] = 1
        r0 += 3 * N
        for n in range(N - 1):
            for k in range(3):
                S[r0 + 3 * n + k, n_glob + 3 * n + k] = 1
                S[r0 + 3 * n + k, n_glob + 3 * (n + 1) + k] = 1
        sparsity = S
    sol = least_squares(resid, v0, loss="soft_l1", f_scale=8.0, max_nfev=300,
                        jac_sparsity=sparsity, x_scale="jac")
    uv, X, T = project(sol.x)
    err = np.linalg.norm(uv - obs, axis=2) * f_px  # N x 21
    return sol.x[:n_glob], err, X, T


def degrees(v):
    """Angle parameters in degrees, wrapped to (-180, 180]."""
    return {n: round(float((np.degrees(v[PARAMS.index(n)]) + 180) % 360 - 180), 1) for n in ANGLE_PARAMS}


def offsets(v):
    return {n: round(float(v[PARAMS.index(n)]), 1) for n in OFFSETS}
