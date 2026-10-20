"""Independent verification helpers (own code; only numpy/scipy/mathutils)."""
import bpy  # noqa: makes mathutils importable
import numpy as np
from scipy.spatial import cKDTree
from mathutils.bvhtree import BVHTree
from mathutils import Vector

def load_stage(path):
    d = np.load(path)
    return d['V'].astype(float), d['T'].astype(np.int64)

def bvh(V, T):
    return BVHTree.FromPolygons(V.tolist(), T.tolist(), all_triangles=True)

def face_normals(V, T):
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    return n  # area-weighted (2A)

def orient_outward(V, T):
    vol = np.einsum('ij,ij->i', V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])).sum() / 6
    return T if vol > 0 else T[:, ::-1].copy()

def vertex_normals(V, T):
    fn = face_normals(V, T)
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, T[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)

def vertex_area(V, T):
    a = 0.5 * np.linalg.norm(face_normals(V, T), axis=1) / 3
    A = np.zeros(len(V))
    for k in range(3):
        np.add.at(A, T[:, k], a)
    return A

DIRS = [Vector(d).normalized() for d in ((0.577, 0.571, 0.583), (-0.61, 0.33, 0.72), (0.23, -0.87, 0.43), (-0.37, -0.52, -0.77), (0.81, -0.21, -0.55))]

def parity_inside(tree, P):
    """Inside test by ray parity in 3 skew directions, majority vote. Own loop."""
    out = np.zeros(len(P), bool)
    for i, p in enumerate(P):
        votes = 0
        for d in DIRS:
            o = Vector(p); n = 0
            while True:
                hit = tree.ray_cast(o, d)
                if hit[0] is None:
                    break
                n += 1
                o = hit[0] + d * 2e-3
                if n > 200:
                    break
            votes += n & 1
        out[i] = votes >= 3
    return out

def nearest(tree, P):
    D = np.empty(len(P)); Q = np.empty((len(P), 3))
    for i, p in enumerate(P):
        q, _, _, d = tree.find_nearest(Vector(p))
        D[i] = d; Q[i] = q
    return D, Q

def signed_dist(tree, P):
    D, Q = nearest(tree, P)
    ins = parity_inside(tree, P)
    return np.where(ins, -D, D), Q

def seg_points(a, b, step=0.25):
    L = np.linalg.norm(b - a); n = max(2, int(np.ceil(L / step)) + 1)
    t = np.linspace(0, 1, n)
    return a[None] + t[:, None] * (b - a)[None], t

def shape_operator(V, N, idx, R, tree=None, min_dot=0.34, sigma=None):
    """Independent curvature: least-squares shape operator from NORMAL differences (dn = S dp) over a ball R,
    Gaussian weights (sigma = R/2), same-sheet neighbours only (n_i.n_j > min_dot). Returns k1 >= k2 (+ convex)."""
    tree = tree or cKDTree(V)
    sigma = sigma or R / 2.0
    k1 = np.full(len(idx), np.nan); k2 = np.full(len(idx), np.nan); d2 = np.zeros((len(idx), 3))
    nb = tree.query_ball_point(V[idx], R)
    for ii, (i, js) in enumerate(zip(idx, nb)):
        js = np.asarray(js)
        n = N[i]
        ok = N[js] @ n > min_dot
        js = js[ok]
        if len(js) < 8:
            continue
        a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(n, a); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
        dp = V[js] - V[i]; dn = N[js] - n
        w = np.exp(-(dp * dp).sum(1) / (2 * sigma * sigma))
        U = np.stack([dp @ e1, dp @ e2], 1); W = np.stack([dn @ e1, dn @ e2], 1)
        A = (U * w[:, None]).T @ U; B = (W * w[:, None]).T @ U
        try:
            S = B @ np.linalg.inv(A)
        except np.linalg.LinAlgError:
            continue
        S = 0.5 * (S + S.T)
        ev, evec = np.linalg.eigh(S)
        k1[ii], k2[ii] = ev[1], ev[0]
        d2[ii] = evec[0, 0] * e1 + evec[1, 0] * e2
    return k1, k2, d2

def smooth_field(V, N, f, idx, sigma, tree=None, min_dot=0.34, A=None, src=None, chunk=1500):
    """Gaussian average of f over same-sheet neighbours (area weighted), chunked; src = optional source subset."""
    src = np.arange(len(V)) if src is None else src
    A = np.ones(len(V)) if A is None else A
    st = cKDTree(V[src])
    out = np.full(len(idx), np.nan)
    for c0 in range(0, len(idx), chunk):
        ii = idx[c0:c0 + chunk]
        nb = st.query_ball_point(V[ii], 2.5 * sigma)
        for k, (i, js) in enumerate(zip(ii, nb)):
            js = src[np.asarray(js, dtype=np.int64)]
            js = js[N[js] @ N[i] > min_dot]
            d2 = ((V[js] - V[i]) ** 2).sum(1)
            w = np.exp(-d2 / (2 * sigma * sigma)) * A[js]
            out[c0 + k] = (w * f[js]).sum() / w.sum()
    return out
