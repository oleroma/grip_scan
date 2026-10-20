"""Linearised curvature operator of a moved mesh P = V + h D (the acceptance's quadric fit, without the Huber
re-weighting): for each vertex i of a set, the local height quadric's Hessian [[2a, b], [b, 2c]] over a ball of
`radius` mm is linear in h. Returns rows of directional second derivatives e' H e for a set of directions."""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def vnormals(P, T):
    cr = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    N = np.zeros_like(P)
    for k in range(3):
        np.add.at(N, T[:, k], cr)
    return N / np.maximum(np.linalg.norm(N, axis=1), 1e-15)[:, None]


def frames(n):
    a = np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(n, e1)


def quadric_operator(P, T, idx, D, radius=3.0, K=None, chunk=3000, N=None, tree=None):
    """For each vertex idx[k]: G_k (3 x nb) mapping neighbour heights to (2a, b, 2c) and the neighbour indices.
    Heights are measured along the fitted normal n0 at the current P; a change dh_j of h moves the height of
    neighbour j by dh_j * (D_j . n0). Returns (rows (3 per vertex) as a sparse matrix over all vertices,
    current Hessians (n,3) = (Huu, Huv, Hvv), frames e1, e2, n0)."""
    if N is None:
        N = vnormals(P, T)
    if tree is None:
        tree = cKDTree(P)
    if K is None:
        A = np.linalg.norm(np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]), axis=1) / 6.0
        m = np.zeros(len(P))
        for k in range(3):
            np.add.at(m, T[:, k], A)
        apv = float(np.median(m[np.unique(T)]))
        K = int(min(400, max(40, 1.3 * np.pi * radius ** 2 / max(apv, 1e-6))))
    sig = radius / 2.0
    n_out = len(idx)
    Hc = np.zeros((n_out, 3)); E1 = np.zeros((n_out, 3)); E2 = np.zeros((n_out, 3)); N0 = np.zeros((n_out, 3))
    rows, cols, vals = [], [], []
    for s in range(0, n_out, chunk):
        ii = idx[s:s + chunk]
        dd, jj = tree.query(P[ii], k=min(K, len(P)), distance_upper_bound=radius)
        valid = np.isfinite(dd)
        jj = np.where(valid, jj, 0)
        dd = np.where(valid, dd, radius)
        Dl = P[jj] - P[ii][:, None, :]
        w0 = np.exp(-0.5 * (dd / sig) ** 2) * valid
        Nj = N[jj]
        same = (np.einsum('nkc,nc->nk', Nj, N[ii]) > 0.0)
        n0 = (w0[..., None] * Nj * same[..., None]).sum(1)
        n0 /= np.maximum(np.linalg.norm(n0, axis=1, keepdims=True), 1e-12)
        for rnd in range(2):
            compat = np.clip(((Nj @ n0[..., None])[..., 0] - 0.3) / 0.3, 0.0, 1.0)
            ww = w0 * compat
            e1, e2 = frames(n0)
            u = (Dl @ e1[..., None])[..., 0]
            v = (Dl @ e2[..., None])[..., 0]
            hgt = (Dl @ n0[..., None])[..., 0]
            A = np.empty(u.shape + (6,))
            A[..., 0] = u * u; A[..., 1] = u * v; A[..., 2] = v * v; A[..., 3] = u; A[..., 4] = v; A[..., 5] = 1.0
            AW = (A * ww[..., None]).transpose(0, 2, 1)
            Mx = AW @ A + np.eye(6) * 1e-10
            Gm = np.linalg.solve(Mx, AW)                    # (n, 6, k)
            c = (Gm @ hgt[..., None])[..., 0]
            gu, gv = c[:, 3], c[:, 4]
            nf = n0 - gu[:, None] * e1 - gv[:, None] * e2
            nf /= np.linalg.norm(nf, axis=1, keepdims=True)
            if rnd == 0:
                n0 = nf
        # Hessian of the height in the frame (e1, e2, n0) of the last fit
        Hc[s:s + chunk] = np.c_[2 * c[:, 0], c[:, 1], 2 * c[:, 2]]
        E1[s:s + chunk], E2[s:s + chunk], N0[s:s + chunk] = e1, e2, n0
        # linear map: d(height_j) = dh_j * (D_j . n0)
        dn = np.einsum('nkc,nc->nk', D[jj], n0) * (ww > 1e-8)
        for r, sc in ((0, 2.0), (1, 1.0), (2, 2.0)):
            g = sc * Gm[:, r, :] * dn                           # (n, k)
            rr = (np.arange(len(ii)) + s) * 3 + r
            rows.append(np.repeat(rr, g.shape[1])); cols.append(jj.ravel()); vals.append(g.ravel())
    R = np.concatenate(rows); C = np.concatenate(cols); Vv = np.concatenate(vals)
    keep = Vv != 0
    Gs = sp.coo_matrix((Vv[keep], (R[keep], C[keep])), shape=(3 * n_out, len(P))).tocsr()
    return Gs, Hc, E1, E2, N0, K


def principal(Hc):
    """Eigen-decomposition of [[a, b], [b, c]]: lam_max, lam_min, angle of lam_max's direction."""
    a, b, c = Hc[:, 0], Hc[:, 1], Hc[:, 2]
    tr = (a + c) / 2
    disc = np.sqrt(((a - c) / 2) ** 2 + b * b)
    th = 0.5 * np.arctan2(2 * b, a - c)
    return tr + disc, tr - disc, th
