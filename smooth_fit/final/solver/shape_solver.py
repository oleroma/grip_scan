"""shape_solver.py: the smooth hand-fit shape of GripV as ONE signed normal offset h of the stage input mesh.

Solved offline (numpy + scipy.sparse), stored on a mesh object 'shape offset' and applied in the stage by
mod_shape.py (offset -> moved mesh -> SDF round trip -> thenar module). See README_shape.txt.

Unknown: h (mm) per vertex of the right handle of the stage input mesh (subdivided GripV basemesh, level 'high');
the new skin is  p + h * n  (n = the vertex normal of the input mesh). The left handle is the mirror image.

    minimise   Bend * |Laplacian h|^2  (thin plate / bi-Laplacian, cotangent Laplacian, lumped mass)
             + Membrane * |grad h|^2 + Tension * |h|^2               (small: keeps h local)
             + Contact Weight * sum over the contact-zone axis samples (gap - (-Press))^2   (soft, robust)
    subject to h = 0 on the frozen region (+ a margin)      -> zero value AND zero slope at its edge (C1)
               skin on the R2 / R4 seat plane over the opening + rim  (hard: Dirichlet values)
               no hand capsule deeper than Max Dig in the skin   (hard: per-vertex upper bounds)
               y <= Max Y - band, z <= Frozen Z, bbox min y / z kept, cutter wall, keep-outs   (bounds)

Non-linear parts (nearest skin points, capsule distances, seat projections) are linearised at the current
skin and the QP is solved again (outer iterations). The QP with bounds is a primal active-set method on
sparse direct solves (scipy.sparse.linalg.splu).
"""
import time
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.spatial import cKDTree

try:
    import bpy  # noqa: F401  (the bpy wheel provides mathutils)
except Exception:  # pragma: no cover
    pass
from mathutils import Vector
from mathutils.bvhtree import BVHTree

CONTACT_ZONES = ('middle root pad', 'middle PIP', 'ring root pad', 'ring PIP', 'pinky root pad', 'pinky PIP',
                 'pinky distal', 'palm on swell')

# ------------------------------------------------------------------------------------------- parameters
# The six inputs Olex is meant to touch are marked (OLEX). Everything else is internal tuning.
DEFAULTS = dict(
    press=0.5,                 # (OLEX 'Press', general stage input) target sink of the rigid finger at each contact (mm)
    max_dig=0.8,               # (OLEX) no hand capsule deeper than this anywhere (acceptance limit 1.0; margin for the remesh)
    smoothness=1.0,            # (OLEX) bending stiffness multiplier: higher = smoother, wider, but contacts less exact
    contact_weight=1.0,        # (OLEX) how hard the contacts pull (multiplier)
    contact_reach=3.0,         # (OLEX) within a zone, axis samples up to this much farther than the zone's closest
                               #        sample are pulled at full weight (more = more of each finger supported)
    seat_rim=3.0,              # (OLEX) flush seat rim around each trigger opening (acceptance rim 2.5 mm + 0.5 margin)
    # internal
    bend=1.0, membrane=0.002, tension=2e-4,
    contact_scale=40.0,        # base weight per mm of zone axis
    zone_scale=None,           # per-zone multipliers {zone: w}
    zone_reach=None,           # per-zone contact_reach {zone: mm}
    contact_mode='hug',        # 'hug': every zone sample within contact_reach pulled to -Press; 'deep': see contact_rows
    w_deep=1.0, w_support=0.1,
    keep_features=0.0,         # soft weight keeping the generator's own features (creases, sharp edges) unchanged
    feature_hard=0.0,          # > 0: |h| <= this on the features (hard; the acceptance then excludes them as unchanged)
    feature_seat_clear=6.0,    # features this close to a seat zone are not kept (the seat reshapes them)
    feature_dilate=2.5,        # features dilated by this (mm) for the keep weight
    seat_feature_soft=True,    # seat targets on a feature are soft (seat_soft_far) instead of hard
    fair_curv=0.0,             # phase 2: fairing weight grows where the current skin is tighter than the limits
    fair_curv_concave=14.0, fair_curv_convex=3.5,   # (radii, mm) with a margin over the acceptance (12 / 3)
    fair_curv_max=30.0,
    fair=0.0,                  # fairing of the FINAL surface (mean curvature^2) inside the edited region (phase 2)
    fair_after=4,              # outer iterations of phase 1 (thin plate on h only) before the fairing mask is made
    fair_edit=0.15, fair_dilate=4.0, fair_fade=3.0,
    seat_mode='graded',        # 'graded' (soft rim, weight falling with the needed move) or 'hard' (Dirichlet rim)
    seat_w=1.0, seat_move_scale=2.0,
    seat_hard_max=99.0,        # a seat vertex that must move more than this is a soft target (seat_soft_far)
    seat_soft_far=0.05,
    frozen_margin=1.0,         # h = 0 also within this distance of the frozen region (mm)
    frozen_ring=0,             # and on this many edge rings around it (1: no interpolated leak into the frozen region)
    edit_min_x=25.0, frozen_z=38.0, max_y=-4.158, front_band=0.5,
    entry_margin=0.15,         # a moved vertex must stay this far out of the frozen region
    bbox_keep_band=0.4,        # vertices this close to bbox min y / min z (and max x when keep_bbox_x) stay fixed
    keep_bbox_x=False,
    cutter_wall=1.2, keepout_wall=1.5,
    seat_proud=0.2, seat_cos_min=0.25, seat_window=10.0,
    seat_core=4.0,             # hard flush disc around each press-face centre (mm)
    seat_corridor=10.0,        # rim / opening parts closer than this to the other trigger's opening: soft
    seat_soft=0.0,             # weight (per mm2) of the soft seat targets in the corridor (0 = none)
    dig_consider=4.0,          # capsule-vertex pairs considered when within this (mm)
    dir_smooth=3.0,            # offset direction = normal smoothed over this (mm)
    outer_iters=14, tol=0.01, qp_iters=40, damping=0.6, damp_from=3,
    verbose=True,
)


def log(opt, *a):
    if opt.get('verbose', True):
        print('[shape]', *a, flush=True)


# ------------------------------------------------------------------------------------------ mesh basics
def orient(V, T):
    """Faces oriented outward (positive signed volume)."""
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    vol = np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0
    return (T[:, ::-1].copy() if vol < 0 else T.copy())


def face_normals(V, T):
    cr = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    A2 = np.linalg.norm(cr, axis=1)
    return cr / np.maximum(A2, 1e-15)[:, None], 0.5 * A2


def vertex_normals(V, T):
    cr = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, T[:, k], cr)
    return N / np.maximum(np.linalg.norm(N, axis=1), 1e-15)[:, None]


def smooth_directions(V, T, N, sigma, sheet_cos=0.0):
    """Unit vector field: vertex normals averaged with Gaussian weights (sigma mm, Euclidean, same sheet only)."""
    if sigma <= 0:
        return N.copy()
    _, A = face_normals(V, T)
    m = np.zeros(len(V))
    for k in range(3):
        np.add.at(m, T[:, k], A / 3.0)
    tree = cKDTree(V)
    D = np.zeros_like(N)
    nb = tree.query_ball_point(V, 2.5 * sigma, workers=2)
    for i, lst in enumerate(nb):
        lst = np.asarray(lst)
        w = m[lst] * np.exp(-((V[lst] - V[i]) ** 2).sum(1) / (2 * sigma * sigma))
        w = w * ((N[lst] @ N[i]) > sheet_cos)
        D[i] = (w[:, None] * N[lst]).sum(0)
    return D / np.maximum(np.linalg.norm(D, axis=1), 1e-12)[:, None]


def cotan_laplacian(V, T):
    """L (positive semi-definite, L = D - W with cotangent weights) and lumped mass (barycentric areas)."""
    n = len(V)
    i0, i1, i2 = T[:, 0], T[:, 1], T[:, 2]

    def cot(a, b, c):
        u, v = V[b] - V[a], V[c] - V[a]
        return np.einsum('ij,ij->i', u, v) / np.maximum(np.linalg.norm(np.cross(u, v), axis=1), 1e-12)
    c0, c1, c2 = cot(i0, i1, i2), cot(i1, i2, i0), cot(i2, i0, i1)
    I = np.concatenate([i1, i2, i2, i0, i0, i1])
    J = np.concatenate([i2, i1, i0, i2, i1, i0])
    W = 0.5 * np.concatenate([c0, c0, c1, c1, c2, c2])
    Wm = sp.coo_matrix((W, (I, J)), shape=(n, n)).tocsr()
    L = sp.diags(np.asarray(Wm.sum(1)).ravel()) - Wm
    _, A = face_normals(V, T)
    m = np.zeros(n)
    for k in range(3):
        np.add.at(m, T[:, k], A / 3.0)
    return L.tocsr(), np.maximum(m, 1e-9)


def bvh(V, T):
    return BVHTree.FromPolygons([tuple(p) for p in V], [tuple(int(i) for i in t) for t in T], all_triangles=True)


DIRS = [np.array(d) / np.linalg.norm(d) for d in ((0.31, 0.53, 0.79), (-0.67, 0.21, -0.71), (0.12, -0.91, 0.40))]


def inside(tree, p, maxhits=400):
    """Ray parity, majority of 3 directions (as gripmetrics.inside)."""
    votes = 0
    for d in DIRS:
        cnt, o, dv = 0, Vector(p), Vector(d)
        for _ in range(maxhits):
            hit = tree.ray_cast(o, dv)
            if hit[0] is None:
                break
            cnt += 1
            o = hit[0] + dv * 1e-5
        votes += cnt % 2
    return votes >= 2


def barycentric(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-18:
        return np.array([1.0, 0.0, 0.0])
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    bc = np.clip(np.array([1 - v - w, v, w]), 0, 1)
    return bc / bc.sum()


def seg_closest(P, a, b):
    """Closest points on segment a-b for points P (k,3)."""
    u = b - a
    t = np.clip(((P - a) @ u) / max(u @ u, 1e-12), 0.0, 1.0)
    return a + t[:, None] * u


def dist_to_segments(P, S):
    a, b = S[:, 0], S[:, 1]
    ab = b - a
    L2 = np.maximum((ab * ab).sum(1), 1e-12)
    out = np.full(len(P), np.inf)
    for s in range(0, len(P), 2000):
        q = P[s:s + 2000, None, :] - a[None]
        t = np.clip((q * ab[None]).sum(2) / L2[None], 0, 1)
        d = np.linalg.norm(q - t[..., None] * ab[None], axis=2)
        out[s:s + 2000] = d.min(1)
    return out


def inside_polygon_2d(Q, S2):
    """Crossing-number test of 2-D points Q (k,2) against a set of 2-D segments S2 (m,2,2) (closed loops)."""
    x, y = Q[:, 0:1], Q[:, 1:2]
    x1, y1, x2, y2 = S2[None, :, 0, 0], S2[None, :, 0, 1], S2[None, :, 1, 0], S2[None, :, 1, 1]
    cond = (y1 > y) != (y2 > y)
    xin = x1 + (y - y1) * (x2 - x1) / np.where(np.abs(y2 - y1) < 1e-15, 1e-15, y2 - y1)
    return ((cond & (x < xin)).sum(1) % 2) == 1


# ------------------------------------------------------------------------------------------- the problem
class Problem:
    """Everything the solve needs, as plain arrays.

    V, T      stage input mesh (subdivided GripV basemesh), world = object space
    caps      {bone: (a, b, r)} the 20 rigid hand capsules ('hand points')
    zones     {zone: [(a, b, r), ...]} contact-zone axis segments (gripmetrics.zones(caps))
    seats     {'R2': dict(center, normal, segments (opening outline in the seat plane), cutter=(V, T))}
    keepouts  {name: (V, T, required_wall_mm)} internal objects close to the handle
    """

    def __init__(self, V, T, caps, zones, seats, keepouts=None, opt=None):
        self.opt = dict(DEFAULTS)
        self.opt.update(opt or {})
        o = self.opt
        t0 = time.time()
        self.V = np.asarray(V, float)
        self.T = orient(self.V, np.asarray(T, np.int64))
        self.N = vertex_normals(self.V, self.T)
        # offset direction: the normal smoothed over dir_smooth mm (so a strong recession of a tight convex
        # edge translates the edge instead of collapsing it into a cusp)
        self.D = smooth_directions(self.V, self.T, self.N, self.opt['dir_smooth'])
        self.caps = caps
        self.zones = zones
        self.seats = seats
        self.keepouts = keepouts or {}
        V, N = self.V, self.N
        n = len(V)
        # ---- regions
        self.frozen = (np.abs(V[:, 0]) < o['edit_min_x']) | (V[:, 2] > o['frozen_z']) | (V[:, 1] > o['max_y'] - o['front_band'])
        fz = np.nonzero(self.frozen)[0]
        dfz, _ = cKDTree(V[fz]).query(V)
        self.d_frozen = dfz
        fixed = (V[:, 0] <= 0) | self.frozen | (dfz < o['frozen_margin'])
        # one edge ring around the frozen region is fixed too: a mesh edge that crosses into the frozen region
        # then has h = 0 at both ends, so the linearly interpolated skin cannot leak into it (the input mesh has
        # edges up to ~3 mm long on the outer side wall at z 38)
        if o.get('frozen_ring', 0) > 0:
            E = np.r_[self.T[:, [0, 1]], self.T[:, [1, 2]], self.T[:, [2, 0]]]
            fr = self.frozen.copy()
            for _ in range(int(o['frozen_ring'])):
                hit = fr[E[:, 0]] | fr[E[:, 1]]
                fr[E[hit, 0]] = True
                fr[E[hit, 1]] = True
            fixed |= fr
        # bbox keepers (screws: bbox min y / z; optional x extent)
        zmin, ymin, xmax = V[:, 2].min(), V[:, 1].min(), V[:, 0].max()
        self.bbox = dict(zmin=zmin, ymin=ymin, xmax=xmax)
        fixed |= V[:, 2] < zmin + o['bbox_keep_band']
        fixed |= V[:, 1] < ymin + o['bbox_keep_band']
        if o['keep_bbox_x']:
            fixed |= V[:, 0] > xmax - o['bbox_keep_band']
        self.fixed0 = fixed
        self.F = np.nonzero(~fixed)[0]
        log(o, f'mesh {n} verts, free {len(self.F)}, fixed {fixed.sum()}  ({time.time() - t0:.2f} s)')
        # ---- operators
        L, m = cotan_laplacian(V, self.T)
        self.L, self.m = L, m
        Minv = sp.diags(1.0 / m)
        self.K = (L.T @ Minv @ L).tocsr()
        self.Mass = sp.diags(m).tocsr()
        log(o, f'operators ({time.time() - t0:.2f} s)')
        self.feat_w = np.zeros(n)
        if o['keep_features'] > 0 or o['seat_feature_soft'] or o['fair_curv'] > 0:
            self.feat_w = self.feature_weight()
            log(o, f"features: {int((self.feat_w > 0.5).sum())} verts ({time.time() - t0:.2f} s)")

    def feature_weight(self):
        """0..1 on the generator's own design features of the input mesh in the editable region (concave R < 12,
        convex R < 3, concave turn > 10 or convex turn > 25 deg/mm: the same rule as the acceptance's 'unchanged
        design features'), dilated by feature_dilate mm, fading over 1.5 mm."""
        import gripmetrics as GM
        o = self.opt
        V, T = self.V, self.T
        reg = np.nonzero((V[:, 0] >= o['edit_min_x'] - 3) & (V[:, 2] < o['frozen_z'] + 3))[0]
        tree = cKDTree(V)
        C = GM.curvature(V, T, idx=reg, tree=tree)
        _, kc, kv = GM.normal_turn(V, T, reg, tree=tree, signed=True)
        f = (C['k2'] < -1 / 12.0) | (C['k1'] > 1 / 3.0) | (kc > 10.0) | (kv > 25.0)
        self.curv0 = dict(idx=reg, k1=C['k1'], k2=C['k2'])
        fi = reg[f]
        w = np.zeros(len(V))
        if len(fi):
            d, _ = cKDTree(V[fi]).query(V, distance_upper_bound=o['feature_dilate'] + 2)
            t = np.clip((d - o['feature_dilate']) / 1.5, 0, 1)
            w = np.where(np.isfinite(d), 1 - t * t * (3 - 2 * t), 0.0)
        return w

    def curv_fair_mask(self, P, h):
        """Phase-2 fairing weight: grows where the current skin is tighter than fair_curv_concave (concave) or
        fair_curv_convex (convex), except on kept features that are not edited."""
        import gripmetrics as GM
        o = self.opt
        reg = self.F
        tree = cKDTree(P)
        C = GM.curvature(P, self.T, idx=reg, tree=tree)
        vc = np.maximum(-C['k2'] * o['fair_curv_concave'] - 1.0, 0.0)
        vv = np.maximum(C['k1'] * o['fair_curv_convex'] - 1.0, 0.0)
        w = np.zeros(len(P))
        w[reg] = np.minimum(vc * vc + vv * vv, o['fair_curv_max'])
        w[(self.feat_w > 0.5) & (np.abs(h) < 0.15)] = 0.0
        # spread over the 3 mm neighbourhood (the quadric reads over 3 mm too)
        nb = tree.query_ball_point(P[reg], 2.0, workers=2)
        w2 = w.copy()
        for i, lst in zip(reg, nb):
            w2[i] = w[lst].max()
        self._curv_now = (reg, C['k1'], C['k2'])
        return w2

    # ------------------------------------------------------------------ pieces linearised at the current skin
    def moved(self, h):
        return self.V + h[:, None] * self.D

    def contact_rows(self, h, P, tree):
        """Soft targets: for every contact-zone axis sample, the skin point nearest to it should sit at
        gap = -Press (the rigid finger sinks Press mm). Linearised at the nearest point of the current skin."""
        o = self.opt
        rows, cols, vals, tgt, wts, info = [], [], [], [], [], {}
        r = 0
        T = self.T
        for zname in CONTACT_ZONES:
            segs = self.zones.get(zname)
            if not segs:
                continue
            pts, rad, wlen = [], [], []
            for a, b, rr in segs:
                L = float(np.linalg.norm(b - a))
                k = max(int(np.ceil(L / 0.5)), 1)
                for t in np.linspace(0, 1, k + 1):
                    pts.append(a + t * (b - a)); rad.append(rr); wlen.append(L / (k + 1))
            pts, rad, wlen = np.array(pts), np.array(rad), np.array(wlen)
            gaps, feet = [], []
            for p, rr in zip(pts, rad):
                loc, nrm, fi, d = tree.find_nearest(Vector(p))
                if loc is None:
                    gaps.append(1e9); feet.append(None); continue
                s = -d if inside(tree, p) else d
                gaps.append(s - rr)
                feet.append((np.array(loc), fi))
            gaps = np.array(gaps)
            res = gaps + o['press']
            rmin = res.min()
            zsc = (o['zone_scale'] or {}).get(zname, 1.0)
            c = max((o['zone_reach'] or {}).get(zname, o['contact_reach']), 0.25)
            wrob = np.exp(-(np.maximum(res - rmin, 0.0) / c) ** 2)
            if o['contact_mode'] == 'deep':
                # the deepest part of the zone pulled to exactly -Press (two-sided); the rest only toward the finger
                # (one-sided, weak): the finger is supported where the case comes close, without hugging it
                wdeep = np.exp(-((res - rmin) / 0.3) ** 2)
                wrob = o['w_deep'] * wdeep + o['w_support'] * wrob * (res > 0)
            info[zname] = dict(min_gap=float(gaps.min()), penetration=float(max(0.0, -gaps.min())),
                               support=float(wlen[gaps <= 1.5].sum() / wlen.sum() * 100))
            for i in range(len(pts)):
                if feet[i] is None:
                    continue
                q, fi = feet[i]
                tri = T[fi]
                bc = barycentric(q, P[tri[0]], P[tri[1]], P[tri[2]])
                nf = (bc[:, None] * self.D[tri]).sum(0)
                nf /= np.linalg.norm(nf)
                mdir = pts[i] - q
                dd = np.linalg.norm(mdir)
                if dd < 1e-6:
                    continue
                mdir /= dd
                if gaps[i] + rad[i] < 0:     # axis inside the case: the gap grows when the skin recedes
                    mdir = -mdir
                cc = max(float(nf @ mdir), 0.3)
                delta = res[i] / cc            # move the foot out by this (along n) to reach gap = -Press
                hcur = float(bc @ h[tri])
                w = o['contact_scale'] * o['contact_weight'] * zsc * wlen[i] * wrob[i]
                for k in range(3):
                    rows.append(r); cols.append(tri[k]); vals.append(bc[k])
                tgt.append(hcur + delta); wts.append(w)
                r += 1
        A = sp.coo_matrix((vals, (rows, cols)), shape=(r, len(self.V))).tocsr()
        return A, np.array(tgt), np.array(wts), info

    def bounds(self, h, P):
        """Per-vertex box bounds lo <= h <= hi from the no-dig rule, the frozen-region entry limits, bbox,
        the cutter wall and the keep-outs, linearised at the current skin P."""
        o = self.opt
        V, N = self.V, self.D
        n = len(V)
        hi = np.full(n, np.inf)
        lo = np.full(n, -np.inf)
        F = self.F
        PF, NF, hF = P[F], N[F], h[F]
        D = o['max_dig']
        # -- no dig: every capsule (all 20)
        for name, (a, b, r) in self.caps.items():
            C = seg_closest(PF, a, b)
            u = PF - C
            du = np.linalg.norm(u, axis=1)
            d = du - r
            near = d < o['dig_consider']
            if not near.any():
                continue
            u = u / np.maximum(du, 1e-9)[:, None]
            s = np.einsum('ij,ij->i', NF, u)
            m1 = near & (s < -0.15)                 # growing moves toward the capsule axis
            ub = hF + (d + D) / np.maximum(-s, 1e-3)
            hi[F[m1]] = np.minimum(hi[F[m1]], ub[m1])
            m2 = near & (s > 0.15) & (d < -D)       # skin behind the axis side: growing frees it
            lb = hF + (-D - d) / np.maximum(s, 1e-3)
            lo[F[m2]] = np.maximum(lo[F[m2]], lb[m2])
        # -- never into the frozen region: y <= max_y - band - margin, z <= frozen_z - margin, x >= edit_min_x + margin
        #    (a vertex already beyond a limit may stay where it is: the bound is never below 0 / above 0)
        em = o['entry_margin']
        def upper(m, lim, coord, nc):          # coord + h * nc <= lim with nc > 0
            hi[F[m]] = np.minimum(hi[F[m]], np.maximum((lim - coord[m]) / nc[m], 0.0))
        def lower(m, lim, coord, nc):          # coord + h * nc <= lim with nc < 0  ->  h >= ...
            lo[F[m]] = np.maximum(lo[F[m]], np.minimum((lim - coord[m]) / nc[m], 0.0))
        ny, nz, nx = NF[:, 1], NF[:, 2], NF[:, 0]
        VF = V[F]
        ylim = o['max_y'] - o['front_band'] - em
        upper(ny > 1e-3, ylim, VF[:, 1], ny); lower(ny < -1e-3, ylim, VF[:, 1], ny)
        zlim = o['frozen_z'] - em
        upper(nz > 1e-3, zlim, VF[:, 2], nz); lower(nz < -1e-3, zlim, VF[:, 2], nz)
        # x >= xlim  <=>  -x + h * (-nx) <= -xlim
        xlim = o['edit_min_x'] + em
        upper(nx < -1e-3, -xlim, -VF[:, 0], -nx); lower(nx > 1e-3, -xlim, -VF[:, 0], -nx)
        # -- bbox min y / min z (screws) and optionally max x: nothing may pass them
        bb = self.bbox
        m = nz < -1e-3
        hi[F[m]] = np.minimum(hi[F[m]], np.maximum((V[F[m], 2] - (bb['zmin'] + 0.02)) / (-nz[m]), 0.0))
        m = ny < -1e-3
        hi[F[m]] = np.minimum(hi[F[m]], np.maximum((V[F[m], 1] - (bb['ymin'] + 0.02)) / (-ny[m]), 0.0))
        if o['keep_bbox_x']:
            m = nx > 1e-3
            hi[F[m]] = np.minimum(hi[F[m]], np.maximum(((bb['xmax'] - 0.02) - V[F[m], 0]) / nx[m], 0.0))
        # -- the generator's own features stay unchanged (hard), except around the seats
        if o['feature_hard'] > 0:
            fm = self.feat_keep_mask[F]
            hi[F[fm]] = np.minimum(hi[F[fm]], o['feature_hard'])
            lo[F[fm]] = np.maximum(lo[F[fm]], -o['feature_hard'])
        # -- cutter wall and keep-outs: a receding skin keeps its wall over them (lower bounds)
        for P_, cfg in self.seats.items():
            Vc, Tc = cfg['cutter']
            self._wall_bound(cfg['_cutter_bvh'], PF, NF, hF, F, lo, o['cutter_wall'], skip=cfg.get('_in_opening'))
        for name, (Vk, Tk, need) in self.keepouts.items():
            self._wall_bound(self._ko_bvh[name], PF, NF, hF, F, lo, need)
        return lo, hi

    def _wall_bound(self, tree, PF, NF, hF, F, lo, need, skip=None, reach=4.0):
        """A receding skin keeps min(need, its no-fit wall) over the object: it never removes wall there (it does
        not add any where the no-fit wall is already thinner: that is the generator's own design)."""
        key = id(tree)
        cache = self.__dict__.setdefault('_wall0', {})
        if key not in cache:
            w0 = np.full(len(F), np.inf)
            for j in range(len(F)):
                loc, nrm, fi, d = tree.find_nearest(Vector(self.V[F[j]]), reach + need + 6.0)
                if loc is None:
                    continue
                u = self.V[F[j]] - np.array(loc)
                du = np.linalg.norm(u)
                w0[j] = du if float(u @ np.array(nrm)) > 0 else -du
            cache[key] = w0
        w0 = cache[key]
        for j in range(len(F)):
            if skip is not None and skip[F[j]]:
                continue
            if w0[j] <= 0:          # the no-fit skin is inside the object here (a cut): not a wall case
                continue
            need_j = min(need, w0[j])
            loc, nrm, fi, d = tree.find_nearest(Vector(PF[j]), reach + need)
            if loc is None:
                continue
            u = PF[j] - np.array(loc)
            du = np.linalg.norm(u)
            if du < 1e-9 or float(u @ np.array(nrm)) <= 0.0:   # inside the object (or on it): not a wall case
                continue
            u /= du
            s = float(NF[j] @ u)
            if s > 0.15:   # growing moves away from the object: receding is limited
                lo[F[j]] = max(lo[F[j]], hF[j] + (need_j - du) / s)

    def seat_targets(self, h=None, P=None):
        """Seat plane targets (deterministic: they do not depend on the current h).
        For each trigger, a vertex on the sheet facing the seat plane whose landing point on the plane (moved
        along its own normal) lies inside the opening or within seat_rim mm of it should land there.
        A vertex that qualifies for both seats belongs to the one with the smaller move.
          hard (Dirichlet): the core within seat_core mm of the press-face centre, and every rim / opening
                            vertex farther than seat_corridor mm from the OTHER trigger's opening outline;
          corridor:         the rest. R2 and R4 seat planes are offset by 6.4-9.5 mm while the openings are only
                            ~5 mm apart in the plane, so no smooth skin can be flush with both there: no target
                            (or a soft one, weight seat_soft per mm2), the bending energy shapes the ramp."""
        o = self.opt
        if getattr(self, '_seat_cache', None) is not None:
            return self._seat_cache
        V, N = self.V, self.D
        names = list(self.seats)
        per = {}
        for name, cfg in self.seats.items():
            c, nn, S = cfg['center'], cfg['normal'], cfg['segments']
            s0 = c - o['seat_proud'] * nn
            e1 = np.cross(nn, [0.0, 0.0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(nn, e1)
            cosv = N @ nn
            hv = np.where(cosv > 1e-3, -((V - s0) @ nn) / np.maximum(cosv, 1e-3), np.inf)
            X = V + np.where(np.isfinite(hv), hv, 0.0)[:, None] * N          # landing points on the plane
            rho = np.linalg.norm(X - s0, axis=1)
            q = S.reshape(-1, 3) - s0
            ext = float(np.linalg.norm(q - np.outer(q @ nn, nn), axis=1).max())
            cand = np.nonzero((np.abs(hv) < o['seat_window']) & (rho < ext + o['seat_rim'] + 1.0)
                              & (cosv > o['seat_cos_min']) & ~self.fixed0)[0]
            Q2 = np.c_[(X[cand] - s0) @ e1, (X[cand] - s0) @ e2]
            S2 = np.stack([np.c_[(S[:, k] - s0) @ e1, (S[:, k] - s0) @ e2] for k in (0, 1)], axis=1)
            ins = inside_polygon_2d(Q2, S2)
            dop = dist_to_segments(X[cand], S)
            sel = cand[ins | (dop <= o['seat_rim'])]
            inop = np.zeros(len(V), bool)
            inop[cand[ins | (dop <= 0.75)]] = True
            cfg['_in_opening'] = inop
            per[name] = dict(sel=sel, hv=hv[sel], core=rho[sel] <= o['seat_core'], X=X[sel])
        # a vertex claimed by two seats goes to the smaller move
        owner = {}
        for name, d in per.items():
            for i, v in zip(d['sel'], d['hv']):
                if i not in owner or abs(v) < abs(owner[i][1]):
                    owner[i] = (name, v)
        hard_i, hard_v, soft_i, soft_v, soft_w = [], [], [], [], []
        for name, d in per.items():
            keep = np.array([owner[i][0] == name for i in d['sel']], bool) if len(d['sel']) else np.zeros(0, bool)
            sel, hv, core, X = d['sel'][keep], d['hv'][keep], d['core'][keep], d['X'][keep]
            others = [m for m in names if m != name]
            if others and o['seat_corridor'] > 0:
                d_other = np.min([dist_to_segments(X, self.seats[m]['segments']) for m in others], axis=0)
                contested = (d_other < o['seat_corridor']) & ~core
            else:
                contested = np.zeros(len(sel), bool)
            if o['seat_mode'] == 'graded':
                # hard only the core under the press-face centre; elsewhere a soft target whose weight falls with
                # the move it needs (a flush rim where it is cheap, no platform where the skin is far from the plane)
                ws = o['seat_w'] * np.exp(-(np.abs(hv) / o['seat_move_scale']) ** 2)
                if o['seat_feature_soft']:
                    ws = np.where(self.feat_w[sel] > 0.5, ws * 0.05, ws)
                hard = core
                far = ~core
                hard_i.append(sel[hard]); hard_v.append(hv[hard])
                soft_i.append(sel[far]); soft_v.append(hv[far]); soft_w.append(ws[far] * self.m[sel[far]])
            else:
                far = (np.abs(hv) > o['seat_hard_max']) & ~core & ~contested
                if o['seat_feature_soft']:
                    far |= (self.feat_w[sel] > 0.5) & ~core & ~contested
                hard = ~contested & ~far
                hard_i.append(sel[hard]); hard_v.append(hv[hard])
                soft_i.append(sel[far]); soft_v.append(hv[far]); soft_w.append(o['seat_soft_far'] * self.m[sel[far]])
                if o['seat_soft'] > 0:
                    soft_i.append(sel[contested]); soft_v.append(hv[contested]); soft_w.append(o['seat_soft'] * self.m[sel[contested]])
            self.seats[name]['_n_far'] = int(far.sum())
            self.seats[name]['_n_seat'] = (int((~contested).sum()), int(contested.sum()))
        cat = lambda L, dt: np.concatenate(L).astype(dt) if L else np.zeros(0, dt)
        res = (cat(hard_i, int), cat(hard_v, float), cat(soft_i, int), cat(soft_v, float))
        self._seat_soft_w = cat(soft_w, float)
        self._seat_cache = res
        return res

    # ------------------------------------------------------------------------------------- the QP
    def qp(self, Q, b, lo, hi, x0, dmask, dval):
        """min 1/2 x'Qx - b'x  s.t. x[d] = dval, lo <= x <= hi (box), over the free set. Primal active set."""
        o = self.opt
        n = Q.shape[0]
        x = x0.copy()
        x[dmask] = dval[dmask]
        lo = np.where(dmask, -np.inf, lo)
        hi = np.where(dmask, np.inf, hi)
        bad = lo > hi                                  # inconsistent bounds: keep the no-dig (upper) one
        lo[bad] = hi[bad]
        act_hi = (~dmask) & (x >= hi - 1e-9)
        act_lo = (~dmask) & (x <= lo + 1e-9) & ~act_hi
        Qc = Q.tocsc()
        hist = []
        for it in range(o['qp_iters']):
            Dm = dmask | act_hi | act_lo
            xd = np.where(dmask, x, np.where(act_hi, hi, np.where(act_lo, lo, 0.0)))
            U = np.nonzero(~Dm)[0]
            Di = np.nonzero(Dm)[0]
            rhs = b[U] - Qc[U][:, Di] @ xd[Di]
            QUU = Qc[U][:, U].tocsc()
            xu = spl.splu(QUU, permc_spec='COLAMD').solve(rhs)
            x = xd.copy()
            x[U] = xu
            g = Q @ x - b
            add_hi = (~Dm) & (x > hi + 1e-6)
            add_lo = (~Dm) & (x < lo - 1e-6)
            rel_hi = act_hi & (g > 1e-7)
            rel_lo = act_lo & (g < -1e-7)
            if it >= o['qp_iters'] // 2:          # anti-cycling: only add from here on
                rel_hi[:] = False; rel_lo[:] = False
            hist.append((int(add_hi.sum() + add_lo.sum()), int(rel_hi.sum() + rel_lo.sum())))
            if not (add_hi.any() or add_lo.any() or rel_hi.any() or rel_lo.any()):
                break
            act_hi = (act_hi & ~rel_hi) | add_hi
            act_lo = (act_lo & ~rel_lo) | add_lo
        x = np.where(dmask, x, np.clip(x, lo, hi))
        return x, dict(iters=it + 1, active_hi=int(act_hi.sum()), active_lo=int(act_lo.sum()), hist=hist[-3:])

    # ------------------------------------------------------------------------------------- solve
    def solve(self, h0=None):
        o = self.opt
        t0 = time.time()
        n = len(self.V)
        F = self.F
        h = np.zeros(n) if h0 is None else h0.copy()
        for cfg in self.seats.values():
            Vc, Tc = cfg['cutter']
            cfg['_cutter_bvh'] = bvh(Vc, orient(Vc, Tc))
        self._ko_bvh = {k: bvh(Vk, orient(Vk, Tk)) for k, (Vk, Tk, need) in self.keepouts.items()}
        self._seat_cache = None
        self._wc_acc = None
        self._wall0 = {}
        self.feat_keep_mask = np.zeros(n, bool)
        if o['feature_hard'] > 0:
            self.seat_targets()
            near_seat = np.zeros(n, bool)
            for cfg in self.seats.values():
                c, nn = cfg['center'], cfg['normal']
                S = cfg['segments']
                q = S.reshape(-1, 3) - c
                ext = float(np.linalg.norm(q - np.outer(q @ nn, nn), axis=1).max())
                near_seat |= np.linalg.norm(self.V - c, axis=1) < ext + o['seat_rim'] + o['feature_seat_clear']
            self.feat_keep_mask = (self.feat_w > 0.5) & ~near_seat & ~self.fixed0
            log(o, f'features kept (hard): {int(self.feat_keep_mask.sum())} verts')
        sm = o['smoothness']
        Qb = (o['bend'] * sm) * self.K + (o['membrane'] * sm) * self.L + o['tension'] * self.Mass
        if o['keep_features'] > 0:
            Qb = Qb + o['keep_features'] * sp.diags(self.m * self.feat_w)
        Fm = np.zeros(n, bool)
        Fm[F] = True
        iF = -np.ones(n, int)
        iF[F] = np.arange(len(F))
        hist = []
        for it in range(o['outer_iters']):
            ti = time.time()
            P = self.moved(h)
            tree = bvh(P, self.T)
            A, tgt, w, cinfo = self.contact_rows(h, P, tree)
            sidx, sval, qidx, qval = self.seat_targets(h, P)
            lo, hi = self.bounds(h, P)
            if len(qidx):   # soft seat rows (the corridor between the two triggers)
                As = sp.coo_matrix((np.ones(len(qidx)), (np.arange(len(qidx)), qidx)), shape=(len(qidx), n)).tocsr()
                ws = self._seat_soft_w
                A = sp.vstack([A, As]).tocsr(); tgt = np.r_[tgt, qval]; w = np.r_[w, ws]
            # restrict to the free set (fixed vertices are h = 0)
            AF = A[:, F]
            Wd = sp.diags(w)
            Q = (Qb[F][:, F] + AF.T @ Wd @ AF).tocsr()
            b = AF.T @ (w * tgt)
            if o['fair'] > 0 and it == o['fair_after']:
                self.fair_mask = self.make_fair_mask(h)
            if o['fair'] > 0 and it >= o['fair_after']:
                Qf, bf = self.fair_terms(P)
                Q = (Q + o['fair'] * Qf[F][:, F]).tocsr()
                b = b + o['fair'] * bf[F]
            if o['fair_curv'] > 0 and it >= o['fair_after']:
                wc = self.curv_fair_mask(P, h)
                self._wc_acc = wc if getattr(self, '_wc_acc', None) is None else np.maximum(self._wc_acc * 0.8, wc)
                Qf, bf = self.fair_terms(P, self._wc_acc)
                Q = (Q + o['fair_curv'] * Qf[F][:, F]).tocsr()
                b = b + o['fair_curv'] * bf[F]
                log(o, f'   curvature fairing on {int((self._wc_acc > 0.05).sum())} verts')
            dmask = np.zeros(len(F), bool)
            dval = np.zeros(len(F))
            ok = iF[sidx] >= 0
            dmask[iF[sidx[ok]]] = True
            dval[iF[sidx[ok]]] = sval[ok]
            # a seat value that would dig a capsule: the no-dig bound wins (reported)
            conflict = dmask & (dval > hi[F] + 1e-6)
            dval[conflict] = hi[F][conflict]
            xF, qinfo = self.qp(Q, b, lo[F], hi[F], h[F], dmask, dval)
            hn = np.zeros(n)
            hn[F] = xF
            if it >= o['damp_from']:
                hn = h + o['damping'] * (hn - h)
            dh = float(np.abs(hn - h).max())
            jd = int(np.argmax(np.abs(hn - h)))
            self._hhist = getattr(self, '_hhist', []) + [hn.copy()]
            log(o, f'   largest change at {self.V[jd].round(1)} ({h[jd]:.2f} -> {hn[jd]:.2f}), lo {lo[jd]:.2f} hi {hi[jd]:.2f}')
            h = hn
            rec = dict(it=it, dh=round(dh, 4), seconds=round(time.time() - ti, 2), seat_verts=int(dmask.sum()),
                       seat_conflicts=int(conflict.sum()), qp=qinfo,
                       contacts={k: (round(v['penetration'], 2), round(v['min_gap'], 2), round(v['support'], 1)) for k, v in cinfo.items()},
                       h_range=(round(float(h.min()), 2), round(float(h.max()), 2)))
            hist.append(rec)
            log(o, f"it {it}: dh {dh:.3f}  h {h.min():.2f}..{h.max():.2f}  seat {int(dmask.sum())}+{len(qidx)} soft (conflicts {int(conflict.sum())})  "
                   f"qp {qinfo['iters']} it, active {qinfo['active_hi']}/{qinfo['active_lo']}  {time.time() - ti:.1f} s")
            if dh < o['tol'] and it > 1:
                break
        self.h = h
        self.history = hist
        self.seconds = time.time() - t0
        log(o, f'solve {self.seconds:.1f} s')
        return h

    def make_fair_mask(self, h):
        """Smooth 0..1 weight: 1 within fair_dilate mm of where phase 1 moved the skin more than fair_edit mm."""
        o = self.opt
        ed = np.abs(h) > o['fair_edit']
        w = np.zeros(len(h))
        if ed.any():
            d, _ = cKDTree(self.V[ed]).query(self.V, distance_upper_bound=o['fair_dilate'] + o['fair_fade'] + 1)
            t = np.clip((d - o['fair_dilate']) / max(o['fair_fade'], 1e-6), 0, 1)
            w = np.where(np.isfinite(d), 1 - t * t * (3 - 2 * t), 0.0)
        w[self.fixed0] = 0.0
        log(o, f'fair mask: {int((w > 0.5).sum())} verts')
        return w

    def fair_terms(self, P, mask=None):
        """Quadratic form of sum_v w_v |(L_P (V + h D))_v|^2 / m_v (mean curvature^2 of the final surface, masked),
        with the cotangent Laplacian of the current skin P: returns (Q, b) for 1/2 h'Qh - b'h (factor 2 absorbed)."""
        Lp, mp = cotan_laplacian(P, self.T)
        W = sp.diags((self.fair_mask if mask is None else mask) / mp)
        Q = None
        b = np.zeros(len(P))
        for c in range(3):
            Bc = (Lp @ sp.diags(self.D[:, c])).tocsr()
            r0 = Lp @ self.V[:, c]
            Qc = Bc.T @ W @ Bc
            Q = Qc if Q is None else Q + Qc
            b -= Bc.T @ (W @ r0)
        return Q.tocsr(), b

    def offset_vectors(self, h=None):
        h = self.h if h is None else h
        return h[:, None] * self.D


# ------------------------------------------------------------------------------------------ helpers
def zones_from_caps(caps):
    """gripmetrics.zones(caps), re-implemented (no import cycle): {zone: [(a, b, r), ...]}."""
    import gripmetrics as GM
    return GM.zones(caps)


def load_targets(path='/tmp/claude-0/sdf/smooth/targets.npz', json_path='/tmp/claude-0/sdf/smooth/targets.json'):
    """caps, seats and keep-outs from the shared targets (development; refit.py reads them live from the .blend)."""
    t = np.load(path)
    J = json.load(open(json_path))
    names = J['caps_names']
    caps = {k: (t['caps_a'][i], t['caps_b'][i], float(t['caps_r'][i])) for i, k in enumerate(names)}
    seats = {}
    for P in ('R2', 'R4'):
        seats[P] = dict(center=t[f'seat_{P}_center'], normal=t[f'seat_{P}_normal'], segments=t[f'seat_{P}_opening_segments'],
                        cutter=(t[f'cutter_{P}_V'], t[f'cutter_{P}_T']))
    keep = {}
    for k, v in J['keepouts'].items():
        if v['kind'] != 'internal' or v['nofit_wall_to_skin_mm'] is None:
            continue
        need = min(v['nofit_wall_to_skin_mm'], 1.5)
        if need <= 0:
            continue
        keep[k] = (t[k + '_V'], t[k + '_T'], need)
    return caps, seats, keep
