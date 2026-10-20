"""lowfreq.py: SPEC v2 shape solve. ONE smooth, low-frequency signed normal offset of the GripV handle.

The new skin is  P = V + h * D  on the stage input mesh (subdivided GripV basemesh, level 'high'), right
handle only (the stage mirrors it to the left one). Unlike v1 (one free height per vertex), h lives in a
LOW-FREQUENCY BASIS:

    h = W * (B c)

  B   compactly supported radial basis functions (Wendland C4, C4-smooth), support radius 'Smoothness'
      (default 34 mm, i.e. a support of 68 mm across), centres spread over the handle every ~rho/3.5 mm.
      A single basis function is a bump ~0.45 rho wide at half height (15 mm at rho 34), and any
      combination of them is C4-smooth: the shape cannot hold a finger groove or a capsule imprint.
  W   a smooth (C2) window: 0 on the frozen region (|x| < 25, z > 38 - Top Clear, y > -4.658) and
      1 beyond 'taper' mm from it, so the edit fades out exactly and smoothly.
  c   the unknowns (one weight per basis function, ~150-300 numbers).

c minimises   bending |Laplacian h|^2  +  |h|^2 / Spread^4          (keep the shape calm and local)
subject to (squared-hinge penalties, semismooth Newton on the small dense system, re-linearised at the
moved skin in a few outer iterations):
  TOUCH (one-sided, soft tissue): every contact zone has some point with gap <= Touch (0.3 mm): where the
        case is clear of a zone it grows until it touches; where it already overlaps nothing happens.
  GIVE: no capsule overlaps the skin deeper than its soft-tissue allowance ('Palm Give' 3.5 for the palm,
        the metacarpals and the pinky, 'Finger Give' 1.5 for the other phalanges). Only the excess recedes.
  STROKE: the middle / ring distal capsules, moved along their trigger's travel by the full stroke, keep
        'Stroke Clear' (0.3 mm) from every skin point outside the opening (cutter).
  WALLS: >= 'Screw Wall' (3 mm) between the skin and the 'screw bools' channel (or the plain value when it
        is already less); >= min(plain, 1.5) to the other internal objects; >= min(plain, 1.2) to the
        trigger cutters outside the opening.
  LIMITS: y <= max Y - 0.5, no growth below the bbox min y / z, no growth beyond the plain x extent.
  CREASES: near GripV's own concave creases growth is capped (cap0 + slope * distance, i.e. it tapers in
        over ~10 mm), near its sharp convex edges recession is capped the same way.
"""
import time
import json
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
from scipy.spatial import cKDTree

DEFAULTS = dict(
    # ---- inputs Olex may adjust (refit reads them from the file when present)
    smoothness=34.0,     # 'Smoothness' (mm): support radius of the basis; larger = broader, calmer, fewer exact touches
    palm_give=3.5,       # 'Palm Give' (mm): allowed soft-tissue overlap of palm, metacarpals and the pinky
    finger_give=1.5,     # 'Finger Give' (mm): allowed overlap of the other phalanges
    touch=0.3,           # gap at which a zone counts as touching (mm)
    stroke_clear=0.3,    # 'Stroke Clear' (mm): fingertip vs case beside the opening at full trigger travel
    screw_wall=3.0,      # 'Screw Wall' (mm)
    middle_pip=0.0,      # 'Middle PIP' pull (0 = do not chase the middle finger's PIP: a calm shape cannot reach it)
    top_clear=3.0,       # 'Top Clear' (mm): the change is zero above z 38 - this (case_top / middle plate untouched)
    # ---- internal
    spacing=None,        # centre spacing (mm); None = smoothness / 3.5
    taper=10.0,          # window ramp width from the body |x| < 25 (mm)
    taper_top=10.0,      # window ramp width below the top band and behind the front band (mm; 6 left a visible step)
    h_max=8.0,           # safety: |h| <= this everywhere (mm)
    tikhonov=1e-4,       # small pull of the basis weights to 0 (no large cancelling weights)
    touch_huber=0.5,     # touch rows pull linearly (not quadratically) beyond this violation (mm)
    spread=22.0,         # decay length of the |h|^2 term (mm)
    bend=1.0,            # thin-plate weight
    membrane=0.02,       # membrane (|grad h|^2) weight (per mm^2): dominates above ~sqrt(bend/membrane) = 7 mm
    w_touch=3e3, w_hard=1e5, w_give=3e4, w_crease=1e4,
    touch_margin=0.15,   # solve for gap <= touch - margin (the remesh and the nearest-vertex distance add ~0.1)
    hard_margin=0.12,    # keep-outs and allowances are solved this much stricter
    zone_weights=None,   # {zone: factor}
    crease_cap0=0.3, crease_slope=0.35, crease_reach=10.0,
    crease_concave_R=12.0, crease_convex_R=4.0,
    outer_iters=18, trust=2.0, tol=0.01, newton_iters=40, damping=0.6, damp_from=3,
    stroke_steps=3, reach=8.0,
    xmax_hold=True,
    bbox_pin=0.6,        # vertices this close to the bbox min y / z planes keep h = 0 (mm)
    bbox_ramp=6.0,       # the window fades to 0 at the bbox min y / z planes over this (mm)
    stroke_margin=0.2,   # extra margin of the stroke keep-out (mm)
    rim_band=1.5,        # cut-away vertices this close to the cutter surface still count as the opening rim (mm)
    h_cap=6.0,          # largest |h| a row may demand at a fully free vertex (scaled by the window)
    # relief limit (the acceptance metric as a soft constraint): |h - mean of h on the 12-16 mm ring| <= relief_max
    w_relief=3000.0, relief_max=0.55, relief_bump=1.0, relief_ring=(12.0, 16.0), relief_step=2.5, relief_wmin=0.2,
    verbose=True,
)
FING3 = ('middle', 'ring', 'pinky')
FINGER_GIVE = ('middle_proximal', 'middle_middle', 'ring_proximal', 'ring_middle', 'index_proximal', 'index_middle', 'index_distal')
PALM_GIVE = ('palm', 'middle_metacarpal', 'ring_metacarpal', 'pinky_metacarpal', 'index_metacarpal',
             'pinky_proximal', 'pinky_middle', 'pinky_distal', 'thumb_metacarpal', 'thumb_proximal', 'thumb_distal')
STROKE = {'R2': 'middle_distal', 'R4': 'ring_distal'}
MAX_Y = -4.158
NOFIT_CURV = '/tmp/claude-0/sdf/smooth/baseline/nofit/curvature.npz'


# ------------------------------------------------------------------------------------------ mesh helpers
def orient(V, T):
    vol = np.einsum('ij,ij->i', V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])).sum() / 6.0
    return T[:, ::-1].copy() if vol < 0 else T.copy()


def vertex_normals(V, T):
    cr = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, T[:, k], cr)
    return N / np.maximum(np.linalg.norm(N, axis=1), 1e-15)[:, None]


def vertex_area(V, T):
    A = 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
    m = np.zeros(len(V))
    for k in range(3):
        np.add.at(m, T[:, k], A / 3.0)
    return np.maximum(m, 1e-9)


def smooth_directions(V, T, N, sigma=3.0, sheet_cos=0.0):
    """The v1 offset direction (shape_solver.smooth_directions): normals averaged over sigma mm, same sheet."""
    m = vertex_area(V, T)
    tree = cKDTree(V)
    D = np.zeros_like(N)
    for i, lst in enumerate(tree.query_ball_point(V, 2.5 * sigma, workers=2)):
        lst = np.asarray(lst)
        w = m[lst] * np.exp(-((V[lst] - V[i]) ** 2).sum(1) / (2 * sigma * sigma)) * ((N[lst] @ N[i]) > sheet_cos)
        D[i] = (w[:, None] * N[lst]).sum(0)
    return D / np.maximum(np.linalg.norm(D, axis=1), 1e-12)[:, None]


def cotan_laplacian(V, T):
    n = len(V)
    i0, i1, i2 = T[:, 0], T[:, 1], T[:, 2]

    def cot(a, b, c):
        u, v = V[b] - V[a], V[c] - V[a]
        return np.einsum('ij,ij->i', u, v) / np.maximum(np.linalg.norm(np.cross(u, v), axis=1), 1e-12)
    c0, c1, c2 = cot(i0, i1, i2), cot(i1, i2, i0), cot(i2, i0, i1)
    I = np.concatenate([i1, i2, i2, i0, i0, i1]); J = np.concatenate([i2, i1, i0, i2, i1, i0])
    W = 0.5 * np.concatenate([c0, c0, c1, c1, c2, c2])
    Wm = sp.coo_matrix((W, (I, J)), shape=(n, n)).tocsr()
    return (sp.diags(np.asarray(Wm.sum(1)).ravel()) - Wm).tocsr()


def frozen_mask(V, top_clear=0.0):
    return (np.abs(V[:, 0]) < 25.0) | (V[:, 2] > 38.0 - top_clear) | (V[:, 1] > MAX_Y - 0.5)


def wendland(q):
    q = np.clip(q, 0.0, 1.0)
    return (1 - q) ** 6 * (35 * q * q + 18 * q + 3) / 3.0


def smootherstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


def seg_closest(P, a, b):
    ab = b - a
    t = np.clip((P - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
    return a[None] + t[:, None] * ab[None], t


def seg_samples(a, b, step=0.5):
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
    t = np.linspace(0, 1, n)
    return a[None] + t[:, None] * (b - a)[None], t


def sample_surface(V, T, step=0.6, seed=0):
    A = 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
    n = int(A.sum() / (step * step)) + 1
    rng = np.random.default_rng(seed)
    f = rng.choice(len(T), n, p=A / A.sum())
    u, v = rng.random(n), rng.random(n)
    s = u + v > 1; u[s], v[s] = 1 - u[s], 1 - v[s]
    return V[T[f, 0]] + u[:, None] * (V[T[f, 1]] - V[T[f, 0]]) + v[:, None] * (V[T[f, 2]] - V[T[f, 0]])


def parity_inside(V, T, P):
    """Ray parity, 5 skew rays, majority (mathutils BVH)."""
    import bpy  # noqa: F401
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    tree = BVHTree.FromPolygons(V.tolist(), T.tolist(), all_triangles=True)
    dirs = [Vector(d).normalized() for d in ((0.577, 0.571, 0.583), (-0.61, 0.33, 0.72), (0.23, -0.87, 0.43),
                                             (-0.37, -0.52, -0.77), (0.81, -0.21, -0.55))]
    out = np.zeros(len(P), bool)
    for i, p in enumerate(P):
        votes = 0
        for d in dirs:
            o = Vector(p); n = 0
            while n < 200:
                hit = tree.ray_cast(o, d)
                if hit[0] is None:
                    break
                n += 1
                o = hit[0] + d * 2e-3
            votes += n & 1
        out[i] = votes >= 3
    return out


def zones(caps):
    """gripmetrics.zones as {zone: [(a, b, r)]} (axis sub-segments)."""
    Z = {}
    for f in FING3:
        wrist, mcp, rm = caps[f + '_metacarpal']
        a, b, r = caps[f + '_proximal']
        Z[f'{f} root pad'] = [(mcp, mcp + 0.2 * (wrist - mcp), rm), (a, a + 0.35 * (b - a), r)]
        a, b, r = caps[f + '_middle']
        Z[f'{f} PIP'] = [(a, a + 0.4 * (b - a), r)]
    Z['pinky distal'] = [caps['pinky_distal']]
    Z['palm on swell'] = [(caps[f + '_metacarpal'][1], caps[f + '_metacarpal'][0], caps[f + '_metacarpal'][2]) for f in FING3]
    return Z


# ------------------------------------------------------------------------------------------ the fit
class LowFreqFit:
    def __init__(self, g, opt=None, D=None):
        o = dict(DEFAULTS); o.update(opt or {})
        self.opt = o
        t0 = time.time()
        V = np.asarray(g['V'], float)
        T = orient(V, np.asarray(g['T'], np.int64))
        self.V, self.T = V, T
        self.N = vertex_normals(V, T)
        self.D = D if D is not None else smooth_directions(V, T, self.N, 3.0)
        self.A = vertex_area(V, T)
        self.caps = {k: (np.asarray(a, float), np.asarray(b, float), float(r)) for k, (a, b, r) in g['caps'].items()}
        self.trig = g['trig']
        self.tree = cKDTree(V)
        n = len(V)
        # ---- window: 0 on the frozen region (+ Top Clear), a smooth (C2) ramp over 'taper' mm from the body
        #      (|x| < 25) and over 'taper_top' mm from the top band and the front band
        fz = frozen_mask(V, o['top_clear'])
        W = np.where(fz, 0.0, 1.0)
        for sub, tau in (((np.abs(V[:, 0]) < 25.0), o['taper']), ((V[:, 2] > 38.0 - o['top_clear']), o['taper_top']),
                         ((V[:, 1] > MAX_Y - 0.5), o['taper_top'])):
            if sub.any():
                dist, _ = cKDTree(V[sub]).query(V, distance_upper_bound=tau + 1.0, workers=2)
                W *= smootherstep(np.where(np.isfinite(dist), dist, tau + 1.0) / tau)
        # the bbox min y / z planes (the screws): the window also fades to exactly 0 there, over bbox_ramp mm
        for ax in (1, 2):
            W *= smootherstep((V[:, ax] - V[:, ax].min()) / o['bbox_ramp'])
        self.W = W
        self.free = np.nonzero(self.W > 1e-6)[0]
        # ---- basis: Wendland C4, support rho, centres by Poisson-disk sampling over the free region
        rho = float(o['smoothness'])
        s = float(o['spacing'] or rho / 3.5)
        rng = np.random.default_rng(1)
        cand = self.free[rng.permutation(len(self.free))]
        cand = cand[np.argsort(-self.W[cand], kind='stable')]   # inner points first (stable for a given mesh)
        cent = []
        ct = None
        for i in cand:
            if cent and np.min(np.linalg.norm(V[cent] - V[i], axis=1)) < s:
                continue
            cent.append(i)
        self.centres = V[np.array(cent)]
        m = len(cent)
        rows, cols, vals = [], [], []
        for j, lst in enumerate(self.tree.query_ball_point(self.centres, rho, workers=2)):
            lst = np.asarray(lst)
            lst = lst[self.W[lst] > 0]
            q = np.linalg.norm(V[lst] - self.centres[j], axis=1) / rho
            rows.append(lst); cols.append(np.full(len(lst), j)); vals.append(wendland(q) * self.W[lst])
        self.B = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, m))
        # ---- quadratic energy: thin plate + magnitude / spread^4
        L = cotan_laplacian(V, T)
        Y = (L @ self.B).tocsc()
        Qb = (Y.T @ sp.diags(1.0 / self.A) @ Y).toarray()
        Qm = (self.B.T @ sp.diags(self.A) @ self.B).toarray() / o['spread'] ** 4
        Qg = (self.B.T @ (L @ self.B)).toarray()        # membrane (Dirichlet) energy: keeps a maximum principle,
        Qg = 0.5 * (Qg + Qg.T)                           # i.e. no overshoot dents / bumps beside a pulled place
        self.Q = o['bend'] * Qb + o['membrane'] * Qg + Qm + o['tikhonov'] * np.eye(m)
        # ---- relief operator (the acceptance metric, linear in h): h_i - mean of h on the 12-16 mm ring
        self._prep_relief()
        # ---- static data of the keep-outs
        self._prep_keepouts(g)
        self._prep_creases()
        self.log(f'basis {m} functions (rho {rho:.0f} mm, spacing {s:.1f}), free verts {len(self.free)}, '
                 f'keep-out rows ready ({time.time() - t0:.1f} s)')
        self.history = []

    def log(self, *a):
        if self.opt['verbose']:
            print('[lowfreq]', *a, flush=True)

    # -------------------------------------------------------------------------------- static keep-out data
    def _prep_keepouts(self, g):
        o = self.opt
        V, F = self.V, self.free
        Vf = V[F]
        # trigger cutters: which free vertices are inside (cut away), sample points of the cutter surface
        self.cut_inside = np.zeros(len(V), bool)
        self.cutter_pts = {}
        for name, (Vc, Tc) in g['cutters'].items():
            Vc = np.asarray(Vc, float); Tc = np.asarray(Tc, np.int64)
            lo, hi = Vc.min(0) - 12, Vc.max(0) + 12
            near = F[((Vf > lo) & (Vf < hi)).all(1)]
            if len(near):
                self.cut_inside[near[parity_inside(Vc, Tc, V[near])]] = True
            self.cutter_pts[name] = sample_surface(Vc, Tc, 0.5)
        # vertices inside a cutter but within rim_band of its surface still describe the rim edge of the
        # opening (a triangle across the cut is partly real material): the stroke keep-out sees them too
        self.near_rim = np.zeros(len(V), bool)
        for name, S in self.cutter_pts.items():
            ins = np.nonzero(self.cut_inside)[0]
            if len(ins):
                d, _ = cKDTree(S).query(V[ins], workers=2)
                self.near_rim[ins[d < o['rim_band']]] = True
        # internal objects: surface samples inside the plain case (exits / openings excluded)
        self.internal_pts = {}
        for name, (Vi, Ti) in g['internals'].items():
            Vi = np.asarray(Vi, float); Ti = np.asarray(Ti, np.int64)
            Ps = sample_surface(Vi, Ti, 0.6)
            Ps = Ps[(Ps[:, 0] > 20) & (Ps[:, 2] < 40)]
            if len(Ps) == 0:
                continue
            lo, hi = V[F].min(0) - 6, V[F].max(0) + 6
            Ps = Ps[((Ps > lo) & (Ps < hi)).all(1)]
            if len(Ps) == 0:
                continue
            ins = parity_inside(V, self.T, Ps)
            if ins.any():
                self.internal_pts[name] = Ps[ins]
        # plain walls (distance of each free vertex to each keep-out)
        self.wall0 = {}
        for name, Ps in list(self.internal_pts.items()) + list(self.cutter_pts.items()):
            d, _ = cKDTree(Ps).query(Vf, distance_upper_bound=12.0, workers=2)
            self.wall0[name] = d
        # bbox / position limits of the plain input mesh
        self.ymin0, self.zmin0, self.xmax0 = V[:, 1].min(), V[:, 2].min(), V[:, 0].max()

    def _prep_relief(self):
        o = self.opt
        V, N, A, F = self.V, self.N, self.A, self.free
        pts = F[self.W[F] > o['relief_wmin']]
        rng = np.random.default_rng(2)
        pts = pts[rng.permutation(len(pts))]
        keep, tr = [], None
        grid = {}
        cs = o['relief_step']
        for i in pts:                                    # Poisson-disk subsample (grid hashing)
            key = tuple((V[i] // cs).astype(int))
            ok = True
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                            if np.sum((V[j] - V[i]) ** 2) < cs * cs:
                                ok = False; break
                        if not ok: break
                    if not ok: break
                if not ok: break
            if ok:
                keep.append(i); grid.setdefault(key, []).append(i)
        keep = np.array(keep)
        rows, cols, vals = [], [], []
        r0, r1 = o['relief_ring']
        for k, (i, lst) in enumerate(zip(keep, self.tree.query_ball_point(V[keep], r1, workers=2))):
            lst = np.asarray(lst)
            dd = np.linalg.norm(V[lst] - V[i], axis=1)
            lst = lst[(dd > r0) & (N[lst] @ N[i] > 0.0)]
            if len(lst) < 10:
                continue
            w = A[lst] / A[lst].sum()
            rows += [k] * (len(lst) + 1); cols += [i] + lst.tolist(); vals += [1.0] + (-w).tolist()
        R = sp.csr_matrix((vals, (rows, cols)), shape=(len(keep), len(V)))
        nz = np.diff(R.indptr) > 0
        self.relief_pts = keep[nz]
        self.Grel = (R[nz] @ self.B).tocsr()         # (h_i - ring mean) as rows in c

    def _prep_creases(self):
        """GripV's own concave creases / sharp convex edges (no-fit skin, shared curvature at 3 mm + kinks)."""
        o = self.opt
        z = np.load(NOFIT_CURV)
        Vb = z['V']; reg = z['mask']
        cc = reg & ((z['k2'] < -1 / o['crease_concave_R']) | (z['kink_concave'] > 10))
        cv = reg & ((z['k1'] > 1 / o['crease_convex_R']) | (z['kink_convex'] > 25))
        Vf = self.V[self.free]
        self.d_cc = cKDTree(Vb[cc]).query(Vf, distance_upper_bound=o['crease_reach'] + 1, workers=2)[0] if cc.any() else np.full(len(Vf), np.inf)
        self.d_cv = cKDTree(Vb[cv]).query(Vf, distance_upper_bound=o['crease_reach'] + 1, workers=2)[0] if cv.any() else np.full(len(Vf), np.inf)

    # -------------------------------------------------------------------------------- rows
    def rows(self, h):
        """Linearised one-sided rows  a * h_j >= beta  (vertex j), weight w, tag. Returns arrays."""
        o = self.opt
        V, D, F = self.V, self.D, self.free
        P = V + h[:, None] * D
        J, Aa, Bb, Ww, tags = [], [], [], [], []

        def add(j, a, beta, w, tag):
            j = np.asarray(j); n = len(j)
            if n == 0:
                return
            J.append(j); Aa.append(np.broadcast_to(a, n).astype(float)); Bb.append(np.broadcast_to(beta, n).astype(float))
            Ww.append(np.broadcast_to(w, n).astype(float)); tags.append(np.full(n, tag, object))

        def dist_rows(j, q, dmin, w, tag):
            """|P_j - q| >= dmin, linearised at P."""
            v = P[j] - q
            d = np.linalg.norm(v, axis=1)
            a = np.einsum('ij,ij->i', D[j], v / np.maximum(d, 1e-9)[:, None])
            add(j, a, dmin - d + a * h[j], w, tag)

        Pf = P[F]
        treeP = cKDTree(Pf)
        # ---- soft-tissue allowance (all capsules but the trigger fingertips)
        for name, (a, b, r) in self.caps.items():
            if name in STROKE.values():
                continue
            give = o['palm_give'] if name in PALM_GIVE else o['finger_give']
            lo = np.minimum(a, b) - r - o['reach']; hi = np.maximum(a, b) + r + o['reach']
            sel = np.nonzero(((Pf > lo) & (Pf < hi)).all(1))[0]
            if len(sel) == 0:
                continue
            q, _ = seg_closest(Pf[sel], a, b)
            d = np.linalg.norm(Pf[sel] - q, axis=1)
            k = d < r + o['reach']
            dist_rows(F[sel[k]], q[k], r - give + o['hard_margin'], o['w_give'], 'give:' + name)
        # ---- trigger stroke: distal capsule swept along the travel, vs skin outside the opening
        for P_, capn in STROKE.items():
            if P_ not in self.trig:
                continue
            a, b, r = self.caps[capn]
            t = self.trig[P_]
            for s in np.linspace(0.0, 1.0, o['stroke_steps']):
                sh = np.asarray(t['travel']) * t['travel_mm'] * s
                aa, bb = a + sh, b + sh
                q, _ = seg_closest(Pf, aa, bb)
                d = np.linalg.norm(Pf - q, axis=1)
                k = (d < r + o['reach']) & (~self.cut_inside[F] | self.near_rim[F])
                dist_rows(F[k], q[k], r + o['stroke_clear'] + o['stroke_margin'], o['w_hard'], 'stroke:' + P_)
        # ---- walls to the internal objects and the trigger cutters
        for name, Ps in list(self.internal_pts.items()) + list(self.cutter_pts.items()):
            w0 = self.wall0[name]
            if name == 'screw bools':          # hard: never thinner than plain where plain is <= Screw Wall
                need = np.minimum(w0, o['screw_wall'] + o['hard_margin']) + 0.02   # (+0.02 cancels the slack below: never thinner than plain)
            elif name in self.cutter_pts:
                need = np.where(w0 >= 0.5, np.minimum(w0, 1.2), -np.inf)
            else:
                need = np.minimum(w0, 1.5)
            k = np.nonzero(np.isfinite(w0) & (w0 < need + 6.0) & np.isfinite(need) & ~self.cut_inside[F])[0]
            if len(k) == 0:
                continue
            _, jj = cKDTree(Ps).query(Pf[k], workers=2)
            if name in self.cutter_pts:     # only where the channel lies under the skin (receding thins the wall)
                v = Pf[k] - Ps[jj]
                under = np.einsum('ij,ij->i', self.D[F[k]], v / np.maximum(np.linalg.norm(v, axis=1), 1e-9)[:, None]) > 0.7
                k, jj = k[under], jj[under]
            dist_rows(F[k], Ps[jj], need[k] - 0.02, o['w_hard'], 'wall:' + name)
        # ---- position limits (exact, linear)
        j = F[V[F, 1] > MAX_Y - 12]
        add(j, -D[j, 1], V[j, 1] - (MAX_Y - 0.5 - 0.05), o['w_hard'], 'maxY')
        j = F[V[F, 1] < self.ymin0 + 10]
        add(j, D[j, 1], self.ymin0 - V[j, 1], o['w_hard'], 'ymin')
        j = F[V[F, 2] < self.zmin0 + 10]
        add(j, D[j, 2], self.zmin0 - V[j, 2], o['w_hard'], 'zmin')
        # bbox min y / z (the screws): the extreme vertices do not move at all
        j = F[(V[F, 1] < self.ymin0 + o['bbox_pin']) | (V[F, 2] < self.zmin0 + o['bbox_pin'])]
        add(j, 1.0, -0.01, o['w_hard'], 'bbox_pin'); add(j, -1.0, -0.01, o['w_hard'], 'bbox_pin')
        if o['xmax_hold']:
            j = F[V[F, 0] > self.xmax0 - 10]
            add(j, -D[j, 0], V[j, 0] - self.xmax0, o['w_hard'], 'xmax')
        # ---- creases: growth capped near concave creases, recession near convex edges
        k = np.nonzero(self.d_cc < o['crease_reach'])[0]
        add(F[k], -1.0, -(o['crease_cap0'] + o['crease_slope'] * self.d_cc[k]), o['w_crease'], 'crease_cc')
        k = np.nonzero(self.d_cv < o['crease_reach'])[0]
        add(F[k], 1.0, -(o['crease_cap0'] + o['crease_slope'] * self.d_cv[k]), o['w_crease'], 'crease_cv')
        # ---- safety: |h| <= h_max
        add(F, 1.0, -o['h_max'], o['w_hard'], 'hmax'); add(F, -1.0, -o['h_max'], o['w_hard'], 'hmax')
        # ---- touch: each zone's closest point gets gap <= touch - margin (one-sided: only pulls when clear).
        #      The pulled vertex is the one with the smallest gap + 3 mm * (1 - window): a free vertex is
        #      preferred over one at the frozen edge that cannot move.
        zinfo = {}
        zw = o['zone_weights'] or {}
        for zn, segs in zones(self.caps).items():
            best = (np.inf, None, None, None)
            gmin = np.inf
            for a, b, r in segs:
                S, _ = seg_samples(a, b, 0.5)
                dd, jj = treeP.query(S, k=64, workers=2)
                g_ = dd - r
                gmin = min(gmin, float(g_[:, 0].min()))
                sc = g_ + 3.0 * (1.0 - self.W[F[jj]])
                i, k = np.unravel_index(np.argmin(sc), sc.shape)
                if sc[i, k] < best[0]:
                    best = (sc[i, k], S[i], F[jj[i, k]], r)
            _, pz, jz, r = best
            g0 = gmin
            zinfo[zn] = (float(g0), jz)
            w = o['w_touch'] * zw.get(zn, 1.0) * (o['middle_pip'] if zn == 'middle PIP' else 1.0)
            if w <= 0:
                continue
            v = pz - P[jz]; d = np.linalg.norm(v)
            a = -float(D[jz] @ v / max(d, 1e-9))    # d(gap)/dh
            U = r + o['touch'] - o['touch_margin']   # want d <= U
            add([jz], -a, d - U - a * h[jz], w, 'touch:' + zn)
            if g0 <= o['touch'] - o['touch_margin']:     # already touching elsewhere: no pull
                Ww[-1] = Ww[-1] * 0.0
        self._nvert_rows = sum(len(x) for x in J)
        J = np.concatenate(J); Aa = np.concatenate(Aa); Bb = np.concatenate(Bb); Ww = np.concatenate(Ww); tags = np.concatenate(tags)
        # never demand more than the window allows: |h_j| <= W_j * h_cap (rows at the frozen edge cannot
        # be met by moving it; they are reported, not chased)
        Bb = np.minimum(Bb, np.abs(Aa) * self.W[J] * o['h_cap'])
        return J, Aa, Bb, Ww, tags, zinfo

    # -------------------------------------------------------------------------------- solve
    def newton(self, G, beta, w, delta, c0):
        """min 0.5 c'Qc + sum_r w_r * huber_hinge(beta_r - G_r c; delta_r)  (semismooth Newton + backtracking).
        huber_hinge(v) = 0 (v <= 0), v^2/2 (0 < v <= delta), delta (v - delta/2) (v > delta)."""
        Q = self.Q
        c = c0.copy()

        def F(c):
            v = np.maximum(beta - G @ c, 0.0)
            pen = np.where(v <= delta, 0.5 * v * v, delta * (v - 0.5 * delta))
            return 0.5 * c @ Q @ c + (w * pen).sum()
        f = F(c)
        state_prev = None
        for it in range(self.opt['newton_iters']):
            v = beta - G @ c
            quad = (v > 0) & (v <= delta)
            lin = v > delta
            state = (quad, lin)
            if state_prev is not None and np.array_equal(quad, state_prev[0]) and np.array_equal(lin, state_prev[1]):
                break
            GA = G[quad]
            H = Q + (GA.T @ sp.diags(w[quad]) @ GA).toarray()
            rhs = GA.T @ (w[quad] * beta[quad]) + G[lin].T @ (w[lin] * delta[lin])
            cn = sla.solve(H, rhs, assume_a='pos')
            step = 1.0
            while step > 1e-3:
                ct = c + step * (cn - c)
                ft = F(ct)
                if ft <= f + 1e-12:
                    break
                step *= 0.5
            c, f = ct, ft
            state_prev = state
        return c, it + 1

    def solve(self, c0=None):
        o = self.opt
        t0 = time.time()
        m = self.B.shape[1]
        c = np.zeros(m) if c0 is None else c0.copy()
        h = self.B @ c
        fac = o['damping']
        for k in range(o['outer_iters']):
            J, Aa, Bb, Ww, tags, zinfo = self.rows(h)
            G = sp.diags(Aa) @ self.B[J]
            nr = self.Grel.shape[0]
            if o['w_relief'] > 0 and nr:
                G = sp.vstack([G, self.Grel, -self.Grel])
                Bb = np.r_[Bb, np.full(nr, -o['relief_max']), np.full(nr, -o['relief_max'])]
                # first block: h - ring mean >= -max (no DENT); second: h - ring mean <= max (no BUMP, weak:
                # up to 2 broad bumps are allowed where a touch needs them)
                Ww = np.r_[Ww, np.full(nr, o['w_relief']), np.full(nr, o['w_relief'] * o['relief_bump'])]
                Aa = np.r_[Aa, np.ones(2 * nr)]
                tags = np.r_[tags, np.full(nr, 'relief:dent', object), np.full(nr, 'relief:bump', object)]
            G = G.tocsr()
            delta = np.where(np.char.startswith(tags.astype(str), 'touch:'), o['touch_huber'], np.inf)
            cn, nit = self.newton(G, Bb, Ww, delta, c)
            hn = self.B @ cn
            if k >= o['damp_from']:
                # cycle breaker: when the step stops shrinking (a row set flipping between two states), halve it
                if len(self.history) >= 2 and self.history[-1]['move'] >= 0.9 * self.history[-2]['move'] and self.history[-1]['move'] > o['tol']:
                    fac = 0.5 * fac
                cn = c + fac * (cn - c)
                hn = self.B @ cn
            dh = np.abs(hn - h).max()
            if dh > o['trust']:
                cn = c + (o['trust'] / dh) * (cn - c)
                hn = self.B @ cn
            move = float(np.abs(hn - h).max())
            c, h = cn, hn
            viol = (G @ c - Bb)
            bad = {}
            for t in np.unique(tags):
                mm = tags == t
                v = -viol[mm].min() if mm.any() else 0.0
                if v > 0.02:
                    bad[t] = round(float(v / max(np.abs(Aa[mm][np.argmin(viol[mm])]), 1e-3)), 2)
            rec = dict(it=k, move=round(move, 3), newton=nit, h=(round(float(h.min()), 2), round(float(h.max()), 2)),
                       zones={z: round(v[0], 2) for z, v in zinfo.items()}, viol=bad, rows=len(J))
            self.history.append(rec)
            self.log(json.dumps(rec))
            if move < o['tol'] and k >= 2:
                break
        self.c, self.h = c, h
        self.seconds = time.time() - t0
        _, _, _, _, _, zinfo = self.rows(h)
        self.zones_final = {z: round(v[0], 3) for z, v in zinfo.items()}
        self.log(f'solve {self.seconds:.1f} s, h {h.min():.2f}..{h.max():.2f}, zones (solver gap) {self.zones_final}')
        return h

    def offset_vectors(self, h=None):
        h = self.h if h is None else h
        off = h[:, None] * self.D
        off[self.V[:, 0] <= 0] = 0.0
        return off
