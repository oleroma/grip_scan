"""dense_fit.py: the smooth hand-fit shape of GripV as ONE signed normal offset h of the dense stage input mesh.

Unknown: h (mm) per vertex of the right handle of the stage input mesh (subdivided GripV basemesh, level 'high');
the new skin is  P = V + h * D  (D = the vertex normal smoothed over 3 mm). The left handle is the mirror image.

    minimise   Bend * |Laplacian h|^2  (thin plate / bi-Laplacian, cotangent Laplacian, lumped mass)
             + Contact Weight * sum over contact-zone axis samples (gap + Press)^2       (soft)
             + soft seat targets in the corridor between R2 and R4
    subject to (one-sided penalties, semismooth Newton, i.e. an active set)
       curvature of the FINAL skin P: on every vertex the acceptance judges, the height-quadric Hessian over a
           3 mm ball (the acceptance's own fit, linearised in h) in 6 directions:
               concave curvature <= 1 / Min Concave Radius,  convex curvature <= 1 / Min Convex Radius
           (or the no-fit value, when that is already tighter: the shape never makes a place worse)
       no hand capsule deeper than Max Dig;  the frozen region h = 0;  y / z / x entry limits;  bbox min y / z;
       the generator's own design features (creases, sharp edges) unchanged |h| <= Feature Lock;
       skin on the R2 / R4 seat planes over the opening + rim (box +-Seat Tol) outside the corridor.
Non-linear parts (nearest skin points, capsule distances, curvature frames) are re-linearised at the current
skin each outer iteration.
"""
import sys
import time
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.spatial import cKDTree

HERE = '/tmp/claude-0/sdf/smooth/final/solver'   # final copy of p_dense/v2/dense_fit.py
sys.path.insert(0, HERE)
sys.path.insert(0, '/tmp/claude-0/sdf/smooth/lib')
import shape_solver as SS   # noqa: E402  (mesh basics, contacts, no-dig bounds: the earlier p_dense solver)
import curvop as CO          # noqa: E402

NOFIT_CURV = '/tmp/claude-0/sdf/smooth/baseline/nofit/curvature.npz'

DEFAULTS2 = dict(
    # ---- the inputs Olex adjusts (refit.py reads them from the modifier's Shape panel)
    press=0.5,                 # 'Press' (general): target sink of the rigid finger at each contact (mm)
    max_dig=0.8,               # 'Max Dig': no capsule deeper than this (acceptance 1.0; margin for the remesh)
    min_concave_radius=14.0,   # 'Min Concave Radius': no groove tighter than this on the edited skin (accept 12 / 8)
    min_convex_radius=4.0,     # 'Min Convex Radius': no ridge sharper than this (accept 3)
    contact_weight=1.0,        # 'Contact Weight': how hard the contacts pull (multiplier)
    seat_rim=4.0,              # 'Seat Rim': flush rim around each trigger opening (acceptance 2.5 + 1.5 margin: the
                               # acceptance projects along the plane normal, the solve lands along D)
    # ---- internal
    curv_weight=3e4,           # penalty weight of the curvature limits (per mm2 of skin, per (1/mm)^2)
    curv_dirs=6,               # directions per vertex (30 deg steps)
    curv_radius=3.0,
    curv_base_margin=0.003,    # a vertex already tighter in the no-fit may stay so (+ this, 1/mm)
    curv_region_xmin=21.0, curv_region_zmax=40.6,
    box_weight=3e4,            # penalty weight of the box limits (no dig, seats, features, entry)
    feature_lock=0.12,         # |h| <= this on the no-fit design features (dilated 2 mm, as the acceptance)
    feature_seat_clear=5.0,    # features this close to a seat platform are not locked (the seat reshapes them)
    unlock_features_near=None, # [(point, radius)]: also unlock features there
    seat_tol=0.12,             # seat box half-width (mm, along the plane normal)
    seat_corridor=9.0,         # rim points closer than this to the OTHER opening outline: soft target
    seat_corridor_w=0.3,       # weight (per mm2) of the corridor targets
    seat_core=4.0,
    contact_sigma=1.5,         # contact rows act on h smoothed over this (mm): no spikes under a contact foot
    lock_wins=True,            # a locked feature inside a seat rim stays locked (the rim deviates there)
    newton_iters=30, newton_tol=1e-4, newton_verbose=False,
    trust=2.0,                 # max change of h per outer iteration (mm): keeps the linearisation valid
    curv_schedule=(0.01, 0.1, 0.3, 1.0),   # curvature weight multipliers of the first outer iterations (then 1)
    outer_iters=10, damping=0.7, damp_from=2, tol=0.02,
    pcg_tol=1e-9, pcg_max=25,
    slope_max=0.8,             # |grad h| limit (edge differences): steeper growth breaks the offset (folds, cusps)
    slope_weight=3e3,          # penalty weight of the slope limit (per edge, per mm^2 of excess)
    true_search=True,          # outer step length chosen on the true curvature of the moved skin
    prox=0.0,                  # proximal weight (per mm2) on the change of h in one outer iteration
)


def _pcg(Aop, b, x0, Minv, tol, maxit):
    x = x0.copy()
    r = b - Aop(x)
    nb = max(np.linalg.norm(b), 1e-30)
    z = Minv(r)
    p = z.copy()
    rz = r @ z
    for k in range(maxit):
        if np.linalg.norm(r) / nb < tol:
            return x, k, True
        Ap = Aop(p)
        al = rz / (p @ Ap)
        x += al * p
        r -= al * Ap
        z = Minv(r)
        rz2 = r @ z
        p = z + (rz2 / rz) * p
        rz = rz2
    return x, maxit, np.linalg.norm(r) / nb < tol


class Fit(SS.Problem):
    def __init__(self, V, T, caps, zones, seats, keepouts=None, opt=None):
        o = dict(SS.DEFAULTS)
        o.update(DEFAULTS2)
        o['seat_feature_soft'] = False
        o.update(opt or {})
        super().__init__(V, T, caps, zones, seats, keepouts, o)
        o = self.opt
        t0 = time.time()
        V = self.V
        n = len(V)
        # ---- design features of the no-fit skin, exactly as the acceptance defines them (dilated 2 mm)
        z = np.load(NOFIT_CURV)
        fb = z['mask'] & ((z['k2'] < -1 / 12.0) | (z['k1'] > 1 / 3.0) | (z['kink_concave'] > 10) | (z['kink_convex'] > 25))
        Vb = z['V']
        tb = cKDTree(Vb)
        fd = np.zeros(len(Vb), bool)
        for lst in tb.query_ball_point(Vb[np.nonzero(fb)[0]], 2.0, workers=2):
            fd[lst] = True
        _, jn = tb.query(V, workers=2)
        self.feat = fd[jn] & (V[:, 0] > 0)
        # ---- curvature-controlled vertices (right side, where the acceptance can judge)
        self.CS = np.nonzero((V[:, 0] >= o['curv_region_xmin']) & (V[:, 2] < o['curv_region_zmax']))[0]
        # base curvature (input mesh) for the 'never worse than the no-fit' limits
        G0, H0, *_ = CO.quadric_operator(V, self.T, self.CS, self.D, radius=o['curv_radius'])
        self.lmax0, self.lmin0, _ = CO.principal(H0)
        CO_log(o, f'features {int(self.feat.sum())} verts, curvature set {len(self.CS)} ({time.time() - t0:.1f} s)')

    def smoother(self, sigma):
        """Row-normalised Gaussian kernel matrix S (n x n) over vertices within 2.5 sigma on the same sheet."""
        V, N = self.V, self.N
        tree = cKDTree(V)
        dd, jj = tree.query(V, k=96, distance_upper_bound=2.5 * sigma, workers=2)
        ok = np.isfinite(dd)
        jj = np.where(ok, jj, 0)
        w = np.exp(-0.5 * (np.where(ok, dd, 0) / sigma) ** 2) * ok
        w = w * (np.einsum('nkc,nc->nk', N[jj], N) > 0.5)
        w = w / w.sum(1, keepdims=True)
        rows = np.repeat(np.arange(len(V)), jj.shape[1])
        S = sp.coo_matrix((w.ravel(), (rows, jj.ravel())), shape=(len(V), len(V))).tocsr()
        S.eliminate_zeros()
        return S

    def contact_rows(self, h, P, tree):
        A, tgt, w, info = super().contact_rows(h, P, tree)
        sg = self.opt.get('contact_sigma', 0.0)
        if sg > 0:
            if getattr(self, '_Sc', None) is None:
                self._Sc = self.smoother(sg)
            A2 = (A @ self._Sc).tocsr()
            tgt = tgt - A @ h + A2 @ h
            A = A2
        return A, tgt, w, info

    # ------------------------------------------------------------------------------- seats
    def seat_boxes(self):
        """Per vertex: seat target hv (landing on the plane along D), kind (1 hard box, 2 soft corridor)."""
        o = self.opt
        V, N = self.V, self.D
        names = list(self.seats)
        per = {}
        for name, cfg in self.seats.items():
            c, nn, S = cfg['center'], cfg['normal'], cfg['segments']
            s0 = c - o['seat_proud'] * nn
            e1 = np.cross(nn, [0.0, 0.0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(nn, e1)
            cosv = N @ nn
            hv = np.where(cosv > 1e-3, -((V - s0) @ nn) / np.maximum(cosv, 1e-3), np.inf)
            X = V + np.where(np.isfinite(hv), hv, 0.0)[:, None] * N
            rho = np.linalg.norm(X - s0, axis=1)
            q = S.reshape(-1, 3) - s0
            ext = float(np.linalg.norm(q - np.outer(q @ nn, nn), axis=1).max())
            cand = np.nonzero((np.abs(hv) < o['seat_window']) & (rho < ext + o['seat_rim'] + 1.0)
                              & (cosv > o['seat_cos_min']) & ~self.fixed0)[0]
            Q2 = np.c_[(X[cand] - s0) @ e1, (X[cand] - s0) @ e2]
            S2 = np.stack([np.c_[(S[:, k] - s0) @ e1, (S[:, k] - s0) @ e2] for k in (0, 1)], axis=1)
            ins = SS.inside_polygon_2d(Q2, S2)
            dop = SS.dist_to_segments(X[cand], S)
            m = ins | (dop <= o['seat_rim'])
            sel = cand[m]
            inop = np.zeros(len(V), bool)
            inop[cand[ins | (dop <= 0.75)]] = True
            cfg['_in_opening'] = inop
            cfg['_ext'] = ext
            per[name] = dict(sel=sel, hv=hv[sel], X=X[sel], core=rho[sel] <= o['seat_core'])
        owner = {}
        for name, d in per.items():
            for i, v in zip(d['sel'], d['hv']):
                if i not in owner or abs(v) < abs(owner[i][1]):
                    owner[i] = (name, v)
        tgt = np.full(len(V), np.nan)
        kind = np.zeros(len(V), np.int8)
        which = np.full(len(V), -1, np.int8)
        for si, (name, d) in enumerate(per.items()):
            keep = np.array([owner[i][0] == name for i in d['sel']], bool) if len(d['sel']) else np.zeros(0, bool)
            sel, hv, X, core = d['sel'][keep], d['hv'][keep], d['X'][keep], d['core'][keep]
            others = [m for m in names if m != name]
            if others:
                d_other = np.min([SS.dist_to_segments(X, self.seats[m]['segments']) for m in others], axis=0)
                corr = (d_other < o['seat_corridor']) & ~core
            else:
                corr = np.zeros(len(sel), bool)
            tgt[sel] = hv
            kind[sel] = np.where(corr, 2, 1)
            which[sel] = si
            self.seats[name]['_n'] = (int((~corr).sum()), int(corr.sum()))
        self.seat_tgt, self.seat_kind, self.seat_which = tgt, kind, which
        return tgt, kind

    # ------------------------------------------------------------------------------- solve
    def solve(self, h0=None):
        o = self.opt
        t0 = time.time()
        n = len(self.V)
        V = self.V
        for cfg in self.seats.values():
            Vc, Tc = cfg['cutter']
            cfg['_cutter_bvh'] = SS.bvh(Vc, SS.orient(Vc, Tc))
        self._ko_bvh = {k: SS.bvh(Vk, SS.orient(Vk, Tk)) for k, (Vk, Tk, need) in self.keepouts.items()}
        self._wall0 = {}
        self._seat_cache = None
        self.feat_keep_mask = np.zeros(n, bool)
        stgt, skind = self.seat_boxes() if self.seats else (np.full(n, np.nan), np.zeros(n, np.int8))
        # features locked unless near a seat platform (or unlocked by the caller)
        near = np.zeros(n, bool)
        for cfg in self.seats.values():
            c = cfg['center']
            near |= np.linalg.norm(V - c, axis=1) < cfg['_ext'] + o['seat_rim'] + o['feature_seat_clear']
        for p, r in (o['unlock_features_near'] or []):
            near |= np.linalg.norm(V - np.asarray(p), axis=1) < r
        if o['lock_wins']:
            lock = self.feat & ~self.fixed0
        else:
            lock = self.feat & ~near & ~self.fixed0
        self.lock = lock
        F = self.F
        nF = len(F)
        iF = -np.ones(n, int)
        iF[F] = np.arange(nF)
        # curvature rows only where not locked (locked features are 'unchanged' and excluded by the acceptance)
        cs = self.CS[~lock[self.CS]]
        self.cs = cs
        csi = np.searchsorted(self.CS, cs)
        kcc = np.maximum(1.0 / o['min_concave_radius'], self.lmax0[csi] + o['curv_base_margin'])
        kcv = np.maximum(1.0 / o['min_convex_radius'], -self.lmin0[csi] + o['curv_base_margin'])
        self.kcc, self.kcv = kcc, kcv
        nd = o['curv_dirs']
        th = np.arange(nd) * np.pi / nd
        cc, cs_, ss = np.cos(th) ** 2, 2 * np.cos(th) * np.sin(th), np.sin(th) ** 2      # e'He = c2 Huu + 2cs Huv + s2 Hvv
        wv = o['curv_weight'] * self.m[cs]
        sm = o['smoothness']
        Qb = ((o['bend'] * sm) * self.K + (o['membrane'] * sm) * self.L + o['tension'] * self.Mass)[F][:, F].tocsr()
        # slope rows: (h_i - h_j) / |e_ij| for edges with at least one free end (fixed ends are h = 0)
        E = np.unique(np.sort(np.r_[self.T[:, [0, 1]], self.T[:, [1, 2]], self.T[:, [2, 0]]], axis=1), axis=0)
        E = E[(iF[E[:, 0]] >= 0) | (iF[E[:, 1]] >= 0)]
        el = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
        ne = len(E)
        rr = np.r_[np.arange(ne), np.arange(ne)]
        cc_ = np.r_[iF[E[:, 0]], iF[E[:, 1]]]
        vv = np.r_[1.0 / el, -1.0 / el]
        okc = cc_ >= 0
        Bs = sp.coo_matrix((vv[okc], (rr[okc], cc_[okc])), shape=(ne, nF)).tocsr()
        ws = o['slope_weight'] * el            # per edge (length-weighted)
        smax = o['slope_max']
        h = np.zeros(n) if h0 is None else h0.copy()
        hist = []
        self._lu = None
        for it in range(o['outer_iters']):
            ti = time.time()
            P = self.moved(h)
            tree = SS.bvh(P, self.T)
            A, tgt, w, cinfo = self.contact_rows(h, P, tree)
            lo, hi = self.bounds(h, P)
            # seats: hard box around the landing value (the no-dig bound wins), soft rows in the corridor
            hs = skind == 1
            hi_s = np.where(hs, np.minimum(hi, stgt + o['seat_tol']), hi)
            lo_s = np.where(hs, np.maximum(lo, np.minimum(stgt - o['seat_tol'], hi_s)), lo)
            conflict = int((hs & (stgt - o['seat_tol'] > hi)).sum())
            hi, lo = hi_s, lo_s
            # locked features
            hi = np.where(lock, np.minimum(hi, o['feature_lock']), hi)
            lo = np.where(lock, np.maximum(lo, -o['feature_lock']), lo)
            lo = np.minimum(lo, hi)                      # inconsistent: the upper bound (no dig) wins
            # trust region around the current h (the nearest point of it when the bounds lie outside)
            tr = o['trust']
            down = hi < h - tr
            up = lo > h + tr
            lo_t, hi_t = np.maximum(lo, h - tr), np.minimum(hi, h + tr)
            lo = np.where(down, h - tr, np.where(up, h + tr, lo_t))
            hi = np.where(down, h - tr, np.where(up, h + tr, hi_t))
            soft = np.nonzero(skind == 2)[0]
            if len(soft):
                As = sp.coo_matrix((np.ones(len(soft)), (np.arange(len(soft)), soft)), shape=(len(soft), n)).tocsr()
                A = sp.vstack([A, As]).tocsr()
                tgt = np.r_[tgt, stgt[soft]]
                w = np.r_[w, o['seat_corridor_w'] * self.m[soft]]
            AF = A[:, F]
            Qs = (Qb + AF.T @ sp.diags(w) @ AF).tocsr()
            bs = AF.T @ (w * tgt)
            if o['prox'] > 0:
                pw = o['prox'] * self.m[F]
                Qs = (Qs + sp.diags(pw)).tocsr()
                bs = bs + pw * h[F]
            # curvature operator at the current skin
            G, Hc, *_ = CO.quadric_operator(P, self.T, cs, self.D, radius=o['curv_radius'])
            GF = G[:, F].tocsr()
            Guu, Guv, Gvv = GF[0::3], GF[1::3], GF[2::3]
            hF = h[F]
            Hoff = Hc - (GF @ hF).reshape(-1, 3)        # values(x) = Hoff + (GF x)
            loF, hiF = lo[F], hi[F]
            x = hF.copy()
            # the true curvature of the current skin vs the limits (area share of violating vertices)
            Hn = Hc[:, 0:1] * cc[None] + Hc[:, 1:2] * cs_[None] + Hc[:, 2:3] * ss[None]
            ecc = (Hn - kcc[:, None]).max(1)
            ecv = (-kcv[:, None] - Hn).max(1)
            mcs = self.m[cs]
            true_stats = dict(cc_area_pct=round(100 * float(mcs[ecc > 0].sum() / mcs.sum()), 2),
                              cc_p99=round(float(np.percentile(ecc, 99)), 4), cc_max=round(float(ecc.max()), 3),
                              cv_area_pct=round(100 * float(mcs[ecv > 0].sum() / mcs.sum()), 2), cv_max=round(float(ecv.max()), 3))
            sch = o['curv_schedule']
            wcur = wv * (sch[it] if it < len(sch) else 1.0)

            def curv_vals(xx):
                Hx = Hoff + (GF @ xx).reshape(-1, 3)
                vals = Hx[:, 0:1] * cc[None] + Hx[:, 1:2] * cs_[None] + Hx[:, 2:3] * ss[None]   # (ncs, nd)
                return vals

            def fval(xx):
                vals = curv_vals(xx)
                rc = np.maximum(vals - kcc[:, None], 0.0)
                rv = np.maximum(-kcv[:, None] - vals, 0.0)
                pen = 0.5 * (wcur[:, None] * (rc * rc + rv * rv)).sum()
                bx = 0.5 * o['box_weight'] * (np.maximum(xx - hiF, 0) ** 2 + np.maximum(loF - xx, 0) ** 2).sum()
                g = Bs @ xx
                sl = 0.5 * (ws * (np.maximum(g - smax, 0) ** 2 + np.maximum(-smax - g, 0) ** 2)).sum()
                return 0.5 * xx @ (Qs @ xx) - bs @ xx + pen + bx + sl

            f = fval(x)
            nact_prev = None
            for nit in range(o['newton_iters']):
                vals = curv_vals(x)
                acc = vals > kcc[:, None]
                acv = vals < -kcv[:, None]
                ka, da = np.nonzero(acc | acv)
                sg = np.where(acc[ka, da], 1.0, -1.0)
                lim = np.where(sg > 0, kcc[ka], kcv[ka])
                # rows: sg * (c2 Guu + cs Guv + s2 Hvv)[ka] ; offsets sg * (Hoff . dir) - lim
                Rr = (sp.diags(sg * cc[da]) @ Guu[ka] + sp.diags(sg * cs_[da]) @ Guv[ka] + sp.diags(sg * ss[da]) @ Gvv[ka]).tocsr()
                off = sg * (Hoff[ka, 0] * cc[da] + Hoff[ka, 1] * cs_[da] + Hoff[ka, 2] * ss[da]) - lim
                wr = wcur[ka]
                ah = x > hiF
                al = x < loF
                dB = o['box_weight'] * (ah.astype(float) + al)
                g = Bs @ x
                sa = np.nonzero(np.abs(g) > smax)[0]
                Ba = Bs[sa]
                sgn = np.sign(g[sa])
                Mx = (Qs + Rr.T @ sp.diags(wr) @ Rr + sp.diags(dB) + Ba.T @ sp.diags(ws[sa]) @ Ba).tocsr()
                rhs = bs - Rr.T @ (wr * off) + o['box_weight'] * (ah * hiF + al * loF) + Ba.T @ (ws[sa] * sgn * smax)
                # solve: PCG preconditioned by the last factorisation; refactor when it struggles
                ok = False
                sig = (len(ka), int(ah.sum() + al.sum()) + len(sa))
                same_ish = nact_prev is not None and abs(sig[0] - nact_prev[0]) + abs(sig[1] - nact_prev[1]) < 0.03 * (sum(sig) + 1)
                if self._lu is not None and self._lu_n == nF and same_ish:
                    xn, k, ok = _pcg(lambda v: Mx @ v, rhs, x, self._lu.solve, o['pcg_tol'], o['pcg_max'])
                if not ok:
                    self._lu = spl.splu(Mx.tocsc(), permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                                        options=dict(SymmetricMode=True))
                    self._lu_n = nF
                    xn = self._lu.solve(rhs)
                    k = -1
                # line search on the true (piecewise quadratic) objective
                dx = xn - x
                if o['newton_verbose']:
                    res = np.linalg.norm(Mx @ xn - rhs) / max(np.linalg.norm(rhs), 1e-30)
                    gr = Mx @ x - rhs
                    eps = 1e-6
                    fd = (fval(x + eps * dx) - fval(x - eps * dx)) / (2 * eps)
                    CO_log(o, f'      solve rel.res {res:.2e}  g.dx {gr @ dx:.4g}  fd {fd:.4g}')
                tstep = 1.0
                for _ in range(12):
                    fn = fval(x + tstep * dx)
                    if fn <= f + 1e-12 * abs(f):
                        break
                    tstep *= 0.5
                else:
                    tstep, fn = 0.0, f
                x = x + tstep * dx
                fprev, f = f, fn
                nact = (len(ka), int(ah.sum() + al.sum()) + len(sa))
                if o['newton_verbose']:
                    CO_log(o, f'      newton {nit}: f {f:.6g} step {tstep:.3f} act {nact} pcg {k} |dx| {np.abs(dx).max():.3f}')
                if abs(fprev - f) <= o['newton_tol'] * max(abs(f), 1.0) and nact == nact_prev:
                    break
                nact_prev = nact
            hn = np.zeros(n)
            hn[F] = x
            if it >= o['damp_from']:
                hn = h + o['damping'] * (hn - h)
            if o['true_search']:
                # the model's smooth part + box/slope at x, but the curvature penalty from the true moved skin
                def true_obj(hh):
                    xx = hh[F]
                    Gt, Ht, *_ = CO.quadric_operator(self.moved(hh), self.T, cs, self.D, radius=o['curv_radius'])
                    vals = Ht[:, 0:1] * cc[None] + Ht[:, 1:2] * cs_[None] + Ht[:, 2:3] * ss[None]
                    rc = np.maximum(vals - kcc[:, None], 0.0); rv = np.maximum(-kcv[:, None] - vals, 0.0)
                    pen = 0.5 * (wcur[:, None] * (rc * rc + rv * rv)).sum()
                    bx = 0.5 * o['box_weight'] * (np.maximum(xx - hiF, 0) ** 2 + np.maximum(loF - xx, 0) ** 2).sum()
                    g = Bs @ xx
                    sl = 0.5 * (ws * (np.maximum(g - smax, 0) ** 2 + np.maximum(-smax - g, 0) ** 2)).sum()
                    return 0.5 * xx @ (Qs @ xx) - bs @ xx + pen + bx + sl
                j0 = true_obj(h)
                best = (j0, 0.0, h)
                for tt in (1.0, 0.5, 0.25):
                    hc = h + tt * (hn - h)
                    jt = true_obj(hc)
                    if jt < best[0]:
                        best = (jt, tt, hc)
                    if jt < j0 and tt == 1.0:
                        break
                if best[1] == 0.0:
                    best = (None, 0.1, h + 0.1 * (hn - h))
                CO_log(o, f'   true search: step {best[1]}  J {j0:.4g} -> {best[0] if best[0] is None else round(best[0], 1)}')
                hn = best[2]
            dh = float(np.abs(hn - h).max())
            # diagnostics: curvature violations (linear model) at the new x
            vals = curv_vals(x)
            vcc = float((vals - kcc[:, None]).max())
            vcv = float((-kcv[:, None] - vals).max())
            h = hn
            rec = dict(it=it, dh=round(dh, 4), seconds=round(time.time() - ti, 1), newton=nit + 1,
                       active_curv=int(len(ka)), active_box=int(ah.sum() + al.sum()), seat_conflicts=conflict,
                       max_concave_excess=round(vcc, 4), max_convex_excess=round(vcv, 4),
                       contacts={k: (round(v['penetration'], 2), round(v['min_gap'], 2), round(v['support'], 1)) for k, v in cinfo.items()},
                       h_range=(round(float(h.min()), 2), round(float(h.max()), 2)))
            hist.append(rec)
            rec['true_at_start'] = true_stats
            CO_log(o, f"   skin now: concave excess area {true_stats['cc_area_pct']}% p99 {true_stats['cc_p99']} max {true_stats['cc_max']} | convex area {true_stats['cv_area_pct']}% max {true_stats['cv_max']}")
            CO_log(o, '   contacts ' + ' '.join(f"{k.split()[0][:2]}{k.split()[1][:3]} {v['penetration']:.2f}/{v['support']:.0f}" for k, v in cinfo.items()))
            CO_log(o, f"it {it}: dh {dh:.3f} h {h.min():.2f}..{h.max():.2f} newton {nit + 1} act curv {len(ka)} box {rec['active_box']} "
                      f"excess cc {vcc:+.4f} cv {vcv:+.4f} seat-dig conflicts {conflict}  {time.time() - ti:.1f} s")
            if dh < o['tol'] and it > 1:
                break
        self.h = h
        self.history = hist
        self.seconds = time.time() - t0
        CO_log(o, f'solve {self.seconds:.1f} s')
        return h


def CO_log(o, *a):
    if o.get('verbose', True):
        print('[fit]', *a, flush=True)
