"""relief.py (final/lib_v2, from v2/lib): SPEC v2 acceptance of a stage skin (v2 addition to the shared metrics).

    PY=/tmp/claude-0/venv/bin/python
    $PY relief.py -- --run RUN_DIR --out OUT.json [--label NAME] [--inputs INPUTS.npz]
                     [--plain RUN] [--nofit RUN] [--nofit-parts RUN] [--give-palm 3.5] [--give-finger 1.5]

RUN_DIR is an eval_all run (stage.npz, caps.json, internals.npz, parts/case_bottom.npz ...).
Measured, all on the right handle (x >= 25, z <= 38.5); every inside/outside test by ray parity
(vlib.parity_inside, 5 skew rays, majority), never by nearest-face normals.

1. RELIEF (primary)
   offset h = signed distance of each skin vertex to the no-fit skin (+ grown, - receded);
   judged region = |h| > 0.2 mm dilated 3 mm, z <= 38 (as the independent verifier, v_fit/analyze_skin.py);
   extrema = local max / min of h (7 mm neighbourhood) whose prominence against the median of h on the
   12-16 mm ring is >= 1 mm (v_fit/extrema.py), deduplicated within 6 mm.
   ACCEPT: 0 dents, <= 2 bumps.
2. CURVATURE at 6 mm scale (normal-difference shape operator, v_fit/vlib.shape_operator, validated on a
   sphere r20 / cylinder r10 / groove r8): % area concave R < 20 and convex R < 8 on the judged region vs
   plain GripV on the same footprint. ACCEPT: ratio <= 1.5 each. Also the whole handle.
3. CREASES (SPEC v2 item 5): at every no-fit vertex with concave R < 12 (or convex R < 5) at 6 mm scale,
   the candidate's radius at the same place / the no-fit radius (min, p5, count < 0.8).
4. CONTACTS: per zone (gripmetrics.zones) the min gap (sd(axis) - r; < 0 = overlap). Touch when gap <= 0.3.
   Per capsule the max overlap on editable skin vs its soft-tissue allowance (Palm Give for palm,
   metacarpals and the pinky; Finger Give for the other phalanges; fingertips on R2 / R4 are judged by
   the stroke check instead).
5. KEEP-OUTS: trigger stroke (middle / ring distal capsule moved along the trigger travel by travel_mm,
   gap to case_bottom with the openings cut, accept >= 0.3 mm), screw-channel wall (distance from the
   'screw bools' surface to the skin where plain GripV has >= 3 mm, accept >= 3 mm).
6. INVARIANTS (cheap own checks): frozen-region change vs no-fit, mirror symmetry, max Y, top face, bbox,
   case_top / case_middle_plate deviation vs the no-fit parts.
"""
import os
import sys
import json
import time
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as L          # noqa: E402
from mathutils import Vector  # noqa: E402

PLAIN = '/tmp/claude-0/sdf/smooth/final/runs/base_Grip_smooth_high_low'
NOFIT = '/tmp/claude-0/sdf/smooth/final/runs/nofit_fix'        # Fit 0, repaired (outward) stage
NOFIT_PARTS = '/tmp/claude-0/sdf/smooth/final/runs/nofit_fix'
INPUTS = '/tmp/claude-0/sdf/smooth/final/solve/inputs_Grip_smooth.npz'   # written by final/refit.py (trigger travel)
FING3 = ('middle', 'ring', 'pinky')
FINGER_GIVE = ('middle_proximal', 'middle_middle', 'ring_proximal', 'ring_middle')
PALM_GIVE = ('palm', 'middle_metacarpal', 'ring_metacarpal', 'pinky_metacarpal', 'pinky_proximal', 'pinky_middle', 'pinky_distal')
STROKE = {'R2': 'middle_distal', 'R4': 'ring_distal'}
TOUCH = 0.3


def frozen(Q):
    return (np.abs(Q[:, 0]) < 25) | (Q[:, 2] > 38) | (Q[:, 1] > -4.658)


def region(X):
    return (X[:, 0] >= 25) & (X[:, 2] <= 38.5)


def load(run):
    V, T = L.load_stage(os.path.join(run, 'stage.npz'))
    return V, L.orient_outward(V, T)


def zones(caps):
    """gripmetrics.zones (axis sub-segments) as {zone: [(name, t0, t1)]}."""
    Z = {}
    for f in FING3:
        Z[f + ' root pad'] = [(f + '_metacarpal', 0.8, 1.0), (f + '_proximal', 0.0, 0.35)]
        Z[f + ' PIP'] = [(f + '_middle', 0.0, 0.4)]
    Z['pinky distal'] = [('pinky_distal', 0.0, 1.0)]
    Z['palm on swell'] = [(f + '_metacarpal', 0.0, 1.0) for f in FING3]
    return Z


def stats(k1, k2, Ar, sel):
    w = Ar[sel]; W = max(w.sum(), 1e-9)
    cc = np.where(k2[sel] < 0, -1 / np.minimum(k2[sel], -1e-9), np.inf)
    cv = np.where(k1[sel] > 0, 1 / np.maximum(k1[sel], 1e-9), np.inf)
    o = dict(area_mm2=round(float(w.sum()), 1))
    for r in (8, 12, 20):
        o[f'pct_concave_R<{r}'] = round(100 * w[cc < r].sum() / W, 2)
    for r in (3, 5, 8):
        o[f'pct_convex_R<{r}'] = round(100 * w[cv < r].sum() / W, 2)
    o['min_concave_R'] = round(float(cc.min()), 2) if len(cc) else None
    o['min_convex_R'] = round(float(cv.min()), 2) if len(cv) else None
    return o


def curv6(V, N, idx, R=6.0):
    tr = cKDTree(V)
    outs = [L.shape_operator(V, N, idx[c:c + 4000], R, tr) for c in range(0, len(idx), 4000)]
    return np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs])


def extrema(P, N, h, judge, caps, thr=1.0):
    tr = cKDTree(P)

    def capname(p):
        best = (1e9, '')
        for n, (a, b, r) in caps.items():
            ab = b - a; t = np.clip((p - a) @ ab / (ab @ ab), 0, 1)
            d = np.linalg.norm(p - (a + t * ab)) - r
            if d < best[0]:
                best = (d, n)
        return f'{best[1]} {best[0]:+.1f}'
    res = []
    cand = np.where(judge)[0]          # every judged vertex (the verifier sampled every 2nd one, which can
    for sign, lab in ((1, 'bump'), (-1, 'dent')):   # miss the single top vertex of a smooth broad bump)
        for i in cand:
            js = np.asarray(tr.query_ball_point(P[i], 7.0)); js = js[N[js] @ N[i] > 0.3]
            if sign > 0 and h[i] < h[js].max() - 0.01:
                continue
            if sign < 0 and h[i] > h[js].min() + 0.01:
                continue
            ring = np.asarray(tr.query_ball_point(P[i], 16.0)); ring = ring[N[ring] @ N[i] > 0.0]
            far = ring[np.linalg.norm(P[ring] - P[i], axis=1) > 12.0]
            if len(far) < 10:
                continue
            prom = sign * (h[i] - np.median(h[far]))
            if prom >= thr:
                res.append(dict(kind=lab, at=np.round(P[i], 1).tolist(), h=round(float(h[i]), 2),
                                prominence=round(float(prom), 2), nearest_capsule=capname(P[i]),
                                rim=bool(abs(h[i]) < 0.2)))   # an unedited point beside an edit (edge of the change)
    out = []
    for r in sorted(res, key=lambda r: -r['prominence']):
        if all(np.linalg.norm(np.array(r['at']) - np.array(o['at'])) > 6 for o in out):
            out.append(r)
    return out


def seg_samples(a, b, step=0.25):
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
    t = np.linspace(0, 1, n)
    return a[None] + t[:, None] * (b - a)[None], t


def capsule_gaps(tree, caps, names=None, shift=None):
    out = {}
    for name, (a, b, r) in caps.items():
        if names and name not in names:
            continue
        P, t = seg_samples(a, b)
        if shift is not None:
            P = P + shift
        sd, Q = L.signed_dist(tree, P)
        out[name] = (P, t, sd - r, Q)
    return out


def sample_surface(V, T, step=0.6, seed=0):
    A = 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
    n = int(A.sum() / (step * step)) + 1
    rng = np.random.default_rng(seed)
    f = rng.choice(len(T), n, p=A / A.sum())
    u, v = rng.random(n), rng.random(n)
    s = u + v > 1; u[s], v[s] = 1 - u[s], 1 - v[s]
    return V[T[f, 0]] + u[:, None] * (V[T[f, 1]] - V[T[f, 0]]) + v[:, None] * (V[T[f, 2]] - V[T[f, 0]])


def screw_walls(tree_c, tree_p, Ps):
    sdc, _ = L.signed_dist(tree_c, Ps)
    sdp, _ = L.signed_dist(tree_p, Ps)
    wc, wp = -sdc, -sdp                     # > 0: screw surface point inside the case, wall = distance to skin
    judged = wp >= 3.0
    bands = {}
    for z0 in range(-16, 38, 4):
        m = judged & (Ps[:, 2] >= z0) & (Ps[:, 2] < z0 + 4)
        if m.any():
            bands[f'z{z0}..{z0 + 4}'] = [round(float(wc[m].min()), 2), round(float(wp[m].min()), 2)]
    i = int(np.argmin(np.where(judged, wc, np.inf)))
    return dict(min_wall_mm=round(float(wc[i]), 2), at=np.round(Ps[i], 1).tolist(), plain_at_same=round(float(wp[i]), 2),
                n_judged=int(judged.sum()), bands_candidate_vs_plain=bands, accept=bool(wc[i] >= 3.0 - 0.02))   # 0.02: the sampled wall of plain itself reads 3.00


def mesh_dev(Va, Ta, Vb, Tb):
    tb = L.bvh(Vb, Tb)
    d = np.array([tb.find_nearest(Vector(p))[3] for p in Va])
    i = int(np.argmax(d))
    return round(float(d[i]), 3), np.round(Va[i], 1).tolist(), int((d > 0.05).sum())


def analyze(run, out, label='', inputs=INPUTS, plain=PLAIN, nofit=NOFIT, nofit_parts=NOFIT_PARTS,
            give_palm=3.5, give_finger=1.5, quick=False):
    t0 = time.time()
    rep = dict(label=label, run=run)
    V, T = load(run); Vn, Tn = load(nofit); Vb, Tb = load(plain)
    caps = {k: (np.array(v['a']), np.array(v['b']), float(v['r'])) for k, v in json.load(open(os.path.join(run, 'caps.json'))).items()}
    N = L.vertex_normals(V, T); A = L.vertex_area(V, T)
    Nn = L.vertex_normals(Vn, Tn); An = L.vertex_area(Vn, Tn)
    Nb = L.vertex_normals(Vb, Tb); Ab = L.vertex_area(Vb, Tb)
    idx, idn, idb = np.where(region(V))[0], np.where(region(Vn))[0], np.where(region(Vb))[0]
    tn = L.bvh(Vn, Tn); tc = L.bvh(V, T); tb = L.bvh(Vb, Tb)
    # ---- 1. offset vs no-fit and relief extrema
    h, _ = L.signed_dist(tn, V[idx])
    tree = cKDTree(V)
    ed = np.abs(h) > 0.2
    dil = np.zeros(len(V), bool)
    if ed.any():
        for js in tree.query_ball_point(V[idx[ed]], 3.0):
            dil[js] = True
    judge = dil[idx] & (V[idx, 2] <= 38.0)
    w = A[idx]
    rep['offset'] = dict(grow_max=round(float(h.max()), 2), recede_max=round(float(-h.min()), 2),
                         edited_area_mm2=round(float(w[ed].sum()), 1), judged_area_mm2=round(float(w[judge].sum()), 1),
                         grow_at=np.round(V[idx][np.argmax(h)], 1).tolist(), recede_at=np.round(V[idx][np.argmin(h)], 1).tolist())
    ex = extrema(V[idx], N[idx], h, judge, caps) if judge.any() else []
    nb, nd = sum(e['kind'] == 'bump' for e in ex), sum(e['kind'] == 'dent' for e in ex)
    nbr, ndr = sum(e['kind'] == 'bump' and not e['rim'] for e in ex), sum(e['kind'] == 'dent' and not e['rim'] for e in ex)
    rep['relief'] = dict(bumps=nb, dents=nd, bumps_edited=nbr, dents_edited=ndr, extrema=ex, accept=bool(nd == 0 and nb <= 2))
    print(f'[relief] {label}: offset {h.min():.2f}..{h.max():.2f}, bumps {nb} dents {nd} ({time.time() - t0:.0f} s)', flush=True)
    # ---- 2. curvature at 6 mm, judged footprint vs plain / no-fit on the same footprint
    if not quick:
        k1, k2 = curv6(V, N, idx)
        k1n, k2n = curv6(Vn, Nn, idn)
        k1b, k2b = curv6(Vb, Nb, idb)
        m = np.zeros(len(V), bool); m[idx[judge]] = True

        def foot(Vx, ix):
            d, j = tree.query(Vx[ix]); return m[j] & (d < 6)
        fn, fb = foot(Vn, idn), foot(Vb, idb)
        c = dict(candidate=stats(k1, k2, A[idx], judge & np.isfinite(k2)),
                 plain_same_footprint=stats(k1b, k2b, Ab[idb], fb & np.isfinite(k2b)),
                 nofit_same_footprint=stats(k1n, k2n, An[idn], fn & np.isfinite(k2n)),
                 candidate_whole_handle=stats(k1, k2, A[idx], np.isfinite(k2) & (V[idx, 2] <= 38)),
                 plain_whole_handle=stats(k1b, k2b, Ab[idb], np.isfinite(k2b) & (Vb[idb, 2] <= 38)))
        cp, pp = c['candidate'], c['plain_same_footprint']
        rc = cp['pct_concave_R<20'] / max(pp['pct_concave_R<20'], 1e-6)
        rv = cp['pct_convex_R<8'] / max(pp['pct_convex_R<8'], 1e-6)
        c['ratio_concave_R<20_vs_plain'] = round(rc, 2) if judge.any() else 0.0
        c['ratio_convex_R<8_vs_plain'] = round(rv, 2) if judge.any() else 0.0
        c['accept'] = bool(c['ratio_concave_R<20_vs_plain'] <= 1.5 and c['ratio_convex_R<8_vs_plain'] <= 1.5)
        rep['curv6'] = c
        # ---- 3. creases: radius change at the no-fit's own tight places
        cr = {}
        okn = np.isfinite(k2n) & (Vn[idn, 2] <= 38) & (Vn[idn, 0] >= 27)
        d, j = cKDTree(V[idx]).query(Vn[idn])
        for kind, kn, kc, lim in (('concave', -k2n, -k2[j], 1 / 12.0), ('convex', k1n, k1[j], 1 / 5.0)):
            sel = okn & (kn > lim) & np.isfinite(kc) & (d < 4)
            if not sel.any():
                continue
            ratio = np.where(kc[sel] > 1e-6, kn[sel] / np.maximum(kc[sel], 1e-9), 10.0)   # R_cand / R_nofit
            i = int(np.argmin(ratio))
            cr[kind] = dict(n=int(sel.sum()), min_ratio=round(float(ratio[i]), 2), p5_ratio=round(float(np.percentile(ratio, 5)), 2),
                            pct_tighter_20=round(100 * float((ratio < 0.8).mean()), 2),
                            worst_at=np.round(Vn[idn][sel][i], 1).tolist(),
                            worst_R_nofit=round(float(1 / kn[sel][i]), 2), worst_R_cand=round(float(1 / max(kc[sel][i], 1e-9)), 2))
            # robust 'same place': the tightest radius within 2 mm on each skin (a crease that only MOVED by the
            # offset is not tighter; the pointwise compare above reads a moved sharp crease as a change)
            kna = np.where(np.isfinite(kn), kn, -np.inf); kca_full = -k2 if kind == 'concave' else k1
            kca = np.where(np.isfinite(kca_full), kca_full, -np.inf)
            Pq = Vn[idn][sel]
            ln = cKDTree(Vn[idn]).query_ball_point(Pq, 2.0); lc = cKDTree(V[idx]).query_ball_point(Pq, 2.0)
            mn = np.array([kna[l].max() if len(l) else np.nan for l in ln]); mc = np.array([kca[l].max() if len(l) else np.nan for l in lc])
            okk = np.isfinite(mn) & np.isfinite(mc) & (mc > 1e-6)
            rn = mn[okk] / mc[okk]
            i2 = int(np.argmin(rn))
            cr[kind].update(nbhd2_min_ratio=round(float(rn[i2]), 2), nbhd2_p5_ratio=round(float(np.percentile(rn, 5)), 2),
                            nbhd2_pct_tighter_20=round(100 * float((rn < 0.8).mean()), 2), nbhd2_worst_at=np.round(Pq[okk][i2], 1).tolist(),
                            nbhd2_worst_R_nofit=round(float(1 / mn[okk][i2]), 2), nbhd2_worst_R_cand=round(float(1 / mc[okk][i2]), 2))
        rep['creases'] = cr
        print(f'[relief] {label}: curvature done ({time.time() - t0:.0f} s)', flush=True)
    # ---- 4. contacts and soft-tissue allowance
    G = capsule_gaps(tc, caps)
    Z = {}
    for zn, parts in zones(caps).items():
        g, at, wsum, wsup = [], [], 0.0, 0.0
        for name, lo, hi in parts:
            P, t, gg, Q = G[name]
            mm = (t >= lo - 1e-9) & (t <= hi + 1e-9)
            g.append(gg[mm]); at.append(Q[mm])
        g = np.concatenate(g); at = np.concatenate(at)
        i = int(np.argmin(g))
        Z[zn] = dict(gap=round(float(g[i]), 2), overlap=round(max(0.0, -float(g[i])), 2), touch=bool(g[i] <= TOUCH),
                     support_pct=round(100 * float((g <= 1.5).mean()), 1), at=np.round(at[i], 1).tolist())
    rep['contacts'] = Z
    al = {}
    for name, (P, t, gg, Q) in G.items():
        fz = frozen(Q)
        ge = np.where(fz, np.inf, gg)
        give = give_palm if name in PALM_GIVE else give_finger if name in FINGER_GIVE else None
        j = int(np.argmin(ge)); i = int(np.argmin(gg))
        ov = max(0.0, -float(ge[j])) if np.isfinite(ge[j]) else 0.0
        al[name] = dict(overlap_editable=round(ov, 2), at=np.round(Q[j], 1).tolist(), overlap_any=round(max(0.0, -float(gg[i])), 2),
                        frozen_at_max=bool(fz[i]), give=give, accept=None if give is None else bool(ov <= give + 1e-6))
    rep['allowance'] = al
    rep['allowance_accept'] = bool(all(v['accept'] is not False for v in al.values()))
    # ---- 5. keep-outs: trigger stroke vs case_bottom, screw walls
    ko = {}
    cb = os.path.join(run, 'parts', 'case_bottom.npz')
    if os.path.exists(inputs) and os.path.exists(cb):
        import importlib.util
        spec = importlib.util.spec_from_file_location('gather_np', '/tmp/claude-0/sdf/smooth/final/solver_v2/gather_np.py')
        GN = importlib.util.module_from_spec(spec); spec.loader.exec_module(GN)
        g = GN.load_npz(inputs)
        d = np.load(cb); Vc, Tc = d['V'], d['T']
        tcb = L.bvh(Vc, Tc)
        for P, capn in STROKE.items():
            tr = g['trig'][P]
            rest = capsule_gaps(tcb, caps, [capn])[capn]
            pres = capsule_gaps(tcb, caps, [capn], shift=tr['travel'] * tr['travel_mm'])[capn]
            i, j = int(np.argmin(rest[2])), int(np.argmin(pres[2]))
            ko['stroke_' + P] = dict(capsule=capn, travel_mm=round(tr['travel_mm'], 3), gap_rest=round(float(rest[2][i]), 2),
                                     gap_pressed=round(float(pres[2][j]), 2), at=np.round(pres[3][j], 1).tolist(),
                                     accept=bool(pres[2][j] >= 0.3 - 1e-6))
    zi = np.load(os.path.join(run, 'internals.npz'))
    if 'screw bools|V' in zi.files:
        Vs, Ts = zi['screw bools|V'], zi['screw bools|T']
        Ps = sample_surface(Vs, Ts, 0.6)
        Ps = Ps[(Ps[:, 0] > 25) & (Ps[:, 2] < 38)]
        ko['screw_wall'] = screw_walls(tc, tb, Ps)
    rep['keepouts'] = ko
    print(f'[relief] {label}: contacts + keep-outs done ({time.time() - t0:.0f} s)', flush=True)
    # ---- 6. invariants (own, cheap)
    inv = {}
    fzm = frozen(V) & (V[:, 0] >= 0)
    sub = np.nonzero(fzm)[0][::3]
    dfz = np.array([tn.find_nearest(Vector(p))[3] for p in V[sub]])
    inv['frozen_change_vs_nofit_max_mm'] = round(float(dfz.max()), 4)
    inv['frozen_change_at'] = np.round(V[sub][np.argmax(dfz)], 1).tolist()
    msub = np.arange(0, len(V), 7)
    Vm = V[msub] * np.array([-1, 1, 1])
    inv['mirror_dev_max_mm'] = round(float(max(tc.find_nearest(Vector(p))[3] for p in Vm)), 4)
    inv['max_y'] = round(float(V[:, 1].max()), 4)
    inv['max_z'] = round(float(V[:, 2].max()), 4)
    inv['bbox_min'] = np.round(V.min(0), 3).tolist()
    inv['bbox_x_extent'] = round(float(V[:, 0].max()), 3)
    inv['nofit_bbox_x_extent'] = round(float(Vn[:, 0].max()), 3)
    top = np.abs(V[:, 2] - V[:, 2].max()) < 1e-3
    inv['top_verts_at_max_z'] = int(top.sum())
    vol = np.einsum('ij,ij->i', V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])).sum() / 6
    inv['volume_mm3'] = round(float(vol), 1)
    if not quick and os.path.exists(os.path.join(nofit_parts, 'parts')):
        for part in ('case_top', 'case_middle_plate'):
            pa, pb = os.path.join(run, 'parts', part + '.npz'), os.path.join(nofit_parts, 'parts', part + '.npz')
            if os.path.exists(pa) and os.path.exists(pb):
                a_, b_ = np.load(pa), np.load(pb)
                inv[part + '_dev_vs_nofit'] = mesh_dev(a_['V'], a_['T'], b_['V'], b_['T'])
    rep['invariants'] = inv
    rp = os.path.join(run, 'report.json')
    if os.path.exists(rp):
        r = json.load(open(rp))
        sk = r.get('skin', {})
        rep['shared'] = dict(topology=sk.get('topology'), top_face=sk.get('top_face'), max_y_ok=sk.get('max_y_ok'),
                             section_z40=sk.get('section_z40'), bbox=sk.get('bbox'), case_top=r.get('case_top_summary'),
                             acceptance={k: v for k, v in r.get('acceptance', {}).items() if not k.startswith('_')})
    rep['seconds'] = round(time.time() - t0, 1)
    json.dump(rep, open(out, 'w'), indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print(f'[relief] {label}: done {rep["seconds"]} s -> {out}', flush=True)
    return rep


if __name__ == '__main__':
    a = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    o = {'--label': '', '--inputs': INPUTS, '--plain': PLAIN, '--nofit': NOFIT, '--nofit-parts': NOFIT_PARTS,
         '--give-palm': '3.5', '--give-finger': '1.5'}
    quick = '--quick' in a
    a = [x for x in a if x != '--quick']
    for i in range(0, len(a), 2):
        o[a[i]] = a[i + 1]
    analyze(o['--run'], o['--out'], o['--label'], o['--inputs'], o['--plain'], o['--nofit'], o['--nofit-parts'],
            float(o['--give-palm']), float(o['--give-finger']), quick)
