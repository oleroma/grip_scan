"""Validate gripmetrics.curvature() / normal_turn() on analytic shapes meshed like the stage
(make_analytic.py: analytic SDF -> Field to Grid 0.5 mm -> Grid to Mesh).
    /tmp/claude-0/venv/bin/python /tmp/claude-0/sdf/smooth/lib/validate_curvature.py
Writes lib/validation/curvature_validation.json; prints a table. Exit 0 when the errors are inside the
tolerances below (radius error <= 5 % median / 10 % p95 on the clean regions, sign never wrong)."""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gripmetrics as GM  # noqa: E402
VAL = os.path.join(HERE, 'validation')


def stats(name, k, kref, sel):
    e = k[sel] - kref[sel]
    r = {'n': int(sel.sum()), 'ref_mean': round(float(kref[sel].mean()), 5), 'mean': round(float(k[sel].mean()), 5),
         'bias': round(float(e.mean()), 5), 'rms': round(float(np.sqrt((e * e).mean())), 5), 'max_abs': round(float(np.abs(e).max()), 5)}
    nz = sel & (np.abs(kref) > 1e-6)
    if nz.any():
        re = np.abs(1 / k[nz] - 1 / kref[nz]) / np.abs(1 / kref[nz]) * 100
        r.update(radius_err_pct_median=round(float(np.median(re)), 2), radius_err_pct_p95=round(float(np.percentile(re, 95)), 2),
                 radius_err_pct_max=round(float(re.max()), 2), sign_wrong=int((np.sign(k[nz]) != np.sign(kref[nz])).sum()))
    else:
        r['radius_where_ref_flat_min_mm'] = round(float(1 / max(np.abs(k[sel]).max(), 1e-9)), 1)
    return r


def run(name, region_fn):
    z = np.load(os.path.join(VAL, name + '.npz'))
    V, T = z['V'], z['T']
    T, _ = GM.orient(V, T)
    t = time.time()
    C = GM.curvature(V, T)
    tc = time.time() - t
    t = time.time()
    kink, kink_cc, kink_cv = GM.normal_turn(V, T, np.arange(len(V)), signed=True)
    tk = time.time() - t
    out = {'verts': int(len(V)), 'seconds_curvature': round(tc, 2), 'seconds_kink': round(tk, 2),
           'neighbours_median': int(np.median(C['nn'])), 'K': C['K'], 'fit_rms_median_mm': round(float(np.median(C['rms'])), 4)}
    out.update(region_fn(V, C, kink))
    out.update(signed_kinks(name, V, kink_cc, kink_cv))
    return out


def signed_kinks(name, V, cc, cv):
    """Concave / convex split of the normal turn (normals converging / diverging along the pair)."""
    if name == 'sphere_r20':
        return {'signed kink sphere: convex median (ref 2.86) / concave max (ref 0)': [round(float(np.median(cv)), 2), round(float(cc.max()), 2)]}
    if name == 'cylinder_r10':
        b = np.abs(V[:, 0]) < 25
        return {'signed kink cylinder body: convex median (ref 5.73) / concave max (ref 0)': [round(float(np.median(cv[b])), 2), round(float(cc[b].max()), 2)],
                'signed kink cap crease: convex max / concave max': [round(float(cv[np.abs(V[:, 0]) > 29].max()), 1), round(float(cc[np.abs(V[:, 0]) > 29].max()), 1)]}
    rho = np.hypot(V[:, 0], V[:, 1]); dt = np.hypot(rho - 40, V[:, 2]); th = np.degrees(np.arctan2(V[:, 2], 40 - rho))
    bot = (np.abs(dt - 8) < 0.5) & (rho < 40) & (np.abs(th) < 10)
    rim = (np.abs(np.abs(V[:, 2]) - 8) < 1) & (np.abs(rho - 40) < 1)
    return {'signed kink groove bottom: concave median (ref 7.16) / max': [round(float(np.median(cc[bot])), 2), round(float(cc[bot].max()), 2)],
            'signed kink groove rims (convex 90 deg crease): convex median / max': [round(float(np.median(cv[rim])), 1), round(float(cv[rim].max()), 1)]}


def sphere(V, C, kink):
    sel = np.ones(len(V), bool)
    ref = np.full(len(V), 1 / 20)
    return {'k1 (ref 1/20)': stats('k1', C['k1'], ref, sel), 'k2 (ref 1/20)': stats('k2', C['k2'], ref, sel),
            'kink_deg_per_mm': {'ref': round(np.degrees(1 / 20), 2), 'median': round(float(np.median(kink)), 2),
                                'p99': round(float(np.percentile(kink, 99)), 2), 'max': round(float(kink.max()), 2)}}


def cylinder(V, C, kink):
    rr = np.hypot(V[:, 1], V[:, 2])
    sel = (np.abs(V[:, 0]) < 25) & (np.abs(rr - 10) < 0.5)
    return {'k1 (ref 1/10)': stats('k1', C['k1'], np.full(len(V), 0.1), sel), 'k2 (ref 0)': stats('k2', C['k2'], np.zeros(len(V)), sel),
            'dir2 vs axis (deg, median)': round(float(np.median(np.degrees(np.arccos(np.clip(np.abs(C['dir2'][sel, 0]), 0, 1))))), 2),
            'kink_deg_per_mm': {'ref': round(np.degrees(1 / 10), 2), 'median': round(float(np.median(kink[sel])), 2),
                                'p99': round(float(np.percentile(kink[sel], 99)), 2), 'max': round(float(kink[sel].max()), 2)},
            'cap_edge_kink_max (90 deg convex crease)': round(float(kink[np.abs(V[:, 0]) > 29].max()), 1),
            'cap_edge_min_convex_radius_mm': round(float(1 / C['k1'][np.abs(V[:, 0]) > 29].max()), 2)}


def groove(V, C, kink):
    rho = np.hypot(V[:, 0], V[:, 1])
    dt = np.hypot(rho - 40, V[:, 2])
    th = np.degrees(np.arctan2(V[:, 2], 40 - rho))
    on_tube = (np.abs(dt - 8) < 0.5) & (rho < 40)
    sel = on_tube & (np.abs(th) < 50)
    cth = np.cos(np.radians(th))
    k1ref = cth / np.maximum(40 - 8 * cth, 1e-6)
    wall = (np.abs(rho - 40) < 0.4) & (np.abs(V[:, 2]) > 12) & (np.abs(V[:, 2]) < 16)
    rim = (np.abs(V[:, 2] - 8) < 1.0) & (np.abs(rho - 40) < 1.0) | (np.abs(V[:, 2] + 8) < 1.0) & (np.abs(rho - 40) < 1.0)
    bottom = on_tube & (np.abs(th) < 10)
    return {'groove k2 (ref -1/8, across)': stats('k2', C['k2'], np.full(len(V), -1 / 8), sel),
            'groove k1 (ref cos/(40-8cos), along)': stats('k1', C['k1'], k1ref, sel),
            'dir2 across the groove (deg from the tube plane, median)': round(float(np.median(np.degrees(np.arcsin(np.clip(np.abs(
                (C['dir2'][sel] * np.c_[-V[sel, 1], V[sel, 0], np.zeros(sel.sum())] / rho[sel, None]).sum(1)), 0, 1))))), 2),
            'wall k1 (ref 1/40)': stats('k1', C['k1'], np.full(len(V), 1 / 40), wall),
            'wall k2 (ref 0)': stats('k2', C['k2'], np.zeros(len(V)), wall),
            'groove bottom min concave radius (ref 8)': round(float(1 / (-C['k2'][bottom]).max()), 2),
            'groove whole tube min concave radius (incl. near the rims)': round(float(1 / (-C['k2'][on_tube]).max()), 2),
            'kink groove bottom (ref 7.16 deg/mm)': {'median': round(float(np.median(kink[bottom])), 2), 'max': round(float(kink[bottom].max()), 2)},
            'kink at the rims (90 deg convex crease)': {'median': round(float(np.median(kink[rim])), 1), 'max': round(float(kink[rim].max()), 1)},
            'rim min convex radius_mm': round(float(1 / C['k1'][rim].max()), 2)}


res = {'method': {'curvature': f'robust weighted quadric fit, radius {GM.CURV_RADIUS} mm (Gaussian sigma r/2, Huber x3, refit on the fitted normal)',
                  'kink': f'max normal turn per mm between vertices {GM.KINK_D} mm apart, normals averaged over {GM.KINK_SMOOTH} mm',
                  'meshes': 'analytic SDF -> Field to Grid (0.5 mm, centre off the lattice by (0.13, 0.27, 0.41)) -> Grid to Mesh, as the stage'}}
for n, f in (('sphere_r20', sphere), ('cylinder_r10', cylinder), ('torus_groove', groove)):
    res[n] = run(n, f)
    print(n, json.dumps(res[n], indent=1), flush=True)
json.dump(res, open(os.path.join(VAL, 'curvature_validation.json'), 'w'), indent=1)
ok = True
for n, keys in (('sphere_r20', ('k1 (ref 1/20)', 'k2 (ref 1/20)')), ('cylinder_r10', ('k1 (ref 1/10)',)),
                ('torus_groove', ('groove k2 (ref -1/8, across)',))):
    for k in keys:
        s = res[n][k]
        ok &= s['radius_err_pct_median'] <= 5 and s['radius_err_pct_p95'] <= 10 and s['sign_wrong'] == 0
print('VALIDATION', 'OK' if ok else 'BAD')
sys.exit(0 if ok else 1)
