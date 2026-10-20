"""gripmetrics: shared measurements for the hand-fit stage of Olex's GripV case.

Pure numpy / scipy plus mathutils BVH trees (the bpy wheel provides mathutils; importing bpy
first makes it available). No Blender scene access in here: every function takes plain arrays,
so it can be used from measure_case.py, from eval_all.py or from a module agent's own script.

Sign convention: signed distance < 0 inside the case, > 0 outside. Inside/outside is decided by
ray parity (majority vote of 3 fixed directions) or by the generalized winding number, never by
the nearest-face normal (that gave false -28.9 / -20.2 mm readings before).

Gap of a capsule (axis a-b, radius r) to a surface = min over axis points (0.5 mm steps) of
(signed distance - r). Gap < 0: the rigid finger sinks into the case by -gap mm.

Zones (as grip_case_rig.solve / the verified V_C_hand_sdf/v_analyze.py):
  '<f> root pad' : 20 % of the metacarpal next to the knuckle + 35 % of the proximal phalanx
  '<f> PIP'      : 40 % of the middle phalanx next to the PIP joint
  'pinky distal' : whole pinky distal phalanx (wraps)
  'palm on swell': metacarpal capsules of middle, ring, pinky
  'R2 fingertip' / 'R4 fingertip': middle / ring distal phalanx (vs the case skin here; vs the
  trigger parts in trigger_report)
for f in middle, ring, pinky. Index phalanges are reported too (they rest on R1).
"""
import json
import math
import numpy as np

try:  # mathutils ships with the bpy wheel
    import bpy  # noqa: F401
except Exception:  # pragma: no cover
    pass
from mathutils import Vector
from mathutils.bvhtree import BVHTree

FING3 = ('middle', 'ring', 'pinky')
PHALANGES = ('middle_proximal', 'middle_middle', 'middle_distal', 'ring_proximal', 'ring_middle',
             'ring_distal', 'pinky_proximal', 'pinky_middle', 'pinky_distal')
ZONE_ORDER = ('middle root pad', 'middle PIP', 'ring root pad', 'ring PIP', 'pinky root pad',
              'pinky PIP', 'pinky distal', 'R2 fingertip', 'R4 fingertip', 'palm on swell',
              'index proximal', 'index middle', 'index distal')
DIRS = [np.array(d) / np.linalg.norm(d) for d in ((0.31, 0.53, 0.79), (-0.67, 0.21, -0.71), (0.12, -0.91, 0.40))]
TOP_Z = 59.04          # flat top of GripV
MAX_Y = -4.158         # GripV max Y (R1/L1 bumper position)
PARTING_Z = 40.0       # case_top / case_bottom parting plane

V_C_OUT = '/tmp/claude-0/sdf/V_C_hand_sdf/out'
DEFAULT_CAPS = '/tmp/claude-0/sdf/smooth/lib/v27_caps.json'


# ----------------------------------------------------------------------------------- I/O
def load_mesh(path):
    """npz with V/T (or verts/tris) [+ chg] -> (V float64 (n,3), T int64 (m,3), chg or None)."""
    z = np.load(path, allow_pickle=True)
    V = np.asarray(z['V'] if 'V' in z.files else z['verts'], float).reshape(-1, 3)
    T = np.asarray(z['T'] if 'T' in z.files else z['tris'], np.int64).reshape(-1, 3)
    chg = np.asarray(z['chg'], float) if 'chg' in z.files else None
    return V, T, chg


def save_mesh(path, V, T, chg=None, **extra):
    d = dict(V=np.asarray(V, np.float64), T=np.asarray(T, np.int64))
    if chg is not None:
        d['chg'] = np.asarray(chg, np.float64)
    d.update(extra)
    np.savez_compressed(path, **d)


def load_caps(path=DEFAULT_CAPS):
    """Capsules {name: (a, b, r)} from json {name: {'a': [..], 'b': [..], 'r': float}}."""
    c = json.load(open(path))
    return {k: (np.array(v['a'], float), np.array(v['b'], float), float(v['r']))
            for k, v in c.items() if v.get('r') is not None}


def save_caps(path, caps, note=''):
    json.dump({**({'_note': {'a': None, 'b': None, 'r': None, 'text': note}} if note else {}),
               **{k: {'a': list(map(float, a)), 'b': list(map(float, b)), 'r': float(r)} for k, (a, b, r) in caps.items()}},
              open(path, 'w'), indent=1)


def caps_from_points(P, rad, bid, names):
    """Capsules from the 'hand points' cloud (first/last point of each bone, its radius)."""
    caps = {}
    for i, n in enumerate(names):
        m = bid == i
        if m.sum() >= 2:
            caps[str(n)] = (P[m][0].astype(float), P[m][-1].astype(float), float(rad[m][0]))
    return caps


# ------------------------------------------------------------------------- mesh basics
def orient(V, T):
    """Faces outward (positive signed volume). Returns (T, |volume|)."""
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    vol = np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0
    return (T if vol > 0 else T[:, [0, 2, 1]]), abs(float(vol))


def bvh_of(V, T):
    return BVHTree.FromPolygons(np.asarray(V, float).tolist(), np.asarray(T, np.int64).tolist())


def vnormals(V, T):
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    vn = np.zeros_like(V)
    for j in range(3):
        np.add.at(vn, T[:, j], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def vert_area(V, T):
    fa = 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
    va = np.zeros(len(V))
    for j in range(3):
        np.add.at(va, T[:, j], fa / 3)
    return va


def ray_count(bvh, p, d, maxhits=400):
    n = 0
    o = Vector(p)
    dv = Vector(d)
    for _ in range(maxhits):
        hit = bvh.ray_cast(o, dv)
        if hit[0] is None:
            break
        n += 1
        o = hit[0] + dv * 1e-4
    return n


def inside(bvh, p):
    """Ray parity, majority of 3 directions."""
    return sum(ray_count(bvh, p, d) % 2 for d in DIRS) >= 2


def sd(bvh, p):
    """Signed distance (parity sign), < 0 inside."""
    d = bvh.find_nearest(Vector(p))[3]
    if d is None:
        return 1e9
    return -d if inside(bvh, p) else d


def sd_many(bvh, P, far=None):
    """Signed distances for many points. With far=x, points farther than x from the surface skip the
    parity test and get +distance only if `far_sign` is not needed (use for clearance checks)."""
    out = np.empty(len(P))
    for i, p in enumerate(P):
        d = bvh.find_nearest(Vector(p))[3]
        if d is None:
            out[i] = 1e9
            continue
        if far is not None and d > far:
            out[i] = d if not inside(bvh, p) else -d
        else:
            out[i] = -d if inside(bvh, p) else d
    return out


def winding(V, T, P, chunk=1500):
    """Generalized winding number of points P w.r.t. a closed triangle mesh. ~1 inside, ~0 outside."""
    A0, B0, C0 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    out = np.empty(len(P))
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk, None, :]
        a, b, c = A0[None] - p, B0[None] - p, C0[None] - p
        la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
        det = np.einsum('ijk,ijk->ij', a, np.cross(b, c))
        den = la * lb * lc + np.einsum('ijk,ijk->ij', a, b) * lc + np.einsum('ijk,ijk->ij', b, c) * la + np.einsum('ijk,ijk->ij', c, a) * lb
        out[s:s + chunk] = np.arctan2(det, den).sum(1) / (2 * np.pi)
    return out


def edges_info(T, nV):
    E = np.sort(np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1)
    Eu, cnt = np.unique(E, axis=0, return_counts=True)
    return Eu, cnt


def components(T, nV):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    r = np.r_[T[:, 0], T[:, 1], T[:, 2]]
    c = np.r_[T[:, 1], T[:, 2], T[:, 0]]
    A = coo_matrix((np.ones(len(r)), (r, c)), shape=(nV, nV))
    k, lab = connected_components(A, directed=False)
    used = np.zeros(nV, bool)
    used[T.ravel()] = True
    labs = np.unique(lab[used])
    return len(labs), lab, used


def topology(V, T):
    """closed / manifold / islands / genus (sum over islands, from the Euler characteristic)."""
    nV = len(V)
    Eu, cnt = edges_info(T, nV)
    ncomp, lab, used = components(T, nV)
    chi = int(used.sum()) - len(Eu) + len(T)
    sizes = np.bincount(lab[used])
    sizes = sorted([int(s) for s in sizes if s > 0], reverse=True)
    return {'verts': int(nV), 'tris': int(len(T)), 'boundary_edges': int((cnt == 1).sum()),
            'nonmanifold_edges': int((cnt > 2).sum()), 'closed_manifold': bool((cnt == 2).all()),
            'islands': int(ncomp), 'island_sizes_top5': sizes[:5], 'euler_char': chi,
            'genus': (2 * ncomp - chi) / 2.0}


def volume(V, T):
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0)


# --------------------------------------------------------------------------- contacts
def seg_pts(a, b, step=0.5):
    n = max(3, int(np.linalg.norm(b - a) / step) + 1)
    return [a + t * (b - a) for t in np.linspace(0, 1, n)]


def seg_gap(bvh, a, b, r, step=0.5):
    """(gap, axis point, nearest surface point) of the deepest axis point."""
    best = (1e9, None, None)
    for p in seg_pts(a, b, step):
        loc, nor, _, d = bvh.find_nearest(Vector(p))
        if loc is None:
            continue
        s = -d if inside(bvh, p) else d
        if s - r < best[0]:
            best = (s - r, p, np.array(loc))
    return best


def zones(caps):
    Z = {}
    for f in FING3:
        wrist, mcp, rm = caps[f + '_metacarpal']
        a, b, r = caps[f + '_proximal']
        Z[f'{f} root pad'] = [(mcp, mcp + 0.2 * (wrist - mcp), rm), (a, a + 0.35 * (b - a), r)]
        a, b, r = caps[f + '_middle']
        Z[f'{f} PIP'] = [(a, a + 0.4 * (b - a), r)]
    Z['pinky distal'] = [caps['pinky_distal']]
    Z['R2 fingertip'] = [caps['middle_distal']]
    Z['R4 fingertip'] = [caps['ring_distal']]
    Z['palm on swell'] = [(caps[f + '_metacarpal'][1], caps[f + '_metacarpal'][0], caps[f + '_metacarpal'][2]) for f in FING3]
    for s in ('proximal', 'middle', 'distal'):
        if 'index_' + s in caps:
            Z['index ' + s] = [caps['index_' + s]]
    return Z


def contacts(bvh, caps, step=0.5):
    """{zone: gap mm} (+ where). Target for the stage: about -Press (-0.5) at the 8 grip zones."""
    out, where = {}, {}
    for k, segs in zones(caps).items():
        best = (1e9, None, None)
        for a, b, r in segs:
            g = seg_gap(bvh, a, b, r, step)
            if g[0] < best[0]:
                best = g
        out[k] = round(float(best[0]), 2)
        where[k] = None if best[2] is None else np.round(best[2], 1).tolist()
    return out, where


def dig_elsewhere(bvh, caps, step=0.7, excl=1.0, skip=()):
    """Deepest dig of any capsule axis point that is not within `excl` mm of a zone axis point.
    Also the min gap of each capsule over its whole axis (no exclusion)."""
    zp = []
    for segs in zones(caps).values():
        for a, b, r in segs:
            zp += seg_pts(a, b, step)
    zp = np.array(zp)
    worst, per = {}, {}
    for name, (a, b, r) in caps.items():
        if name in skip:
            continue
        best_ex, best_all = (1e9, None), 1e9
        for p in seg_pts(a, b, step):
            g = sd(bvh, p) - r
            best_all = min(best_all, g)
            if np.min(np.linalg.norm(zp - p, axis=1)) < excl:
                continue
            if g < best_ex[0]:
                best_ex = (g, p)
        worst[name] = best_ex
        per[name] = round(float(best_all), 2)
    name = min(worst, key=lambda k: worst[k][0])
    g = worst[name][0]
    return {'max_dig_mm': round(float(-min(0.0, g)), 2), 'where': name,
            'at': None if worst[name][1] is None else np.round(worst[name][1], 1).tolist(),
            'per_capsule_min_gap_outside_zones': {k: round(float(v[0]), 2) for k, v in worst.items()},
            'per_capsule_min_gap_all': per}


def wrap_deg(bvh, caps, names=PHALANGES, nring=36, step=1.0, ring_off=1.0, near=0.3):
    """How far the case wraps around each phalanx: share of a ring of `nring` points at radius
    r + ring_off around the axis that is inside the case or within `near` mm of it, max over the
    axis stations, in degrees (same as V_C_hand_sdf/v_extra.py). ~180 = half the finger enclosed."""
    out = {}
    for k in names:
        if k not in caps:
            continue
        a, b, r = caps[k]
        u = (b - a) / np.linalg.norm(b - a)
        e1 = np.cross(u, [0, 0, 1.0])
        e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 1e-6 else np.cross(u, [1.0, 0, 0])
        e2 = np.cross(u, e1)
        worst = (0.0, None)
        for p in seg_pts(a, b, step)[1:-1]:
            ins = 0
            for j in range(nring):
                ang = 2 * np.pi * j / nring
                q = p + (r + ring_off) * (np.cos(ang) * e1 + np.sin(ang) * e2)
                d = bvh.find_nearest(Vector(q))[3]
                if (d is not None and d < near) or inside(bvh, q):
                    ins += 1
            if ins / nring > worst[0]:
                worst = (ins / nring, p)
        out[k] = int(round(worst[0] * 360))
    return out


# ------------------------------------------------------------------------- thickness
def thickness_ray(bvh, V, N, idx):
    th = np.full(len(idx), 1e9)
    for k, i in enumerate(idx):
        hit = bvh.ray_cast(Vector(V[i] - N[i] * 0.05), Vector(-N[i]))
        if hit[0] is not None:
            th[k] = hit[3] + 0.05
    return th


def wall_thickness(bvh, V, T, xmin=30.0, zmax=38.0, mirror=False):
    """Ray from each handle vertex (x >= xmin, z <= zmax; |x| with mirror) straight inwards."""
    N = vnormals(V, T)
    x = np.abs(V[:, 0]) if mirror else V[:, 0]
    idx = np.nonzero((x >= xmin) & (V[:, 2] <= zmax))[0]
    if len(idx) == 0:
        return {'n': 0}
    th = thickness_ray(bvh, V, N, idx)
    va = vert_area(V, T)[idx]
    thin = idx[th < 2]
    r = {'n': int(len(idx)), 'min': round(float(th.min()), 2), 'min_at': V[idx[np.argmin(th)]].round(1).tolist(),
         'p1': round(float(np.percentile(th, 1)), 2), 'p5': round(float(np.percentile(th, 5)), 2),
         'pct_verts_lt2': round(100 * float((th < 2).mean()), 2),
         'pct_area_lt2': round(100 * float(va[th < 2].sum() / va.sum()), 2)}
    if len(thin):
        r['lt2_bbox'] = [V[thin].min(0).round(1).tolist(), V[thin].max(0).round(1).tolist()]
    return r


# ----------------------------------------------------------------------------- top face
def top_face(V, T, eps=1e-4):
    """Flat top plate (verts within eps of max Z), the generator's invariant."""
    zt = float(V[:, 2].max())
    m = np.abs(V[:, 2] - zt) < eps
    fm = m[T].all(1)
    a = 0.5 * np.linalg.norm(np.cross(V[T[fm, 1]] - V[T[fm, 0]], V[T[fm, 2]] - V[T[fm, 0]]), axis=1).sum() if fm.any() else 0.0
    r = {'max_z': round(zt, 5), 'max_z_ok': bool(abs(zt - TOP_Z) < 1e-3), 'verts_at_max_z': int(m.sum()),
         'tris_at_max_z': int(fm.sum()), 'top_area_mm2': round(float(a), 1),
         'verts_above_59.04': int((V[:, 2] > TOP_Z + 1e-4).sum())}
    if m.any():
        P = V[m]
        r['top_y_min'] = round(float(P[:, 1].min()), 3)      # back edge of the flat top (thenar term moves it)
        r['top_y_max'] = round(float(P[:, 1].max()), 3)
        for side, sel in (('right', P[:, 0] > 20), ('left', P[:, 0] < -20)):
            if sel.any():
                r[f'top_y_min_{side}'] = round(float(P[sel, 1].min()), 3)
    return r


def battery_face(V, T):
    """The generator's 'battery face' group: verts with x == 0 EXACTLY (epsilon 0), their min Z, then the
    faces whose centre Z equals that min within 1e-5. A remesh must keep verts on x = 0 and a flat face there."""
    m0 = V[:, 0] == 0.0
    r = {'verts_at_x0': int(m0.sum())}
    if not m0.any():
        r['ok'] = False
        return r
    zmin = float(V[m0, 2].min())
    cz = V[T][:, :, 2].mean(1)
    f = np.abs(cz - zmin) < 1e-5
    a = 0.5 * np.linalg.norm(np.cross(V[T[f, 1]] - V[T[f, 0]], V[T[f, 2]] - V[T[f, 0]]), axis=1).sum() if f.any() else 0.0
    r.update(min_z_at_x0=round(zmin, 4), tris=int(f.sum()), area_mm2=round(float(a), 1), ok=bool(f.any()))
    return r


def ripple_near_top(V, band=(58.0, TOP_Z - 1e-4)):
    """Verts just below the top plate: a ripple of the top face shows up as many verts in this band
    that are not part of the side wall (informative only)."""
    m = (V[:, 2] > band[0]) & (V[:, 2] < band[1])
    return int(m.sum())


# ------------------------------------------------------------------------------ sections
def section_segments(V, T, z):
    """Segments of the plane section at height z, oriented with the (outward) face normal so the
    shoelace sum gives the enclosed area. Returns (S (n, 2, 3), oriented)."""
    s = V[:, 2] - z
    st = s[T]
    m = (st.min(1) < 0) & (st.max(1) > 0)
    Tm = T[m]
    fn = np.cross(V[Tm[:, 1]] - V[Tm[:, 0]], V[Tm[:, 2]] - V[Tm[:, 0]])
    segs = []
    for tri, ss, n in zip(Tm, st[m], fn):
        P = V[tri]
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if (ss[i] < 0) != (ss[j] < 0):
                t = ss[i] / (ss[i] - ss[j])
                pts.append(P[i] + t * (P[j] - P[i]))
        if len(pts) == 2:
            a, b = pts
            d = b - a
            if d[0] * n[1] - d[1] * n[0] > 0:      # keep the outward normal on the right: CCW outer loops
                a, b = b, a
            segs.append((a, b))
    return np.array(segs).reshape(-1, 2, 3) if segs else np.zeros((0, 2, 3))


def _densify(S, step=0.05):
    out = []
    for a, b in S:
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        t = np.linspace(0, 1, n + 1)[:, None]
        out.append(a + t * (b - a))
    return np.concatenate(out) if out else np.zeros((0, 3))


def section_stats(V, T, z):
    S = section_segments(V, T, z + 1e-5)
    if len(S) == 0:
        return {'n_segments': 0}, np.zeros((0, 3))
    a, b = S[:, 0, :2], S[:, 1, :2]
    area = 0.5 * np.sum(a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1])
    P = S.reshape(-1, 3)
    return {'n_segments': int(len(S)), 'perimeter': round(float(np.linalg.norm(b - a, axis=1).sum()), 2),
            'area_mm2': round(float(area), 1), 'bbox': [P.min(0).round(2).tolist(), P.max(0).round(2).tolist()]}, _densify(S)


def section_change(V, T, Vb, Tb, z=PARTING_Z):
    """Change of the Z=40 parting section vs the base: max distance between the two section curves
    (densified to 0.05 mm, so the value is good to ~0.03 mm), where, and the area change."""
    from scipy.spatial import cKDTree
    r1, P1 = section_stats(V, T, z)
    r0, P0 = section_stats(Vb, Tb, z)
    if r1['n_segments'] == 0 or r0['n_segments'] == 0:
        return {'new': r1, 'base': r0}
    d10 = cKDTree(P0[:, :2]).query(P1[:, :2])[0]
    d01 = cKDTree(P1[:, :2]).query(P0[:, :2])[0]
    worst = P1[np.argmax(d10)] if d10.max() >= d01.max() else P0[np.argmax(d01)]
    return {'new': r1, 'base': r0, 'max_dev_mm': round(float(max(d10.max(), d01.max())), 3),
            'max_dev_at': np.round(worst, 1).tolist(),
            'p99_dev_mm': round(float(np.percentile(np.r_[d10, d01], 99)), 3),
            'area_change_mm2': round(r1['area_mm2'] - r0['area_mm2'], 1)}


# ------------------------------------------------------------------------------ change
def change_map(V, bvh_base, sample_step=1):
    """Signed distance of the new surface's verts to the base (+ grown, - carved), parity sign."""
    idx = np.arange(0, len(V), sample_step)
    chg = np.zeros(len(V))
    for i in idx:
        p = V[i]
        loc, nor, _, d = bvh_base.find_nearest(Vector(p))
        if d is None:
            continue
        if d < 0.02:
            chg[i] = 0.0
            continue
        chg[i] = -d if inside(bvh_base, p) else d
    return chg


def change_bands(V, chg, bands=((-20, 0), (0, 20), (20, 38), (38, 40), (40, 44), (44, 50), (50, 56), (56, 58), (58, 59.1)), thr=0.1):
    out = {}
    for lo, hi in bands:
        m = (V[:, 2] >= lo) & (V[:, 2] < hi)
        if not m.any():
            continue
        c = chg[m]
        out[f'z{lo}..{hi}'] = {'grow_max': round(float(c.max()), 2), 'carve_min': round(float(c.min()), 2),
                               'n_changed': int((np.abs(c) > thr).sum()), 'n': int(m.sum())}
    return out


# --------------------------------------------------------------------------- clearances
def cutout_clearance(V, cutouts, zlo=30.0, zhi=58.95):
    """Unsigned distance from the skin verts (zlo < z < zhi, the top plate excluded) to each cutout
    surface (same as V_C_hand_sdf/v_analyze.py). cutouts: {name: (Vc, Tc)}."""
    sel = np.nonzero((V[:, 2] < zhi) & (V[:, 2] > zlo))[0]
    out = {}
    for n, (Vc, Tc) in cutouts.items():
        b = bvh_of(Vc, Tc)
        best = (1e9, None)
        for i in sel:
            d = b.find_nearest(Vector(V[i]))[3]
            if d is not None and d < best[0]:
                best = (d, i)
        out[n] = {'mm': round(float(best[0]), 2), 'at': None if best[1] is None else V[best[1]].round(1).tolist()}
    if out:
        k = min(out, key=lambda n: out[n]['mm'])
        out['_min'] = {'mm': out[k]['mm'], 'cutout': k}
    return out


def sample_surface(V, T, n=3000, seed=0):
    A = 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
    if A.sum() <= 0:
        return V.copy()
    rng = np.random.default_rng(seed)
    i = rng.choice(len(T), n, p=A / A.sum())
    u = rng.random((n, 2))
    f = u.sum(1) > 1
    u[f] = 1 - u[f]
    S = V[T[i, 0]] + u[:, :1] * (V[T[i, 1]] - V[T[i, 0]]) + u[:, 1:] * (V[T[i, 2]] - V[T[i, 0]])
    return np.r_[S, V]


def internal_clearance(bvh_new, internals, bvh_base=None, n=2500, far=12.0):
    """Wall left between each internal object and the outer skin: min over points on the object's
    surface of (distance to the skin, + inside the skin / - outside). With a base skin also the same
    for the base, and the min over the points that were inside the base skin (the wall that the
    stage may have thinned). Points farther than `far` from the skin are skipped."""
    out = {}
    for name, (Vi, Ti) in internals.items():
        if len(Ti) == 0:
            continue
        P = sample_surface(Vi, Ti, n)
        dn = np.array([bvh_new.find_nearest(Vector(p))[3] or 1e9 for p in P])
        keep = dn < far
        if bvh_base is not None:
            db = np.array([bvh_base.find_nearest(Vector(p))[3] or 1e9 for p in P])
            keep |= db < far
        if not keep.any():
            out[name] = {'min_mm': None, 'note': f'farther than {far} mm from the skin'}
            continue
        Pk = P[keep]
        sn = -sd_many(bvh_new, Pk)            # + inside the skin
        r = {'min_mm': round(float(sn.min()), 2), 'at': Pk[np.argmin(sn)].round(1).tolist()}
        if bvh_base is not None:
            sb = -sd_many(bvh_base, Pk)
            r['base_min_mm'] = round(float(sb.min()), 2)
            ins = sb > 0
            if ins.any():
                r['min_mm_where_inside_base'] = round(float(sn[ins].min()), 2)
                r['base_min_mm_where_inside_base'] = round(float(sb[ins].min()), 2)
                loss = sb[ins] - sn[ins]
                r['max_wall_loss_mm'] = round(float(loss.max()), 2)
                r['max_wall_loss_at'] = Pk[ins][np.argmax(loss)].round(1).tolist()
        out[name] = r
    return out


# ------------------------------------------------------------------------------ triggers
def press_face_from_mesh(V, T, top_mask):
    """Area-weighted centre and normal of the faces flagged 'top' on a trigger part."""
    a, b, c = V[T[top_mask, 0]], V[T[top_mask, 1]], V[T[top_mask, 2]]
    cr = np.cross(b - a, c - a)
    A = 0.5 * np.linalg.norm(cr, axis=1)
    C = (a + b + c) / 3
    n = cr.sum(0)
    return (C * A[:, None]).sum(0) / A.sum(), n / np.linalg.norm(n)


def capsule_vs_solid(V, T, caps, names=None, step=0.5):
    """Min gap of each capsule to a small closed solid (winding-number sign). Overlap = -gap."""
    b = bvh_of(V, T)
    out = {}
    for k, (a, bb, r) in caps.items():
        if names and k not in names:
            continue
        P = np.array(seg_pts(a, bb, step))
        d = np.array([b.find_nearest(Vector(p))[3] for p in P])
        w = winding(V, T, P)
        s = np.where(w > 0.5, -d, d)
        i = int(np.argmin(s - r))
        out[k] = (float(s[i] - r), P[i])
    return out


def trigger_report(triggers, caps, press=None, seats=None, skin_bvh=None, skin=None):
    """triggers: {'R2': (V, T), 'R4': (V, T), ...}; press: {'R2': {'center':[..], 'normal':[..]}};
    seats: {'R2': [x, y, z]}; skin: (V, T) of the stage output (for the protrusion of the trigger)."""
    tip = {'R2': 'middle_distal', 'R4': 'ring_distal'}
    out = {}
    for part, (Vt, Tt) in triggers.items():
        if part not in ('R2', 'R4'):
            continue
        g = capsule_vs_solid(Vt, Tt, caps)
        worst = min(g, key=lambda k: g[k][0])
        r = {'max_hand_overlap_mm': round(max(0.0, -g[worst][0]), 2), 'worst_capsule': worst,
             'overlapping_capsules': {k: round(-v[0], 2) for k, v in g.items() if v[0] < 0},
             'fingertip_gap_to_part_mm': round(g[tip[part]][0], 2)}
        if press and part in press:
            c = np.array(press[part]['center'])
            n = np.array(press[part]['normal'])
            a, bb, rr = caps[tip[part]]
            u = bb - a
            t = np.clip((c - a) @ u / (u @ u), 0, 1)
            q = a + t * u
            r['press_face_center'] = np.round(c, 2).tolist()
            r['pad_to_press_face_center_gap_mm'] = round(float(np.linalg.norm(c - q) - rr), 2)
            r['press_normal_vs_pad_direction_deg'] = round(float(np.degrees(np.arccos(np.clip(n @ ((q - c) / max(np.linalg.norm(q - c), 1e-9)), -1, 1)))), 1)
            if seats and part in seats:
                r['seat_to_press_face_center_mm'] = round(float(np.linalg.norm(np.array(seats[part]) - c)), 2)
        if skin is not None:
            Vs, Ts = skin
            bs = skin_bvh or bvh_of(Vs, Ts)
            d = np.array([bs.find_nearest(Vector(p))[3] for p in Vt])
            cand = np.nonzero(d > 0.02)[0]
            if len(cand):
                outside = np.array([not inside(bs, Vt[i]) for i in cand])
                r['protrusion_out_of_skin_mm'] = round(float(d[cand][outside].max()), 2) if outside.any() else 0.0
                if press and part in press:
                    c = np.array(press[part]['center'])
                    r['press_face_center_to_skin_signed_mm'] = round(float(sd(bs, c)), 2)
        out[part] = r
    return out


# ---------------------------------------------------------------------------- full skin
def skin_report(V, T, caps, base=None, cutouts=None, internals=None, triggers=None, press=None,
                seats=None, quick=False, chg=None):
    """Everything about one stage output (the evaluated 'basemesh GripV' skin).
    base: (Vb, Tb) plain GripV at the same subdivision (for change / section / clearance deltas)."""
    import time
    t0 = time.time()
    if len(V) == 0 or len(T) == 0:
        return {'empty': True, 'contacts_mm': {}, 'palm_mm': None}
    raw = volume(V, T)          # raw winding as stored, before orienting (fix 2026-10-20)
    T, vol = orient(V, T)
    bvh = bvh_of(V, T)
    rep = {'volume_mm3': round(vol, 1), 'raw_signed_volume_mm3': round(raw, 1), 'outward_winding': bool(raw > 0)}
    c, where = contacts(bvh, caps)
    rep['contacts_mm'] = {k: c[k] for k in ZONE_ORDER if k in c}
    rep['contacts_where'] = where
    rep['palm_mm'] = c.get('palm on swell')
    rep['dig_elsewhere'] = dig_elsewhere(bvh, caps)
    rep['topology'] = topology(V, T)
    rep['top_face'] = top_face(V, T)
    rep['battery_face'] = battery_face(V, T)
    rep['bbox'] = [V.min(0).round(3).tolist(), V.max(0).round(3).tolist()]
    rep['max_y'] = round(float(V[:, 1].max()), 3)
    rep['max_y_ok'] = bool(V[:, 1].max() <= MAX_Y + 1e-3)
    rep['wall_thickness_right_handle'] = wall_thickness(bvh, V, T)
    if base is not None:
        Vb, Tb = base
        Tb, _ = orient(Vb, Tb)
        bb = bvh_of(Vb, Tb)
        rep['section_z40'] = section_change(V, T, Vb, Tb)
        if chg is None:
            chg = change_map(V, bb)
        rep['change_bands'] = change_bands(V, chg)
        rep['volume_change_mm3'] = round(vol - orient(Vb, Tb)[1], 1)
    else:
        bb = None
        rep['section_z40'] = section_stats(V, T, PARTING_Z)[0]
        if chg is not None:
            rep['change_bands'] = change_bands(V, chg)
    if chg is not None:
        rep['change_range_mm'] = [round(float(chg.min()), 2), round(float(chg.max()), 2)]
    if cutouts:
        rep['stick_cutout_clearance'] = cutout_clearance(V, cutouts)
    if triggers:
        rep['triggers'] = trigger_report(triggers, caps, press, seats, skin_bvh=bvh, skin=(V, T))
    if not quick:
        rep['wrap_deg'] = wrap_deg(bvh, caps)
        if internals:
            rep['internal_clearance'] = internal_clearance(bvh, internals, bb)
    rep['seconds'] = round(time.time() - t0, 1)
    return rep


def summary_line(rep):
    if rep.get('empty'):
        return 'EMPTY STAGE OUTPUT'
    c = rep['contacts_mm']
    s = ' '.join(f"{c[k]:+.2f}" for k in ZONE_ORDER[:7] if k in c)
    t = rep['topology']
    w = rep.get('wall_thickness_right_handle', {})
    return (f"contacts {s} palm {rep['palm_mm']:+.2f} | dig {rep['dig_elsewhere']['max_dig_mm']:.2f} ({rep['dig_elsewhere']['where']}) | "
            f"genus {t['genus']:g} islands {t['islands']} | thin<2mm {w.get('pct_verts_lt2')}% p1 {w.get('p1')} | "
            f"top {rep['top_face']['verts_at_max_z']} verts @ {rep['top_face']['max_z']} | maxY {rep['max_y']}")


# =====================================================================================================
# smooth/ extension (2026-10-19): smoothness metrics, contact support for a smooth surface, seats,
# frozen region, the fixed press face.  Everything below is new in /tmp/claude-0/sdf/smooth/lib.
# =====================================================================================================
EDIT_MIN_X = 25.0          # |x| below this: frozen (body between the handles, mirror seam)
FROZEN_Z = 38.0            # above this: frozen (parting band Z 38-40 and case_top), except the thenar module
FRONT_BAND = 0.5           # y above MAX_Y - FRONT_BAND: frozen (the verts that hold max Y = R1/L1 position)
EDITED_MM = 0.2            # a vertex is 'edited' when it is more than this from the no-fit baseline surface
FINGER_R = 12.0            # concave radius below this reads as a groove (a finger is ~16-18 mm across)
GROOVE_R = 8.0             # concave radius below this: a groove narrower than a finger, never acceptable
CURV_RADIUS = 3.0          # neighbourhood of the quadric fit (mm)
KINK_SMOOTH = 1.0          # normals are averaged over this radius before the kink test (mm)
KINK_D = (1.0, 2.0)        # kink test: normal turn between points this far apart (mm)
CONTACT_ZONES = ('middle root pad', 'middle PIP', 'ring root pad', 'ring PIP', 'pinky root pad', 'pinky PIP',
                 'pinky distal', 'palm on swell')
SUPPORT_MM = 1.5           # a zone axis point is 'supported' when its gap is at most this (mm)
FEATURE_KINK_CONCAVE = 10.0   # concave normal turn above this (deg/mm) = a groove crease (noise floor on smooth skin: max 4.5)
FEATURE_KINK_CONVEX = 25.0    # convex normal turn above this (deg/mm) = a sharp edge (a convex R 3 mm rounding reads 19)


def frozen_mask(V, edit_min_x=EDIT_MIN_X, frozen_z=FROZEN_Z, front_band=FRONT_BAND, max_y=MAX_Y):
    """True where the smooth shape must not change the skin (the thenar module's own area is above
    frozen_z, so it is inside this mask: the shape module must leave it alone too)."""
    return (np.abs(V[:, 0]) < edit_min_x) | (V[:, 2] > frozen_z) | (V[:, 1] > max_y - front_band)


def handle_mask(V, edit_min_x=EDIT_MIN_X, zmax=PARTING_Z):
    """Right handle below the parting plane (the left one is its mirror image)."""
    return (V[:, 0] >= edit_min_x) & (V[:, 2] < zmax)


def _tangent_frames(n):
    a = np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(n, e1)


def curvature(V, T, idx=None, radius=CURV_RADIUS, iters=3, chunk=2500, N=None, tree=None, kcap=700):
    """Principal curvatures at the vertices `idx` by a robust weighted quadric fit.

    For each vertex: neighbours within `radius` (Euclidean ball, Gaussian weights sigma = radius/2,
    neighbours whose normal faces away (< ~70 deg) are dropped, so the far side of a thin wall never
    counts), local frame from the neighbourhood's mean normal, weighted least squares of
    h = a u^2 + b uv + c v^2 + d u + e v + f, Huber re-weighting (scale = 1.4826 MAD), then the frame is
    re-centred on the fitted normal and fitted again. Curvatures from the shape operator of the graph.

    Sign: + convex (sphere seen from outside), - concave (groove, the inside of a bowl), 1/mm,
    w.r.t. the outward normal (faces are oriented by the signed volume first).
    Returns dict: k1 >= k2 (n,), dir2 (n,3) direction of k2 (across a groove), normal (n,3),
    rms (fit residual, mm), nn (neighbours used)."""
    from scipy.spatial import cKDTree
    V = np.asarray(V, float)
    T, _ = orient(V, T)
    if N is None:
        N = vnormals(V, T)
    if tree is None:
        tree = cKDTree(V)
    idx = np.arange(len(V)) if idx is None else np.asarray(idx, np.int64)
    apv = float(np.median(vert_area(V, T)[np.unique(T)]))
    K = int(min(kcap, max(40, 1.3 * np.pi * radius ** 2 / max(apv, 1e-6))))
    sig = radius / 2.0
    n_out = len(idx)
    k1 = np.zeros(n_out); k2 = np.zeros(n_out); rms = np.zeros(n_out); nn = np.zeros(n_out, np.int32)
    d2 = np.zeros((n_out, 3)); nf_all = np.zeros((n_out, 3))
    eye = np.eye(6) * 1e-10
    for s in range(0, n_out, chunk):
        ii = idx[s:s + chunk]
        dd, jj = tree.query(V[ii], k=min(K, len(V)), distance_upper_bound=radius, workers=2)
        valid = np.isfinite(dd)
        jj = np.where(valid, jj, 0)
        dd = np.where(valid, dd, radius)
        D = V[jj] - V[ii][:, None, :]
        w0 = np.exp(-0.5 * (dd / sig) ** 2) * valid
        Nj = N[jj]
        n0 = N[ii]
        same = (np.einsum('nkc,nc->nk', Nj, n0) > 0.0)
        n0 = (w0[..., None] * Nj * same[..., None]).sum(1)
        n0 /= np.maximum(np.linalg.norm(n0, axis=1, keepdims=True), 1e-12)
        for rnd in range(2):
            compat = np.clip(((Nj @ n0[..., None])[..., 0] - 0.3) / 0.3, 0.0, 1.0)
            ww = w0 * compat
            e1, e2 = _tangent_frames(n0)
            u = (D @ e1[..., None])[..., 0]
            v = (D @ e2[..., None])[..., 0]
            h = (D @ n0[..., None])[..., 0]
            A = np.empty(u.shape + (6,))
            A[..., 0] = u * u; A[..., 1] = u * v; A[..., 2] = v * v; A[..., 3] = u; A[..., 4] = v; A[..., 5] = 1.0
            wr = np.ones_like(u)
            nval = (ww > 1e-6).sum(1)
            rows = np.arange(len(ii))
            for it in range(iters if rnd else 1):
                W = ww * wr
                AW = (A * W[..., None]).transpose(0, 2, 1)
                Mx = AW @ A + eye
                b = AW @ h[..., None]
                c = np.linalg.solve(Mx, b)[..., 0]
                r = h - (A @ c[..., None])[..., 0]
                ar = np.sort(np.where(ww > 1e-6, np.abs(r), np.inf), axis=1)
                sc = np.maximum(1.4826 * ar[rows, np.maximum(nval // 2, 0)], 0.005)   # MAD scale
                wr = np.minimum(1.0, 1.345 * sc[:, None] / np.maximum(np.abs(r), 1e-12))
            gu, gv = c[:, 3], c[:, 4]
            nf = n0 - gu[:, None] * e1 - gv[:, None] * e2
            nf /= np.linalg.norm(nf, axis=1, keepdims=True)
            if rnd == 0:
                n0 = nf
        huu, huv, hvv = 2 * c[:, 0], c[:, 1], 2 * c[:, 2]
        E, F, G_ = 1 + gu * gu, gu * gv, 1 + gv * gv
        Wn = np.sqrt(1 + gu * gu + gv * gv)
        L, Mm, Nn = huu / Wn, huv / Wn, hvv / Wn
        det = E * G_ - F * F
        S11 = (G_ * L - F * Mm) / det; S12 = (G_ * Mm - F * Nn) / det
        S21 = (E * Mm - F * L) / det; S22 = (E * Nn - F * Mm) / det
        tr = S11 + S22
        dt = S11 * S22 - S12 * S21
        disc = np.sqrt(np.maximum(tr * tr / 4 - dt, 0.0))
        lam1, lam2 = tr / 2 + disc, tr / 2 - disc            # w.r.t. the outward normal: convex -> negative
        k1[s:s + chunk] = -lam2
        k2[s:s + chunk] = -lam1
        phi = 0.5 * np.arctan2(2 * huv, huu - hvv)          # direction of the largest II eigenvalue = most concave
        dv = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2
        d2[s:s + chunk] = dv
        nf_all[s:s + chunk] = nf
        Wf = ww * wr
        rms[s:s + chunk] = np.sqrt((Wf * r * r).sum(1) / np.maximum(Wf.sum(1), 1e-12))
        nn[s:s + chunk] = (ww > 0.05).sum(1)
    return {'k1': k1, 'k2': k2, 'dir2': d2, 'normal': nf_all, 'rms': rms, 'nn': nn, 'K': K}


def smoothed_normals(V, T, idx, N=None, tree=None, radius=KINK_SMOOTH):
    """Area-weighted vertex normals averaged with Gaussian weights (sigma = radius/2) within `radius`."""
    from scipy.spatial import cKDTree
    T, _ = orient(V, T)
    if N is None:
        fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        N = np.zeros_like(V)
        for j in range(3):
            np.add.at(N, T[:, j], fn)        # area-weighted (unnormalised)
    if tree is None:
        tree = cKDTree(V)
    out = np.zeros((len(idx), 3))
    for s in range(0, len(idx), 20000):
        ii = idx[s:s + 20000]
        dd, jj = tree.query(V[ii], k=64, distance_upper_bound=radius)
        valid = np.isfinite(dd)
        jj = np.where(valid, jj, 0)
        w = np.exp(-0.5 * (np.where(valid, dd, radius) / (radius / 2)) ** 2) * valid
        Nj = N[jj]
        same = np.einsum('nkc,nc->nk', Nj, N[ii]) > 0
        n = (w[..., None] * Nj * same[..., None]).sum(1)
        out[s:s + 20000] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return out


def normal_turn(V, T, idx, tree=None, dmin=KINK_D[0], dmax=KINK_D[1], smooth=KINK_SMOOTH, signed=False):
    """Kink detector: per vertex, the largest turn of the (smoothed) surface normal per mm towards any
    vertex dmin..dmax mm away on the same sheet (normals within ~100 deg), in deg/mm.
    A smooth surface of radius R gives 57.3 / R deg/mm (R 8 mm: 7.2; R 12 mm: 4.8); a crease far more."""
    from scipy.spatial import cKDTree
    if tree is None:
        tree = cKDTree(V)
    idx = np.asarray(idx, np.int64)
    # smoothed normals of every vertex within dmax of the query set
    nb = tree.query_ball_point(V[idx], dmax + 1e-6) if len(idx) < 4000 else None
    if nb is not None:
        need = np.unique(np.concatenate([np.asarray(x, np.int64) for x in nb] + [idx]))
    else:
        lo, hi = V[idx].min(0) - dmax - 0.1, V[idx].max(0) + dmax + 0.1
        need = np.nonzero(np.all((V >= lo) & (V <= hi), axis=1))[0]
    Ns = np.zeros_like(V)
    Ns[need] = smoothed_normals(V, T, need, tree=tree, radius=smooth)
    out = np.zeros(len(idx)); occ = np.zeros(len(idx)); ocv = np.zeros(len(idx))
    K = 160
    for s in range(0, len(idx), 10000):
        ii = idx[s:s + 10000]
        dd, jj = tree.query(V[ii], k=K, distance_upper_bound=dmax)
        valid = np.isfinite(dd) & (dd >= dmin)
        jj = np.where(np.isfinite(dd), jj, 0)
        c = np.clip(np.einsum('nkc,nc->nk', Ns[jj], Ns[ii]), -1, 1)
        valid &= c > -0.2
        ang = np.where(valid, np.degrees(np.arccos(c)) / np.where(valid, dd, 1.0), 0.0)
        out[s:s + 10000] = ang.max(1)
        if signed:
            # normals converging along the pair = concave (groove side), diverging = convex (ridge side)
            conv = np.einsum('nkc,nkc->nk', Ns[jj] - Ns[ii][:, None, :], V[jj] - V[ii][:, None, :]) > 0
            occ[s:s + 10000] = np.where(~conv, ang, 0.0).max(1)
            ocv[s:s + 10000] = np.where(conv, ang, 0.0).max(1)
    return (out, occ, ocv) if signed else out


def _wpct(x, w, q):
    """Weighted percentile (q in 0..100)."""
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    if cw[-1] <= 0:
        return float('nan')
    return float(x[o][min(np.searchsorted(cw, q / 100.0 * cw[-1]), len(x) - 1)])


def smooth_stats(V, A, k1, k2, kink, sel, label=''):
    """Smoothness numbers of the vertices `sel` (bool over the arrays k1/k2/kink, which belong to the
    vertices V/A given): concave / convex radii, groove area shares, kink (normal turn per mm)."""
    sel = np.asarray(sel, bool)
    if not sel.any():
        return {'n': 0, 'area_mm2': 0.0}
    kd = kink if isinstance(kink, dict) else {'all': kink}
    a = A[sel]; c2 = k2[sel]; c1 = k1[sel]; kk = kd['all'][sel]; P = V[sel]
    tot = float(a.sum())
    conc = np.maximum(-c2, 0.0)              # concave curvature (1/mm)
    conv = np.maximum(c1, 0.0)
    i_cc = int(np.argmax(conc)); i_cv = int(np.argmax(conv)); i_k = int(np.argmax(kk))
    r = {'n': int(sel.sum()), 'area_mm2': round(tot, 1),
         'min_concave_radius_mm': round(float(1 / conc[i_cc]), 2) if conc[i_cc] > 1e-9 else None,
         'min_concave_radius_at': P[i_cc].round(1).tolist(),
         'concave_radius_p0.5_mm': round(float(1 / max(_wpct(conc, a, 99.5), 1e-9)), 2),
         'concave_radius_p2_mm': round(float(1 / max(_wpct(conc, a, 98.0), 1e-9)), 2),
         'pct_area_concave_r_lt12': round(100 * float(a[conc > 1 / FINGER_R].sum()) / tot, 2),
         'pct_area_concave_r_lt8': round(100 * float(a[conc > 1 / GROOVE_R].sum()) / tot, 2),
         'area_concave_r_lt12_mm2': round(float(a[conc > 1 / FINGER_R].sum()), 1),
         'area_concave_r_lt8_mm2': round(float(a[conc > 1 / GROOVE_R].sum()), 1),
         'min_convex_radius_mm': round(float(1 / conv[i_cv]), 2) if conv[i_cv] > 1e-9 else None,
         'min_convex_radius_at': P[i_cv].round(1).tolist(),
         'convex_radius_p0.5_mm': round(float(1 / max(_wpct(conv, a, 99.5), 1e-9)), 2),
         'max_normal_turn_deg_per_mm': round(float(kk[i_k]), 2), 'max_normal_turn_at': P[i_k].round(1).tolist(),
         'normal_turn_p99_deg_per_mm': round(_wpct(kk, a, 99.0), 2),
         'normal_turn_p99.9_deg_per_mm': round(_wpct(kk, a, 99.9), 2)}
    for sk in ('concave', 'convex'):
        if sk in kd:
            x = kd[sk][sel]
            i = int(np.argmax(x))
            r[f'max_{sk}_turn_deg_per_mm'] = round(float(x[i]), 2)
            r[f'max_{sk}_turn_at'] = P[i].round(1).tolist()
            r[f'{sk}_turn_p99_deg_per_mm'] = round(_wpct(x, a, 99.0), 2)
    if label:
        r['label'] = label
    return r


# ------------------------------------------------------------------- contacts for a smooth surface
def zone_support(bvh, caps, step=0.5, near=SUPPORT_MM, names=None):
    """Per zone (as zones()): deepest penetration of the rigid capsule (target ~0.5 mm, accept 0.2-1.0),
    min gap, and the share of the zone's axis length whose gap is at most `near` (the part of the
    finger the case supports)."""
    out = {}
    for k, segs in zones(caps).items():
        if names and k not in names:
            continue
        g, w, loc, ax = [], [], [], []
        for a, b, r in segs:
            P = seg_pts(a, b, step)
            L = float(np.linalg.norm(b - a))
            for p in P:
                hit, _, _, d = bvh.find_nearest(Vector(p))
                if hit is None:
                    continue
                s = -d if inside(bvh, p) else d
                g.append(s - r); w.append(L / len(P)); loc.append(np.array(hit)); ax.append(p)
        g, w = np.array(g), np.array(w)
        i = int(np.argmin(g))
        sup = g <= near
        out[k] = {'min_gap_mm': round(float(g[i]), 2), 'penetration_mm': round(float(max(0.0, -g[i])), 2),
                  'support_pct': round(100 * float(w[sup].sum() / w.sum()), 1),
                  'supported_mm': round(float(w[sup].sum()), 1), 'zone_axis_mm': round(float(w.sum()), 1),
                  'mean_gap_supported_mm': round(float(g[sup].mean()), 2) if sup.any() else None,
                  'deepest_at_surface': loc[i].round(1).tolist(), 'deepest_at_axis': np.round(ax[i], 1).tolist(),
                  'penetration_ok': bool(0.2 <= max(0.0, -g[i]) <= 1.0)}
    return out


def max_dig(bvh, caps, step=0.5, skip=()):
    """Deepest dig of any hand capsule anywhere (limit 1.0 mm), split by where it lands: in the editable
    region or in the frozen one (top / body / front band: the shape cannot change those)."""
    per, worst = {}, {'editable': (1e9, None, None, None), 'frozen': (1e9, None, None, None)}
    for name, (a, b, r) in caps.items():
        if name in skip:
            continue
        best = 1e9
        for p in seg_pts(a, b, step):
            hit, _, _, d = bvh.find_nearest(Vector(p))
            if hit is None:
                continue
            s = -d if inside(bvh, p) else d
            gap = s - r
            best = min(best, gap)
            reg = 'frozen' if frozen_mask(np.array([hit]))[0] else 'editable'
            if gap < worst[reg][0]:
                worst[reg] = (gap, name, np.array(hit), p)
        per[name] = round(float(best), 2)
    r = {'per_capsule_min_gap_mm': per}
    for reg, (gap, name, hit, p) in worst.items():
        r[f'max_dig_{reg}_mm'] = round(float(max(0.0, -gap)), 2) if name else 0.0
        r[f'max_dig_{reg}_capsule'] = name
        r[f'max_dig_{reg}_at'] = None if hit is None else hit.round(1).tolist()
    r['max_dig_mm'] = max(r['max_dig_editable_mm'], r['max_dig_frozen_mm'])
    return r


# --------------------------------------------------------------------------------------- seats
SEAT_SKIN = 0.2            # skin plane = press face - SEAT_SKIN along its normal
SEAT_RADIUS = 2.5          # flush rim around the trigger opening
SEAT_FADE = 5.0            # suggested blend width beyond the rim


def plane_section(V, T, c, n, reach=20.0, xmin=0.0):
    """Segments (m, 2, 3) of the section of a mesh with the plane (c, n), only near c (and x > xmin)."""
    h = (V - c) @ n
    st = h[T]
    m = (st.min(1) < 0) & (st.max(1) > 0)
    segs = []
    for tri, ss in zip(T[m], st[m]):
        P = V[tri]
        pts = [P[i] + ss[i] / (ss[i] - ss[j]) * (P[j] - P[i]) for i, j in ((0, 1), (1, 2), (2, 0)) if (ss[i] < 0) != (ss[j] < 0)]
        if len(pts) == 2 and min(np.linalg.norm(pts[0] - c), np.linalg.norm(pts[1] - c)) < reach and min(pts[0][0], pts[1][0]) > xmin:
            segs.append(pts)
    return np.array(segs).reshape(-1, 2, 3)


def _dist_to_segments(P, S):
    """Distance of points P (k,3) to segments S (m,2,3)."""
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


def seat_report(V, T, press, cutters, bvh=None, radius=SEAT_RADIUS, skin_off=SEAT_SKIN):
    """How flush the skin is with each trigger's seat plane (press face - skin_off along n) on a rim of
    `radius` mm around the case opening. Opening = section of the cutter '<P> bool' (right half) with the
    seat plane, near the press face; rim = skin vertices within 8 mm of the plane whose projection onto it
    lies outside the cutter and within `radius` of that outline (exact in-plane distance)."""
    if bvh is None:
        T, _ = orient(V, T)
        bvh = bvh_of(V, T)
    out = {}
    for part, pf in (press or {}).items():
        if part not in ('R2', 'R4') or part not in cutters:
            continue
        c, n = np.array(pf['center'], float), np.array(pf['normal'], float)
        s0 = c - skin_off * n
        Vc, Tc = cutters[part]
        Tc, _ = orient(Vc, Tc)
        S = plane_section(Vc, Tc, s0, n)
        r = {'seat_plane_point': s0.round(3).tolist(), 'normal': n.round(5).tolist(), 'rim_radius_mm': radius,
             'skin_at_press_center_signed_mm': round(float(sd(bvh, c)), 3), 'target_skin_at_press_center_mm': skin_off}
        if len(S) == 0:
            r['rim_verts'] = 0
            r['note'] = 'cutter does not cross the seat plane'
            out[part] = r
            continue
        q = S.reshape(-1, 3) - s0
        ext = float(np.linalg.norm(q - np.outer(q @ n, n), axis=1).max())
        h = (V - s0) @ n
        P = V - h[:, None] * n
        cand = np.nonzero((np.abs(h) < 8.0) & (np.linalg.norm(P - s0, axis=1) < ext + radius + 0.5) & (V[:, 0] > 0))[0]
        dop = _dist_to_segments(P[cand], S)
        bc = bvh_of(Vc, Tc)
        near = dop <= radius
        outside = np.array([not inside(bc, P[i]) if ok else False for i, ok in zip(cand, near)])
        rim = cand[near & outside]
        r.update(opening_extent_mm=round(ext, 2), opening_segments=int(len(S)))
        if len(rim):
            dv = h[rim]
            r.update(rim_verts=int(len(rim)), rim_dev_mean_mm=round(float(dv.mean()), 3),
                     rim_dev_max_abs_mm=round(float(np.abs(dv).max()), 3), rim_dev_p95_abs_mm=round(float(np.percentile(np.abs(dv), 95)), 3),
                     rim_dev_min_mm=round(float(dv.min()), 3), rim_dev_max_mm=round(float(dv.max()), 3),
                     rim_worst_at=V[rim[np.argmax(np.abs(dv))]].round(1).tolist())
        else:
            r['rim_verts'] = 0
        out[part] = r
    return out


# --------------------------------------------------------------------------- change vs a reference
def mesh_deviation(V, T, Vr, Tr):
    """Two-sided max unsigned surface distance between two meshes (vertices of each to the other surface)."""
    from mathutils import Vector
    out = {}
    for name, (P, (Vb, Tb)) in (('to_ref', (V, (Vr, Tr))), ('from_ref', (Vr, (V, T)))):
        b = bvh_of(Vb, Tb)
        d = np.array([b.find_nearest(Vector(p))[3] for p in P])
        i = int(np.argmax(d))
        out[name] = {'max_mm': round(float(d[i]), 4), 'at': P[i].round(2).tolist(), 'n_gt_0.01': int((d > 0.01).sum())}
    out['max_mm'] = max(out['to_ref']['max_mm'], out['from_ref']['max_mm'])
    return out


def change_vs(V, bvh_ref, idx, thr=EDITED_MM, sign=True):
    """Distance of the vertices idx to a reference surface (the no-fit baseline): unsigned for all,
    signed (+ grown = outside the reference, - receded; ray parity) where it is above thr."""
    d = np.zeros(len(idx))
    for k, i in enumerate(idx):
        hit = bvh_ref.find_nearest(Vector(V[i]))
        d[k] = hit[3] if hit[0] is not None else 1e9
    s = d.copy()
    if sign:
        for k in np.nonzero(d > thr)[0]:
            if inside(bvh_ref, V[idx[k]]):
                s[k] = -d[k]
    return d, s


def press_face_poly(V, T, tri_mask):
    """Area-weighted centre / normal of the triangles tri_mask (a press face exported from Blender)."""
    a, b, c = V[T[tri_mask, 0]], V[T[tri_mask, 1]], V[T[tri_mask, 2]]
    cr = np.cross(b - a, c - a)
    A = 0.5 * np.linalg.norm(cr, axis=1)
    n = cr.sum(0)
    return {'center': ((a + b + c) / 3 * A[:, None]).sum(0) / A.sum(), 'normal': n / np.linalg.norm(n), 'area': float(A.sum())}


# ------------------------------------------------------------------------------ smooth report
def curvature_fields(V, T, sel_mask, tree=None):
    """k1, k2, kink for the vertices of sel_mask (arrays over all V, 0 elsewhere) + the computed mask."""
    from scipy.spatial import cKDTree
    tree = tree or cKDTree(V)
    idx = np.nonzero(sel_mask)[0]
    k1 = np.zeros(len(V)); k2 = np.zeros(len(V)); kink = np.zeros(len(V)); kcc = np.zeros(len(V)); kcv = np.zeros(len(V))
    if len(idx):
        C = curvature(V, T, idx=idx, tree=tree)
        k1[idx], k2[idx] = C['k1'], C['k2']
        kink[idx], kcc[idx], kcv[idx] = normal_turn(V, T, idx, tree=tree, signed=True)
    return k1, k2, {'all': kink, 'concave': kcc, 'convex': kcv}, sel_mask.copy()


def curv_region(V):
    """Where curvature is computed on a skin: the right handle and its surroundings (x >= 20, z < 42)."""
    return (V[:, 0] >= EDIT_MIN_X - 5.0) & (V[:, 2] < PARTING_Z + 2.0)


def smooth_report(V, T, caps, nofit=None, nofit_curv=None, press=None, cutters=None, curv_out=None, dilate=3.0):
    """Smoothness, contact support, dig, seats, frozen-region change vs the no-fit baseline.
    nofit: (Vn, Tn) the no-fit baseline skin (base.blend: shape pass-through, thenar on).
    nofit_curv: dict(V=Vn, k1, k2, kink, mask) of the baseline (cached) for the same statistics on the
    baseline in the same footprint. curv_out: path to save this skin's curvature fields (npz)."""
    import time
    from scipy.spatial import cKDTree
    t0 = time.time()
    T, _ = orient(V, T)
    bvh = bvh_of(V, T)
    tree = cKDTree(V)
    A = vert_area(V, T)
    rep = {}
    # ---- change vs the no-fit baseline (right half; the left half is the mirror image)
    right = np.nonzero(V[:, 0] >= 0)[0]
    edited = np.zeros(len(V), bool)
    if nofit is not None:
        Vn, Tn = nofit
        Tn, _ = orient(Vn, Tn)
        bn = bvh_of(Vn, Tn)
        d, s = change_vs(V, bn, right)
        dd = np.zeros(len(V)); ss = np.zeros(len(V))
        dd[right], ss[right] = d, s
        edited = dd > EDITED_MM
        fz = frozen_mask(V)
        fr = right[fz[right]]
        rep['change_vs_nofit'] = {
            'edited_verts': int(edited.sum()), 'edited_area_mm2': round(float(A[edited].sum()), 1),
            'grow_max_mm': round(float(ss.max()), 2), 'recede_max_mm': round(float(-ss.min()), 2),
            'grow_max_at': V[np.argmax(ss)].round(1).tolist(), 'recede_max_at': V[np.argmin(ss)].round(1).tolist(),
            'frozen_max_change_mm': round(float(dd[fr].max()), 4) if len(fr) else 0.0,
            'frozen_max_change_at': V[fr[np.argmax(dd[fr])]].round(1).tolist() if len(fr) else None,
            'edited_outside_handle_verts': int((edited & ~handle_mask(V)).sum()),
            'edited_bbox': [V[edited].min(0).round(1).tolist(), V[edited].max(0).round(1).tolist()] if edited.any() else None}
        rep['_change_signed'] = ss
    # ---- curvature
    reg = curv_region(V)
    k1, k2, kink, cm = curvature_fields(V, T, reg, tree)
    if curv_out:
        np.savez_compressed(curv_out, V=V, k1=k1, k2=k2, kink=kink['all'], kink_concave=kink['concave'], kink_convex=kink['convex'],
                            mask=cm, edited=edited)
    regions = {'handle_below_z40': handle_mask(V)}
    if edited.any():
        e_idx = np.nonzero(edited)[0]
        dil = np.zeros(len(V), bool)
        for lst in tree.query_ball_point(V[e_idx], dilate, workers=2):
            dil[lst] = True
        dil &= V[:, 0] >= 0
        regions['edited'] = edited & cm
        regions[f'edited_plus_{dilate:g}mm'] = dil & cm
    if nofit_curv is not None and edited.any():
        # unchanged design features of the generator (the body-handle crease at z ~30, the outline edges, ...):
        # not edited (<= EDITED_MM from the no-fit skin) AND a feature on the no-fit skin at the nearest vertex
        Vb = nofit_curv['V']
        _, jn = cKDTree(Vb).query(V, workers=2)
        kb = nofit_curv['kink']
        kb_cc = kb['concave'] if isinstance(kb, dict) else kb
        kb_cv = kb['convex'] if isinstance(kb, dict) else kb
        fb = nofit_curv['mask'] & ((nofit_curv['k2'] < -1 / FINGER_R) | (nofit_curv['k1'] > 1 / 3.0) |
                                   (kb_cc > FEATURE_KINK_CONCAVE) | (kb_cv > FEATURE_KINK_CONVEX))
        # dilate the features by the kink pair distance: an unchanged vertex next to a design crease reads the crease
        fd = np.zeros(len(Vb), bool)
        fi = np.nonzero(fb)[0]
        if len(fi):
            for lst in cKDTree(Vb).query_ball_point(Vb[fi], KINK_D[1], workers=2):
                fd[lst] = True
        unchanged_feat = (~edited) & fd[jn]
        key = f'edited_plus_{dilate:g}mm'
        regions[key + '_excl_unchanged_features'] = regions[key] & ~unchanged_feat
        rep['change_vs_nofit']['unchanged_design_feature_verts_excluded'] = int((regions[key] & unchanged_feat).sum())
    rep['smoothness'] = {k: smooth_stats(V, A, k1, k2, kink, m & cm) for k, m in regions.items()}
    # the same statistics of the no-fit baseline in the same footprint
    if nofit_curv is not None and edited.any():
        Vb = nofit_curv['V']
        nb_tree = cKDTree(V)
        _, j = nb_tree.query(Vb, workers=2)
        Ab = vert_area(Vb, orient(Vb, nofit[1])[0]) if nofit is not None else np.ones(len(Vb))
        for k in ('edited', f'edited_plus_{dilate:g}mm'):
            mb = regions[k][j] & nofit_curv['mask'] & (np.linalg.norm(Vb - V[j], axis=1) < 6.0)
            rep['smoothness'][k + ' (no-fit baseline, same footprint)'] = smooth_stats(Vb, Ab, nofit_curv['k1'], nofit_curv['k2'], nofit_curv['kink'], mb)
    # ---- contacts / dig / seats
    rep['zone_support'] = zone_support(bvh, caps)
    rep['max_dig'] = max_dig(bvh, caps)
    if press and cutters:
        rep['seats'] = seat_report(V, T, press, cutters, bvh)
    rep['seconds'] = round(time.time() - t0, 1)
    return rep


# --------------------------------------------------------------------------------- acceptance
ACCEPT = {
    'zone_penetration_mm': (0.2, 1.0),       # every contact zone: deepest penetration of the rigid capsule
    'zone_support_pct_min': 50.0,            # root pads / PIPs / palm: share of the zone axis within 1.5 mm (target)
    'max_dig_editable_mm': 1.0,              # any capsule, anywhere the shape may change
    'pct_area_concave_lt8': 0.5,             # edited + 3 mm: area share with concave radius < 8 mm (%)
    'pct_area_concave_lt12': 5.0,            # edited + 3 mm: area share with concave radius < 12 mm (%)
    'concave_radius_p0.5_mm': 8.0,           # edited + 3 mm: 99.5 % of the area has a concave radius above this
    'convex_radius_p0.5_mm': 3.0,            # edited + 3 mm: no sharp convex ridge
    'max_concave_turn_deg_per_mm': 10.0,     # edited + 3 mm: concave kink (groove R 8 reads 7.2; noise <= 4.5; a crease > 30)
    'max_convex_turn_deg_per_mm': 25.0,      # edited + 3 mm: convex kink (rounding R 3 reads 19; the handle's own edges 15-20)
    'frozen_max_change_mm': 0.05,
    'section_z40_dev_mm': 0.05,
    'seat_skin_at_press_center_mm': (0.1, 0.3),   # skin 0.2 mm under the press face
    'seat_rim_dev_max_abs_mm': 0.3,
    'pad_gap_mm': (-0.55, -0.45),            # fingertip pad vs press-face centre (the solved triggers: -0.50)
    'trigger_hand_overlap_mm': 0.55,
    'cutout_clearance_mm': 0.8,
    'wall_p1_mm': 2.0,
    'internal_wall_mm': 1.5,                 # wall to each internal object: >= min(its no-fit wall, this)
    'case_top_dev_mm': 0.05,                 # case_top surface vs no-fit case_top, two-sided max (noise 0.018 measured)
    'part_volume_vs_plain_mm3': 1.0,         # every evaluated part except case_top / case_bottom vs plain GripV (fix 2026-10-20)
    'case_top_volume_mm3': 10.0,             # case_top volume vs no-fit (noise 3.3 mm3 measured with a bit-identical
                                             # skin above z 38: the generator re-triangulates the parting band z 40-45
                                             # when the skin's vertex order changes)
}
SUPPORT_ZONES = ('middle root pad', 'middle PIP', 'ring root pad', 'ring PIP', 'pinky root pad', 'pinky PIP', 'palm on swell')


def acceptance(rep, nofit_rep=None):
    """Checks of a prototype run against PROBLEM.md. Returns {check: {'value', 'limit', 'pass'}} plus
    'n_fail' / 'failed' / 'hard_fail' lists. 'info' checks (pass None) are reported, not judged."""
    A, sk, sm = {}, rep.get('skin', {}), rep.get('smooth', {})

    def put(name, value, limit, ok, kind='hard'):
        A[name] = {'value': value, 'limit': limit, 'pass': None if ok is None else bool(ok), 'kind': kind}

    lo, hi = ACCEPT['zone_penetration_mm']
    for z, r in sm.get('zone_support', {}).items():
        if z in CONTACT_ZONES:
            put(f'contact {z}: penetration mm', r['penetration_mm'], f'{lo}..{hi} (target 0.5)', lo <= r['penetration_mm'] <= hi)
        if z in SUPPORT_ZONES:
            put(f'support {z}: % of zone within {SUPPORT_MM} mm', r['support_pct'], f">= {ACCEPT['zone_support_pct_min']:g}",
                r['support_pct'] >= ACCEPT['zone_support_pct_min'], 'soft')
    md = sm.get('max_dig', {})
    if md:
        put('max dig, editable region mm', md['max_dig_editable_mm'], f"<= {ACCEPT['max_dig_editable_mm']}", md['max_dig_editable_mm'] <= ACCEPT['max_dig_editable_mm'])
        put('max dig, frozen region mm (info: the shape cannot change it)', md['max_dig_frozen_mm'], 'info', None, 'info')
    regs = sm.get('smoothness', {})
    key = next((k for k in regs if k.endswith('_excl_unchanged_features')), None) or \
        next((k for k in regs if k.startswith('edited_plus_') and '(' not in k), None)
    if key and regs[key].get('n', 0):
        r = regs[key]
        put(f'{key}: % area concave R < 8 mm', r['pct_area_concave_r_lt8'], f"<= {ACCEPT['pct_area_concave_lt8']}", r['pct_area_concave_r_lt8'] <= ACCEPT['pct_area_concave_lt8'])
        put(f'{key}: % area concave R < 12 mm', r['pct_area_concave_r_lt12'], f"<= {ACCEPT['pct_area_concave_lt12']}", r['pct_area_concave_r_lt12'] <= ACCEPT['pct_area_concave_lt12'])
        put(f'{key}: concave radius p0.5 mm', r['concave_radius_p0.5_mm'], f">= {ACCEPT['concave_radius_p0.5_mm']}", r['concave_radius_p0.5_mm'] >= ACCEPT['concave_radius_p0.5_mm'])
        put(f'{key}: min concave radius mm (info)', r['min_concave_radius_mm'], 'info', None, 'info')
        put(f'{key}: convex radius p0.5 mm', r['convex_radius_p0.5_mm'], f">= {ACCEPT['convex_radius_p0.5_mm']}", r['convex_radius_p0.5_mm'] >= ACCEPT['convex_radius_p0.5_mm'])
        put(f'{key}: max concave normal turn deg/mm (kink)', r.get('max_concave_turn_deg_per_mm'), f"<= {ACCEPT['max_concave_turn_deg_per_mm']}",
            r.get('max_concave_turn_deg_per_mm', 1e9) <= ACCEPT['max_concave_turn_deg_per_mm'])
        put(f'{key}: max convex normal turn deg/mm (kink)', r.get('max_convex_turn_deg_per_mm'), f"<= {ACCEPT['max_convex_turn_deg_per_mm']}",
            r.get('max_convex_turn_deg_per_mm', 1e9) <= ACCEPT['max_convex_turn_deg_per_mm'])
        put(f'{key}: max normal turn deg/mm (info, unsigned)', r['max_normal_turn_deg_per_mm'], 'info', None, 'info')
    elif 'change_vs_nofit' in sm:
        put('edited region', 0, 'nothing edited', None, 'info')
    cv = sm.get('change_vs_nofit')
    if cv:
        put('frozen region max change mm', cv['frozen_max_change_mm'], f"<= {ACCEPT['frozen_max_change_mm']}", cv['frozen_max_change_mm'] <= ACCEPT['frozen_max_change_mm'])
        put('grow max / recede max vs no-fit mm (info)', [cv['grow_max_mm'], cv['recede_max_mm']], 'info', None, 'info')
    vn = sm.get('vs_nofit')
    if vn:
        sec = vn['section_z40'].get('max_dev_mm')
        put('Z40 section vs no-fit, max dev mm', sec, f"<= {ACCEPT['section_z40_dev_mm']}", sec is not None and sec <= ACCEPT['section_z40_dev_mm'])
        tf = vn['top_face']
        put('top face verts / back edge Y = no-fit', [tf['verts_at_max_z'], tf['top_y_min']], 'equal', tf['verts_at_max_z'][0] == tf['verts_at_max_z'][1] and tf['top_y_min'][0] == tf['top_y_min'][1])
        put('bbox min y / z shift vs no-fit mm (screws)', vn['bbox_min_yz_shift_mm'], '<= 0.01', vn['bbox_min_yz_shift_mm'] <= 0.01)
        put('bbox x extent vs no-fit mm (|x| max 82.896; OPEN QUESTION: it sits on the palm contact)', vn['bbox_x_shift_mm'], '<= 0.01',
            vn['bbox_x_shift_mm'] <= 0.01, 'bbox_x')
        if 'case_top_volume_change_mm3' in vn:
            put('case_top volume change vs no-fit mm3', vn['case_top_volume_change_mm3'], f"|d| <= {ACCEPT['case_top_volume_mm3']:g} (noise 3.3)",
                abs(vn['case_top_volume_change_mm3']) <= ACCEPT['case_top_volume_mm3'])
        if 'case_middle_plate_volume_change_mm3' in vn:
            dv, db = vn['case_middle_plate_volume_change_mm3'], vn.get('case_middle_plate_bbox_shift_mm')
            put('case_middle_plate vs no-fit: [volume change mm3, bbox shift mm]', [dv, db],
                f"|dV| <= {ACCEPT['part_volume_vs_plain_mm3']:g}, bbox <= 0.05",
                abs(dv) <= ACCEPT['part_volume_vs_plain_mm3'] and (db is None or db <= 0.05))
        if 'case_top_dev' in vn:
            put('case_top surface vs no-fit case_top, max dev mm', vn['case_top_dev']['max_mm'], f"<= {ACCEPT['case_top_dev_mm']}",
                vn['case_top_dev']['max_mm'] <= ACCEPT['case_top_dev_mm'])
    t = sk.get('topology', {})
    if t:
        put('skin closed manifold, 1 island, genus 0', [t['closed_manifold'], t['islands'], t['genus']], 'True, 1, 0', t['closed_manifold'] and t['islands'] == 1 and t['genus'] == 0)
    # ---- raw winding + parts vs plain (fix 2026-10-20): measured on the triangles as wound, never after orient()
    if 'raw_signed_volume_mm3' in sk:
        put('skin wound outward: raw signed volume mm3', sk['raw_signed_volume_mm3'], '> 0', sk['raw_signed_volume_mm3'] > 0)
    parts = rep.get('parts') or {}
    if parts:
        inv = [[p, r['raw_signed_volume_mm3']] for p, r in parts.items() if not r.get('empty') and r.get('raw_signed_volume_mm3') is not None and r['raw_signed_volume_mm3'] <= 0]
        n = sum(1 for r in parts.values() if not r.get('empty'))
        put('parts wound outward: raw signed volume > 0', inv or f'all {n} ok', 'no part inside-out', not inv)
        lim = ACCEPT['part_volume_vs_plain_mm3']
        bad, n_cmp = [], 0
        for p, r in parts.items():
            if p in ('case_top', 'case_bottom', 'case_middle_plate'):   # shaped by the skin by design: judged vs no-fit
                continue
            if r.get('empty') or r.get('base_empty'):
                if r.get('empty') != bool(r.get('base_empty')) and 'base_empty' in r:
                    bad.append([p, 'empty' if r.get('empty') else 'not empty', 'plain ' + ('empty' if r.get('base_empty') else 'not empty')])
                n_cmp += 'base_empty' in r
                continue
            if r.get('base_volume_mm3') is None:
                continue
            n_cmp += 1
            # Blender's n-gon volume when both runs have it (less triangulation noise), else the triangle volume
            vb = (r.get('volume_bmesh_mm3'), r.get('base_volume_bmesh_mm3'))
            dv = round(vb[0] - vb[1], 2) if None not in vb else round(r['volume_mm3'] - r['base_volume_mm3'], 2)
            chi, chib = r.get('euler_char'), r.get('base_euler_char')
            if abs(dv) > lim or (chi is not None and chib is not None and chi != chib):
                bad.append([p, dv, chi, chib])
        if n_cmp:
            put(f'parts vs plain GripV (all but case_top/bottom/middle_plate): |volume change| <= {lim:g} mm3, same Euler char',
                bad or f'all {n_cmp} ok', '[part, dV mm3, chi, plain chi]', not bad)
        mp = parts.get('case_middle_plate')
        if mp and mp.get('base_volume_mm3') is not None:
            put('case_middle_plate vs plain GripV: volume change mm3 (info; the stage remesh alone gives about +1.9)',
                round(mp['volume_mm3'] - mp['base_volume_mm3'], 2), 'info', None, 'info')
        ch = parts.get('cable holder')
        if ch and not ch.get('empty') and ch.get('base_volume_mm3') is not None:
            dv = round(ch['volume_mm3'] - ch['base_volume_mm3'], 2)
            put('cable holder vs plain: [volume change mm3, Euler char, plain Euler char]', [dv, ch.get('euler_char'), ch.get('base_euler_char')],
                f'|dV| <= {lim:g}, same chi', abs(dv) <= lim and ch.get('euler_char') == ch.get('base_euler_char'))
    tf = sk.get('top_face', {})
    if tf:
        put('top face max Z 59.04, nothing above', [tf['max_z'], tf['verts_above_59.04']], '59.04, 0', tf['max_z_ok'] and tf['verts_above_59.04'] == 0)
    if 'max_y' in sk:
        put('max Y (R1/L1 position)', sk['max_y'], '= -4.158', abs(sk['max_y'] - MAX_Y) <= 1e-3)
    bf = sk.get('battery_face', {})
    if bf:
        put('battery face (verts at x = 0, flat face)', [bf.get('verts_at_x0'), bf.get('min_z_at_x0')], 'ok', bf.get('ok'))
    sc = sk.get('stick_cutout_clearance', {}).get('_min')
    if sc:
        put('stick cutout clearance mm', sc['mm'], f">= {ACCEPT['cutout_clearance_mm']}", sc['mm'] >= ACCEPT['cutout_clearance_mm'])
    w = sk.get('wall_thickness_right_handle', {})
    if w.get('p1') is not None:
        put('handle wall p1 mm', w['p1'], f">= {ACCEPT['wall_p1_mm']}", w['p1'] >= ACCEPT['wall_p1_mm'])
    seats = sm.get('seats', {})
    lo, hi = ACCEPT['seat_skin_at_press_center_mm']
    for p, r in seats.items():
        put(f'{p} seat: skin at press-face centre mm', r['skin_at_press_center_signed_mm'], f'{lo}..{hi} (target 0.2)', lo <= r['skin_at_press_center_signed_mm'] <= hi)
        if r.get('rim_verts'):
            put(f'{p} seat: rim max |dev| from seat plane mm', r['rim_dev_max_abs_mm'], f"<= {ACCEPT['seat_rim_dev_max_abs_mm']}", r['rim_dev_max_abs_mm'] <= ACCEPT['seat_rim_dev_max_abs_mm'])
    lo, hi = ACCEPT['pad_gap_mm']
    for p, r in sk.get('triggers', {}).items():
        g = r.get('pad_to_press_face_center_gap_mm')
        if g is not None:
            put(f'{p} fingertip pad to press-face centre mm', g, f'{lo}..{hi} (-0.5)', lo <= g <= hi)
        put(f'{p} hand overlap with the trigger part mm', r['max_hand_overlap_mm'], f"<= {ACCEPT['trigger_hand_overlap_mm']}", r['max_hand_overlap_mm'] <= ACCEPT['trigger_hand_overlap_mm'])
    ic = sk.get('internal_clearance', {})
    icn = (nofit_rep or {}).get('skin', {}).get('internal_clearance', {})
    if ic and icn:
        bad = []
        for k, v in ic.items():
            b = icn.get(k, {}).get('min_mm')
            if v.get('min_mm') is None or b is None:
                continue
            need = min(b, ACCEPT['internal_wall_mm']) - 0.05
            if v['min_mm'] < need:
                bad.append([k, v['min_mm'], b])
        put(f"internal objects: wall >= min(no-fit wall, {ACCEPT['internal_wall_mm']}) mm", bad or 'all ok', 'no object closer', not bad)
    if ic:
        worst = min(((k, v['min_mm']) for k, v in ic.items() if v.get('min_mm') is not None), key=lambda x: x[1], default=None)
        loss = max(((k, v.get('max_wall_loss_mm', 0.0)) for k, v in ic.items()), key=lambda x: x[1], default=None)
        if worst:
            put('internal objects: min wall to the skin mm (info)', worst, 'info', None, 'info')
        if loss:
            put('internal objects: max wall loss vs plain mm (info)', loss, 'info', None, 'info')
    if 'R1L1_shift_mm' in rep:
        put('R1/L1 shift mm', rep['R1L1_shift_mm'], '0', rep['R1L1_shift_mm'] <= 1e-3)
    hard = [k for k, v in A.items() if v['pass'] is False and v['kind'] in ('hard', 'bbox_x')]
    soft = [k for k, v in A.items() if v['pass'] is False and v['kind'] == 'soft']
    A['_summary'] = {'hard_fail': hard, 'soft_fail': soft,
                     'n_hard_checks': sum(1 for v in A.values() if isinstance(v, dict) and v.get('kind') in ('hard', 'bbox_x')),
                     'accepted': not hard,
                     'accepted_except_bbox_x': not [k for k in hard if A[k]['kind'] != 'bbox_x']}
    return A


def acceptance_line(A):
    s = A.get('_summary', {})
    hf = s.get('hard_fail', [])
    return (f"ACCEPT {'YES' if s.get('accepted') else 'NO'} (without the bbox-x check: {'YES' if s.get('accepted_except_bbox_x') else 'NO'}): "
            f"{len(hf)}/{s.get('n_hard_checks')} hard checks fail"
            + (': ' + '; '.join(h.split(':')[0] if h.startswith('contact') else h for h in hf[:6]) if hf else '')
            + (f" | soft fails {len(s.get('soft_fail', []))}" if s.get('soft_fail') else ''))
