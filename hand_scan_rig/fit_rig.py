"""Module B: map the hand rig into a hand scan.

    /tmp/claude-0/hs/venv/bin/python fit_rig.py SCAN.(ply|obj|glb) [--out DIR] [--units auto|mm|m|cm]
        [--lengths rig|olex|FILE.json] [--no-render] [--name NAME]

Input: any right-hand scan mesh (photogrammetry: arbitrary scale, orientation and
units; may have noise, holes, a table plane, background blobs, a forearm of any
length, a texture (ignored)).
Output (DIR, default /mnt/project-files/hand_scan_rig/fit/<name>/):
  hand_rigged.blend  objects 'hand scan' (watertight skin, mm) + 'hand rig'
                     (armature, SPEC bone names, rest pose = scan pose),
                     Armature modifier 'Armature', vertex groups named after bones
                     (initial distance-based weights; the skin module replaces them)
  scan_clean.ply     the cleaned, watertight, decimated skin (mm, scan orientation)
  joints.json        SPEC format (+ "fit" block: scale, residuals, diagnostics)
  check.png          palm-view and side-view renders with the fitted skeleton drawn in

Frame: the output keeps the scan's orientation and origin; only a uniform scale
is applied so that 1 unit = 1 mm (scale chosen from the measured segment lengths
when the units are unknown).

Method (geometric, no ML):
 1. clean: RANSAC plane -> drop a large planar (table) patch; keep the largest
    connected component; Blender: fill holes, voxel remesh (watertight), Taubin
    smoothing, decimate. Scale-free first pass (voxel = bbox/300) for detection.
 2. fingertips = geodesic extremities that are finger-like (ring width w* where the
    geodesic ring radius t first exceeds 1.5 w is small); thumb = the tip whose
    nearest other tip is geodesically farthest; index = long finger geodesically
    nearest the thumb; others chained by geodesic proximity.
 3. finger centre-lines: geodesic level-set ring centroids, refined by planar
    cross-sections perpendicular to the local axis (area centroid of the loop that
    contains the axis point), from the tip until the loop merges into the palm.
    Tip landmark = skin apex along the distal axis minus TIP_INSET (2.5 mm).
    DIP / PIP = points on the centre-line at the bone lengths (chord) counted back
    from the tip, refined by the dorsal knuckle / palmar crease profile where the
    evidence is clear; MCP = PIP - L_prox * (proximal axis, fitted on the free
    proximal centre-line). Thumb: tip, IP, MCP the same way; CMC and wrist from the
    rig palm layout (hold_rig.blend, arch kept 30% as in the stand-in), fitted
    rigidly (scale fixed) to the four MCPs, the palm normal and the wrist
    cross-section centroid (lateral / depth); the wrist position ALONG the
    wrist -> knuckle axis is then re-solved from the four metacarpal lines
    (MCP_i = wrist + L_i * layout direction_i, L_i free with a 4% prior on the rig
    length; wrist_free_lengths()), which removes most of the metacarpal-length and
    scale error; CMC rides with the refitted layout, then is projected onto the
    sphere |CMC - thumb_MCP| = L_thumb_meta.
 4. scale: s = argmin sum (measured segment length - s * L_ref)^2 over the
    distal / middle (and thumb distal / proximal) segments whose both ends come
    from geometric evidence, iterated with the joint placement; residuals reported.
 5. armature: bones exactly at the joint centres, bone Z = palmar direction
    (perpendicular to the bone), so local X = flexion axis and a POSITIVE rotation
    about local X curls the finger TOWARD THE PALM (verified in code: verify_rig()).
    The 'palm' bone (wrist -> knuckle-row centre) is a non-deforming handle.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RIG_BLEND = "/tmp/claude-0/hs/hold_rig.blend"
OUT_ROOT = "/mnt/project-files/hand_scan_rig/fit"
WORK = "/tmp/claude-0/hs/work/fit"

NAMES = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_mcp", "index_pip",
         "index_dip", "index_tip", "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
         "ring_mcp", "ring_pip", "ring_dip", "ring_tip", "pinky_mcp", "pinky_pip", "pinky_dip",
         "pinky_tip"]
FINGERS4 = ["index", "middle", "ring", "pinky"]
FINGERS5 = ["thumb"] + FINGERS4
SEGS4 = ["proximal", "middle", "distal"]
SEGS_T = ["metacarpal", "proximal", "distal"]
JN4 = ["mcp", "pip", "dip", "tip"]
JNT = ["cmc", "mcp", "ip", "tip"]

TIP_INSET = 2.5          # mm, tip landmark inside the skin apex along the distal axis
FINAL_VOXEL = 0.6        # mm, voxel remesh of the output skin
TARGET_VERTS = 50000
FOREARM_KEEP = 40.0      # mm of forearm kept proximal of the wrist
W_REF = 16.95            # mm, median ring width w* of the 4 long fingers (stand-in calibration)
MCP_DEPTH = 0.41         # MCP centre at this fraction of the knuckle-row thickness from the dorsal skin
PROX_FIT_FRAC = float(os.environ.get('FIT_PROX_FRAC', 0.6))  # proximal axis: fit on this part of the free centre-line (webs bias the rest)
CMC_SPHERE_W = float(os.environ.get('FIT_CMC_W', 1.0))
WRIST_LS = int(os.environ.get('FIT_WRIST_LS', 1))   # wrist from metacarpal lines with free lengths
ANALYSIS_SMOOTH = int(os.environ.get('FIT_SMOOTH', 20))   # Taubin iterations on the analysis mesh (scan noise)
WRIST_DEPTH = 0.5        # wrist joint centre at mid-thickness of the wrist
# width-peak position relative to the joint centre (mm, + = peak lies proximal of the
# joint): fingers taper distally, so the knuckle width maximum sits slightly proximal.
# Calibrated on the stand-in (base + 2 variants: DIP 0.46+-0.04, PIP 0.35+-0.08, IP 0.60+-0.07; IP re-checked 0.74 after the noise-robust width profile)
PEAK_OFF = {0: 0.46, 1: 0.35}    # segment index distal->proximal: 0 = DIP / thumb IP, 1 = PIP
PEAK_OFF_THUMB = 0.74
OLEX_LENGTHS = {"index": {"proximal": 42.7, "middle": 24.4, "distal": 19.8},
                "middle": {"proximal": 46.2, "middle": 26.5, "distal": 20.4},
                "ring": {"proximal": 42.7, "middle": 26.5, "distal": 20.3},
                "pinky": {"proximal": 32.8, "middle": 19.7, "distal": 17.7},
                "thumb": {"metacarpal": 36.5, "proximal": 33.1, "distal": 23.9}}


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def log(*a):
    print("[fit_rig %6.1fs]" % (time.time() - T0), *a, flush=True)


T0 = time.time()


# ======================================================================= rig reference
def read_rig(path=RIG_BLEND):
    """Bone heads/tails/lengths of 'hand rig' (rest pose, world mm). Cached."""
    os.makedirs(WORK, exist_ok=True)
    cache = os.path.join(WORK, "rig_rest.json")
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
        return json.load(open(cache))
    import bpy
    bpy.ops.wm.open_mainfile(filepath=path)
    arm = bpy.data.objects["hand rig"]
    mw = arm.matrix_world
    out = {b.name: {"head": list(mw @ b.head_local), "tail": list(mw @ b.tail_local),
                    "length": b.length} for b in arm.data.bones}
    json.dump(out, open(cache, "w"), indent=1)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return out


def palm_layout(R, arch_keep=0.3):
    """Wrist (origin), MCP heads and thumb CMC in the palm frame (x across to the
    index side, y wrist -> middle MCP, z palmar), arch flattened like the stand-in;
    plus bone lengths."""
    W = np.array(R["palm"]["head"])
    mcp = {f: np.array(R[f + "_metacarpal"]["tail"]) for f in FINGERS4}
    y = unit(mcp["middle"] - W)
    a = mcp["index"] - mcp["pinky"]
    x = unit(a - (a @ y) * y)
    z = np.cross(x, y)
    B = np.stack([x, y, z])
    lay = {"mcp": {}, "len": {}}
    for f in FINGERS4:
        m = B @ (mcp[f] - W)
        m[2] *= arch_keep
        lay["mcp"][f] = m
    lay["cmc"] = B @ (np.array(R["thumb_metacarpal"]["head"]) - W)
    for f in FINGERS4:
        lay["len"][f] = {s: R[f"{f}_{s}"]["length"] for s in SEGS4}
        lay["len"][f]["metacarpal"] = float(np.linalg.norm(lay["mcp"][f]))
    lay["len"]["thumb"] = {s: R[f"thumb_{s}"]["length"] for s in SEGS_T}
    return lay


# ======================================================================= mesh io / clean
def load_mesh(path):
    import trimesh
    m = trimesh.load(path, force="mesh", process=False)
    if not hasattr(m, "faces") or len(m.faces) == 0:
        raise SystemExit(f"{path}: no faces (point clouds are not supported, mesh it first)")
    # Textured photogrammetry meshes (OpenMVS / texrecon OBJ, GLB) carry one vertex per
    # UV corner, so a plain load splits the surface at every UV seam into hundreds of
    # pieces.  Drop the visuals and weld by position before any component analysis.
    n_in = len(m.vertices)
    m = trimesh.Trimesh(np.asarray(m.vertices, float), np.asarray(m.faces, np.int64), process=False)
    D = float(np.linalg.norm(np.ptp(m.vertices, 0))) or 1.0
    m.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=max(0, int(np.ceil(-np.log10(D * 1e-7)))))
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    V = np.asarray(m.vertices, float)
    F = np.asarray(m.faces, np.int64)
    if len(V) < n_in:
        log(f"welded {n_in} -> {len(V)} vertices (UV seams / duplicate positions)")
    return V, F


def face_stuff(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    return (a + b + c) / 3, cr / np.maximum(2 * area[:, None], 1e-30), area


def face_adjacency_graph(F, nv):
    """Faces connected when they share a vertex (cheap, good enough for regions)."""
    nf = len(F)
    rows = np.repeat(np.arange(nf), 3)
    M = sp.csr_matrix((np.ones(3 * nf), (rows, F.ravel())), shape=(nf, nv))
    return M


def components_faces(F, nv, mask=None):
    """Connected components of faces (shared vertices). Returns labels (-1 = masked out)."""
    idx = np.arange(len(F)) if mask is None else np.flatnonzero(mask)
    sub = F[idx]
    # union through vertices: vertex graph of the sub-faces
    e = np.concatenate([sub[:, [0, 1]], sub[:, [1, 2]], sub[:, [2, 0]]])
    G = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv))
    _, vlab = connected_components(G, directed=False)
    lab = np.full(len(F), -1)
    lab[idx] = vlab[sub[:, 0]]
    return lab


def _plane_frame(n):
    u = unit(np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0]))
    return u, np.cross(n, u)


def _quad_design(P, o, n):
    u, v = _plane_frame(n)
    a, b = (P - o) @ u, (P - o) @ v
    return np.stack([np.ones_like(a), a, b, a * a, a * b, b * b], 1)


def components_faces_edge(F, manifold_only=False):
    """Connected components of faces that share an EDGE (pieces touching only at a
    vertex are separate; with manifold_only, also pieces joined only through edges
    with 3+ faces).  Returns a label per face."""
    nf = len(F)
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    fid = np.tile(np.arange(nf), 3)
    _, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    ne = inv.max() + 1
    ok = cnt[inv] == 2 if manifold_only else np.ones(len(inv), bool)
    M = sp.csr_matrix((np.ones(ok.sum()), (fid[ok], inv[ok])), shape=(nf, ne))
    A = M @ M.T
    _, lab = connected_components(A, directed=False)
    return lab


def remove_table(V, F, rng, info):
    """Remove the table / ground the hand lies on (up to 2 surfaces).

    RANSAC finds the dominant plane; the table surface is then modelled as a
    low-order height field over that plane (quadric: photogrammetry tables bow by
    a few mm) fitted robustly to its inliers, with the inlier band set from the
    inlier spread (MAD), so noisy (0.8 mm) or warped (2 mm bowl) tables are found.
    A region is accepted as a table when it is large and flat, and either very
    flat (old rule) or it spans the whole scene in plan (its in-plane bounding
    box covers most of the mesh's footprint: a table extends past the hand, the
    hand's own flat palm never does).  Returns V, F and the removed surfaces
    [{o, n, coef, thr, keep_sign}] (keep_sign: side of the plane the hand is on)."""
    planes = []
    for rep in range(2):
        C, N, A = face_stuff(V, F)
        if len(F) < 100:
            break
        D = np.linalg.norm(V.max(0) - V.min(0))
        p = A / A.sum()
        best = (0, None)
        thr0 = 0.004 * D
        for it in range(400):
            i = rng.choice(len(F), 3, p=p, replace=False)
            n = np.cross(C[i[1]] - C[i[0]], C[i[2]] - C[i[0]])
            if np.linalg.norm(n) < 1e-12:
                continue
            n = unit(n)
            dist = np.abs((C - C[i[0]]) @ n)
            sc = A[dist < thr0].sum()
            if sc > best[0]:
                best = (sc, (n, C[i[0]]))
        if best[1] is None:
            break
        n, o = best[1]
        thr = thr0
        coef = np.zeros(6)
        sig = np.inf
        for k in range(6):
            res = (C - o) @ n - _quad_design(C, o, n) @ coef
            inl = np.abs(res) < thr
            if inl.sum() < 10:
                break
            if k < 2:      # plane refit first (stable frame), then the quadric height field
                P = C[inl]
                o = P.mean(0)
                _, _, vt = np.linalg.svd(P - o, full_matrices=False)
                n = vt[2]
                coef = np.zeros(6)
                r = (P - o) @ n
            else:
                X = _quad_design(C[inl], o, n)
                y = (C[inl] - o) @ n
                wts = np.sqrt(A[inl] / A[inl].mean())
                coef = np.linalg.lstsq(X * wts[:, None], y * wts, rcond=None)[0]
                r = y - X @ coef
            sig = 1.4826 * np.median(np.abs(r - np.median(r)))
            thr = float(np.clip(3.5 * sig, 0.0008 * D, 2.0 * thr0))
        res = (C - o) @ n - _quad_design(C, o, n) @ coef
        inl = np.abs(res) < thr
        if inl.sum() == 0:
            break
        lab = components_faces(F, len(V), inl)
        areas = np.bincount(lab[inl], weights=A[inl])
        big = np.argmax(areas)
        reg = lab == big
        P = C[reg]
        u, v = _plane_frame(n)
        ext = np.ptp(P @ u) * np.ptp(P @ v)
        frac = A[reg].sum() / A.sum()
        fill = A[reg].sum() / max(ext, 1e-30)
        rr = res[reg]
        sig = 1.4826 * np.median(np.abs(rr - np.median(rr)))
        # plan coverage: how much of the rest of the mesh lies inside the region's plan bbox
        rest = ~reg
        Q = C[rest]
        lo_u, hi_u = np.percentile(P @ u, [1, 99])
        lo_v, hi_v = np.percentile(P @ v, [1, 99])
        inside = ((Q @ u >= lo_u) & (Q @ u <= hi_u) & (Q @ v >= lo_v) & (Q @ v <= hi_v))
        cover = float((A[rest] * inside).sum() / max(A[rest].sum(), 1e-30))
        bow = float(np.ptp(_quad_design(P, o, n) @ coef)) if np.any(coef) else 0.0
        # a table has the hand on ONE side only; a curved palm region has the rest of
        # the hand (back, thumb) on both sides of it
        hr = res[rest]
        far = np.abs(hr) > 3 * thr
        a_pos = float(A[rest][far & (hr > 0)].sum())
        a_neg = float(A[rest][far & (hr < 0)].sum())
        two_side = min(a_pos, a_neg) / max(a_pos + a_neg, 1e-30)
        log(f"plane {rep}: region area frac {frac:.3f} fill {fill:.2f} thr {thr:.4g} "
            f"sig/thr0 {sig / thr0:.3f} plan cover {cover:.2f} bow {bow / thr0:.2f} thr0 "
            f"two-sided {two_side:.3f}")
        flat_old = frac >= 0.12 and fill >= 0.3 and sig <= 0.2 * thr0
        table_like = (frac >= 0.25 and fill >= 0.4 and cover >= 0.85 and sig <= 0.5 * thr0 and two_side < 0.05
                      and rep == 0)      # a 2nd surface (wall, 2nd table) must pass the strict flat rule
        if not (flat_old or table_like):
            break
        # which side is the hand on: area-weighted signed height of the remaining faces
        hs = res[rest]
        keep_sign = 1.0 if (A[rest] * np.sign(hs)).sum() >= 0 else -1.0
        F = F[~reg]
        planes.append({"o": o, "n": n, "coef": coef, "thr": thr, "keep_sign": keep_sign,
                       "sig": float(sig)})
        info.setdefault("planes_removed", []).append(
            {"normal": [round(float(x), 4) for x in n], "area_frac": round(float(frac), 3),
             "thr": float(thr), "sig": float(sig), "plan_cover": round(cover, 3),
             "bow": round(bow, 5), "two_sided": round(two_side, 4),
             "rule": "flat" if flat_old else "table"})
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F], planes


def keep_largest(V, F, info):
    lab = components_faces(F, len(V))
    _, _, A = face_stuff(V, F)
    areas = np.bincount(lab, weights=A)
    big = np.argmax(areas)
    info["components"] = int((areas > 0).sum())
    info["kept_area_frac"] = round(float(areas[big] / areas.sum()), 4)
    F = F[lab == big]
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F]


def taubin(V, F, it=8, lamb=0.5, mu=-0.53):
    nv = len(V)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    L = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv))
    L.data[:] = 1.0
    deg = np.asarray(L.sum(1)).ravel()
    V = V.copy()
    for _ in range(it):
        for f in (lamb, mu):
            V = V + f * (L @ V / deg[:, None] - V)
    return V


def fill_all_holes(bm, bmesh, rounds=6):
    """Fill every boundary loop. Loops that touch themselves (bow-tie vertices with
    more than two boundary edges) are not filled by holes_fill; erode the faces at
    such vertices and retry. Last resort: fan-fill each remaining loop."""
    for r in range(rounds):
        bnd = [e for e in bm.edges if e.is_boundary]
        if not bnd:
            return 0
        ret = bmesh.ops.holes_fill(bm, edges=bnd, sides=0)
        for f in ret.get("faces", []):
            f.material_index = 0          # mark as filled (no scan data)
        bnd = [e for e in bm.edges if e.is_boundary]
        if not bnd:
            return 0
        bad = set()
        for e in bnd:
            for v in e.verts:
                if sum(1 for x in v.link_edges if x.is_boundary) > 2 or not v.is_manifold:
                    bad.add(v)
        faces = {f for v in bad for f in v.link_faces}
        if not faces:   # simple loops holes_fill refused: erode one ring anyway
            faces = {f for e in bnd for f in e.link_faces}
        bmesh.ops.delete(bm, geom=list(faces), context="FACES")
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bnd = [e for e in bm.edges if e.is_boundary]
    return len(bnd)


def bmesh_area(bm):
    return float(sum(f.calc_area() for f in bm.faces))


def skirt(bm, bmesh, o, n, hlim, drop):
    """Extrude every boundary loop whose median height above plane (o, n) is below
    hlim straight down (along -n) to `drop` below the plane."""
    o, n = np.asarray(o, float), unit(np.asarray(n, float))
    bnd = [e for e in bm.edges if e.is_boundary]
    if not bnd:
        return 0
    bm.verts.index_update()
    ev = np.array([[e.verts[0].index, e.verts[1].index] for e in bnd])
    nv = len(bm.verts)
    G = sp.coo_matrix((np.ones(len(ev)), (ev[:, 0], ev[:, 1])), shape=(nv, nv))
    _, lab = connected_components(G, directed=False)
    co = np.array([v.co[:] for v in bm.verts])
    h = (co - o) @ n
    elab = lab[ev[:, 0]]
    nloops = 0
    for L in np.unique(elab):
        sel = np.flatnonzero(elab == L)
        vids = np.unique(ev[sel])
        if np.median(h[vids]) > hlim:
            continue
        edges = [bnd[i] for i in sel]
        ret = bmesh.ops.extrude_edge_only(bm, edges=edges)
        newv = [g for g in ret["geom"] if isinstance(g, bmesh.types.BMVert)]
        for v in newv:
            p = np.array(v.co[:])
            v.co = (p - ((p - o) @ n + drop) * n).tolist()
        for f in (g for g in ret["geom"] if isinstance(g, bmesh.types.BMFace)):
            f.material_index = 0
        nloops += 1
    return nloops


def bpy_rebuild(V, F, voxel, target_verts, plane=None, smooth_it=6, cuts=(), stats=None,
                genus_retry=False, use_skirt=True):
    """Blender: optional plane cuts (keep the side each normal points to, cap it),
    fill holes, voxel remesh (watertight), decimate. Returns V, F (triangles).
    `cuts` (table caps) are applied and filled one after another BEFORE `plane`
    (forearm cut), so every cap is a planar loop (one combined L-shaped loop over
    table + forearm opening is what folded the old lid).  `stats` (dict) receives
    the filled area fraction and the Euler number of the remeshed surface; with
    genus_retry the remesh is repeated once at 0.6x voxel when it is not genus 0."""
    import bpy
    import bmesh
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bm = bmesh.new()
    vs = [bm.verts.new(p) for p in V.tolist()]
    bm.verts.ensure_lookup_table()
    for f in F.tolist():
        try:
            bm.faces.new([vs[i] for i in f])
        except ValueError:
            pass
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=voxel * 0.01)
    for f in bm.faces:            # scan faces = material 1; faces made by hole filling get 0
        f.material_index = 1
    a_in = bmesh_area(bm)
    a_fill = 0.0
    seq = [tuple(c) for c in cuts] + ([tuple(plane)] if plane is not None else [])
    if not seq:
        seq = [None]
    Dm = float(np.linalg.norm(np.ptp(V, 0)))
    for ci, cut in enumerate(seq):
        if cut is not None and ci < len(cuts) and use_skirt:
            # table cut: the opening where the hand lay on the table is a ragged,
            # non-planar rim.  Drop a vertical skirt from every rim loop that runs
            # near the table down through the cut plane, so the cut leaves a clean
            # planar loop (the hand's footprint) that fills as a flat cap; filling
            # the rim directly folds a lid through the hand.
            o, n = cut
            skirt(bm, bmesh, o, n, hlim=0.04 * Dm, drop=2.0 * voxel)
        if cut is not None:
            o, n = cut
            geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
            bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-6 * voxel, plane_co=o, plane_no=n,
                                   clear_inner=True)
            loose = [v for v in bm.verts if not v.link_faces]
            if loose:
                bmesh.ops.delete(bm, geom=loose, context="VERTS")
        a0 = bmesh_area(bm)
        n_open = fill_all_holes(bm, bmesh)
        a_fill += max(bmesh_area(bm) - a0, 0.0)
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")
    if True:
        # the cuts leave table specks (noise above the cut plane) as small separate
        # pieces and hole filling makes fins on bow-tie loops; the voxel remesh would
        # weld those that touch the skin into handles.  Keep the largest manifold
        # piece (split at vertices and at edges with 3+ faces), re-close, repeat.
        for _ in range(3):
            bm.verts.index_update()
            Ft = np.array([[v.index for v in f.verts] for f in bm.faces], np.int64)
            lab = components_faces_edge(Ft, manifold_only=True)
            big = np.argmax(np.bincount(lab))
            bm.faces.ensure_lookup_table()
            drop = [bm.faces[i] for i in np.flatnonzero(lab != big)]
            if not drop:
                break
            bmesh.ops.delete(bm, geom=drop, context="FACES")
            loose = [v for v in bm.verts if not v.link_faces]
            if loose:
                bmesh.ops.delete(bm, geom=loose, context="VERTS")
            a0 = bmesh_area(bm)
            n_open = fill_all_holes(bm, bmesh)
            a_fill += max(bmesh_area(bm) - a0, 0.0)
            bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")
    a_out = bmesh_area(bm)
    # sample points on the filled (no-data) faces, about one per voxel^2
    fill_pts = []
    for f in bm.faces:
        if f.material_index == 0:
            co = np.array([v.co[:] for v in f.verts])
            k = int(min(200, 1 + f.calc_area() / voxel ** 2))
            r = np.random.default_rng(len(fill_pts)).dirichlet(np.ones(len(co)), k)
            fill_pts.append(r @ co)
    fill_pts = np.concatenate(fill_pts) if fill_pts else np.zeros((0, 3))
    n_nonman = sum(1 for e in bm.edges if not e.is_manifold)
    if os.environ.get("FIT_DUMP_FILLED"):
        Vd = np.array([v.co[:] for v in bm.verts]); bm.verts.index_update()
        Fd = np.array([[v.index for v in f.verts] for f in bm.faces])
        np.savez(os.environ["FIT_DUMP_FILLED"] + f"_{len(Vd)}.npz", V=Vd, F=Fd)
    me = bpy.data.meshes.new("m")
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("m", me)
    bpy.context.scene.collection.objects.link(ob)
    rm = ob.modifiers.new("remesh", "REMESH")
    rm.mode = "VOXEL"
    rm.voxel_size = voxel
    rm.adaptivity = 0.0

    def _remesh():
        dg = bpy.context.evaluated_depsgraph_get()
        em = ob.evaluated_get(dg).to_mesh()
        Vr = np.array([v.co[:] for v in em.vertices])
        em.calc_loop_triangles()
        Fr = np.array([t.vertices[:] for t in em.loop_triangles], np.int64)
        ob.evaluated_get(dg).to_mesh_clear()
        if len(Fr) == 0:
            raise RuntimeError("voxel remesh produced an empty mesh")
        # keep largest piece (voxel remesh can leave dust)
        return keep_largest(Vr, Fr, {})

    V2, F2 = _remesh()
    eul = int(len(V2) - len(np.unique(np.sort(np.concatenate([F2[:, [0, 1]], F2[:, [1, 2]], F2[:, [2, 0]]]), 1), axis=0)) + len(F2))
    eul0, vox_used = eul, voxel
    if genus_retry and eul != 2:
        rm.voxel_size = 0.6 * voxel
        V3, F3 = _remesh()
        e3 = int(len(V3) - len(np.unique(np.sort(np.concatenate([F3[:, [0, 1]], F3[:, [1, 2]], F3[:, [2, 0]]]), 1), axis=0)) + len(F3))
        log(f"remesh Euler {eul} at voxel {voxel:.3g}; retry at {0.6 * voxel:.3g}: Euler {e3}")
        if abs(e3 - 2) < abs(eul - 2):
            V2, F2, eul, vox_used = V3, F3, e3, 0.6 * voxel
    if stats is not None:
        stats["_fill_pts"] = fill_pts
        stats.update({"open_edges_after_fill": int(n_open), "nonmanifold_edges": int(n_nonman),
                      "filled_area_frac": round(a_fill / max(a_out, 1e-30), 4),
                      "input_area_kept_frac": round((a_out - a_fill) / max(a_in, 1e-30), 4),
                      "remesh_euler": eul, "remesh_euler_first": eul0, "remesh_voxel": vox_used})
    if len(V2) < 2000:
        raise RuntimeError(f"voxel remesh failed ({len(V2)} vertices): the mesh could not be closed")
    V2 = taubin(V2, F2, it=smooth_it)
    if len(V2) > target_verts:
        me2 = bpy.data.meshes.new("d")
        me2.from_pydata(V2.tolist(), [], F2.tolist())
        ob2 = bpy.data.objects.new("d", me2)
        bpy.context.scene.collection.objects.link(ob2)
        dm = ob2.modifiers.new("dec", "DECIMATE")
        dm.ratio = target_verts / len(V2)
        dm.use_collapse_triangulate = True
        dg = bpy.context.evaluated_depsgraph_get()
        em = ob2.evaluated_get(dg).to_mesh()
        V2 = np.array([v.co[:] for v in em.vertices])
        em.calc_loop_triangles()
        F2 = np.array([t.vertices[:] for t in em.loop_triangles], np.int64)
    return V2, F2


def mesh_check(V, F):
    import trimesh
    m = trimesh.Trimesh(V, F, process=False)
    return {"vertices": int(len(V)), "faces": int(len(F)), "watertight": bool(m.is_watertight),
            "winding_consistent": bool(m.is_winding_consistent), "euler": int(m.euler_number),
            "volume_ml": round(float(abs(m.volume)) / 1000.0, 1) if m.is_watertight else None}


def orient_outward(V, F):
    import trimesh
    m = trimesh.Trimesh(V, F, process=False)
    trimesh.repair.fix_normals(m)
    if m.is_watertight and m.volume < 0:
        m.invert()
    return np.asarray(m.vertices), np.asarray(m.faces)


# ======================================================================= geometry helpers
class Surf:
    """Triangle mesh with edge graph, KD-trees and slicing / ray helpers."""

    def __init__(self, V, F):
        self.V = np.asarray(V, float)
        self.F = np.asarray(F, np.int64)
        e = np.concatenate([self.F[:, [0, 1]], self.F[:, [1, 2]], self.F[:, [2, 0]]])
        e = np.unique(np.sort(e, 1), axis=0)
        self.E = e
        w = np.linalg.norm(self.V[e[:, 0]] - self.V[e[:, 1]], axis=1)
        self.h = float(np.mean(w))
        n = len(self.V)
        self.G = sp.csr_matrix((np.concatenate([w, w]), (np.concatenate([e[:, 0], e[:, 1]]),
                                                         np.concatenate([e[:, 1], e[:, 0]]))),
                               shape=(n, n))
        self.C, self.N, self.A = face_stuff(self.V, self.F)
        self.ctree = cKDTree(self.C)
        self.vtree = cKDTree(self.V)
        self.rmax = float(np.max(np.linalg.norm(self.V[self.F] - self.C[:, None], axis=2)))

    def geo(self, src, limit=np.inf):
        return dijkstra(self.G, indices=src, limit=limit)

    def local_maxima(self, d):
        G = self.G.tocoo()
        nb = np.full(len(self.V), -np.inf)
        np.maximum.at(nb, G.row, d[G.col])
        return np.flatnonzero(d >= nb)

    # ---- planar cross-section: loop around point p in plane (p, n)
    @staticmethod
    def _split_loop(q, ref_area, p2=np.zeros(2)):
        """Touching fingers give one peanut-shaped loop around both.  Cut the loop at
        its narrowest neck (two points close in space but far apart along the loop)
        while it is much bigger than ref_area; keep the part around p2."""
        for _ in range(5):
            x, y = q[:, 0], q[:, 1]
            Ar = abs((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()) / 2
            N = len(q)
            if Ar < 1.4 * ref_area or N < 16:
                return q, False
            D = np.linalg.norm(q[:, None] - q[None], axis=2)
            i, j = np.triu_indices(N, 1)
            sep = np.minimum(j - i, N - (j - i))
            ok = sep >= N // 6
            if not ok.any():
                return q, False
            k = np.argmin(np.where(ok, D[i, j], np.inf))
            a, b = i[k], j[k]
            if D[a, b] > 0.8 * np.sqrt(ref_area):      # no neck: a genuinely wide section
                return q, False
            parts = [q[a:b + 1], np.concatenate([q[b:], q[:a + 1]])]

            def wind(qq):
                ang = np.arctan2(qq[:, 1] - p2[1], qq[:, 0] - p2[0])
                da = np.diff(np.concatenate([ang, ang[:1]]))
                da = (da + np.pi) % (2 * np.pi) - np.pi
                return abs(da.sum()) / (2 * np.pi)
            w = [wind(pp) for pp in parts]
            if max(w) < 0.5:          # p outside both: take the one whose centroid is nearer
                w = [-np.linalg.norm(pp.mean(0) - p2) for pp in parts]
            q = parts[int(np.argmax(w))]
        return q, True

    def section(self, p, n, R, ref_area=None):
        n = unit(n)
        fi = self.ctree.query_ball_point(p, R + self.rmax)
        if len(fi) == 0:
            return None
        fi = np.asarray(fi)
        Fs = self.F[fi]
        s = (self.V - p) @ n
        sf = s[Fs]
        pos = sf > 0
        npos = pos.sum(1)
        cut = (npos == 1) | (npos == 2)
        Fs = Fs[cut]
        if len(Fs) < 3:
            return None
        sf = s[Fs]
        pos = sf > 0
        # each crossing face: the two edges whose endpoints differ in sign
        segs = []
        keys = []
        for (i, j) in ((0, 1), (1, 2), (2, 0)):
            m = pos[:, i] != pos[:, j]
            keys.append(np.where(m[:, None], np.sort(Fs[:, [i, j]], 1), -1))
        K = np.stack(keys, 1)                       # (m, 3, 2)
        has = K[:, :, 0] >= 0
        # pick the two crossing edges per face
        idx = np.argsort(~has, axis=1, kind="stable")[:, :2]
        k1 = K[np.arange(len(K)), idx[:, 0]]
        k2 = K[np.arange(len(K)), idx[:, 1]]
        nv = len(self.V)
        a1 = k1[:, 0] * nv + k1[:, 1]
        a2 = k2[:, 0] * nv + k2[:, 1]
        allk, inv = np.unique(np.concatenate([a1, a2]), return_inverse=True)
        ea, eb = allk // nv, allk % nv
        t = s[ea] / (s[ea] - s[eb])
        P = self.V[ea] + t[:, None] * (self.V[eb] - self.V[ea])
        m = len(a1)
        u1, u2 = inv[:m], inv[m:]
        # graph of crossing points, components = loops (or open chains at R)
        G = sp.coo_matrix((np.ones(m), (u1, u2)), shape=(len(allk), len(allk)))
        ncomp, lab = connected_components(G, directed=False)
        deg = np.bincount(np.concatenate([u1, u2]), minlength=len(allk))
        # 2D frame
        ax = unit(np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0]))
        ay = np.cross(n, ax)
        Q = np.stack([(P - p) @ ax, (P - p) @ ay], 1)
        best = None
        for c in range(ncomp):
            sel = lab == c
            if sel.sum() < 4 or np.any(deg[sel] != 2):
                continue  # open chain (left the ball) or degenerate
            # order the loop
            ids = np.flatnonzero(sel)
            adj = {}
            for x, y in zip(u1, u2):
                if sel[x]:
                    adj.setdefault(x, []).append(y)
                    adj.setdefault(y, []).append(x)
            order = [ids[0]]
            prev = -1
            cur = ids[0]
            for _ in range(len(ids) - 1):
                nx = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
                prev, cur = cur, nx
                order.append(cur)
            q = Q[order]
            split = False
            if ref_area is not None:
                q, split = self._split_loop(q, ref_area)
            # winding number of origin (p)
            ang = np.arctan2(q[:, 1], q[:, 0])
            dang = np.diff(np.concatenate([ang, ang[:1]]))
            dang = (dang + np.pi) % (2 * np.pi) - np.pi
            wn = abs(dang.sum()) / (2 * np.pi)
            if wn < 0.5:
                continue
            x, y = q[:, 0], q[:, 1]
            x1, y1 = np.roll(x, -1), np.roll(y, -1)
            cr = x * y1 - x1 * y
            Ar = cr.sum() / 2
            if abs(Ar) < 1e-12:
                continue
            cx = ((x + x1) * cr).sum() / (6 * Ar)
            cy = ((y + y1) * cr).sum() / (6 * Ar)
            if best is None or abs(Ar) < best["area"]:
                best = {"area": abs(Ar), "centroid": p + cx * ax + cy * ay,
                        "pts": p + q[:, :1] * ax + q[:, 1:] * ay,
                        "perim": float(np.linalg.norm(q - np.roll(q, -1, 0), axis=1).sum()),
                        "split": split}
        return best

    # ---- ray cast (first hit, t > 0)
    def ray(self, o, d, R=200.0):
        d = unit(d)
        # candidate faces: near the ray segment
        fi = np.flatnonzero(np.linalg.norm(np.cross(self.C - o, d), axis=1) < self.rmax * 1.5 + 1e-9)
        if len(fi) == 0:
            return None
        T = self.V[self.F[fi]]
        e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
        pv = np.cross(d, e2)
        det = (e1 * pv).sum(1)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
        tv = o - T[:, 0]
        u = (tv * pv).sum(1) * inv
        qv = np.cross(tv, e1)
        v = (qv @ d) * inv
        t = (qv * e2).sum(1) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9) & (t < R)
        if not hit.any():
            return None
        return float(t[hit].min())

    def ray_all(self, o, d, R=400.0):
        d = unit(d)
        fi = np.flatnonzero(np.linalg.norm(np.cross(self.C - o, d), axis=1) < self.rmax * 1.5 + 1e-9)
        T = self.V[self.F[fi]]
        e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
        pv = np.cross(d, e2)
        det = (e1 * pv).sum(1)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
        tv = o - T[:, 0]
        u = (tv * pv).sum(1) * inv
        qv = np.cross(tv, e1)
        v = (qv @ d) * inv
        t = (qv * e2).sum(1) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (np.abs(t) < R)
        return np.sort(t[hit])

    def dist(self, P):
        """Approximate unsigned distance to the surface (nearest vertex / face centroid)."""
        d1, _ = self.vtree.query(P)
        d2, _ = self.ctree.query(P)
        return np.minimum(d1, d2)


# ======================================================================= step 2a: fingertips
def ring_profile(S, g, tmax, step=None):
    """Width w(t) (2*sqrt(2*lambda_max) of the geodesic ring at distance t) and ring
    centroids, for t = step .. tmax."""
    step = step or 1.5 * S.h
    order = np.argsort(g)
    gs = g[order]
    ts = np.arange(step, tmax, step)
    W, Cn, Nr = [], [], []
    for t in ts:
        a, b = np.searchsorted(gs, [t - 0.75 * S.h, t + 0.75 * S.h])
        idx = order[a:b]
        if len(idx) < 6:
            W.append(np.nan); Cn.append(np.full(3, np.nan)); Nr.append(len(idx))
            continue
        P = S.V[idx]
        c = P.mean(0)
        lam = np.linalg.eigvalsh(np.cov((P - c).T))
        W.append(2 * np.sqrt(2 * max(lam[-1], 0)))
        Cn.append(c)
        Nr.append(len(idx))
    return ts, np.array(W), np.array(Cn), np.array(Nr)


def w_star(ts, W, k=1.5):
    ok = np.isfinite(W) & (ts >= k * np.nan_to_num(W, nan=np.inf))
    if not ok.any():
        return np.inf, None
    i = np.flatnonzero(ok)[0]
    return float(W[i]), i


SPREAD_ADVICE = ("Check the scan: all five fingers must be separate (spread them so there is at least "
                 "5 mm of air between neighbouring fingers and between the thumb and the index), the "
                 "fingertips must be complete, and no table or background piece may stick out like a finger.")


def find_fingertips(S, info):
    """Five fingertip vertices, finger order thumb,index,middle,ring,pinky."""
    n = len(S.V)
    d0 = S.geo(int(np.argmin(np.linalg.norm(S.V - S.V.mean(0), axis=1))))
    a = int(np.argmax(d0))
    da = S.geo(a)
    b = int(np.argmax(da))
    db = S.geo(b)
    D = float(np.linalg.norm(np.ptp(S.V, 0)))
    rho = 0.04 * D
    cand = set()
    for dd in (d0, da, db):
        for c in S.local_maxima(dd):
            g = S.geo(int(c), limit=rho)
            if dd[c] >= dd[np.isfinite(g)].max() - 1e-9:
                cand.add(int(c))
    profs = []
    for c in sorted(cand):
        g = S.geo(c, limit=0.6 * D)
        profs.append((int(c), g, ring_profile(S, g, 0.6 * D)))

    def pick(k, win):
        tips = []
        for c, g, (ts, W, Cn, Nr) in profs:
            ws, i = w_star(ts, W, k)
            if np.isfinite(ws):
                # a finger stays thin for a while beyond t*: reject rims / corners / bumps
                seg = W[(ts >= ts[i]) & (ts <= ts[i] + win * ws)]
                if len(seg) == 0 or not np.all(np.isfinite(seg)) or seg.max() > 1.5 * ws:
                    ws = np.inf
            tips.append({"v": c, "w": ws, "t": (float(ts[i]) if i is not None else None),
                         "g": g, "prof": (ts, W, Cn)})
        tips.sort(key=lambda r: r["w"])
        # merge candidates on the same protrusion (geodesic distance < w*)
        uniq = []
        for r in tips:
            if not np.isfinite(r["w"]):
                continue
            if any(u["g"][r["v"]] < 1.2 * u["w"] for u in uniq):
                continue
            uniq.append(r)
        return uniq

    wmin = 0.015 * D          # a finger is >= ~0.027 D wide; thinner = cap / debris spike
    uniq = [r for r in pick(1.5, 1.2) if r["w"] >= wmin]
    info["tip_test"] = "strict"
    if len(uniq) < 5 or max(r["w"] for r in uniq[:5]) > 2.2 * min(r["w"] for r in uniq[:5]):
        # Fingers that touch: the geodesic rings of a finger spread onto its neighbour
        # just proximal of the contact, so the strict test (ring thin for 1.2 w beyond
        # t = 1.5 w) fails.  Relaxed test: t >= w, thin for 0.5 w; a relaxed candidate
        # must be about as wide as the strict ones (spikes are not).
        relaxed = [r for r in pick(1.0, 0.5) if r["w"] >= wmin]
        if len(uniq) >= 2:
            wm = float(np.median([r["w"] for r in uniq]))
            relaxed = [r for r in relaxed if 0.6 * wm <= r["w"] <= 1.7 * wm]
        merged = list(uniq)
        for r in relaxed:
            if not any(u["g"][r["v"]] < 1.2 * min(u["w"], r["w"]) for u in merged):
                merged.append(r)
        merged.sort(key=lambda r: r["w"])
        if len(merged) >= 5 and len(merged) > len(uniq):
            uniq = merged
            info["tip_test"] = "relaxed (fingers touching?)"
            log("WARNING: fingertips needed the relaxed test: fingers probably touch")
            info.setdefault("warnings", []).append(
                "fingertips found only with the relaxed test: fingers probably touch each other; "
                "joint positions near the contact are unreliable. Re-scan with the fingers spread "
                "(at least 5 mm gaps).")
    info["tip_candidates"] = [round(r["w"], 4) for r in uniq[:8]]
    if len(uniq) < 5:
        raise RuntimeError(f"found only {len(uniq)} finger-like extremities. " + SPREAD_ADVICE)
    five = uniq[:5]
    if max(r["w"] for r in five) > 2.2 * min(r["w"] for r in five):
        raise RuntimeError("the 5 fingertip candidates differ too much in width "
                           f"({', '.join(f'{r['w']:.3g}' for r in five)}): one is probably not a finger. "
                           + SPREAD_ADVICE)
    wmed = np.median([r["w"] for r in five])
    if len(uniq) > 5 and uniq[5]["w"] < 1.5 * wmed:
        log(f"WARNING: 6th finger-like extremity (w {uniq[5]['w']:.3g} vs median {wmed:.3g})")
        info.setdefault("warnings", []).append("ambiguous 6th fingertip candidate")
    for r in five:
        r["g"] = S.geo(r["v"])
    # thumb: its nearest other tip is geodesically farthest
    Gm = np.array([[r["g"][q["v"]] for q in five] for r in five])
    Gm = np.minimum(Gm, Gm.T)
    np.fill_diagonal(Gm, np.inf)
    nn = Gm.min(1)
    th = int(np.argmax(nn))
    rest = [i for i in range(5) if i != th]
    if info.get("tip_test", "strict") != "strict":
        # touching fingers short-cut the geodesics between tips: order the four
        # fingers by Euclidean nearest-neighbour chain from the thumb instead
        Pt = np.array([S.V[r["v"]] for r in five])
        Gm = np.linalg.norm(Pt[:, None] - Pt[None], axis=2)
        np.fill_diagonal(Gm, np.inf)
    idx = min(rest, key=lambda i: Gm[th, i])
    order = [idx]
    while len(order) < 4:
        cur = order[-1]
        nxt = min([i for i in rest if i not in order], key=lambda i: Gm[cur, i])
        order.append(nxt)
    fing = {"thumb": five[th]}
    for f, i in zip(FINGERS4, order):
        fing[f] = five[i]
    info["tip_geodesic_nn"] = [round(float(x), 4) for x in nn]
    info["w_star"] = {f: round(fing[f]["w"], 4) for f in FINGERS5}
    return fing


# ======================================================================= step 2b: centre-lines
def resample_polyline(P, step):
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] < step:
        return P[[0, -1]], s[[0, -1]]
    u = np.arange(0, s[-1] + 1e-9, step)
    return np.stack([np.interp(u, s, P[:, k]) for k in range(3)], 1), u


def smooth_polyline(P, k=2):
    Q = P.copy()
    for _ in range(k):
        Q[1:-1] = 0.25 * Q[:-2] + 0.5 * Q[1:-1] + 0.25 * Q[2:]
    return Q


def tangents(P):
    T = np.gradient(P, axis=0)
    return unit(T)


def finger_centreline(S, tip_v, w, step=1.25, tmax=140.0):
    """Centre-line of one finger from the tip toward the palm, by planar sections.
    Returns dict with points (tip -> proximal), tangents, areas, widths, loops."""
    g = S.geo(tip_v, limit=tmax + 10)
    ts, W, Cn, Nr = ring_profile(S, g, tmax, step=step)
    ok = np.isfinite(W)
    ref = np.nanmedian(W[(ts > 0.6 * w) & (ts < 1.8 * w)]) if np.any((ts > 0.6 * w) & (ts < 1.8 * w)) else w
    stop = len(ts)
    for i in range(len(ts)):
        if ts[i] < 0.6 * w:
            continue
        if not ok[i] or W[i] > 1.6 * ref:
            stop = i
            break
    sel = (ts >= 0.35 * w) & (np.arange(len(ts)) < stop) & ok
    P = Cn[sel]
    if len(P) < 5:
        raise RuntimeError("finger centre-line too short. " + SPREAD_ADVICE)
    P = smooth_polyline(P, 3)
    R = 1.3 * w
    # touching fingers (relaxed fingertip test): split merged sections at their neck
    split = getattr(S, "split_touching", False)
    aref = 0.67 * w * w if split else None
    if split:              # the merged loop around two fingers must close inside the ball
        R = 4.0 * w
    for it in range(3):
        P, _ = resample_polyline(P, step)
        T = tangents(smooth_polyline(P, 4))
        out, area, loops = [], [], []
        for p, t in zip(P, T):
            sec = S.section(p, t, R, ref_area=aref)
            if sec is None:
                break
            out.append(sec["centroid"])
            area.append(sec["area"])
            loops.append(sec)
        n = len(out)
        if n < 5:
            raise RuntimeError("finger sections failed. " + SPREAD_ADVICE)
        area = np.array(area)
        amed = np.median(area[: max(5, n // 2)])
        cut = n
        for i in range(n):
            if i > 3 and area[i] > 1.7 * amed and (i + 1 >= n or area[i + 1] > 1.7 * amed):
                cut = i
                break
        P = smooth_polyline(np.array(out[:cut]), 1)
        # extend proximally along the last tangent while the section stays finger-like
        if it == 2:
            break
        T = tangents(smooth_polyline(P, 4))
        p, t = P[-1].copy(), T[-1].copy()
        ext = []
        for k in range(40):
            p = p + step * t
            ra = amed if split else None
            sec = S.section(p, t, R, ref_area=ra)
            if sec is None or sec["area"] > 1.7 * amed:
                sec2 = S.section(p + step * t, t, R, ref_area=ra)
                if sec is None or sec2 is None or sec2["area"] > 1.7 * amed:
                    break
            c = sec["centroid"]
            ext.append(c)
            if len(ext) >= 3:
                t = unit(0.7 * t + 0.3 * unit(ext[-1] - ext[-3]))
            p = c
        if ext:
            P = np.concatenate([P, np.array(ext)])
    T = tangents(smooth_polyline(P, 4))
    return {"P": P, "T": T, "area": np.array([l["area"] for l in loops[:len(P)]]),
            "loops": loops[:len(P)], "amed": amed}


# ======================================================================= step 3: joints
def arclen(P):
    return np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def point_at_chord(P, u, start, L, u0):
    """First point on polyline P (arc length u), beyond arc length u0, at Euclidean
    distance L from `start`. Extrapolates along the last segment if needed.
    Returns (point, arc length)."""
    d = np.linalg.norm(P - start, axis=1)
    for i in range(len(P) - 1):
        if u[i + 1] <= u0:
            continue
        if d[i] <= L <= d[i + 1] or (d[i] < L and i == len(P) - 2):
            a, b = P[i], P[i + 1]
            # solve |a + t (b - a) - start| = L
            ab = b - a
            qa = ab @ ab
            qb = 2 * ab @ (a - start)
            qc = (a - start) @ (a - start) - L * L
            disc = max(qb * qb - 4 * qa * qc, 0)
            t = (-qb + np.sqrt(disc)) / (2 * qa)
            return a + t * ab, u[i] + t * np.linalg.norm(ab)
    return None, None


def section_widths(S, P, T, pal, R, ref_area=None):
    """Lateral width, dorsal and palmar heights of the sections along a centre-line."""
    out = []
    for p, t in zip(P, T):
        sec = S.section(p, t, R, ref_area=ref_area)
        if sec is None:
            out.append((np.nan, np.nan, np.nan))
            continue
        q = sec["pts"] - sec["centroid"]
        pz = unit(pal - (pal @ t) * t)
        lat = np.cross(t, pz)
        ql, qz = q @ lat, q @ pz
        # robust extents (percentiles: scan noise inflates max/min)
        out.append((np.percentile(ql, 98) - np.percentile(ql, 2), -np.percentile(qz, 2), np.percentile(qz, 98)))
    out = np.array(out)
    # light smoothing along the centre-line (sigma ~ 1 mm at 1.25 mm spacing)
    k = np.array([0.25, 0.5, 0.25])
    for j in range(out.shape[1]):
        col = out[:, j]
        ok = np.isfinite(col)
        if ok.sum() > 3:
            f = np.interp(np.arange(len(col)), np.flatnonzero(ok), col[ok])
            f = np.convolve(np.pad(f, 1, mode="edge"), k, mode="valid")
            out[:, j] = np.where(ok, f, np.nan)
    return out


def find_peak(u, w, u_pred, win, min_prom, span=8.0):
    """Most prominent local maximum of w within [u_pred - win, u_pred + win]."""
    best = None
    for i in range(1, len(w) - 1):
        if not (abs(u[i] - u_pred) <= win and np.isfinite(w[i])):
            continue
        if not (w[i] >= w[i - 1] and w[i] >= w[i + 1]):
            continue
        # prominence: drop to the minima within 8 mm either side
        lw = w[(u >= u[i] - span) & (u < u[i])]
        rw = w[(u > u[i]) & (u <= u[i] + span)]
        if len(lw) == 0 or len(rw) == 0:
            continue
        prom = w[i] - max(np.nanmin(lw), np.nanmin(rw))
        if prom >= min_prom and (best is None or prom > best[1]):
            # parabolic refinement
            den = w[i - 1] - 2 * w[i] + w[i + 1]
            off = 0.5 * (w[i - 1] - w[i + 1]) / den if abs(den) > 1e-12 else 0.0
            off = float(np.clip(off, -1, 1))
            best = (u[i] + off * (u[i + 1] - u[i]), prom)
    return best


def fit_line(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[0]


class FingerFit:
    def __init__(self, S, name, cl, pal):
        self.S, self.f, self.cl = S, name, cl
        P, T = cl["P"], cl["T"]
        self.P, self.T, self.u = P, T, arclen(P)
        # distal axis from the first ~10 units of the centre-line
        k = max(4, int(np.searchsorted(self.u, 9.0)))
        c, d = fit_line(P[:k])
        if d @ (P[0] - P[k - 1]) < 0:
            d = -d
        self.d_dist = d                              # points distally
        o = P[min(2, len(P) - 1)]
        t = S.ray(o, d)
        self.apex = o + t * d if t is not None else P[0] + 5 * d
        self.prof = None
        self.pal = pal
        self.cscale = 1.0          # mesh unit -> mm (set by place_all)

    def profile(self, R):
        ra = self.cl.get("amed") if getattr(self.S, "split_touching", False) else None
        self.prof = section_widths(self.S, self.P, self.T, self.pal, R, ref_area=ra)
        return self.prof

    def support_axis(self, base, d, r, iters=6):
        """Distal axis through `base` (the DIP / IP): iterate d <- unit(apex(d) - base),
        apex(d) = skin point near the tip farthest along d. Converges to the axis of a
        rounded finger end (contraction ~ r_tip / (|cap centre - base| + r_tip))."""
        S = self.S
        apex = self.apex
        for _ in range(iters):
            idx = S.vtree.query_ball_point(apex, r)
            if len(idx) == 0:
                break
            Q = S.V[idx]
            if getattr(S, "split_touching", False):
                # a touching neighbour's tip lies inside the ball: keep this finger only
                wq = np.sqrt(self.cl["amed"] / 0.67)
                rel = Q - base
                lat = np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)
                Q = Q[lat < 0.6 * wq] if (lat < 0.6 * wq).sum() >= 3 else Q
            h = (Q - base) @ d
            cap = Q[h >= h.max() - 0.8 / self.cscale]     # centroid of the top 0.8 mm cap (noise-robust)
            apex = cap.mean(0)
            d = unit(apex - base)
        # exact apex: ray along the axis from the base
        t = S.ray(base, d)
        if t is not None:
            apex = base + t * d
        return d, apex

    def joint_from(self, Pc, uc, cur, ucur, L, si, use_evidence, sig_l_frac, min_prom):
        u = self.u
        w = self.prof[:, 0] if self.prof is not None else None
        q, uq = point_at_chord(Pc, uc, cur, L, ucur)
        rec = {"len_pred_u": None if uq is None else round(float(uq), 2)}
        if q is not None and use_evidence and w is not None:
            cs = self.cscale
            pk = find_peak(u, w, uq, max(4.0 / cs, 0.18 * L), min_prom / cs, 8.0 / cs)
            if pk is not None:
                ue, prom = pk
                ue = ue - (PEAK_OFF_THUMB if self.f == "thumb" else PEAK_OFF.get(si, 0.4)) / cs
                rec["peak_pt"] = np.array([np.interp(ue, uc, Pc[:, k]) for k in range(3)])
                sl = sig_l_frac * L
                sig_e = float(np.clip(0.25 / max(prom * cs, 1e-6), 0.3, 1.5)) / cs
                wt_e, wt_l = 1 / sig_e ** 2, 1 / sl ** 2
                uf = (wt_e * ue + wt_l * uq) / (wt_e + wt_l)
                q = np.array([np.interp(uf, uc, Pc[:, k]) for k in range(3)])
                rec.update({"peak_u": round(float(ue), 2), "prom": round(float(prom * cs), 3),
                            "used_u": round(float(uf), 2)})
                uq = uf
        if q is None:
            raise RuntimeError(f"{self.f}: centre-line too short for segment {si}")
        return q, uq, rec

    def place(self, lens, tip_inset, use_evidence=True, sig_l_frac=0.03, min_prom=0.25):
        """lens: segment lengths distal -> proximal (mesh units), e.g. long finger
        [distal, middle, proximal]; the last one is extrapolated along the fitted
        proximal axis. Returns joints distal -> proximal [tip, j1, ...] + evidence."""
        P, u = self.P, self.u
        d, apex = self.d_dist, self.apex
        r_cap = 12.0 / self.cscale
        for rep in range(3):
            tip = apex - tip_inset * d
            ut = float((tip - P[0]) @ (-d))
            Pc = np.concatenate([[tip], P]) if ut < 0 else P
            uc = np.concatenate([[ut], u]) if ut < 0 else u
            q1, u1, rec1 = self.joint_from(Pc, uc, tip, ut, lens[0], 0, use_evidence, sig_l_frac, min_prom)
            d_new, apex_new = self.support_axis(q1, d, r_cap)
            moved = np.linalg.norm(apex_new - apex)
            d, apex = d_new, apex_new
            if moved < 0.05 / self.cscale:
                break
        tip = apex - tip_inset * d
        ut = float((tip - P[0]) @ (-d))
        Pc = np.concatenate([[tip], P]) if ut < 0 else P
        uc = np.concatenate([[ut], u]) if ut < 0 else u
        self.d_tip, self.apex_fit = d, apex
        joints = [tip]
        ev = []
        cur, ucur = tip, ut
        for si, L in enumerate(lens):
            if si < len(lens) - 1:
                q, uq, rec = self.joint_from(Pc, uc, cur, ucur, L, si, use_evidence, sig_l_frac, min_prom)
                joints.append(q)
                ev.append(rec)
                cur, ucur = q, uq
            else:
                # proximal segment: extrapolate the line fitted on the free centre-line
                free = u[-1] - ucur
                seg = P[(u >= ucur) & (u <= ucur + max(PROX_FIT_FRAC * free, 8.0 / self.cscale))]
                if len(seg) >= 4 and free > 6 / self.cscale:
                    c, dd = fit_line(np.concatenate([[cur], seg]))
                    if dd @ (seg[-1] - cur) < 0:
                        dd = -dd
                else:
                    dd = self.T[min(np.searchsorted(u, ucur), len(u) - 1)]
                joints.append(cur + L * dd)
                ev.append({"free_len_mm": round(float(u[-1] - ucur) * self.cscale, 2)})
        return joints, ev


def palm_normal_est(S, J, along, r=30.0):
    """Palm normal (out of the palm) from the face normals of the palm region:
    dominant direction of the area-weighted normal moment (palmar and dorsal skin
    both contribute, +-n); sign from the right-hand rule across x along."""
    across = J["index_mcp"] - J["pinky_mcp"]
    pc = np.mean([J[f + "_mcp"] for f in FINGERS4], 0) - 30.0 * along
    idx = np.asarray(S.ctree.query_ball_point(pc, r))
    N, A = S.N[idx], S.A[idx]
    M = (N * A[:, None]).T @ N
    w, v = np.linalg.eigh(M)
    n = v[:, -1]
    ref = np.cross(across, along)          # right hand: across x along = out of the palm
    return n if n @ ref > 0 else -n


def mcp_depth(S, m0, n, frac, lo_hi=(12.0, 50.0)):
    """Move m0 along n to `frac` of the local palm thickness, measured from the dorsal skin."""
    ts = S.ray_all(m0, n)
    if len(ts) < 2 or not (np.any(ts < 0) and np.any(ts > 0)):
        return m0, None
    lo, hi = ts[ts < 0].max(), ts[ts > 0].min()
    th = hi - lo
    if not (lo_hi[0] <= th <= lo_hi[1]):
        return m0, float(th)
    return m0 + n * (lo + frac * th), float(th)


def kabsch(A, B):
    """R, t with R @ A_i + t ~ B_i."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def fit_layout(J, n, lay, c):
    """Rigidly place the rig palm layout (scale 1/c) on the 4 MCPs + palm normal.
    Returns wrist, cmc (mesh units) and fit residuals."""
    A = np.array([lay["mcp"][f] / c for f in FINGERS4])
    B = np.array([J[f + "_mcp"] for f in FINGERS4])
    k = 25.0
    A2 = np.concatenate([A, A + k * np.array([0, 0, 1.0])])
    B2 = np.concatenate([B, B + k * n])
    R, t = kabsch(A2, B2)
    res = np.linalg.norm((A @ R.T + t) - B, axis=1)
    return R @ np.zeros(3) + t, R @ (lay["cmc"] / c) + t, R, t, res


def wrist_free_lengths(J, n, lay, c, R, w_sec, s_m=0.7, s_len=float(os.environ.get('FIT_WRIST_SL', 0.04)),
                       s_w=float(os.environ.get('FIT_WRIST_SW', 1.0)), iters=4):
    """Wrist from the four metacarpal lines: MCP_i = w + L_i * R d_i, with the rig
    layout directions d_i, free lengths L_i (prior: rig length, sigma s_len*L) and
    w_sec (rigid layout fit through the wrist section centroid) fixing the lateral
    and depth position (sigma s_w); only the axial position comes from the lines.
    Linear LS in (w, L_1..4); R is refitted to (scaled layout, wrist, normal) and the
    solve repeated. Returns wrist, R, t (layout -> mesh), MCP residuals, lengths."""
    d = np.array([unit(lay["mcp"][f]) for f in FINGERS4])
    L0 = np.array([np.linalg.norm(lay["mcp"][f]) / c for f in FINGERS4])
    M = np.array([J[f + "_mcp"] for f in FINGERS4])
    sm, sw = s_m / c, s_w / c
    for _ in range(iters):
        D = d @ R.T
        a = R @ np.array([0, 1.0, 0])        # wrist -> knuckle-row axis: left free;
        Pp = np.eye(3) - np.outer(a, a)      # lateral + depth held at w_sec
        rows, rhs = [], []
        for i in range(4):
            Ai = np.zeros((3, 7)); Ai[:, :3] = np.eye(3); Ai[:, 3 + i] = D[i]
            rows.append(Ai / sm); rhs.append(M[i] / sm)
            r = np.zeros((1, 7)); r[0, 3 + i] = 1.0
            sl = s_len * L0[i]
            rows.append(r / sl); rhs.append(np.array([L0[i] / sl]))
        Aw = np.zeros((3, 7)); Aw[:, :3] = Pp
        rows.append(Aw / sw); rhs.append(Pp @ w_sec / sw)
        x = np.linalg.lstsq(np.concatenate(rows), np.concatenate(rhs), rcond=None)[0]
        w, Lm = x[:3], x[3:]
        A = np.concatenate([d * Lm[:, None], np.zeros((1, 3))])
        k = 25.0 / c
        A2 = np.concatenate([A, A + k * np.array([0, 0, 1.0])])
        B = np.concatenate([M, w[None]])
        B2 = np.concatenate([B, B + k * n])
        R, t = kabsch(A2, B2)
    res = np.linalg.norm(M - (w + Lm[:, None] * (d @ R.T)), axis=1)
    # report the layout transform anchored at the solved wrist
    t = w
    return w, R, t, res, Lm


def project_sphere(p, centre, r):
    return centre + r * unit(p - centre)


def ref_lengths(lay, which="rig", path=None):
    L = {f: dict(lay["len"][f]) for f in FINGERS5}
    if which == "olex":
        for f in FINGERS5:
            L[f].update(OLEX_LENGTHS[f])
    elif which not in ("rig", None):
        user = json.load(open(which))
        user = user.get("lengths", user)
        for f in user:
            L[f].update(user[f])
    return L


def place_all(ffs, L, c, lay, n=None, evidence=True):
    J, EV = {}, {}
    for f in FINGERS5:
        ffs[f].cscale = c
    for f in FINGERS4:
        ls = [L[f]["distal"] / c, L[f]["middle"] / c, L[f]["proximal"] / c]
        js, ev = ffs[f].place(ls, TIP_INSET / c, use_evidence=evidence)
        for nm, p in zip(["tip", "dip", "pip", "mcp"], js):
            J[f"{f}_{nm}"] = p
        EV[f] = ev
    ls = [L["thumb"]["distal"] / c, L["thumb"]["proximal"] / c]
    js, ev = ffs["thumb"].place(ls, TIP_INSET / c, use_evidence=evidence)
    for nm, p in zip(["tip", "ip", "mcp"], js):
        J[f"thumb_{nm}"] = p
    EV["thumb"] = ev
    if n is None:
        along = unit(np.mean([unit(J[f + "_pip"] - J[f + "_mcp"]) for f in FINGERS4], 0))
        n = np.cross(J["index_mcp"] - J["pinky_mcp"], along)
        n = unit(n)
    else:
        S = ffs["index"].S
        for f in FINGERS4:
            m, th = mcp_depth(S, J[f + "_mcp"], n, MCP_DEPTH, (12.0 / c, 50.0 / c))
            pip = J[f + "_pip"]
            J[f + "_mcp"] = pip + (L[f]["proximal"] / c) * unit(m - pip)
            EV[f][-1]["knuckle_thickness_mm"] = None if th is None else round(th * c, 2)
    wrist, cmc, R, t, res = fit_layout(J, n, lay, c)
    wth = None
    X_capf = [None]
    if evidence:
        # wrist joint centre at mid-thickness of the wrist; refit the layout through it
        S = ffs["index"].S
        w1, wth = mcp_depth(S, wrist, n, WRIST_DEPTH, (20.0 / c, 70.0 / c))
        # lateral + depth: centroid of the wrist cross-section (plane through the
        # layout wrist, normal = layout wrist -> middle MCP direction)
        ax_w = R @ np.array([0, 1.0, 0])
        sec = S.section(wrist, ax_w, 70.0 / c)
        if sec is None or not (800 / c ** 2 < sec["area"] < 6000 / c ** 2):
            # a stray fold / handle can break the loop exactly at the wrist: use the
            # nearest good section within 6 mm, moved back into the wrist plane
            for dd in (3.0, -3.0, 6.0, -6.0):
                s2 = S.section(wrist + dd / c * ax_w, ax_w, 70.0 / c)
                if s2 is not None and 800 / c ** 2 < s2["area"] < 6000 / c ** 2:
                    s2 = dict(s2)
                    s2["centroid"] = s2["centroid"] - (dd / c) * ax_w
                    s2["pts"] = s2["pts"] - (dd / c) * ax_w
                    sec = s2
                    break
        if sec is not None and 800 / c ** 2 < sec["area"] < 6000 / c ** 2:
            w1 = sec["centroid"]
            wth = ("section", float(sec["area"]) * c * c)
            ft = getattr(S, "fill_tree", None)
            if ft is not None:
                dfill, _ = ft.query(sec["pts"])
                capf = float(np.mean(dfill < 2.0 / c))
                X_capf[0] = capf
                if capf > 0.15:
                    # much of the wrist outline is cap (no data, e.g. the back lay on the
                    # table): its centroid is biased toward the scanned side.  Keep the
                    # lateral position, take the depth from the MCP-based layout.
                    w1 = w1 - ((w1 - wrist) @ n) * n
                    wth = ("section+layout depth (wrist section %.0f%% cap)" % (100 * capf),
                           float(sec["area"]) * c * c)
        A = np.array([lay["mcp"][f] / c for f in FINGERS4] + [np.zeros(3)] * 2)
        B = np.array([J[f + "_mcp"] for f in FINGERS4] + [w1] * 2)
        R, t = kabsch(A, B)
        res = np.linalg.norm(A[:4] @ R.T + t - B[:4], axis=1)
        wrist, cmc = t.copy(), R @ (lay["cmc"] / c) + t
        if WRIST_LS:
            # keep that wrist's lateral / depth position, re-solve its position along
            # the wrist -> knuckle axis from the metacarpal lines (lengths free)
            wrist, R, t, res, Lm = wrist_free_lengths(J, n, lay, c, R, wrist)
            cmc = R @ (lay["cmc"] / c) + t
            wth = (wth if isinstance(wth, tuple) else ("mid", wth)) + (
                {f: round(float(l * c), 2) for f, l in zip(FINGERS4, Lm)},)
    J["wrist"] = wrist
    cmc_s = project_sphere(cmc, J["thumb_mcp"], L["thumb"]["metacarpal"] / c)
    J["thumb_cmc"] = CMC_SPHERE_W * cmc_s + (1 - CMC_SPHERE_W) * cmc
    return J, EV, {"layout_R": R, "layout_t": t, "layout_res": res, "cmc_layout": cmc,
                   "wrist_thickness": wth, "normal": n, "wrist_cap_frac": X_capf[0]}


def estimate_scale(J, EV, L, c_prev):
    """LS scale from the segments whose both ends carry geometric evidence."""
    m, Lr, names = [], [], []
    for f in FINGERS4:
        ev = EV[f]
        tip = J[f + "_tip"]
        d = ev[0].get("peak_pt")
        p = ev[1].get("peak_pt")
        if d is not None:
            m.append(np.linalg.norm(tip - d)); Lr.append(L[f]["distal"]); names.append(f + "_distal")
        if d is not None and p is not None:
            m.append(np.linalg.norm(d - p)); Lr.append(L[f]["middle"]); names.append(f + "_middle")
    ip = EV["thumb"][0].get("peak_pt")
    if ip is not None:
        m.append(np.linalg.norm(J["thumb_tip"] - ip)); Lr.append(L["thumb"]["distal"]); names.append("thumb_distal")
    m, Lr = np.array(m), np.array(Lr)
    if len(m) < 3:
        return c_prev, {"n": int(len(m)), "note": "too little evidence, scale kept"}
    keep = np.ones(len(m), bool)
    for _ in range(3):
        c = (Lr[keep] * m[keep]).sum() / (m[keep] ** 2).sum()
        r = Lr - c * m
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep]))) + 0.3
        keep = np.abs(r) < 3 * mad
    c = (Lr[keep] * m[keep]).sum() / (m[keep] ** 2).sum()
    r = Lr - c * m
    return float(c), {"n": int(keep.sum()), "segments": {nm: {"measured_mm": round(float(c * mm), 2),
                                                               "ref_mm": round(float(l), 2),
                                                               "resid_mm": round(float(rr), 2),
                                                               "used": bool(k)}
                                                          for nm, mm, l, rr, k in zip(names, m, Lr, r, keep)},
                      "rms_mm": round(float(np.sqrt(np.mean(r[keep] ** 2))), 3)}


# ======================================================================= pipeline
UNITS = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}


def analyse(S, tipv, wts, L, lay, c0, scale_known, info):
    """Centre-lines, joints, scale on surface S (units: c0 * unit = mm, approx)."""
    ffs = {}
    for f in FINGERS5:
        cl = finger_centreline(S, int(tipv[f]), wts[f])
        ffs[f] = FingerFit(S, f, cl, None)
    c = c0
    J, EV, X = place_all(ffs, L, c, lay, evidence=False)
    along = unit(np.mean([unit(J[f + "_pip"] - J[f + "_mcp"]) for f in FINGERS4], 0))
    n = palm_normal_est(S, J, along, r=30.0 / c)
    for f in FINGERS5:
        ffs[f].pal = n
        ffs[f].profile(1.3 * wts[f])
    hist = []
    sc_info = {"note": "units given, scale fixed"}
    for it in range(4):
        J, EV, X = place_all(ffs, L, c, lay, n=n, evidence=True)
        if scale_known:
            _, sc_info = estimate_scale(J, EV, L, c)     # residuals only
            sc_info["note"] = "units given, scale fixed"
            break
        c_new, sc_info = estimate_scale(J, EV, L, c)
        hist.append(round(c_new, 5))
        if abs(c_new - c) < 1e-4 * c:
            c = c_new
            J, EV, X = place_all(ffs, L, c, lay, n=n, evidence=True)
            break
        c = c_new
    else:
        J, EV, X = place_all(ffs, L, c, lay, n=n, evidence=True)
    sc_info["iterations"] = hist
    info["scale_fit"] = sc_info
    info["evidence"] = {f: [{k: v for k, v in r.items() if k != "peak_pt"} for r in EV[f]] for f in FINGERS5}
    info["layout_resid_mm"] = [round(float(x) * c, 2) for x in X["layout_res"]]
    wt = X["wrist_thickness"]
    info["wrist_section_cap_frac"] = None if X.get("wrist_cap_frac") is None else round(X["wrist_cap_frac"], 3)
    if isinstance(wt, tuple) and str(wt[0]).startswith("section"):
        info["wrist_from"] = (f"wrist section centroid, area {wt[1]:.0f} mm2" if wt[0] == "section"
                              else f"{wt[0]}, area {wt[1]:.0f} mm2")
        if wt[0] != "section":
            info.setdefault("warnings", []).append(
                "the wrist cross-section is largely cap (no scan data): wrist depth taken from the "
                "knuckle-row layout instead; wrist / thumb_cmc are less certain.")
    else:
        info["wrist_from"] = f"mid-thickness {wt[1] if isinstance(wt, tuple) else wt}"
    if isinstance(wt, tuple) and len(wt) == 3:
        info["wrist_from"] += "; axial position from the 4 metacarpal lines (free lengths)"
        info["metacarpal_lengths_mm"] = wt[2]
    return J, c, X["normal"], ffs, X


def forearm_axis(S, J, n, c):
    """Forearm direction (pointing proximally) from section centroids proximal of the wrist."""
    along = unit(J["middle_mcp"] - J["wrist"])
    along = unit(along - (along @ n) * n)
    pts = []
    for d in np.arange(10.0, 46.0, 4.0) / c:
        sec = S.section(J["wrist"] - d * along, along, 70.0 / c)
        if sec is None:
            break
        pts.append(sec["centroid"])
    if len(pts) >= 4:
        _, ax = fit_line(np.array(pts))
        if ax @ along > 0:
            ax = -ax
        if ax @ (-along) > np.cos(np.radians(40)):
            return ax, len(pts)
    return -along, len(pts)


def table_cuts(V, planes, margin_frac=1.5):
    """One planar cut per removed table surface, in V's units: the plane that best
    fits the (quadric) table height under the hand's footprint, moved toward the
    hand by the largest height residual + margin_frac * the inlier band, so the
    fillet where hand and table were fused goes and the cap is a planar loop.
    Returns [(point, normal pointing to the hand)]."""
    cuts = []
    for pl in planes:
        o, n, coef, thr, ks = pl["o"], pl["n"], pl["coef"], pl["thr"], pl["keep_sign"]
        X = _quad_design(V, o, n)
        h_tab = X @ coef
        hgt = ks * ((V - o) @ n - h_tab)          # height of the hand above the table
        foot = hgt < 6 * thr + 0.02 * np.linalg.norm(np.ptp(V, 0))
        if foot.sum() < 10:
            continue
        # plane through the table heights under the footprint: h ~ X[:, :3] b
        b = np.linalg.lstsq(X[foot][:, :3], h_tab[foot], rcond=None)[0]
        u, v = _plane_frame(n)
        # height field h = b0 + b1 a + b2 c  ->  plane normal n - b1 u - b2 v
        nn = unit(n - b[1] * u - b[2] * v)
        top = ks * (h_tab[foot] - X[foot][:, :3] @ b)
        off = float(top.max()) + margin_frac * thr
        p0 = o + (b[0] + ks * off) * n
        cuts.append((p0, ks * nn))
    return cuts


def fit_scan(path, units="auto", lengths="rig", seed=0):
    info = {"input": os.path.abspath(path)}
    rng = np.random.default_rng(seed)
    V, F = load_mesh(path)
    info["input_mesh"] = {"vertices": int(len(V)), "faces": int(len(F))}
    log("loaded", path, V.shape)
    known = units != "auto"
    if known:
        V = V * UNITS[units]
    V, F, planes = remove_table(V, F, rng, info)
    V, F = keep_largest(V, F, info)
    cuts = table_cuts(V, planes)
    D = float(np.linalg.norm(np.ptp(V, 0)))
    # ---- coarse, scale-free pass: fingertips and the unit guess
    Vc, Fc = bpy_rebuild(V, F, D / 300.0, 30000, cuts=cuts)
    Sc = Surf(Vc, Fc)
    fing = find_fingertips(Sc, info)
    wl = np.median([fing[f]["w"] for f in FINGERS4])
    s0 = 1.0 if known else W_REF / wl
    info["s0"] = s0
    log(f"fingertips found; unit guess s0 = {s0:.6g} (ring width {wl:.4g})")
    lay = palm_layout(read_rig())
    L = ref_lengths(lay, lengths)
    # coarse joints for the forearm cut
    touching = info.get("tip_test", "strict") != "strict"
    Sc = Surf(Vc * s0, Fc)
    Sc.split_touching = touching
    tipv = {f: fing[f]["v"] for f in FINGERS5}
    wts = {f: fing[f]["w"] * s0 for f in FINGERS5}
    Jc, cc, nc, _, _ = analyse(Sc, tipv, wts, L, lay, 1.0, known, {})
    ax, _ = forearm_axis(Sc, Jc, nc, cc)
    cut = (Jc["wrist"] + ax * (FOREARM_KEEP + 15.0) / cc, -ax)   # keep side = -ax (distal)
    # ---- fine pass (s0 units ~ mm)
    cuts_s0 = [(p * s0, nn) for p, nn in cuts]
    ast = {}
    Vf, Ff = bpy_rebuild(V * s0, F, FINAL_VOXEL / cc, 80000, plane=(cut[0], cut[1]), smooth_it=ANALYSIS_SMOOTH,
                         cuts=cuts_s0, stats=ast, genus_retry=True)
    fp = ast.pop("_fill_pts")
    info["analysis_rebuild"] = ast
    S = Surf(Vf, Ff)
    S.split_touching = touching
    S.fill_tree = cKDTree(fp) if len(fp) else None
    tipv = {f: int(S.vtree.query(Sc.V[fing[f]["v"]])[1]) for f in FINGERS5}
    J, c, n, ffs, X = analyse(S, tipv, wts, L, lay, cc, known, info)
    # consistency: each tip landmark must sit just inside its own fingertip
    for f in FINGERS5:
        dt = float(np.linalg.norm(J[f + "_tip"] - S.V[tipv[f]]) * c)
        if dt > 8.0:
            info.setdefault("warnings", []).append(
                f"{f}_tip is {dt:.1f} mm from the {f} fingertip found on the skin: the {f} chain is "
                "probably on the wrong finger. " + SPREAD_ADVICE)
    ax, nsec = forearm_axis(S, J, n, c)
    info["forearm_axis_sections"] = nsec
    k = s0 * c                    # raw (after units) -> mm
    info["scale_raw_to_mm"] = k * (UNITS[units] if known else 1.0)
    log(f"scale: raw unit -> mm = {info['scale_raw_to_mm']:.6g}")
    Jmm = {nm: J[nm] * c for nm in NAMES}
    cut = (Jmm["wrist"] + ax * FOREARM_KEEP, -ax)
    # final skin.  The voxel remesh welds crossing sheets into handles; where the forearm cap,
    # the table cap and a ragged rim meet near the wrist this happens for some cut positions
    # (seen intermittently on a noisy-table case).  Fall back to a forearm cut further from the
    # wrist, then without the table skirt, and keep the first genus-0 result.
    tries = [("as planned", cut, True), ("forearm cut 8 mm further", (cut[0] + 8.0 * ax, cut[1]), True),
             ("forearm cut 15 mm further", (cut[0] + 15.0 * ax, cut[1]), True),
             ("no table skirt", cut, False)]
    best = None
    for ti_, (lab_, pl_, sk_) in enumerate(tries):
        st_ = {}
        Vo_, Fo_ = bpy_rebuild(V * k, F, FINAL_VOXEL, TARGET_VERTS, plane=pl_, cuts=[(p * k, nn) for p, nn in cuts],
                               stats=st_, genus_retry=True, use_skirt=sk_)
        st_.pop("_fill_pts", None)
        st_["attempt"] = lab_
        if best is None or abs(st_["remesh_euler"] - 2) < abs(best[2]["remesh_euler"] - 2):
            best = (Vo_, Fo_, st_)
        if st_["remesh_euler"] == 2:
            break
        log(f"final remesh Euler {st_['remesh_euler']} ({lab_}); trying the next cut")
    Vo, Fo, rst = best
    rst["attempts"] = ti_ + 1
    info["rebuild"] = rst
    if rst.get("remesh_euler", 2) != 2:
        g = (2 - rst["remesh_euler"]) // 2
        info.setdefault("warnings", []).append(
            f"the closed skin has {g} handle(s) (Euler {rst['remesh_euler']}, a hand is 2): touching "
            "fingers or scan debris were welded together. Look at check.png; " + SPREAD_ADVICE)
    if rst.get("filled_area_frac", 0) > 0.05:
        info.setdefault("warnings", []).append(
            f"{100 * rst['filled_area_frac']:.0f}% of the skin had no scan data and was closed with flat "
            "caps (table side, holes, forearm cut): the shape there is a guess (cap area includes the "
            "forearm cut).")
    Vo, Fo = orient_outward(Vo, Fo)
    info["mesh"] = mesh_check(Vo, Fo)
    info["forearm_cut"] = {"point": [round(float(x), 3) for x in cut[0]],
                           "normal": [round(float(x), 4) for x in cut[1]]}
    return {"J": Jmm, "normal": n, "V": Vo, "F": Fo, "info": info, "c": c, "k": k,
            "centrelines": {f: ffs[f].P * c for f in FINGERS5}, "L": L, "ffs": ffs}


# ======================================================================= step 4: armature
def bone_list():
    """(name, head joint, tail joint, parent, connected) in SPEC hierarchy."""
    out = [("palm", "wrist", "knuckles", None, False)]
    for f in FINGERS4:
        out.append((f"{f}_metacarpal", "wrist", f"{f}_mcp", "palm", False))
        prev = f"{f}_metacarpal"
        for s, a, b in zip(SEGS4, ["mcp", "pip", "dip"], ["pip", "dip", "tip"]):
            out.append((f"{f}_{s}", f"{f}_{a}", f"{f}_{b}", prev, True))
            prev = f"{f}_{s}"
    prev = "palm"
    for s, a, b in zip(SEGS_T, JNT[:-1], JNT[1:]):
        out.append((f"thumb_{s}", f"thumb_{a}", f"thumb_{b}", prev, s != "metacarpal"))
        prev = f"thumb_{s}"
    return out


def palmar_dirs(J, n):
    """Per-bone palmar direction (bone Z after align_roll). Fingers: palm normal.
    Thumb: its own bending direction if the chain is bent, else the default
    (palm normal + toward the little-finger side, as in the stand-in)."""
    across = unit(J["index_mcp"] - J["pinky_mcp"])
    pt = unit(n - across)
    a, b, c = J["thumb_mcp"], J["thumb_ip"], J["thumb_tip"]
    d1 = unit(b - a)
    bend = (c - b) - ((c - b) @ d1) * d1
    if np.linalg.norm(bend) > np.sin(np.radians(8)) * np.linalg.norm(c - b) and unit(bend) @ pt > 0:
        pt = unit(bend)
    out = {}
    for name, h, t, _, _ in bone_list():
        out[name] = pt if name.startswith("thumb") else n
    return out


def skin_weights(V, J, bones, sigma=3.0, keep=3):
    """Initial weights: soft nearest-bone-segment assignment (distance falloff)."""
    names, D = [], []
    for name, h, t, _, _ in bones:
        if name == "palm":
            continue
        a, b = J[h], J[t]
        ab = b - a
        tt = np.clip(((V - a) @ ab) / (ab @ ab), 0, 1)
        D.append(np.linalg.norm(V - (a + tt[:, None] * ab), axis=1))
        names.append(name)
    D = np.stack(D, 1)
    dmin = D.min(1, keepdims=True)
    W = np.exp(-((D - dmin) / sigma) ** 2)
    order = np.argsort(-W, axis=1)
    mask = np.zeros_like(W, bool)
    np.put_along_axis(mask, order[:, :keep], True, axis=1)
    W = np.where(mask & (W > 0.02), W, 0)
    W /= W.sum(1, keepdims=True)
    return names, W


def build_blend(V, F, J, n, out_blend, info):
    import bpy
    from mathutils import Vector
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 0.001
    sc.unit_settings.length_unit = "MILLIMETERS"
    me = bpy.data.meshes.new("hand scan")
    me.from_pydata(V.tolist(), [], F.tolist())
    me.update()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new("hand scan", me)
    sc.collection.objects.link(ob)
    mat = bpy.data.materials.new("skin")
    mat.diffuse_color = (0.95, 0.68, 0.55, 1)
    me.materials.append(mat)

    Jx = dict(J)
    Jx["knuckles"] = np.mean([J[f + "_mcp"] for f in FINGERS4], 0)
    bones = bone_list()
    pal = palmar_dirs(J, n)
    ad = bpy.data.armatures.new("hand rig")
    arm = bpy.data.objects.new("hand rig", ad)
    sc.collection.objects.link(arm)
    arm.show_in_front = True
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = ad.edit_bones
    for name, h, t, par, con in bones:
        b = eb.new(name)
        b.head, b.tail = Vector(Jx[h]), Vector(Jx[t])
        b.align_roll(Vector(pal[name]))
    for name, h, t, par, con in bones:
        if par:
            eb[name].parent = eb[par]
            eb[name].use_connect = con
    bpy.ops.object.mode_set(mode="OBJECT")
    for b in ad.bones:
        b.use_deform = b.name != "palm"
    for pb in arm.pose.bones:
        pb.rotation_mode = "XYZ"
        if pb.name != "palm":
            pb.lock_location = (True, True, True)
        pb.lock_scale = (True, True, True)
    # skin binding
    names, W = skin_weights(V, J | {"knuckles": Jx["knuckles"]}, bones)
    for k, nm in enumerate(names):
        vg = ob.vertex_groups.new(name=nm)
        nz = np.flatnonzero(W[:, k] > 0)
        for w in np.unique(np.round(W[nz, k], 3)):
            ids = nz[np.round(W[nz, k], 3) == w]
            vg.add(ids.tolist(), float(w), "REPLACE")
    ob.parent = arm
    mod = ob.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    mod.use_vertex_groups = True
    arm["grip_scan"] = json.dumps({"names": NAMES, "fingers": {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8],
                                   "middle": [9, 10, 11, 12], "ring": [13, 14, 15, 16],
                                   "pinky": [17, 18, 19, 20]},
                                   "convention": "bone local X = flexion axis; +rotation about X curls toward the palm"})
    arm["fit_rig"] = json.dumps({k: info[k] for k in ("input", "scale_raw_to_mm") if k in info})
    info["rig_check"] = verify_rig(arm, pal)
    os.makedirs(os.path.dirname(out_blend), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    return out_blend


def verify_rig(arm, pal):
    """Pose every finger bone +20 deg about local X and check the tail moves toward
    the palmar side; also check heads/tails sit on the joints (rest)."""
    import bpy
    res = {}
    ok_all = True
    for pb in arm.pose.bones:
        if pb.name == "palm":
            continue
        rest_tail = np.array(arm.matrix_world @ pb.bone.tail_local)
        pb.rotation_euler = (np.radians(20), 0, 0)
        bpy.context.view_layer.update()
        posed_tail = np.array(arm.matrix_world @ pb.tail)
        pb.rotation_euler = (0, 0, 0)
        bpy.context.view_layer.update()
        mv = posed_tail - rest_tail
        dot = float(unit(mv) @ pal[pb.name]) if np.linalg.norm(mv) > 1e-9 else 0.0
        res[pb.name] = round(dot, 3)
        ok_all &= dot > 0.5
    return {"positive_x_curls_palmward": bool(ok_all), "tail_move_dot_palmar_min": min(res.values())}


# ======================================================================= check image
FCOL = {"thumb": (40, 40, 220), "index": (30, 160, 30), "middle": (200, 120, 20),
        "ring": (160, 40, 160), "pinky": (20, 160, 200), "palm": (60, 60, 60)}   # BGR


def render_check(V, F, J, n, out_png, title, samples=16):
    """Cycles CPU orthographic renders (palm, side from the thumb side, back) of the
    cleaned skin, with the fitted joints and bones drawn on top (x-ray overlay)."""
    import bpy
    import cv2
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Matrix, Vector
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    me = bpy.data.meshes.new("skin")
    me.from_pydata(V.tolist(), [], F.tolist())
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new("skin", me)
    sc.collection.objects.link(ob)
    mat = bpy.data.materials.new("skin")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.80, 0.60, 0.50, 1)
    bsdf.inputs["Roughness"].default_value = 0.6
    me.materials.append(mat)
    world = bpy.data.worlds.new("w")
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.9, 0.9, 0.92, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.5
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = False
    sc.view_settings.view_transform = "Standard"
    cam_d = bpy.data.cameras.new("cam")
    cam_d.type = "ORTHO"
    cam_d.clip_end = 2000
    cam = bpy.data.objects.new("cam", cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam
    sun_d = bpy.data.lights.new("sun", "SUN")
    sun_d.energy = 3.0
    sun = bpy.data.objects.new("sun", sun_d)
    sc.collection.objects.link(sun)
    up = unit(J["middle_mcp"] - J["wrist"])
    up = unit(up - (up @ n) * n)
    across = np.cross(up, n)               # right hand: points to the thumb side (x)
    if across @ (J["index_mcp"] - J["pinky_mcp"]) < 0:
        across = -across
    c = sum(ax * 0.5 * ((V @ ax).min() + (V @ ax).max()) for ax in (across, up, n))
    H, Wd = 860, 560
    sc.render.resolution_x, sc.render.resolution_y = Wd, H
    views = [("palm view", n, up), ("thumb-side view", across, up), ("back view", -n, up)]
    ext = max(np.ptp(V @ up), np.ptp(V @ across) * H / Wd, np.ptp(V @ n) * H / Wd) * 1.08
    cam_d.ortho_scale = ext
    chains = [("palm", ["wrist", "thumb_cmc"])] + \
        [(f, ["wrist", f"{f}_mcp", f"{f}_pip", f"{f}_dip", f"{f}_tip"]) for f in FINGERS4] + \
        [("thumb", ["thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"]),
         ("palm", ["index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"])]
    imgs = []
    for vt, vd, vu in views:
        vd, vu = Vector(unit(vd)), Vector(vu)
        cx = vu.cross(vd)
        cam.matrix_world = Matrix.Translation(Vector(c) + vd * 600) @ Matrix(
            [[cx[i], vu[i], vd[i], 0] for i in range(3)] + [[0, 0, 0, 1]])
        ld = (vd + vu * 0.6 - cx * 0.5).normalized()
        sx = vu.cross(ld).normalized()
        sy = ld.cross(sx)
        sun.matrix_world = Matrix([[sx[i], sy[i], ld[i], 0] for i in range(3)] + [[0, 0, 0, 1]])
        sc.render.filepath = os.path.join(WORK, f"check_{os.getpid()}.png")
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True)
        im = cv2.imread(sc.render.filepath)

        def px(p):
            q = world_to_camera_view(sc, cam, Vector(p))
            return int(round(q.x * Wd)), int(round((1 - q.y) * H))
        ov = im.copy()
        for f, ch in chains:
            for a, b in zip(ch[:-1], ch[1:]):
                cv2.line(ov, px(J[a]), px(J[b]), FCOL[f], 3, cv2.LINE_AA)
        for nm in NAMES:
            f = nm.split("_")[0]
            cv2.circle(ov, px(J[nm]), 5, FCOL.get(f, FCOL["palm"]), -1, cv2.LINE_AA)
            cv2.circle(ov, px(J[nm]), 5, (255, 255, 255), 1, cv2.LINE_AA)
        im = cv2.addWeighted(ov, 0.85, im, 0.15, 0)
        im = cv2.copyMakeBorder(im, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(im, vt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
        imgs.append(im)
    out = np.concatenate([np.pad(i, ((0, 0), (4, 4), (0, 0)), constant_values=255) for i in imgs], 1)
    bar = np.full((40, out.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.imwrite(out_png, np.concatenate([bar, out], 0))
    return out_png


# ======================================================================= output
def joints_json(J, n, info):
    def ln(chain):
        return [round(float(np.linalg.norm(J[b] - J[a])), 2) for a, b in zip(chain[:-1], chain[1:])]
    lengths = {"thumb": dict(zip(SEGS_T, ln([f"thumb_{x}" for x in JNT])))}
    for f in FINGERS4:
        lengths[f] = dict(zip(["metacarpal"] + SEGS4, ln(["wrist"] + [f"{f}_{x}" for x in JN4])))
    out = {"side": "right", "units": "mm", "pose": "open",
           "landmarks": {nm: [round(float(x), 3) for x in J[nm]] for nm in NAMES},
           "palm_normal": [round(float(x), 5) for x in unit(n)],
           "lengths": lengths}
    try:
        sys.path.insert(0, "/tmp/claude-0/hs/pg")
        from gripscan.relaxed_pose import angles
        a = angles(np.array([J[nm] for nm in NAMES]))
        pip = np.median([v for k, v in a.items() if k.endswith("_pip")])
        if pip < 0:      # relaxed_pose's axis sign depends on the hand frame; make curl positive
            a = {k: (v if (k.startswith("spread") or k == "thumb_cmc") else -v) for k, v in a.items()}
        out["angles_deg"] = {k: round(float(v), 2) for k, v in a.items()}
    except Exception as e:  # noqa
        out["angles_deg_error"] = str(e)
    out["fit"] = info
    return out


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def run(path, out_dir, units="auto", lengths="rig", render=True, name=None):
    import trimesh
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    r = fit_scan(path, units, lengths)
    J, n, info = r["J"], r["normal"], r["info"]
    info["lengths_ref"] = lengths
    trimesh.Trimesh(r["V"], r["F"], process=False).export(os.path.join(out_dir, "scan_clean.ply"))
    blend = build_blend(r["V"], r["F"], J, n, os.path.join(out_dir, "hand_rigged.blend"), info)
    info["t_total_s"] = round(time.time() - T0, 1)
    jj = to_jsonable(joints_json(J, n, info))
    json.dump(jj, open(os.path.join(out_dir, "joints.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(out_dir, "centrelines.npz"), **r["centrelines"])
    if render:
        render_check(r["V"], r["F"], J, n, os.path.join(out_dir, "check.png"),
                     f"fit_rig: {name or os.path.basename(path)}  (scale {info['scale_raw_to_mm']:.5g} mm/unit,"
                     f" seg rms {info['scale_fit'].get('rms_mm', float('nan'))} mm)")
    log("wrote", out_dir)
    for w in info.get("warnings", []):
        log("WARNING: " + w)
    return r, jj


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scan")
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--units", default="auto", choices=["auto"] + list(UNITS))
    ap.add_argument("--lengths", default="rig", help="rig | olex | path to a json with {finger:{segment:mm}}")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    name = a.name or os.path.splitext(os.path.basename(a.scan))[0]
    out = a.out or os.path.join(OUT_ROOT, name)
    run(a.scan, out, a.units, a.lengths, not a.no_render, name)


if __name__ == "__main__":
    main()
