"""rig_core.py: numpy-only pieces of the GripV grip rig (no bpy).

The grip rig deforms the basemesh GripV *cage* (half mesh, X >= 0) with an armature.
Everything downstream (MirrorX, subdivision surface, every case part that reads
'basemesh variant') follows the cage.

What lives here:
  * the exception zone (Olex's STL, +X half, basemesh frame) and an inside test
  * landmarks: where each bone starts and ends, found from the grip surface and the
    scanned hand in the approved hold (hold_upright)
  * pinned cage vertices (exception zone, cage vertices whose subdivision reaches into
    the zone, battery guard) and per-axis masks (mirror seam, flat top)
  * smooth, low-frequency skin weights
  * a linear-blend-skinning simulator and smoothness / invariance metrics, used to
    check poses without Blender (the subdivision is linear: out = A @ cage)
"""
import numpy as np

ZONE_STL = "/mnt/project-files/hand_scan_rig/grip_rig/exception_zone.stl"
TOP_Z = 59.04          # flat top of the case (must stay exact)

# bone names as Olex uses them
ROOT = "grip root"
PALM = "palm swell"
FINS = ("fin middle", "fin ring", "fin pinky")
STAND = "stand"
THENAR = "thenar surface"
TOP_EDGE = "top back edge"
DEFORM = (PALM,) + FINS + (STAND, THENAR, TOP_EDGE)

PARAMS = {
    "sigma_mm": {PALM: 20.0, "fin middle": 13.0, "fin ring": 14.0, "fin pinky": 14.0,
                 STAND: 15.0, THENAR: 16.0},
    "pin_ramp_mm": 26.0,      # weights rise from 0 at a pinned cage vertex to full over this distance
    "trigger_keep_mm": 4.0,   # cage vertices this close to the R2/R4 trigger parts are pinned ...
    "trigger_ramp_mm": 16.0,  # ... and weights reach full strength this far from them
    "battery_margin_mm": 6.0,  # battery box (battery bool) grown by this much is pinned
    "share_q": 3.0,           # bones share a vertex in proportion to (gaussian)^q
    "saturate": 3.0,          # rigged amount = 1 - exp(-saturate * sum of gaussians)
    "palm_tail_mm": 15.0,     # palm swell bone points this far out along the surface normal
}


# ---------------------------------------------------------------- small helpers
def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def read_stl(path=ZONE_STL):
    """ASCII or binary STL -> (V, F) with merged vertices."""
    raw = open(path, "rb").read()
    tris = None
    if raw[:5].lower() == b"solid" and b"facet" in raw[:400]:
        pts = [list(map(float, ln.split()[1:4])) for ln in raw.decode("ascii", "ignore").splitlines()
               if ln.strip().startswith("vertex")]
        tris = np.array(pts).reshape(-1, 3, 3)
    else:
        n = int(np.frombuffer(raw[80:84], "<u4")[0])
        rec = np.frombuffer(raw[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        tris = rec["v"].astype(float)
    P = tris.reshape(-1, 3)
    key = np.round(P, 4)
    V, inv = np.unique(key, axis=0, return_inverse=True)
    return V, inv.reshape(-1, 3)


def zone_mesh(path=ZONE_STL):
    """The zone STL is the +X half (open at X=0); mirrored copy closes it."""
    V, F = read_stl(path)
    Vm = V * [-1.0, 1.0, 1.0]
    return np.vstack([V, Vm]), np.vstack([F, F[:, ::-1] + len(V)]), (V, F)


def winding(V, F, P, chunk=3000):
    """Generalised winding number of the closed triangle mesh (V, F) at points P (|w| ~ 1 inside)."""
    P = np.asarray(P, float)
    out = np.zeros(len(P))
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk, None, :]
        a, b, c = A[None] - p, B[None] - p, C[None] - p
        la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
        num = np.einsum("ijk,ijk->ij", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("ijk,ijk->ij", a, b) * lc + np.einsum("ijk,ijk->ij", b, c) * la
               + np.einsum("ijk,ijk->ij", c, a) * lb)
        out[s:s + chunk] = 2.0 * np.arctan2(num, den).sum(1) / (4.0 * np.pi)
    return out


def in_zone(P, zone=None):
    ZV, ZF, _ = zone if zone is not None else zone_mesh()
    return np.abs(winding(ZV, ZF, P)) > 0.5


def ray_hit(V, T, o, d):
    """First hit (t > 0) of the ray o + t d with triangles (V, T); None if it misses."""
    o, d = np.asarray(o, float), np.asarray(d, float)
    A = V[T[:, 0]]
    E1, E2 = V[T[:, 1]] - A, V[T[:, 2]] - A
    p = np.cross(d, E2)
    det = (E1 * p).sum(1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = o - A
    u = (s * p).sum(1) * inv
    q = np.cross(s, E1)
    v = (q @ d) * inv
    t = (E2 * q).sum(1) * inv
    m = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
    return None if not m.any() else o + d * t[m].min()


def vertex_normals(V, T):
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, T[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def orient_outward(V, T):
    """Flip T if its signed volume is negative."""
    vol = np.einsum("ij,ij->i", V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])).sum() / 6.0
    return T if vol > 0 else T[:, ::-1]


# ---------------------------------------------------------------- landmarks
def landmarks(S, T, hand_bones, palm_swell, out_free=None, thenar_min_free=0.7, middle_min_free=0.5):
    """Bone end points on the (subdivided, full) grip surface S, T.

    hand_bones: {bone: [head, tail]} of the scanned hand in the approved hold (world mm).
    palm_swell: the 'palm swell' marker of the hold (on the grip surface).
    Returns {name: {'p': point, 'n': outward normal, 'how': text}}.
    """
    T = orient_outward(S, T)
    N = vertex_normals(S, T)
    R = S[:, 0] > 0.0
    B = {k: np.asarray(v, float) for k, v in hand_bones.items()}

    def snap(p):
        i = np.argmin(np.where(R, np.linalg.norm(S - p, axis=1), 1e9))
        return S[i], N[i]

    def extreme(direction, mask):
        s = S @ np.asarray(direction, float)
        s[~(R & mask)] = -1e9
        i = int(np.argmax(s))
        return S[i], N[i]

    out = {}
    p, n = snap(np.asarray(palm_swell, float))
    out[PALM] = {"p": p, "n": n, "how": "the 'palm swell' marker of the approved hold (hold_upright), on the grip surface"}
    # middle finger: GripV has no lobe there; the tip is where the middle finger's middle-joint crease
    # (the inside of its curl) faces the handle: ray from the PIP joint along the inner bisector of the bend
    u1 = B["middle_proximal"][1] - B["middle_proximal"][0]
    u2 = B["middle_middle"][1] - B["middle_middle"][0]
    bis = -u1 / np.linalg.norm(u1) + u2 / np.linalg.norm(u2)
    hit = ray_hit(S, T, B["middle_proximal"][1], bis / np.linalg.norm(bis))
    okm = R if out_free is None else R & (out_free >= middle_min_free)
    i = int(np.argmin(np.where(okm, np.linalg.norm(S - hit, axis=1), 1e9)))
    out["fin middle"] = {"p": S[i], "n": N[i], "crease_hit": hit,
                         "how": "where the middle finger's middle-joint crease faces the handle (GripV has no lobe there), "
                                f"moved to the nearest surface that is at least {middle_min_free:.0%} free to move "
                                f"({np.linalg.norm(S[i] - hit):.1f} mm away; the crease point itself sits just under the exception zone)"}
    # ring finger: its crease ray lands on the front lobe; the tip is the lobe's point (down and forward)
    p, n = extreme((0.0, 0.6, -0.8), S[:, 1] > -42.0)
    out["fin ring"] = {"p": p, "n": n, "how": "tip of the front lobe; the ring finger curls round it"}
    # pinky: curls round the lowest point of the handle
    p, n = extreme((0.0, 0.0, -1.0), np.ones(len(S), bool))
    out["fin pinky"] = {"p": p, "n": n, "how": "tip of the bottom lobe = lowest point of the case; the pinky curls round it"}
    # stand: the end of the handle at the back, the lowest point of the back lobe (under the palm swell)
    p, n = extreme((0.0, 0.0, -1.0), S[:, 1] < -62.0)
    out[STAND] = {"p": p, "n": n, "how": "lowest point of the back lobe (handle end under the palm swell)"}
    # thenar: grip surface closest to the thumb metacarpal (thumb-base pad rests here)
    a, b = B["thumb_metacarpal"]
    t = np.clip(((S - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
    d = np.linalg.norm(S - (a + t[:, None] * (b - a)), axis=1)
    ok = R if out_free is None else R & (out_free >= thenar_min_free)
    i = int(np.argmin(np.where(ok, d, 1e9)))
    out[THENAR] = {"p": S[i], "n": N[i], "how": "grip surface closest to the thumb metacarpal (thumb-base pad) where the "
                                                 f"surface is at least {thenar_min_free:.0%} free to move (the closest point overall "
                                                 "touches the exception zone)"}
    return out


# ---------------------------------------------------------------- pins and weights
def zone_pins(co, A, S, zone=None):
    """Cage vertices that must not move so the subdivided surface inside the zone stays exact:
    those inside the zone plus every cage vertex whose subdivision weight reaches an output vertex
    inside the zone. A: (n_out, n_cage) subdivision matrix, S: rest outputs."""
    zone = zone if zone is not None else zone_mesh()
    cz = in_zone(co, zone)
    oz = in_zone(S, zone)
    infl = (np.abs(A[oz]) > 1e-4).any(0)
    return cz, infl & ~cz, oz


def battery_pins(co, box_min, box_max, margin):
    lo, hi = np.asarray(box_min) - margin, np.asarray(box_max) + margin
    c = np.abs(co) * [1, 0, 0] + co * [0, 1, 1]          # |x|: battery box is symmetric
    return np.all((c >= lo) & (c <= hi), axis=1)


def top_face(co, eps=0.005):
    return np.abs(co[:, 2] - TOP_Z) < eps


def nearest_dist(P, Q):
    from scipy.spatial import cKDTree
    return cKDTree(Q).query(P)[0] if len(Q) else np.full(len(P), 1e9)


def compute_weights(co, marks, pinned, trig_pts, top_edge, params=PARAMS):
    """Skin weights of the cage. Gaussian of the distance to each bone's region centre (palm swell:
    its head; the others: their tip), times a freedom field that is 0 on every pinned vertex and
    rises smoothly away from pinned vertices and from the triggers, softly normalised.
    The top back edge vertices belong to 'top back edge' alone."""
    sig = params["sigma_mm"]
    d_pin = nearest_dist(co, co[pinned])
    free = smoothstep(0.0, params["pin_ramp_mm"], d_pin)
    d_tr = nearest_dist(co, trig_pts)
    free *= smoothstep(params["trigger_keep_mm"], params["trigger_ramp_mm"], d_tr)
    free[pinned] = 0.0
    raw = {}
    for name, s in sig.items():
        c = marks[name]["p"]
        d = np.linalg.norm(co - c, axis=1)
        r = np.exp(-0.5 * (d / s) ** 2)
        r[d > 3.5 * s] = 0.0
        raw[name] = r
    # how much of a vertex is rigged at all (smoothly saturating), then shared between the bones
    # in proportion to r^q: each bone owns the middle of its region, neighbours blend smoothly
    tot = sum(raw.values())
    q = params["share_q"]
    rq = {n: r ** q for n, r in raw.items()}
    sq = np.maximum(sum(rq.values()), 1e-30)
    amount = free * (1.0 - np.exp(-params["saturate"] * tot))
    W = {n: amount * rq[n] / sq for n in raw}
    W[TOP_EDGE] = top_edge.astype(float)
    for n in sig:
        W[n][top_edge] = 0.0
    W[ROOT] = np.clip(1.0 - sum(W[n] for n in DEFORM), 0.0, 1.0)
    return W, free


def axis_mask(co, pinned, top_edge, seam_eps=1e-4):
    """Per-vertex, per-axis freedom applied after the armature: pinned (0,0,0); mirror seam X locked;
    flat top: Z locked everywhere, top back edge only Y."""
    m = np.ones((len(co), 3))
    m[np.abs(co[:, 0]) < seam_eps, 0] = 0.0
    m[top_face(co), 2] = 0.0
    m[top_edge] = (0.0, 1.0, 0.0)
    m[pinned] = 0.0
    return m


# ---------------------------------------------------------------- simulation and metrics
def lbs(co, W, G):
    """Linear blend skinning: G {bone: 4x4 world pose matrix relative to rest}."""
    out = np.zeros_like(co)
    tot = np.zeros(len(co))
    for n, w in W.items():
        M = G.get(n, np.eye(4))
        out += w[:, None] * (co @ M[:3, :3].T + M[:3, 3])
        tot += w
    out = np.where(tot[:, None] > 0, out / np.maximum(tot, 1e-12)[:, None], co)
    return out


def subdivide(A, base, cage):
    """Outputs of MirrorX + subdivision for a moved cage (linear). base: rest outputs (to know the side)."""
    S = A @ cage
    left = base[:, 0] < -1e-6
    S[left, 0] = -S[left, 0]
    S[np.abs(base[:, 0]) <= 1e-6, 0] = 0.0      # mirror seam (symmetric, stays on X = 0)
    return S


def edge_dihedrals(V, T):
    """Dihedral angle (deg) across each interior edge of the triangle mesh."""
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    e = np.sort(np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], 1).reshape(-1, 2), 1)
    f = np.repeat(np.arange(len(T)), 3)
    key = e[:, 0] * (len(V) + 1) + e[:, 1]
    order = np.argsort(key)
    k, ff = key[order], f[order]
    same = k[1:] == k[:-1]
    a, b = ff[:-1][same], ff[1:][same]
    cosv = np.clip((fn[a] * fn[b]).sum(1), -1, 1)
    return np.degrees(np.arccos(cosv)), e[order][:-1][same]


def smooth_report(S0, S1, T, region=None):
    """How the pose changed the surface: displacement, change of the angle between neighbouring faces
    (kinks), and the roughness of the displacement field (its umbrella Laplacian)."""
    d0, edges = edge_dihedrals(S0, T)
    d1, _ = edge_dihedrals(S1, T)
    dd = np.abs(d1 - d0)
    disp = S1 - S0
    nb = [[] for _ in range(len(S0))]
    for a, b in edges:
        nb[a].append(b)
        nb[b].append(a)
    lap = np.array([np.linalg.norm(disp[i] - disp[nb[i]].mean(0)) if nb[i] else 0.0 for i in range(len(S0))])
    moved = np.linalg.norm(disp, axis=1)
    i = int(np.argmax(dd))
    return {"max_move_mm": round(float(moved.max()), 3),
            "max_face_angle_change_deg": round(float(dd.max()), 2),
            "p99_face_angle_change_deg": round(float(np.percentile(dd, 99)), 2),
            "worst_edge_at": np.round(S0[edges[i]].mean(0), 1).tolist(),
            "max_disp_laplacian_mm": round(float(lap.max()), 3),
            "disp_laplacian_over_move": round(float(lap.max() / max(moved.max(), 1e-9)), 3)}
