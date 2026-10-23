"""solve_grip.py: pose the grip rig so the soft hand (approved hold) closes round a smooth handle.

    $PY solve_grip.py [--work /tmp/claude-0/hs/work/grip_rig] [--palm-give 3.5] [--finger-give 1.5]

Idea (Olex, 2026-10-22): a hand takes the shape of what it holds, so the grip does not copy the small
bumps of the skin. The grip only moves its few bones (palm swell, three fins, stand, thenar surface)
until the inside of the hand rests on it with a light press everywhere it should touch:

  contact   each contact part of the hand (palm pads over the metacarpals, the inner side of the middle,
            ring and pinky fingers, the thumb base) should be pressed in by about half its soft-tissue
            give at its closest spot (gaps are filled, deep presses are relieved)
  give      no skin sample may be pressed in deeper than its give (palm 3.5 mm, fingers 1.5 mm)
  stand     the controller must still stand on the two pinky fins and the two handle ends: nothing else
            touches the table and the centre of mass sits between those supports
  calm      small bone moves (a soft pull back to the original shape)

Everything is measured on the exact basemesh surface the rig produces (solve_core.Model). The pose is
written to <work>/solve.json; pose_in_blender.py puts it on the rig and checks it in Blender.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rig_core as rc  # noqa: E402
import solve_core as sc  # noqa: E402

# hand parts that should rest on the grip, and which give applies
CONTACT = {
    "index_metacarpal": "palm", "middle_metacarpal": "palm", "ring_metacarpal": "palm",
    "pinky_metacarpal": "palm", "thumb_metacarpal": "palm",
    "middle_proximal": "finger", "middle_middle": "finger",
    "ring_proximal": "finger", "ring_middle": "finger",
    "pinky_proximal": "finger", "pinky_middle": "finger",
}
REACH_MM = 8.0          # a part further than this from the original grip is not expected to touch it
NEAR_MM = 14.0          # skin samples further than this are ignored
SOFTMIN_FRAC = 0.05     # a part's contact depth = mean of its closest 5% samples (at least 8)
W_CONTACT, W_GIVE, W_CALM, W_STAND, W_SMOOTH = 1.0, 3.0, 0.4, 2.0, 5.0
LAP_FREE_MM = 0.2     # bending of the change field below this (umbrella Laplacian, mm) is free
STAND_CLEAR_MM = 1.0    # other parts of the case must stay this far above the table
COM_MARGIN_MM = 8.0     # centre of mass at least this far inside the supports


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/claude-0/hs/work/grip_rig")
    ap.add_argument("--palm-give", type=float, default=3.5)
    ap.add_argument("--finger-give", type=float, default=1.5)
    ap.add_argument("--smooth", type=float, default=W_SMOOTH, help="weight of the smoothness term")
    ap.add_argument("--out", default=None)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return ap.parse_args(argv)


# ---------------------------------------------------------------- geometry helpers
def closest_on_mesh(P, V, T, k=16):
    """Closest triangle and barycentric coordinates of each point (candidates: k nearest centroids)."""
    C = V[T].mean(1)
    _, idx = cKDTree(C).query(P, k=k)
    best_d = np.full(len(P), np.inf)
    best_t = np.zeros(len(P), np.int64)
    best_b = np.zeros((len(P), 3))
    for j in range(k):
        t = idx[:, j]
        a, b, c = V[T[t, 0]], V[T[t, 1]], V[T[t, 2]]
        q, bc = _closest_tri(P, a, b, c)
        d = np.linalg.norm(P - q, axis=1)
        m = d < best_d
        best_d[m], best_t[m], best_b[m] = d[m], t[m], bc[m]
    return best_t, best_b


def _closest_tri(p, a, b, c):
    """Closest point on triangles (vectorised, Ericson) and its barycentric coordinates."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-18, 1e-18, den)
    v, w = vb / den, vc / den
    bc = np.stack([1 - v - w, v, w], 1)

    def put(m, val):
        bc[m] = val[m] if val.ndim == 2 else val

    put((d1 <= 0) & (d2 <= 0), np.array([1.0, 0, 0]))
    put((d3 >= 0) & (d4 <= d3), np.array([0, 1.0, 0]))
    put((d6 >= 0) & (d5 <= d6), np.array([0, 0, 1.0]))
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = np.clip(d1 / np.where(np.abs(d1 - d3) < 1e-18, 1e-18, d1 - d3), 0, 1)
    put(m, np.stack([1 - t, t, 0 * t], 1))
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = np.clip(d2 / np.where(np.abs(d2 - d6) < 1e-18, 1e-18, d2 - d6), 0, 1)
    put(m, np.stack([1 - t, 0 * t, t], 1))
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = np.clip((d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) < 1e-18, 1e-18, (d4 - d3) + (d5 - d6)), 0, 1)
    put(m, np.stack([0 * t, 1 - t, t], 1))
    q = bc[:, :1] * a + bc[:, 1:2] * b + bc[:, 2:] * c
    return q, bc


def signed_distance(P, V, T):
    """Exact signed distance of points to the closed grip (negative = inside the plastic)."""
    t, b = closest_on_mesh(P, V, T)
    q = (b[:, :, None] * V[T[t]]).sum(1)
    d = np.linalg.norm(P - q, axis=1)
    w = rc.winding(V, T, P)
    return np.where(np.abs(w) > 0.5, -d, d)


def com(V, T):
    P = V[T]
    v = np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 1], P[:, 2])) / 6.0
    return (v[:, None] * P.sum(1) / 4.0).sum(0) / v.sum()


# ---------------------------------------------------------------- the stand
class Stand:
    """The controller on a table, resting on the two pinky fins and the two handle ends (the case is
    mirror-symmetric, so the table plane contains X and only tilts in the YZ plane)."""

    def __init__(self, S0, marks):
        pk, st = np.array(marks["fin pinky"]["p"]), np.array(marks["stand"]["p"])
        Sa = S0 * [-1, 1, 1]
        d_pk = np.minimum(np.linalg.norm(S0 - pk, axis=1), np.linalg.norm(Sa - pk, axis=1))
        self.pinky = d_pk < 18.0
        self.end = (S0[:, 1] < -62.0) & (S0[:, 2] < 18.0)
        self.other = ~(self.pinky | self.end)
        self.theta = np.radians(np.arange(-70.0, 20.01, 0.25))
        self.N = np.stack([0 * self.theta, np.sin(self.theta), -np.cos(self.theta)], 1)   # "down"

    def measure(self, S, T):
        H = S @ self.N.T                                   # (n_out, n_theta): depth along each down
        hp, he, ho = H[self.pinky].max(0), H[self.end].max(0), H[self.other].max(0)
        f = hp - he
        k = np.nonzero(np.sign(f[:-1]) != np.sign(f[1:]))[0]
        if not len(k):
            return {"ok": False, "theta_deg": None, "clearance_mm": -10.0, "com_margin_mm": -10.0}
        k = k[0]
        a = f[k] / (f[k] - f[k + 1])
        th = self.theta[k] + a * (self.theta[k + 1] - self.theta[k])
        n = np.array([0.0, np.sin(th), -np.cos(th)])
        h = S @ n
        table = max(h[self.pinky].max(), h[self.end].max())
        clear = table - h[self.other].max()
        u = np.cross(n, [1.0, 0.0, 0.0])                   # along the table, front-back
        p_pk = S[self.pinky][np.argmax(h[self.pinky])]
        p_en = S[self.end][np.argmax(h[self.end])]
        c = com(S, T)
        lo, hi = sorted([p_pk @ u, p_en @ u])
        margin = min(c @ u - lo, hi - c @ u)
        return {"ok": True, "theta_deg": float(np.degrees(th)), "clearance_mm": float(clear),
                "com_margin_mm": float(margin), "support_span_mm": float(hi - lo),
                "pinky_support": p_pk.round(2).tolist(), "end_support": p_en.round(2).tolist(),
                "com": c.round(2).tolist()}


# ---------------------------------------------------------------- the solve
class Problem:
    def __init__(self, work, palm_give, finger_give):
        self.m = sc.Model(work)
        self.T = rc.orient_outward(self.m.S0, self.m.T)
        h = np.load(os.path.join(work, "hand_hold.npz"))
        info = json.load(open(os.path.join(work, "rig.json")))
        self.marks = info["landmarks"]
        names = list(h["seg_names"])
        V, seg = h["V"], h["seg"]
        sd0 = signed_distance(V, self.m.S0, self.T)
        keep = sd0 < NEAR_MM
        self.P, self.seg, self.sd0 = V[keep], seg[keep], sd0[keep]
        self.seg_names = names
        give = {"palm": palm_give, "finger": finger_give}
        self.give = np.full(len(self.P), finger_give)        # every skin sample has a give
        for i, n in enumerate(names):
            if n in CONTACT:
                self.give[self.seg == i] = give[CONTACT[n]]
            elif n.endswith("metacarpal") or n == "palm":
                self.give[self.seg == i] = palm_give
        self.parts = []
        for i, n in enumerate(names):
            mk = self.seg == i
            if n in CONTACT and mk.any() and self.sd0[mk].min() < REACH_MM:
                self.parts.append((n, np.nonzero(mk)[0], give[CONTACT[n]]))
        self.stand = Stand(self.m.S0, self.marks)
        from scipy.sparse import coo_matrix, diags
        e = np.concatenate([self.T[:, [0, 1]], self.T[:, [1, 2]], self.T[:, [2, 0]]])
        e = np.unique(np.sort(e, 1), axis=0)
        n = len(self.m.S0)
        Adj = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
        self.L = diags(np.ones(n)) - diags(1.0 / np.asarray(Adj.sum(1)).ravel()) @ Adj   # umbrella Laplacian
        self.update_corr(np.zeros(self.m.n))

    def update_corr(self, x):
        S = self.m.surface(x)
        self.tri, self.bary = closest_on_mesh(self.P, S, self.T)
        self.inside0 = np.abs(rc.winding(S, self.T, self.P)) > 0.5

    def sd(self, S):
        """Signed distance from the current correspondences (exact enough for moves of a few mm)."""
        A = S[self.T[self.tri]]
        q = (self.bary[:, :, None] * A).sum(1)
        n = np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        d = np.linalg.norm(self.P - q, axis=1)
        s = np.sign(((self.P - q) * n).sum(1))
        # samples sitting on an edge/vertex: trust the winding sign of the last exact update
        s = np.where(np.abs(((self.P - q) * n).sum(1)) < 0.3 * d, np.where(self.inside0, -1.0, 1.0), s)
        return s * d

    def contact_depths(self, sd):
        out = {}
        for n, idx, g in self.parts:
            k = max(8, int(SOFTMIN_FRAC * len(idx)))
            out[n] = float(np.sort(sd[idx])[:k].mean())
        return out

    def residuals(self, z):
        x = z * self.m.scale
        S = self.m.surface(x)
        sd = self.sd(S)
        r = []
        for n, idx, g in self.parts:
            k = max(8, int(SOFTMIN_FRAC * len(idx)))
            depth = np.sort(sd[idx])[:k].mean()
            r.append(W_CONTACT * (depth + 0.5 * g))           # target: pressed in by half the give
        over = np.maximum(0.0, -sd - self.give)                 # pressed deeper than the give
        r.append(W_GIVE * np.sqrt((over ** 2).sum()))
        r.extend(W_CALM * z)
        lap = np.linalg.norm(self.L @ (S - self.m.S0), axis=1)
        r.append(self.w_smooth * np.sqrt((np.maximum(0.0, lap - LAP_FREE_MM) ** 2).sum()))
        st = self.stand.measure(S, self.T)
        r.append(W_STAND * max(0.0, STAND_CLEAR_MM - st["clearance_mm"]))
        r.append(W_STAND * max(0.0, COM_MARGIN_MM - st["com_margin_mm"]))
        return np.array(r)

    def jac(self, z, h=0.05):
        """Central differences (the residuals are only piecewise smooth below ~0.01 mm)."""
        J = np.zeros((len(self.residuals(z)), len(z)))
        for j in range(len(z)):
            e = np.zeros(len(z))
            e[j] = h
            J[:, j] = (self.residuals(z + e) - self.residuals(z - e)) / (2 * h)
        return J

    def report(self, x, exact=True):
        S = self.m.surface(x)
        sd = signed_distance(self.P, S, self.T) if exact else self.sd(S)
        rep = {"contact_depth_mm (negative = pressed in)": {k: round(v, 2) for k, v in self.contact_depths(sd).items()},
               "deepest_press_mm": round(float(-sd.min()), 2),
               "over_give_samples": int((-sd > self.give + 0.05).sum()),
               "over_give_max_mm": round(float(np.max(-sd - self.give)), 2),
               "stand": self.stand.measure(S, self.T)}
        by = {}
        for i, n in enumerate(self.seg_names):
            mk = self.seg == i
            if mk.any():
                by[n] = round(float(-sd[mk].min()), 2)
        rep["deepest_press_by_part_mm"] = dict(sorted(by.items(), key=lambda kv: -kv[1])[:8])
        mv = np.linalg.norm(S - self.m.S0, axis=1)
        rep["largest_move_mm"] = round(float(mv.max()), 2)
        rep["largest_move_at"] = np.round(self.m.S0[int(mv.argmax())], 1).tolist()
        rep["zone_max_move_mm"] = round(float(mv[self.m.out_in_zone].max()), 4)
        rep["smooth"] = rc.smooth_report(self.m.S0, S, self.T)
        return rep


def main():
    a = args()
    t0 = time.time()
    cache = os.path.join(a.work, f"problem_{a.palm_give}_{a.finger_give}.pkl")
    if os.path.exists(cache):
        pb = pickle.load(open(cache, "rb"))
    else:
        pb = Problem(a.work, a.palm_give, a.finger_give)
        pickle.dump(pb, open(cache, "wb"))
    pb.w_smooth = a.smooth
    print("[solve] skin samples", len(pb.P), "contact parts", [p[0] for p in pb.parts], f"{time.time() - t0:.1f} s")
    rest = pb.report(np.zeros(pb.m.n))
    print("[solve] rest", json.dumps(rest["contact_depth_mm (negative = pressed in)"]), "stand", json.dumps(rest["stand"]))
    z = np.zeros(pb.m.n)
    for it in range(10):
        res = least_squares(pb.residuals, z, jac=pb.jac, x_scale=1.0, max_nfev=60)
        moved = np.abs(res.x - z).max()
        z = res.x
        pb.update_corr(z * pb.m.scale)
        print(f"[solve] round {it}: cost {res.cost:.3f}, change {moved:.3f}, {time.time() - t0:.0f} s")
        if moved < 0.01:
            break
    x = z * pb.m.scale
    posed = pb.report(x)
    out = {"channels": [{"bone": b, "kind": k, "axis": int(ax), "value": float(v),
                         "unit": "mm" if k == "loc" else "rad"} for (b, k, ax, _), v in zip(sc.CHANNELS, x)],
           "x": x.tolist(),
           "settings": {"palm_give_mm": a.palm_give, "finger_give_mm": a.finger_give, "reach_mm": REACH_MM,
                        "weights": [W_CONTACT, W_GIVE, W_CALM, W_STAND, a.smooth], "stand_clear_mm": STAND_CLEAR_MM,
                        "com_margin_mm": COM_MARGIN_MM},
           "rest": rest, "posed": posed, "seconds": round(time.time() - t0, 1)}
    path = a.out or os.path.join(a.work, "solve.json")
    json.dump(out, open(path, "w"), indent=1)
    print("[solve] posed", json.dumps({k: v for k, v in posed.items() if k != "smooth"}))
    print("[solve] smooth", json.dumps(posed["smooth"]))
    print("[solve] pose", [(c["bone"], c["kind"], c["axis"], round(c["value"], 3)) for c in out["channels"]])
    print("[solve] wrote", path)


if __name__ == "__main__":
    main()
