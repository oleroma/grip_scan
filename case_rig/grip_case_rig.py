"""Grip case rig: fit the GripV case to a measured hand by kinematics.

Runs inside Blender (Text Editor > Run Script) or with the `bpy` module.
Needs only numpy (bundled with Blender); no scipy.

The procedure follows Olex's description (2026-10-18):
  1. The hand keeps the orientation of the hand-posed reference hold
     (palm roughly perpendicular to X, fingers along +Y).
  2. The index wraps the fixed top part and lands on R1. `r1_overlap_mm`
     says where along the index the R1 centre sits (measured from the
     fingertip end). This slides the hand in Y; the palm pocket and so the
     palm bump move with it.
  3. Middle and ring fingers curl until their fingertip pads sit on the
     R2/R4 centres, with the distal joint relaxed (`dip_release_deg`).
  4. The case walls under the proximal and middle phalanges of middle and
     ring move until those phalanges just touch the case (`finger_press_mm`).
  5. The pinky curls until it touches the case (reported only).

Everything the solver moves is a bone of the "case rig" armature, so the
case can also be adjusted by hand afterwards.
"""
import json
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

HAND = "hand rig"
CASE = "basemesh GripV"
RIG = "case rig"
FINGERS = ["index", "middle", "ring", "pinky"]
SEGS = ["proximal", "middle", "distal"]

DEFAULTS = {
    "r1_overlap_mm": 24.0,      # R1 centre distance from index fingertip end along the finger
    "dip_release_deg": 20.0,    # middle/ring distal joint flex on R2/R4
    "tip_pad_mm": 9.0,          # fingertip pad point that lands on R2/R4 centre (from tip end)
    "finger_press_mm": 0.5,     # proximal/middle phalanges press this far into the case
    "knuckle_z": 58.4,          # index knuckle (MCP) height; reference hold = 58.4
    "fixed_above_z": 38.0,      # case above this Z never moves (top part, index wrap)
    "free_below_z": 32.0,       # case below this Z moves fully
    "wall_sigma_mm": 12.0,      # reach of each finger-wall handle
    "bump_sigma_mm": 16.0,      # reach of the palm bump handle
    "button_keep_mm": 9.0,      # case within this distance of R2/R4 centres stays put
}


# ---------------------------------------------------------------- helpers
def rot(axis, ang):
    """Rotation matrix about a unit axis (radians)."""
    a = axis / np.linalg.norm(axis)
    x, y, z = a
    c, s = math.cos(ang), math.sin(ang)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def signed_angle(u, v, n):
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return math.atan2(np.dot(np.cross(u, v), n), np.dot(u, v))


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def lm_solve(fun, x0, steps, iters=40, lam=1e-2, tol=1e-4):
    """Small Levenberg-Marquardt with a finite-difference Jacobian."""
    x = np.array(x0, float)
    r = fun(x)
    cost = r @ r
    for _ in range(iters):
        J = np.empty((len(r), len(x)))
        for j in range(len(x)):
            dx = np.zeros_like(x)
            dx[j] = steps[j]
            J[:, j] = (fun(x + dx) - r) / steps[j]
        A = J.T @ J
        g = J.T @ r
        improved = False
        for _ in range(12):
            step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            xn = x + step
            rn = fun(xn)
            cn = rn @ rn
            if cn < cost:
                x, r, cost = xn, rn, cn
                lam = max(lam / 3, 1e-7)
                improved = True
                break
            lam *= 4
        if not improved or np.max(np.abs(step)) < tol:
            break
    return x, r


# ---------------------------------------------------------------- hand model
class Hand:
    """Rigid capsule hand built from the rest pose of the hand rig (= reference hold)."""

    def __init__(self, arm):
        self.arm = arm
        bones = arm.data.bones
        mw = np.array(arm.matrix_world)

        def w(v):
            return (mw @ np.array([*v, 1.0]))[:3]

        self.head = {b.name: w(b.head_local) for b in bones}
        self.tail = {b.name: w(b.tail_local) for b in bones}
        self.radius = {}
        for b in bones:
            skin = bpy.data.objects.get(b.name + " skin")
            self.radius[b.name] = self._skin_radius(skin, b) if skin else 8.0
        self.f = {}
        for f in FINGERS:
            mcp = self.tail[f + "_metacarpal"]
            pts = [mcp] + [self.tail[f"{f}_{s}"] for s in SEGS]
            d = [pts[i + 1] - pts[i] for i in range(3)]
            L = [np.linalg.norm(v) for v in d]
            d = [v / l for v, l in zip(d, L)]
            P = np.array(pts)
            n = np.linalg.svd(P - P.mean(0))[2][2]
            if signed_angle(d[0], d[1], n) < 0:
                n = -n
            meta = mcp - self.head[f + "_metacarpal"]
            meta /= np.linalg.norm(meta)
            ang = [signed_angle(meta, d[0], n), signed_angle(d[0], d[1], n), signed_angle(d[1], d[2], n)]
            self.f[f] = dict(mcp=mcp, d=d, L=L, n=n, ang=ang,
                             r=[self.radius[f"{f}_{s}"] for s in SEGS])
        # palm normal (pointing from the palm towards the case, ~ -X)
        P = np.array([self.tail[f + "_metacarpal"] for f in FINGERS] + [self.head["palm"]])
        pn = np.linalg.svd(P - P.mean(0))[2][2]
        self.palm_normal = pn if pn[0] < 0 else -pn

    @staticmethod
    def _skin_radius(skin, bone):
        dg = bpy.context.evaluated_depsgraph_get()
        me = skin.evaluated_get(dg).to_mesh()
        mw = np.array(skin.matrix_world)
        P = np.array([(mw @ np.array([*v.co, 1.0]))[:3] for v in me.vertices])
        skin.evaluated_get(dg).to_mesh_clear()
        pb = bpy.data.objects[HAND].pose.bones[bone.name]
        a, b = np.array(pb.head), np.array(pb.tail)
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
        return float(np.median(np.linalg.norm(P - (a + t[:, None] * ab), axis=1)))

    def chain(self, f, t, flex, abd=0.0):
        """Joint positions of finger f for hand translation t, absolute flex
        angles (MCP, PIP, DIP; radians) and MCP abduction (radians)."""
        F = self.f[f]
        n = F["n"]
        Ra = rot(self.palm_normal, abd)
        delta = np.cumsum(np.array(flex) - np.array(F["ang"]))
        pts = [F["mcp"] + t]
        dirs = []
        for k in range(3):
            dk = Ra @ (rot(n, delta[k]) @ F["d"][k])
            dirs.append(dk)
            pts.append(pts[-1] + F["L"][k] * dk)
        return pts, dirs, Ra @ n

    def point_from_tip(self, f, pts, dirs, s):
        """Axis point at distance s (mm) back from the fingertip end."""
        F = self.f[f]
        end = pts[3] + F["r"][2] * dirs[2]
        s_left = s
        p = end
        seglen = [F["L"][2] + F["r"][2], F["L"][1], F["L"][0]]
        for k, ln in zip([2, 1, 0], seglen):
            if s_left <= ln:
                return p - s_left * dirs[k], k
            p = p - ln * dirs[k]
            s_left -= ln
        return p, 0


# ---------------------------------------------------------------- case model
class Case:
    def __init__(self, obj):
        self.obj = obj

    def bvh(self):
        dg = bpy.context.evaluated_depsgraph_get()
        return BVHTree.FromObject(self.obj, dg)

    @staticmethod
    def sd(bvh, p):
        loc, nor, _, d = bvh.find_nearest(Vector(p))
        loc, nor = np.array(loc), np.array(nor)
        return (d if np.dot(p - loc, nor) >= 0 else -d), loc, nor

    def segment_gap(self, bvh, a, b, r, samples=13):
        best = (1e9, None, None, None)
        for t in np.linspace(0, 1, samples):
            p = a + t * (b - a)
            d, loc, nor = self.sd(bvh, p)
            if d - r < best[0]:
                best = (d - r, loc, nor, p)
        return best


# ---------------------------------------------------------------- case rig
HANDLES = ["bump", "middle_proximal", "middle_middle", "ring_proximal", "ring_middle"]


def ensure_case_rig(case_obj):
    rig = bpy.data.objects.get(RIG)
    if rig is None:
        data = bpy.data.armatures.new(RIG)
        rig = bpy.data.objects.new(RIG, data)
        for c in case_obj.users_collection:
            c.objects.link(rig)
        data.display_type = "STICK"
        rig.show_in_front = True
    if not any(m.type == "ARMATURE" and m.object == rig for m in case_obj.modifiers):
        mod = case_obj.modifiers.new("case rig", "ARMATURE")
        mod.object = rig
        mod.use_vertex_groups = True
        mod.use_bone_envelopes = False
        case_obj.modifiers.move(case_obj.modifiers.find(mod.name), 0)
    for k, v in DEFAULTS.items():
        if k not in rig:
            rig[k] = v
    return rig


def build_handles(rig, case_obj, centers, normals, params, button_pts):
    """(Re)create rig bones at the given surface points and weight the cage."""
    bpy.context.view_layer.objects.active = rig
    for o in bpy.context.selected_objects:
        o.select_set(False)
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = rig.data.edit_bones
    for b in list(eb):
        eb.remove(b)
    root = eb.new("case_fixed")
    root.head = (0, -50, 40)
    root.tail = (0, -50, 50)
    for name in HANDLES:
        b = eb.new(name)
        c = Vector(centers[name])
        n = Vector(normals[name]).normalized()
        b.head = c
        b.tail = c + n * 12.0
        b.parent = root
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in rig.pose.bones:
        pb.lock_rotation = (True, True, True)
        pb.lock_scale = (True, True, True)
        pb.rotation_mode = "XYZ"
        if pb.name.startswith(("middle", "ring")):
            pb.lock_location = (True, False, True)  # slide along the wall normal only
    # weights on the cage (half mesh, X >= 0)
    co = np.array([v.co for v in case_obj.data.vertices])
    mask = 1.0 - smoothstep(params["free_below_z"], params["fixed_above_z"], co[:, 2])
    for bp in button_pts:
        dist = np.linalg.norm(co - bp, axis=1)
        mask *= smoothstep(params["button_keep_mm"], params["button_keep_mm"] + 8.0, dist)
    W = {}
    for name in HANDLES:
        sig = params["bump_sigma_mm"] if name == "bump" else params["wall_sigma_mm"]
        dist = np.linalg.norm(co - np.array(centers[name]), axis=1)
        w = np.exp(-0.5 * (dist / sig) ** 2) * mask
        w[dist > 3 * sig] = 0.0
        W[name] = w
    tot = sum(W.values())
    scale = np.where(tot > 1.0, 1.0 / np.maximum(tot, 1e-9), 1.0)
    for name in HANDLES:
        W[name] *= scale
    W["case_fixed"] = np.clip(1.0 - sum(W[n] for n in HANDLES), 0.0, 1.0)
    for name, w in W.items():
        vg = case_obj.vertex_groups.get(name) or case_obj.vertex_groups.new(name=name)
        idx_all = list(range(len(co)))
        vg.remove(idx_all)
        for i in np.nonzero(w > 1e-4)[0]:
            vg.add([int(i)], float(w[i]), "REPLACE")
    return W


def set_handles(rig, bump_vec, wall_off):
    """bump_vec: world translation of the bump; wall_off: mm along each wall normal."""
    for pb in rig.pose.bones:
        pb.location = (0, 0, 0)
    pb = rig.pose.bones["bump"]
    R = np.array(pb.bone.matrix_local.to_3x3())
    pb.location = Vector(R.T @ np.array(bump_vec))
    for name, s in wall_off.items():
        rig.pose.bones[name].location = (0.0, float(s), 0.0)
    bpy.context.view_layer.update()


# ---------------------------------------------------------------- posing the hand rig
def pose_hand(hand, t, angles):
    """Write solved joint positions back onto the hand rig pose bones."""
    arm = hand.arm
    target = {}
    for name in hand.head:
        target[name] = (hand.head[name] + t, hand.tail[name] + t)
    for f, (flex, abd) in angles.items():
        pts, _, _ = hand.chain(f, t, flex, abd)
        for k, s in enumerate(SEGS):
            target[f"{f}_{s}"] = (pts[k], pts[k + 1])
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
    bpy.context.view_layer.update()
    order = sorted(arm.pose.bones, key=lambda p: len(p.parent_recursive))
    for pb in order:
        h, e = target[pb.name]
        rest = pb.bone.matrix_local
        y_rest = Vector(rest.col[1][:3]).normalized()
        y_new = Vector(e - h).normalized()
        q = y_rest.rotation_difference(y_new)
        m = (q.to_matrix() @ rest.to_3x3()).to_4x4()
        m.translation = Vector(h)
        pb.matrix = m
        bpy.context.view_layer.update()


# ---------------------------------------------------------------- the solve
def solve(params=None, build=True, log=print):
    arm = bpy.data.objects[HAND]
    case_obj = bpy.data.objects[CASE]
    rig = ensure_case_rig(case_obj)
    P = dict(DEFAULTS)
    P.update({k: rig[k] for k in DEFAULTS if k in rig})
    if params:
        P.update(params)
        for k, v in params.items():
            rig[k] = v
    # neutral rig, neutral hand
    if rig.pose and len(rig.pose.bones):
        set_handles(rig, (0, 0, 0), {})
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
    bpy.context.view_layer.update()
    hand = Hand(arm)
    case = Case(case_obj)
    bvh0 = case.bvh()
    R1 = np.array(bpy.data.objects["R1 target"].location)
    R2 = np.array(bpy.data.objects["R2 press"].location)
    R4 = np.array(bpy.data.objects["R4 press"].location)
    swell = np.array(bpy.data.objects["palm swell"].location)
    rep = {"params": P}

    # surface points + normals of the targets on the undeformed case
    def surf(p):
        _, loc, nor = case.sd(bvh0, p)
        return loc, nor

    r1_loc, r1_n = surf(R1)
    r2_loc, r2_n = surf(R2)
    r4_loc, r4_n = surf(R4)

    # ---- 1+2: index on R1 -> hand translation
    I = hand.f["index"]
    k_dip = I["ang"][2] / I["ang"][1]
    ref_ang = np.array(I["ang"])
    # longitudinal axis of R1: the index runs along the R1 face; compare along
    # the face tangent closest to the index direction at the reference.
    def index_res(x):
        ty, tz, dm, dp = x
        t = np.array([0.0, ty, tz])
        flex = [ref_ang[0] + dm, ref_ang[1] + dp, ref_ang[2] + k_dip * dp]
        pts, dirs, n = hand.chain("index", t, flex)
        a, k = hand.point_from_tip("index", pts, dirs, P["r1_overlap_mm"])
        # finger axis direction at that point, projected into the R1 face
        u = dirs[k] - np.dot(dirs[k], r1_n) * r1_n
        u /= np.linalg.norm(u)
        along = np.dot(a - r1_loc, u)                     # R1 centre at the chosen point
        res = [3.0 * along]
        # middle + distal phalanges lie on the case (touch, no gap)
        for kk in (1, 2):
            g = case.segment_gap(bvh0, pts[kk], pts[kk + 1], I["r"][kk], 9)[0]
            res.append(1.0 * (g + 0.5))
        # proximal may not dig in more than 1 mm
        g0 = case.segment_gap(bvh0, pts[0], pts[1], I["r"][0], 9)[0]
        res.append(2.0 * min(0.0, g0 + 1.0))
        res.append(1.0 * (pts[0][2] - P["knuckle_z"]))
        res += list(0.3 * np.degrees([dm, dp]) / 10.0)
        return np.array(res)

    x, r = lm_solve(index_res, [0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.01, 0.01])
    ty, tz, dm, dp = x
    t = np.array([0.0, ty, tz])
    flex_i = [ref_ang[0] + dm, ref_ang[1] + dp, ref_ang[2] + k_dip * dp]
    pts, dirs, _ = hand.chain("index", t, flex_i)
    a, k = hand.point_from_tip("index", pts, dirs, P["r1_overlap_mm"])
    u = dirs[k] - np.dot(dirs[k], r1_n) * r1_n
    u /= np.linalg.norm(u)
    a_idx = a
    rep["hand_shift_mm"] = {"y": round(float(ty), 1), "z": round(float(tz), 1)}
    rep["index"] = {
        "flex_deg": [round(float(v), 1) for v in np.degrees(flex_i)],
        "ref_flex_deg": [round(float(v), 1) for v in np.degrees(ref_ang)],
        "r1_along_error_mm": round(float(np.dot(a - r1_loc, u)), 1),
        "r1_height_offset_mm": round(float(np.dot(r1_loc - a, np.cross(u, r1_n))), 1),
        "knuckle_z": round(float(pts[0][2]), 1),
        "segment_gaps_mm": [round(float(case.segment_gap(bvh0, pts[i], pts[i + 1], I["r"][i])[0]), 1) for i in range(3)],
        "on_phalanx": SEGS[k],
    }
    angles = {"index": (flex_i, 0.0)}

    # ---- bump follows the palm pocket (orientation is unchanged, so by t)
    pocket = swell + t
    rep["bump"] = {"from": [round(float(v), 1) for v in swell], "to": [round(float(v), 1) for v in pocket],
                   "dy_from_r1_mm": round(float(pocket[1] - r1_loc[1]), 1)}

    # ---- 3: middle and ring tips onto R2 / R4 with relaxed DIP
    dip = math.radians(P["dip_release_deg"])
    for f, loc, nor in (("middle", r2_loc, r2_n), ("ring", r4_loc, r4_n)):
        F = hand.f[f]
        goal = loc + nor * F["r"][2]   # axis point one pad radius above the button centre

        def tip_res(x, F=F, f=f, goal=goal):
            m, p, ab, s = x
            pts, dirs, _ = hand.chain(f, t, [F["ang"][0] + m, F["ang"][1] + p, dip], ab)
            a, _ = hand.point_from_tip(f, pts, dirs, s)
            lo, hi = P["tip_pad_mm"] - 4.0, P["tip_pad_mm"] + 4.0
            return np.concatenate([a - goal,
                                   [0.6 * math.degrees(ab) / 5.0,
                                    0.2 * (s - P["tip_pad_mm"]),
                                    3.0 * (min(0.0, s - lo) + max(0.0, s - hi))]])

        x, r = lm_solve(tip_res, [0.0, 0.0, 0.0, P["tip_pad_mm"]], [0.01, 0.01, 0.01, 0.2])
        m, p, ab, s_pad = x
        flex = [F["ang"][0] + m, F["ang"][1] + p, dip]
        pts, dirs, _ = hand.chain(f, t, flex, ab)
        a, _ = hand.point_from_tip(f, pts, dirs, s_pad)
        angles[f] = (flex, ab)
        rep[f] = {
            "flex_deg": [round(float(v), 1) for v in np.degrees(flex)],
            "ref_flex_deg": [round(float(v), 1) for v in np.degrees(F["ang"])],
            "abduction_deg": round(float(np.degrees(ab)), 1),
            "tip_error_mm": round(float(np.linalg.norm(a - goal)), 1),
            "pad_point_mm_from_tip": round(float(s_pad), 1),
            "distal_vs_button_deg": round(float(90 - np.degrees(np.arccos(abs(np.dot(dirs[2], nor))))), 1),
        }

    # ---- 4: case walls under proximal + middle phalanges
    centers = {"bump": swell}
    normals = {"bump": np.array([1.0, 0.0, 0.0])}
    segs = {}
    for f in ("middle", "ring"):
        flex, ab = angles[f]
        pts, dirs, _ = hand.chain(f, t, flex, ab)
        for kk, s in ((0, "proximal"), (1, "middle")):
            g, loc, nor, p = case.segment_gap(bvh0, pts[kk], pts[kk + 1], hand.f[f]["r"][kk])
            centers[f"{f}_{s}"] = loc
            normals[f"{f}_{s}"] = nor
            segs[f"{f}_{s}"] = (pts[kk], pts[kk + 1], hand.f[f]["r"][kk], g)
    if build or not (rig.pose and len(rig.pose.bones)):
        build_handles(rig, case_obj, centers, normals, P, [r2_loc, r4_loc])
    walls = [n for n in HANDLES if n != "bump"]
    rep["walls_before_mm"] = {n: round(float(segs[n][3]), 1) for n in walls}

    def wall_res(x):
        set_handles(rig, t, dict(zip(walls, x)))
        bvh = case.bvh()
        return np.array([case.segment_gap(bvh, a, b, r)[0] + P["finger_press_mm"] for (a, b, r, _) in
                         (segs[n] for n in walls)])

    x0 = np.array([segs[n][3] for n in walls])
    x, r = lm_solve(wall_res, x0, [0.5] * 4, iters=15)
    set_handles(rig, t, dict(zip(walls, x)))
    bvh1 = case.bvh()
    rep["wall_moves_mm"] = {n: round(float(v), 1) for n, v in zip(walls, x)}
    rep["walls_after_mm"] = {n: round(float(case.segment_gap(bvh1, a, b, rr)[0]), 1)
                             for n, (a, b, rr, _) in segs.items()}
    # distal phalanges should stay free (released)
    for f in ("middle", "ring"):
        flex, ab = angles[f]
        pts, dirs, _ = hand.chain(f, t, flex, ab)
        g = case.segment_gap(bvh1, pts[2], pts[2] + 0.6 * (pts[3] - pts[2]), hand.f[f]["r"][2])[0]
        rep[f]["distal_clearance_mm"] = round(float(g), 1)
    # buttons stayed put?
    rep["button_shift_mm"] = {n: round(float(abs(case.sd(bvh1, p)[0] - case.sd(bvh0, p)[0])), 2)
                              for n, p in (("R1", R1), ("R2", R2), ("R4", R4))}

    # ---- 5: pinky curls until it touches
    Fp = hand.f["pinky"]
    base = np.array(Fp["ang"])
    curl_dir = np.array([0.6, 1.0, 0.7])

    def pinky_gap(c):
        flex = base + c * curl_dir
        pts, _, _ = hand.chain("pinky", t, flex)
        return min(case.segment_gap(bvh1, pts[i], pts[i + 1], Fp["r"][i], 9)[0] for i in range(3))

    lo, hi = math.radians(-60), math.radians(40)
    if pinky_gap(lo) > 0 and pinky_gap(hi) < 0:
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if pinky_gap(mid) > 0:
                lo = mid
            else:
                hi = mid
        c = lo
    else:
        c = 0.0
    angles["pinky"] = (list(base + c * curl_dir), 0.0)
    rep["pinky"] = {"flex_deg": [round(float(v), 1) for v in np.degrees(base + c * curl_dir)],
                    "ref_flex_deg": [round(float(v), 1) for v in np.degrees(base)]}

    pose_hand(hand, t, angles)
    # markers
    put_marker("index on R1 (solved)", a_idx, 2.5)
    put_marker("palm pocket (solved)", pocket, 3.0)
    rep["summary"] = (f"R1 overlap {P['r1_overlap_mm']:.0f} mm from index tip -> hand shift Y {ty:+.1f} mm, "
                      f"Z {tz:+.1f} mm; bump at Y {pocket[1]:.1f} ({pocket[1] - r1_loc[1]:.1f} mm from R1); "
                      "walls " + ", ".join(f"{n} {v:+.1f}" for n, v in rep["wall_moves_mm"].items()))
    txt = bpy.data.texts.get("case_rig_report") or bpy.data.texts.new("case_rig_report")
    txt.from_string(json.dumps(rep, indent=1))
    log(rep["summary"])
    return rep


def put_marker(name, p, size):
    o = bpy.data.objects.get(name)
    if o is None:
        o = bpy.data.objects.new(name, None)
        o.empty_display_type = "SPHERE"
        bpy.context.scene.collection.objects.link(o)
    o.empty_display_size = size
    o.location = Vector(p)


if __name__ == "__main__":
    solve()
