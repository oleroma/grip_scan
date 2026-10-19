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
     ring move until those phalanges just touch the case (`finger_press_mm`):
     the middle phalanges grab the front of the handle next to their PIP
     joint (`pip_pad_frac`) and press it into the pads at the finger roots,
     next to the knuckles (`root_pad_frac`). Elsewhere they may only clear.
  5. The pinky has no button: it is turned so it curls in the same plane
     as the ring finger (wrapping the side of the handle, not its bottom)
     and changes its knuckle and middle-joint curl by as much as the ring
     finger did (plus `pinky_extra_curl_deg`); its fingertip keeps its
     grabbing bend. The case walls under all three pinky phalanges then
     move until they touch.
  6. The palm cups: the ring and pinky metacarpals flex towards the palm
     (`palm_cup_deg`), deepening the knuckle arch into a cup around the
     palm swell.
  7. The back edge of the top face moves forward (towards the sticks) so
     the thenar rests on a smooth slope, but the case never comes closer
     than `stick_clearance_mm` to the stick and home button cutouts
     (collection "stick cutouts", baked from Grip.blend).

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
    "pinky_extra_curl_deg": 0.0,  # pinky curls this much more (+) or less (-) than its posed curl
    "min_thickness_mm": 10.0,   # walls may not push the handle thinner than this
    "root_pad_frac": 0.35,      # proximal phalanges touch over this part next to the knuckle
    "pip_pad_frac": 0.4,        # middle phalanges touch over this part next to the PIP joint
    "palm_cup_deg": 10.0,       # pinky metacarpal flexes this much towards the palm (ring half)
    "top_back_max_mm": 30.0,    # top face back edge may move forward up to this far
    "stick_clearance_mm": 0.8,  # case stays at least this far from the stick/home cutouts
}
CUTOUTS = "stick cutouts"


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

    def cup(self, ang):
        """Flex the ring and pinky metacarpals towards the palm (ring half as much),
        so the knuckle arch deepens into a cup. Their fingers move along."""
        for f, k in (("ring", 0.5), ("pinky", 1.0)):
            F = self.f[f]
            c = self.head[f + "_metacarpal"]
            meta = F["mcp"] - c
            R = rot(np.cross(meta / np.linalg.norm(meta), self.palm_normal), k * ang)
            F["mcp"] = c + R @ meta
            F["d"] = [R @ v for v in F["d"]]
            F["n"] = R @ F["n"]
            self.tail[f + "_metacarpal"] = F["mcp"]

    def knuckle_arch(self):
        """How far each knuckle sits towards the palm side, relative to the middle knuckle (mm)."""
        m = self.f["middle"]["mcp"]
        return {f: round(float(np.dot(self.f[f]["mcp"] - m, self.palm_normal)), 1) for f in FINGERS}

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

    def min_thickness(self, bvh, below_z, min_x=30.0):
        """Thinnest point of the handle: ray from each surface point straight inwards."""
        dg = bpy.context.evaluated_depsgraph_get()
        ev = self.obj.evaluated_get(dg)
        me = ev.to_mesh()
        best = (1e9, None)
        for v in me.vertices:
            if v.co.x < min_x or v.co.z > below_z:
                continue
            hit = bvh.ray_cast(v.co - v.normal * 0.05, -v.normal)
            if hit[0] is not None and hit[3] + 0.05 < best[0]:
                best = (hit[3] + 0.05, np.array(v.co))
        ev.to_mesh_clear()
        return best

    def side_bvh(self):
        """BVH of the case surface without its flat top face (the cutouts pass through that)."""
        dg = bpy.context.evaluated_depsgraph_get()
        ev = self.obj.evaluated_get(dg)
        me = ev.to_mesh()
        me.calc_loop_triangles()
        V = [tuple(v.co) for v in me.vertices]
        T = [tuple(t.vertices) for t in me.loop_triangles if min(V[i][2] for i in t.vertices) < 58.95]
        ev.to_mesh_clear()
        return BVHTree.FromPolygons(V, T)

    @staticmethod
    def cutout_samples(step=0.7):
        """Points on the stick/home cutout surfaces (inside the case, Z <= 59)."""
        col = bpy.data.collections.get(CUTOUTS)
        pts, names = [], []
        for o in (col.objects if col else []):
            if o.type != "MESH":
                continue
            me = o.data
            me.calc_loop_triangles()
            mw = np.array(o.matrix_world)
            V = np.array([(mw @ np.array([*v.co, 1.0]))[:3] for v in me.vertices])
            for tri in me.loop_triangles:
                a, b, c = V[list(tri.vertices)]
                k = max(2, int(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)) / step))
                for i in range(k + 1):
                    for j in range(k + 1 - i):
                        p = a + (b - a) * i / k + (c - a) * j / k
                        if p[2] <= 59.0:
                            pts.append(p)
                            names.append(o.name)
        return np.array(pts), names

    @staticmethod
    def clearance(bvh, pts, names):
        best = (1e9, None)
        for p, n in zip(pts, names):
            d = bvh.find_nearest(Vector(p))[3]
            if d is not None and d < best[0]:
                best = (d, n)
        return best

    def segment_gap(self, bvh, a, b, r, samples=13):
        best = (1e9, None, None, None)
        for t in np.linspace(0, 1, samples):
            p = a + t * (b - a)
            d, loc, nor = self.sd(bvh, p)
            if d - r < best[0]:
                best = (d - r, loc, nor, p)
        return best


# ---------------------------------------------------------------- case rig
HANDLES = ["bump", "middle_proximal", "middle_middle", "ring_proximal", "ring_middle",
           "pinky_proximal", "pinky_middle", "pinky_distal", "top_back"]


def top_back_weights(co):
    """Back edge of the top face (full), fading down the back of the shoulder; the
    outer side and everything in front of the sticks stay put."""
    wz = np.interp(co[:, 2], [38.0, 47.0, 56.0, 58.1, 59.0], [0.0, 0.15, 0.45, 0.75, 1.0])
    wy = 1.0 - smoothstep(-62.0, -54.0, co[:, 1])
    wx = 1.0 - smoothstep(50.0, 70.0, co[:, 0])
    return wz * wy * wx


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
        if pb.name.startswith(("middle", "ring", "pinky")) or pb.name == "top_back":
            pb.lock_location = (True, False, True)  # slide along the wall normal only
    # weights on the cage (half mesh, X >= 0)
    co = np.array([v.co for v in case_obj.data.vertices])
    mask = 1.0 - smoothstep(params["free_below_z"], params["fixed_above_z"], co[:, 2])
    for bp in button_pts:
        dist = np.linalg.norm(co - bp, axis=1)
        mask *= smoothstep(params["button_keep_mm"], params["button_keep_mm"] + 8.0, dist)
    W = {}
    for name in HANDLES:
        if name == "top_back":
            W[name] = top_back_weights(co)
            continue
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
    arch0 = hand.knuckle_arch()
    hand.cup(math.radians(P["palm_cup_deg"]))
    rep["knuckle_arch_mm"] = {"before": arch0, "after": hand.knuckle_arch()}

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

    # ---- 5 (posed before the walls): pinky wraps like the ring finger
    Fp = hand.f["pinky"]
    ring_flex, ring_ab = angles["ring"]
    r_pts, _, r_n = hand.chain("ring", t, ring_flex, ring_ab)
    extra = math.radians(P["pinky_extra_curl_deg"])
    ring_delta = np.array(ring_flex) - np.array(hand.f["ring"]["ang"])
    flex_p = np.array(Fp["ang"]) + np.array([ring_delta[0], ring_delta[1], 0.0]) + extra * np.array([0.6, 1.0, 0.7])

    def capsule_gap(pa, pb, ra, qa, qb, rb):
        u = np.linspace(0, 1, 9)
        A = pa + u[:, None] * (pb - pa)
        B = qa + u[:, None] * (qb - qa)
        return float(np.min(np.linalg.norm(A[:, None] - B[None], axis=2))) - ra - rb

    def pinky_res(x):
        pts, _, n = hand.chain("pinky", t, flex_p, x[0])
        res = list(10.0 * (n - r_n))                       # same curl plane as the ring finger
        for i in range(3):                                 # but never inside the ring finger
            for j in range(3):
                g = capsule_gap(pts[i], pts[i + 1], Fp["r"][i], r_pts[j], r_pts[j + 1], hand.f["ring"]["r"][j])
                res.append(5.0 * min(0.0, g - 0.5))
        return np.array(res)

    x, _ = lm_solve(pinky_res, [0.0], [0.005])
    ab_p = float(x[0])
    angles["pinky"] = (list(flex_p), ab_p)
    p_pts, _, p_n = hand.chain("pinky", t, flex_p, ab_p)
    rep["pinky"] = {"flex_deg": [round(float(v), 1) for v in np.degrees(flex_p)],
                    "ref_flex_deg": [round(float(v), 1) for v in np.degrees(Fp["ang"])],
                    "abduction_deg": round(math.degrees(ab_p), 1),
                    "plane_vs_ring_deg": round(math.degrees(math.acos(np.clip(np.dot(p_n, r_n), -1, 1))), 1),
                    "gap_to_ring_mm": round(min(capsule_gap(p_pts[i], p_pts[i + 1], Fp["r"][i], r_pts[j], r_pts[j + 1],
                                                            hand.f["ring"]["r"][j]) for i in range(3) for j in range(3)), 1),
                    "lowest_z": round(float(min(p[2] for p in p_pts)) - Fp["r"][2], 1)}

    # ---- 4: case walls. Each wall touches over its contact zone: the finger-root pads (palm over
    # the knuckle and the base of the proximal phalanx), the middle phalanx next to its PIP joint,
    # the pinky's fingertip all along. Everywhere else the hand may only clear the case.
    centers = {"bump": swell}
    normals = {"bump": np.array([1.0, 0.0, 0.0])}
    segs = {}

    def zone_gap(bvh, caps):
        return min((case.segment_gap(bvh, a, b, r) for a, b, r in caps), key=lambda g: g[0])

    for f in ("middle", "ring", "pinky"):
        flex, ab = angles[f]
        pts, dirs, _ = hand.chain(f, t, flex, ab)
        mcp, wrist = pts[0], hand.head[f + "_metacarpal"] + t
        r_meta = hand.radius[f + "_metacarpal"]
        for kk, s in enumerate(SEGS if f == "pinky" else SEGS[:2]):
            a, b, r = pts[kk], pts[kk + 1], hand.f[f]["r"][kk]
            if kk == 0:
                zone = [(mcp, mcp + 0.2 * (wrist - mcp), r_meta), (a, a + P["root_pad_frac"] * (b - a), r)]
                full = [(a, b, r), (mcp, mcp + 0.4 * (wrist - mcp), r_meta)]
            elif kk == 1:
                zone = [(a, a + P["pip_pad_frac"] * (b - a), r)]
                full = [(a, b, r)]
            else:
                zone = full = [(a, b, r)]
            g, loc, nor, p = zone_gap(bvh0, zone)
            centers[f"{f}_{s}"] = loc
            normals[f"{f}_{s}"] = nor
            segs[f"{f}_{s}"] = (zone, full, g)
    top = case_obj.data.vertices
    y_top = min(v.co.y for v in top if v.co.z > 58.9)
    centers["top_back"] = np.array([30.0, y_top, 59.0])
    normals["top_back"] = np.array([0.0, 1.0, 0.0])
    if build or not (rig.pose and len(rig.pose.bones)) or "top_back" not in rig.pose.bones:
        build_handles(rig, case_obj, centers, normals, P, [r2_loc, r4_loc])
    walls = [n for n in HANDLES if n not in ("bump", "top_back")]
    rep["walls_before_mm"] = {n: round(float(segs[n][2]), 1) for n in walls}

    thick0 = case.min_thickness(bvh0, P["fixed_above_z"])[0]
    t_min = min(P["min_thickness_mm"], thick0)

    def wall_res(x):
        set_handles(rig, t, dict(zip(walls, x)))
        bvh = case.bvh()
        res = []
        for n in walls:
            zone, full, _ = segs[n]
            res.append(zone_gap(bvh, zone)[0] + P["finger_press_mm"])                       # touch in the zone
            res.append(2.0 * min(0.0, zone_gap(bvh, full)[0] + P["finger_press_mm"]))       # no digging
        res.append(4.0 * min(0.0, case.min_thickness(bvh, P["fixed_above_z"])[0] - t_min))
        res += list(0.1 * np.asarray(x))                   # smallest moves that do the job
        return np.array(res)

    x0 = np.array([segs[n][2] for n in walls])
    x, r = lm_solve(wall_res, x0, [0.5] * len(walls), iters=15)
    wall_off = dict(zip(walls, x))
    set_handles(rig, t, wall_off)
    bvh1 = case.bvh()
    thk, thk_at = case.min_thickness(bvh1, P["fixed_above_z"])
    rep["min_thickness_mm"] = {"before": round(float(thick0), 1), "after": round(float(thk), 1),
                               "at": [round(float(c)) for c in thk_at]}
    rep["wall_moves_mm"] = {n: round(float(v), 1) for n, v in zip(walls, x)}
    rep["palm_on_swell_mm"] = round(float(min(
        case.segment_gap(bvh1, hand.f[f]["mcp"] + t, hand.head[f + "_metacarpal"] + t,
                         hand.radius[f + "_metacarpal"])[0] for f in ("middle", "ring", "pinky"))), 1)
    rep["walls_after_mm"] = {n: {"contact_zone": round(float(zone_gap(bvh1, zone)[0]), 1),
                                 "anywhere": round(float(zone_gap(bvh1, full)[0]), 1)}
                             for n, (zone, full, _) in segs.items()}

    # ---- 7: top face back edge forward, as far as the stick cutouts allow
    pts_c, names_c = case.cutout_samples()
    if len(pts_c) and P["top_back_max_mm"] > 0:
        def clear_at(m):
            set_handles(rig, t, {**wall_off, "top_back": m})
            return case.clearance(case.side_bvh(), pts_c, names_c)

        c0 = clear_at(0.0)
        lo, hi = 0.0, float(P["top_back_max_mm"])
        if clear_at(hi)[0] >= P["stick_clearance_mm"]:
            lo = hi
        elif c0[0] > P["stick_clearance_mm"]:
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                if clear_at(mid)[0] >= P["stick_clearance_mm"]:
                    lo = mid
                else:
                    hi = mid
        c1 = clear_at(lo)
        wall_off["top_back"] = lo
        rep["top_back"] = {"moved_forward_mm": round(lo, 1), "clearance_before_mm": round(c0[0], 2),
                           "clearance_mm": round(c1[0], 2), "closest_cutout": c1[1]}
    else:
        rep["top_back"] = {"moved_forward_mm": 0.0, "note": f"no '{CUTOUTS}' collection" if not len(pts_c) else "off"}
    set_handles(rig, t, wall_off)
    bvh1 = case.bvh()
    # distal phalanges should stay free (released)
    for f in ("middle", "ring"):
        flex, ab = angles[f]
        pts, dirs, _ = hand.chain(f, t, flex, ab)
        g = case.segment_gap(bvh1, pts[2], pts[2] + 0.6 * (pts[3] - pts[2]), hand.f[f]["r"][2])[0]
        rep[f]["distal_clearance_mm"] = round(float(g), 1)
    # buttons stayed put?
    rep["button_shift_mm"] = {n: round(float(abs(case.sd(bvh1, p)[0] - case.sd(bvh0, p)[0])), 2)
                              for n, p in (("R1", R1), ("R2", R2), ("R4", R4))}

    pose_hand(hand, t, angles)
    # markers
    put_marker("index on R1 (solved)", a_idx, 2.5)
    put_marker("palm pocket (solved)", pocket, 3.0)
    rep["summary"] = (f"R1 overlap {P['r1_overlap_mm']:.0f} mm from index tip -> hand shift Y {ty:+.1f} mm, "
                      f"Z {tz:+.1f} mm; bump at Y {pocket[1]:.1f} ({pocket[1] - r1_loc[1]:.1f} mm from R1); "
                      "walls " + ", ".join(f"{n} {v:+.1f}" for n, v in rep["wall_moves_mm"].items()) +
                      f"; top face back edge +{rep['top_back']['moved_forward_mm']:.1f} mm")
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
