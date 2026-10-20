"""trigger_solver.py: fit Olex's GripV R2 / R4 trigger parameters to the posed hand.

    PY=/tmp/claude-0/venv/bin/python
    $PY trigger_solver.py -- --in IN.blend --out OUT.blend [--parts R2,R4] [--press 0.5]
                             [--use-x-offset] [--report report.json] [--check-case]
    $PY trigger_solver.py -- --in IN.blend --out OUT.blend --apply values.json     (write known values, no solve)
    $PY trigger_solver.py -- --in IN.blend --show                                   (print the current state only)

What it does (all in a scratch scene: only the triggers, their cutters, the PCB switches and the hand are
evaluated, ~0.1 s per trigger):
  1. Target of each trigger from the posed hand ('hand points' + 'hand rig'):
       R2 <- middle fingertip, R4 <- ring fingertip.
       pad point q   = on the distal phalanx axis, TIP_PAD mm back from the fingertip end
                       (middle 9.0, ring 6.4: the pad points of the v_27 solve), kept on the bone;
       press dir p   = the bone's Z axis of the distal bone in 'hand rig' = the way the fingertip moves
                       when the finger curls (perpendicular to the bone, in its bending plane);
       target centre = q + p * (radius - Press)   (the rigid fingertip sinks Press mm into the face)
       target normal = -p                          (press face perpendicular to the pressing direction)
  2. Solve the trigger parameters in 'GripV R2 parameters' / 'GripV R4 parameters' > node
     'trigger parameters.001' so that the press-face centre lands on the target and its normal matches.
     Press face = the outer cap of the finger end (1 mm 'top bevel' above the largest face flagged 'top';
     see press_face(): the 'top' faces themselves lie 1 mm lower and include a stopper step, so their
     average is off by 1-2 mm). Free by default: trigger length, trigger radius, button radius offset,
     shift z, button rotation, tip angle offset, tip shear (x offset only with --use-x-offset; tip shear
     is needed to turn the face out of the arc plane, R4 needs ~20 deg of it).
     The PCB switches ('R2_pcb_button', 'R4_pcb_button') never move.
     Soft constraints in the solve: no hand capsule may sink more than Press into the trigger part; the
     lever must stay under its switch (same switch-to-lever gap and the same number of lever hits under the
     switch footprint as before); the travel direction may not stray more than 25 deg (or than today) from
     the pressing direction; smallest change of the parameters. Hard bounds: R2 shift z >= -0.5
     (lever path to the switch stays open), sensible ranges for everything else.
     Takes ~30 s per trigger (least squares, forward differences).
  3. Checks: switch-to-lever gap (81 rays from the switch bottom down over its footprint), hand overlap per
     capsule, press-face centre / normal error, travel direction vs pressing direction; with --check-case
     also the case-dependent checks (needs the whole case, ~30 s): case material between switch and lever
     (81 rays, must be 0), trigger vertices inside case_bottom material.
  4. Writes the values into OUT.blend (never into Grip.blend or build/Grip_hand.blend) and prints a table
     old -> new. L2 / L4 are the mirrored R2 / R4 (same parameter groups), so they follow automatically.

Angles are radians in the node (button rotation, tip angle offset, tip shear); the tables print degrees.
"""
import os
import sys
import json
import math
import time
import numpy as np
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/tmp/claude-0/sdf/smooth/lib')
import gripmetrics as GM  # noqa: E402

PROTECTED = {'/home/claude/Grip/blender/Grip.blend', '/tmp/claude-0/sdf/build/Grip_hand.blend', '/tmp/claude-0/sdf/smooth/base.blend',
             '/tmp/claude-0/work/v_27.blend'}
PARTS = {
    'R2': dict(group='GripV R2 parameters', node='trigger parameters.001', finger='middle', switch='R2_pcb_button',
               cutter='R2 bool', left='L2', tip_pad=9.0),
    'R4': dict(group='GripV R4 parameters', node='trigger parameters.001', finger='ring', switch='R4_pcb_button',
               cutter='R4 bool', left='L4', tip_pad=6.4),
}
SHOWN = ('trigger length', 'trigger radius', 'button radius offset', 'shift z', 'button rotation', 'tip angle offset',
         'tip shear', 'x offset')
ANGLES = {'button rotation', 'tip angle offset', 'tip shear'}
FREE_DEFAULT = ['trigger length', 'trigger radius', 'button radius offset', 'shift z', 'button rotation', 'tip angle offset',
                'tip shear']
STEP = {'trigger length': 0.3, 'trigger radius': 1.0, 'button radius offset': 0.3, 'shift z': 0.3, 'x offset': 0.3,
        'button rotation': math.radians(0.5), 'tip angle offset': math.radians(0.5), 'tip shear': math.radians(0.5)}
SCALE = {'trigger length': 3.0, 'trigger radius': 15.0, 'button radius offset': 3.0, 'shift z': 3.0, 'x offset': 1.0,
         'button rotation': math.radians(10), 'tip angle offset': math.radians(10), 'tip shear': math.radians(10)}
BOUNDS = {'trigger length': (8.0, 70.0), 'trigger radius': (15.0, 250.0), 'button radius offset': (-6.0, 15.0),
          'shift z': (-0.5, 60.0), 'x offset': (-6.0, 6.0), 'button rotation': (-math.pi, math.pi),
          'tip angle offset': (-1.3, 1.3), 'tip shear': (-1.0, 1.0)}
OBJECTS = ('R2', 'R4', 'L2', 'L4', 'R2 bool', 'R4 bool', 'R2_pcb_button', 'R4_pcb_button', 'hand points', 'hand rig')

W_NORMAL = 30.0      # mm per unit normal error (1 deg ~ 0.5 mm)
W_OVERLAP = 6.0      # per mm of capsule overlap beyond Press
W_GAP = 4.0          # per mm of switch-gap change
W_HITS = 0.3         # per lever ray lost under the switch footprint
W_REG = 0.06         # smallest change: weight per SCALE unit
W_TRAVEL = 8.0       # per radian the travel direction strays further from the pressing direction than ...
TRAVEL_MAX_DEG = 25.0  # ... this (or than today, whichever is larger)


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------- press face
def press_face(me, M, tol=0.02):
    """The surface the fingertip presses. The generator flags two polygons 'top' on a trigger: the flat cut
    face at the finger end (the largest one) and a small step face of the stopper. The finger end is then
    extruded by 'top bevel' (1 mm) and bevelled, so the outermost surface is the polygon(s) 1 mm above the
    flagged cut face and parallel to it: that is the press face used here (centre = area-weighted centre,
    normal = area-weighted normal). Also returns the generator's own 'top' average ('top_*') for comparison."""
    P = np.empty(len(me.polygons) * 3)
    me.polygons.foreach_get('center', P); C = P.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    me.polygons.foreach_get('normal', P); N = P.reshape(-1, 3) @ M[:3, :3].T
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    A = np.empty(len(me.polygons)); me.polygons.foreach_get('area', A)
    top = me.attributes.get('top')
    sel = np.empty(len(me.polygons), bool); top.data.foreach_get('value', sel)
    V = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', V); V = V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    ls = np.empty(len(me.polygons), np.int64); me.polygons.foreach_get('loop_start', ls)
    lv = np.empty(len(me.loops), np.int64); me.loops.foreach_get('vertex_index', lv)
    j = np.nonzero(sel)[0][np.argmax(A[sel])]                 # flagged cut face of the finger end
    n0 = N[j]
    h = (V - C[j]) @ n0
    hmin_poly = np.minimum.reduceat(h[lv], ls)
    outer = (hmin_poly > h.max() - tol) & (N @ n0 > 0.95)
    if not outer.any():
        outer = np.zeros(len(A), bool); outer[j] = True
    n = (N[outer] * A[outer, None]).sum(0)
    nt = (N[sel] * A[sel, None]).sum(0)
    lt = np.empty(len(me.loop_triangles), np.int64); me.loop_triangles.foreach_get('polygon_index', lt)
    return dict(center=(C[outer] * A[outer, None]).sum(0) / A[outer].sum(), normal=n / np.linalg.norm(n),
                area=float(A[outer].sum()), proud_of_cut_face_mm=float(h.max()),
                top_center=(C[sel] * A[sel, None]).sum(0) / A[sel].sum(), top_normal=nt / np.linalg.norm(nt),
                press_tris=outer[lt], press_pts=V[np.unique(lv[np.isin(np.repeat(np.arange(len(A)), np.diff(np.r_[ls, len(lv)])), np.nonzero(outer)[0])])])


# ------------------------------------------------------------------------------ scene
class Session:
    """Open a .blend (agent copy or any GripV working copy), switch to GripV, evaluate in a scratch scene."""

    def __init__(self, blend):
        t = time.time()
        bpy.ops.wm.open_mainfile(filepath=blend)
        self.blend = blend
        self.open_s = round(time.time() - t, 2)
        gp = bpy.data.node_groups['global'].nodes['global parameters']
        gp.inputs['variant'].default_value = 'GripV'
        self.saved_scene = bpy.context.window.scene
        self.sc = bpy.data.scenes.new('ts eval')
        for n in OBJECTS:
            if n in bpy.data.objects:
                self.sc.collection.objects.link(bpy.data.objects[n])
        bpy.context.window.scene = self.sc
        self.orig = {p: self.get(p) for p in PARTS}

    # ---- parameters
    def node(self, part):
        return bpy.data.node_groups[PARTS[part]['group']].nodes[PARTS[part]['node']]

    def get(self, part, names=SHOWN):
        nd = self.node(part)
        return {k: float(nd.inputs[k].default_value) for k in names}

    def set(self, part, vals):
        nd = self.node(part)
        for k, v in vals.items():
            s = nd.inputs[k]
            if s.is_linked:
                raise RuntimeError(f'{part}: input {k} is linked, cannot set it')
            s.default_value = float(v) if not isinstance(v, bool) else v

    # ---- evaluation
    def depsgraph(self):
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        return dg

    def mesh(self, name, dg=None, press=False):
        dg = dg or self.depsgraph()
        ob = bpy.data.objects[name]
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        me.calc_loop_triangles()
        M = np.array(ob.matrix_world)
        V = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', V); V = V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        T = np.empty(len(me.loop_triangles) * 3, np.int64); me.loop_triangles.foreach_get('vertices', T); T = T.reshape(-1, 3)
        out = {'V': V, 'T': T}
        if press:
            out.update(press_face(me, M))
        ev.to_mesh_clear()
        return out

    def trigger(self, part, travel=True):
        """Press face of the trigger (+ the travel direction from the 'action' preview)."""
        r = self.mesh(part, press=True)
        if travel:
            nd = self.node(part)
            nd.inputs['action'].default_value = True
            try:
                rp = self.mesh(part, press=True)
            finally:
                nd.inputs['action'].default_value = False
            d = rp['center'] - r['center']
            r['travel'] = d / max(np.linalg.norm(d), 1e-9)
            r['travel_mm'] = float(np.linalg.norm(d))
        return r

    # ---- hand
    def hand(self):
        """Capsules {name: (a, b, r)} of the posed hand + the distal bones' frames from 'hand rig'."""
        dg = self.depsgraph()
        hp = bpy.data.objects['hand points']
        ev = hp.evaluated_get(dg)
        me = ev.to_mesh()
        P = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', P)
        M = np.array(hp.matrix_world)
        P = P.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        rad = np.empty(len(me.vertices)); me.attributes['radius'].data.foreach_get('value', rad)
        bid = np.empty(len(me.vertices), np.int32); me.attributes['bone_id'].data.foreach_get('value', bid)
        ev.to_mesh_clear()
        arm = bpy.data.objects['hand rig']
        names = [b.name for b in arm.data.bones]
        caps = GM.caps_from_points(P, rad, bid, names)
        frames = {}
        for b in arm.pose.bones:
            R = np.array((arm.matrix_world @ b.matrix).to_3x3().normalized())
            frames[b.name] = R
        return caps, frames

    def save(self, out):
        out = os.path.abspath(out)
        assert out not in PROTECTED, f'refusing to write {out}'
        bpy.context.window.scene = self.saved_scene
        sc = self.sc
        for ob in list(sc.collection.objects):
            sc.collection.objects.unlink(ob)
        bpy.data.scenes.remove(sc)
        bpy.ops.wm.save_as_mainfile(filepath=out)
        log('SAVED', out)


# ----------------------------------------------------------------------------- targets
def chain_palmar(caps, finger):
    """Bending-plane direction of the distal phalanx toward the palm side, from the capsule chain
    (fallback / cross-check of the rig's bone axis)."""
    u = lambda k: (caps[k][1] - caps[k][0]) / np.linalg.norm(caps[k][1] - caps[k][0])
    ud, um = u(finger + '_distal'), u(finger + '_middle')
    p = -(um - (um @ ud) * ud)
    return p / np.linalg.norm(p)


def targets(caps, frames, press=0.5, tip_pad=None):
    """Target press-face centre and normal for each trigger (see the module doc)."""
    out = {}
    for part, cfg in PARTS.items():
        f = cfg['finger']
        a, b, r = caps[f + '_distal']
        L = np.linalg.norm(b - a)
        u = (b - a) / L
        pad = (tip_pad or {}).get(part, cfg['tip_pad'])
        s = float(np.clip(L + r - pad, 0.0, L))          # distance of the pad point from the DIP joint, on the bone
        q = a + s * u
        pc = chain_palmar(caps, f)
        if f + '_distal' in frames:
            p = frames[f + '_distal'][:, 2].copy()
            p -= (p @ u) * u
            p /= np.linalg.norm(p)
            if p @ pc < 0:
                p = -p
            src = 'hand rig bone Z'
        else:
            p, src = pc, 'capsule chain'
        c = q + p * (r - press)
        out[part] = dict(finger=f, capsule=f + '_distal', pad_axis_point=q, press_dir=p, center=c, normal=-p, radius=r,
                         pad_from_tip_mm=pad, press_dir_source=src,
                         rig_vs_chain_deg=float(np.degrees(np.arccos(np.clip(p @ pc, -1, 1)))))
    return out


# ------------------------------------------------------------------------------ checks
def switch_gap(sess, part, lever=None, n=9):
    """Rays straight down from the switch bottom over its footprint: gap to the lever (mm)."""
    sw = sess.mesh(PARTS[part]['switch'])['V']
    lo, hi = sw.min(0), sw.max(0)
    lever = lever if lever is not None else sess.mesh(part)
    tl = BVHTree.FromPolygons(lever['V'].tolist(), lever['T'].tolist())
    g = []
    for x in np.linspace(lo[0] + 0.05, hi[0] - 0.05, n):
        for y in np.linspace(lo[1] + 0.05, hi[1] - 0.05, n):
            h = tl.ray_cast(Vector((x, y, lo[2] - 1e-3)), Vector((0, 0, -1)), 20.0)
            if h[0] is not None:
                g.append(lo[2] - h[0].z)
    g = np.array(g)
    return {'rays': n * n, 'hits': int(len(g)), 'min': float(g.min()) if len(g) else None,
            'median': float(np.median(g)) if len(g) else None, 'max': float(g.max()) if len(g) else None,
            'switch_bottom_z': float(lo[2])}


def overlaps(Vt, Tt, caps, names=None):
    """{capsule: overlap mm (>0 = sinks into the trigger part)} via winding-number sign."""
    g = GM.capsule_vs_solid(Vt, Tt, caps, names=names)
    return {k: -v[0] for k, v in g.items()}


def lever_path_blocked(sess, part, case_V, case_T, lever=None, n=9):
    """Case material between the switch and the lever (81 rays from the switch bottom down)."""
    sw = sess.mesh(PARTS[part]['switch'])['V']
    lo, hi = sw.min(0), sw.max(0)
    lever = lever if lever is not None else sess.mesh(part)
    tl = BVHTree.FromPolygons(lever['V'].tolist(), lever['T'].tolist())
    tc = BVHTree.FromPolygons(case_V.tolist(), case_T.tolist())
    hits = blocked = 0
    for x in np.linspace(lo[0] + 0.05, hi[0] - 0.05, n):
        for y in np.linspace(lo[1] + 0.05, hi[1] - 0.05, n):
            o, d = Vector((x, y, lo[2] - 1e-3)), Vector((0, 0, -1))
            hl, hc = tl.ray_cast(o, d, 20.0), tc.ray_cast(o, d, 20.0)
            if hl[0] is not None:
                hits += 1
                if hc[0] is not None and hc[0].z > hl[0].z + 1e-3:
                    blocked += 1
    return {'rays': n * n, 'lever_hits': hits, 'blocked': blocked}


def trigger_in_case(Vt, case_V, case_T, depth=0.05):
    """Trigger vertices inside case material deeper than `depth` (ray parity, 3 directions)."""
    Tq, _ = GM.orient(case_V, case_T)
    b = GM.bvh_of(case_V, Tq)
    deep, worst = 0, 0.0
    for p in Vt:
        d = b.find_nearest(Vector(p))[3]
        if d is not None and d > depth and GM.inside(b, p):
            deep += 1
            worst = max(worst, d)
    return {'verts': int(len(Vt)), 'inside_deeper_than_mm': depth, 'count': deep, 'max_depth_mm': round(worst, 3)}


def state(sess, part, tg, caps, press):
    """Everything the solver and the report need about one trigger in its current parameters."""
    tr = sess.trigger(part)
    ov = overlaps(tr['V'], tr['T'], caps)
    gap = switch_gap(sess, part, tr)
    t = tg[part]
    c, n = tr['center'], tr['normal']
    a, b, r = caps[t['capsule']]
    u = b - a
    q = a + np.clip((c - a) @ u / (u @ u), 0, 1) * u
    worst = max(ov, key=ov.get)
    return dict(center=c, normal=n, area=tr['area'], travel=tr['travel'], travel_mm=tr['travel_mm'],
                top_center=tr['top_center'], top_normal=tr['top_normal'],
                center_err_mm=float(np.linalg.norm(c - t['center'])), center_err=c - t['center'],
                normal_err_deg=float(np.degrees(np.arccos(np.clip(n @ t['normal'], -1, 1)))),
                pad_gap_mm=float(np.linalg.norm(c - q) - r),
                normal_vs_pad_dir_deg=float(np.degrees(np.arccos(np.clip(n @ ((q - c) / np.linalg.norm(q - c)), -1, 1)))),
                travel_vs_press_deg=float(np.degrees(np.arccos(np.clip(tr['travel'] @ t['press_dir'], -1, 1)))),
                travel_vs_normal_deg=float(np.degrees(np.arccos(np.clip(-tr['travel'] @ n, -1, 1)))),
                overlap=ov, worst=worst, max_overlap=float(ov[worst]), gap=gap, V=tr['V'], T=tr['T'])


# ------------------------------------------------------------------------------- solve
def arc_frame(sess, part, vals=None):
    """Axis and centre of the arc the trigger slides on (as node group 'trigger align' builds it):
    local arc centre (0,0,0), axis local X; offset (x offset, sqrt(R^2 - shift z^2) - button radius offset, shift z),
    turned about world Z by 'button rotation', moved to the switch location - (0, 0, 1.6 + button pressed height)."""
    v = vals or sess.get(part, ('trigger radius', 'shift z', 'button radius offset', 'x offset', 'button rotation',
                                'button pressed height'))
    th = v['button rotation']
    Rz = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    R, z = v['trigger radius'], v['shift z']
    loc = Rz @ np.array([v['x offset'], math.sqrt(max(R * R - z * z, 0.0)) - v['button radius offset'], z])
    S = np.array(bpy.data.objects[PARTS[part]['switch']].location)
    C = loc + S - np.array([0.0, 0.0, 1.6 + v['button pressed height']])
    return Rz @ np.array([1.0, 0.0, 0.0]), C


def travel_dir(axis, C, c, n):
    """Direction the press-face centre c moves when the trigger is pressed (into the case, against n)."""
    t = np.cross(axis, c - C)
    t /= np.linalg.norm(t)
    return -t if t @ n > 0 else t


def near_capsules(caps, V, margin=12.0):
    lo, hi = V.min(0) - margin, V.max(0) + margin
    out = []
    for k, (a, b, r) in caps.items():
        if k.startswith('palm'):
            continue
        P = np.array(GM.seg_pts(a, b, 1.0))
        if np.any(np.all((P > lo - r) & (P < hi + r), axis=1)):
            out.append(k)
    return out


def solve(sess, part, tg, caps, press=0.5, free=None, max_nfev=80, log_every=True):
    from scipy.optimize import least_squares
    free = list(free or FREE_DEFAULT)
    x0 = np.array([sess.get(part, free)[k] for k in free])
    s0 = state(sess, part, tg, caps, press)
    gap0, hits0 = s0['gap']['min'], s0['gap']['hits']
    trav0 = max(s0['travel_vs_press_deg'], TRAVEL_MAX_DEG)
    t = tg[part]
    lo = np.array([BOUNDS[k][0] for k in free])
    hi = np.array([BOUNDS[k][1] for k in free])
    if part != 'R2' and 'shift z' in free:          # only R2 has the measured shift z >= -0.5 limit
        lo[free.index('shift z')] = -30.0
    x0c = np.clip(x0, lo + 1e-9, hi - 1e-9)
    sc = np.array([SCALE[k] for k in free])
    near = near_capsules(caps, s0['V'])
    fixed = sess.get(part)
    fixed['button pressed height'] = float(sess.node(part).inputs['button pressed height'].default_value)
    cache = {}

    def resid(x):
        key = tuple(np.round(x, 9))
        if key in cache:
            return cache[key]
        vals = dict(zip(free, x))
        sess.set(part, vals)
        tr = sess.trigger(part, travel=False)
        ov = overlaps(tr['V'], tr['T'], caps, names=near)
        gp = switch_gap(sess, part, tr)
        ax, C = arc_frame(sess, part, {**fixed, **vals})
        tv = travel_dir(ax, C, tr['center'], tr['normal'])
        ang = math.degrees(math.acos(float(np.clip(tv @ t['press_dir'], -1, 1))))
        r = list(tr['center'] - t['center'])
        r += list(W_NORMAL * (tr['normal'] - t['normal']))
        r += [W_OVERLAP * max(0.0, ov[k] - press) for k in near]
        g = gp['min'] if gp['min'] is not None else gap0 + 3.0
        r += [W_GAP * (g - gap0), W_HITS * max(0, hits0 - gp['hits'])]
        r += [W_TRAVEL * math.radians(max(0.0, ang - trav0))]
        r += list(W_REG * (x - x0c) / sc)
        r = np.array(r)
        cache[key] = r
        return r

    def jac(x):
        r0 = resid(x)
        J = np.zeros((len(r0), len(x)))
        for j, k in enumerate(free):
            h = STEP[k] if x[j] + STEP[k] <= hi[j] else -STEP[k]
            xx = x.copy(); xx[j] += h
            J[:, j] = (resid(xx) - r0) / h
        return J

    t0 = time.time()
    res = least_squares(resid, x0c, jac=jac, bounds=(lo, hi), x_scale=sc, method='trf', max_nfev=max_nfev,
                        xtol=1e-4, ftol=1e-6, gtol=1e-6)
    sess.set(part, dict(zip(free, res.x)))
    s1 = state(sess, part, tg, caps, press)
    if log_every:
        log(f'SOLVE {part}: {res.nfev} steps, {len(cache)} evaluations, {time.time() - t0:.1f} s, cost {res.cost:.4f}, '
            f'status {res.status} ({res.message}); capsules checked {near}')
    return dict(free=free, x0=x0, x=res.x, before=s0, after=s1, nfev=int(res.nfev), evals=len(cache),
                seconds=round(time.time() - t0, 1), cost=float(res.cost), capsules=near)


# ------------------------------------------------------------------------------ report
def fmt_val(k, v):
    return f'{math.degrees(v):8.2f} deg' if k in ANGLES else f'{v:8.3f} mm '


def table(old, new, parts=PARTS):
    lines = [f"{'part':4s} {'parameter':22s} {'old':>12s} {'new':>12s} {'change':>12s}"]
    for p in parts:
        for k in SHOWN:
            o, n = old[p][k], new[p][k]
            d = n - o
            ch = (f'{math.degrees(d):+8.2f} deg' if k in ANGLES else f'{d:+8.3f} mm ') if abs(d) > 1e-6 else '   unchanged'
            lines.append(f'{p:4s} {k:22s} {fmt_val(k, o):>12s} {fmt_val(k, n):>12s} {ch:>12s}')
    return '\n'.join(lines)


def state_summary(s):
    return dict(press_face_center=np.round(s['center'], 2).tolist(), press_face_normal=np.round(s['normal'], 4).tolist(),
                press_face_area_mm2=round(s['area'], 1),
                generator_top_faces_center=np.round(s['top_center'], 2).tolist(),
                generator_top_faces_normal=np.round(s['top_normal'], 4).tolist(),
                center_to_target_mm=round(s['center_err_mm'], 3), center_minus_target=np.round(s['center_err'], 2).tolist(),
                normal_to_target_deg=round(s['normal_err_deg'], 2),
                fingertip_pad_gap_mm=round(s['pad_gap_mm'], 2), normal_vs_pad_direction_deg=round(s['normal_vs_pad_dir_deg'], 1),
                travel_vs_press_direction_deg=round(s['travel_vs_press_deg'], 1),
                travel_vs_face_normal_deg=round(s['travel_vs_normal_deg'], 1),
                max_hand_overlap_mm=round(s['max_overlap'], 2), worst_capsule=s['worst'],
                overlapping_capsules={k: round(v, 2) for k, v in s['overlap'].items() if v > 0},
                switch_gap_mm=dict(hits=s['gap']['hits'], rays=s['gap']['rays'],
                                   min=None if s['gap']['min'] is None else round(s['gap']['min'], 3),
                                   median=None if s['gap']['median'] is None else round(s['gap']['median'], 3),
                                   max=None if s['gap']['max'] is None else round(s['gap']['max'], 3)))


def case_checks(sess, parts, stage_dirs=None):
    """Evaluate case_bottom (with whatever stage the file has) and run the case-dependent checks."""
    ob = bpy.data.objects['case_bottom']
    base = bpy.data.objects['basemesh GripV']
    linked = []
    for o in (ob, base):
        if o.name not in sess.sc.collection.objects:
            sess.sc.collection.objects.link(o); linked.append(o)
    t = time.time()
    m = sess.mesh('case_bottom')
    out = {'case_bottom_eval_s': round(time.time() - t, 1)}
    for p in parts:
        tr = sess.mesh(p)
        out[p] = dict(lever_path=lever_path_blocked(sess, p, m['V'], m['T'], tr),
                      trigger_in_case=trigger_in_case(tr['V'], m['V'], m['T']))
    for o in linked:
        sess.sc.collection.objects.unlink(o)
    return out


# --------------------------------------------------------------------------------- main
def parse(argv):
    o = dict(inp=None, out=None, parts=['R2', 'R4'], press=0.5, use_x=False, report=None, apply=None, show=False,
             check_case=False, free=None, max_nfev=60)
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--in': o['inp'] = argv[i + 1]; i += 2
        elif a == '--out': o['out'] = argv[i + 1]; i += 2
        elif a == '--parts': o['parts'] = argv[i + 1].split(','); i += 2
        elif a == '--press': o['press'] = float(argv[i + 1]); i += 2
        elif a == '--use-x-offset': o['use_x'] = True; i += 1
        elif a == '--report': o['report'] = argv[i + 1]; i += 2
        elif a == '--apply': o['apply'] = argv[i + 1]; i += 2
        elif a == '--show': o['show'] = True; i += 1
        elif a == '--check-case': o['check_case'] = True; i += 1
        elif a == '--free': o['free'] = argv[i + 1].split(','); i += 2
        elif a == '--max-nfev': o['max_nfev'] = int(argv[i + 1]); i += 2
        else: raise SystemExit(f'unknown argument {a}\n{__doc__}')
    if not o['inp']:
        raise SystemExit(__doc__)
    return o


def main(argv):
    o = parse(argv)
    t_all = time.time()
    sess = Session(o['inp'])
    log(f"opened {o['inp']} in {sess.open_s} s")
    caps, frames = sess.hand()
    tg = targets(caps, frames, o['press'])
    rep = {'input': o['inp'], 'press_mm': o['press'], 'targets': {}, 'parts': {}}
    for p in o['parts']:
        t = tg[p]
        rep['targets'][p] = dict(finger=t['finger'], pad_from_tip_mm=t['pad_from_tip_mm'],
                                 pad_axis_point=np.round(t['pad_axis_point'], 2).tolist(),
                                 press_direction=np.round(t['press_dir'], 4).tolist(), press_dir_source=t['press_dir_source'],
                                 rig_vs_chain_deg=round(t['rig_vs_chain_deg'], 1),
                                 target_center=np.round(t['center'], 2).tolist(), target_normal=np.round(t['normal'], 4).tolist())
    old = {p: sess.get(p) for p in PARTS}
    if o['apply']:
        vals = json.load(open(o['apply']))
        vals = vals.get('values', vals)
        for p in o['parts']:
            if p in vals:
                sess.set(p, {k: v for k, v in vals[p].items() if k in SHOWN})
    for p in o['parts']:
        before = state(sess, p, tg, caps, o['press']) if not o['apply'] else None
        if not o['show'] and not o['apply']:
            free = o['free'] or (FREE_DEFAULT + (['x offset'] if o['use_x'] else []))
            res = solve(sess, p, tg, caps, o['press'], free, o['max_nfev'])
            after = res['after']
            rep['parts'][p] = dict(solve=dict(free=free, steps=res['nfev'], evaluations=res['evals'], seconds=res['seconds'],
                                              cost=round(res['cost'], 5)),
                                   before=state_summary(res['before']), after=state_summary(after))
        else:
            after = state(sess, p, tg, caps, o['press'])
            rep['parts'][p] = dict(current=state_summary(after))
            if before is not None:
                rep['parts'][p]['before'] = state_summary(before)
    new = {p: sess.get(p) for p in PARTS}
    rep['values_old'] = old
    rep['values'] = new
    log('\nTRIGGER PARAMETERS (GripV R2/R4 parameters > trigger parameters.001)\n' + table(old, new, o['parts']))
    for p in o['parts']:
        for k, s in rep['parts'][p].items():
            if k == 'solve':
                continue
            log(f"{p} {k:7s}: centre->target {s['center_to_target_mm']:.2f} mm, normal->target {s['normal_to_target_deg']:.1f} deg, "
                f"pad gap {s['fingertip_pad_gap_mm']:+.2f}, normal vs pad dir {s['normal_vs_pad_direction_deg']:.1f} deg, "
                f"travel vs press {s['travel_vs_press_direction_deg']:.1f} deg, max overlap {s['max_hand_overlap_mm']:.2f} "
                f"({s['worst_capsule']}), switch gap {s['switch_gap_mm']}")
    if o['check_case']:
        rep['case_checks'] = case_checks(sess, o['parts'])
        log('CASE CHECKS', json.dumps(rep['case_checks']))
    if o['report']:
        json.dump(rep, open(o['report'], 'w'), indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
        log('REPORT', o['report'])
    if o['out'] and not o['show']:
        sess.save(o['out'])
    log(f'total {time.time() - t_all:.1f} s')
    return rep


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
