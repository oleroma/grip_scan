"""make_targets.py: shared problem data for the smooth-shape prototypes.

    /tmp/claude-0/venv/bin/python /tmp/claude-0/sdf/smooth/make_targets.py [-- --nofit-run DIR --blend F]

Reads base.blend (hand points, hand rig, R2/R4 triggers and their cutters, evaluated in a scratch scene) and
the no-fit baseline run (baseline/nofit: skin, static cutouts, internal objects) and writes
    /tmp/claude-0/sdf/smooth/targets.npz   arrays (see targets.json 'arrays' for every key)
    /tmp/claude-0/sdf/smooth/targets.json  names, constants, zones, seats, frozen region, how each array was made
"""
import os
import sys
import json
import time
import numpy as np
import bpy
from mathutils import Vector

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'lib'))
sys.path.insert(0, os.path.join(ROOT, 'stage'))
import gripmetrics as GM  # noqa: E402
import trigger_solver as TS  # noqa: E402

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
opt = {'--nofit-run': os.path.join(ROOT, 'baseline', 'nofit'), '--blend': os.path.join(ROOT, 'base.blend')}
for i in range(0, len(args) - 1, 2):
    opt[args[i]] = args[i + 1]
NOFIT = opt['--nofit-run']
PRESS = 0.5
STEP = 0.5
t0 = time.time()

# ------------------------------------------------------------------------------ Blender side
bpy.ops.wm.open_mainfile(filepath=opt['--blend'])
gp = bpy.data.node_groups['global'].nodes['global parameters']
gp.inputs['variant'].default_value = 'GripV'
settings = {k: gp.inputs[k].default_value for k in ('variant', 'subdivision level', 'voxel resolution')}
sc = bpy.data.scenes.new('targets eval')
for n in ('hand points', 'hand rig', 'R2', 'R4', 'R2 bool', 'R4 bool', 'R2_pcb_button', 'R4_pcb_button'):
    sc.collection.objects.link(bpy.data.objects[n])
bpy.context.window.scene = sc
dg = bpy.context.evaluated_depsgraph_get()
dg.update()

hp = bpy.data.objects['hand points']
arm = bpy.data.objects['hand rig']
bone_names = [b.name for b in arm.data.bones]
ev = hp.evaluated_get(dg)
me = ev.to_mesh()
HP = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', HP)
Mw = np.array(hp.matrix_world)
HP = HP.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]
hp_rad = np.empty(len(me.vertices)); me.attributes['radius'].data.foreach_get('value', hp_rad)
hp_bid = np.empty(len(me.vertices), np.int32); me.attributes['bone_id'].data.foreach_get('value', hp_bid)
ev.to_mesh_clear()
# 'fill zones' weights live on the original mesh (vertex group)
vg = hp.vertex_groups['fill zones']
hp_fz = np.zeros(len(hp.data.vertices))
for v in hp.data.vertices:
    for g in v.groups:
        if g.group == vg.index:
            hp_fz[v.index] = g.weight
caps = GM.caps_from_points(HP, hp_rad, hp_bid, bone_names)
frames = {b.name: np.array((arm.matrix_world @ b.matrix).to_3x3().normalized()) for b in arm.pose.bones}


def mesh_of(name, press=False):
    ob = bpy.data.objects[name]
    e = ob.evaluated_get(dg)
    m = e.to_mesh()
    m.calc_loop_triangles()
    M = np.array(ob.matrix_world)
    V = np.empty(len(m.vertices) * 3); m.vertices.foreach_get('co', V); V = V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    T = np.empty(len(m.loop_triangles) * 3, np.int64); m.loop_triangles.foreach_get('vertices', T); T = T.reshape(-1, 3)
    pf = TS.press_face(m, M) if press else None
    e.to_mesh_clear()
    return V, T, pf


trig = {p: mesh_of(p, press=True) for p in ('R2', 'R4')}
cutters = {p: mesh_of(p + ' bool')[:2] for p in ('R2', 'R4')}
switches = {p: mesh_of(p + '_pcb_button')[:2] for p in ('R2', 'R4')}
tp = {}
for p, cfg in TS.PARTS.items():
    nd = bpy.data.node_groups[cfg['group']].nodes[cfg['node']]
    tp[p] = {k: float(nd.inputs[k].default_value) for k in TS.SHOWN}
tg = TS.targets(caps, frames, PRESS)

# ------------------------------------------------------------------------------ no-fit skin
Vn, Tn, _ = GM.load_mesh(os.path.join(NOFIT, 'stage.npz'))
Tn, _ = GM.orient(Vn, Tn)
bvh = GM.bvh_of(Vn, Tn)


def load_multi(path):
    if not os.path.exists(path):
        return {}
    z = np.load(path, allow_pickle=True)
    names = sorted({k.rsplit('|', 1)[0] for k in z.files})
    return {n: (np.asarray(z[n + '|V'], float), np.asarray(z[n + '|T'], np.int64)) for n in names}


statics = load_multi(os.path.join(NOFIT, 'cutouts_static.npz'))
internals = load_multi(os.path.join(NOFIT, 'internals.npz'))
nofit_rep = json.load(open(os.path.join(NOFIT, 'report.json'))) if os.path.exists(os.path.join(NOFIT, 'report.json')) else {}


# ------------------------------------------------------------------------------ palmar directions
def unit(x):
    return x / np.linalg.norm(x)


CHILD = {}
for f in ('index', 'middle', 'ring', 'pinky', 'thumb'):
    chain = [f + '_metacarpal', f + '_proximal', f + '_middle', f + '_distal'] if f != 'thumb' else ['thumb_metacarpal', 'thumb_proximal', 'thumb_distal']
    for a, b in zip(chain[:-1], chain[1:]):
        CHILD[a] = b
PARENT = {v: k for k, v in CHILD.items()}


def flexion(name):
    """Chain flexion direction of a bone, perpendicular to it (the child bends towards the palm; a distal
    bone: away from its parent), or None."""
    a, b, _ = caps[name]
    u = unit(b - a)
    if name in CHILD:
        c = caps[CHILD[name]]
        w = unit(c[1] - c[0])
        pc = w - (w @ u) * u
    elif name in PARENT:
        c = caps[PARENT[name]]
        w = unit(c[1] - c[0])
        pc = -(w - (w @ u) * u)
    else:
        return None
    return unit(pc) if np.linalg.norm(pc) > 1e-6 else None


def palmar(name):
    """Palmar direction of a bone, perpendicular to it: the rig bone axis (X or Z) closest to the flexion
    direction of the chain (the child bends towards the palm; for a distal bone, away from the parent)."""
    a, b, _ = caps[name]
    u = unit(b - a)
    if name in CHILD:
        c = caps[CHILD[name]]
        w = unit(c[1] - c[0])
        pc = w - (w @ u) * u
    elif name in PARENT:
        c = caps[PARENT[name]]
        w = unit(c[1] - c[0])
        pc = -(w - (w @ u) * u)
    else:
        return None, None, None
    if np.linalg.norm(pc) < 1e-6:
        return None, None, None
    pc = unit(pc)
    R = frames.get(name)
    best = None
    if R is not None:
        for ax in (0, 2):
            q = R[:, ax] - (R[:, ax] @ u) * u
            q = unit(q)
            if q @ pc < 0:
                q = -q
            ang = float(np.degrees(np.arccos(np.clip(q @ pc, -1, 1))))
            if best is None or ang < best[1]:
                best = (q, ang, 'XZ'[ax // 2])
    if best is None:
        return pc, None, 'chain'
    return best[0], best[1], f'rig bone {best[2]} axis (sign from the chain flexion)'


pal = {k: palmar(k) for k in caps}

# ------------------------------------------------------------------------------ zone samples
ZONE_KIND = {z: 'contact' for z in GM.CONTACT_ZONES}
ZONE_KIND.update({'R2 fingertip': 'trigger', 'R4 fingertip': 'trigger',
                  'index proximal': 'frozen (R1, top)', 'index middle': 'frozen (R1, top)', 'index distal': 'frozen (R1, top)'})
zl = GM.zones(caps)
zone_names = [z for z in GM.ZONE_ORDER if z in zl]
bone_of_seg = {}
for k, (a, b, r) in caps.items():
    bone_of_seg[(tuple(np.round(a, 4)), tuple(np.round(b, 4)))] = k


def seg_bone(a, b):
    for k, (ca, cb, r) in caps.items():
        u = cb - ca
        L2 = u @ u
        if L2 < 1e-9:
            continue
        for p in (a, b):
            t = (p - ca) @ u / L2
            if not (-1e-3 <= t <= 1 + 1e-3) or np.linalg.norm(ca + t * u - p) > 1e-3:
                break
        else:
            return k
    return None


Z = {k: [] for k in ('pts', 'r', 'zone', 'bone', 'w', 'dir_nearest', 'dir_palmar', 'gap_nofit', 'axis_inside', 'surf_nofit', 't')}
zone_info = []
for zi, zname in enumerate(zone_names):
    n0 = len(Z['pts'])
    for a, b, r in zl[zname]:
        bone = seg_bone(a, b)
        L = float(np.linalg.norm(b - a))
        P = GM.seg_pts(a, b, STEP)
        for j, p in enumerate(P):
            hit, _, _, d = bvh.find_nearest(Vector(p))
            ins = GM.inside(bvh, p)
            s = -d if ins else d
            dn = unit(np.array(hit) - p) * (-1.0 if ins else 1.0)
            pp = pal[bone][0] if bone and pal[bone][0] is not None else dn
            fl = flexion(bone) if bone else None
            Z.setdefault('dir_flexion', []).append(fl if fl is not None else pp)
            Z['pts'].append(p); Z['r'].append(r); Z['zone'].append(zi); Z['bone'].append(bone_names.index(bone) if bone else -1)
            Z['w'].append(L / len(P)); Z['dir_nearest'].append(dn); Z['dir_palmar'].append(pp)
            Z['gap_nofit'].append(s - r); Z['axis_inside'].append(ins); Z['surf_nofit'].append(np.array(hit)); Z['t'].append(j / max(len(P) - 1, 1))
    g = np.array(Z['gap_nofit'][n0:])
    zone_info.append({'id': zi, 'name': zname, 'kind': ZONE_KIND.get(zname, 'contact'), 'n_points': len(Z['pts']) - n0,
                      'segments': [{'bone': seg_bone(a, b), 'a': np.round(a, 3).tolist(), 'b': np.round(b, 3).tolist(), 'r': round(float(r), 3)} for a, b, r in zl[zname]],
                      'nofit_min_gap_mm': round(float(g.min()), 2), 'nofit_max_gap_mm': round(float(g.max()), 2),
                      'nofit_support_pct': round(100 * float((g <= GM.SUPPORT_MM).mean()), 1)})
for k in Z:
    Z[k] = np.array(Z[k])
Z['target_nearest'] = Z['pts'] + Z['dir_nearest'] * (Z['r'] - PRESS)[:, None]
Z['target_palmar'] = Z['pts'] + Z['dir_palmar'] * (Z['r'] - PRESS)[:, None]
Z['target_flexion'] = Z['pts'] + Z['dir_flexion'] * (Z['r'] - PRESS)[:, None]
Z['frozen'] = GM.frozen_mask(Z['surf_nofit'])


# ------------------------------------------------------------------------------ seats
plane_section = GM.plane_section


seats, seat_arrays = {}, {}
for p in ('R2', 'R4'):
    Vt, Tt, pf = trig[p]
    c, n = np.asarray(pf['center'], float), np.asarray(pf['normal'], float)
    s0 = c - GM.SEAT_SKIN * n
    Vc, Tc = cutters[p]
    sec = plane_section(Vc, Tc, s0, n)              # right-hand opening only (x > 0), near the press face
    fin = caps[tg[p]['capsule']]
    seats[p] = {'press_face_center': np.round(c, 4).tolist(), 'press_face_normal': np.round(n, 5).tolist(),
                'press_face_area_mm2': round(pf['area'], 2), 'press_face_source': 'trigger_solver.press_face(): outer cap of the finger end',
                'generator_top_center (1 mm lower, do not use)': np.round(pf['top_center'], 3).tolist(),
                'seat_plane_point': np.round(s0, 4).tolist(), 'skin_under_face_mm': GM.SEAT_SKIN,
                'seat_radius_mm': GM.SEAT_RADIUS, 'seat_fade_mm (suggested)': GM.SEAT_FADE,
                'opening_section_segments': int(len(sec)), 'cutter_wall_mm': 1.2,
                'finger': tg[p]['finger'], 'fingertip_capsule': tg[p]['capsule'],
                'pad_axis_point': np.round(tg[p]['pad_axis_point'], 3).tolist(), 'press_dir': np.round(tg[p]['press_dir'], 5).tolist(),
                'target_center': np.round(tg[p]['center'], 4).tolist(), 'center_to_target_mm': round(float(np.linalg.norm(c - tg[p]['center'])), 4),
                'normal_to_target_deg': round(float(np.degrees(np.arccos(np.clip(n @ tg[p]['normal'], -1, 1)))), 3),
                'trigger_parameters': tp[p],
                'nofit_skin_at_press_center_signed_mm': round(float(GM.sd(bvh, c)), 3)}
    seat_arrays.update({f'seat_{p}_center': c, f'seat_{p}_normal': n, f'seat_{p}_plane_point': s0, f'seat_{p}_opening_segments': sec,
                        f'seat_{p}_press_pts': np.asarray(pf['press_pts'], float), f'trigger_{p}_V': Vt, f'trigger_{p}_T': Tt,
                        f'cutter_{p}_V': Vc, f'cutter_{p}_T': Tc, f'switch_{p}_V': switches[p][0], f'switch_{p}_T': switches[p][1]})

# ------------------------------------------------------------------------------ keep-outs near the handle
near_lo, near_hi = np.array([GM.EDIT_MIN_X - 5, -120, -30]), np.array([120, 10, GM.PARTING_Z + 2])
keep_arrays, keep_info = {}, {}
for kind, d in (('static_cutout', statics), ('internal', internals)):
    for name, (Vk, Tk) in d.items():
        if len(Tk) == 0:
            continue
        if not np.any(np.all((Vk > near_lo - 15) & (Vk < near_hi + 15), axis=1)):
            continue
        key = f"{kind}_{name.replace(' ', '_').replace('.', '_')}"
        keep_arrays[key + '_V'], keep_arrays[key + '_T'] = Vk, Tk
        ic = nofit_rep.get('skin', {}).get('internal_clearance', {}).get(name, {}) if kind == 'internal' else {}
        cc = nofit_rep.get('skin', {}).get('stick_cutout_clearance', {}).get('static: ' + name, {}) if kind == 'static_cutout' else {}
        keep_info[key] = {'object': name, 'kind': kind, 'verts': int(len(Vk)),
                          'nofit_wall_to_skin_mm': ic.get('min_mm'), 'nofit_cutout_clearance_mm': cc.get('mm')}

# ------------------------------------------------------------------------------ write
caps_names = list(caps)
arrays = dict(
    zone_pts=Z['pts'], zone_r=Z['r'], zone_id=Z['zone'].astype(np.int32), zone_bone=Z['bone'].astype(np.int32), zone_w=Z['w'],
    zone_t=Z['t'], zone_dir_nearest=Z['dir_nearest'], zone_dir_palmar=Z['dir_palmar'],
    zone_target_nearest=Z['target_nearest'], zone_target_palmar=Z['target_palmar'],
    zone_dir_flexion=Z['dir_flexion'], zone_target_flexion=Z['target_flexion'],
    zone_gap_nofit=Z['gap_nofit'], zone_axis_inside_nofit=Z['axis_inside'], zone_surf_nofit=Z['surf_nofit'], zone_frozen=Z['frozen'],
    caps_a=np.array([caps[k][0] for k in caps_names]), caps_b=np.array([caps[k][1] for k in caps_names]),
    caps_r=np.array([caps[k][2] for k in caps_names]),
    caps_palmar=np.array([pal[k][0] if pal[k][0] is not None else np.zeros(3) for k in caps_names]),
    bone_frames=np.array([frames[k] for k in caps_names]),
    hand_points=HP, hand_points_radius=hp_rad, hand_points_bone_id=hp_bid, hand_points_fill_zones=hp_fz,
    nofit_V=Vn, nofit_T=Tn, nofit_frozen=GM.frozen_mask(Vn), nofit_handle=GM.handle_mask(Vn),
    **seat_arrays, **keep_arrays)
np.savez_compressed(os.path.join(ROOT, 'targets.npz'), **arrays)

desc = {
    'zone_pts': 'zone sample points ON THE CAPSULE AXIS, every 0.5 mm along each zone segment (N,3)',
    'zone_r': 'capsule radius at each point (N,)', 'zone_id': "zone index into 'zones' (N,)",
    'zone_bone': "bone index into 'caps_names' (N,)", 'zone_w': 'axis length each point stands for (mm)',
    'zone_t': 'position along its segment 0..1 (0 = segment start a)',
    'zone_dir_nearest': 'unit direction from the axis point to the nearest point of the NO-FIT skin (flipped if the axis is inside it)',
    'zone_dir_palmar': 'unit palmar direction of the bone (rig bone axis, perpendicular to the bone, sign from the chain flexion)',
    'zone_target_nearest': 'axis + dir_nearest * (r - Press): where the case surface should be for the rigid finger to sink Press = 0.5 mm',
    'zone_target_palmar': 'axis + dir_palmar * (r - Press): the same along the palmar direction',
    'zone_dir_flexion': 'unit flexion direction from the capsule chain only (child bone bends towards the palm), perpendicular to the bone; differs from dir_palmar by the bone roll (see palmar.rig_vs_chain_deg)',
    'zone_target_flexion': 'axis + dir_flexion * (r - Press)',
    'zone_gap_nofit': 'signed gap of each axis point to the no-fit skin minus r (< 0 = the rigid finger sinks in)',
    'zone_axis_inside_nofit': 'axis point inside the no-fit case', 'zone_surf_nofit': 'nearest no-fit skin point',
    'zone_frozen': 'that nearest skin point lies in the frozen region (the shape cannot move it)',
    'caps_a / caps_b / caps_r': "every hand capsule (axis a-b, radius r), order of 'caps_names': the no-dig set",
    'caps_palmar': 'palmar unit direction per capsule (zeros where undefined: palm heel)',
    'bone_frames': "world 3x3 rotation of each pose bone of 'hand rig' (columns X, Y = along the bone, Z), order of caps_names",
    'hand_points*': "the 'hand points' cloud as evaluated (positions, radius, bone_id, 'fill zones' weight)",
    'nofit_V / nofit_T': 'the no-fit baseline skin (base.blend evaluated: shape pass-through, thenar on)',
    'nofit_frozen': 'per no-fit vertex: in the frozen region', 'nofit_handle': 'per no-fit vertex: right handle x >= 25, z < 40',
    'seat_<P>_center / _normal': 'press face of trigger P (outer cap, area-weighted centre and normal, from the evaluated trigger in base.blend)',
    'seat_<P>_plane_point': 'a point of the seat plane = centre - 0.2 * normal (the skin around the opening lies in this plane)',
    'seat_<P>_opening_segments': "section of the right-hand cutter '<P> bool' with the seat plane (M,2,3): the opening outline",
    'seat_<P>_press_pts': 'vertices of the press-face polygons', 'trigger_<P>_V/T': 'evaluated trigger part',
    'cutter_<P>_V/T': "case opening cutter '<P> bool' (lever channel; both sides, Mirror)", 'switch_<P>_V/T': 'PCB switch (fixed)',
    'static_cutout_* / internal_*': 'keep-out meshes near the right handle (x >= 5, z < 57 within 15 mm), from the no-fit run',
}
info = {
    '_about': 'Shared problem data for the smooth-shape prototypes (see PROBLEM.md). Units mm, Blender scene coordinates. '
              'Made by make_targets.py from base.blend and the no-fit run ' + NOFIT,
    'made': time.strftime('%Y-%m-%d %H:%M'), 'seconds': round(time.time() - t0, 1), 'blend': opt['--blend'], 'settings': settings,
    'constants': {'press_mm': PRESS, 'penetration_accept_mm': list(GM.ACCEPT['zone_penetration_mm']), 'penetration_target_mm': PRESS,
                  'support_within_mm': GM.SUPPORT_MM, 'max_dig_mm': GM.ACCEPT['max_dig_editable_mm'],
                  'seat_skin_under_face_mm': GM.SEAT_SKIN, 'seat_radius_mm': GM.SEAT_RADIUS, 'cutter_wall_mm': 1.2,
                  'cutout_clearance_mm': 0.8, 'top_z': GM.TOP_Z, 'max_y': GM.MAX_Y, 'parting_z': GM.PARTING_Z,
                  'finger_concave_radius_mm': GM.FINGER_R, 'groove_radius_mm': GM.GROOVE_R, 'step_mm': STEP},
    'frozen_region': {'rule': 'frozen = |x| < 25 (Edit Min X: body, mirror seam) OR z > 38 (parting band 38-40 + case_top; the thenar module, '
                              'which runs after the shape, owns z > 40 behind the sticks) OR y > max_y - 0.5 = -4.658 (front band that '
                              'holds max Y = the R1/L1 position)',
                      'edit_min_x': GM.EDIT_MIN_X, 'frozen_z': GM.FROZEN_Z, 'front_band_mm': GM.FRONT_BAND, 'max_y': GM.MAX_Y,
                      'hard_limits': 'never y > -4.158, never z > 59.04, top plate exactly flat, field exactly symmetric for |x| < 1 (mirror seam), '
                                     'edits mirrored to the left handle (use Mirrored Position)',
                      'python': 'gripmetrics.frozen_mask(V)', 'nofit_verts_frozen': int(GM.frozen_mask(Vn).sum()), 'nofit_verts': int(len(Vn))},
    'zones': zone_info,
    'caps_names': caps_names,
    'palmar': {k: {'dir': None if v[0] is None else np.round(v[0], 4).tolist(), 'rig_vs_chain_deg': None if v[1] is None else round(v[1], 1), 'source': v[2]}
               for k, v in pal.items()},
    'seats': seats,
    'keepouts': keep_info,
    'hand': {'object_points': 'hand points', 'object_rig': 'hand rig', 'collection': 'hand fit',
             'points_attributes': {'radius': 'capsule radius at the point (float, point)', 'bone_id': "index into the 'hand rig' bones (int, point)",
                                   'seg_first / seg_last': 'first / last point of its bone (bool, point)'},
             'vertex_group': "'fill zones' (1 = metacarpals of middle/ring/pinky and pinky distal; ~0.4 = parts of the proximal / middle phalanges; 0 = index, thumb, palm heel, R2/R4 fingertips)",
             'bones': bone_names, 'points': int(len(HP)), 'skins': "'<bone> skin' meshes in 'hand fit' (display only)",
             'stage_input': "modifier 'hand fit' input 'Hand Points' (object 'hand points'); context socket 'Hand Points' in a module",
             'seat_empties': ['R2 seat', 'R4 seat', 'R1 seat'],
             'pose_vs_v27_max_mm': round(max(max(np.linalg.norm(caps[k][0] - v[0]), np.linalg.norm(caps[k][1] - v[1])) for k, v in GM.load_caps().items() if k in caps), 4)},
    'triggers_objects': {'parts': ['R2', 'R4', 'L2', 'L4'], 'cutters': ['R2 bool', 'R4 bool'], 'switches': ['R2_pcb_button', 'R4_pcb_button'],
                         'parameters': "node groups 'GripV R2 parameters' / 'GripV R4 parameters' > node 'trigger parameters.001' (solved, do not change)",
                         'stage_reads': "a module may read 'R2', 'R4', 'R2 bool', 'R4 bool' with Object Info (no cycle; mod_seats.py does)"},
    'arrays': {k: list(np.shape(v)) for k, v in arrays.items()},
    'arrays_meaning': desc,
}
json.dump(info, open(os.path.join(ROOT, 'targets.json'), 'w'), indent=1)
print('TARGETS', os.path.join(ROOT, 'targets.npz'), len(arrays), 'arrays,', len(Z['pts']), 'zone points,', round(time.time() - t0, 1), 's', flush=True)
for zinfo in zone_info:
    print(f"  {zinfo['id']:2d} {zinfo['name']:16s} {zinfo['kind']:18s} n {zinfo['n_points']:4d} no-fit gap {zinfo['nofit_min_gap_mm']:+6.2f} .. {zinfo['nofit_max_gap_mm']:+6.2f} support {zinfo['nofit_support_pct']}%")
for p, s in seats.items():
    print(' ', p, 'centre', s['press_face_center'], 'normal', s['press_face_normal'], 'opening segs', s['opening_section_segments'],
          'no-fit skin at centre', s['nofit_skin_at_press_center_signed_mm'])
print('  palmar rig-vs-chain deg', {k: v[1] if v[1] is None else round(v[1], 1) for k, v in pal.items()})
print('  keepouts', list(keep_info))
