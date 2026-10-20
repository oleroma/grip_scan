"""export_check.py: evaluate Grip_smooth.blend at the export settings of Olex's presets 'Case standard' + 'Triggers'
(subdivision 'very high', voxel resolution 'high', tolerance 'tight', press tolerance 'tight', hollow top, body
'standard'), time it, check every exported part at triangle level (closed, manifold, consistently oriented,
islands, genus, degenerate / sliver triangles) and write binary STLs (mm) to final/export/.

    /tmp/claude-0/venv/bin/python /tmp/claude-0/sdf/smooth/final/export_check.py -- [--blend F] [--out DIR] [--fit X] [--no-stl] [--plain]

Every part is also checked for its RAW winding (signed volume of the triangles as wound, > 0 = faces point
outward, the way the STL is written); 'cable holder' (preset 'Parts') is included because the stage's winding
once broke its boolean. --fit X evaluates with the Fit control at X (e.g. 0) without saving.

The file itself is not changed (nothing is saved). It refuses to run when the hidden 'View' input of the 'hand fit'
modifier is not 0 (review round 2).
"""
import os, sys, json, time, struct
import numpy as np
import bpy
from mathutils.bvhtree import BVHTree
from mathutils import Vector
FINAL = '/tmp/claude-0/sdf/smooth/final'
sys.path.insert(0, FINAL + '/stage')
import build_stage as BS
import eval_all as EA
import gripmetrics as GM

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
blend = args[args.index('--blend') + 1] if '--blend' in args else FINAL + '/Grip_smooth.blend'
out = args[args.index('--out') + 1] if '--out' in args else FINAL + '/export'
PLAIN = '--plain' in args          # reference: the same check with the hand fit switched off (plain GripV)
NO_STL = '--no-stl' in args
PARTS = ('case_top', 'case_bottom', 'R2', 'R4', 'L2', 'L4', 'cable holder')
FIT = float(args[args.index('--fit') + 1]) if '--fit' in args else None
SETTINGS = {'subdivision level': 'very high', 'voxel resolution': 'high', 'tolerance': 'tight', 'press tolerance': 'tight',
            'hollow top': True, 'body type': 'standard', 'variant': 'GripV'}
os.makedirs(out, exist_ok=True)


def write_stl(path, V, T):
    P = V[T].astype(np.float32)
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    rec = np.zeros(len(T), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    rec['n'] = n; rec['v'] = P
    with open(path, 'wb') as f:
        f.write(b'GripV smooth hand fit, mm'.ljust(80, b' '))
        f.write(struct.pack('<I', len(T)))
        f.write(rec.tobytes())


def tri_check(V, T):
    r = {}
    e = np.sort(np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    r['tri_boundary_edges'] = int((cnt == 1).sum())
    r['tri_nonmanifold_edges'] = int((cnt > 2).sum())
    ue = np.unique(e, axis=0)
    r['_nm_mid'] = ((V[ue[cnt > 2][:, 0]] + V[ue[cnt > 2][:, 1]]) / 2) if (cnt > 2).any() else np.zeros((0, 3))
    # orientation: every interior directed edge must appear once in each direction
    d = np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]
    _, dc = np.unique(d, axis=0, return_counts=True)
    r['tri_inconsistent_directed_edges'] = int((dc > 1).sum())
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    L = np.stack([np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1), np.linalg.norm(a - b, axis=1)], 1)
    lmax = L.max(1)
    h = 2 * area / np.maximum(lmax, 1e-20)          # height on the longest edge
    def ang(x, y, z):
        return np.degrees(np.arccos(np.clip(((y - x) * (z - x)).sum(1) / np.maximum(np.linalg.norm(y - x, axis=1) * np.linalg.norm(z - x, axis=1), 1e-20), -1, 1)))
    mina = np.minimum(np.minimum(ang(a, b, c), ang(b, c, a)), ang(c, a, b))
    r['tris'] = int(len(T)); r['area_mm2'] = round(float(area.sum()), 1)
    r['degenerate_tris (area < 1e-6 mm2)'] = int((area < 1e-6).sum())
    r['sliver_tris (min angle < 1 deg)'] = int((mina < 1).sum())
    r['sliver_tris (min angle < 5 deg)'] = int((mina < 5).sum())
    r['needle_tris (height < 0.001 mm)'] = int((h < 1e-3).sum())
    r['sliver_area_share_pct (min angle < 5 deg)'] = round(100 * float(area[mina < 5].sum() / area.sum()), 4)
    r['min_angle_deg'] = round(float(mina.min()), 4); r['min_angle_p1_deg'] = round(float(np.percentile(mina, 1)), 2)
    r['shortest_edge_mm'] = round(float(L.min()), 5); r['longest_edge_mm'] = round(float(L.max()), 3)
    raw = GM.volume(V, T)            # as wound, before orienting (world space; parts here have det > 0)
    Tq, vol = GM.orient(V, T)
    topo = GM.topology(V, Tq)
    r.update(volume_mm3=round(float(vol), 1), islands=topo['islands'], genus=topo['genus'],
             raw_signed_volume_mm3=round(float(raw), 1), outward_winding=bool(raw > 0))
    r['tri_closed_manifold'] = r['tri_boundary_edges'] == 0 and r['tri_nonmanifold_edges'] == 0
    return r


t0 = time.time()
bpy.ops.wm.open_mainfile(filepath=blend)
t_open = time.time() - t0
gp = bpy.data.node_groups['global'].nodes['global parameters']
before = {k: gp.inputs[k].default_value for k in SETTINGS}
for k, v in SETTINGS.items():
    gp.inputs[k].default_value = v
mod = bpy.data.objects[BS.TARGET].modifiers[BS.MOD_NAME]
_view = BS.get_inputs(mod).get('View', 0)
if _view not in (0, None):     # review round 2: View 1 / 2 replace the basemesh every exported part is built from
    raise SystemExit(f"[export_check] REFUSED: the 'hand fit' modifier's View is {_view}, not 0. Every exported part would be built "
                     f"from an inspection mesh (View 1: plain remesh, View 2: coarse pass, case_top collapses). Set View back to 0.")
BS.set_inputs(mod, {'Enable': not PLAIN})
if FIT is not None:
    bpy.data.node_groups[BS.CONTROLS_GROUP].nodes[BS.VALUES_NODE].inputs['Fit'].default_value = FIT
sc = bpy.data.scenes.new('hf export'); win = bpy.context.window.scene; bpy.context.window.scene = sc
statics = [ob.name for ob in bpy.data.collections['cutout flat'].objects] if 'cutout flat' in bpy.data.collections else []
names = [BS.TARGET] + list(PARTS) + statics + [n for n in EA.CUTTERS + EA.DYN_CUTOUTS + EA.INTERNALS if n in bpy.data.objects]
for n in dict.fromkeys(names):
    sc.collection.objects.link(bpy.data.objects[n])
t = time.time()
dg = bpy.context.evaluated_depsgraph_get(); dg.update()
t_eval = time.time() - t
rep = {'blend': blend, 'hand_fit': not PLAIN, 'fit': FIT, 'settings_before': {k: (v if isinstance(v, (bool, int, float, str)) else str(v)) for k, v in before.items()},
       'settings_export': SETTINGS, 'timing': {'open_s': round(t_open, 1), 'evaluate_all_parts_s': round(t_eval, 1)}, 'parts': {}}
ev_mod = bpy.data.objects[BS.TARGET].evaluated_get(dg).modifiers.get(BS.MOD_NAME)
if ev_mod is not None and hasattr(ev_mod, 'execution_time'):
    rep['timing']['hand_fit_stage_s'] = round(ev_mod.execution_time, 2)
Vs, Ts, _, _ = EA.mesh_arrays(bpy.data.objects[BS.TARGET], dg)
rep['stage_skin'] = tri_check(Vs, Ts)
rep['stage_skin'].pop('_nm_mid', None)
# the export skin vs the design skin (final run at subdivision 'high'): nearest-surface distance
try:
    # design-level reference run of the installed shape (v2, 2026-10-20; was runs/final = the v1 shape)
    d = np.load(args[args.index('--design-run') + 1] + '/stage.npz' if '--design-run' in args else FINAL + '/runs/v2_final/stage.npz')
    bvh = BVHTree.FromPolygons([Vector(v) for v in d['V']], d['T'].tolist())
    idx = np.random.default_rng(0).choice(len(Vs), min(40000, len(Vs)), replace=False)
    dist = np.array([bvh.find_nearest(Vector(Vs[i]))[3] for i in idx])
    rep['stage_skin_vs_design_high_mm'] = {'p50': round(float(np.median(dist)), 4), 'p99': round(float(np.percentile(dist, 99)), 4),
                                           'max': round(float(dist.max()), 4), 'at': Vs[idx[np.argmax(dist)]].round(1).tolist()}
    rep['stage_skin_max_z'] = round(float(Vs[:, 2].max()), 5); rep['stage_skin_max_y'] = round(float(Vs[:, 1].max()), 4)
except Exception as ex:
    rep['stage_skin_vs_design_high_mm'] = f'not measured: {ex}'
CUT_BVH = {}
for n in ('R2 bool', 'R4 bool'):
    if n in bpy.data.objects:
        Vc, Tc, _, _ = EA.mesh_arrays(bpy.data.objects[n], dg)
        if len(Tc):
            CUT_BVH[n] = BVHTree.FromPolygons(Vc.tolist(), Tc.tolist(), all_triangles=True)
for p in PARTS:
    t = time.time()
    V, T, at, _ = EA.mesh_arrays(bpy.data.objects[p], dg)
    r = dict(at.get('_topo') or {})
    if not len(T):
        rep['parts'][p] = {'empty': True}
        print(f'PART {p}: EMPTY', flush=True)
        continue
    r.update(tri_check(V, T))
    if 'raw_signed_volume_mm3' in (at.get('_topo') or {}):     # det-corrected value from eval_all.mesh_arrays
        r['raw_signed_volume_mm3'] = at['_topo']['raw_signed_volume_mm3']; r['outward_winding'] = bool(r['raw_signed_volume_mm3'] > 0)
    path = os.path.join(out, f'{p}.stl')
    if not NO_STL:
        write_stl(path, V, T)
        r['stl'] = path; r['stl_bytes'] = os.path.getsize(path); r['seconds'] = round(time.time() - t, 1)
    nm = r.pop('_nm_mid')
    if len(nm):      # where the triangle-level non-manifold edges sit: on an R2 / R4 cutter rim or elsewhere
        dc = np.array([min(b.find_nearest(Vector(q))[3] for b in CUT_BVH.values()) for q in nm]) if CUT_BVH else np.full(len(nm), np.inf)
        r['tri_nonmanifold_on_trigger_cutter_rims (<= 0.05 mm from R2/R4 bool)'] = int((dc <= 0.05).sum())
        r['tri_nonmanifold_edge_midpoints'] = [[*map(float, q.round(2)), round(float(d), 3)] for q, d in zip(nm[:80], dc[:80])]
    rep['parts'][p] = r
    print(f"PART {p}: {r['tris']} tris, closed manifold (tri) {r['tri_closed_manifold']} (Blender polys: {r.get('bl_closed_manifold')}, "
          f"non-manifold {r.get('bl_nonmanifold_edges')}), islands {r['islands']}, genus {r['genus']}, slivers<1deg {r['sliver_tris (min angle < 1 deg)']}, "
          f"degenerate {r['degenerate_tris (area < 1e-6 mm2)']}, min angle {r['min_angle_deg']}, vol {r['volume_mm3']}, "
          f"raw signed vol {r['raw_signed_volume_mm3']} ({'outward' if r['outward_winding'] else 'INSIDE-OUT'}), "
          f"tri non-manifold {r['tri_nonmanifold_edges']} (on R2/R4 cutter rims {r.get('tri_nonmanifold_on_trigger_cutter_rims (<= 0.05 mm from R2/R4 bool)', 0)})", flush=True)
inv = [p for p, r in rep['parts'].items() if not r.get('empty') and not r['outward_winding']]
if not rep['stage_skin']['outward_winding']:
    inv = ['stage skin'] + inv
rep['winding_check'] = {'pass': not inv, 'inside_out': inv, 'rule': 'raw signed volume > 0 for the stage skin and every part (as wound)'}
print('WINDING', 'ok: every part outward' if not inv else 'FAIL, inside-out: ' + ', '.join(inv), flush=True)
bpy.context.window.scene = win
rep['timing']['total_s'] = round(time.time() - t0, 1)
json.dump(rep, open(os.path.join(out, 'export_report.json'), 'w'), indent=1)
print('TIMING', rep['timing']); print('SKIN', rep['stage_skin']); print('SKIN vs design', rep.get('stage_skin_vs_design_high_mm'))
