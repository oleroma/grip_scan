"""eval_all.py: rebuild the hand-fit stage in a .blend copy, evaluate the case, measure, render.

    PY=/tmp/claude-0/venv/bin/python
    $PY /tmp/claude-0/sdf/smooth/stage/eval_all.py -- --blend MY_COPY.blend --tag NAME [options]

Options
  --blend F            the .blend to evaluate (your own copy of Grip_hand.blend; it is NOT saved unless --save)
  --tag NAME           run name; results go to --out-root/NAME (default root /tmp/claude-0/sdf/smooth/runs)
  --out-root DIR       where run folders go (module agents: use your own folder)
  --modules-dir DIR    look for mod_<name>.py here first (repeatable); default modules from build/stage
  --no-rebuild         do not rebuild the stage node groups (use what is in the file)
  --set 'Name=value'   set a 'hand fit' modifier input (repeatable), e.g. --set 'Carve Max Z=40'
  --subdiv LEVEL       global 'subdivision level' (low/medium/high/very high), default: as saved (high)
  --voxel LEVEL        global 'voxel resolution', default: as saved (low)
  --parts a,b,..       case parts to evaluate (default: case_top,case_bottom,R2,R4,L2,L4,R1,L1,case_middle_plate)
  --skin-only          evaluate only the stage output (fast loop: ~stage time + measure)
  --base-run DIR       plain-GripV run to compare with (default: cached <out-root>/base_<blend stem>_<subdiv>_<voxel>,
                       made with the stage switched off when missing; --refresh-base remakes it)
  --no-base            no comparison run
  --quick              measure without wrap angles / internal clearances
  --no-render          skip the picture sheet
  --save               save the rebuilt .blend back to --blend
Outputs in the run folder: stage.npz, parts/*.npz, parts.json, caps.json, cutouts_*.npz, internals.npz,
report.json (measure_case.py), sheet.jpg + view_*.png, log lines 'RESULT ...' on stdout.
"""
import os
import sys
import json
import time
import math
import hashlib
import bpy
import bmesh
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = '/tmp/claude-0/sdf/smooth/lib'
sys.path.insert(0, HERE)
sys.path.insert(0, LIB)
import build_stage as BS  # noqa: E402
import gripmetrics as GM  # noqa: E402
import trigger_solver as TS  # noqa: E402  (press_face: the outer cap of the finger end)

PARTS = ('case_top', 'case_bottom', 'R2', 'R4', 'L2', 'L4', 'R1', 'L1', 'case_middle_plate',
         # fix 2026-10-20: every other object Olex's GripV export presets write that reads 'basemesh variant'
         # (dependency walk: final/fix/work/inspect_before.json). measure_case checks each against plain GripV
         # (volume within 1 mm3, same Euler characteristic). The dpad-stick parts are empty in GripV (also in plain).
         'cable holder', 'capacitive', 'nut_lock', 'StickL', 'abxy button', 'select cap down right', 'abxy cap left',
         'home cap', 'abxy cap down', 'dpad cap right', 'abxy cap up', 'abxy cap right',
         'rocker ball', 'rocker', 'dpad stick cap', 'rocker support', 'ball support', 'stick')
DYN_CUTOUTS = ('Stick R cutout', 'StickL cutout', 'Stick R hole', 'StickL hole')
INTERNALS = ('pcb cutout', 'marmota cutout', 'corner relief', 'middle supports', 'side pillars', 'middle pillars',
             'jst2p bool', 'battery wire space', 'battery wire space.001', 'wire bool', 'wire bool 2', 'cutout.004',
             'screw bools', 'hollow cutout', 'battery bool', 'jst2p pass')
SEATS = ('R2 seat', 'R4 seat', 'R1 seat')
TRIGGERS = ('R2', 'R4')                          # also evaluated in --skin-only runs (they never read the basemesh)
CUTTERS = ('R2 bool', 'R4 bool', 'R2_pcb_button', 'R4_pcb_button')   # opening cutters + switches (cutters.npz)
NOFIT_RUN = '/tmp/claude-0/sdf/smooth/baseline/nofit'   # the no-fit baseline run (reference of the smooth metrics)
PROTECTED_BLENDS = {'/tmp/claude-0/sdf/smooth/base.blend', '/tmp/claude-0/sdf/build/Grip_hand.blend', '/home/claude/Grip/blender/Grip.blend'}
OUT_ROOT = '/tmp/claude-0/sdf/smooth/runs'


def log(*a):
    print(*a, flush=True)


def parse(argv):
    o = dict(blend=None, tag=None, out_root=OUT_ROOT, modules_dirs=[], rebuild=True, sets={}, subdiv=None, voxel=None,
             parts=list(PARTS), skin_only=False, base_run=None, no_base=False, refresh_base=False, quick=False,
             render=True, save=False, nofit_run=None)
    i = 0
    while i < len(argv):
        a = argv[i]
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if a == '--blend': o['blend'] = nxt; i += 2
        elif a == '--tag': o['tag'] = nxt; i += 2
        elif a == '--out-root': o['out_root'] = nxt; i += 2
        elif a == '--modules-dir': o['modules_dirs'].append(nxt); i += 2
        elif a == '--no-rebuild': o['rebuild'] = False; i += 1
        elif a == '--allow-view': o['allow_view'] = True; i += 1
        elif a == '--set':
            k, v = nxt.split('=', 1); o['sets'][k.strip()] = BS.parse_value(v); i += 2
        elif a == '--subdiv': o['subdiv'] = nxt; i += 2
        elif a == '--voxel': o['voxel'] = nxt; i += 2
        elif a == '--parts': o['parts'] = [p for p in nxt.split(',') if p]; i += 2
        elif a == '--skin-only': o['skin_only'] = True; i += 1
        elif a == '--base-run': o['base_run'] = nxt; i += 2
        elif a == '--no-base': o['no_base'] = True; i += 1
        elif a == '--refresh-base': o['refresh_base'] = True; i += 1
        elif a == '--quick': o['quick'] = True; i += 1
        elif a == '--no-render': o['render'] = False; i += 1
        elif a == '--save': o['save'] = True; i += 1
        elif a == '--nofit-run': o['nofit_run'] = nxt; i += 2
        else:
            raise SystemExit(f'unknown argument {a}')
    if not o['blend']:
        raise SystemExit('--blend is required')
    o['tag'] = o['tag'] or os.path.splitext(os.path.basename(o['blend']))[0]
    return o


# ---------------------------------------------------------------------------- evaluation
def mesh_arrays(ob, dg, attrs=(), bm_volume=False):
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    if me is None or len(me.vertices) == 0:
        ev.to_mesh_clear()
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), {}, None
    me.calc_loop_triangles()
    # topology on the real polygons/edges (triangulated n-gons of boolean results can fake non-manifold edges)
    le = np.empty(len(me.loops), np.int64); me.loops.foreach_get('edge_index', le)
    cnt = np.bincount(le, minlength=len(me.edges))
    topo = {'bl_boundary_edges': int((cnt == 1).sum()), 'bl_nonmanifold_edges': int((cnt > 2).sum()),
            'bl_wire_edges': int((cnt == 0).sum()), 'bl_faces': int(len(me.polygons)),
            'bl_verts': int(len(me.vertices)), 'bl_edges': int(len(me.edges))}
    topo['bl_closed_manifold'] = topo['bl_boundary_edges'] == 0 and topo['bl_nonmanifold_edges'] == 0 and topo['bl_wire_edges'] == 0
    # raw winding (fix 2026-10-20): signed volume of the loop triangles AS WOUND, in world space (x the sign of
    # the transform's determinant, so a mirrored object counts the way an exporter writes it). > 0 = faces point
    # outward. Everything else here orients first (GM.orient), which hid the inside-out stage output.
    if len(me.loop_triangles):
        _V = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', _V); _V = _V.reshape(-1, 3)
        _T = np.empty(len(me.loop_triangles) * 3, np.int64); me.loop_triangles.foreach_get('vertices', _T); _T = _T.reshape(-1, 3)
        _M = np.array(ob.matrix_world)
        _P = (_V @ _M[:3, :3].T + _M[:3, 3])[_T]
        _raw = float(np.einsum('ij,ij->i', _P[:, 0], np.cross(_P[:, 1], _P[:, 2])).sum() / 6.0) * float(np.sign(np.linalg.det(_M[:3, :3])))
        topo['raw_signed_volume_mm3'] = round(_raw, 2)
        topo['outward_winding'] = bool(_raw > 0)
    if bm_volume:
        # Blender's own volume (bmesh.calc_volume on the real n-gons, object space; parts have identity
        # transforms). Differs from the triangle volume by ~0.1 % where boolean n-gons are not planar.
        bm = bmesh.new(); bm.from_mesh(me)
        topo['bl_volume_mm3'] = round(float(bm.calc_volume(signed=False)), 1)
        bm.free()
    V = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', V); V = V.reshape(-1, 3)
    T = np.empty(len(me.loop_triangles) * 3, np.int64); me.loop_triangles.foreach_get('vertices', T); T = T.reshape(-1, 3)
    M = np.array(ob.matrix_world)
    V = V @ M[:3, :3].T + M[:3, 3]
    out = {}
    for a in attrs:
        at = me.attributes.get(a)
        if at is not None and at.domain == 'POINT' and at.data_type == 'FLOAT':
            x = np.empty(len(me.vertices)); at.data.foreach_get('value', x); out[a] = x
    for at in me.attributes:
        if at.name.startswith('hf_') and at.domain == 'POINT' and at.data_type == 'FLOAT':
            x = np.empty(len(me.vertices)); at.data.foreach_get('value', x); out[at.name] = x
    press = None
    top = me.attributes.get('top')
    if top is not None and top.domain == 'FACE' and len(me.polygons) and ob.name in ('R2', 'R4', 'L2', 'L4'):
        sel = np.empty(len(me.polygons), bool); top.data.foreach_get('value', sel)
        if sel.any():
            # smooth/ fix: the press face is the OUTER CAP of the finger end (trigger_solver.press_face), 1 mm
            # above the generator's 'top'-flagged faces (whose average the old code used, 1-2 mm off)
            pf = TS.press_face(me, M)
            press = {'center': np.round(pf['center'], 4).tolist(), 'normal': np.round(pf['normal'], 5).tolist(),
                     'area': round(float(pf['area']), 2), 'proud_of_cut_face_mm': round(float(pf['proud_of_cut_face_mm']), 3),
                     'generator_top_center': np.round(pf['top_center'], 4).tolist(),
                     'generator_top_normal': np.round(pf['top_normal'], 5).tolist(),
                     'source': 'trigger_solver.press_face (outer cap)'}
            out['_press_tris'] = np.asarray(pf['press_tris'], bool)
    out['_topo'] = topo
    ev.to_mesh_clear()
    return V, T, out, press


def save_multi(path, d):
    z = {}
    for n, (V, T) in d.items():
        z[n + '|V'] = V
        z[n + '|T'] = T
    np.savez_compressed(path, **z)


def global_settings():
    gp = bpy.data.node_groups['global'].nodes['global parameters']
    out = {}
    for s in gp.inputs:
        if hasattr(s, 'default_value'):
            v = s.default_value
            out[s.name] = v if isinstance(v, (int, float, str, bool)) else str(v)
    return out


def hand_caps(dg):
    hp = bpy.data.objects.get('hand points')
    arm = bpy.data.objects.get('hand rig')
    if hp is None or arm is None:
        return None
    ev = hp.evaluated_get(dg)
    me = ev.to_mesh()
    P = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', P); P = P.reshape(-1, 3)
    M = np.array(hp.matrix_world)
    P = P @ M[:3, :3].T + M[:3, 3]
    rad = np.empty(len(me.vertices)); me.attributes['radius'].data.foreach_get('value', rad)
    bid = np.empty(len(me.vertices), np.int32); me.attributes['bone_id'].data.foreach_get('value', bid)
    ev.to_mesh_clear()
    names = [b.name for b in arm.data.bones]
    return GM.caps_from_points(P, rad, bid, names)


def evaluate(o, run_dir, stage_on=True):
    """Evaluate the stage output and the case parts in a scratch scene (one depsgraph evaluation);
    export everything to run_dir."""
    os.makedirs(os.path.join(run_dir, 'parts'), exist_ok=True)
    mod = bpy.data.objects[BS.TARGET].modifiers[BS.MOD_NAME]
    BS.set_inputs(mod, {'Enable': stage_on})
    sc = bpy.data.scenes.new('hf eval ' + os.path.basename(run_dir))
    win_scene = bpy.context.window.scene
    bpy.context.window.scene = sc
    info = {'settings': global_settings(), 'stage_inputs': BS.get_inputs(mod), 'stage_on': stage_on, 'parts': {}, 'timing': {}}
    try:
        base_ob = bpy.data.objects[BS.TARGET]
        objs = [p for p in (TRIGGERS if o['skin_only'] else o['parts']) if p in bpy.data.objects]
        statics = []
        try:
            col = getattr(mod.properties.inputs, BS.input_ids(mod)['Static Cutouts']).value
            if col is not None:
                statics = [ob.name for ob in col.all_objects]
        except Exception:
            statics = [ob.name for ob in bpy.data.collections['cutout flat'].objects] if 'cutout flat' in bpy.data.collections else []
        cutters = [n for n in CUTTERS if n in bpy.data.objects]
        others = list(dict.fromkeys(statics + cutters)) if o['skin_only'] else list(dict.fromkeys(statics + cutters + [n for n in DYN_CUTOUTS + INTERNALS if n in bpy.data.objects]))
        for n in [BS.TARGET] + objs + others:
            if n not in sc.collection.objects:
                sc.collection.objects.link(bpy.data.objects[n])
        t = time.time()
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        info['timing']['eval_s'] = round(time.time() - t, 2)     # stage + all linked parts, one evaluation
        V, T, at, _ = mesh_arrays(base_ob, dg, attrs=('change_mm',))
        info['stage_topology'] = at.pop('_topo', None)
        ev_mod = base_ob.evaluated_get(dg).modifiers.get(BS.MOD_NAME)
        if ev_mod is not None and hasattr(ev_mod, 'execution_time'):
            info['timing']['stage_s'] = round(ev_mod.execution_time, 2)
        warns = []
        for mm in (ev_mod, mod):
            if mm is not None and hasattr(mm, 'node_warnings'):
                try:
                    warns += [f'{w.type}: {w.message}' for w in mm.node_warnings]
                except Exception:
                    pass
        info['stage_warnings'] = sorted(set(warns))
        chg = at.get('change_mm')
        extra = {k: v for k, v in at.items() if k.startswith('hf_')}
        GM.save_mesh(os.path.join(run_dir, 'stage.npz'), V, T, chg if chg is not None else np.zeros(len(V)), **extra)
        info['stage_mesh'] = {'verts': int(len(V)), 'tris': int(len(T))}
        caps = hand_caps(dg)
        if caps:
            GM.save_caps(os.path.join(run_dir, 'caps.json'), caps)
            ref = GM.load_caps()
            err = max(max(np.linalg.norm(caps[k][0] - ref[k][0]), np.linalg.norm(caps[k][1] - ref[k][1])) for k in ref if k in caps)
            info['hand_pose_vs_v27_max_mm'] = round(float(err), 4)
        info['seats'] = {n.split()[0]: list(map(float, bpy.data.objects[n].matrix_world.translation)) for n in SEATS if n in bpy.data.objects}
        press = {}
        for p in objs:
            Vp, Tp, pat, pf = mesh_arrays(bpy.data.objects[p], dg, bm_volume=True)
            ptris = pat.pop('_press_tris', None)
            GM.save_mesh(os.path.join(run_dir, 'parts', p + '.npz'), Vp, Tp, **({'press_tris': ptris} if ptris is not None else {}))
            r = {'verts': int(len(Vp))}
            r.update(pat.get('_topo') or {})
            if len(Tp):
                Tq, vol = GM.orient(Vp, Tp)
                topo = GM.topology(Vp, Tq)
                r.update(volume_mm3=round(vol, 1), islands=topo['islands'], genus=topo['genus'])
            if pf:
                press[p] = pf
            info['parts'][p] = r
        info['press_faces'] = press
        save_multi(os.path.join(run_dir, 'cutouts_static.npz'), {n: mesh_arrays(bpy.data.objects[n], dg)[:2] for n in statics})
        save_multi(os.path.join(run_dir, 'cutters.npz'), {n: mesh_arrays(bpy.data.objects[n], dg)[:2] for n in cutters})
        if not o['skin_only']:
            save_multi(os.path.join(run_dir, 'cutouts_dynamic.npz'), {n: mesh_arrays(bpy.data.objects[n], dg)[:2] for n in DYN_CUTOUTS if n in bpy.data.objects})
            save_multi(os.path.join(run_dir, 'internals.npz'), {n: mesh_arrays(bpy.data.objects[n], dg)[:2] for n in INTERNALS if n in bpy.data.objects})
    finally:
        bpy.context.window.scene = win_scene
        for ob in list(sc.collection.objects):
            sc.collection.objects.unlink(ob)
        bpy.data.scenes.remove(sc)
        BS.set_inputs(mod, {'Enable': True})
    json.dump(info, open(os.path.join(run_dir, 'parts.json'), 'w'), indent=1)
    return info


# ------------------------------------------------------------------------------- render
def _mat_change(name='hf change'):
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes['Principled BSDF']
    bsdf.inputs['Roughness'].default_value = 0.55
    at = nt.nodes.new('ShaderNodeAttribute'); at.attribute_name = 'chg'; at.attribute_type = 'GEOMETRY'
    sk = nt.nodes.new('ShaderNodeAttribute'); sk.attribute_name = 'skin'; sk.attribute_type = 'GEOMETRY'
    mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = -4.0; mr.inputs['From Max'].default_value = 4.0
    nt.links.new(at.outputs['Fac'], mr.inputs['Value'])
    cr = nt.nodes.new('ShaderNodeValToRGB')
    els = cr.color_ramp.elements
    els[0].position = 0.0; els[0].color = (0.10, 0.25, 0.95, 1)
    els[1].position = 1.0; els[1].color = (0.95, 0.18, 0.10, 1)
    for pos in (0.4875, 0.5, 0.5125):
        e = els.new(pos); e.color = (0.82, 0.82, 0.80, 1)
    nt.links.new(mr.outputs[0], cr.inputs['Fac'])
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'
    mix.inputs['A'].default_value = (0.35, 0.36, 0.38, 1)     # inner surfaces (not on the skin)
    nt.links.new(sk.outputs['Fac'], mix.inputs['Factor'])
    nt.links.new(cr.outputs['Color'], mix.inputs['B'])
    nt.links.new(mix.outputs['Result'], bsdf.inputs['Base Color'])
    return m


def _mat(name, rgba):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = rgba
    b.inputs['Roughness'].default_value = 0.5
    return m


def _mesh_obj(sc, name, V, T, attrs=None, mat=None, right_half=True):
    if right_half:
        keep = (V[T][:, :, 0] > -0.5).all(1)
        T = T[keep]
    used = np.unique(T)
    remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    V2, T2 = V[used], remap[T]
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V2)); me.vertices.foreach_set('co', V2.astype(np.float32).ravel())
    me.loops.add(len(T2) * 3); me.loops.foreach_set('vertex_index', T2.astype(np.int32).ravel())
    me.polygons.add(len(T2)); me.polygons.foreach_set('loop_start', np.arange(0, len(T2) * 3, 3, dtype=np.int32))
    me.update(calc_edges=True)
    me.polygons.foreach_set('use_smooth', np.ones(len(T2), bool))
    for k, v in (attrs or {}).items():
        a = me.attributes.new(k, 'FLOAT', 'POINT'); a.data.foreach_set('value', v[used].astype(np.float32))
    if mat:
        me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    sc.collection.objects.link(ob)
    return ob


def _capsule_mesh(caps, names):
    bm = bmesh.new()
    for k in names:
        a, b, r = caps[k]
        L = float(np.linalg.norm(b - a))
        geom = bmesh.ops.create_uvsphere(bm, u_segments=20, v_segments=12, radius=r)
        vs = [v for v in geom['verts']]
        u = Vector((b - a) / max(L, 1e-9))
        rot = u.to_track_quat('Z', 'Y').to_matrix()
        c = Vector((a + b) / 2)
        for v in vs:
            z = v.co.z
            v.co.z = z + (L / 2 if z > 1e-6 else (-L / 2 if z < -1e-6 else 0))
            v.co = rot @ v.co + c
    me = bpy.data.meshes.new('hand capsules')
    bm.to_mesh(me); bm.free()
    for p in me.polygons:
        p.use_smooth = True
    return me


def _mat_curv(name='hf curvature'):
    """Skin coloured by its smallest principal curvature (attribute 'cc' = max(-k2, 0), 1/mm) in bands of
    concave radius: grey > 20 mm, yellow 12-20, orange 8-12, red 5-8, magenta < 5; blue = convex radius
    < 3 mm (attribute 'cv' = max(k1, 0)); dark = not computed ('kc' = 0)."""
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes['Principled BSDF']
    bsdf.inputs['Roughness'].default_value = 0.6
    cc = nt.nodes.new('ShaderNodeAttribute'); cc.attribute_name = 'cc'; cc.attribute_type = 'GEOMETRY'
    cv = nt.nodes.new('ShaderNodeAttribute'); cv.attribute_name = 'cv'; cv.attribute_type = 'GEOMETRY'
    kc = nt.nodes.new('ShaderNodeAttribute'); kc.attribute_name = 'kc'; kc.attribute_type = 'GEOMETRY'
    cr = nt.nodes.new('ShaderNodeValToRGB')
    cr.color_ramp.interpolation = 'CONSTANT'
    els = cr.color_ramp.elements
    els[0].position = 0.0; els[0].color = (0.78, 0.78, 0.76, 1)
    els[1].position = 1.0; els[1].color = (0.75, 0.05, 0.75, 1)
    for pos, col in ((1 / 20, (0.95, 0.88, 0.25, 1)), (1 / 12, (0.98, 0.55, 0.08, 1)), (1 / 8, (0.88, 0.08, 0.06, 1)), (1 / 5, (0.75, 0.05, 0.75, 1))):
        e = els.new(pos); e.color = col
    nt.links.new(cc.outputs['Fac'], cr.inputs['Fac'])                 # ramp input 0..1 = curvature 0..1 /mm
    gt = nt.nodes.new('ShaderNodeMath'); gt.operation = 'GREATER_THAN'; gt.inputs[1].default_value = 1 / 3
    nt.links.new(cv.outputs['Fac'], gt.inputs[0])
    mb = nt.nodes.new('ShaderNodeMix'); mb.data_type = 'RGBA'
    mb.inputs['B'].default_value = (0.1, 0.3, 0.95, 1)
    nt.links.new(gt.outputs[0], mb.inputs['Factor']); nt.links.new(cr.outputs['Color'], mb.inputs['A'])
    mk = nt.nodes.new('ShaderNodeMix'); mk.data_type = 'RGBA'
    mk.inputs['A'].default_value = (0.3, 0.3, 0.32, 1)
    nt.links.new(kc.outputs['Fac'], mk.inputs['Factor']); nt.links.new(mb.outputs['Result'], mk.inputs['B'])
    nt.links.new(mk.outputs['Result'], bsdf.inputs['Base Color'])
    return m


def render_sheet(run_dir, tag, report=None, samples=16):
    """3 x 3 sheet: change colour (outside, front, inside), hand front, underside, curvature (inside),
    R2 close-up, R4 close-up, curvature (front-below)."""
    from scipy.spatial import cKDTree
    t0 = time.time()
    sc = bpy.data.scenes.new('hf render')
    bpy.context.window.scene = sc
    Vs, Ts, chg = GM.load_mesh(os.path.join(run_dir, 'stage.npz'))
    kd = cKDTree(Vs)
    mc = _mat_change()
    shown = []
    for p in ('case_bottom', 'case_top'):
        f = os.path.join(run_dir, 'parts', p + '.npz')
        if not os.path.exists(f):
            continue
        V, T, _ = GM.load_mesh(f)
        if len(T) == 0:
            continue
        d, i = kd.query(V)
        on = d < 0.3
        shown.append(_mesh_obj(sc, p, V, T, {'chg': np.where(on, chg[i], 0.0), 'skin': on.astype(float)}, mc))
    if not shown:   # skin only
        shown.append(_mesh_obj(sc, 'skin', Vs, Ts, {'chg': chg, 'skin': np.ones(len(Vs))}, mc))
    curv_obj = None
    cf = os.path.join(run_dir, 'curvature.npz')
    if os.path.exists(cf):
        z = np.load(cf)
        if len(z['k2']) == len(Vs):
            curv_obj = _mesh_obj(sc, 'skin curvature', Vs, Ts, {'cc': np.maximum(-z['k2'], 0.0), 'cv': np.maximum(z['k1'], 0.0),
                                                                 'kc': z['mask'].astype(float)}, _mat_curv())
    trig = {}
    for p, col in (('R2', (0.95, 0.55, 0.1, 1)), ('R4', (0.2, 0.75, 0.3, 1)), ('R1', (0.55, 0.6, 0.75, 1))):
        f = os.path.join(run_dir, 'parts', p + '.npz')
        if os.path.exists(f):
            V, T, _ = GM.load_mesh(f)
            if len(T):
                trig[p] = _mesh_obj(sc, p, V, T, None, _mat('hf ' + p, col), right_half=False)
    caps_f = os.path.join(run_dir, 'caps.json')
    caps = GM.load_caps(caps_f if os.path.exists(caps_f) else GM.DEFAULT_CAPS)
    hand = bpy.data.objects.new('hand', _capsule_mesh(caps, [k for k in caps if not k.startswith('palm')]))
    hand.data.materials.append(_mat('hf skin', (0.80, 0.56, 0.43, 1)))
    sc.collection.objects.link(hand)
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = samples
    sc.cycles.device = 'CPU'
    sc.cycles.use_denoising = True
    sc.render.threads_mode = 'FIXED'
    sc.render.threads = 2
    sc.render.resolution_x, sc.render.resolution_y = 640, 480
    sc.render.film_transparent = False
    w = bpy.data.worlds.new('hf world'); sc.world = w
    w.use_nodes = True
    w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.16, 0.16, 0.17, 1)
    w.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.8
    for i, (loc, e) in enumerate((((300, 200, 300), 3.0), ((-200, 300, 100), 1.5), ((200, -300, -150), 1.4))):
        ld = bpy.data.lights.new(f'hf sun{i}', 'SUN'); ld.energy = e
        lo = bpy.data.objects.new(f'hf sun{i}', ld); sc.collection.objects.link(lo)
        lo.rotation_euler = (Vector((60, -40, 25)) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    sc.view_settings.view_transform = 'Standard'
    cam_d = bpy.data.cameras.new('hf cam'); cam_d.type = 'ORTHO'
    cam = bpy.data.objects.new('hf cam', cam_d); sc.collection.objects.link(cam); sc.camera = cam
    centre = Vector((58, -45, 22))
    pj = json.load(open(os.path.join(run_dir, 'parts.json'))) if os.path.exists(os.path.join(run_dir, 'parts.json')) else {}
    pf = pj.get('press_faces', {})
    fallback = {'R2': (Vector((39.2, -25.1, 19.8)), Vector((-0.71, 0.67, -0.22))), 'R4': (Vector((41.5, -39.5, 4.5)), Vector((-0.74, 0.56, -0.38)))}
    tview = {}
    for p in ('R2', 'R4'):
        if p in pf:
            tview[p] = (Vector(pf[p]['center']), Vector(pf[p]['normal']).normalized())
        else:
            tview[p] = fallback[p]
    # (name, view direction (camera sits at centre + d), centre, ortho scale, show hand, curvature colouring)
    views = [('case outside', Vector((1, -0.15, 0.1)), centre, 150, False, False),
             ('case front', Vector((-0.25, 1, -0.35)), centre, 150, False, False),
             ('case inside', Vector((-1, 0.35, -0.25)), centre, 150, False, False),
             ('hand front', Vector((0.35, 1, -0.45)), centre, 175, True, False),
             ('case back (underside)', Vector((0.25, 0.35, -1)), centre, 150, False, False),
             ('curvature inside', Vector((-1, 0.35, -0.25)), Vector((62, -45, 15)), 115, False, True),
             ('R2 close-up', tview['R2'][1], tview['R2'][0], 45, False, False),
             ('R4 close-up', tview['R4'][1], tview['R4'][0], 45, False, False),
             ('curvature front-below', Vector((0.15, 1, -0.6)), Vector((62, -40, 12)), 115, False, True)]
    files = []
    for name, d, c, scale, show_hand, curv in views:
        if curv and curv_obj is None:
            continue
        hand.hide_render = not show_hand
        for ob in shown:
            ob.hide_render = curv
        if curv_obj is not None:
            curv_obj.hide_render = not curv
        cam_d.ortho_scale = scale
        cam.location = c + d.normalized() * 400
        cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
        f = os.path.join(run_dir, 'view_' + name.split(' (')[0].replace(' ', '_').replace('-', '') + '.png')
        sc.render.filepath = f
        bpy.ops.render.render(write_still=True)
        files.append((name, f))
    from PIL import Image, ImageDraw
    ims = [Image.open(f).convert('RGB') for _, f in files]
    W, H = ims[0].size
    head = 84
    rows = (len(ims) + 2) // 3
    S = Image.new('RGB', (W * 3, H * rows + head), (36, 36, 38))
    dr = ImageDraw.Draw(S)
    dr.text((10, 6), f'hand fit run: {tag}   change colour = skin vs plain GripV: blue receded, white unchanged (|d| < 0.1 mm), red grown, full colour at 4 mm. '
                     'Triggers: R2 orange, R4 green, R1 blue-grey.', fill=(235, 235, 235))
    dr.text((10, 22), 'curvature views: concave radius grey > 20 mm, yellow 12-20, orange 8-12, red 5-8, magenta < 5; blue = convex radius < 3 mm; dark = not measured',
            fill=(235, 235, 235))
    if report:
        dr.text((10, 40), report.get('summary', '')[:230], fill=(235, 235, 200))
        dr.text((10, 54), report.get('summary2', '')[:230], fill=(235, 235, 200))
        dr.text((10, 68), report.get('summary3', '')[:230], fill=(200, 235, 200))
    for k, ((name, _), im) in enumerate(zip(files, ims)):
        x, y = (k % 3) * W, head + (k // 3) * H
        S.paste(im, (x, y))
        dr.text((x + 8, y + 6), name, fill=(240, 240, 240))
    out = os.path.join(run_dir, 'sheet.jpg')
    S.save(out, quality=88)
    log('RENDER', out, round(time.time() - t0, 1), 's')
    return out


# --------------------------------------------------------------------------------- main
def main(argv):
    o = parse(argv)
    t_all = time.time()
    run_dir = os.path.join(o['out_root'], o['tag'])
    os.makedirs(run_dir, exist_ok=True)
    t = time.time()
    bpy.ops.wm.open_mainfile(filepath=o['blend'])
    t_open = round(time.time() - t, 2)
    # review round 2: 'View' != 0 replaces the basemesh every printed part is built from (View 1 / 2 broke the prints).
    # The rebuild below would silently reset it, so the file's own value is checked first.
    m0 = bpy.data.objects[BS.TARGET].modifiers.get(BS.MOD_NAME) if BS.TARGET in bpy.data.objects else None
    if m0 is not None and m0.node_group is not None:
        fview = BS.get_inputs(m0).get('View', 0)
        if fview not in (0, None) and not o.get('allow_view'):
            raise SystemExit(f"[eval_all] the file's 'hand fit' View is {fview}, not 0: every part would be built from an "
                             f"inspection mesh. Set View back to 0 (modifier 'hand fit' on 'basemesh GripV') and save, or pass --allow-view.")
    gp = bpy.data.node_groups['global'].nodes['global parameters']
    gp.inputs['variant'].default_value = 'GripV'
    if o['subdiv']:
        gp.inputs['subdivision level'].default_value = o['subdiv']
    if o['voxel']:
        gp.inputs['voxel resolution'].default_value = o['voxel']
    saved_scene = bpy.context.window.scene
    idle = bpy.data.scenes.new('hf eval idle')
    bpy.context.window.scene = idle          # never evaluate the whole GripV scene
    t = time.time()
    if o['rebuild']:
        mod, binfo = BS.build(o['modules_dirs'], keep_values=False, values=o['sets'], verbose=False)
    else:
        mod = bpy.data.objects[BS.TARGET].modifiers[BS.MOD_NAME]
        BS.set_inputs(mod, o['sets'])
        binfo = {'modules': 'as saved in the file'}
    t_build = round(time.time() - t, 2)
    log('BUILD', json.dumps(binfo), t_build, 's')
    if o['save']:
        assert os.path.abspath(o['blend']) not in PROTECTED_BLENDS, 'refusing to save over ' + o['blend'] + ' (work in a private copy)'
        bpy.context.window.scene = saved_scene      # keep the file's own active scene
        bpy.data.scenes.remove(idle)
        bpy.ops.wm.save_as_mainfile(filepath=o['blend'])
        idle = bpy.data.scenes.new('hf eval idle')
        bpy.context.window.scene = idle
        log('SAVED', o['blend'])
    info = evaluate(o, run_dir, stage_on=True)
    info['build'] = binfo
    info['timing']['build_s'] = t_build
    info['timing']['open_s'] = t_open
    json.dump(info, open(os.path.join(run_dir, 'parts.json'), 'w'), indent=1)
    log('EVAL', json.dumps(info['timing']), 'stage mesh', info['stage_mesh'], 'warnings', info.get('stage_warnings'),
        'hand pose vs v27 (mm):', info.get('hand_pose_vs_v27_max_mm'))
    base_run = None
    if not o['no_base']:
        s = info['settings']
        base_run = o['base_run'] or os.path.join(o['out_root'], 'base_{}_{}_{}'.format(
            os.path.splitext(os.path.basename(o['blend']))[0], s['subdivision level'].replace(' ', ''), s['voxel resolution'].replace(' ', '')))
        need = ['stage.npz'] if o['skin_only'] else [os.path.join('parts', p + '.npz') for p in o['parts'] if p in bpy.data.objects]
        stale = not all(os.path.exists(os.path.join(base_run, f)) for f in need)
        if not stale:     # a base run from before the raw-winding fields: remake it (fix 2026-10-20)
            try:
                bpj = json.load(open(os.path.join(base_run, 'parts.json')))
                stale = 'raw_signed_volume_mm3' not in (bpj.get('stage_topology') or {})
            except Exception:
                stale = True
        if o['refresh_base'] or stale:
            t = time.time()
            ob = dict(o); ob['skin_only'] = o['skin_only']
            evaluate(ob, base_run, stage_on=False)
            info['timing']['base_s'] = round(time.time() - t, 1)
            log('BASE RUN', base_run, info['timing']['base_s'], 's')
    import measure_case
    margs = ['--run', run_dir] + (['--base-run', base_run] if base_run else []) + (['--quick'] if o['quick'] else [])
    nofit = o.get('nofit_run') or NOFIT_RUN
    if os.path.abspath(nofit) != os.path.abspath(run_dir) and os.path.exists(os.path.join(nofit, 'stage.npz')):
        margs += ['--nofit-run', nofit]
    t = time.time()
    rep = measure_case.main(margs)
    info['timing']['measure_s'] = round(time.time() - t, 1)
    if base_run and 'base_s' not in info['timing']:
        info['timing']['base_s'] = 0.0
    sk = rep['skin']
    summary2 = []
    if 'case_top_summary' in rep:
        c = rep['case_top_summary']
        summary2.append(f"case_top vol {c['volume_mm3']:.0f} (base {c['base_volume_mm3']}; bmesh {c.get('volume_bmesh_mm3')} base {c.get('base_volume_bmesh_mm3')}) top verts {c['top_verts_at_max_z']} closed {c['closed']}")
    if 'R1L1_shift_mm' in rep:
        summary2.append(f"R1/L1 shift {rep['R1L1_shift_mm']} mm")
    if 'section_z40' in sk and 'max_dev_mm' in sk['section_z40']:
        summary2.append(f"Z40 section dev {sk['section_z40']['max_dev_mm']} mm")
    if 'wrap_deg' in sk:
        summary2.append('wrap max {} deg'.format(max(sk['wrap_deg'].values())))
    tr = sk.get('triggers', {})
    for p in ('R2', 'R4'):
        if p in tr:
            summary2.append(f"{p} hand overlap {tr[p]['max_hand_overlap_mm']} pad-face gap {tr[p].get('pad_to_press_face_center_gap_mm')}")
    rep['summary2'] = ' | '.join(summary2)
    json.dump(rep, open(os.path.join(run_dir, 'report.json'), 'w'), indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    log('RESULT', rep['summary'])
    log('RESULT2', rep['summary2'])
    if o['render']:
        try:
            t = time.time()
            render_sheet(run_dir, o['tag'], rep)
            info['timing']['render_s'] = round(time.time() - t, 1)
        except Exception as e:   # rendering must never hide the measurements
            import traceback
            traceback.print_exc()
            log('RENDER FAILED', e)
    info['timing']['total_s'] = round(time.time() - t_all, 1)
    json.dump(info, open(os.path.join(run_dir, 'parts.json'), 'w'), indent=1)
    log('DONE', run_dir, json.dumps(info['timing']))


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
