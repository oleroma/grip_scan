"""measure_case.py: measure one hand-fit result and write a JSON report.

Usage (Blender's python, because mathutils comes with the bpy wheel):
  PY=/tmp/claude-0/venv/bin/python
  # a run folder written by stage/eval_all.py (skin + case parts + cutouts + internals):
  $PY measure_case.py -- --run /tmp/claude-0/sdf/build/out/<tag> [--base-run <dir>] [--quick]
  # just a skin mesh (npz with V,T or verts,tris):
  $PY measure_case.py -- --stage skin.npz [--base base.npz] [--caps caps.json] [--out report.json] [--quick]
  # a mesh object inside a .blend (evaluated, world space):
  $PY measure_case.py -- --blend file.blend --object "basemesh GripV" [--variant GripV] [--out r.json]

Run folder layout (what eval_all.py writes and this script reads):
  stage.npz            evaluated 'basemesh GripV' (= outer skin of the case) V, T [, chg]
  caps.json            hand capsules used for the run (from 'hand points' in the .blend)
  parts/<name>.npz     evaluated case parts (case_top, case_bottom, R2, R4, L2, L4, R1, L1, case_middle_plate)
  parts.json           Blender-side facts: settings, timings, press faces of R2/R4, seat empties
  cutouts_static.npz   'Stick R cutout flat', 'StickL cutout flat', 'Stick R cutout appendix'  (<name>|V, <name>|T)
  cutouts_dynamic.npz  'Stick R cutout', 'StickL cutout', 'Stick R hole', 'StickL hole' as evaluated
  internals.npz        internal static objects (pcb cutout, R4 wire, battery wire space, screws, supports, ...)
--base-run: a run of the same file with the stage switched off (the plain GripV at the same settings);
gives change maps, Z=40 section change, wall loss to internal objects, R1/L1 shift, case_top volume change.
"""
import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gripmetrics as GM  # noqa: E402

PARTS = ('case_top', 'case_bottom', 'R2', 'R4', 'L2', 'L4', 'R1', 'L1', 'case_middle_plate')


def load_multi(path):
    if not path or not os.path.exists(path):
        return {}
    z = np.load(path, allow_pickle=True)
    names = sorted({k.rsplit('|', 1)[0] for k in z.files})
    return {n: (np.asarray(z[n + '|V'], float), np.asarray(z[n + '|T'], np.int64)) for n in names}


def part_report(run, base_run=None):
    out = {}
    pj = json.load(open(os.path.join(run, 'parts.json'))) if os.path.exists(os.path.join(run, 'parts.json')) else {}
    bj = json.load(open(os.path.join(base_run, 'parts.json'))) if base_run and os.path.exists(os.path.join(base_run, 'parts.json')) else {}
    for p in dict.fromkeys(list(PARTS) + list(pj.get('parts', {}))):    # every part the run evaluated (fix 2026-10-20)
        f = os.path.join(run, 'parts', p + '.npz')
        if not os.path.exists(f):
            continue
        V, T, _ = GM.load_mesh(f)
        fb0 = os.path.join(base_run, 'parts', p + '.npz') if base_run else None
        if len(T) == 0:
            out[p] = {'empty': True}
            if fb0 and os.path.exists(fb0):
                out[p]['base_empty'] = len(GM.load_mesh(fb0)[1]) == 0
            continue
        e0 = pj.get('parts', {}).get(p, {})
        # raw winding, BEFORE orienting (eval_all stores the det-corrected value; the npz T is as wound)
        raw = e0.get('raw_signed_volume_mm3', round(GM.volume(V, T), 2))
        T, vol = GM.orient(V, T)
        r = {'volume_mm3': round(vol, 1), 'bbox': [V.min(0).round(3).tolist(), V.max(0).round(3).tolist()],
             'raw_signed_volume_mm3': raw, 'outward_winding': bool(raw > 0)}
        r.update({k: v for k, v in GM.topology(V, T).items() if k in ('verts', 'closed_manifold', 'boundary_edges', 'nonmanifold_edges', 'islands', 'genus')})
        if p == 'case_top':
            r['top_face'] = GM.top_face(V, T)
        if p in pj.get('parts', {}):
            e = pj['parts'][p]
            # Blender's own edge topology (real polygons) is the reference for closed / manifold
            for k in ('bl_closed_manifold', 'bl_boundary_edges', 'bl_nonmanifold_edges'):
                if k in e:
                    r[k[3:]] = e[k]
            # genus from the real polygons: a triangulation diagonal of an n-gon can coincide with an existing
            # edge, which merges two triangle edges and makes the triangle Euler characteristic off by one
            if all(k in e for k in ('bl_verts', 'bl_edges', 'bl_faces')) and r.get('islands'):
                chi = e['bl_verts'] - e['bl_edges'] + e['bl_faces']
                r['euler_char'] = int(chi)
                r['genus'] = (2 * r['islands'] - chi) / 2
            elif r.get('genus') is not None and r['genus'] != int(r['genus']):
                r['genus_note'] = 'triangle genus (n-gon diagonals merge edges); not exact for this part'
            if 'bl_volume_mm3' in e:
                r['volume_bmesh_mm3'] = e['bl_volume_mm3']
                if 'bl_volume_mm3' in bj.get('parts', {}).get(p, {}):
                    r['base_volume_bmesh_mm3'] = bj['parts'][p]['bl_volume_mm3']
            be = bj.get('parts', {}).get(p, {})
            if all(k in be for k in ('bl_verts', 'bl_edges', 'bl_faces')):
                r['base_euler_char'] = int(be['bl_verts'] - be['bl_edges'] + be['bl_faces'])
            if 'raw_signed_volume_mm3' in be:
                r['base_raw_signed_volume_mm3'] = be['raw_signed_volume_mm3']
            if p in bj.get('parts', {}) and 'bl_nonmanifold_edges' in bj['parts'][p]:
                r['base_nonmanifold_edges'] = bj['parts'][p]['bl_nonmanifold_edges']
                r['base_closed_manifold'] = bj['parts'][p]['bl_closed_manifold']
        fb = fb0
        if fb and os.path.exists(fb):
            Vb, Tb, _ = GM.load_mesh(fb)
            r['base_empty'] = len(Tb) == 0
            if len(Tb):
                Tb, volb = GM.orient(Vb, Tb)
                r['base_volume_mm3'] = round(volb, 1)
                r['volume_change_mm3'] = round(vol - volb, 1)
                bmin, bmax = Vb.min(0), Vb.max(0)
                r['bbox_shift_max_mm'] = round(float(max(np.abs(V.min(0) - bmin).max(), np.abs(V.max(0) - bmax).max())), 3)
                if p in ('R1', 'L1', 'R2', 'R4', 'L2', 'L4'):
                    same = len(Vb) == len(V)
                    r['vertex_shift_max_mm'] = round(float(np.linalg.norm(V - Vb, axis=1).max()), 3) if same else None
                if p == 'case_top':
                    r['base_top_face'] = GM.top_face(Vb, Tb)
        out[p] = r
    return out, pj, bj


def _plain_reference(base_run):
    """Smoothness of plain GripV (the Enable-off base run's skin) on the right handle below Z 40, cached
    in <base_run>/smooth_reference.json."""
    if not base_run or not os.path.exists(os.path.join(base_run, 'stage.npz')):
        return None
    f = os.path.join(base_run, 'smooth_reference.json')
    if os.path.exists(f):
        return json.load(open(f))
    Vb, Tb, _ = GM.load_mesh(os.path.join(base_run, 'stage.npz'))
    Tb, _ = GM.orient(Vb, Tb)
    k1, k2, kink, cm = GM.curvature_fields(Vb, Tb, GM.handle_mask(Vb))
    r = {'plain GripV handle_below_z40': GM.smooth_stats(Vb, GM.vert_area(Vb, Tb), k1, k2, kink, cm), 'mesh': os.path.join(base_run, 'stage.npz')}
    json.dump(r, open(f, 'w'), indent=1)
    return r


def smooth_part(a, rep, V, T, caps, press, base):
    """smooth/ extension: smoothness, contact support, max dig, seats, change vs the no-fit baseline,
    invariants vs it, acceptance."""
    nofit = nofit_curv = nofit_rep = None
    if a.nofit_run and (not a.run or os.path.abspath(a.nofit_run) != os.path.abspath(a.run)):
        Vn, Tn, _ = GM.load_mesh(os.path.join(a.nofit_run, 'stage.npz'))
        nofit = (Vn, Tn)
        cf = os.path.join(a.nofit_run, 'curvature.npz')
        if os.path.exists(cf):
            z = np.load(cf)
            nofit_curv = {k: z[k] for k in ('V', 'k1', 'k2', 'mask')}
            nofit_curv['kink'] = {'all': z['kink'], 'concave': z['kink_concave'], 'convex': z['kink_convex']} if 'kink_concave' in z.files else z['kink']
        rf = os.path.join(a.nofit_run, 'report.json')
        nofit_rep = json.load(open(rf)) if os.path.exists(rf) else None
    cutters = {}
    if a.run:
        cm = load_multi(os.path.join(a.run, 'cutters.npz'))
        cutters = {p: cm[p + ' bool'] for p in ('R2', 'R4') if p + ' bool' in cm}
        for p in ('R2', 'R4'):                  # in-plane extent of each press face (seat rim reach)
            f = os.path.join(a.run, 'parts', p + '.npz')
            if press and p in press and os.path.exists(f):
                z = np.load(f)
                if 'press_tris' in z.files:
                    Vt, Tt = z['V'], z['T']
                    P = Vt[np.unique(Tt[z['press_tris'].astype(bool)])]
                    c, n = np.array(press[p]['center']), np.array(press[p]['normal'])
                    q = P - c - np.outer((P - c) @ n, n)
                    press[p]['extent_mm'] = round(float(np.linalg.norm(q, axis=1).max()), 3)
    out_dir = a.run or (os.path.dirname(os.path.abspath(a.out)) if a.out else None)
    curv_out = os.path.join(out_dir, 'curvature.npz') if out_dir else None
    sm = GM.smooth_report(V, T, caps, nofit=nofit, nofit_curv=nofit_curv, press=press, cutters=cutters, curv_out=curv_out)
    ss = sm.pop('_change_signed', None)
    if ss is not None and curv_out:
        z = dict(np.load(curv_out)); z['change_vs_nofit'] = ss
        np.savez_compressed(curv_out, **z)
    ref = {}
    pr = _plain_reference(a.base_run) if a.run else None
    if pr:
        ref['plain GripV handle_below_z40'] = pr['plain GripV handle_below_z40']
    if nofit_rep and 'smooth' in nofit_rep:
        ref['no-fit baseline handle_below_z40'] = nofit_rep['smooth']['smoothness'].get('handle_below_z40')
    sm['reference'] = ref
    if nofit is not None:
        Vn, Tn = nofit
        Tn, _ = GM.orient(Vn, Tn)
        T, _ = GM.orient(V, T)
        tf, tfn = GM.top_face(V, T), GM.top_face(Vn, Tn)
        sm['vs_nofit'] = {'section_z40': GM.section_change(V, T, Vn, Tn),
                          'top_face': {k: [tf.get(k), tfn.get(k)] for k in ('max_z', 'verts_at_max_z', 'tris_at_max_z', 'top_area_mm2', 'top_y_min', 'verts_above_59.04')},
                          'bbox_min_shift_mm': round(float(np.abs(V.min(0) - Vn.min(0)).max()), 4),
                          'bbox_min_yz_shift_mm': round(float(np.abs(V.min(0)[1:] - Vn.min(0)[1:]).max()), 4),
                          'bbox_x_shift_mm': round(float(max(abs(V[:, 0].min() - Vn[:, 0].min()), abs(V[:, 0].max() - Vn[:, 0].max()))), 4),
                          'bbox_min': V.min(0).round(4).tolist(), 'bbox_min_nofit': Vn.min(0).round(4).tolist(),
                          'bbox_max_shift_mm': round(float(np.abs(V.max(0) - Vn.max(0)).max()), 4),
                          'max_y': [round(float(V[:, 1].max()), 4), round(float(Vn[:, 1].max()), 4)],
                          'volume_change_mm3': round(GM.orient(V, T)[1] - GM.orient(Vn, Tn)[1], 1)}
        if nofit_rep and 'parts' in rep and 'case_top' in rep['parts'] and 'parts' in nofit_rep and 'case_top' in nofit_rep['parts']:
            sm['vs_nofit']['case_top_volume_change_mm3'] = round(rep['parts']['case_top']['volume_mm3'] - nofit_rep['parts']['case_top']['volume_mm3'], 1)
            sm['vs_nofit']['case_top_top_verts'] = [rep['parts']['case_top']['top_face']['verts_at_max_z'], nofit_rep['parts']['case_top']['top_face']['verts_at_max_z']]
            mp, mpn = rep['parts'].get('case_middle_plate'), nofit_rep['parts'].get('case_middle_plate')
            if mp and mpn and not mp.get('empty') and not mpn.get('empty'):
                sm['vs_nofit']['case_middle_plate_volume_change_mm3'] = round(mp['volume_mm3'] - mpn['volume_mm3'], 2)
                sm['vs_nofit']['case_middle_plate_bbox_shift_mm'] = round(float(np.abs(np.array(mp['bbox']) - np.array(mpn['bbox'])).max()), 3)
            fa, fb = os.path.join(a.run or '', 'parts', 'case_top.npz'), os.path.join(a.nofit_run, 'parts', 'case_top.npz')
            if a.run and os.path.exists(fa) and os.path.exists(fb):
                za, zb = np.load(fa), np.load(fb)
                sm['vs_nofit']['case_top_dev'] = GM.mesh_deviation(za['V'], za['T'], zb['V'], zb['T'])
    rep['smooth'] = sm
    rep['acceptance'] = GM.acceptance(rep, nofit_rep)
    rep['summary3'] = GM.acceptance_line(rep['acceptance'])
    print('ACCEPTANCE', rep['summary3'], flush=True)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--run')
    ap.add_argument('--base-run')
    ap.add_argument('--stage')
    ap.add_argument('--base')
    ap.add_argument('--caps')
    ap.add_argument('--blend')
    ap.add_argument('--object', default='basemesh GripV')
    ap.add_argument('--variant', default='GripV')
    ap.add_argument('--cutouts')
    ap.add_argument('--triggers', help='dir with R2.npz/R4.npz (e.g. a parts/ folder)')
    ap.add_argument('--out')
    ap.add_argument('--quick', action='store_true', help='skip wrap angles and internal clearances')
    ap.add_argument('--nofit-run', help='the no-fit baseline run (smooth/ extension): edited region, smoothness reference, invariants vs it')
    ap.add_argument('--no-smooth', action='store_true', help='skip the smooth/ extension')
    a = ap.parse_args(argv)
    t0 = time.time()
    rep = {'inputs': vars(a)}
    chg = None
    base = None
    cutouts = internals = triggers = press = seats = None
    if a.run:
        V, T, chg = GM.load_mesh(os.path.join(a.run, 'stage.npz'))
        caps_path = a.caps or (os.path.join(a.run, 'caps.json') if os.path.exists(os.path.join(a.run, 'caps.json')) else GM.DEFAULT_CAPS)
        if a.base_run:
            Vb, Tb, _ = GM.load_mesh(os.path.join(a.base_run, 'stage.npz'))
            base = (Vb, Tb)
        cutouts = {**{'static: ' + k: v for k, v in load_multi(os.path.join(a.run, 'cutouts_static.npz')).items()},
                   **{'dynamic: ' + k: v for k, v in load_multi(os.path.join(a.run, 'cutouts_dynamic.npz')).items()}}
        internals = load_multi(os.path.join(a.run, 'internals.npz'))
        triggers = {}
        for p in ('R2', 'R4'):
            f = os.path.join(a.run, 'parts', p + '.npz')
            if os.path.exists(f):
                Vt, Tt, _ = GM.load_mesh(f)
                triggers[p] = (Vt, Tt)
        rep['parts'], pj, bj = part_report(a.run, a.base_run)
        press = pj.get('press_faces')
        seats = pj.get('seats')
        rep['settings'] = pj.get('settings')
        rep['stage_inputs'] = pj.get('stage_inputs')
        rep['timing'] = pj.get('timing')
        if 'R1' in rep['parts'] and 'L1' in rep['parts']:
            rep['R1L1_shift_mm'] = max(rep['parts'][p].get('bbox_shift_max_mm', 0.0) or 0.0 for p in ('R1', 'L1'))
        if 'case_top' in rep['parts']:
            ct = rep['parts']['case_top']
            rep['case_top_summary'] = {'volume_mm3': ct['volume_mm3'], 'base_volume_mm3': ct.get('base_volume_mm3'),
                                       'top_verts_at_max_z': ct['top_face']['verts_at_max_z'], 'max_z': ct['top_face']['max_z'],
                                       'closed': ct['closed_manifold'], 'islands': ct['islands'],
                                       'volume_bmesh_mm3': ct.get('volume_bmesh_mm3'), 'base_volume_bmesh_mm3': ct.get('base_volume_bmesh_mm3')}
    elif a.blend:
        import bpy
        bpy.ops.wm.open_mainfile(filepath=a.blend)
        gp = bpy.data.node_groups.get('global')
        if gp and a.variant and 'global parameters' in gp.nodes:
            gp.nodes['global parameters'].inputs['variant'].default_value = a.variant
        dg = bpy.context.evaluated_depsgraph_get()
        ob = bpy.data.objects[a.object]
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        me.calc_loop_triangles()
        V = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', V); V = V.reshape(-1, 3)
        T = np.empty(len(me.loop_triangles) * 3, np.int64); me.loop_triangles.foreach_get('vertices', T); T = T.reshape(-1, 3)
        if 'change_mm' in me.attributes:
            chg = np.empty(len(me.vertices)); me.attributes['change_mm'].data.foreach_get('value', chg)
        M = np.array(ob.matrix_world)
        V = V @ M[:3, :3].T + M[:3, 3]
        ev.to_mesh_clear()
        caps_path = a.caps or GM.DEFAULT_CAPS
    else:
        V, T, chg = GM.load_mesh(a.stage)
        caps_path = a.caps or GM.DEFAULT_CAPS
        if a.base:
            Vb, Tb, _ = GM.load_mesh(a.base)
            base = (Vb, Tb)
        if a.cutouts:
            cutouts = load_multi(a.cutouts)
        if a.triggers:
            triggers = {}
            for p in ('R2', 'R4'):
                f = os.path.join(a.triggers, p + '.npz')
                if not os.path.exists(f):
                    f = os.path.join(a.triggers, f'base_{p}.npz')
                if os.path.exists(f):
                    Vt, Tt, _ = GM.load_mesh(f)
                    triggers[p] = (Vt, Tt)
    caps = GM.load_caps(caps_path)
    rep['caps'] = caps_path
    rep['skin'] = GM.skin_report(V, T, caps, base=base, cutouts=cutouts, internals=internals, triggers=triggers,
                                 press=press, seats=seats, quick=a.quick, chg=chg)
    rep['summary'] = GM.summary_line(rep['skin'])
    if not a.no_smooth:
        smooth_part(a, rep, V, T, caps, press, base)
    rep['seconds_total'] = round(time.time() - t0, 1)
    out = a.out or (os.path.join(a.run, 'report.json') if a.run else None)
    if out:
        json.dump(rep, open(out, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print('SUMMARY', rep['summary'])
    print('REPORT', out, round(time.time() - t0, 1), 's', flush=True)
    return rep


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
