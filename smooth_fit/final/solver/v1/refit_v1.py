"""refit.py: re-solve the smooth hand-fit shape and store it in the .blend. THE button to press after
re-posing the hand (or after changing Smoothness / Press / Max Dig / Seat Pull / Seat Proud).

    /tmp/claude-0/venv/bin/python /tmp/claude-0/sdf/smooth/final/refit.py -- --blend /tmp/claude-0/sdf/smooth/final/Grip_smooth.blend --triggers

  --triggers  first re-fit the R2 / R4 triggers to the fingertips (stage/trigger_solver.py, in place, ~1-2 min).
              Use it whenever the fingers moved. Without it the triggers stay as they are.
  --cold      start the shape from zero instead of the stored one (after a big pose change; ~6-7 min instead of ~1.5)
  --ignore-trigger-check   solve the shape even when the trigger check fails (only when Claude tells you to)

Timing (measured): a re-pose with --triggers takes about 4-4.5 min; a control change without --triggers ~1.5-2 min.

Trigger check (before anything is solved): the trigger solver's residuals are read for R2 and R4. The script stops
with a message, and leaves the file unchanged, when a press-face centre is more than 0.1 mm from its fingertip
target, its normal more than 1 deg off, the trigger travel more than 26 deg from the pressing direction (spec 25;
the accepted v_27 triggers sit at 25.4 / 25.6, the solver's own limit is max(25, the angle it started from)), or
a hand capsule sinks more than 0.55 mm into a trigger part (TRIGGER_GATE below). Without --triggers the current
triggers are checked against the current pose (catches a re-pose without --triggers).

What it does
  1. Evaluates the stage input mesh (the subdivided GripV basemesh, hand fit switched off) and the hand capsules
     (object 'hand points', which follows 'hand rig') of the file itself.
  2. Reads Olex's values from the node group 'GripV hand fit parameters' (dashboard 'GripV controls', frame HAND FIT):
        Smoothness   tightest hollow the shape may make (radius, mm)
        Press        how far each finger contact sinks in (mm)
        Max Dig      no part of the hand deeper than this anywhere (mm)
        Seat Pull    how hard the case around R2 / R4 is pulled flush with the press faces
        Seat Proud   how far each press face stands out of the case around it (mm)
     (advanced, hidden on the 'hand fit' modifier: Contact Pull, Middle PIP Pull)
  3. Solves the offset with SHAPE_SOLVER (default solve_shape_dense: final/solver/dense_fit.py, thin-plate bending +
     soft contacts + soft seats + no-dig / frozen / curvature limits; the other options are in final/fit_opts.json),
     warm-started from the stored shape. Another solver drops in as one function with the same signature: set
     "solver": "/abs/path/file.py:function" in fit_opts.json (see solve_shape_dense for the contract).
  4. Writes it into the mesh object 'shape offset' (point attribute 'offset') and saves the file in place.
The case then shows the new shape (mod_shape reads 'shape offset').
The trigger seats (press faces, openings) are read live from the file's R2 / R4 and 'R2 bool' / 'R4 bool'.
Fixed inputs that do NOT follow automatically: the internal keep-outs and GripV's own design features (creases the
shape must leave alone) come from /tmp/claude-0/sdf/smooth/targets.npz and the no-fit baseline: they only change
if the BODY changes (another body preset, subdivision or basemesh edit); then they must be rebuilt (GUIDE.md).
"""
import os
import sys
import json
import time
import numpy as np
import bpy

HERE = '/tmp/claude-0/sdf/smooth/final'
for p in (HERE + '/solver', '/tmp/claude-0/sdf/smooth/lib', HERE + '/stage'):
    sys.path.insert(0, p)
import gripmetrics as GM      # noqa: E402
import shape_solver as SS     # noqa: E402
import dense_fit as DF        # noqa: E402
import build_stage as BS      # noqa: E402
import eval_all as EA         # noqa: E402

PROTECTED = ('/tmp/claude-0/sdf/smooth/base.blend', '/tmp/claude-0/sdf/build/Grip_hand.blend', '/home/claude/Grip/blender/Grip.blend')
# trigger check (fix 2026-10-20): stop before solving the shape when R2 / R4 do not sit on the fingertips
TRIGGER_GATE = {   # trigger_solver state key -> (limit, words for Olex)
    'center_to_target_mm': (0.1, 'press-face centre to the fingertip target', 'mm'),
    'normal_to_target_deg': (1.0, 'press-face angle to the fingertip target', 'deg'),
    'travel_vs_press_direction_deg': (26.0, 'trigger travel vs the way the fingertip presses', 'deg'),
    'max_hand_overlap_mm': (0.55, 'hand sunk into the trigger part', 'mm'),
}
FINGER = {'R2': 'middle fingertip', 'R4': 'ring fingertip'}
OLEX = {  # control / modifier input -> solver option
    'Press': 'press', 'Max Dig': 'max_dig', 'Smoothness': 'min_concave_radius', 'Seat Pull': 'seat_corridor_w',
    'Seat Proud': 'seat_proud', 'Contact Pull': 'contact_weight', 'Middle PIP Pull': 'middle_pip_weight'}


def store_offset(V, T, off):
    off = np.array(off, float).reshape(-1, 3)
    off[V[:, 0] <= 0] = 0.0          # right handle only; the stage mirrors it to the left one
    me = bpy.data.meshes.new('shape offset')
    me.vertices.add(len(V)); me.vertices.foreach_set('co', V.ravel())
    me.loops.add(3 * len(T)); me.loops.foreach_set('vertex_index', T.ravel())
    me.polygons.add(len(T)); me.polygons.foreach_set('loop_start', np.arange(0, 3 * len(T), 3))
    me.update()
    me.attributes.new('offset', 'FLOAT_VECTOR', 'POINT').data.foreach_set('vector', off.ravel())
    so = bpy.data.objects.get('shape offset')
    if so is None:
        so = bpy.data.objects.new('shape offset', me)
        bpy.data.collections['hand fit'].objects.link(so)   # never 'basemesh GripV' (the generator reads that collection)
    else:
        old = so.data; so.data = me
        if old.users == 0:
            bpy.data.meshes.remove(old)
    me.name = 'shape offset'
    so.hide_render = True
    so.display_type = 'BOUNDS'
    return so


def live_seats(dg):
    """R2 / R4 seats from the evaluated triggers and cutters of the open file (same recipe as make_targets.py)."""
    import trigger_solver as TS
    seats = {}
    for P in ('R2', 'R4'):
        out = {}
        for n in (P, P + ' bool'):
            ob = bpy.data.objects[n]
            e = ob.evaluated_get(dg); m = e.to_mesh(); m.calc_loop_triangles()
            M = np.array(ob.matrix_world)
            V = np.empty(len(m.vertices) * 3); m.vertices.foreach_get('co', V); V = V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
            T = np.empty(len(m.loop_triangles) * 3, np.int64); m.loop_triangles.foreach_get('vertices', T); T = T.reshape(-1, 3)
            out[n] = (V, T, TS.press_face(m, M) if n == P else None)
            e.to_mesh_clear()
        pf = out[P][2]
        c, nn = np.asarray(pf['center'], float), np.asarray(pf['normal'], float)
        Vc, Tc = out[P + ' bool'][:2]
        seats[P] = dict(center=c, normal=nn, segments=GM.plane_section(Vc, Tc, c - GM.SEAT_SKIN * nn, nn), cutter=(Vc, Tc))
    return seats


def trigger_gate(report_path):
    """Read the trigger solver's report; return (ok, lines for Olex, numbers)."""
    rep = json.load(open(report_path))
    ok, lines, nums = True, [], {}
    for P in ('R2', 'R4'):
        r = rep.get('parts', {}).get(P, {})
        st = r.get('after') or r.get('current')
        if st is None:
            ok = False; lines.append(f'  {P}: no result from the trigger solver'); continue
        nums[P] = {k: st.get(k) for k in TRIGGER_GATE}
        for k, (lim, words, unit) in TRIGGER_GATE.items():
            v = st.get(k)
            bad = v is None or v > lim
            ok &= not bad
            lines.append(f"  {P} ({FINGER[P]}): {words} {v} {unit} (limit {lim}){'   <-- TOO FAR' if bad else ''}")
    return ok, lines, nums


def solve_shape_dense(ctx):
    """THE shape solver call (pluggable). ctx keys:
         V, T        stage input mesh (subdivided GripV basemesh, hand fit off), (N, 3) / (M, 3)
         caps        hand capsules {name: (a, b, r)};  zones = GM.zones(caps)
         seats       live R2 / R4 seats (press face centre / normal, opening segments, cutter mesh)
         keep        fixed keep-outs from smooth/targets.npz (SS.load_targets)
         opt         solver options (fit_opts.json 'opt' + Olex's controls), best = the whole fit_opts.json
         cold        True: start from zero;  stored  (N, 3) stored offset of 'shape offset' or None
       returns dict(offset=(N, 3) offset vectors per vertex of V (right handle; left is zeroed and mirrored by the
       stage), T=triangles to store, npz={extra arrays for solve/refit_last.npz}, lines=[report lines]).
    A replacement (e.g. the v2 solver) implements the same contract and is named in fit_opts.json "solver"."""
    V, T, opt, best = ctx['V'], ctx['T'], dict(ctx['opt']), ctx['best']
    fit = DF.Fit(V, T, ctx['caps'], ctx['zones'], ctx['seats'], ctx['keep'], opt)
    h0 = None
    if ctx['stored'] is not None and not ctx['cold']:
        h0 = (ctx['stored'] * fit.D).sum(1)
        h0[fit.fixed0] = 0.0
        print(f'[refit] warm start from the stored shape ({h0.min():.2f}..{h0.max():.2f} mm)', flush=True)
    elif not ctx['cold']:
        print('[refit] the stored shape does not match the mesh (basemesh changed?): cold start', flush=True)
        opt.update(best.get('cold', {}))
        fit = DF.Fit(V, T, ctx['caps'], ctx['zones'], ctx['seats'], ctx['keep'], opt)
    h = fit.solve(h0)
    last = fit.history[-1].get('contacts', {}) if fit.history else {}
    lines = ['[refit] contacts (penetration mm / support %): ' + ', '.join(f'{k} {v[0]:.2f}/{v[2]:.0f}' for k, v in last.items()),
             f'[refit] shape {h.min():.2f}..{h.max():.2f} mm, solve {fit.seconds:.0f} s']
    return dict(offset=h[:, None] * fit.D, T=fit.T, npz=dict(h=h, D=fit.D, P=fit.moved(h)), lines=lines)


SHAPE_SOLVER = solve_shape_dense


def load_solver(best):
    """fit_opts.json "solver": "/abs/path/file.py:function" replaces SHAPE_SOLVER (same contract)."""
    spec = best.get('solver')
    if not spec:
        return SHAPE_SOLVER
    import importlib.util
    path, fn = spec.rsplit(':', 1)
    sp = importlib.util.spec_from_file_location('refit_plugin_solver', path)
    m = importlib.util.module_from_spec(sp)
    sys.path.insert(0, os.path.dirname(path))
    sp.loader.exec_module(m)
    print(f'[refit] shape solver: {spec}', flush=True)
    return getattr(m, fn)


def main(argv):
    o = {'--blend': None, '--cold': False, '--out-npz': None, '--triggers': False, '--ignore-trigger-check': False}
    i = 0
    while i < len(argv):
        if argv[i] in ('--cold', '--triggers', '--ignore-trigger-check'):
            o[argv[i]] = True; i += 1
        else:
            o[argv[i]] = argv[i + 1]; i += 2
    blend = os.path.abspath(o['--blend'])
    assert blend not in PROTECTED, 'refit writes the file: never pass ' + blend
    t0 = time.time()
    import subprocess
    ts = os.path.join(HERE, 'stage', 'trigger_solver.py')
    rdir = os.path.join(HERE, 'runs', 'refit')
    os.makedirs(rdir, exist_ok=True)
    rpt = os.path.join(rdir, 'trigger_report.json')
    tmp = None
    if o['--triggers']:
        # the solver writes a temporary copy next to the file; Olex's file is only written once everything passed
        tmp = os.path.splitext(blend)[0] + '.refit_triggers.blend'
        print('[refit] 1/3 re-fitting R2 / R4 to the fingertips:', ts, flush=True)
        subprocess.run([sys.executable, ts, '--', '--in', blend, '--out', tmp, '--report', rpt], check=True)
    else:
        print('[refit] 1/3 checking that R2 / R4 still sit on the fingertips (no --triggers):', ts, flush=True)
        subprocess.run([sys.executable, ts, '--', '--in', blend, '--show', '--report', rpt], check=True)
    ok, lines, _ = trigger_gate(rpt)
    print('[refit] trigger check:\n' + '\n'.join(lines), flush=True)
    if not ok and not o['--ignore-trigger-check']:
        for f in (tmp, tmp + '1' if tmp else None):
            if f and os.path.exists(f):
                os.remove(f)
        print('\n[refit] STOPPED before solving the shape: R2 / R4 do not sit on your fingertips (lines marked TOO FAR).\n'
              '        Your file was NOT changed.\n'
              + ('        You ran without --triggers: if you moved the fingers, run the same command again with --triggers.\n'
                 if not o['--triggers'] else
                 '        The triggers could not follow the new fingertip positions (their switches and lever paths stay fixed).\n'
                 '        Pose the middle / ring fingertips closer to where they were, or ask Claude to look at the report.\n')
              + f'        Trigger report: {rpt}', flush=True)
        sys.exit(2)
    src = tmp or blend
    bpy.ops.wm.open_mainfile(filepath=src)
    win = bpy.context.window.scene
    ob = bpy.data.objects[BS.TARGET]
    mod = ob.modifiers[BS.MOD_NAME]
    vals = BS.get_inputs(mod)
    best = json.load(open(os.path.join(HERE, 'fit_opts.json')))
    opt = dict(best['opt'])
    if o['--cold']:
        opt.update(best.get('cold', {}))
    for k, sk in OLEX.items():
        if k in vals and vals[k] is not None:
            opt[sk] = float(vals[k])
    w = opt.pop('middle_pip_weight', 1.0)
    zs = dict(opt.get('zone_scale') or {})
    zs['middle PIP'] = w
    opt['zone_scale'] = zs
    opt['verbose'] = True
    # ---- the stage input mesh and the hand, evaluated with the hand fit off, in a scratch scene
    enable0 = vals.get('Enable', True)
    BS.set_inputs(mod, {'Enable': False})
    sc = bpy.data.scenes.new('hf refit')
    bpy.context.window.scene = sc
    for n in (BS.TARGET, 'hand points', 'hand rig', 'R2', 'R4', 'R2 bool', 'R4 bool'):
        sc.collection.objects.link(bpy.data.objects[n])
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    V, T, _, _ = EA.mesh_arrays(ob, dg)
    caps = EA.hand_caps(dg)
    seats = live_seats(dg)
    bpy.context.window.scene = win
    bpy.data.scenes.remove(sc)
    BS.set_inputs(mod, {'Enable': enable0})
    _, seats_shared, keep = SS.load_targets()
    for P in seats:
        dc = float(np.linalg.norm(seats[P]['center'] - seats_shared[P]['center']))
        print(f'[refit] seat {P}: press face {np.round(seats[P]["center"], 3).tolist()} ({dc:.3f} mm from the design run), '
              f'opening {len(seats[P]["segments"])} segments', flush=True)
    print(f'[refit] 2/3 solving the shape: input mesh {len(V)} verts, {len(caps)} capsules, solver inputs ' +
          json.dumps({k: opt.get(v) for k, v in OLEX.items() if v in opt}) + f', middle PIP pull {w}', flush=True)
    stored = None
    so = bpy.data.objects.get('shape offset')
    if so is not None and len(so.data.vertices) == len(V) and 'offset' in so.data.attributes:
        stored = np.empty(3 * len(V)); so.data.attributes['offset'].data.foreach_get('vector', stored)
        stored = stored.reshape(-1, 3)
    solver = load_solver(best)
    res = solver(dict(V=V, T=T, caps=caps, zones=GM.zones(caps), seats=seats, keep=keep, opt=opt, best=best,
                      cold=o['--cold'], stored=stored))
    out = o['--out-npz'] or os.path.join(HERE, 'solve', 'refit_last.npz')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, V=V, T=res['T'], offset=res['offset'], **res.get('npz', {}))
    store_offset(V, res['T'], res['offset'])
    print('[refit] 3/3 saving', blend, flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    for f in (tmp, tmp + '1' if tmp else None):
        if f and os.path.exists(f):
            os.remove(f)
    for ln in res.get('lines', []):
        print(ln, flush=True)
    print(f'[refit] done: total {time.time() - t0:.0f} s -> {blend}', flush=True)
    print('[refit] check it: /tmp/claude-0/venv/bin/python /tmp/claude-0/sdf/smooth/final/check.py -- --blend ' + blend, flush=True)


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
