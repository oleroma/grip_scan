"""refit.py: re-solve the smooth hand-fit shape (SPEC v2) and store it in the .blend. THE button to press after
re-posing the hand, or after changing Shape Size / Palm Give / Finger Give / Stroke Clear / Screw Wall.

    /tmp/claude-0/venv/bin/python /tmp/claude-0/sdf/smooth/final/refit.py -- --blend /tmp/claude-0/sdf/smooth/final/Grip_smooth.blend --triggers

  --triggers  first re-fit the R2 / R4 triggers to the fingertips (stage/trigger_solver.py, about 1 min).
              Use it whenever the fingers moved. Without it the triggers stay as they are (and are only checked).
  --ignore-trigger-check   solve the shape even when the trigger check fails (only when Claude tells you to)
  --skip-case-check        do not run the case-dependent trigger checks on the result (about 25 s; tests only)
  --cold      accepted for old command lines; the v2 solve always starts from zero (it takes 5-15 s)
  --opts FILE.json         solver options (default: final/fit_opts.json)
  --out-npz FILE.npz       where the solve arrays go (default final/solve/refit_<file name>.npz)

Runs only in Claude's environment (this venv's bpy + scipy, paths under /tmp/claude-0): Olex poses the hand, saves
and sends the file; Claude runs refit and sends it back.
Timing (measured 2026-10-20 on this machine): see GUIDE.md (the case check adds about 25 s: 23.5-25.2 s measured).

Trigger check (before anything is solved): the trigger solver's residuals are read for R2 and R4. The script stops
with a message, and leaves the file unchanged, when a press-face centre is more than 0.1 mm from its fingertip
target, its normal more than 1 deg off, the trigger travel more than 26 deg from the pressing direction (the accepted
v_27 triggers sit at 25.4 / 25.6), a hand capsule sinks more than 0.55 mm into a trigger part (TRIGGER_GATE), a value
is missing or not a number, or the lever is no longer under its switch (all 81 rays from the switch footprint must
hit the lever, gap min / max within 0.05 mm of the accepted 0.438..0.484 / 0.470..0.491 mm).
Case check (after the solve, review round 2): the result is saved to <file>.refit_result.blend and checked with
trigger_solver --check-case (no case material between switch and lever, no trigger vertex inside case_bottom); only
then does it replace the file (the previous file is kept as <file>.blend1). Exit codes: 2 trigger check, 3 case check.

What it does
  1. Reads everything from the file itself, evaluated with the hand fit switched off (solver_v2/gather.py): the stage
     input mesh (subdivided GripV basemesh), the hand capsules ('hand points', which follows 'hand rig'), the R2 / R4
     press faces and their stroke (the generator's own 'action' preview), the 'R2 bool' / 'R4 bool' cutters and the
     internal keep-outs ('screw bools', 'battery wire space', pcb, pillars ...).
  2. Reads Olex's values from 'GripV hand fit parameters' (dashboard 'GripV controls', frame HAND FIT):
        Shape Size    support radius of the smooth bumps the shape is made of (mm, >= 25 enforced)
        Palm Give     how far palm, metacarpals and pinky may sink in before the case gives way (mm)
        Finger Give   the same for the other finger segments (mm)
        Stroke Clear  middle / ring fingertip vs the case beside the openings over the whole trigger stroke (mm)
        Screw Wall    wall kept between the skin and the screw channels (mm)
     advanced, hidden on the 'hand fit' modifier: Middle PIP Pull (0), Top Clear (3 mm), Touch Reach (1 mm)
  3. Solves the offset with solver_v2/lowfreq.py (h = window x low-frequency basis; see DESIGN.md).
  4. Writes it into the mesh object 'shape offset' (point attribute 'offset') and saves the file in place.
     Also writes final/solve/inputs_<file name>.npz (the gathered inputs; check.py uses its trigger travel) and
     final/solve/refit_<file name>.json (the values used, the contact gaps, the timing).
The v1 refit (dense solve) is kept for reference in final/solver/v1/ (it does not fit this stage any more).
"""
import os
import sys
import json
import math
import time
import shutil
import subprocess
import numpy as np
import bpy

HERE = '/tmp/claude-0/sdf/smooth/final'
for p in (HERE + '/solver_v2', HERE + '/stage', '/tmp/claude-0/sdf/smooth/lib'):
    if p not in sys.path:
        sys.path.insert(0, p)
import lowfreq as LF          # noqa: E402
import store as ST            # noqa: E402
import gather as GA           # noqa: E402
import gather_np as GN        # noqa: E402

PROTECTED = ('/tmp/claude-0/sdf/smooth/base.blend', '/tmp/claude-0/sdf/build/Grip_hand.blend',
             '/home/claude/Grip/blender/Grip.blend')
PROTECTED_NAMES = ('Grip.blend', 'v_27.blend')
PROTECTED_DIRS = ('/tmp/claude-0/sdf/build/', '/mnt/project-files/')
TRIGGER_GATE = {   # trigger_solver state key -> (limit, words for Olex, unit)
    'center_to_target_mm': (0.1, 'press-face centre to the fingertip target', 'mm'),
    'normal_to_target_deg': (1.0, 'press-face angle to the fingertip target', 'deg'),
    'travel_vs_press_direction_deg': (26.0, 'trigger travel vs the way the fingertip presses', 'deg'),
    'max_hand_overlap_mm': (0.55, 'hand sunk into the trigger part', 'mm'),
}
# switch reach (review round 2): the lever must stay under its PCB switch with the gap of the accepted v_27 triggers
# (trigger_solver keeps it only as a soft penalty). Every one of the 81 rays from the switch footprint must hit the
# lever, and the smallest / largest gap may move at most SWITCH_GAP_TOL mm from these values.
SWITCH_GAP_REF = {'R2': (0.438, 0.484), 'R4': (0.470, 0.491)}   # (min, max) mm, accepted v_27 triggers
SWITCH_GAP_TOL = 0.05
FINGER = {'R2': 'middle fingertip', 'R4': 'ring fingertip'}
CONTROLS = {   # dashboard / modifier input -> lowfreq option
    'Shape Size': 'smoothness', 'Palm Give': 'palm_give', 'Finger Give': 'finger_give', 'Stroke Clear': 'stroke_clear',
    'Screw Wall': 'screw_wall', 'Middle PIP Pull': 'middle_pip', 'Top Clear': 'top_clear', 'Touch Reach': 'touch_reach'}
MIN_SHAPE_SIZE = 25.0   # SPEC v2: the basis's smallest support is >= 25 mm


def guard(path):
    p = os.path.abspath(path)
    bad = p in PROTECTED or os.path.basename(p) in PROTECTED_NAMES or any(p.startswith(d) for d in PROTECTED_DIRS)
    assert not bad, 'refit writes the file: never pass ' + p


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
            # a missing or non-finite value (NaN: NaN > lim is False) fails too
            bad = not isinstance(v, (int, float)) or not math.isfinite(v) or v > lim
            ok &= not bad
            lines.append(f"  {P} ({FINGER[P]}): {words} {v} {unit} (limit {lim}){'   <-- TOO FAR' if bad else ''}")
        # switch reach: all rays from the switch footprint hit the lever, gap min / max as accepted
        sg = st.get('switch_gap_mm') or {}
        nums[P]['switch_gap_mm'] = sg
        lo, hi = SWITCH_GAP_REF[P]
        vals = [sg.get('hits'), sg.get('rays'), sg.get('min'), sg.get('max')]
        fin = all(isinstance(x, (int, float)) and math.isfinite(x) for x in vals)
        bad = (not fin) or sg['hits'] != sg['rays'] or abs(sg['min'] - lo) > SWITCH_GAP_TOL or abs(sg['max'] - hi) > SWITCH_GAP_TOL
        ok &= not bad
        lines.append(f"  {P} ({FINGER[P]}): lever under its switch: {sg.get('hits')}/{sg.get('rays')} rays hit, gap "
                     f"{sg.get('min')}..{sg.get('max')} mm (accepted {lo}..{hi} +- {SWITCH_GAP_TOL})"
                     f"{'   <-- LEVER OFF ITS SWITCH' if bad else ''}")
    return ok, lines, nums


def case_gate(report_path):
    """Read trigger_solver --check-case on the NEW file: no case material between switch and lever, every lever ray
    hits, no trigger vertex inside case_bottom material. Returns (ok, lines, numbers)."""
    rep = json.load(open(report_path))
    cc = rep.get('case_checks') or {}
    ok, lines, nums = True, [], {}
    for P in ('R2', 'R4'):
        r = cc.get(P)
        if not r:
            ok = False; lines.append(f'  {P}: no case check result'); continue
        lp, tc = r.get('lever_path', {}), r.get('trigger_in_case', {})
        nums[P] = dict(lever_path=lp, trigger_in_case=tc)
        bad = lp.get('blocked') != 0 or lp.get('lever_hits') != lp.get('rays') or tc.get('count') != 0
        ok &= not bad
        lines.append(f"  {P}: lever path {lp.get('lever_hits')}/{lp.get('rays')} rays reach the lever, {lp.get('blocked')} blocked by case "
                     f"material; trigger vertices inside the case: {tc.get('count')}{'   <-- BLOCKED' if bad else ''}")
    return ok, lines, nums


def options(controls, opts_path=None):
    """fit_opts.json 'opt' (or lowfreq DEFAULTS), then the file's controls when the v2 controls exist."""
    f = opts_path or os.path.join(HERE, 'fit_opts.json')
    opt = dict(json.load(open(f)).get('opt', {})) if os.path.exists(f) else {}
    src = {}
    if 'Palm Give' in controls:
        for k, sk in CONTROLS.items():
            if controls.get(k) is not None:
                opt[sk] = float(controls[k]); src[k] = opt[sk]
    else:
        print('[refit] this file has no v2 controls (Palm Give ...): using final/fit_opts.json. Rebuild its stage with '
              'final/stage/build_stage.py to get the controls.', flush=True)
    if opt.get('smoothness', LF.DEFAULTS['smoothness']) < MIN_SHAPE_SIZE:
        print(f"[refit] Shape Size {opt['smoothness']} is below {MIN_SHAPE_SIZE} mm: using {MIN_SHAPE_SIZE}", flush=True)
        opt['smoothness'] = MIN_SHAPE_SIZE
    return opt, src


def main(argv):
    o = {'--blend': None, '--cold': False, '--out-npz': None, '--triggers': False, '--ignore-trigger-check': False,
         '--opts': None, '--skip-case-check': False}
    i = 0
    while i < len(argv):
        if argv[i] in ('--cold', '--triggers', '--ignore-trigger-check', '--skip-case-check'):
            o[argv[i]] = True; i += 1
        else:
            o[argv[i]] = argv[i + 1]; i += 2
    blend = os.path.abspath(o['--blend'] or os.path.join(HERE, 'Grip_smooth.blend'))
    guard(blend)
    stem = os.path.splitext(os.path.basename(blend))[0]
    t0 = time.time()
    ts = os.path.join(HERE, 'stage', 'trigger_solver.py')
    rdir = os.path.join(HERE, 'runs', 'refit')
    os.makedirs(rdir, exist_ok=True)
    rpt = os.path.join(rdir, f'trigger_report_{stem}.json')
    tmp = None
    if o['--triggers']:
        # the trigger solver writes a temporary copy next to the file; the file itself is only written once all passed
        tmp = os.path.splitext(blend)[0] + '.refit_triggers.blend'
        print('[refit] 1/3 re-fitting R2 / R4 to the fingertips:', ts, flush=True)
        subprocess.run([sys.executable, ts, '--', '--in', blend, '--out', tmp, '--report', rpt], check=True)
    else:
        print('[refit] 1/3 checking that R2 / R4 still sit on the fingertips (no --triggers):', ts, flush=True)
        subprocess.run([sys.executable, ts, '--', '--in', blend, '--show', '--report', rpt], check=True)
    t_trig = time.time() - t0
    ok, lines, tnums = trigger_gate(rpt)
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
    bpy.ops.wm.open_mainfile(filepath=tmp or blend)
    t1 = time.time()
    g = GA.gather()
    opt, src = options(g['controls'], o['--opts'])
    opt['verbose'] = True
    print(f'[refit] 2/3 solving the shape: {len(g["V"])} verts, {len(g["caps"])} capsules, controls from the file {src}',
          flush=True)
    fit = LF.LowFreqFit(g, opt)
    fit.solve()
    off = fit.offset_vectors()
    out = o['--out-npz'] or os.path.join(HERE, 'solve', f'refit_{stem}.npz')
    guard(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # the solve arrays go to .pending files and replace the stored ones only together with the .blend (review round 2:
    # a rejected result must not leave solve/ out of step with the file)
    inp = os.path.join(HERE, 'solve', f'inputs_{stem}.npz')
    out_p, inp_p = os.path.splitext(out)[0] + '.pending.npz', os.path.splitext(inp)[0] + '.pending.npz'
    np.savez_compressed(out_p, V=fit.V, T=fit.T, offset=off, h=fit.h, c=fit.c, D=fit.D, W=fit.W, centres=fit.centres,
                        opt=json.dumps(fit.opt), history=json.dumps(fit.history), zones=json.dumps(fit.zones_final))
    GN.save_npz(inp_p, g)
    ST.store_offset(fit.V, fit.T, off)
    # the result goes to a side file first; it replaces the file only after the case-dependent trigger checks pass
    res = os.path.splitext(blend)[0] + '.refit_result.blend'
    guard(res)
    bpy.ops.wm.save_as_mainfile(filepath=res)
    for f in (tmp, tmp + '1' if tmp else None):
        if f and os.path.exists(f):
            os.remove(f)
    case = dict(ok=None, skipped=bool(o['--skip-case-check']))
    if not o['--skip-case-check']:
        t2 = time.time()
        crpt = os.path.join(rdir, f'trigger_case_{stem}.json')
        print('[refit] 3/3 checking the new case against the triggers (lever path, trigger inside the case):', flush=True)
        subprocess.run([sys.executable, ts, '--', '--in', res, '--show', '--check-case', '--report', crpt], check=True)
        cok, clines, cnums = case_gate(crpt)
        gok, glines, gnums = trigger_gate(crpt)        # the saved file's trigger state, read once more
        case = dict(ok=bool(cok and gok), case=cnums, triggers=gnums, seconds=round(time.time() - t2, 1), report=crpt)
        print('[refit] case check:\n' + '\n'.join(clines + glines), flush=True)
        if not (cok and gok):
            print('\n[refit] STOPPED: the new shape blocks a trigger (lines marked BLOCKED / TOO FAR / LEVER OFF ITS SWITCH).\n'
                  '        Your file was NOT changed. The rejected result is kept for Claude to look at: ' + res
                  + f'\n        (solve arrays: {out_p})', flush=True)
            sys.exit(3)
    if os.path.exists(blend):
        shutil.copy2(blend, blend + '1')               # the previous file, as Blender's own .blend1 backup
    os.replace(res, blend)
    os.replace(out_p, out)
    os.replace(inp_p, inp)
    if os.path.exists(res + '1'):
        os.remove(res + '1')
    print('[refit] saved', blend, '(previous version: ' + blend + '1)', flush=True)
    t_all = time.time() - t0
    summ = dict(blend=blend, controls=src, options={k: fit.opt.get(k) for k in CONTROLS.values()},
                zones_gap_mm=fit.zones_final, h_range_mm=[round(float(fit.h.min()), 3), round(float(fit.h.max()), 3)],
                basis_functions=int(fit.B.shape[1]), trigger_check=dict(ok=bool(ok), numbers=tnums, ignored=bool(not ok)),
                case_check=case, zones_status=getattr(fit, 'zones_status', None), last_violations=fit.history[-1].get('viol'),
                seconds=dict(triggers=round(t_trig, 1), gather_solve=round(time.time() - t1, 1), solve=round(fit.seconds, 1),
                             total=round(t_all, 1)), npz=out, triggers_refitted=bool(o['--triggers']))
    json.dump(summ, open(os.path.join(HERE, 'solve', f'refit_{stem}.json'), 'w'), indent=1, default=str)
    print('[refit] contact gaps (mm; <= 0.3 touches, negative = the hand sinks in): '
          + ', '.join(f'{k} {v:+.2f}' for k, v in fit.zones_final.items()), flush=True)
    print(f'[refit] shape {fit.h.min():.2f}..{fit.h.max():.2f} mm, {fit.B.shape[1]} smooth bumps; times: triggers '
          f'{t_trig:.0f} s, gather + solve {time.time() - t1:.0f} s (solve {fit.seconds:.1f} s), total {t_all:.0f} s -> {blend}',
          flush=True)
    print('[refit] check it: /tmp/claude-0/venv/bin/python /tmp/claude-0/sdf/smooth/final/check.py -- --blend ' + blend, flush=True)


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
