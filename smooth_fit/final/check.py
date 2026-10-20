"""check.py: measure and render Grip_smooth.blend (or a copy): shared metrics + the SPEC v2 relief acceptance, ~3.5 min.

    /tmp/claude-0/venv/bin/python /tmp/claude-0/sdf/smooth/final/check.py -- [--blend F] [--tag NAME] [--fast] [--nofit-run DIR] [--no-rebuild] [--no-relief] [--no-render]

Results in final/runs/NAME/: report.json (shared acceptance table), relief.json (SPEC v2: relief, 6 mm curvature,
creases, one-sided contacts, soft-tissue allowance, trigger stroke, screw wall, invariants), spec_v2.json (the SPEC v2
table printed below) and sheet.jpg (3 x 3 views). The file is not changed.
--fast: skin + triggers only, no render, relief without curvature (~1.5 min).
It refuses to run when the file's hidden 'View' input is not 0 (that would replace every part's basemesh).
The trigger stroke check needs the inputs refit.py saved for this file (final/solve/inputs_<file name>.npz).
Rows of the shared table that encode the v1 spec (contact penetration 0.2-1.0 mm, support %, max dig 1.0, flush
seat rims) are printed as 'v1' and are not SPEC v2 failures: SPEC v2 replaced them by one-sided touches + Palm Give /
Finger Give + the stroke keep-out.
"""
import os, sys, json
FINAL = '/tmp/claude-0/sdf/smooth/final'
sys.path.insert(0, FINAL + '/stage')
import eval_all
args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
blend = args[args.index('--blend') + 1] if '--blend' in args else FINAL + '/Grip_smooth.blend'
tag = args[args.index('--tag') + 1] if '--tag' in args else 'check'
a = ['--blend', blend, '--modules-dir', FINAL + '/modules', '--out-root', FINAL + '/runs', '--tag', tag]
# no-fit reference (Fit 0) made with the repaired, outward-wound stage (fix 2026-10-20)
nofit = args[args.index('--nofit-run') + 1] if '--nofit-run' in args else FINAL + '/runs/nofit_fix'
if os.path.exists(os.path.join(nofit, 'stage.npz')):
    a += ['--nofit-run', nofit]
if '--no-rebuild' in args:
    a += ['--no-rebuild']
if '--fast' in args:
    a += ['--skin-only', '--quick', '--no-render']
elif '--no-render' in args:
    a += ['--no-render']
eval_all.main(a)
run = f'{FINAL}/runs/{tag}'
rep = json.load(open(f'{run}/report.json'))
V1_RULES = ('penetration', 'support ', 'max dig, editable', 'seat: rim', 'seat: skin at press-face')
print('SHARED METRICS (smooth/lib/gripmetrics.py)')
for k, v in rep['acceptance'].items():
    if k.startswith('_'):
        continue
    mark = {True: 'ok  ', False: 'FAIL', None: 'info'}[v['pass']]
    if v['pass'] is False and v['kind'] == 'soft':
        mark = 'soft'
    if any(s in k for s in V1_RULES):
        mark = 'v1  '
    print(f"  {mark} {k}: {v['value']}  [{v['limit']}]")
print(rep.get('summary3', ''))
if '--no-relief' in args:
    sys.exit(0)
# ---- SPEC v2 acceptance (final/lib_v2/relief.py)
sys.path.insert(0, FINAL + '/lib_v2')
import relief as RL
stem = os.path.splitext(os.path.basename(blend))[0]
inputs = f'{FINAL}/solve/inputs_{stem}.npz'
opts = {}
if os.path.exists(f'{FINAL}/solve/refit_{stem}.json'):
    opts = json.load(open(f'{FINAL}/solve/refit_{stem}.json')).get('options', {})
pg, fg = float(opts.get('palm_give', 3.5)), float(opts.get('finger_give', 1.5))
sc, sw = float(opts.get('stroke_clear', 0.3)), float(opts.get('screw_wall', 3.0))
if not os.path.exists(inputs):
    print(f'[check] no {inputs}: the trigger stroke is not checked (run refit.py on this file first)')
nf_parts = nofit if os.path.exists(os.path.join(nofit, 'parts')) else RL.NOFIT_PARTS
r = RL.analyze(run, f'{run}/relief.json', tag, inputs, RL.PLAIN, nofit, nf_parts, pg, fg, quick='--fast' in args)
rows = []
def row(name, value, ok, limit):
    rows.append(dict(name=name, value=value, pass_=ok, limit=limit))
rl = r['relief']
row('relief: dents / bumps (prominence >= 1 mm vs the 12-16 mm ring)', [rl['dents'], rl['bumps']], rl['dents'] == 0 and rl['bumps'] <= 2, '0 dents, <= 2 bumps')
if 'curv6' in r:
    c = r['curv6']
    row('6 mm: area concave R<20 vs plain GripV, same footprint (ratio)', c['ratio_concave_R<20_vs_plain'], c['ratio_concave_R<20_vs_plain'] <= 1.5, '<= 1.5')
    row('6 mm: area convex R<8 vs plain GripV, same footprint (ratio)', c['ratio_convex_R<8_vs_plain'], c['ratio_convex_R<8_vs_plain'] <= 1.5, '<= 1.5')
    for kind in ('concave', 'convex'):
        cr = r.get('creases', {}).get(kind)
        if cr:
            # review round 2: the full (same-vertex) row is shown next to the 2 mm-neighbourhood one
            row(f'GripV {kind} creases: min R ratio vs no-fit, % > 20% tighter (same place / 2 mm neighbourhood)',
                [[cr['min_ratio'], cr['pct_tighter_20']], [cr['nbhd2_min_ratio'], cr['nbhd2_pct_tighter_20']]], None,
                'info (v1: 0.48, 7.56 %); the same-place value is a single-vertex measure')
for z, v in r['contacts'].items():
    row(f'touch {z}: gap mm', v['gap'], v['gap'] <= 0.3, '<= 0.3 (touching); negative = overlap')
bad = {k: v['overlap_editable'] for k, v in r['allowance'].items() if v['accept'] is False}
# index capsules lie on the frozen R1 region: the shape cannot change them; shown with their plain value (review round 2)
try:
    import numpy as _np
    _c = json.load(open(f'{run}/caps.json')); _d = _np.load(f'{run}/stage.npz'); _P = _d['V'][_d['V'][:, 0] > 0]
    def _gap(c):
        a, b, rr = _np.array(c['a']), _np.array(c['b']), c['r']; ab = b - a
        t = _np.clip(((_P - a) @ ab) / (ab @ ab), 0, 1)
        return round(float(_np.linalg.norm(_P - (a + t[:, None] * ab), axis=1).min() - rr), 2)
    row('index capsules (frozen R1 region, plain value; Finger Give does not apply): gap mm', {k: _gap(v) for k, v in _c.items() if k.startswith('index')}, None, 'info')
except Exception as ex:   # noqa: BLE001
    print('[check] index capsules not measured:', ex)
row(f'soft-tissue allowance (Palm Give {pg}, Finger Give {fg}): capsules over it', bad or 'none', not bad, 'none')
ko = r.get('keepouts', {})
for P in ('R2', 'R4'):
    s = ko.get('stroke_' + P)
    if s:
        row(f'trigger stroke {P}: fingertip gap at rest / fully pressed mm', [s['gap_rest'], s['gap_pressed']], s['gap_pressed'] >= sc - 1e-6, f'pressed >= {sc}')
if 'screw_wall' in ko:
    row('screw-channel wall min mm (where plain GripV has >= 3)', ko['screw_wall'].get('min_wall_mm'), ko['screw_wall'].get('min_wall_mm', 0) >= sw - 0.05, f'>= {sw}')
inv = r['invariants']
row('frozen region change vs no-fit mm', inv['frozen_change_vs_nofit_max_mm'], inv['frozen_change_vs_nofit_max_mm'] <= 0.05, '<= 0.05')
row('mirror deviation mm', inv['mirror_dev_max_mm'], inv['mirror_dev_max_mm'] <= 0.05, '<= 0.05')
for part in ('case_top', 'case_middle_plate'):
    d = inv.get(part + '_dev_vs_nofit')
    if d:
        row(f'{part} vs no-fit: max dev mm, verts > 0.05 mm', [d[0], d[2]], d[0] <= 0.05 or d[2] <= 2, '<= 0.05 (or <= 2 isolated verts: remesh noise)')
row('bbox x extent vs no-fit mm (open question)', round(inv['bbox_x_extent'] - inv['nofit_bbox_x_extent'], 3), None, 'info')
json.dump(dict(blend=blend, run=run, options=dict(palm_give=pg, finger_give=fg, stroke_clear=sc, screw_wall=sw), rows=rows),
          open(f'{run}/spec_v2.json', 'w'), indent=1, default=str)
print('SPEC v2 ACCEPTANCE (final/lib_v2/relief.py)')
for x in rows:
    mark = {True: 'ok  ', False: 'FAIL', None: 'info'}[x['pass_']]
    print(f"  {mark} {x['name']}: {x['value']}  [{x['limit']}]")
print('RUN', run)
