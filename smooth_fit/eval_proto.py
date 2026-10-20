"""eval_proto.py: evaluate one smooth-shape prototype with one command.

    PY=/tmp/claude-0/venv/bin/python
    $PY /tmp/claude-0/sdf/smooth/eval_proto.py -- --proto DIR --tag NAME [--fast] [--set 'Name=value' ...] [--blend F]

  --proto DIR   your prototype folder; it holds your mod_shape.py (and any helper files it imports).
                A private copy DIR/proto.blend of /tmp/claude-0/sdf/smooth/base.blend is made on first use
                (base.blend itself is never written); pass --blend to use another private copy instead.
  --tag NAME    run name: results in DIR/runs/NAME
  --fast        fast loop: stage output + R2/R4 only, no wrap angles / internal clearances, no render (~35 s)
  --set         set a 'hand fit' modifier input by name (repeatable), e.g. --set 'My Param=3.5'
  --save        save the rebuilt stage into the private copy (DIR/proto.blend), e.g. to hand it over

Steps (all in one process): open the private copy -> rebuild the stage from DIR/mod_shape.py + the
smooth/stage modules (fill, carve, seats OFF; thenar ON) -> evaluate the case (stage + 9 case parts +
cutouts + internal objects, one depsgraph evaluation) -> measure (lib/measure_case.py: gripmetrics +
smooth/ extension vs the no-fit baseline) -> render the standard 3 x 3 sheet (change colour views, hand,
underside, R2 / R4 close-ups, two curvature views). Prints the acceptance table at the end.
Outputs: DIR/runs/NAME/report.json (keys 'skin', 'parts', 'smooth', 'acceptance', 'summary*'), sheet.jpg,
view_*.png, stage.npz, curvature.npz, parts/, ...
"""
import os
import sys
import json
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, 'base.blend')
sys.path.insert(0, os.path.join(ROOT, 'stage'))


def main(argv):
    o = {'proto': None, 'tag': None, 'fast': False, 'sets': [], 'blend': None, 'save': False}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--proto': o['proto'] = os.path.abspath(argv[i + 1]); i += 2
        elif a == '--tag': o['tag'] = argv[i + 1]; i += 2
        elif a == '--fast': o['fast'] = True; i += 1
        elif a == '--set': o['sets'].append(argv[i + 1]); i += 2
        elif a == '--blend': o['blend'] = os.path.abspath(argv[i + 1]); i += 2
        elif a == '--save': o['save'] = True; i += 1
        else:
            raise SystemExit(f'unknown argument {a}\n{__doc__}')
    if not o['proto'] or not o['tag']:
        raise SystemExit(__doc__)
    if not os.path.exists(os.path.join(o['proto'], 'mod_shape.py')):
        raise SystemExit(f"{o['proto']}/mod_shape.py not found (the prototype's module)")
    blend = o['blend'] or os.path.join(o['proto'], 'proto.blend')
    assert os.path.abspath(blend) != BASE, 'use a private copy, not base.blend'
    if not os.path.exists(blend):
        shutil.copy2(BASE, blend)
        print('PRIVATE COPY', blend, flush=True)
    args = ['--blend', blend, '--modules-dir', o['proto'], '--out-root', os.path.join(o['proto'], 'runs'), '--tag', o['tag']]
    for s in o['sets']:
        args += ['--set', s]
    if o['fast']:
        args += ['--skin-only', '--quick', '--no-render']
    if o['save']:
        args += ['--save']
    import eval_all
    eval_all.main(args)
    rep = json.load(open(os.path.join(o['proto'], 'runs', o['tag'], 'report.json')))
    A = rep.get('acceptance', {})
    print('\nACCEPTANCE TABLE (report.json > acceptance)')
    for k, v in A.items():
        if k.startswith('_'):
            continue
        mark = {True: 'ok  ', False: 'FAIL', None: 'info'}[v['pass']]
        if v['pass'] is False and v['kind'] == 'soft':
            mark = 'soft'
        print(f"  {mark} {k}: {v['value']}  [{v['limit']}]")
    print(rep.get('summary3', ''))
    print('RUN', os.path.join(o['proto'], 'runs', o['tag']))


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
