"""Re-check gripmetrics against the verified reference numbers.
    /tmp/claude-0/venv/bin/python /tmp/claude-0/sdf/smooth/lib/validate.py
Exit code 0 = all reference numbers reproduced (to 0.01 mm)."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gripmetrics as GM  # noqa: E402

VC = '/tmp/claude-0/sdf/V_C_hand_sdf/out'
KEYS = ('middle root pad', 'middle PIP', 'ring root pad', 'ring PIP', 'pinky root pad', 'pinky PIP', 'pinky distal')
REF = {'rig_v27': ([0.35, -0.18, 0.47, 0.91, 1.07, -0.13, -0.09], -1.62, 0.0),
       'default': ([-0.36, -0.18, -0.51, -0.28, -0.51, -0.52, -0.53], -0.51, 6.0)}
caps = GM.load_caps()
ok = True
for tag, (ref, palm, genus) in REF.items():
    V, T, _ = GM.load_mesh(f'{VC}/{tag}.npz')
    T, _ = GM.orient(V, T)
    b = GM.bvh_of(V, T)
    c, _ = GM.contacts(b, caps)
    got = [c[k] for k in KEYS]
    g = GM.topology(V, T)['genus']
    good = np.allclose(got, ref, atol=0.011) and abs(c['palm on swell'] - palm) < 0.011 and g == genus
    ok &= good
    print(f"{tag:8s} {'OK ' if good else 'BAD'} contacts {got} palm {c['palm on swell']} genus {g}   (ref {ref} palm {palm} genus {genus})")
# hand vs today's trigger parts (critic: middle distal 2.55 mm into R2, ring distal 2.02 mm into R4)
tr = {}
for p in ('R2', 'R4'):
    V, T, _ = GM.load_mesh(f'/tmp/claude-0/sdf/B_eval/out/base_{p}.npz')
    tr[p] = (V, T)
r = GM.trigger_report(tr, caps)
good = abs(r['R2']['max_hand_overlap_mm'] - 2.55) < 0.011 and abs(r['R4']['max_hand_overlap_mm'] - 2.02) < 0.011
ok &= good
print(f"triggers {'OK ' if good else 'BAD'} R2 {r['R2']['max_hand_overlap_mm']} ({r['R2']['worst_capsule']}) R4 {r['R4']['max_hand_overlap_mm']} ({r['R4']['worst_capsule']})   (ref 2.55 / 2.02)")
sys.exit(0 if ok else 1)
