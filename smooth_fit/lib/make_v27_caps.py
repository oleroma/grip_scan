"""Write lib/v27_caps.json: hand capsules of Olex's v_27 hold, taken straight from v_27.blend by the
verified V_C_hand_sdf/v_caps.py (bones + skin median radius), plus the palm heel capsule (PCA fit, the
prototype's 'hand points' bone 'palm'; v_27 has no palm skin capsule)."""
import json, numpy as np
c = json.load(open('/tmp/claude-0/sdf/V_C_hand_sdf/out/v27_caps.json'))
hp = np.load('/tmp/claude-0/sdf/V_C_hand_sdf/out/hand_points.npz')
out = {k: {'a': v['a'], 'b': v['b'], 'r': v['r']} for k, v in c.items() if v['r'] is not None}
names = list(hp['names']); i = names.index('palm'); m = hp['bid'] == i
out['palm'] = {'a': hp['P'][m][0].tolist(), 'b': hp['P'][m][-1].tolist(), 'r': float(hp['rad'][m][0])}
json.dump(out, open('/tmp/claude-0/sdf/smooth/lib/v27_caps.json', 'w'), indent=1)
print(len(out), 'capsules', sorted(out))
