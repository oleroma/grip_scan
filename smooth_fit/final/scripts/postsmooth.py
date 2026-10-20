"""postsmooth.py SRC_TAG SIGMA OUT_TAG: Gaussian-smooth a solved offset h (same sheet), quick metrics."""
import sys, json, numpy as np, scipy.sparse as sp
from scipy.spatial import cKDTree
sys.path.insert(0, '/tmp/claude-0/sdf/smooth/p_dense/v2'); sys.path.insert(0, '/tmp/claude-0/sdf/smooth/p_dense')
import shape_solver as SS, solve_dev as SD, gripmetrics as GM
src, sig, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
d = dict(np.load(f'/tmp/claude-0/sdf/smooth/final/solve/{src}.npz'))
V, T, D, h = d['V'], d['T'], d['D'], d['h']
tree = cKDTree(V)
dd, jj = tree.query(V, k=64, distance_upper_bound=2.5 * sig, workers=2)
ok = np.isfinite(dd); jj = np.where(ok, jj, 0)
w = np.exp(-0.5 * (np.where(ok, dd, 0) / sig) ** 2) * ok * (np.einsum('nkc,nc->nk', D[jj], D) > 0.5)
w /= w.sum(1, keepdims=True)
hs = (w * h[jj]).sum(1)
hs[h == 0] = 0.0          # frozen / fixed stays exactly 0
P = V + hs[:, None] * D
caps, seats, keep = SS.load_targets()
press = {k: {'center': v['center'], 'normal': v['normal']} for k, v in seats.items()}
q = SD.quick(P, T, caps, press, {k: v['cutter'] for k, v in seats.items()})
q['smooth'], _ = SD.smooth_quick(P, T, hs)
q['solve_s'] = 0; q['h'] = (round(float(hs.min()), 2), round(float(hs.max()), 2))
d.update(h=hs, P=P)
np.savez(f'/tmp/claude-0/sdf/smooth/final/solve/{out}.npz', **d)
json.dump(dict(opt={'postsmooth_of': src, 'sigma': sig}, q=q), open(f'/tmp/claude-0/sdf/smooth/final/solve/{out}.json', 'w'), indent=1, default=str)
print('RESULT', out, json.dumps(q)[:300])
