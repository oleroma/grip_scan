"""Resume driver: dense_fit solve with a warm start (h zeroed on the fixed set), quick metrics, npz + json out."""
import sys, json, os, numpy as np
sys.path.insert(0, '/tmp/claude-0/sdf/smooth/p_dense')        # solve_dev (quick metrics only)
sys.path.insert(0, '/tmp/claude-0/sdf/smooth/final/solver')   # the final solver copy
import dense_fit as DF
import shape_solver as SS
import solve_dev as SD
import gripmetrics as GM
OUT = '/tmp/claude-0/sdf/smooth/final/solve'
os.makedirs(OUT, exist_ok=True)
opt = json.loads(sys.argv[1])
tag = opt.pop('tag')
warm = opt.pop('warm', None)
V, T = SD.load_input()
caps, seats, keep = SS.load_targets()
pr = DF.Fit(V, T, caps, GM.zones(caps), seats, keep, opt)
h0 = None
if warm:
    h0 = np.load(warm)['h'].copy()
    h0[pr.fixed0] = 0.0
h = pr.solve(h0)
P = pr.moved(h)
press = {k: {'center': v['center'], 'normal': v['normal']} for k, v in seats.items()}
q = SD.quick(P, pr.T, caps, press, {k: v['cutter'] for k, v in seats.items()})
q['smooth'], _ = SD.smooth_quick(P, pr.T, h)
q['solve_s'] = round(pr.seconds, 1)
q['h'] = (round(float(h.min()), 2), round(float(h.max()), 2))
np.savez(f'{OUT}/{tag}.npz', h=h, V=V, T=pr.T, D=pr.D, P=P, lock=pr.lock, seat_kind=pr.seat_kind)
json.dump(dict(opt=opt, warm=warm, q=q, hist=pr.history), open(f'{OUT}/{tag}.json', 'w'), indent=1, default=str)
print('RESULT', tag, json.dumps(q), flush=True)
