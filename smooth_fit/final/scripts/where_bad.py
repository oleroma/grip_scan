"""where_bad.py RUN [RUN...]: area of each smoothness failure in the judged region, split by location."""
import sys, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, '/tmp/claude-0/sdf/smooth/lib')
import gripmetrics as GM
t = np.load('/tmp/claude-0/sdf/smooth/targets.npz')
nf = np.load('/tmp/claude-0/sdf/smooth/baseline/nofit/curvature.npz')
fb = nf['mask'] & ((nf['k2'] < -1/12) | (nf['k1'] > 1/3) | (nf['kink_concave'] > 10) | (nf['kink_convex'] > 25))
tb = cKDTree(nf['V']); fd = np.zeros(len(fb), bool)
for l in tb.query_ball_point(nf['V'][fb], 2.0, workers=2): fd[l] = True
caps = list(zip(t['caps_a'], t['caps_b'], t['caps_r']))
def capdist(P):
    best = np.full(len(P), 1e9)
    for a, b, r in caps:
        ab = b - a; s = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
        best = np.minimum(best, np.linalg.norm(P - (a + s[:, None] * ab), axis=1) - r)
    return best
for run in sys.argv[1:]:
    z = np.load(run + '/curvature.npz'); V = z['V']; ed = z['edited']
    st = np.load(run + '/stage.npz'); T = st['T'] if 'T' in st else None
    A = GM.vert_area(V, T) if T is not None and len(V) == st['V'].shape[0] else np.ones(len(V))
    tr = cKDTree(V); dil = np.zeros(len(V), bool)
    for l in tr.query_ball_point(V[ed], 3.0, workers=2): dil[l] = True
    _, jn = tb.query(V, workers=2)
    J = dil & z['mask'] & (V[:, 0] >= 0) & ~((~ed) & fd[jn])
    ds = np.minimum(np.linalg.norm(V - t['seat_R2_center'], axis=1), np.linalg.norm(V - t['seat_R4_center'], axis=1))
    dc = capdist(V)
    loc = np.where(ds < 14, 'seats', np.where(V[:, 2] > 34, 'z>34', np.where(dc < 2.5, 'contact', np.where(fd[jn], 'feature-adj', 'free'))))
    tot = A[J].sum()
    print(run, 'judged area', round(tot), 'mm2')
    for nm, b in [('R<8', z['k2'] < -1/8), ('R<12', z['k2'] < -1/12), ('cvxR<3', z['k1'] > 1/3), ('kcc>10', z['kink_concave'] > 10), ('kcv>25', z['kink_convex'] > 25)]:
        bb = b & J
        parts = {k: round(100 * A[bb & (loc == k)].sum() / tot, 2) for k in ('seats', 'z>34', 'contact', 'feature-adj', 'free')}
        print(f'  {nm:7s} {100*A[bb].sum()/tot:6.2f}%  ', parts)
