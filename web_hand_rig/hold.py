"""Hold-pose solver for the MakeHuman-based hand on the grip (numpy LBS, scipy least squares).
python hold.py solve OUT.json [--palm_flat W] [--arch SIG] ; the pose is stored as posed world matrices per bone."""
import json, sys, time, numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares
S = '/tmp/claude-0/-home-claude/f25e5681-7fdb-5318-adab-0634023a58c3/scratchpad/'
rest = np.load(S + 'work/hand_rest.npz'); sub = np.load(S + 'work/hand_sub1.npz')
NAMES = [str(b) for b in rest['bones']]
PARENT = {'lowerarm01.R': None, 'lowerarm02.R': 'lowerarm01.R', 'wrist.R': 'lowerarm02.R', 'finger1-1.R': 'wrist.R'}
for i in range(1, 5): PARENT[f'metacarpal{i}.R'] = 'wrist.R'
for f in range(1, 6):
    PARENT.setdefault(f'finger{f}-1.R', f'metacarpal{f-1}.R'); PARENT[f'finger{f}-2.R'] = f'finger{f}-1.R'; PARENT[f'finger{f}-3.R'] = f'finger{f}-2.R'
ORDER = ['lowerarm01.R', 'lowerarm02.R', 'wrist.R'] + [f'metacarpal{i}.R' for i in range(1, 5)] + [f'finger{f}-{k}.R' for f in range(1, 6) for k in range(1, 4)]
PN = np.array([0, 0, 1.0])  # palm normal of the rest (Olex canonical) frame
def rest_frame(h, t):
    y = t - h; y /= np.linalg.norm(y); z = -PN - (-PN @ y) * y; z /= np.linalg.norm(z); x = np.cross(y, z)
    M = np.eye(4); M[:3, :3] = np.stack([x, y, z], 1); M[:3, 3] = h; return M
M = {b: rest_frame(rest['heads'][i], rest['tails'][i]) for i, b in enumerate(NAMES)}
Minv = {b: np.linalg.inv(M[b]) for b in NAMES}
REL = {b: (Minv[PARENT[b]] @ M[b]) if PARENT[b] else M[b] for b in NAMES}
Vs = sub['V']; Ws = sub['W'][:, [list(map(str, sub['names'])).index(b) for b in NAMES]]
Fs = sub['F']
SEG = {b: Ws[:, NAMES.index(b)] for b in NAMES}
own = Ws.argmax(1)
def verts_of(*bs, thr=0.5):
    idx = [NAMES.index(b) for b in bs]; return np.where(np.isin(own, idx) & (Ws[:, idx].sum(1) > thr))[0]
TIP = {'index_md': verts_of('finger2-2.R', 'finger2-3.R'), 'index': verts_of('finger2-3.R'), 'middle': verts_of('finger3-3.R'), 'ring': verts_of('finger4-3.R'), 'pinky': verts_of('finger5-3.R')}
PALMV = verts_of('wrist.R', 'metacarpal1.R', 'metacarpal2.R', 'metacarpal3.R', 'metacarpal4.R')
PALMSIDE = PALMV[(Vs[PALMV, 2] > 4) & (Vs[PALMV, 1] > 15) & (Vs[PALMV, 1] < 75)]  # palm (front) skin, +Z side
# ---- grip as dense surface samples with normals
gd = np.load(S + 'work/grip.npz'); Vg, Fg = gd['V'], gd['F']
def sample_surface(V, F, step=0.35):
    pts = []; nrm = []
    for f in F:
        a, b, c = V[f]; n = np.cross(b - a, c - a); A = np.linalg.norm(n) / 2
        if A < 1e-9: continue
        n = n / (2 * A); k = max(1, int(np.ceil(np.sqrt(2 * A) / step)))
        u, v = np.meshgrid(np.arange(k + 1), np.arange(k + 1)); m = (u + v) <= k
        u, v = u[m] / k, v[m] / k; p = a + np.outer(u, b - a) + np.outer(v, c - a)
        pts.append(p); nrm.append(np.repeat(n[None], len(p), 0))
    return np.concatenate(pts), np.concatenate(nrm)
GP, GN = sample_surface(Vg, Fg); TREE = cKDTree(GP)
def sdf(P):
    d, i = TREE.query(P, workers=-1); s = np.sign(np.einsum('ij,ij->i', P - GP[i], GN[i])); s[s == 0] = 1; return d * s
MK = json.load(open(S + 'work/markers.json'))
# R1's marker sits 4.6 mm under the face (button centre): use the nearest grip-surface point as the touch target
MK_RAW = dict(MK); MK = {k: GP[TREE.query(np.array(v))[1]].tolist() for k, v in MK.items()}
# ---- v2 transfer
v2 = json.load(open(S + 'work/v2_mats.json')); MW = np.array(v2['mw'])
MAP = {'wrist.R': 'palm', 'finger1-1.R': 'thumb_metacarpal', 'finger1-2.R': 'thumb_proximal', 'finger1-3.R': 'thumb_distal'}
for i, n in enumerate(['index', 'middle', 'ring', 'pinky']):
    MAP[f'metacarpal{i+1}.R'] = f'{n}_metacarpal'
    for k, s in enumerate(['proximal', 'middle', 'distal']): MAP[f'finger{i+2}-{k+1}.R'] = f'{n}_{s}'
def v2_world(b):
    return MW @ np.array(v2[b]['pose'])
def v2_delta(b):
    return (MW @ np.array(v2[b]['pose']) @ np.linalg.inv(np.array(v2[b]['rest'])))[:3, :3]
V2J = {}
for b, n in MAP.items():
    P = v2_world(n); V2J[b] = (P[:3, 3], (P @ np.array([0, np.linalg.norm(np.array(v2[n]['tail']) - np.array(v2[n]['head'])), 0, 1]))[:3])
# ---- forward kinematics
def fk(place, L):
    """place: 4x4 world matrix applied to the root (lowerarm01 rest); L: dict bone -> 3x3 local rot."""
    P = {}
    for b in ORDER:
        T = np.eye(4); T[:3, :3] = L.get(b, np.eye(3))
        P[b] = (place @ M[b] @ T) if PARENT[b] is None else (P[PARENT[b]] @ REL[b] @ T)
    return P
VH = np.c_[Vs, np.ones(len(Vs))]
SKIDX = [(i, b, np.where(Ws[:, i] > 0)[0]) for i, b in enumerate(NAMES)]
SKIDX = [(i, b, m, Ws[m, i][:, None], VH[m]) for i, b, m in SKIDX if len(m)]
def skin(P, V=None, W=None):
    if V is not None:
        out = np.zeros_like(V); Vh = np.c_[V, np.ones(len(V))]
        for i, b in enumerate(NAMES):
            w = W[:, i]; m = w > 0
            if m.any(): out[m] += w[m, None] * (Vh[m] @ (P[b] @ Minv[b]).T)[:, :3]
        return out
    out = np.zeros_like(Vs)
    for i, b, m, w, vh in SKIDX:
        out[m] += w * (vh @ (P[b] @ Minv[b])[:3].T)
    return out
def initial():
    """Posed world rotations = v2 world delta applied to our rest frames; heads by FK from the v2 wrist."""
    Rw = {}
    for b in ORDER:
        D = v2_delta(MAP.get(b, 'palm')); Rw[b] = D @ M[b][:3, :3]
    L = {}
    for b in ORDER:
        p = PARENT[b]
        if p is None: continue
        Rpar = Rw[p] @ REL[b][:3, :3]; L[b] = Rpar.T @ Rw[b]
    place = np.eye(4); place[:3, :3] = Rw['lowerarm01.R'] @ M['lowerarm01.R'][:3, :3].T
    P0 = fk(place, L); off = V2J['wrist.R'][0] - P0['wrist.R'][:3, 3]
    place[:3, 3] += off
    return place, L
PLACE0, L0 = initial()
FREE = [b for b in ORDER if b not in ('lowerarm01.R', 'lowerarm02.R')]
FINGERS = [2, 3, 4, 5]
NX = 6 + 3 + 9 + 4 * len(FINGERS)
def RX(a): return Rot.from_euler('x', a).as_matrix()
def RZ(a): return Rot.from_euler('z', a).as_matrix()
def unpack(x):
    rv, t = x[:3], x[3:6]
    place = np.eye(4); R = Rot.from_rotvec(rv).as_matrix()
    c = V2J['wrist.R'][0]
    place[:3, :3] = R @ PLACE0[:3, :3]; place[:3, 3] = R @ (PLACE0[:3, 3] - c) + c + t
    L = dict(L0)
    L['wrist.R'] = L0['wrist.R'] @ Rot.from_rotvec(x[6:9]).as_matrix()
    for i in range(1, 5): L[f'metacarpal{i}.R'] = np.eye(3)  # flat palm: metacarpals stay as in the open hand
    for k in range(3): L[f'finger1-{k+1}.R'] = L0[f'finger1-{k+1}.R'] @ Rot.from_rotvec(x[9 + 3 * k: 12 + 3 * k]).as_matrix()
    for j, f in enumerate(FINGERS):
        mf, ms, pf, df = x[18 + 4 * j: 22 + 4 * j]
        L[f'finger{f}-1.R'] = L0[f'finger{f}-1.R'] @ RX(mf) @ RZ(ms)
        L[f'finger{f}-2.R'] = L0[f'finger{f}-2.R'] @ RX(pf)
        L[f'finger{f}-3.R'] = L0[f'finger{f}-3.R'] @ RX(df)
    return place, L
def palm_normal(P, X):
    """normal of the plane fitted to the posed palm-side skin, pointing out of the palm."""
    Q = X[PALMSIDE]; c = Q.mean(0); _, _, vt = np.linalg.svd(Q - c); n = vt[2]
    ref = P['wrist.R'][:3, :3] @ np.array([0, 0, -1.0])  # bone -Z = palm side
    return n if n @ ref > 0 else -n
def softmin(d, k=0.5):
    return -k * np.log(np.exp(-(d - d.min()) / k).sum()) + d.min()
OPT = dict(palm_flat=8.0, arch_sig=3.0, palm_vert=6.0)
def residuals(x, info=False):
    place, L = unpack(x); P = fk(place, L); X = skin(P)
    sd = sdf(X); r = []
    pen = np.maximum(0, -sd - 0.4); r.append(4 * pen * 0.35)  # per-vertex penetration (free 0.4 mm squash)
    for mk, f in [('R1 target', 'index_md'), ('R2 press', 'middle'), ('R4 press', 'ring')]:
        d = np.linalg.norm(X[TIP[f]] - MK[mk], axis=1); r.append([3 * (softmin(d) - 0.3)])
    d = np.linalg.norm(X[PALMV] - MK['palm swell'], axis=1); r.append([3 * (softmin(d) - 0.3)])
    r.append([2 * (softmin(sd[TIP['pinky']]) - 0.3)])
    # thumb: keep the v2 thumb tip / IP (sigma 4 mm)
    for b in ['finger1-2.R', 'finger1-3.R']:
        r.append((P[b][:3, 3] - V2J[b][0]) / 4.0)
    tt = (P['finger1-3.R'] @ np.array([0, np.linalg.norm(rest['tails'][NAMES.index('finger1-3.R')] - rest['heads'][NAMES.index('finger1-3.R')]), 0, 1]))[:3]
    r.append((tt - V2J['finger1-3.R'][1]) / 4.0)
    # every joint near v2 (sigma 15 mm)
    for b in FREE: r.append((P[b][:3, 3] - V2J.get(b, V2J['wrist.R'])[0]) / 15.0 * (b in V2J))
    # palm like the side of a box: palm normal horizontal (no roll toward the floor/ceiling)
    n = palm_normal(P, X); r.append([OPT['palm_vert'] * n[2]])
    # finger joint priors: change vs the transferred v2 pose (sigma 20 deg flexion, 8 deg spread), limits on the total flexion
    r.append(np.degrees(x[18:]) / np.tile([20.0, 8.0, 20.0, 20.0], len(FINGERS)))
    r.append(np.degrees(x[9:18]) / 10.0)
    for b in FREE:
        if b.startswith(('metacarpal', 'finger1', 'wrist')): continue
        e = Rot.from_matrix(L[b]).as_euler('XYZ', degrees=True)
        fl = -e[0]  # flexion toward the palm = -X in these bone frames (Z points to the back of the hand)
        r.append([max(0, -15 - fl) / 2, max(0, fl - 100) / 2])
    # wrist relative to forearm small
    r.append(np.degrees(Rot.from_matrix(L['wrist.R']).as_rotvec()) / 25.0)
    res = np.concatenate([np.atleast_1d(np.asarray(a, float)).ravel() for a in r])
    if info == 'parts':
        return [float(0.5 * (np.atleast_1d(np.asarray(a, float)) ** 2).sum()) for a in r]
    if info:
        return dict(X=X, P=P, sd=sd, n=n)
    return res
def report(x):
    I = residuals(x, info=True); X, sd, n, P = I['X'], I['sd'], I['n'], I['P']
    out = {'max_penetration_mm': float(max(0, -sd.min())), 'penetrating_verts_over_0.5mm': int((sd < -0.5).sum()),
           'palm_normal': n.round(3).tolist(), 'palm_tilt_from_vertical_deg': float(np.degrees(np.arcsin(abs(n[2])))),
           'palm_faces_deg_from_minusX_in_plan': float(np.degrees(np.arctan2(n[1], -n[0])))}
    for mk, f in [('R1 target', 'index_md'), ('R2 press', 'middle'), ('R4 press', 'ring')]:
        out[f'{mk}_to_{f}_tip_mm'] = float(np.linalg.norm(X[TIP[f]] - MK[mk], axis=1).min())
    out['palm_swell_to_palm_mm'] = float(np.linalg.norm(X[PALMV] - MK['palm swell'], axis=1).min())
    out['pinky_tip_gap_mm'] = float(sd[TIP['pinky']].min())
    out['arch_deg'] = {f'metacarpal{i}': float(np.degrees(np.linalg.norm(Rot.from_matrix(unpack(x)[1][f'metacarpal{i}.R']).as_rotvec()))) for i in range(1, 5)}
    out['angles_deg'] = {b: [round(v, 1) for v in Rot.from_matrix(unpack(x)[1][b]).as_euler('XYZ', degrees=True)] for b in FREE}
    return out
if __name__ == '__main__':
    cmd, outp = sys.argv[1], sys.argv[2]
    x0 = np.zeros(NX)
    if '--start' in sys.argv: x0 = np.array(json.load(open(sys.argv[sys.argv.index('--start') + 1]))['x'])
    for k in OPT:
        if '--' + k in sys.argv: OPT[k] = float(sys.argv[sys.argv.index('--' + k) + 1])
    if cmd == 'report':
        print(json.dumps(report(x0), indent=1)); sys.exit()
    t0 = time.time(); r0 = residuals(x0); print('start cost', 0.5 * (r0 ** 2).sum(), 'n res', len(r0), json.dumps({k: v for k, v in report(x0).items() if k != 'angles_deg'}))
    nfev = int(sys.argv[sys.argv.index('--nfev') + 1]) if '--nfev' in sys.argv else 60
    sol = least_squares(residuals, x0, method='trf', diff_step=1e-3, max_nfev=nfev, x_scale=0.05, jac='2-point', verbose=1)
    rep = report(sol.x); print('cost', sol.cost, 'secs', round(time.time() - t0), json.dumps({k: v for k, v in rep.items() if k != 'angles_deg'}))
    place, L = unpack(sol.x); P = fk(place, L)
    json.dump({'x': sol.x.tolist(), 'opt': OPT, 'report': rep, 'posed_world': {b: P[b].tolist() for b in ORDER}}, open(outp, 'w'), indent=1)
