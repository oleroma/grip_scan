"""Retarget the MakeHuman right hand to Olex's measured skeleton (fitted/joints.json, open pose,
canonical frame: wrist origin, fingers +Y, palm normal +Z, thumb +X) and measured skin sections.
Output work/hand_rest.npz: V (mm, Olex frame), F (quads/tris as list), W (n x bones), bones, heads, tails."""
import json, sys, numpy as np
sys.path.insert(0, sys.path[0])
from mh_load import load
OUT = sys.argv[1] if len(sys.argv) > 1 else 'work/hand_rest.npz'
V, G, J, Wm, tot, ours, bones = load()
O = json.load(open('/mnt/project-files/hand_scan_rig/fitted/joints.json'))['landmarks']
SEC = json.load(open('/mnt/project-files/hand_scan_rig/fitted/fitted_params.json'))['sections_mm']['fitted']
FING = ['thumb', 'index', 'middle', 'ring', 'pinky']
JN = ['mcp', 'pip', 'dip', 'tip']; TJ = ['cmc', 'mcp', 'ip', 'tip']
def mhj(f, k):  # finger idx 0..4, joint idx 0..3
    return J[f'joint-r-finger-{f+1}-{k+1}']
def olj(f, k):
    return np.array(O[f"{FING[f]}_{(TJ if f == 0 else JN)[k]}"], float)
# --- global similarity MH -> Olex from palm points
src = np.array([J['joint-r-hand']] + [mhj(f, k) for f in range(1, 5) for k in (0, 1)])
dst = np.array([O['wrist']] + [olj(f, k) for f in range(1, 5) for k in (0, 1)])
def similarity(a, b):
    ma, mb = a.mean(0), b.mean(0); A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A); d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt; s = (S * [1, 1, d]).sum() / (A ** 2).sum()
    return s, R, mb - s * R @ ma
s, R, t = similarity(src, dst)
S = lambda p: (s * (R @ np.asarray(p).T)).T + t
print('similarity scale', round(s, 4), 'palm fit rms mm', np.sqrt(((S(src) - dst) ** 2).sum(1).mean()).round(2))
V1 = S(V); J1 = {k: S(v) for k, v in J.items()}
# --- bone list: name -> (src head, src tail, dst head, dst tail)
B = {}
B['lowerarm02.R'] = (J1['joint-r-elbow'] * 0 + (J1['joint-r-hand'] + (J1['joint-r-elbow'] - J1['joint-r-hand']) * 0.5), J1['joint-r-hand'], None, None)
B['lowerarm01.R'] = (J1['joint-r-elbow'], B['lowerarm02.R'][0], None, None)
B['wrist.R'] = (J1['joint-r-hand'], J1['joint-r-hand-3'], np.array(O['wrist'], float), None)
for i, f in enumerate(range(1, 5)):
    h = J1['joint-r-hand-3'] if f <= 2 else J1['joint-r-hand-2']
    B[f'metacarpal{i+1}.R'] = (h, J1[f'joint-r-finger-{f+1}-1'], None, olj(f, 0))
for f in range(5):
    for k in range(3):
        B[f'finger{f+1}-{k+1}.R'] = (J1[f'joint-r-finger-{f+1}-{k+1}'], J1[f'joint-r-finger-{f+1}-{k+2}'], olj(f, k), olj(f, k + 1))
# forearm: straight continuation of the middle metacarpal (neutral wrist), MH lengths
ax = olj(2, 0) - np.array(O['wrist'], float); ax[2] = 0; ax /= np.linalg.norm(ax)
wr = np.array(O['wrist'], float)
L2 = np.linalg.norm(B['lowerarm02.R'][1] - B['lowerarm02.R'][0]); L1 = np.linalg.norm(B['lowerarm01.R'][1] - B['lowerarm01.R'][0])
B['lowerarm02.R'] = (B['lowerarm02.R'][0], B['lowerarm02.R'][1], wr - ax * L2, wr.copy())
B['lowerarm01.R'] = (B['lowerarm01.R'][0], B['lowerarm01.R'][1], wr - ax * (L1 + L2), wr - ax * L2)
# non-measured ends: keep the similarity-mapped positions
for b, (hs, ts, hd, td) in list(B.items()):
    B[b] = (hs, ts, hs.copy() if hd is None else hd, ts.copy() if td is None else td)
# --- palm frame in Olex coords: normal +Z, lateral +X
NZ = np.array([0, 0, 1.0])
def frame(h, tl, nrm):
    y = tl - h; L = np.linalg.norm(y); y /= L
    z = nrm - (nrm @ y) * y; z /= np.linalg.norm(z); x = np.cross(y, z)
    return np.stack([x, y, z]), L
# palm normal in MH after similarity: R @ (MH normal) == Olex normal approx -> use NZ for both
names = list(B)
Wmat = np.zeros((len(V), len(names)))
for v, d in Wm.items():
    for b, w in d.items(): Wmat[v, names.index(b)] = w
keep_v = np.where(ours > 0.6)[0]
# sections: ratios of cross-section along each finger bone (u = 0 at head, 1 at tail)
def map_vertices(scales):
    out = np.zeros_like(V1); wsum = Wmat.sum(1, keepdims=True); wsum[wsum == 0] = 1
    for bi, b in enumerate(names):
        w = Wmat[:, bi]; m = w > 0
        if not m.any(): continue
        hs, ts, hd, td = B[b]
        Fs, Ls = frame(hs, ts, NZ); Fd, Ld = frame(hd, td, NZ)
        loc = (V1[m] - hs) @ Fs.T  # x,y,z local
        u = np.clip(loc[:, 1] / Ls, 0, 1)
        sx0, sx1, sz0, sz1 = scales.get(b, (1, 1, 1, 1))
        sx = sx0 + (sx1 - sx0) * u; sz = sz0 + (sz1 - sz0) * u
        nl = np.stack([loc[:, 0] * sx, loc[:, 1] * Ld / Ls, loc[:, 2] * sz], 1)
        out[m] += w[m, None] * (hd + nl @ Fd)
    return out / wsum
def sections(Vm, Bmap):
    """width (x) / thickness (z) of the skin at stations, in the bone frame."""
    res = {}
    def sec(b, u, key):
        hs, ts, hd, td = Bmap[b]; Fd, Ld = frame(hd, td, NZ)
        bi = names.index(b); m = np.ones(len(Vm), bool)
        loc = (Vm[m] - hd) @ Fd.T; sel = (np.abs(loc[:, 1] - min(max(u * Ld, 3.5), Ld - 3.5)) < 3.5) & (np.hypot(loc[:, 0], loc[:, 2]) < 13)
        if sel.sum() < 6: return
        res['width_' + key] = np.ptp(loc[sel, 0]); res['thick_' + key] = np.ptp(loc[sel, 2])
    for f in range(5):
        n = FING[f]
        if f == 0:
            sec('finger1-2.R', 0.5, 'thumb_proximal_mid'); sec('finger1-3.R', 0.0, 'thumb_ip')
        else:
            sec(f'finger{f+1}-1.R', 0.5, f'{n}_proximal_mid'); sec(f'finger{f+1}-2.R', 0.0, f'{n}_pip'); sec(f'finger{f+1}-3.R', 0.0, f'{n}_dip')
    return res
Bm = {b: (hd, td, hd, td) for b, (hs, ts, hd, td) in B.items()}
scales = {}
for it in range(3):
    Vm = map_vertices(scales); sc = sections(Vm, Bm)
    err = {k: round(sc[k] - SEC[k], 1) for k in sc if SEC.get(k)}
    print('iter', it, 'section error mm', err)
    for f in range(5):
        n = FING[f]
        def r(kind, key):
            k = f'{kind}_{key}'
            return (SEC[k] / sc[k]) if (SEC.get(k) and k in sc) else None
        if f == 0:
            st = [('finger1-2.R', 'thumb_proximal_mid', 'thumb_ip'), ('finger1-3.R', 'thumb_ip', 'thumb_ip')]
        else:
            st = [(f'finger{f+1}-1.R', f'{n}_proximal_mid', f'{n}_pip'), (f'finger{f+1}-2.R', f'{n}_pip', f'{n}_dip'), (f'finger{f+1}-3.R', f'{n}_dip', f'{n}_dip')]
        for b, k0, k1 in st:
            o = scales.get(b, (1, 1, 1, 1))
            rx0 = r('width', k0) or r('width', k1); rx1 = r('width', k1) or r('thick', k1)
            rz0 = r('thick', k0) or 1.0; rz1 = r('thick', k1) or 1.0; rx0 = rx0 or 1.0; rx1 = rx1 or 1.0
            if f == 0:  # thumb: uniform cross-section scale (nail faces sideways)
                g0 = np.sqrt(rx0 * rz0); g1 = np.sqrt(rx1 * rz1); rx0, rx1, rz0, rz1 = g0, g1, g0, g1
            cl = lambda x: float(np.clip(x, 0.75, 1.35))
            scales[b] = (cl(o[0] * rx0), cl(o[1] * rx1), cl(o[2] * rz0), cl(o[3] * rz1))
Vm = map_vertices(scales)
# palm thickness at mid palm: measure along Z at the middle-metacarpal midpoint
mid = 0.5 * (np.array(O['wrist']) + olj(2, 0))
def palm_thick(Vx):
    sel = (np.abs(Vx[:, 1] - mid[1]) < 5) & (np.abs(Vx[:, 0] - mid[0]) < 8) & (ours > 0.6)
    return np.ptp(Vx[sel, 2]) if sel.sum() > 3 else None
def breadth(Vx):
    yrow = np.mean([olj(f, 0)[1] for f in range(1, 5)])
    pb = [names.index(b) for b in ['wrist.R','metacarpal1.R','metacarpal2.R','metacarpal3.R','metacarpal4.R']]
    sel = (np.abs(Vx[:, 1] - yrow + 6) < 4) & (ours > 0.6) & np.isin(Wmat.argmax(1), pb) & (Wmat[:, pb].sum(1) > 0.8)
    return np.ptp(Vx[sel, 0])
sel=(ours>0.6);print('hand bbox',Vm[sel].min(0).round(1),Vm[sel].max(0).round(1),'mid',mid)
pt, br = palm_thick(Vm), breadth(Vm)
print('palm thickness', pt, 'target', SEC['thick_mid_palm'], 'breadth near MCP row', br, 'target', SEC['breadth_mcp_row'])
rz = 1.0  # MH palm already 27-29 mm thick and 86 mm across the MCP row (targets 29.4 / 86.1)
for b in ['wrist.R'] + [f'metacarpal{i}.R' for i in range(1, 5)]:
    scales[b] = (1, 1, rz, rz)
Vm = map_vertices(scales); print('palm thickness after', palm_thick(Vm), 'breadth', breadth(Vm))
dwr = np.linalg.norm(Vm - np.array(O['wrist']), axis=1)
F = [f for f in G['body'] if all(Vm[i, 1] > -62 for i in f) and (all(ours[i] > 0.6 for i in f) or all(tot[i] > 0 and ours[i] > 0.3 for i in f))]
used = np.unique(np.concatenate(F)); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
Fl = [[int(remap[i]) for i in f] for f in F]
heads = np.array([B[b][2] for b in names]); tails = np.array([B[b][3] for b in names])
Wk = Wmat[used]; Wk /= np.maximum(Wk.sum(1, keepdims=True), 1e-9)
np.savez(OUT, V=Vm[used], W=Wk, bones=np.array(names), heads=heads, tails=tails,
         F=np.array([f + [-1] * (4 - len(f)) for f in Fl]), scales=json.dumps({k: list(map(float, v)) for k, v in scales.items()}),
         mh_index=used)
print('verts', len(used), 'faces', len(Fl), 'final sections err', {k: round(v - SEC[k], 1) for k, v in sections(Vm, Bm).items() if SEC.get(k)})
