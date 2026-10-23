"""Load the MakeHuman (CC0) base mesh right hand + default-rig weights. Units: MH decimetres -> mm (x100)."""
import json, gzip, numpy as np
D = '/tmp/claude-0/-home-claude/f25e5681-7fdb-5318-adab-0634023a58c3/scratchpad/dl/mpfb2/src/mpfb/data/'
HAND_BONES = ['wrist.R', 'metacarpal1.R', 'metacarpal2.R', 'metacarpal3.R', 'metacarpal4.R'] + \
    [f'finger{i}-{j}.R' for i in range(1, 6) for j in range(1, 4)]
ARM_BONES = ['lowerarm01.R', 'lowerarm02.R']

def load_obj():
    V = []; F = []; G = {}; g = None
    for line in open(D + '3dobjs/base.obj'):
        if line.startswith('v '):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('g '):
            g = line.split()[1]
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            G.setdefault(g, []).append(idx)
    return np.array(V) * 100.0, G

def load():
    V, G = load_obj()
    joints = {g: V[np.unique(np.concatenate(f))].mean(0) for g, f in G.items() if g.startswith('joint-r-')}
    W = json.load(open(D + 'rigs/standard/weights.default.json'))['weights']
    bones = HAND_BONES + ARM_BONES
    Wm = {}
    for b in bones:
        for v, w in W[b]:
            Wm.setdefault(v, {})[b] = w
    # total weight of every vertex (all bones) to know how much is outside our set
    tot = np.zeros(len(V)); ours = np.zeros(len(V))
    for b, lst in W.items():
        for v, w in lst:
            tot[v] += w
            if b in bones: ours[v] += w
    return V, G, joints, Wm, tot, ours, bones

if __name__ == '__main__':
    V, G, J, Wm, tot, ours, bones = load()
    body = G['body']
    print('body faces', len(body), 'joint groups', len(J))
    for k in ['joint-r-hand', 'joint-r-hand-2', 'joint-r-hand-3', 'joint-r-finger-2-1', 'joint-r-finger-2-4', 'joint-r-finger-1-1', 'joint-r-finger-1-4', 'joint-r-finger-3-4', 'joint-r-elbow']:
        print(k, J.get(k))
    hand = np.where(ours > 0.5)[0]; print('verts >0.5 ours', len(hand), 'x range', V[hand].min(0), V[hand].max(0))
