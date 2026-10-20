"""gather_np.py: numpy-only save / load of the v2 solver inputs (see gather.py)."""
import numpy as np

FINGER_OF = {'R2': 'middle_distal', 'R4': 'ring_distal'}


def save_npz(path, g):
    z = dict(V=g['V'], T=g['T'])
    for k, (a, b, r) in g['caps'].items():
        z['cap|' + k] = np.r_[a, b, [r]]
    for P, t in g['trig'].items():
        for k in ('center', 'normal', 'travel'):
            z[f'trig|{P}|{k}'] = t[k]
        z[f'trig|{P}|travel_mm'] = np.array(t['travel_mm'])
    for n, (Vx, Tx) in list(g['cutters'].items()) + list(g['internals'].items()):
        z['mesh|' + n + '|V'] = Vx; z['mesh|' + n + '|T'] = Tx
    np.savez_compressed(path, **z)


def load_npz(path):
    d = np.load(path)
    g = dict(V=d['V'], T=d['T'], caps={}, trig={}, cutters={}, internals={})
    for k in d.files:
        p = k.split('|')
        if p[0] == 'cap':
            x = d[k]; g['caps'][p[1]] = (x[:3], x[3:6], float(x[6]))
        elif p[0] == 'trig':
            g['trig'].setdefault(p[1], {'capsule': FINGER_OF[p[1]]})[p[2]] = float(d[k]) if p[2] == 'travel_mm' else d[k]
        elif p[0] == 'mesh' and p[2] == 'V':
            tgt = g['cutters'] if p[1].endswith(' bool') and p[1][:2] in ('R2', 'R4') else g['internals']
            tgt[p[1]] = (d[k], d['mesh|' + p[1] + '|T'])
    return g


