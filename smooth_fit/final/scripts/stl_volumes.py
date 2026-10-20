"""stl_volumes.py DIR [OUT.json]: signed volume (> 0 = faces point outward), triangles, open / non-manifold edges of
every binary STL under DIR (vertices welded by exact position, as a slicer does)."""
import os, sys, json
import numpy as np


def load(p):
    with open(p, 'rb') as f:
        f.read(80); n = int(np.frombuffer(f.read(4), '<u4')[0])
        d = np.frombuffer(f.read(n * 50), np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
    P = d['v'].reshape(-1, 3).astype(np.float64)
    U, inv = np.unique(P, axis=0, return_inverse=True)
    return U, inv.reshape(-1, 3)


def check(p):
    V, T = load(p)
    vol = float(np.einsum('ij,ij->i', V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])).sum() / 6.0)
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(E, axis=0, return_counts=True)
    return dict(triangles=int(len(T)), signed_volume_mm3=round(vol, 2), outward=bool(vol > 0),
                open_edges=int((cnt == 1).sum()), nonmanifold_edges=int((cnt > 2).sum()),
                bbox_min=np.round(V.min(0), 3).tolist(), bbox_max=np.round(V.max(0), 3).tolist())


d = sys.argv[1]
res = {}
for root, _, fs in os.walk(d):
    for f in sorted(fs):
        if f.lower().endswith('.stl'):
            p = os.path.join(root, f)
            res[os.path.relpath(p, d)] = check(p)
for k, v in res.items():
    print(f"{k}: vol {v['signed_volume_mm3']} mm3 ({'outward' if v['outward'] else 'INSIDE-OUT'}), {v['triangles']} tris, "
          f"open edges {v['open_edges']}, non-manifold edges {v['nonmanifold_edges']}")
if len(sys.argv) > 2:
    json.dump(res, open(sys.argv[2], 'w'), indent=1)
