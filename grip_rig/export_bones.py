"""export_bones.py: rest matrices of the 'grip rig' bones, and a numpy-vs-Blender check of a random pose.

    $PY export_bones.py [--rig .../grip_rig.blend] [--work /tmp/claude-0/hs/work/grip_rig]

Writes <work>/bones.npz (names, parent index, matrix_local 4x4 per bone) and prints the largest
difference between solve_core's simulated basemesh and Blender's evaluated basemesh for a random pose.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import solve_core as sc  # noqa: E402

import bpy  # noqa: E402
from check_demo import eval_mesh, open_grip, scratch  # noqa: E402


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", default="/tmp/claude-0/hs/work/grip_rig/grip_rig.blend")
    ap.add_argument("--work", default="/tmp/claude-0/hs/work/grip_rig")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return ap.parse_args(argv)


def x_matrix(cage):
    me = cage.data
    co0 = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co0)
    co0 = co0.reshape(-1, 3)
    dg = bpy.context.evaluated_depsgraph_get()
    base, _, _ = eval_mesh(cage, dg, world=False)
    Ax = np.zeros((len(base), len(co0)))
    for i in range(len(co0)):
        if abs(co0[i, 0]) < 1e-4:          # mirror seam: X is locked by the rig (and moving it unmerges the seam)
            continue
        c = co0.copy()
        c[i, 0] += 1.0
        me.vertices.foreach_set("co", c.ravel())
        me.update()
        dg = bpy.context.evaluated_depsgraph_get()
        V, _, _ = eval_mesh(cage, dg, world=False)
        Ax[:, i] = V[:, 0] - base[:, 0]
    me.vertices.foreach_set("co", co0.ravel())
    me.update()
    Ax[np.abs(Ax) < 1e-5] = 0.0
    return Ax


def main():
    a = args()
    open_grip(a.rig)
    rig = bpy.data.objects[sc.RIG]
    names = [b.name for b in rig.data.bones]
    parent = [names.index(b.parent.name) if b.parent else -1 for b in rig.data.bones]
    B = np.array([np.array(b.matrix_local) for b in rig.data.bones])
    np.savez(os.path.join(a.work, "bones.npz"), names=np.array(names), parent=np.array(parent), B=B,
             length=np.array([b.length for b in rig.data.bones]), rig_world=np.array(rig.matrix_world))
    # X needs its own subdivision matrix: the mirrored copy of a cage vertex moves the other way in X,
    # so near the mirror seam X is not A @ cage (A was measured with Z moves)
    scratch([sc.CAGE])
    if not os.path.exists(os.path.join(a.work, "Ax.npy")):
        np.save(os.path.join(a.work, "Ax.npy"), x_matrix(bpy.data.objects[sc.CAGE]))
    # random pose: numpy vs Blender
    model = sc.Model(a.work)
    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, model.n) * model.scale * 0.8
    scratch([sc.CAGE, sc.RIG])
    model.apply_to_rig(rig, x)
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    S_b, _, _ = eval_mesh(bpy.data.objects[sc.CAGE], dg)
    S_n = model.surface(x)
    mv = np.linalg.norm(S_n - model.S0, axis=1)
    print("[export] pose", np.round(x, 3).tolist())
    print("[export] largest move %.2f mm; numpy vs Blender max diff %.5f mm" % (mv.max(), np.abs(S_b - S_n).max()))


if __name__ == "__main__":
    main()
