"""extract_hand.py: the scanned hand in the approved hold, as plain arrays for the solver.

    $PY extract_hand.py [--hold .../hold_upright/hand_on_grip_upright.blend] [--work /tmp/claude-0/hs/work/grip_rig]

Writes <work>/hand_hold.npz:
  V (n,3) world mm, T (m,3) triangles of the posed 'hand scan' skin,
  seg (n,) index of the hand bone that owns each vertex (largest skin weight), seg_names,
  bones_head / bones_tail (k,3) of the posed 'hand rig', bone_names,
  markers: palm swell, R1 target, R2 press, R4 press (world mm)
"""
import argparse
import os
import sys

import numpy as np
import bpy


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", default="/mnt/project-files/hand_scan_rig/hold_upright/hand_on_grip_upright.blend")
    ap.add_argument("--work", default="/tmp/claude-0/hs/work/grip_rig")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return ap.parse_args(argv)


def main():
    a = args()
    os.makedirs(a.work, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=a.hold)
    print("objects:", sorted(o.name for o in bpy.data.objects))
    hs = bpy.data.objects["hand scan"]
    rig = bpy.data.objects["hand rig"]
    dg = bpy.context.evaluated_depsgraph_get()
    e = hs.evaluated_get(dg)
    me = e.to_mesh()
    me.calc_loop_triangles()
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    T = np.empty(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", T)
    T = T.reshape(-1, 3)
    e.to_mesh_clear()
    M = np.array(hs.matrix_world)
    V = V @ M[:3, :3].T + M[:3, 3]
    # dominant bone per vertex (from the undeformed mesh's vertex groups; same vertex order)
    names = [g.name for g in hs.vertex_groups]
    W = np.zeros((len(hs.data.vertices), len(names)))
    for v in hs.data.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    assert len(W) == len(V), (len(W), len(V))
    bone_cols = [i for i, n in enumerate(names) if n != "cs_mask"]     # cs_mask is a helper group, not a bone
    seg = np.array(bone_cols)[W[:, bone_cols].argmax(1)]
    Mr = rig.matrix_world
    bn = [pb.name for pb in rig.pose.bones]
    bh = np.array([list(Mr @ pb.head) for pb in rig.pose.bones])
    bt = np.array([list(Mr @ pb.tail) for pb in rig.pose.bones])
    mk = {}
    for o in bpy.data.objects:
        if o.name in ("palm swell", "R1 target", "R2 press", "R4 press"):
            mk[o.name] = np.array(o.matrix_world.translation)
    np.savez_compressed(os.path.join(a.work, "hand_hold.npz"), V=V, T=T, seg=seg, seg_names=np.array(names),
                        bones_head=bh, bones_tail=bt, bone_names=np.array(bn),
                        marker_names=np.array(list(mk)), markers=np.array(list(mk.values())))
    print("hand", V.shape, T.shape, "groups", names, "markers", {k: np.round(v, 2).tolist() for k, v in mk.items()})


if __name__ == "__main__":
    main()
