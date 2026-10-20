"""gather.py (final/solver_v2, from v2/solver): read everything the v2 shape solve needs from an OPEN Grip_smooth-style .blend (bpy).

    inputs = gather()          # dict of numpy arrays, see below

Evaluated in a scratch scene with the 'hand fit' modifier switched off (the stage input mesh = Olex's
subdivided GripV basemesh), so nothing here can create a dependency cycle (the file is not changed).
Returned:
    V, T           stage input mesh (world = object space), triangles
    caps           {bone: (a, b, r)} hand capsules of 'hand points' (follows 'hand rig')
    trig           {R2/R4: dict(center, normal, travel (unit), travel_mm, capsule)} press faces and the
                   trigger stroke (from the generator's own 'action' preview)
    cutters        {'R2 bool'/'R4 bool': (V, T)} opening cutters
    internals      {name: (V, T)} internal keep-outs (screw bools, pcb cutout, ...)
    controls       values of the modifier / dashboard controls that exist in the file
"""
import os
import sys
import numpy as np
import bpy

FINAL = '/tmp/claude-0/sdf/smooth/final'     # integrated 2026-10-20: the repaired final stage + shared metrics
for p in (FINAL + '/stage', '/tmp/claude-0/sdf/smooth/lib'):
    if p not in sys.path:
        sys.path.insert(0, p)
import build_stage as BS      # noqa: E402
import eval_all as EA         # noqa: E402
import trigger_solver as TS   # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

INTERNALS = tuple(EA.INTERNALS)   # every internal keep-out the shared metrics measure (incl. 'battery wire space')
FINGER_OF = {'R2': 'middle_distal', 'R4': 'ring_distal'}


def _mesh(name, dg):
    ob = bpy.data.objects[name]
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    M = np.array(ob.matrix_world)
    V = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', V); V = V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    T = np.empty(len(me.loop_triangles) * 3, np.int64); me.loop_triangles.foreach_get('vertices', T); T = T.reshape(-1, 3)
    pf = TS.press_face(me, M) if name in TS.PARTS else None
    ev.to_mesh_clear()
    return V, T, pf


def gather():
    gp = bpy.data.node_groups['global'].nodes['global parameters']
    gp.inputs['variant'].default_value = 'GripV'
    ob = bpy.data.objects[BS.TARGET]
    mod = ob.modifiers[BS.MOD_NAME]
    vals = BS.get_inputs(mod)
    enable0 = vals.get('Enable', True)
    BS.set_inputs(mod, {'Enable': False})
    win = bpy.context.window.scene
    sc = bpy.data.scenes.new('hf v2 gather')
    bpy.context.window.scene = sc
    names = [BS.TARGET, 'hand points', 'hand rig', 'R2', 'R4', 'R2 bool', 'R4 bool'] + [n for n in INTERNALS if n in bpy.data.objects]
    for n in names:
        sc.collection.objects.link(bpy.data.objects[n])
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        V, T, _, _ = EA.mesh_arrays(ob, dg)
        caps = EA.hand_caps(dg)
        trig, cutters, internals = {}, {}, {}
        for P in ('R2', 'R4'):
            _, _, pf = _mesh(P, dg)
            nd = bpy.data.node_groups[TS.PARTS[P]['group']].nodes[TS.PARTS[P]['node']]
            nd.inputs['action'].default_value = True
            try:
                dg = bpy.context.evaluated_depsgraph_get(); dg.update()
                _, _, pp = _mesh(P, dg)
            finally:
                nd.inputs['action'].default_value = False
                dg = bpy.context.evaluated_depsgraph_get(); dg.update()
            d = np.asarray(pp['center']) - np.asarray(pf['center'])
            trig[P] = dict(center=np.asarray(pf['center'], float), normal=np.asarray(pf['normal'], float),
                           travel=d / max(np.linalg.norm(d), 1e-9), travel_mm=float(np.linalg.norm(d)), capsule=FINGER_OF[P])
            Vc, Tc, _ = _mesh(P + ' bool', dg)
            cutters[P + ' bool'] = (Vc, Tc)
        for n in INTERNALS:
            if n in bpy.data.objects:
                Vi, Ti, _ = _mesh(n, dg)
                if len(Vi):
                    internals[n] = (Vi, Ti)
    finally:
        bpy.context.window.scene = win
        bpy.data.scenes.remove(sc)
        BS.set_inputs(mod, {'Enable': enable0})
    return dict(V=V, T=T, caps=caps, trig=trig, cutters=cutters, internals=internals, controls=vals)


from gather_np import save_npz, load_npz  # noqa: E402,F401


if __name__ == '__main__':
    a = sys.argv[sys.argv.index('--') + 1:]
    bpy.ops.wm.open_mainfile(filepath=a[0])
    g = gather()
    save_npz(a[1], g)
    print('GATHER', len(g['V']), 'verts', len(g['caps']), 'caps', {P: (np.round(t['center'], 2).tolist(), np.round(t['travel'], 3).tolist(), round(t['travel_mm'], 3)) for P, t in g['trig'].items()},
          'internals', list(g['internals']), 'controls', g['controls'])
