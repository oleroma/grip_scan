"""stage_api.py: keep one Blender session open and re-evaluate the stage quickly (parameter sweeps).

    import sys; sys.path.insert(0, '/tmp/claude-0/sdf/smooth/final/stage')
    import stage_api as SA
    mod = SA.open_copy('/path/to/my_copy.blend', modules_dirs=['/my/folder'])   # rebuilds the stage
    V, T, chg, secs = SA.eval_skin(mod, {'Carve Max Z': 40.0})                  # stage output only
    rep = SA.quick_metrics(V, T)                                                  # contacts, dig, top, thickness, ...
    SA.rebuild(['/my/folder'])            # after editing your mod_<name>.py (same session)

Run it with /tmp/claude-0/venv/bin/python my_sweep.py. Never save over Grip_hand.blend.
Open a copy made by make_agent_copy.py (0.6 s) rather than Grip_hand.blend itself (17 s on load).
Repeated evaluations in one session are bit-identical (between processes C's fill varies by < 0.7 mm).
"""
import os
import sys
import time
import numpy as np
import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/tmp/claude-0/sdf/smooth/lib')
import build_stage as BS  # noqa: E402
import gripmetrics as GM  # noqa: E402

_state = {}


def open_copy(blend, modules_dirs=None, values=None, subdiv=None, voxel=None, rebuild_stage=True):
    bpy.ops.wm.open_mainfile(filepath=blend)
    gp = bpy.data.node_groups['global'].nodes['global parameters']
    gp.inputs['variant'].default_value = 'GripV'
    if subdiv:
        gp.inputs['subdivision level'].default_value = subdiv
    if voxel:
        gp.inputs['voxel resolution'].default_value = voxel
    sc = bpy.data.scenes.new('hf api')
    sc.collection.objects.link(bpy.data.objects[BS.TARGET])
    bpy.context.window.scene = sc            # evaluate only the basemesh (+ its dependencies)
    if rebuild_stage:
        mod, _ = BS.build(modules_dirs, values=values, verbose=False)
    else:
        mod = bpy.data.objects[BS.TARGET].modifiers[BS.MOD_NAME]
        if values:
            BS.set_inputs(mod, values)
    _state['dirs'] = modules_dirs
    return mod


def rebuild(modules_dirs=None, values=None):
    mod, info = BS.build(modules_dirs if modules_dirs is not None else _state.get('dirs'), values=values, verbose=False)
    return mod


def eval_skin(mod, values=None):
    """Set inputs, evaluate 'basemesh GripV' (the stage output). Returns V, T, change_mm, seconds."""
    if values:
        BS.set_inputs(mod, values)
    ob = bpy.data.objects[BS.TARGET]
    t = time.time()
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    dt = time.time() - t
    me.calc_loop_triangles()
    V = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', V); V = V.reshape(-1, 3)
    T = np.empty(len(me.loop_triangles) * 3, np.int64); me.loop_triangles.foreach_get('vertices', T); T = T.reshape(-1, 3)
    chg = np.zeros(len(V))
    if 'change_mm' in me.attributes:
        me.attributes['change_mm'].data.foreach_get('value', chg)
    ev.to_mesh_clear()
    return V, T, chg, dt


def quick_metrics(V, T, caps=None, base=None, wrap=False):
    """Contacts, dig elsewhere, topology, top face, battery face, max Y, wall thickness (+ wrap)."""
    caps = caps or GM.load_caps()
    return GM.skin_report(V, T, caps, base=base, quick=not wrap)
