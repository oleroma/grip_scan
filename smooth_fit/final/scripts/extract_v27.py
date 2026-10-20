"""Read-only: evaluated 'basemesh GripV' of the old rig file /tmp/claude-0/work/v_27.blend -> compare/v27_skin.npz"""
import numpy as np, bpy
bpy.ops.wm.open_mainfile(filepath='/tmp/claude-0/work/v_27.blend', load_ui=False)
gp = bpy.data.node_groups['global'].nodes['global parameters']
print('saved subdivision', gp.inputs['subdivision level'].default_value)
gp.inputs['subdivision level'].default_value = 'high'   # in memory only (never saved)
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()
ob = bpy.data.objects['basemesh GripV']
me = ob.evaluated_get(dg).to_mesh(); me.calc_loop_triangles()
V = np.empty(3 * len(me.vertices)); me.vertices.foreach_get('co', V); V = V.reshape(-1, 3)
T = np.empty(3 * len(me.loop_triangles), np.int64); me.loop_triangles.foreach_get('vertices', T); T = T.reshape(-1, 3)
M = np.array(ob.matrix_world); V = V @ M[:3, :3].T + M[:3, 3]
print('modifiers', [(m.name, m.type) for m in ob.modifiers], 'verts', len(V), 'bbox', V.min(0).round(2), V.max(0).round(2))
np.savez_compressed('/tmp/claude-0/sdf/smooth/final/compare/v27_skin.npz', V=V, T=T, chg=np.zeros(len(V)))
