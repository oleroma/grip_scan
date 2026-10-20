import bpy, sys, numpy as np
sys.path[:0]=['/tmp/claude-0/sdf/smooth/final/stage','/tmp/claude-0/sdf/smooth/lib','/tmp/claude-0/sdf/smooth/final/solver_v2']
bpy.ops.wm.open_mainfile(filepath="/tmp/claude-0/sdf/overlap/Grip_smooth.blend")
import build_stage as BS, gripmetrics as GM
ob=bpy.data.objects[BS.TARGET]; mod=ob.modifiers[BS.MOD_NAME]
print(BS.get_inputs(mod))
hv=bpy.data.node_groups.get('hand fit values'); print([ (n.name,[i.name for i in n.inputs]) for n in hv.nodes] if hv else None)
BS.set_inputs(mod,{'Shape':False}); ob.data.update()
dg=bpy.context.evaluated_depsgraph_get(); dg.update()
ev=ob.evaluated_get(dg); me=ev.to_mesh(); me.calc_loop_triangles()
M=np.array(ob.matrix_world)
V=np.empty(len(me.vertices)*3); me.vertices.foreach_get('co',V); V=V.reshape(-1,3)@M[:3,:3].T+M[:3,3]
T=np.empty(len(me.loop_triangles)*3,np.int64); me.loop_triangles.foreach_get('vertices',T); T=T.reshape(-1,3)
print("nofit mesh",V.shape,T.shape)
T,_=GM.orient(V,T)
k1,k2,kink,cm=GM.curvature_fields(V,T,GM.curv_region(V))
import os; os.makedirs('/tmp/claude-0/sdf/smooth/baseline/nofit',exist_ok=True)
np.savez_compressed('/tmp/claude-0/sdf/smooth/baseline/nofit/curvature.npz',V=V,k1=k1,k2=k2,kink=kink['all'],kink_concave=kink['concave'],kink_convex=kink['convex'],mask=cm,edited=np.zeros(len(V),bool))
np.savez_compressed('/tmp/claude-0/sdf/overlap/nofit_skin.npz',V=V,T=T)
print("DONE")
