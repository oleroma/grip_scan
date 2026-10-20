import bpy, sys, numpy as np
src=sys.argv[sys.argv.index('--')+1]; out=sys.argv[sys.argv.index('--')+2]
bpy.ops.wm.open_mainfile(filepath=src)
sc=bpy.context.scene; dg=bpy.context.evaluated_depsgraph_get()
def grp(n):
    if n.endswith(' skin'): return 'hand'
    if n in ('case_top','case_bottom'): return 'case'
    if n=='R1': return 'r1'
    if n in ('R2','R4'): return 'trig'
    if n in ('seam','hand points','L1','L2','L4'): return None
    return 'parts'
res={}
for o in sc.objects:
    if not(o.visible_get() and o.type=='MESH'): continue
    g=grp(o.name)
    if g is None: continue
    oe=o.evaluated_get(dg); me=oe.to_mesh(); me.calc_loop_triangles()
    V=np.array([oe.matrix_world@v.co for v in me.vertices],np.float32).reshape(-1,3)
    T=np.array([t.vertices[:] for t in me.loop_triangles],np.int32).reshape(-1,3)
    oe.to_mesh_clear()
    if len(T)==0: continue
    res.setdefault(g,[]).append((V,T)); print("got",o.name,len(V),flush=True)
D={}
for g,L in res.items():
    off=0; Vs=[];Ts=[]
    for V,T in L: Vs.append(V);Ts.append(T+off);off+=len(V)
    D[g+'_V']=np.concatenate(Vs); D[g+'_T']=np.concatenate(Ts)
np.savez_compressed(out,**D); print("SAVED",out)
