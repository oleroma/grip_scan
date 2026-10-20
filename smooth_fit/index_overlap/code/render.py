import bpy, sys, math, numpy as np
from mathutils import Vector
a=sys.argv[sys.argv.index('--')+1:]
src, outp = a[0], a[1]; views=a[2].split(',') if len(a)>2 else ['front','top','outside']
ref = a[3] if len(a)>3 and a[3]!='-' else None
bpy.ops.wm.read_factory_settings(use_empty=True)
sc=bpy.context.scene
sc.render.engine='CYCLES'; sc.cycles.device='CPU'; sc.cycles.samples=24; sc.cycles.use_denoising=False
sc.render.resolution_x, sc.render.resolution_y = 900, 700
sc.view_settings.view_transform='Standard'
w=bpy.data.worlds.new('w'); sc.world=w; w.use_nodes=True
w.node_tree.nodes['Background'].inputs[0].default_value=(1,1,1,1); w.node_tree.nodes['Background'].inputs[1].default_value=0.55
D=np.load(src)
def mat(name,col,alpha=1.0,attr=None):
    m=bpy.data.materials.new(name); m.use_nodes=True; b=m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value=(*col,1); b.inputs['Roughness'].default_value=0.5
    if alpha<1: b.inputs['Alpha'].default_value=alpha
    if attr:
        nt=m.node_tree; at=nt.nodes.new('ShaderNodeAttribute'); at.attribute_name=attr; at.attribute_type='GEOMETRY'
        nt.links.new(at.outputs['Color'], b.inputs['Base Color'])
    return m
def obj(g,m,col=None):
    if g+'_V' not in D: return None
    V=D[g+'_V']; T=D[g+'_T']
    me=bpy.data.meshes.new(g); me.vertices.add(len(V)); me.vertices.foreach_set('co',V.ravel())
    me.loops.add(3*len(T)); me.loops.foreach_set('vertex_index',T.ravel())
    me.polygons.add(len(T)); me.polygons.foreach_set('loop_start',np.arange(0,3*len(T),3,dtype=np.int32))
    me.update(); me.polygons.foreach_set('use_smooth',np.ones(len(T),bool))
    if col is not None:
        ca=me.color_attributes.new('dcol','FLOAT_COLOR','POINT'); ca.data.foreach_set('color',col.ravel())
    o=bpy.data.objects.new(g,me); sc.collection.objects.link(o); o.data.materials.append(m); return o
dcol=None
if ref:
    # colour case vertices by signed change vs reference case (nearest-point distance, + = more material)
    from scipy.spatial import cKDTree
    R=np.load(ref); RV=R['case_V']; RT=R['case_T']
    import bmesh
    V=D['case_V']
    # signed distance via nearest ref vertex + its normal
    rn=np.zeros_like(RV); fn=np.cross(RV[RT[:,1]]-RV[RT[:,0]],RV[RT[:,2]]-RV[RT[:,0]])
    for k in range(3): np.add.at(rn,RT[:,k],fn)
    rn/=np.linalg.norm(rn,axis=1,keepdims=True)+1e-12
    d,i=cKDTree(RV).query(V); s=np.einsum('ij,ij->i',V-RV[i],rn[i])
    s=np.where(d<8,s,0)
    t=np.clip(s/1.5,-1,1)   # +-1.5 mm full colour
    grey=np.array([0.62,0.62,0.62]); blu=np.array([0.15,0.35,0.95]); red=np.array([0.95,0.25,0.15])
    c=np.where(t[:,None]>0, grey+(red-grey)*t[:,None], grey+(blu-grey)*(-t[:,None]))
    c=np.where((np.abs(s)<0.2)[:,None],grey,c)
    dcol=np.concatenate([c,np.ones((len(c),1))],1).astype(np.float32)
    np.save(outp+'_dstats.npy',s)
obj('case', mat('case',(0.62,0.62,0.62),attr='dcol' if ref else None), dcol)
obj('r1', mat('r1',(0.1,0.55,0.95)))
obj('trig', mat('trig',(0.98,0.6,0.1)))
obj('parts', mat('parts',(0.35,0.35,0.38)))
show_hand = not ref
hm=mat('hand',(0.93,0.72,0.6))
if show_hand: obj('hand', hm)
for n,(rot,e) in {'k1':((50,0,30),3.0),'k2':((-60,0,-140),1.5)}.items():
    l=bpy.data.objects.new(n,bpy.data.lights.new(n,'SUN')); l.data.energy=e
    l.rotation_euler=[math.radians(x) for x in rot]; sc.collection.objects.link(l)
cam=bpy.data.objects.new('cam',bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera=cam
cam.data.type='ORTHO'; cam.data.clip_end=2000
VIEWS={ # dir from centre to camera, centre, ortho scale
 'front':(Vector((0.15,1,0.35)),Vector((45,-20,30)),150),
 'top':(Vector((0.0,-0.2,1)),Vector((50,-40,25)),190),
 'outside':(Vector((1,0.35,0.25)),Vector((45,-35,20)),190),
 'r1':(Vector((0.12,1,0.45)),Vector((38,-8,40)),95),
 'grip':(Vector((0.55,0.75,-0.35)),Vector((55,-35,15)),150),
}
for v in views:
    dv,C,s=VIEWS[v]; dv=dv.normalized()
    cam.location=C+dv*500
    f=-dv; up=Vector((0,1,0)) if abs(dv.z)>0.9 else Vector((0,0,1))
    r=f.cross(up).normalized(); u=r.cross(f)
    from mathutils import Matrix
    M=Matrix((r,u,-f)).transposed(); cam.rotation_euler=M.to_euler(); cam.data.ortho_scale=s
    hm.node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value=0.45 if v=='r1' else 1.0
    sc.render.filepath=f'{outp}_{v}.png'; bpy.ops.render.render(write_still=True); print('R',sc.render.filepath,flush=True)
