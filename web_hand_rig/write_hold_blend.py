"""hand_rest.blend + solve json -> hold .blend with the grip and markers (mm). usage: write_hold_blend.py solve.json out.blend"""
import bpy, sys, json, numpy as np
from mathutils import Matrix
S = '/tmp/claude-0/-home-claude/f25e5681-7fdb-5318-adab-0634023a58c3/scratchpad/'
sol = json.load(open(sys.argv[1])); out = sys.argv[2]
bpy.ops.wm.open_mainfile(filepath=S + 'work/hand_rest.blend')
sc = bpy.context.scene
a = bpy.data.objects['hand rig']
# keep the rest pose in the canonical frame; place the armature object at the root so the bones keep local rotations only
PW = {b: Matrix(m) for b, m in sol['posed_world'].items()}
root = 'lowerarm01.R'
Mr = a.data.bones[root].matrix_local
a.matrix_world = PW[root] @ Mr.inverted()
bpy.context.view_layer.update()
order = ['lowerarm01.R', 'lowerarm02.R', 'wrist.R'] + [f'metacarpal{i}.R' for i in range(1, 5)] + [f'finger{f}-{k}.R' for f in range(1, 6) for k in range(1, 4)]
inv = a.matrix_world.inverted()
for b in order:
    pb = a.pose.bones[b]; pb.rotation_mode = 'QUATERNION'
    pb.matrix = inv @ PW[b]
    bpy.context.view_layer.update()
# grip + markers
g = np.load(S + 'work/grip.npz'); me = bpy.data.meshes.new('grip'); me.from_pydata(g['V'].tolist(), [], g['F'].tolist())
go = bpy.data.objects.new('grip (current GripV)', me); sc.collection.objects.link(go)
gm = bpy.data.materials.new('grip'); gm.use_nodes = True; gm.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.55, 0.57, 0.6, 1); me.materials.append(gm)
mk = json.load(open(S + 'work/markers.json'))
for n, p in mk.items():
    bpy.ops.mesh.primitive_uv_sphere_add(radius=2.5, location=p); o = bpy.context.active_object; o.name = n
    mm = bpy.data.materials.new(n); mm.use_nodes = True; mm.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.9, 0.1, 0.1, 1); o.data.materials.append(mm)
bpy.ops.wm.save_as_mainfile(filepath=out)
print('saved', out)
