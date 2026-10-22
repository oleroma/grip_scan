"""npz (V,F,W,bones,heads,tails) -> .blend with armature 'hand rig' + mesh 'hand' (Armature modifier). mm units."""
import bpy, sys, numpy as np
from mathutils import Vector
npz, out = sys.argv[1], sys.argv[2]
d = np.load(npz, allow_pickle=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene; sc.unit_settings.scale_length = 0.001; sc.unit_settings.length_unit = 'MILLIMETERS'
names = [str(b) for b in d['bones']]
PARENT = {'lowerarm01.R': None, 'lowerarm02.R': 'lowerarm01.R', 'wrist.R': 'lowerarm02.R',
          'metacarpal1.R': 'wrist.R', 'metacarpal2.R': 'wrist.R', 'metacarpal3.R': 'wrist.R', 'metacarpal4.R': 'wrist.R',
          'finger1-1.R': 'wrist.R'}
for f in range(1, 6):
    PARENT.setdefault(f'finger{f}-1.R', f'metacarpal{f-1}.R')
    PARENT[f'finger{f}-2.R'] = f'finger{f}-1.R'; PARENT[f'finger{f}-3.R'] = f'finger{f}-2.R'
arm = bpy.data.armatures.new('hand rig'); ao = bpy.data.objects.new('hand rig', arm); sc.collection.objects.link(ao)
bpy.context.view_layer.objects.active = ao; bpy.ops.object.mode_set(mode='EDIT')
palm_n = Vector(d['palm_normal']) if 'palm_normal' in d else Vector((0, 0, 1))
for i, b in enumerate(names):
    eb = arm.edit_bones.new(b); eb.head = Vector(d['heads'][i]); eb.tail = Vector(d['tails'][i])
    # roll: bone Z axis toward the BACK of the hand (-palm normal) -> flexion = +X rotation curls toward the palm
    eb.align_roll(-palm_n)
for b in names:
    p = PARENT.get(b)
    if p: arm.edit_bones[b].parent = arm.edit_bones[p]
bpy.ops.object.mode_set(mode='OBJECT')
me = bpy.data.meshes.new('hand'); F = [[int(i) for i in f if i >= 0] for f in d['F']]
me.from_pydata(d['V'].tolist(), [], F); me.update()
import bmesh
bm = bmesh.new(); bm.from_mesh(me)
bound = [e for e in bm.edges if e.is_boundary]
bmesh.ops.holes_fill(bm, edges=bound, sides=0)  # close the forearm cut (and any small hole)
bm.to_mesh(me); bm.free()
for p in me.polygons: p.use_smooth = True
mo = bpy.data.objects.new('hand', me); sc.collection.objects.link(mo); mo.parent = ao
for i, b in enumerate(names):
    vg = mo.vertex_groups.new(name=b); w = d['W'][:, i]
    for vi in np.where(w > 1e-4)[0]: vg.add([int(vi)], float(w[vi]), 'REPLACE')
m = mo.modifiers.new('Armature', 'ARMATURE'); m.object = ao
s = mo.modifiers.new('Subdivision', 'SUBSURF'); s.levels = 1; s.render_levels = 2
mat = bpy.data.materials.new('skin'); mat.use_nodes = True
mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.85, 0.62, 0.5, 1)
me.materials.append(mat)
bpy.ops.wm.save_as_mainfile(filepath=out)
print('saved', out)
