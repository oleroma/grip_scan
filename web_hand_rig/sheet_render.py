# python sheet_render.py file.blend prefix handobj
import bpy, sys
from mathutils import Vector
bpy.ops.wm.open_mainfile(filepath=sys.argv[1]); pre, hand = sys.argv[2], sys.argv[3]
sc = bpy.context.scene
for o in sc.objects:
    if o.type == 'MESH': o.hide_render = o.name not in (hand, 'grip (current GripV)')
h = bpy.data.objects[hand]
skin = bpy.data.materials.new('skin2'); skin.use_nodes = True; skin.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.86, 0.62, 0.5, 1)
h.data.materials.clear(); h.data.materials.append(skin)
for m in h.modifiers:
    if m.type == 'SUBSURF': m.render_levels = 2
g = bpy.data.objects['grip (current GripV)']; gm = bpy.data.materials.new('g2'); gm.use_nodes = True
gm.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.42, 0.45, 0.5, 1); g.data.materials.clear(); g.data.materials.append(gm)
sc.render.engine = 'CYCLES'; sc.cycles.samples = 32; sc.render.resolution_x = 760; sc.render.resolution_y = 600
sc.world = bpy.data.worlds.new('w'); sc.world.use_nodes = True; sc.world.node_tree.nodes['Background'].inputs[0].default_value = (0.92, 0.92, 0.92, 1)
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.lens = 50; cam.data.clip_end = 1e5
sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN')); sc.collection.objects.link(sun); sun.data.energy = 3.5
V = {'behind': ((0, -1, 0.12), (75, -50, 25), 380), 'outside': ((1, 0.15, 0.2), (75, -60, 25), 380),
     'above': ((0.15, -0.25, 1), (70, -55, 20), 400), 'front': ((-0.25, 1, 0.25), (60, -40, 25), 380)}
for k, (d, c, dist) in V.items():
    d = Vector(d).normalized(); cam.location = Vector(c) + d * dist
    cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    sun.rotation_euler = (-(d + Vector((0.3, -0.2, 0.8)))).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = f'{pre}_{k}.png'; bpy.ops.render.render(write_still=True)
