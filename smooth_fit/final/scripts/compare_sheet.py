"""compare_sheet.py: the same 5 views of 4 case skins (right handle), neutral colour, one row each:
plain GripV | old rig v_27 | carved prototype (old fill + carve + seats) | this smooth fit.  -> final/compare_sheet.jpg"""
import os, sys, json, time
import numpy as np
import bpy
from mathutils import Vector
FINAL = '/tmp/claude-0/sdf/smooth/final'
sys.path.insert(0, FINAL + '/stage')
import eval_all as EA
import gripmetrics as GM
ROWS = [('plain GripV (no fit)', FINAL + '/runs/base_Grip_smooth_high_low/stage.npz'),
        ('old rig v_27 (cage deform)', FINAL + '/compare/v27_skin.npz'),
        ('carved prototype (fill + carve + seats)', '/tmp/claude-0/sdf/build/out/baseline_C/stage.npz'),
        ('this smooth fit (Grip_smooth.blend)', FINAL + '/runs/final/stage.npz')]
centre = Vector((58, -45, 22))
VIEWS = [('outside', Vector((1, -0.15, 0.1)), centre, 150, False),
         ('front', Vector((-0.25, 1, -0.35)), centre, 150, False),
         ('inside (palm side)', Vector((-1, 0.35, -0.25)), centre, 150, False),
         ('underside', Vector((0.25, 0.35, -1)), centre, 150, False),
         ('hand front', Vector((0.35, 1, -0.45)), centre, 175, True)]
RES = (480, 360)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.samples = 16; sc.cycles.device = 'CPU'; sc.cycles.use_denoising = True
sc.render.threads_mode = 'FIXED'; sc.render.threads = 3
sc.render.resolution_x, sc.render.resolution_y = RES
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.16, 0.16, 0.17, 1)
w.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.8
for i, (loc, e) in enumerate((((300, 200, 300), 3.0), ((-200, 300, 100), 1.5), ((200, -300, -150), 1.4))):
    ld = bpy.data.lights.new(f'sun{i}', 'SUN'); ld.energy = e
    lo = bpy.data.objects.new(f'sun{i}', ld); sc.collection.objects.link(lo)
    lo.rotation_euler = (Vector((60, -40, 25)) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
sc.view_settings.view_transform = 'Standard'
cam_d = bpy.data.cameras.new('cam'); cam_d.type = 'ORTHO'
cam = bpy.data.objects.new('cam', cam_d); sc.collection.objects.link(cam); sc.camera = cam
mat = EA._mat('case grey', (0.80, 0.80, 0.78, 1))
caps = GM.load_caps(FINAL + '/runs/final/caps.json')
hand = bpy.data.objects.new('hand', EA._capsule_mesh(caps, [k for k in caps if not k.startswith('palm')]))
hand.data.materials.append(EA._mat('skin', (0.80, 0.56, 0.43, 1)))
sc.collection.objects.link(hand)
objs = []
for label, f in ROWS:
    d = np.load(f)
    objs.append(EA._mesh_obj(sc, label[:20], d['V'].astype(float), d['T'].astype(np.int64), None, mat))
files = {}
t0 = time.time()
for r, ob in enumerate(objs):
    for o2 in objs:
        o2.hide_render = o2 is not ob
    for name, dvec, c, scale, show_hand in VIEWS:
        hand.hide_render = not show_hand
        cam_d.ortho_scale = scale
        cam.location = c + dvec.normalized() * 400
        cam.rotation_euler = (-dvec).to_track_quat('-Z', 'Y').to_euler()
        fp = f'{FINAL}/compare/r{r}_{name.split()[0]}.png'
        sc.render.filepath = fp
        bpy.ops.render.render(write_still=True)
        files[(r, name)] = fp
print('rendered', len(files), 'in', round(time.time() - t0), 's')
from PIL import Image, ImageDraw
W, H = RES; head, lab = 40, 210
S = Image.new('RGB', (lab + W * len(VIEWS), head + H * len(ROWS)), (36, 36, 38))
dr = ImageDraw.Draw(S)
dr.text((10, 6), 'Same views of the case skin (right handle), neutral grey; hand capsules = Olex\'s v_27 hold. Rows: plain GripV, the old rig v_27, the carved prototype, this smooth fit.', fill=(235, 235, 235))
for k, (name, *_ ) in enumerate(VIEWS):
    dr.text((lab + k * W + 8, 24), name, fill=(235, 235, 200))
for r, (label, _) in enumerate(ROWS):
    y = head + r * H
    for i, line in enumerate(label.replace(' (', '\n(').split('\n')):
        dr.text((10, y + H // 2 - 10 + 14 * i), line, fill=(240, 240, 240))
    for k, (name, *_ ) in enumerate(VIEWS):
        S.paste(Image.open(files[(r, name)]).convert('RGB'), (lab + k * W, y))
S.save(FINAL + '/compare_sheet.jpg', quality=88)
print('SAVED', FINAL + '/compare_sheet.jpg')
