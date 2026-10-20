"""render_relief.py: zebra (isophote) and grazing-light renders of stage skins, right handle, side by side.

    $PY render_relief.py -- OUT_DIR SHEET.jpg [--views outside,front,...] [--modes zebra,graze] label=run/stage.npz ...

Based on the independent verifier's v_fit/scripts/render_skin2.py (same views, lights and zebra material).
Cycles CPU, 640x480, 16 samples. The sheet has one row per (view, mode) and one column per label.
"""
import bpy
import sys
import os
import math
import numpy as np
from mathutils import Vector, Quaternion

a = sys.argv[sys.argv.index('--') + 1:]
out, sheet = a[0], a[1]
views_sel = modes_sel = None
items = []
i = 2
while i < len(a):
    if a[i] == '--views':
        views_sel = a[i + 1].split(','); i += 2
    elif a[i] == '--modes':
        modes_sel = a[i + 1].split(','); i += 2
    else:
        items.append(a[i].split('=', 1)); i += 1
os.makedirs(out, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = 16
sc.render.resolution_x, sc.render.resolution_y = 640, 480
sc.render.threads_mode = 'FIXED'; sc.render.threads = 2
sc.cycles.use_denoising = False
sc.view_settings.view_transform = 'Standard'
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.25, 0.25, 0.27, 1)
w.node_tree.nodes['Background'].inputs[1].default_value = 1.0


def mesh_obj(name, path):
    d = np.load(path); V = d['V']; T = d['T']
    keep = (V[T][:, :, 0] > 22).all(1) & (V[T][:, :, 2] < 41).all(1)
    T = T[keep]
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V)); me.vertices.foreach_set('co', V.astype(np.float32).ravel())
    me.loops.add(3 * len(T)); me.loops.foreach_set('vertex_index', T.ravel().astype(np.int32))
    me.polygons.add(len(T)); me.polygons.foreach_set('loop_start', np.arange(0, 3 * len(T), 3, dtype=np.int32))
    me.update(); me.polygons.foreach_set('use_smooth', np.ones(len(T), bool))
    # outward winding (the stage output may be inside-out): flip when the signed volume is negative
    vol = np.einsum('ij,ij->i', V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])).sum()
    if vol < 0:
        me.flip_normals()
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob)
    return ob


def mat_grey():
    m = bpy.data.materials.new('grey'); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value = (0.7, 0.7, 0.7, 1)
    b.inputs['Roughness'].default_value = 0.45
    return m


def mat_zebra(dirv, freq=18.0):
    m = bpy.data.materials.new('zebra'); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    g = nt.nodes.new('ShaderNodeNewGeometry')
    dp = nt.nodes.new('ShaderNodeVectorMath'); dp.operation = 'DOT_PRODUCT'; dp.inputs[1].default_value = dirv
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = freq
    fr = nt.nodes.new('ShaderNodeMath'); fr.operation = 'FRACT'
    gt = nt.nodes.new('ShaderNodeMath'); gt.operation = 'GREATER_THAN'; gt.inputs[1].default_value = 0.5
    mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'
    mx.inputs['A'].default_value = (0.05, 0.05, 0.05, 1); mx.inputs['B'].default_value = (0.95, 0.95, 0.95, 1)
    em = nt.nodes.new('ShaderNodeEmission'); o = nt.nodes.new('ShaderNodeOutputMaterial')
    L = nt.links.new
    L(g.outputs['Normal'], dp.inputs[0]); L(dp.outputs['Value'], mul.inputs[0]); L(mul.outputs[0], fr.inputs[0])
    L(fr.outputs[0], gt.inputs[0]); L(gt.outputs[0], mx.inputs['Factor']); L(mx.outputs['Result'], em.inputs[0])
    L(em.outputs[0], o.inputs[0])
    return m


cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
cam.data.type = 'ORTHO'
sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN')); sc.collection.objects.link(sun)
sun.data.energy = 3.0; sun.data.angle = math.radians(1.0)
C = Vector((56, -45, 12))
VIEWS = {   # camera direction from the centre, up, ortho scale, grazing light travel, zebra direction
    'outside': (Vector((1, 0, 0)), Vector((0, 0, 1)), 95, Vector((-0.25, 0, -0.97)), (0, 0.7071, 0.7071)),
    'front': (Vector((0, 1, 0)), Vector((0, 0, 1)), 95, Vector((0, -0.25, -0.97)), (0.7071, 0, 0.7071)),
    'frontout': (Vector((0.7, 0.7, -0.15)), Vector((0, 0, 1)), 95, Vector((-0.2, -0.2, -0.96)), (0.6, -0.6, 0.53)),
    'inside': (Vector((-1, 0, 0.05)), Vector((0, 0, 1)), 95, Vector((0.25, 0, -0.97)), (0, 0.7071, 0.7071)),
    'under': (Vector((0.15, 0.25, -1)), Vector((0, 1, 0)), 105, Vector((-0.97, 0, 0.25)), (0.7071, 0.7071, 0)),
}
objs = {lab: mesh_obj(lab, p) for lab, p in items}
grey = mat_grey()
files = {}
for vname, (dv, up, scale, ldir, zdir) in VIEWS.items():
    if views_sel and vname not in views_sel:
        continue
    dv = dv.normalized()
    cam.location = C + dv * 300
    q = (-dv).to_track_quat('-Z', 'Y')
    cam_up = q @ Vector((0, 1, 0))
    upp = (up - dv * up.dot(dv)).normalized()
    ang = math.atan2(cam_up.cross(upp).dot(-dv), cam_up.dot(upp))
    cam.rotation_euler = (Quaternion(-dv, ang) @ q).to_euler()
    cam.data.ortho_scale = scale; cam.data.clip_end = 1000
    sun.rotation_euler = ldir.normalized().to_track_quat('-Z', 'Y').to_euler()
    for mode in ('zebra', 'graze'):
        if modes_sel and mode not in modes_sel:
            continue
        mat = mat_zebra(zdir) if mode == 'zebra' else grey
        sun.hide_render = mode == 'zebra'
        for lab, ob in objs.items():
            for o2 in objs.values():
                o2.hide_render = o2 is not ob
            ob.data.materials.clear(); ob.data.materials.append(mat)
            f = os.path.join(out, f'{vname}_{mode}_{lab}.png')
            sc.render.filepath = f
            bpy.ops.render.render(write_still=True)
            files[(vname, mode, lab)] = f
# sheet
from PIL import Image, ImageDraw
labs = [l for l, _ in items]
rows = sorted({(v, m) for v, m, _ in files}, key=lambda x: (list(VIEWS).index(x[0]), x[1]))
tw, th = 400, 300
S = Image.new('RGB', (tw * len(labs), th * len(rows) + 24), (30, 30, 30))
dr = ImageDraw.Draw(S)
for ci, lab in enumerate(labs):
    dr.text((ci * tw + 8, 4), lab, fill=(255, 255, 0))
for ri, (v, m) in enumerate(rows):
    for ci, lab in enumerate(labs):
        im = Image.open(files[(v, m, lab)]).convert('RGB').resize((tw, th))
        S.paste(im, (ci * tw, 24 + ri * th))
        ImageDraw.Draw(S).text((ci * tw + 6, 24 + ri * th + 4), f'{lab} {v} {m}', fill=(255, 80, 80))
S.save(sheet, quality=85)
print('SHEET', sheet, len(files), 'renders')
