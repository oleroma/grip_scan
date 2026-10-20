"""Hide the helper 'shape offset' in the viewport (checked: the stage still reads it), open in scene 'GripV', save."""
import sys, numpy as np, bpy
FINAL = '/tmp/claude-0/sdf/smooth/final'
sys.path.insert(0, FINAL + '/stage')
import build_stage as BS
f = FINAL + '/Grip_smooth.blend'
bpy.ops.wm.open_mainfile(filepath=f)
ob = bpy.data.objects[BS.TARGET]; so = bpy.data.objects['shape offset']

def nverts():
    sc = bpy.data.scenes.new('hf t'); win = bpy.context.window.scene; bpy.context.window.scene = sc
    sc.collection.objects.link(ob)
    dg = bpy.context.evaluated_depsgraph_get(); dg.update()
    me = ob.evaluated_get(dg).to_mesh(); n = len(me.vertices)
    V = np.empty(3 * n); me.vertices.foreach_get('co', V); mx = V.reshape(-1, 3)[:, 0].max()
    ob.evaluated_get(dg).to_mesh_clear()
    bpy.context.window.scene = win; bpy.data.scenes.remove(sc)
    return n, round(float(mx), 4)
a = nverts()
so.hide_viewport = True
ob.data.update()
b = nverts()
print('stage with shape offset visible', a, 'disabled in viewport', b)
if a != b:
    so.hide_viewport = False
    print('reverted: the stage needs it enabled')
gp = bpy.data.node_groups['global'].nodes['global parameters']
print('variant', gp.inputs['variant'].default_value, 'subdivision', gp.inputs['subdivision level'].default_value)
bpy.context.window.scene = bpy.data.scenes['GripV']
bpy.ops.wm.save_mainfile(filepath=f)
print('SAVED', f, 'active scene', bpy.context.window.scene.name)
