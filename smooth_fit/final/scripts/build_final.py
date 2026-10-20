"""build_final.py: make final/Grip_smooth.blend from smooth/base.blend: final stage (shape + thenar), Olex's
controls group, the solved shape from an npz.   $PY build_final.py -- [--npz solve/d1.npz] [--test]"""
import os, sys, time, numpy as np, bpy
FINAL = '/tmp/claude-0/sdf/smooth/final'
sys.path.insert(0, FINAL + '/stage'); sys.path.insert(0, FINAL)
import build_stage as BS
import refit
args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
npz = args[args.index('--npz') + 1] if '--npz' in args else FINAL + '/solve/d1.npz'
out = FINAL + '/Grip_smooth.blend'
t0 = time.time()
bpy.ops.wm.open_mainfile(filepath='/tmp/claude-0/sdf/smooth/base.blend')
mod, info = BS.build([FINAL + '/modules'])
d = np.load(npz)
so = refit.store_offset(d['V'].astype(float), d['T'].astype(np.int64), d['h'].astype(float), d['D'].astype(float))
so['source_npz'] = npz
BS.set_inputs(mod, {'Shape Offset': 'shape offset'})
ob = bpy.data.objects[BS.TARGET]
print('modifier inputs visible:', [it.name for it in mod.node_group.interface.items_tree
      if getattr(it, 'in_out', None) == 'INPUT' and it.item_type == 'SOCKET' and not getattr(it, 'hide_in_modifier', False)])
print('controls:', {k: v for k, v in BS.get_inputs(mod).items() if k in BS.CONTROL_NAMES})
print('groups left:', [g for g in ('HF fill', 'HF carve', 'HF seats', 'HF shape', 'HF thenar', 'HF Stage', BS.CONTROLS_GROUP, BS.VALUES_GROUP) if g in bpy.data.node_groups])
print('dashboard node:', bpy.data.node_groups['GripV controls'].nodes.get(BS.CONTROLS_GROUP) is not None)
print('collections of shape offset:', [c.name for c in so.users_collection])
if '--test' in args:
    sc = bpy.data.scenes.new('hf test'); win = bpy.context.window.scene; bpy.context.window.scene = sc
    sc.collection.objects.link(ob)
    for n in ('hand points', 'hand rig', 'shape offset'):
        pass
    res = {}
    for fit in (1.0, 0.0, 1.0):
        BS.set_inputs(mod, {'Fit': fit})
        dg = bpy.context.evaluated_depsgraph_get(); dg.update()
        me = ob.evaluated_get(dg).to_mesh()
        V = np.empty(3 * len(me.vertices)); me.vertices.foreach_get('co', V); V = V.reshape(-1, 3)
        ob.evaluated_get(dg).to_mesh_clear()
        print(f'TEST Fit={fit}: {len(V)} verts, max x {V[:,0].max():.3f}, min y {V[:,1].min():.3f}, z {V[:,2].min():.3f}..{V[:,2].max():.5f}', flush=True)
    bpy.context.window.scene = win; bpy.data.scenes.remove(sc)
    BS.set_inputs(mod, {'Fit': 1.0})
bpy.ops.wm.save_as_mainfile(filepath=out)
print(f'SAVED {out} in {time.time() - t0:.1f} s (active scene {bpy.context.window.scene.name})')
