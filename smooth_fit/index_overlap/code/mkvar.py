import bpy, sys
dx=float(sys.argv[sys.argv.index('--')+1]); out=sys.argv[sys.argv.index('--')+2]
bpy.ops.wm.open_mainfile(filepath="/tmp/claude-0/sdf/overlap/Grip_smooth.blend")
a=bpy.data.objects['hand rig']; a.location.x+=dx
print("hand points mods",[ (m.type, getattr(m,'object',None)) for m in bpy.data.objects['hand points'].modifiers], bpy.data.objects['hand points'].parent)
bpy.ops.wm.save_as_mainfile(filepath=out)
