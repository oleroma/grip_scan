"""Make a private copy of Grip_hand.blend for scripted work (module agents, sweeps).

    /tmp/claude-0/venv/bin/python /tmp/claude-0/sdf/smooth/stage/make_agent_copy.py -- /my/folder/mine.blend [src.blend]

(smooth/ copy: base.blend already opens fast, so a plain `cp base.blend mine.blend` is enough for prototypes.)

Content is identical to the source; only the active window scene becomes an empty scene 'hf idle'.
Opening a .blend evaluates its active scene (GripX/GripV in Grip_hand.blend, ~17 s); with 'hf idle'
active the copy opens in ~0.6 s. eval_all.py and stage_api.py evaluate in their own scratch scene,
so results are the same. To look at it in the Blender UI, switch the scene back to 'GripX'.
"""
import os
import sys
import time
import bpy

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
if not args:
    sys.exit(__doc__)
OUT = os.path.abspath(args[0])
SRC = os.path.abspath(args[1]) if len(args) > 1 else '/tmp/claude-0/sdf/build/Grip_hand.blend'
PROTECTED = {'/home/claude/Grip/blender/Grip.blend', '/tmp/claude-0/sdf/build/Grip_hand.blend', '/tmp/claude-0/sdf/smooth/base.blend'}
if len(args) > 2 and args[2] == '--allow-base':
    PROTECTED.discard('/tmp/claude-0/sdf/smooth/base.blend')     # only for making base.blend itself
assert OUT not in PROTECTED and OUT != SRC, 'write the copy into your own folder'
t = time.time()
bpy.ops.wm.open_mainfile(filepath=SRC, load_ui=False)
idle = bpy.data.scenes.get('hf idle') or bpy.data.scenes.new('hf idle')
bpy.context.window.scene = idle
bpy.ops.wm.save_as_mainfile(filepath=OUT)
print(f'SAVED {OUT} (active scene hf idle) from {SRC} in {time.time() - t:.1f} s', flush=True)
