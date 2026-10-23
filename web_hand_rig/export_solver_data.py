import bpy, numpy as np, json
S = '/tmp/claude-0/-home-claude/f25e5681-7fdb-5318-adab-0634023a58c3/scratchpad/'
# grip + markers from v2
bpy.ops.wm.open_mainfile(filepath='/mnt/project-files/hand_scan_rig/fitted_hold/hand_on_grip_hold_v2.blend')
g = bpy.data.objects['grip (current GripV)']; dg = bpy.context.evaluated_depsgraph_get()
ge = g.evaluated_get(dg); me = ge.to_mesh(); me.calc_loop_triangles()
Vg = np.array([g.matrix_world @ v.co for v in me.vertices]); Fg = np.array([t.vertices[:] for t in me.loop_triangles])
mk = {n: list(bpy.data.objects[n].matrix_world.to_translation()) for n in ['R1 target', 'R2 press', 'R4 press', 'palm swell']}
np.savez(S + 'work/grip.npz', V=Vg, F=Fg); json.dump(mk, open(S + 'work/markers.json', 'w'))
# v2 skin joints for reference (thumb tip etc.) already in v2_mats.json
# subdivided rest hand with interpolated weights
bpy.ops.wm.open_mainfile(filepath=S + 'work/hand_rest.blend')
h = bpy.data.objects['hand']
for m in h.modifiers: m.show_viewport = (m.type == 'SUBSURF')
h.modifiers['Subdivision'].levels = 1
dg = bpy.context.evaluated_depsgraph_get(); he = h.evaluated_get(dg); m2 = bpy.data.meshes.new_from_object(he, depsgraph=dg)
names = [vg.name for vg in h.vertex_groups]
W = np.zeros((len(m2.vertices), len(names)))
for v in m2.vertices:
    for ge_ in v.groups: W[v.index, ge_.group] = ge_.weight
m2.calc_loop_triangles()
np.savez(S + 'work/hand_sub1.npz', V=np.array([v.co for v in m2.vertices]), F=np.array([t.vertices[:] for t in m2.loop_triangles]), W=W, names=np.array(names))
print('grip', Vg.shape, Fg.shape, 'sub1', len(m2.vertices), W.sum(1).min(), W.sum(1).max())
