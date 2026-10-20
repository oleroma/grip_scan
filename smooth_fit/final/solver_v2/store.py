"""store.py: write a solved offset into the mesh object 'shape offset' (same storage as final/refit.py)."""
import numpy as np
import bpy


def store_offset(V, T, off):
    """off: (n, 3) offset vectors per vertex of the stage input mesh (mm); zero for x <= 0 (the stage mirrors)."""
    off = np.array(off, float)
    off[V[:, 0] <= 0] = 0.0
    me = bpy.data.meshes.new('shape offset')
    me.vertices.add(len(V)); me.vertices.foreach_set('co', np.asarray(V, float).ravel())
    me.loops.add(3 * len(T)); me.loops.foreach_set('vertex_index', np.asarray(T).ravel())
    me.polygons.add(len(T)); me.polygons.foreach_set('loop_start', np.arange(0, 3 * len(T), 3))
    me.update()
    me.attributes.new('offset', 'FLOAT_VECTOR', 'POINT').data.foreach_set('vector', off.ravel())
    so = bpy.data.objects.get('shape offset')
    if so is None:
        so = bpy.data.objects.new('shape offset', me)
        bpy.data.collections['hand fit'].objects.link(so)   # never 'basemesh GripV' (the generator reads that collection)
    else:
        old = so.data; so.data = me
        if old.users == 0:
            bpy.data.meshes.remove(old)
    me.name = 'shape offset'
    so.hide_render = True
    so.display_type = 'BOUNDS'
    return so


def stored_offset(n):
    so = bpy.data.objects.get('shape offset')
    if so is None or len(so.data.vertices) != n or 'offset' not in so.data.attributes:
        return None
    off = np.empty(3 * n); so.data.attributes['offset'].data.foreach_get('vector', off)
    return off.reshape(-1, 3)
