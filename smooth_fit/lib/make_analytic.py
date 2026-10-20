"""Analytic test shapes for the curvature code, meshed like the stage meshes its result:
analytic SDF -> Field to Grid on a dense Cube Grid Topology (Voxel 0.5 mm) -> Grid to Mesh (threshold 0).
    /tmp/claude-0/venv/bin/python make_analytic.py [voxel]
Writes lib/validation/{sphere_r20,cylinder_r10,torus_groove}.npz (V, T)."""
import os, sys
import numpy as np
import bpy
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'stage'))
from gn import G  # noqa: E402

VOX = float(sys.argv[sys.argv.index('--') + 1]) if '--' in sys.argv and len(sys.argv) > sys.argv.index('--') + 1 else 0.5
C0 = (0.13, 0.27, 0.41)          # off the voxel lattice
OUT = os.path.join(HERE, 'validation')


def build(name, sdf_fn, lo, hi):
    g = G('analytic ' + name)
    g.inp('Geometry', 'GEOMETRY'); g.out('Geometry', 'GEOMETRY')
    M = g.math
    lo = [np.floor((l + c) / VOX) * VOX for l, c in zip(lo, C0)]
    hi = [np.ceil((h + c) / VOX) * VOX for h, c in zip(hi, C0)]
    res = [int(round((h - l) / VOX)) for l, h in zip(lo, hi)]
    topo = g.node('GeometryNodeCubeGridTopology', {'Bounds Min': tuple(lo), 'Bounds Max': tuple(hi),
                                                  'Resolution X': res[0], 'Resolution Y': res[1], 'Resolution Z': res[2]})
    px, py, pz = g.sep(g.pos())
    x, y, z = M('SUBTRACT', px, C0[0]), M('SUBTRACT', py, C0[1]), M('SUBTRACT', pz, C0[2])
    f = g.node('GeometryNodeFieldToGrid')
    f.grid_items.new('FLOAT', 'Value')
    g.set(f, 'Topology', topo.outputs[0])
    g.set(f, 'Value', sdf_fn(g, x, y, z))
    m = g.node('GeometryNodeGridToMesh', {'Grid': f.outputs[0], 'Threshold': 0.0})
    g.link(m.outputs[0], g.O('Geometry'))
    me = bpy.data.meshes.new(name)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    md = ob.modifiers.new('gn', 'NODES'); md.node_group = g.ng
    dg = bpy.context.evaluated_depsgraph_get(); dg.update()
    ev = ob.evaluated_get(dg); em = ev.to_mesh(); em.calc_loop_triangles()
    V = np.empty(len(em.vertices) * 3); em.vertices.foreach_get('co', V); V = V.reshape(-1, 3) - np.array(C0)
    T = np.empty(len(em.loop_triangles) * 3, np.int64); em.loop_triangles.foreach_get('vertices', T); T = T.reshape(-1, 3)
    ev.to_mesh_clear()
    np.savez_compressed(os.path.join(OUT, name + '.npz'), V=V, T=T, voxel=VOX)
    print(name, len(V), 'verts', flush=True)


def hyp(g, *a):
    return g.vmath('LENGTH', g.xyz(*(list(a) + [0.0] * (3 - len(a)))), out=1)


os.makedirs(OUT, exist_ok=True)
M = None
# sphere r 20
build('sphere_r20', lambda g, x, y, z: g.math('SUBTRACT', hyp(g, x, y, z), 20.0), (-23, -23, -23), (23, 23, 23))
# cylinder r 10 along x, |x| <= 30
build('cylinder_r10', lambda g, x, y, z: g.math('MAXIMUM', g.math('SUBTRACT', hyp(g, y, z), 10.0),
                                                g.math('SUBTRACT', g.math('ABSOLUTE', x), 30.0)), (-32, -12, -12), (32, 12, 12))


# torus groove: cylinder r 40 about z (|z| <= 20) minus a torus (major radius 40 about z, minor 8)
def groove(g, x, y, z):
    M = g.math
    rho = hyp(g, x, y)
    cyl = M('MAXIMUM', M('SUBTRACT', rho, 40.0), M('SUBTRACT', M('ABSOLUTE', z), 20.0))
    tor = M('SUBTRACT', hyp(g, M('SUBTRACT', rho, 40.0), z), 8.0)
    return M('MAXIMUM', cyl, M('MULTIPLY', tor, -1.0))


build('torus_groove', groove, (-42, -42, -22), (42, 42, 22))
