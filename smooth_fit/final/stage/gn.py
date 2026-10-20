"""Tiny helper to build geometry-node trees from Python (Blender 5.2).

Based on /tmp/claude-0/sdf/C_hand_sdf/scripts/gn.py, plus panels and socket structure types.
Every module file gets one `G` bound to its own node group; use
  g.I('Name')                    -> output socket of the Group Input node (context or parameter)
  g.node('GeometryNodeXxx', {'Input': value_or_socket}, label=..., prop=...)
  g.math('ADD', a, b), g.vmath('DOT_PRODUCT', a, b, out=1)
  g.maprange(v, a, b, ta, tb), g.lerp(a, b, w), g.sample(grid, pos), g.mesh_sdf(mesh, vox, band)
"""
import bpy

SOCK = {'FLOAT': 'NodeSocketFloat', 'INT': 'NodeSocketInt', 'BOOLEAN': 'NodeSocketBool',
        'VECTOR': 'NodeSocketVector', 'GEOMETRY': 'NodeSocketGeometry', 'OBJECT': 'NodeSocketObject',
        'COLLECTION': 'NodeSocketCollection', 'MATRIX': 'NodeSocketMatrix'}


class G:
    def __init__(self, name, clear=True):
        ng = bpy.data.node_groups.get(name)
        if ng is None:
            ng = bpy.data.node_groups.new(name, 'GeometryNodeTree')
        elif clear:
            ng.nodes.clear()
            ng.interface.clear()
        self.ng = ng
        self.x = 0
        self.gin = None
        self.gout = None

    # ---- interface
    def inp(self, name, stype, default=None, mn=None, mx=None, desc="", subtype=None, structure=None, parent=None):
        s = self.ng.interface.new_socket(name, in_out='INPUT', socket_type=SOCK.get(stype, stype), parent=parent)
        if default is not None:
            s.default_value = default
        if mn is not None:
            s.min_value = mn
        if mx is not None:
            s.max_value = mx
        if desc:
            s.description = desc
        if subtype:
            s.subtype = subtype
        if structure:
            s.structure_type = structure
        return s

    def out(self, name, stype, structure=None, desc=""):
        s = self.ng.interface.new_socket(name, in_out='OUTPUT', socket_type=SOCK.get(stype, stype))
        if structure:
            s.structure_type = structure
        if desc:
            s.description = desc
        return s

    def panel(self, name, desc="", closed=True):
        p = self.ng.interface.new_panel(name, description=desc, default_closed=closed)
        return p

    def group_in(self):
        if self.gin is None:
            self.gin = self.ng.nodes.new('NodeGroupInput')
            self.gin.location = (-1400, 0)
        return self.gin

    def group_out(self):
        if self.gout is None:
            self.gout = self.ng.nodes.new('NodeGroupOutput')
            self.gout.location = (2600, 0)
        return self.gout

    def I(self, name):
        return self._find(self.group_in().outputs, name)

    def O(self, name):
        return self._find(self.group_out().inputs, name)

    # ---- nodes
    def node(self, typ, inputs=None, label=None, loc=None, **props):
        n = self.ng.nodes.new(typ)
        for k, v in props.items():
            setattr(n, k, v)
        if label:
            n.label = label
        self.x += 1
        n.location = loc if loc else ((self.x % 14) * 220 - 1100, -(self.x // 14) * 300)
        if inputs:
            for k, v in inputs.items():
                self.set(n, k, v)
        return n

    def set(self, n, key, v):
        sock = n.inputs[key] if isinstance(key, int) else self._find(n.inputs, key)
        if isinstance(v, bpy.types.NodeSocket):
            self.ng.links.new(v, sock)
        elif isinstance(v, bpy.types.Node):
            self.ng.links.new(v.outputs[0], sock)
        else:
            sock.default_value = v

    @staticmethod
    def _find(socks, key):
        for s in socks:
            if s.enabled and (s.name == key or s.identifier == key):
                return s
        for s in socks:
            if s.name == key or s.identifier == key:
                return s
        raise KeyError(key)

    def o(self, n, key=0):
        return n.outputs[key] if isinstance(key, int) else self._find(n.outputs, key)

    def link(self, a, b):
        self.ng.links.new(a, b)

    # ---- math helpers
    def math(self, op, a, b=None, c=None, label=None):
        n = self.node('ShaderNodeMath', operation=op, label=label)
        for i, v in enumerate((a, b, c)):
            if v is not None:
                self.set(n, i, v)
        return n.outputs[0]

    def vmath(self, op, a, b=None, c=None, out=0, label=None):
        n = self.node('ShaderNodeVectorMath', operation=op, label=label)
        for i, v in enumerate((a, b, c)):
            if v is not None:
                self.set(n, i, v)
        return n.outputs[out]

    def maprange(self, val, a, b, ta, tb, smooth=True):
        n = self.node('ShaderNodeMapRange', interpolation_type='SMOOTHSTEP' if smooth else 'LINEAR')
        self.set(n, 'Value', val); self.set(n, 'From Min', a); self.set(n, 'From Max', b)
        self.set(n, 'To Min', ta); self.set(n, 'To Max', tb)
        return n.outputs[0]

    def lerp(self, a, b, w):
        return self.math('MULTIPLY_ADD', w, self.math('SUBTRACT', b, a), a)

    def objinfo(self, sock):
        return self.node('GeometryNodeObjectInfo', {'Object': sock}, transform_space='RELATIVE')

    def sample(self, grid, pos, dtype='FLOAT'):
        return self.node('GeometryNodeSampleGrid', {'Grid': grid, 'Position': pos}, data_type=dtype).outputs[0]

    def mesh_sdf(self, mesh, vox, band, label=None):
        return self.node('GeometryNodeMeshToSDFGrid', {'Mesh': mesh, 'Voxel Size': vox, 'Band Width': band}, label=label).outputs[0]

    def to_bool(self, x):
        return self.node('FunctionNodeCompare', {'A': x, 'B': 0.5}, data_type='FLOAT', operation='GREATER_THAN').outputs[0]

    def sw(self, cond, a_false, b_true, typ='FLOAT'):
        n = self.node('GeometryNodeSwitch', input_type=typ)
        self.set(n, 'Switch', cond); self.set(n, 'False', a_false); self.set(n, 'True', b_true)
        return n.outputs[0]

    def xyz(self, x, y, z):
        return self.node('ShaderNodeCombineXYZ', {'X': x, 'Y': y, 'Z': z}).outputs[0]

    def sep(self, v):
        n = self.node('ShaderNodeSeparateXYZ', {'Vector': v})
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def pos(self):
        return self.node('GeometryNodeInputPosition').outputs[0]

    def nattr(self, name, dtype='FLOAT'):
        n = self.node('GeometryNodeInputNamedAttribute', data_type=dtype)
        n.inputs['Name'].default_value = name
        return n.outputs[0]

    def grid_to_mesh(self, grid, thr=0.0):
        return self.node('GeometryNodeGridToMesh', {'Grid': grid, 'Threshold': thr}).outputs[0]

    def points_sdf(self, points_mesh, radius_field, voxel, label=None):
        """SDF grid of spheres at the points of a mesh with the given per-point radius field."""
        pts = self.node('GeometryNodeMeshToPoints', {'Mesh': points_mesh, 'Radius': radius_field})
        s = self.node('GeometryNodePointsToSDFGrid', {'Points': pts.outputs[0], 'Voxel Size': voxel}, label=label)
        self.link(self.node('GeometryNodeInputRadius').outputs[0], s.inputs['Radius'])
        return s.outputs[0]
