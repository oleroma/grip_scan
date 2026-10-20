"""build_stage.py: (re)build the hand-fit stage node groups inside a .blend and install the
'hand fit' modifier on 'basemesh GripV' (after 'subdivision surface').

    PY=/tmp/claude-0/venv/bin/python
    $PY build_stage.py -- in.blend [out.blend] [--modules-dir DIR ...] [--set 'Name=value' ...]

Without out.blend the input file is overwritten (only do that on your own copy).

The stage is an SDF edit of the subdivided GripV basemesh. The main group 'HF Stage' computes
shared context (base SDF grids, mirrored position, hand points, static cutouts) and chains the
module groups in a fixed order:

    base SDF -> shape -> thenar -> mesh (coarse box + fine band) -> output

FINAL (smooth/final) version: only the smooth hand-fit shape and the thenar slope. The old height-field
fill, the capsule carve and the seats-with-re-carve modules are removed. Olex's controls live in the node
group 'GripV hand fit parameters' (shown in his dashboard 'GripV controls', frame 'HAND FIT'); every other
stage input stays on the modifier, hidden at its default.

Each module lives in its own file mod_<name>.py (see INTERFACE.md for the contract):
    NAME, GROUP, LABEL, DESCRIPTION   strings
    USES     list of context socket names it needs (keys of CONTEXT below)
    PARAMS   list of dicts: name, type (FLOAT/INT/BOOLEAN/VECTOR/OBJECT/COLLECTION), default, min, max,
             subtype, desc; shown on the modifier in a panel named LABEL
    OBJECT_DEFAULTS  {param name: object or collection name} set on the modifier after (re)build
    OUTPUTS  subset of ('Mask', 'Debug') besides the mandatory 'Phi'
    build(g) -> {'Phi': socket, 'Mask': socket?, 'Debug': geometry socket?}
             g is a gn.G bound to the module group, with the USES and PARAMS sockets already on
             its Group Input (g.I('Phi'), g.I('Press'), ...).
Module files are searched in the --modules-dir folders in the given order, then in this folder,
so a module agent can test its own mod_<name>.py in its own folder without touching the others.
"""
import os
import sys
import importlib.util
import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gn import G  # noqa: E402

MAIN_GROUP = 'HF Stage'
MOD_NAME = 'hand fit'
TARGET = 'basemesh GripV'
ORDER = ('shape', 'thenar')   # final: smooth shape + thenar (fill, carve and seats removed)

# ---- Olex's controls: these inputs are NOT modifier inputs; they are read from the node group
# 'GripV hand fit parameters' (values on its node 'hand fit values'), which also sits in 'GripV controls'.
CONTROLS_GROUP = 'GripV hand fit parameters'
VALUES_GROUP = 'hand fit values'
VALUES_NODE = 'hand fit values'
CONTROL_NAMES = ('Fit', 'Shape Size', 'Palm Give', 'Finger Give', 'Stroke Clear', 'Screw Wall', 'Thenar')   # SPEC v2 (2026-10-20)
# 'View' is NOT a dashboard control any more (review round 2): it switches the basemesh that every printed part is
# built from, so View 1 / 2 broke the prints (View 2 collapses case_top). It is a hidden modifier input (default 0);
# check.py and export_check.py refuse to run when it is not 0.
CONTROL_DESC = {   # Olex-facing descriptions of the controls taken from the stage / thenar inputs
    'Thenar': 'The slope behind the sticks for the thumb ball (moves the back edge of the top face forward). Acts at once',
    'View': 'What the case shows: 0 the result, 1 the plain remesh of GripV, 2 the coarse pass, 3+ debug meshes',
}
DASHBOARD = 'GripV controls'
DASHBOARD_FRAME = 'HAND FIT'

# name: (socket type, structure type, description)
CONTEXT = {
    'Phi': ('FLOAT', 'FIELD', 'Signed distance so far (mm, < 0 inside the case) after the previous modules'),
    'Base SDF': ('FLOAT', 'FIELD', 'SDF of the plain input mesh at the evaluation point (exact within 2.5 voxels of the surface, wide grid farther out)'),
    'Mirrored Position': ('VECTOR', 'FIELD', 'Evaluation position with x -> |x| when Mirror To Left is on (the hand is on the right handle, x > 0)'),
    'Base Grid': ('FLOAT', 'GRID', 'SDF grid of the input mesh at Voxel Size, narrow band (3 voxels)'),
    'Base Wide Grid': ('FLOAT', 'GRID', 'SDF grid of the input mesh at Fill Voxel Size, band = Wide Band mm'),
    'Base Mesh': ('GEOMETRY', None, 'The modifier input mesh (subdivided GripV basemesh, world = object space)'),
    'Hand Points': ('GEOMETRY', None, "Hand capsule axis points of the right hand: point attributes radius, bone_id, seg_first, seg_last and vertex group 'fill zones'"),
    'Cutouts': ('GEOMETRY', None, 'Static stick cutout meshes (realized), never the ones that read the basemesh'),
    'Voxel Size': ('FLOAT', None, 'Final resolution (mm)'),
    'Fill Voxel Size': ('FLOAT', None, 'Resolution of the coarse pass and of the wide grids (mm)'),
    'Press': ('FLOAT', None, 'How far the rigid hand sinks into the case (mm)'),
    'Edit Min X': ('FLOAT', None, 'Edits only where |x| is larger (handle region)'),
    'Mirror To Left': ('BOOLEAN', None, 'Edits mirrored to the left handle'),
}

GENERAL = [
    dict(name='Enable', type='BOOLEAN', default=True, desc='Off: the basemesh passes through unchanged (plain GripV)'),
    dict(name='View', type='INT', default=0, min=0, max=9, desc='Inspection only, keep 0: it switches the basemesh every part is built from (1 plain SDF remesh of the input, 2 coarse pass: case_top collapses, 3+ module debug meshes). check.py / export_check.py refuse View != 0'),
    dict(name='Hand Points', type='OBJECT', desc="Point cloud along the hand bones with a 'radius' attribute; follows 'hand rig' (Armature modifier)"),
    dict(name='Static Cutouts', type='COLLECTION', desc='Stick cutout meshes that do NOT read the basemesh (else: dependency cycle)'),
    dict(name='Voxel Size', type='FLOAT', default=0.5, min=0.2, max=3.0, subtype='DISTANCE', desc='Final resolution (mm). 0.5 ~ 6 s per update, 0.8 ~ 2-3 s'),
    dict(name='Fill Voxel Size', type='FLOAT', default=1.0, min=0.5, max=3.0, subtype='DISTANCE', desc='Coarse pass / wide grid resolution (mm)'),
    dict(name='Fine Band', type='INT', default=4, min=2, max=20, desc='Width (voxels) of the fine pass around the coarse result'),
    dict(name='Wide Band', type='FLOAT', default=16.0, min=4.0, max=40.0, subtype='DISTANCE', desc='How far from the surface the wide base grid is exact (mm)'),
    dict(name='Box Margin', type='FLOAT', default=10.0, min=2.0, max=40.0, subtype='DISTANCE', desc='The coarse pass covers the input bounding box plus this margin; growth beyond it is clipped'),
    dict(name='Press', type='FLOAT', default=0.5, min=0.0, max=3.0, subtype='DISTANCE', desc='How far the rigid hand sinks into the case (mm)'),
    dict(name='Edit Min X', type='FLOAT', default=25.0, min=0.0, max=80.0, subtype='DISTANCE', desc='Edits only where |X| is larger (handle region)'),
    dict(name='Mirror To Left', type='BOOLEAN', default=True, desc='Apply the same edits to the left handle (mirrored hand)'),
    dict(name='Min Island', type='INT', default=5000, min=0, max=100000, desc='Delete loose mesh pieces with fewer vertices than this'),
    dict(name='Adaptivity', type='FLOAT', default=0.0, min=0.0, max=1.0, subtype='FACTOR', desc='Grid to Mesh adaptivity (keep 0: adaptivity can tilt the flat top)'),
    dict(name='Snap Mirror Seam', type='BOOLEAN', default=True, desc="Put the seam vertices exactly on x = 0 (the generator's 'battery face' group selects verts with x == 0)"),
]
GENERAL_OBJECT_DEFAULTS = {'Hand Points': 'hand points', 'Static Cutouts': 'cutout flat'}


# ------------------------------------------------------------------------------- modules
def find_module_file(name, dirs):
    for d in list(dirs or []) + [HERE]:
        f = os.path.join(d, f'mod_{name}.py')
        if os.path.exists(f):
            return f
    raise FileNotFoundError(f'mod_{name}.py not found in {dirs} or {HERE}')


def load_module(name, dirs=None):
    f = find_module_file(name, dirs)
    spec = importlib.util.spec_from_file_location(f'hf_mod_{name}_{abs(hash(f))}', f)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(f))
    try:
        spec.loader.exec_module(m)
    finally:
        sys.path.pop(0)
    m.__file_path__ = f
    for attr in ('NAME', 'GROUP', 'LABEL', 'USES', 'PARAMS', 'build'):
        if not hasattr(m, attr):
            raise AttributeError(f'{f}: module must define {attr}')
    bad = [u for u in m.USES if u not in CONTEXT]
    if bad:
        raise KeyError(f'{f}: unknown context sockets {bad}; allowed: {list(CONTEXT)}')
    return m


def _hide(sock):
    # Blender 5.2: an interface socket can be hidden from the modifier panel (still settable by script)
    if hasattr(sock, 'hide_in_modifier'):
        sock.hide_in_modifier = True
    return sock


def _socket_value(sock):
    v = sock.default_value
    return tuple(v) if hasattr(v, '__len__') and not isinstance(v, str) else v


def ensure_controls(specs):
    """(Re)build 'hand fit values' (inputs -> same outputs) and 'GripV hand fit parameters' (one 'hand fit values'
    node whose unlinked inputs hold Olex's values -> group outputs), keeping the values already in the file.
    Also put one node of it into the dashboard 'GripV controls' (frame 'HAND FIT')."""
    old = {}
    cg = bpy.data.node_groups.get(CONTROLS_GROUP)
    if cg is not None and cg.nodes.get(VALUES_NODE) is not None:
        for sk in cg.nodes[VALUES_NODE].inputs:
            if hasattr(sk, 'default_value'):
                old[sk.name] = _socket_value(sk)
    vg = G(VALUES_GROUP)
    vg.ng.description = 'The values of the hand fit controls (edit them on the node inside GripV hand fit parameters)'
    specs = [dict(p, desc=CONTROL_DESC.get(p['name'], p.get('desc', ''))) for p in specs]
    for p in specs:
        _add_param(vg, p)
    for p in specs:
        vg.out(p['name'], p['type'], desc=p.get('desc', ''))
    for p in specs:
        vg.link(vg.I(p['name']), vg.O(p['name']))
    c = G(CONTROLS_GROUP)
    c.ng.description = ("Olex's hand fit controls. Fit / Thenar / View act at once; Shape Size, Palm Give, Finger Give, Stroke Clear "
                        "and Screw Wall are inputs of the shape solve: they change nothing until you run final/refit.py")
    for p in specs:
        c.out(p['name'], p['type'], desc=p.get('desc', ''))
    fr = c.node('NodeFrame', label='HAND FIT: edit the values on this node', loc=(-300, 300))
    vn = c.node('GeometryNodeGroup', label=VALUES_NODE, loc=(-200, 0))
    vn.node_tree = vg.ng
    vn.name = VALUES_NODE
    vn.parent = fr
    vn.width = 260
    for p in specs:
        c.link(vn.outputs[p['name']], c.O(p['name']))
        sk = vn.inputs[p['name']]
        if p['name'] in old:
            try:
                sk.default_value = old[p['name']]
            except Exception:
                pass
        elif p.get('default') is not None:
            sk.default_value = p['default']
    # the dashboard
    d = bpy.data.node_groups.get(DASHBOARD)
    if d is not None and d.nodes.get(CONTROLS_GROUP) is None:
        frs = [n for n in d.nodes if n.bl_idname == 'NodeFrame']
        x0 = min((n.location.x for n in d.nodes), default=0.0)
        y0 = min((n.location.y for n in d.nodes), default=0.0)
        f2 = d.nodes.new('NodeFrame')
        f2.label = DASHBOARD_FRAME
        f2.name = 'Frame HAND FIT'
        n2 = d.nodes.new('GeometryNodeGroup')
        n2.node_tree = c.ng
        n2.name = CONTROLS_GROUP
        n2.label = CONTROLS_GROUP
        n2.location = (x0, y0 - 400)
        n2.parent = f2
        n2.width = 240
    return c.ng


def _values_node():
    cg = bpy.data.node_groups.get(CONTROLS_GROUP)
    return cg.nodes.get(VALUES_NODE) if cg is not None else None


def _add_param(g, p, parent=None):
    return g.inp(p['name'], p['type'], p.get('default'), p.get('min'), p.get('max'), p.get('desc', ''),
                 p.get('subtype'), None, parent)


def build_module_group(m):
    """Create the module node group with its interface; call m.build(g) to fill it."""
    g = G(m.GROUP)
    g.ng.description = getattr(m, 'DESCRIPTION', '')
    for u in m.USES:
        typ, struct, desc = CONTEXT[u]
        g.inp(u, typ, desc=desc, structure=struct)
    names = set(m.USES)
    for p in m.PARAMS:
        if p['name'] in names:
            raise KeyError(f"{m.GROUP}: parameter '{p['name']}' clashes with a context socket")
        names.add(p['name'])
        _add_param(g, p)
    g.out('Phi', 'FLOAT', structure='FIELD', desc='Signed distance after this module')
    outs = getattr(m, 'OUTPUTS', ())
    if 'Mask' in outs:
        g.out('Mask', 'FLOAT', structure='FIELD', desc='0..1 where this module acts (stored as attribute hf_<name>)')
    if 'Debug' in outs:
        g.out('Debug', 'GEOMETRY', desc='Inspection mesh shown by View')
    res = m.build(g)
    for k, v in res.items():
        if v is not None:
            g.link(v, g.O(k))
    return g.ng


# ---------------------------------------------------------------------------- main group
def build_main(mods):
    """mods: list of loaded module objects in ORDER."""
    g = G(MAIN_GROUP)
    ng = g.ng
    ng.description = ("Hand fit stage for GripV: smooth hand-fit shape (solved offset) + thenar slope on the subdivided "
                      "basemesh, then a remesh. Every case part reads the result. Olex's controls: node group "
                      "'GripV hand fit parameters' (dashboard 'GripV controls', frame 'HAND FIT').")
    g.inp('Geometry', 'GEOMETRY')
    g.out('Geometry', 'GEOMETRY')
    specs = {p['name']: p for p in GENERAL}
    for m in mods:
        for p in m.PARAMS:
            specs.setdefault(p['name'], p)
    for p in GENERAL:
        if p['name'] not in CONTROL_NAMES:
            _hide(_add_param(g, p))
    seen = {p['name'] for p in GENERAL} | {'Geometry'}
    for m in mods:
        pan = None
        for p in m.PARAMS:
            if p['name'] in seen:
                raise KeyError(f"parameter name '{p['name']}' of module {m.NAME} is already used by another module or the stage")
            seen.add(p['name'])
            if p['name'] in CONTROL_NAMES:
                continue
            if pan is None:
                pan = g.panel(m.LABEL + ' (advanced)', getattr(m, 'DESCRIPTION', ''), closed=True)
            _hide(_add_param(g, p, parent=pan))
    # Olex's controls: one group node of 'GripV hand fit parameters' feeds them (instead of modifier inputs)
    cg = ensure_controls([specs[n] for n in CONTROL_NAMES])
    cn = g.node('GeometryNodeGroup', label=CONTROLS_GROUP, loc=(-1400, 900))
    cn.node_tree = cg
    cn.name = CONTROLS_GROUP
    _orig_I = g.I

    def _I(name):
        return cn.outputs[name] if name in CONTROL_NAMES else _orig_I(name)
    g.I = _I
    I, M, V = g.I, g.math, g.vmath
    vox, vf = I('Voxel Size'), I('Fill Voxel Size')

    # ---- shared context
    base = I('Geometry')
    C = g.mesh_sdf(base, vox, 3, 'Base Grid')
    wide = M('ADD', M('CEIL', M('DIVIDE', I('Wide Band'), vf)), 2.0)
    Cw = g.mesh_sdf(base, vf, wide, 'Base Wide Grid')
    bb = g.node('GeometryNodeBoundBox', {'Geometry': base})
    pos = g.pos()
    px, py, pz = g.sep(pos)
    mx = g.sw(I('Mirror To Left'), px, M('ABSOLUTE', px))
    pm = g.xyz(mx, py, pz)
    # base SDF sampled at the mirrored position when mirroring: the field is then exactly symmetric,
    # so the remesh has exactly mirrored vertex pairs at the seam (GripV itself is symmetric, MirrorX)
    pb = g.sw(I('Mirror To Left'), pos, pm, typ='VECTOR')
    dC = g.sample(C, pb)
    dCw = g.sample(Cw, pb)
    near = M('LESS_THAN', M('ABSOLUTE', dC), M('MULTIPLY', vox, 2.5))
    dCx = g.sw(g.to_bool(near), dCw, dC)
    hand = g.objinfo(I('Hand Points')).outputs['Geometry']
    cut = g.node('GeometryNodeCollectionInfo', {'Collection': I('Static Cutouts')}, transform_space='RELATIVE')
    cut = g.node('GeometryNodeRealizeInstances', {'Geometry': cut.outputs[0]}).outputs[0]
    ctx = {'Base SDF': dCx, 'Mirrored Position': pm, 'Base Grid': C, 'Base Wide Grid': Cw, 'Base Mesh': base,
           'Hand Points': hand, 'Cutouts': cut, 'Voxel Size': vox, 'Fill Voxel Size': vf, 'Press': I('Press'),
           'Edit Min X': I('Edit Min X'), 'Mirror To Left': I('Mirror To Left')}

    # ---- module chain
    phi = dCx
    masks, debugs = [], []
    for i, m in enumerate(mods):
        n = g.node('GeometryNodeGroup', label=m.LABEL, loc=(i * 300, 600))
        n.node_tree = bpy.data.node_groups[m.GROUP]
        for u in m.USES:
            g.set(n, u, phi if u == 'Phi' else ctx[u])
        for p in m.PARAMS:
            g.link(I(p['name']), g._find(n.inputs, p['name']))
        phi = n.outputs['Phi']
        if 'Mask' in getattr(m, 'OUTPUTS', ()):
            masks.append((m.NAME, n.outputs['Mask']))
        if 'Debug' in getattr(m, 'OUTPUTS', ()):
            debugs.append((m.NAME, n.outputs['Debug']))

    # ---- coarse pass: dense box at Fill Voxel Size (aligned to its lattice)
    bmin = g.sep(bb.outputs['Min'])
    bmax = g.sep(bb.outputs['Max'])
    marg = I('Box Margin')
    lo = [M('MULTIPLY', M('FLOOR', M('DIVIDE', M('SUBTRACT', c, marg), vf)), vf) for c in bmin]
    hi = [M('MULTIPLY', M('CEIL', M('DIVIDE', M('ADD', c, marg), vf)), vf) for c in bmax]
    res = [M('ROUND', M('DIVIDE', M('SUBTRACT', h, l), vf)) for l, h in zip(lo, hi)]
    topo = g.node('GeometryNodeCubeGridTopology', {'Bounds Min': g.xyz(*lo), 'Bounds Max': g.xyz(*hi),
                                                   'Resolution X': res[0], 'Resolution Y': res[1], 'Resolution Z': res[2]})
    fa = g.node('GeometryNodeFieldToGrid', label='F2G coarse')
    fa.grid_items.new('FLOAT', 'Value')
    g.set(fa, 'Topology', topo.outputs[0])
    g.set(fa, 'Value', phi)
    mesh_a = g.grid_to_mesh(fa.outputs[0])
    # ---- fine pass: the same field at Voxel Size in a band around the coarse result
    S = g.mesh_sdf(mesh_a, vox, I('Fine Band'), 'S')
    fb = g.node('GeometryNodeFieldToGrid', label='F2G fine')
    fb.grid_items.new('FLOAT', 'Value')
    g.set(fb, 'Topology', S)
    g.set(fb, 'Value', phi)
    hi_bg = g.node('GeometryNodeSetGridBackground', {'Grid': fb.outputs[0], 'Background': 10.0, 'Update Inactive': True}).outputs[0]
    fn = g.node('GeometryNodeFieldToGrid', label='F2G neg')
    fn.grid_items.new('FLOAT', 'Value')
    g.set(fn, 'Topology', S)
    g.set(fn, 'Value', M('MULTIPLY', g.sample(fb.outputs[0], g.pos()), -1.0))
    neg_bg = g.node('GeometryNodeSetGridBackground', {'Grid': fn.outputs[0], 'Background': 10.0, 'Update Inactive': True}).outputs[0]
    u = g.node('GeometryNodeSDFGridBoolean', operation='UNION')          # min(S, phi) in the band
    g.link(S, u.inputs[1]); g.link(hi_bg, u.inputs[1])
    x = g.node('GeometryNodeSDFGridBoolean', operation='DIFFERENCE', label='final')   # max(.., phi): = phi in the band, S outside
    g.link(u.outputs[0], x.inputs[0]); g.link(neg_bg, x.inputs[1])
    final = g.node('GeometryNodeGridToMesh', {'Grid': x.outputs[0], 'Threshold': 0.0, 'Adaptivity': I('Adaptivity')}).outputs[0]
    # winding fix (2026-10-20): Grid to Mesh winds these level-set meshes inward (measured: skin signed volume
    # -437,995 mm3, every case part inside-out, the 'cable holder' boolean doubled to 2,935 mm3). Flip the faces
    # here so the skin points outward like plain GripV; the Enable-off pass-through is not touched.
    final = g.node('GeometryNodeFlipFaces', {'Mesh': final}, label='outward winding').outputs[0]
    # seam: Grid to Mesh puts vertices at cell centres (x = +-Voxel/2), never on x = 0. Snap the first
    # column to x = 0 and weld the (exactly mirrored) pairs; the strip of quads between them collapses.
    ps = g.pos()
    sx, sy, sz = g.sep(ps)
    on_seam = g.node('FunctionNodeBooleanMath', {0: g.to_bool(M('LESS_THAN', M('ABSOLUTE', sx), M('MULTIPLY', vox, 0.75))),
                                                 1: I('Snap Mirror Seam')}, operation='AND').outputs[0]
    snap = g.node('GeometryNodeSetPosition', {'Geometry': final, 'Selection': on_seam, 'Position': g.xyz(0.0, sy, sz)})
    sx2 = g.sep(g.pos())[0]
    at0 = g.node('FunctionNodeBooleanMath', {0: g.to_bool(M('LESS_THAN', M('ABSOLUTE', sx2), 1e-9)), 1: I('Snap Mirror Seam')}, operation='AND').outputs[0]
    weld = g.node('GeometryNodeMergeByDistance', {'Geometry': snap.outputs[0], 'Selection': at0, 'Distance': 1e-4})
    final = weld.outputs[0]

    # ---- attributes: change_mm (+ grown / - carved vs the input) and module masks
    p2 = g.pos()
    d2 = g.sample(C, p2)
    d2w = g.sample(Cw, p2)
    chg = g.node('GeometryNodeStoreNamedAttribute', data_type='FLOAT', domain='POINT')
    g.set(chg, 'Geometry', final); g.set(chg, 'Name', 'change_mm')
    g.set(chg, 'Value', g.sw(g.to_bool(M('LESS_THAN', M('ABSOLUTE', d2), M('MULTIPLY', vox, 2.5))), d2w, d2))
    geo = chg.outputs[0]
    for name, mk in masks:
        st = g.node('GeometryNodeStoreNamedAttribute', data_type='FLOAT', domain='POINT')
        g.set(st, 'Geometry', geo); g.set(st, 'Name', f'hf_{name}'); g.set(st, 'Value', mk)
        geo = st.outputs[0]
    isl = g.node('GeometryNodeInputMeshIsland')
    acc = g.node('GeometryNodeAccumulateField', data_type='FLOAT', domain='POINT')
    g.set(acc, 'Value', 1.0)
    g.link(isl.outputs['Island Index'], acc.inputs['Group ID'])
    small = M('LESS_THAN', acc.outputs['Total'], I('Min Island'))
    dl = g.node('GeometryNodeDeleteGeometry', domain='POINT')
    g.set(dl, 'Geometry', geo); g.set(dl, 'Selection', g.to_bool(small))
    sm = g.node('GeometryNodeSetShadeSmooth', {'Geometry': dl.outputs[0]})

    # ---- view switch + enable
    def outward(m):      # inspection meshes from Grid to Mesh: same winding fix as the result
        return g.node('GeometryNodeFlipFaces', {'Mesh': m}, label='outward winding').outputs[0]
    # View 1 (the plain remesh of the input grid C) already comes out of Grid to Mesh wound outward (measured in
    # review round 2: flipped, its skin had -433,723 mm3): no flip there. View 2 (the coarse pass) needs it.
    views = [sm.outputs[0], g.grid_to_mesh(C), outward(mesh_a)] + [d for _, d in debugs]
    isw = g.node('GeometryNodeIndexSwitch', data_type='GEOMETRY')
    while len(isw.index_switch_items) < len(views):
        isw.index_switch_items.new()
    g.set(isw, 'Index', I('View'))
    for k, v in enumerate(views):
        g.link(v, isw.inputs[k + 1])
    out = g.sw(I('Enable'), base, isw.outputs[0], typ='GEOMETRY')
    g.link(out, g.O('Geometry'))
    vdesc = 'Inspection only, keep 0 (every part is built from this mesh): 0 result, 1 plain SDF remesh of the input, 2 coarse pass' + ''.join(f', {3 + k} {n} debug' for k, (n, _) in enumerate(debugs))
    for it in list(ng.interface.items_tree) + list(bpy.data.node_groups[VALUES_GROUP].interface.items_tree):
        if getattr(it, 'in_out', None) == 'INPUT' and it.name == 'View':
            it.description = vdesc
            it.max_value = len(views) - 1
    return ng, [n for n, _ in debugs]


# ------------------------------------------------------------------------- modifier I/O
def input_ids(mod):
    return {it.name: it.identifier for it in mod.node_group.interface.items_tree
            if getattr(it, 'in_out', None) == 'INPUT' and it.item_type == 'SOCKET'}


def set_inputs(mod, values, update=True):
    """Blender 5.2: modifier inputs live in mod.properties.inputs.<identifier>.value.
    Object / collection inputs accept a name (str) or the datablock."""
    ids = input_ids(mod)
    types = {it.name: it.socket_type for it in mod.node_group.interface.items_tree
             if getattr(it, 'in_out', None) == 'INPUT' and it.item_type == 'SOCKET'}
    vn = _values_node()
    for k, v in values.items():
        if k not in ids and k in CONTROL_NAMES and vn is not None:
            vn.inputs[k].default_value = v
            continue
        if k not in ids:
            raise KeyError(f"modifier has no input '{k}'. Inputs: {list(ids)}")
        if types[k] == 'NodeSocketObject' and isinstance(v, str):
            v = bpy.data.objects[v]
        elif types[k] == 'NodeSocketCollection' and isinstance(v, str):
            v = bpy.data.collections[v]
        getattr(mod.properties.inputs, ids[k]).value = v
    if update:
        # tag only (verified: without ob.data.update() the next evaluation is stale). We do not call
        # view_layer.update() here: in the GripV scene that re-evaluates ~40 objects (~35 s).
        mod.show_viewport = mod.show_viewport
        ob = mod.id_data
        ob.data.update()


def get_inputs(mod):
    out = {}
    for k, i in input_ids(mod).items():
        it = getattr(mod.properties.inputs, i, None)
        if it is None or not hasattr(it, 'value'):
            continue
        v = it.value
        if hasattr(v, 'name') and hasattr(v, 'rna_type') and not isinstance(v, (int, float, str)):
            v = v.name
        elif hasattr(v, '__len__') and not isinstance(v, str):
            v = tuple(v)
        elif isinstance(v, float):
            v = round(v, 5)
        out[k] = v
    vn = _values_node()
    if vn is not None:
        for sk in vn.inputs:
            if sk.name in CONTROL_NAMES and hasattr(sk, 'default_value'):
                v = _socket_value(sk)
                out[sk.name] = round(v, 5) if isinstance(v, float) else v
    return out


def parse_value(s):
    s = s.strip()
    if s.lower() in ('true', 'false'):
        return s.lower() == 'true'
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


# ------------------------------------------------------------------------------- install
def build(modules_dirs=None, keep_values=False, values=None, target=TARGET, verbose=True):
    """Rebuild all stage groups in the open file and make sure the modifier is installed.
    Returns (modifier, info dict)."""
    old = {}
    ob = bpy.data.objects[target]
    mod = ob.modifiers.get(MOD_NAME)
    if mod is not None and mod.node_group is not None and keep_values:
        old = get_inputs(mod)
    mods = [load_module(n, modules_dirs) for n in ORDER]
    for gname in ('HF fill', 'HF carve', 'HF seats'):
        if gname in bpy.data.node_groups:
            bpy.data.node_groups.remove(bpy.data.node_groups[gname])
    for m in mods:
        build_module_group(m)
    ng, debug_names = build_main(mods)
    if mod is None:
        mod = ob.modifiers.new(MOD_NAME, 'NODES')
    mod.node_group = ng
    # keep it right after 'subdivision surface'
    names = [m_.name for m_ in ob.modifiers]
    if 'subdivision surface' in names:
        want = names.index('subdivision surface') + 1
        cur = names.index(MOD_NAME)
        if cur != want:
            with bpy.context.temp_override(object=ob):
                bpy.ops.object.modifier_move_to_index(modifier=MOD_NAME, index=want if cur > want else want - 1)
    defaults = dict(GENERAL_OBJECT_DEFAULTS)
    for m in mods:
        defaults.update(getattr(m, 'OBJECT_DEFAULTS', {}))
    # a rebuilt interface leaves an existing modifier's inputs at 0, not at the socket defaults:
    # write every default explicitly
    vals = {}
    for it in ng.interface.items_tree:
        if getattr(it, 'in_out', None) == 'INPUT' and it.item_type == 'SOCKET' and hasattr(it, 'default_value') \
                and it.socket_type not in ('NodeSocketObject', 'NodeSocketCollection', 'NodeSocketGeometry'):
            dv = it.default_value
            vals[it.name] = tuple(dv) if hasattr(dv, '__len__') else dv
    for k, v in defaults.items():
        if (k in input_ids(mod)) and ((v in bpy.data.objects) or (v in bpy.data.collections)):
            vals[k] = v
    if keep_values:
        vals.update({k: v for k, v in old.items() if k in input_ids(mod)})
    if values:
        vals.update(values)
    set_inputs(mod, vals)
    info = {'modules': {m.NAME: m.__file_path__ for m in mods}, 'debug_views': {3 + i: n for i, n in enumerate(debug_names)},
            'modifier_stack': [m_.name for m_ in ob.modifiers]}
    if verbose:
        print('HF STAGE built:', info, flush=True)
    return mod, info


if __name__ == '__main__':
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    dirs, sets, pos_args = [], {}, []
    i = 0
    while i < len(args):
        if args[i] == '--modules-dir':
            dirs.append(args[i + 1]); i += 2
        elif args[i] == '--set':
            k, v = args[i + 1].split('=', 1); sets[k] = parse_value(v); i += 2
        else:
            pos_args.append(args[i]); i += 1
    bpy.ops.wm.open_mainfile(filepath=pos_args[0])
    build(dirs, keep_values=False, values=sets)
    bpy.ops.wm.save_as_mainfile(filepath=pos_args[1] if len(pos_args) > 1 else pos_args[0])
