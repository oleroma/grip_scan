"""Thenar module: move the back edge of the flat top forward toward the sticks and round it into a
smooth slope behind the sticks, so the thenar (the ball of the thumb) rests on a smooth surface.

What it does (default decision 6; the only module allowed to change the case above Z = 40):
  The case is cut by a 'thenar cutter': seen from the side (the Y-Z plane) its edge is
      flat top (unchanged)  ->  a round of radius 'Thenar Round'  ->  a straight slope at 'Thenar Slope'
  going down and back until it meets the old back wall, where it is blended in over 'Thenar Blend'.
  The flat top ends 'Thenar Move' mm in front of where it ended before (its back edge, the Y min of the
  top face, Y -69.13 on plain GripV). The cutter is the same for every X (both sides alike, so the left
  half is the mirror image); it only removes material, only above 'Thenar Min Z', and never touches the
  flat top in front of the new edge (that stays exactly flat at max Z; nothing is raised).

  Thenar Auto (on): the move is found automatically: the largest move (bisection, 24 steps, 0..25 mm)
  for which the slope stays at least 'Thenar Clearance' away from
      - the static stick cutout meshes (context 'Cutouts' = collection 'cutout flat': 'Stick R cutout flat',
        'StickL cutout flat', 'Stick R cutout appendix'), every vertex checked, and
      - the stick holes, as vertical cylinders computed from 'Thenar Hole R' / 'Thenar Hole L'
        (centre X, centre Y, radius at the top face; the generator's holes are r 9.0 at the top and
        8.58 from 1 mm below it). The 'hole' objects are not read: they read the basemesh (cycle).
  The distance is measured to the round + slope part of the cutter (exact, in the side profile), minus
  the most the wall blend can pull the slope in at that height; the target is Thenar Clearance + half a
  Voxel Size, because the remeshed edge can sit up to half a voxel in front of the exact one (measured:
  0.05 mm closer than the exact cutter without this margin). On plain GripV the stick holes decide it:
  the flat top keeps a rim of 0.8 mm (+ half a voxel) behind the left stick hole (the cutter is the same
  on both sides, so the left hole, 0.5 mm farther back than the right one, sets the move for both).
  Thenar Auto (off): 'Thenar Move' is used as given (no check; read the clearance in the report).

  The move that was used is stored on the Debug geometry (View 5 with the current modules: 3 fill,
  4 carve, 5 thenar) as attribute 'thenar_move_mm' (also 'thenar_edge_y' = the new back edge Y, and
  'thenar_clearance' on the check points); look at it in the spreadsheet.

  Measured on Grip_hand.blend (GripV, subdivision high, Voxel Size 0.5, Carve Max Z 40, 2026-10-19):
  auto move 6.58 mm (exact edge Y -69.13 -> -62.55); meshed top face Y min -68.75 -> -62.25 on both sides;
  nearest stick hole / static cutout to the new slope 1.25 mm (left hole) / 2.58 mm ('StickL cutout flat');
  top face max Z 59.04002 with nothing above; max Y -4.158; case_top closed, 1 island, top plate present;
  R1/L1 unchanged. Slope: 30-40 deg is the useful range; at 45 deg and more the slope runs nearly parallel
  to the back wall and cuts down to Thenar Min Z. Blend: keep it at 3 or less (with Blend 4-6 the generator's
  booleans made a 5-vertex, zero-volume sliver island in case_top in 2 of 7 runs; 0 of 12 with Blend <= 3).
  Beyond |X| ~ 57 the old outline is already in front of the new edge, so the slope ends there (the top
  outline has a small corner, ~27 deg, where the straight new edge meets the old curved outline).

Inputs used (context): Phi, Base Mesh (max Z and the old back edge), Cutouts (static stick cutouts),
Voxel Size (auto margin).
Output: Phi = lerp(Phi, smooth_max(Phi, cutter, k), w); w = Thenar on and above Thenar Min Z (2 mm fade);
k = Thenar Blend below max Z - 5 mm, fading to 0 at max Z - 1.5 mm (so the flat top stays exact).
Mask: where the cutter removed material. Debug: new edge line + clearance check points.
"""
import math
from gn import G

NAME = 'thenar'
GROUP = 'HF thenar'
PROFILE = 'HF thenar profile'
LABEL = 'Thenar'
DESCRIPTION = "Move the back edge of the flat top toward the sticks and round it into a slope for the thenar"
USES = ['Phi', 'Base Mesh', 'Cutouts', 'Voxel Size']
MAX_MOVE = 25.0       # auto search range 0..MAX_MOVE mm
ITER = 24             # bisection steps (25 mm / 2^24: far below a voxel)
HOLE_DEPTH = 13.84    # the stick holes reach 13.84 mm below the top face (Z 45.2 on GripV)
PARAMS = [
    dict(name='Thenar', type='BOOLEAN', default=True,
         desc='Thenar slope on/off. Off: the case top is not changed'),
    dict(name='Thenar Auto', type='BOOLEAN', default=True,
         desc='On: move the back edge as far forward as Thenar Clearance allows (found automatically). Off: use Thenar Move'),
    dict(name='Thenar Move', type='FLOAT', default=6.0, min=0.0, max=25.0, subtype='DISTANCE',
         desc='How far the back edge of the flat top moves forward (mm). Used only when Thenar Auto is off'),
    dict(name='Thenar Clearance', type='FLOAT', default=0.8, min=0.0, max=5.0, subtype='DISTANCE',
         desc='Thenar Auto keeps the slope at least this far from the stick cutouts and stick holes'),
    dict(name='Thenar Slope', type='FLOAT', default=math.radians(35.0), min=math.radians(10.0), max=math.radians(60.0), subtype='ANGLE',
         desc='Steepness of the slope behind the sticks (from horizontal). Steeper = the slope reaches lower down the back'),
    dict(name='Thenar Round', type='FLOAT', default=6.0, min=0.0, max=15.0, subtype='DISTANCE',
         desc='Radius of the rounding where the slope meets the flat top (0 = sharp edge)'),
    dict(name='Thenar Blend', type='FLOAT', default=3.0, min=0.0, max=10.0, subtype='DISTANCE',
         desc='Rounding where the slope meets the old back wall (fades out between 5 and 1.5 mm below the flat top)'),
    dict(name='Thenar Min Z', type='FLOAT', default=40.0, min=-20.0, max=59.0, subtype='DISTANCE',
         desc='The slope never cuts below this height (fades out over 2 mm above it)'),
    dict(name='Thenar Hole R', type='VECTOR', default=(34.0, -52.0, 9.0),
         desc='Right stick hole for the clearance check: centre X, centre Y, radius at the top face'),
    dict(name='Thenar Hole L', type='VECTOR', default=(-23.5, -52.5, 9.0),
         desc='Left stick hole for the clearance check: centre X, centre Y, radius at the top face'),
]
OBJECT_DEFAULTS = {}
OUTPUTS = ('Mask', 'Debug')


# ------------------------------------------------------------------------------------------ profile
def _profile_group():
    """Node group: signed distance to the thenar cutter (side profile, the same for every X).

    Profile in the Y-Z plane: flat top Z = Top Z for Y >= Edge Y, then an arc of radius Round tangent
    to it at Edge Y, then a straight line at Slope going down and back (-Y). Solid = below/in front.
      SDF   exact signed distance to that region (< 0 inside = material kept)
      Clear distance to the arc + slope part only (+ inside, - outside): the clearance of a point
            inside the case to the new slope (the flat top part does not count)
    """
    p = G(PROFILE)
    p.ng.description = 'Thenar cutter profile: exact signed distance (SDF) and clearance to the round + slope (Clear)'
    p.inp('Position', 'VECTOR', desc='Evaluation position')
    p.inp('Edge Y', 'FLOAT', desc='Y where the flat top ends (the new back edge)')
    p.inp('Top Z', 'FLOAT', desc='Height of the flat top')
    p.inp('Slope', 'FLOAT', subtype='ANGLE', desc='Slope angle from horizontal')
    p.inp('Round', 'FLOAT', subtype='DISTANCE', desc='Radius of the round between flat top and slope')
    p.out('SDF', 'FLOAT', desc='Signed distance to the cutter region (< 0 inside)')
    p.out('Clear', 'FLOAT', desc='Signed distance to the round + slope part only (+ inside)')
    I, M, V = p.I, p.math, p.vmath

    def hyp(a, b):
        return V('LENGTH', p.xyz(a, b, 0.0), out=1)

    _, py, pz = p.sep(I('Position'))
    a, R = I('Slope'), I('Round')
    sa, ca = M('SINE', a), M('COSINE', a)
    ry = M('SUBTRACT', py, I('Edge Y'))                               # relative to the arc centre
    rv = M('SUBTRACT', pz, M('SUBTRACT', I('Top Z'), R))
    # wedge W = {rv <= 0} and {B <= 0} with apex at the arc centre; cutter = W dilated by R
    B = M('SUBTRACT', M('MULTIPLY', ca, rv), M('MULTIPLY', sa, ry))
    mAB = M('MAXIMUM', rv, B)
    d1 = hyp(M('MINIMUM', ry, 0.0), rv)                               # to the ray along +Y
    s2 = M('MAXIMUM', M('SUBTRACT', M('MULTIPLY', M('MULTIPLY', ca, -1.0), ry), M('MULTIPLY', sa, rv)), 0.0)
    d2 = hyp(M('MULTIPLY_ADD', s2, ca, ry), M('MULTIPLY_ADD', s2, sa, rv))   # to the ray down the slope
    out = p.to_bool(M('GREATER_THAN', mAB, 0.0))
    sdW = p.sw(out, mAB, M('MINIMUM', d1, d2))
    sdf = M('SUBTRACT', sdW, R)
    # distance to the round + slope only
    rr = hyp(ry, rv)
    th = M('ARCTAN2', M('MULTIPLY', ry, -1.0), rv)                   # 0 = straight up, Slope = slope normal
    in_sec = p.node('FunctionNodeBooleanMath', {0: p.to_bool(M('GREATER_THAN', th, -1e-7)),
                                                1: p.to_bool(M('LESS_THAN', th, M('ADD', a, 1e-7)))}, operation='AND').outputs[0]
    psy = M('MULTIPLY', M('MULTIPLY', R, sa), -1.0)                  # slope tangent point (rel. to the centre)
    psz = M('MULTIPLY', R, ca)
    e0 = hyp(ry, M('SUBTRACT', rv, R))                                # arc end at the flat top
    qy, qz = M('SUBTRACT', ry, psy), M('SUBTRACT', rv, psz)
    e1 = hyp(qy, qz)                                                  # arc end at the slope
    darc = p.sw(in_sec, M('MINIMUM', e0, e1), M('ABSOLUTE', M('SUBTRACT', rr, R)))
    s = M('MAXIMUM', M('SUBTRACT', M('MULTIPLY', M('MULTIPLY', ca, -1.0), qy), M('MULTIPLY', sa, qz)), 0.0)
    dray = hyp(M('MULTIPLY_ADD', s, ca, qy), M('MULTIPLY_ADD', s, sa, qz))
    d = M('MINIMUM', darc, dray)
    clear = p.sw(p.to_bool(M('GREATER_THAN', sdf, 0.0)), d, M('MULTIPLY', d, -1.0))
    p.link(sdf, p.O('SDF'))
    p.link(clear, p.O('Clear'))
    return p.ng


def _profile(g, ng, pos, ye, zt, slope, rnd):
    n = g.node('GeometryNodeGroup', label='thenar profile')
    n.node_tree = ng
    g.set(n, 'Position', pos); g.set(n, 'Edge Y', ye); g.set(n, 'Top Z', zt)
    g.set(n, 'Slope', slope); g.set(n, 'Round', rnd)
    return n.outputs['SDF'], n.outputs['Clear']


# --------------------------------------------------------------------------------------------- build
def build(g):
    I, M = g.I, g.math
    prof = _profile_group()
    phi = I('Phi')
    base = I('Base Mesh')
    slope, rnd, blend, clr = I('Thenar Slope'), I('Thenar Round'), I('Thenar Blend'), I('Thenar Clearance')

    # ---- flat top height and old back edge, from the input mesh
    zt = g.sep(g.node('GeometryNodeBoundBox', {'Geometry': base}).outputs['Max'])[2]
    _, by, bz = g.sep(g.pos())
    top_sel = g.to_bool(M('GREATER_THAN', bz, M('SUBTRACT', zt, 1e-4)))
    st0 = g.node('GeometryNodeAttributeStatistic', {'Geometry': base, 'Selection': top_sel, 'Attribute': by},
                 data_type='FLOAT', domain='POINT', label='old back edge')
    y0 = st0.outputs['Min']

    def k_at(z):            # wall blend radius at height z: 0 within 1.5 mm of the top (the flat top and the voxels
        # just under it stay exact), full Thenar Blend from 5 mm below the top (k/6 stays < the depth below the top,
        # so no false surface appears under the flat top)
        return g.maprange(z, M('SUBTRACT', zt, 5.0), M('SUBTRACT', zt, 1.5), blend, 0.0)

    # ---- clearance check points: static cutout vertices + stick hole rings
    parts = [I('Cutouts')]
    for key in ('Thenar Hole R', 'Thenar Hole L'):
        hx, hy, hr = g.sep(I(key))
        circ = g.node('GeometryNodeMeshCircle', {'Vertices': 96, 'Radius': hr}).outputs[0]
        for dz in (0.01, 0.5, 1.0, HOLE_DEPTH):
            tr = g.node('GeometryNodeTransform', {'Geometry': circ, 'Translation': g.xyz(hx, hy, M('SUBTRACT', zt, dz))})
            parts.append(tr.outputs[0])
    jn = g.node('GeometryNodeJoinGeometry', label='check geometry')
    for gsock in reversed(parts):
        g.link(gsock, jn.inputs[0])
    chk = g.node('GeometryNodeMeshToPoints', {'Mesh': jn.outputs[0]}).outputs[0]

    # ---- auto move: bisection on the clearance (repeat zone)
    ri = g.node('GeometryNodeRepeatInput', label='bisection')
    ro = g.node('GeometryNodeRepeatOutput', label='bisection')
    ri.pair_with_output(ro)
    ro.repeat_items.new('FLOAT', 'Lo')
    ro.repeat_items.new('FLOAT', 'Hi')
    g.set(ri, 'Iterations', ITER)
    g.set(ri, 'Geometry', chk)
    g.set(ri, 'Lo', 0.0)
    g.set(ri, 'Hi', MAX_MOVE)
    lo, hi = g.o(ri, 'Lo'), g.o(ri, 'Hi')
    mid = M('MULTIPLY', M('ADD', lo, hi), 0.5)
    cpos = g.pos()
    _, c_clear = _profile(g, prof, cpos, M('ADD', y0, mid), zt, slope, rnd)
    c_eff = M('SUBTRACT', c_clear, M('DIVIDE', k_at(g.sep(cpos)[2]), 6.0))   # the wall blend pulls in by <= k/6
    stc = g.node('GeometryNodeAttributeStatistic', {'Geometry': g.o(ri, 'Geometry'), 'Attribute': c_eff},
                 data_type='FLOAT', domain='POINT', label='min clearance')
    target = M('MULTIPLY_ADD', I('Voxel Size'), 0.5, clr)          # + half a voxel: the meshed edge may sit
    ok = g.to_bool(M('GREATER_THAN', M('SUBTRACT', stc.outputs['Min'], target), -1e-6))   # that far forward
    g.set(ro, 'Geometry', g.o(ri, 'Geometry'))
    g.set(ro, 'Lo', g.sw(ok, lo, mid))
    g.set(ro, 'Hi', g.sw(ok, mid, hi))
    m_auto = g.o(ro, 'Lo')
    move = g.sw(I('Thenar Auto'), I('Thenar Move'), m_auto)
    ye = M('ADD', y0, move)

    # ---- the cut
    pos = g.pos()
    pz = g.sep(pos)[2]
    h, _ = _profile(g, prof, pos, ye, zt, slope, rnd)
    cut = M('SMOOTH_MAX', phi, h, k_at(pz))
    on = g.sw(I('Thenar'), 0.0, 1.0)
    w = M('MULTIPLY', on, g.maprange(pz, I('Thenar Min Z'), M('ADD', I('Thenar Min Z'), 2.0), 0.0, 1.0))
    phi_out = g.lerp(phi, cut, w)
    mask = M('MULTIPLY', w, g.maprange(M('SUBTRACT', cut, phi), 0.0, 0.25, 0.0, 1.0, smooth=False))

    # ---- debug: the new edge as a line + the check points with their clearance
    line = g.node('GeometryNodeMeshLine', {'Count': 151, 'Start Location': g.xyz(-75.0, ye, zt),
                                          'Offset': g.xyz(1.0, 0.0, 0.0)}).outputs[0]
    _, f_clear = _profile(g, prof, g.pos(), ye, zt, slope, rnd)
    sc = g.node('GeometryNodeStoreNamedAttribute', data_type='FLOAT', domain='POINT')
    g.set(sc, 'Geometry', chk); g.set(sc, 'Name', 'thenar_clearance'); g.set(sc, 'Value', f_clear)
    jd = g.node('GeometryNodeJoinGeometry', label='debug')
    g.link(sc.outputs[0], jd.inputs[0])
    g.link(line, jd.inputs[0])
    dbg = jd.outputs[0]
    for name, val in (('thenar_move_mm', move), ('thenar_edge_y', ye)):
        s_ = g.node('GeometryNodeStoreNamedAttribute', data_type='FLOAT', domain='POINT')
        g.set(s_, 'Geometry', dbg); g.set(s_, 'Name', name); g.set(s_, 'Value', val)
        dbg = s_.outputs[0]
    return {'Phi': phi_out, 'Mask': mask, 'Debug': dbg}
