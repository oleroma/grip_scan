"""Shape module (final, SPEC v2 since 2026-10-20): ONE smooth, low-frequency signed offset of the case, solved
offline by final/refit.py (solver final/solver_v2/lowfreq.py, numpy + scipy).

The solver writes h = W * (B c) on the stage input mesh (the subdivided GripV basemesh, level 'high'):
    B  smooth radial bumps (Wendland C4) with support radius 'Shape Size' (34 mm = 68 mm across), centres every
       ~10 mm over the right handle (the basis alone does not forbid narrow features: the solver's relief rows
       do, and check.py measures them);
    W  a smooth window, 0 on the frozen region (|x| < 25, z > 38 - Top Clear, y > -4.658) and fading in over 10 mm;
    c  ~140 weights: the calmest shape (least bending) that touches every clear contact zone (gap <= 0.3 mm),
       lets palm / metacarpals / pinky sink up to 'Palm Give' and the other phalanges up to 'Finger Give' (only the
       excess recedes), keeps 'Stroke Clear' between the middle / ring fingertips and the case beside the R2 / R4
       openings over the whole trigger stroke, keeps 'Screw Wall' to the screw channels, and has no dent or bump
       of 1 mm against its 12-16 mm surroundings. A zone pulls only when its nearest skin may move and it is within
       'Touch Reach' (1 mm) of touching (v3 settings, review round 2).
The result is stored as the point attribute 'offset' (vector = h * direction, mm) on the mesh object 'shape offset'
(collection 'hand fit'). This module only APPLIES it:
    input mesh -> + Fit * offset (sampled from 'shape offset') -> SDF of the moved mesh
The left handle is the mirror image (the field is sampled at the mirrored position).

After re-posing the hand or changing Shape Size / Palm Give / Finger Give / Stroke Clear / Screw Wall: run
final/refit.py. Until then this module keeps the last solved shape. 'Fit' acts at once (0 = no fit, 1 = the fit).
(v1, 2026-10-19: one free height per vertex, final/solver/v1/; its file is final/Grip_smooth_v1.blend.)
"""
NAME = 'shape'
PIP_DEFAULT = 1.0   # the middle PIP is pulled like every other zone
SEAT_PULL = 0.05   # soft seat weight of the final solve (final/fit_opts.json)
GROUP = 'HF shape'
LABEL = 'Shape'
DESCRIPTION = ("Smooth hand-fit shape: one solved signed offset of the case (object 'shape offset', "
               "made by final/refit.py, SPEC v2 low-frequency solve). Off = the case passes through unchanged")
USES = ['Phi', 'Base Mesh', 'Mirrored Position', 'Voxel Size', 'Fill Voxel Size', 'Edit Min X']
PARAMS = [
    dict(name='Shape', type='BOOLEAN', default=True,
         desc="Smooth hand-fit shape on/off. Off: the case is not changed by the shape (no fit)"),
    dict(name='Shape Offset', type='OBJECT',
         desc="Mesh holding the solved offset (point attribute 'offset', mm). Re-solve with final/refit.py after re-posing the hand"),
    dict(name='Fit', type='FLOAT', default=1.0, min=0.0, max=1.5, subtype='FACTOR',
         desc='How much of the solved hand-fit shape is applied, at once: 0 = plain case (no fit), 1 = the solved fit. Other values are a preview only (they do not keep the contacts)'),
    # ---- SPEC v2 solver inputs: read by final/refit.py (run it after changing them or the hand pose);
    #      the nodes ignore them (the shape itself is the stored 'shape offset')
    dict(name='Shape Size', type='FLOAT', default=34.0, min=25.0, max=60.0, subtype='DISTANCE',
         desc='Re-solve input (run final/refit.py): size of the smallest shape the fit can make (support radius of its smooth basis, mm). 34 = nothing narrower than about two fingers; larger = broader and calmer, fewer exact touches. (Replaces v1 Smoothness, which meant something else)'),
    dict(name='Palm Give', type='FLOAT', default=3.5, min=0.0, max=6.0, subtype='DISTANCE',
         desc='Re-solve input (run final/refit.py): how far palm, metacarpals and the pinky may sink into the case before it gives way (soft tissue, mm)'),
    dict(name='Finger Give', type='FLOAT', default=1.5, min=0.0, max=4.0, subtype='DISTANCE',
         desc='Re-solve input (run final/refit.py): how far the other finger segments may sink into the case before it gives way (mm)'),
    dict(name='Stroke Clear', type='FLOAT', default=0.3, min=0.0, max=2.0, subtype='DISTANCE',
         desc='Re-solve input (run final/refit.py): clearance of the middle / ring fingertips to the case beside the R2 / R4 openings over the whole trigger stroke (mm)'),
    dict(name='Screw Wall', type='FLOAT', default=3.0, min=2.0, max=5.0, subtype='DISTANCE',
         desc="Re-solve input (run final/refit.py): wall kept between the skin and the 'screw bools' (mm)"),
    dict(name='Middle PIP Pull', type='FLOAT', default=0.0, min=0.0, max=1.0, subtype='FACTOR',
         desc='Advanced re-solve input (run final/refit.py): also try to reach the middle-finger PIP (0 = no; a calm shape cannot reach it)'),
    dict(name='Touch Reach', type='FLOAT', default=1.0, min=0.0, max=6.0, subtype='DISTANCE',
         desc='Advanced re-solve input (run final/refit.py): a contact zone pulls the case towards the hand only when it is within this distance of touching (mm). Zones farther away (ring PIP ~3 mm) would need a finger-wide bump; they are reported, not chased'),
    dict(name='Top Clear', type='FLOAT', default=3.0, min=2.7, max=8.0, subtype='DISTANCE',
         desc='Advanced re-solve input (run final/refit.py): the shape is exactly zero this far below z 38 (keeps case_top and the middle plate unchanged; >= one shell thickness 2.7)'),
]
OBJECT_DEFAULTS = {'Shape Offset': 'shape offset'}
OUTPUTS = ('Mask', 'Debug')
WIDE_BAND = 16.0   # mm, as the stage's Base Wide Grid
FROZEN_Z = 38.0    # frozen above this (parting band, case_top, thenar)
FROZEN_Y = -4.658  # frozen in front of this (max Y -4.158 minus the 0.5 mm front band)
FROZEN_BLEND = 1.0 # mm


def build(g):
    I, M, V = g.I, g.math, g.vmath
    vox, vf = I('Voxel Size'), I('Fill Voxel Size')
    pm = I('Mirrored Position')
    phi_in = I('Phi')
    on = I('Shape')
    base = I('Base Mesh')
    # offset per input vertex: nearest point of the stored mesh (same vertices -> exact; a slightly changed
    # basemesh gets the interpolated offset), right handle only (the left one reads the mirrored field)
    src = g.objinfo(I('Shape Offset')).outputs['Geometry']
    p = g.pos()
    px, py, pz = g.sep(p)
    smp = g.node('GeometryNodeSampleNearestSurface', {'Mesh': src, 'Value': g.nattr('offset', 'FLOAT_VECTOR'),
                                                       'Sample Position': g.xyz(M('ABSOLUTE', px), py, pz)},
                 data_type='FLOAT_VECTOR')
    off = smp.outputs['Value']
    w = M('MULTIPLY', M('GREATER_THAN', px, 0.0), I('Fit'))
    off = g.node('ShaderNodeVectorMath', {0: off, 'Scale': w}, operation='SCALE').outputs[0]
    moved = g.node('GeometryNodeSetPosition', {'Geometry': base, 'Offset': off}).outputs[0]
    # SDF of the moved mesh exactly like the stage's base SDF (narrow fine grid + wide coarse grid)
    D = g.mesh_sdf(moved, vox, 3, 'shape SDF')
    wide = M('ADD', M('CEIL', M('DIVIDE', WIDE_BAND, vf)), 2.0)
    Dw = g.mesh_sdf(moved, vf, wide, 'shape wide SDF')
    d = g.sample(D, pm)
    dw = g.sample(Dw, pm)
    near = M('LESS_THAN', M('ABSOLUTE', d), M('MULTIPLY', vox, 2.5))
    phi = g.sw(g.to_bool(near), dw, d)
    # frozen region by construction: there the incoming field passes through exactly, faded in over FROZEN_BLEND
    # mm outside it (the solver already keeps h = 0 there; this removes the leak of long input-mesh edges)
    mx, my, mz = g.sep(pm)

    def ramp(v):
        return M('MINIMUM', M('MAXIMUM', M('DIVIDE', v, FROZEN_BLEND), 0.0), 1.0)
    wz = ramp(M('SUBTRACT', mz, FROZEN_Z - FROZEN_BLEND))
    wy = ramp(M('SUBTRACT', my, FROZEN_Y - FROZEN_BLEND))
    wx = ramp(M('SUBTRACT', M('ADD', I('Edit Min X'), FROZEN_BLEND), M('ABSOLUTE', mx)))
    wf = M('MAXIMUM', M('MAXIMUM', wz, wy), wx)
    phi = M('ADD', M('MULTIPLY', phi, M('SUBTRACT', 1.0, wf)), M('MULTIPLY', phi_in, wf))
    # switch on a single value: Shape off = the incoming field passes through, nothing is evaluated
    phi = g.sw(on, phi_in, phi)
    # mask on the result: how far the shape moved the skin (via the moved mesh, nearest surface)
    mk = g.node('GeometryNodeSampleNearestSurface', {'Mesh': src, 'Value': V('LENGTH', g.nattr('offset', 'FLOAT_VECTOR'), out=1),
                                                      'Sample Position': g.xyz(M('ABSOLUTE', px), py, pz)}, data_type='FLOAT')
    mask = g.sw(on, 0.0, M('MINIMUM', M('DIVIDE', mk.outputs['Value'], 2.0), 1.0))
    return {'Phi': phi, 'Mask': mask, 'Debug': moved}
