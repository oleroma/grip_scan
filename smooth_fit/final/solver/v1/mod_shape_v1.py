"""Shape module (final): ONE smooth signed offset of the case, solved offline on the dense stage input mesh.

Method: the 'dense' prototype (p_dense) with the soft trigger seats of the 'cage' prototype grafted in.
  The solver (final/refit.py -> p_dense/v2/dense_fit.py, numpy + scipy) computes one height h per vertex of the
  stage input mesh (the subdivided GripV basemesh, level 'high'): the new skin is  p + h * normal.
  h is the smoothest field (least bending, thin-plate energy) that
    * makes each contact zone of the hand touch the case with the rigid finger sunk 'Press' mm,
    * keeps every hand capsule at most 'Max Dig' mm deep anywhere,
    * keeps every concave curve of the edited skin no tighter than 'Smoothness' mm (and no worse than GripV),
    * pulls the skin around each R2 / R4 opening towards its seat plane ('Seat Proud' mm under the press face)
      with the soft weight 'Seat Pull' (a soft target, never a carve: smoothness wins where they conflict),
    * is exactly 0 on the frozen region: |x| < 25 (body, mirror seam), z > 38 (parting band, case_top, thenar),
      the front band y > -4.658 (max Y = R1/L1 position).
  The result is stored as the point attribute 'offset' (vector = h * normal, mm) on the mesh object
  'shape offset' (collection 'hand fit'). This module only APPLIES it:
      input mesh -> + Fit * offset (sampled from 'shape offset') -> SDF of the moved mesh
  The left handle is the mirror image (the field is sampled at the mirrored position).

After re-posing the hand or changing Smoothness / Press / Max Dig / Seat Pull / Seat Proud: run final/refit.py.
Until then this module keeps the last solved shape. 'Fit' acts at once (0 = no fit, 1 = the solved fit).
"""
NAME = 'shape'
PIP_DEFAULT = 1.0   # the middle PIP is pulled like every other zone
SEAT_PULL = 0.05   # soft seat weight of the final solve (final/fit_opts.json)
GROUP = 'HF shape'
LABEL = 'Shape'
DESCRIPTION = ("Smooth hand-fit shape: one solved signed offset of the case (object 'shape offset', "
               "made by final/refit.py). Off = the case passes through unchanged")
USES = ['Phi', 'Base Mesh', 'Mirrored Position', 'Voxel Size', 'Fill Voxel Size', 'Edit Min X']
PARAMS = [
    dict(name='Shape', type='BOOLEAN', default=True,
         desc="Smooth hand-fit shape on/off. Off: the case is not changed by the shape (no fit)"),
    dict(name='Shape Offset', type='OBJECT',
         desc="Mesh holding the solved offset (point attribute 'offset', mm). Re-solve with final/refit.py after re-posing the hand"),
    dict(name='Fit', type='FLOAT', default=1.0, min=0.0, max=1.5, subtype='FACTOR',
         desc='How much of the solved hand-fit shape is applied, at once: 0 = plain case (no fit), 1 = the solved fit. Other values are a preview only (they do not keep the contacts)'),
    # ---- solver inputs: read by p_dense/refit.py (run it after changing them or the hand pose); the nodes ignore them
    dict(name='Smoothness', type='FLOAT', default=18.0, min=8.0, max=40.0, subtype='DISTANCE',
         desc='Re-solve input (run refit.py): the tightest hollow the shape may make, as a radius in mm. Larger = smoother and fewer finger contacts; below 12 it starts to read as a finger groove'),
    dict(name='Max Dig', type='FLOAT', default=0.8, min=0.2, max=3.0, subtype='DISTANCE',
         desc='Re-solve input (run refit.py): no part of the hand sinks deeper than this into the case anywhere the shape may change (mm)'),
    dict(name='Seat Pull', type='FLOAT', default=SEAT_PULL, min=0.0, max=2.0,
         desc='Re-solve input (run refit.py): how hard the case around the R2 / R4 openings is pulled flush with the trigger press faces. 0 = not at all; larger = flatter rims but more bumps between R2 and R4'),
    dict(name='Seat Proud', type='FLOAT', default=0.2, min=0.0, max=2.0, subtype='DISTANCE',
         desc='Re-solve input (run refit.py): how far each trigger press face stands out of the case around it (mm)'),
    dict(name='Contact Pull', type='FLOAT', default=1.0, min=0.1, max=10.0,
         desc='Advanced re-solve input (run refit.py): how hard the finger / palm contacts pull against smoothness'),
    dict(name='Middle PIP Pull', type='FLOAT', default=PIP_DEFAULT, min=0.0, max=50.0,
         desc='Advanced re-solve input (run refit.py): extra pull of the middle-finger PIP grab at the handle front (0 = none)'),
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
