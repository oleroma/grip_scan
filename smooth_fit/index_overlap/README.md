# Index finger overlap on R1 (2026-10-20)

Sheet: `index_overlap_sheet.jpg`. Base: smooth fit v3 (`../Grip_smooth.blend`, unchanged).

How the variants were made: the whole 'hand rig' slid along R1 (x) by -6 mm (in) and +6 mm (out), then
`refit.py --triggers` (R2/R4 re-fitted to the fingertips, shape re-solved with the same v3 controls).

| variant | file | R1 under | fingertip past R1's inner edge | checks |
|---|---|---|---|---|
| hand 6 mm in | Grip_smooth_hand_in6.blend (project files only) | middle segment | 20 mm | R2 travel 26.8 deg (limit 26), centre 0.11 mm (limit 0.1): refit refused it, file is the rejected result, for viewing only |
| today (v3) | ../Grip_smooth.blend | middle segment + last knuckle | 14 mm | pass |
| hand 6 mm out | Grip_smooth_hand_out6.blend (project files only) | last knuckle mid-R1 | 8 mm | pass |

R1 front face spans x 36.5-55.5 mm. Shape change vs v3: in6 up to +2.2 / -2.6 mm, out6 +2.5 / -2.3 mm (nearest-point,
includes the moved R2/R4 openings); contact gaps out6: palm -3.38, middle PIP +4.35 (v3 +5.39), ring PIP +3.05 (v3 +3.39).

Code in `code/`: mkvar.py (shift hand), dump.py (evaluated meshes), render.py (Cycles renders, diff colouring),
nofit.py (rebuilds `/tmp/claude-0/sdf/smooth/baseline/nofit/curvature.npz`, which the solver needs and which is not in
smooth_fit_code.tar.gz; with it a re-solve of v3 reproduces -2.07..0.82 mm).

The two variant .blend files (25 MB each, v3 with the hand shifted) are kept in the project files
(sdf_grip/index_overlap/), not in git; rebuild them with code/mkvar.py + refit.py.
