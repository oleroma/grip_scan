# Your smooth hand fit: how to use it

**File:** `/tmp/claude-0/sdf/smooth/final/Grip_smooth.blend` (shape v3, review round 2, 2026-10-20). It is your
GripV with one broad, smooth change of the right handle, mirrored to the left one. v2 is kept as
`final/Grip_smooth_v2.blend`, v1 (palm pocket and finger dish) as `final/Grip_smooth_v1.blend`.

**What v3 is, honestly:** plain GripV with the press of the pinky and the palm eased by up to about 2 mm. It is
**not yet a grip the middle finger grabs**: the middle finger rests on R2 and touches nothing else (see the table
below and the open questions in DESIGN.md).

What the shape does:
- It is a sum of **broad, smooth bumps** (Shape Size 34 mm: each one is about 68 mm across at its base, about
  15 mm wide at half height). The bumps alone could still add up to something narrower; what keeps narrow dents
  and ridges out is a rule in the solve (no dent or bump of 1 mm or more against its 12-16 mm surroundings).
  There are **no finger grooves and no hand imprint**: your skin does the conforming.
- Where the case already presses into the hand, it **leaves it**, as long as the press is no deeper than the soft
  tissue can take ('Palm Give' / 'Finger Give'). Only where the hand would sink in deeper does it recede, broadly.
- It only grows toward a finger that is **within 1 mm of touching** ('Touch Reach'). Farther zones are reported as
  "not chased" (in v2, chasing them made a 0.75 mm ridge under the top band).
- It keeps the middle and ring fingertips 0.3 mm clear of the case beside the R2 / R4 openings over the whole
  trigger press, and keeps the screw-channel wall (see the limits at the end).
- It is **zero** at the body (|x| < 25), in the front band, on the bottom planes and above z 35 by construction;
  after the remesh the case changes by at most 0.012 mm between z 35 and 36.5 and by **0.0 above z 36.5**
  (measured). case_top and the middle plate are unchanged (0.012 / 0.014 mm, remesh-noise level).

The change is small: at most **0.82 mm of growth** and **2.07 mm of recession** (v2: +0.96 / -1.80; v1: +7.7 / -4.9).

## Where things are

- **Your controls:** in the dashboard **'GripV controls'**, frame **HAND FIT**, there is the node
  **'GripV hand fit parameters'**. Select it, press Tab, and edit the values on the node **'hand fit values'**
  (the same pattern as 'GripV R2 parameters').
- **The hand:** collection **'hand fit'**, in the **GripV** scene (the scene the file opens in). 'hand rig' is the
  armature you pose; 'hand points' follows it.
- **The machinery:** 'basemesh GripV' > modifier **'hand fit'**. Turn its eye off to see plain GripV.
- **The solved shape:** the hidden object **'shape offset'** (in 'hand fit'). Do not edit it or unhide it;
  the refit script rewrites it.

## The 7 controls

**"acts at once"**: Blender shows the change right away. **"needs refit"**: the value is only read by the refit
script; changing it in Blender **changes nothing** until refit runs (below).

| control | default | what it really does | when it takes effect |
|---|---|---|---|
| **Fit** | 1.0 | Scales the stored shape. 0 = no fit (GripV through the same remesh), 1 = the solved fit. Values in between (or up to 1.5) only scale the shape; touches and keep-outs are only checked at 1. | acts at once |
| **Shape Size** | 34 mm | The size of the smooth bumps (their radius). Larger = broader and calmer, fewer exact touches. Smaller = more local. Below 25 is raised to 25 by refit. | needs refit |
| **Palm Give** | 3.5 mm | How deep the palm, the metacarpals, the whole pinky and the thumb may press into the case before it gives way. Larger = the case keeps more of its GripV volume there. | needs refit |
| **Finger Give** | 1.5 mm | The same for the other finger segments (middle and ring proximal and middle phalanges). | needs refit |
| **Stroke Clear** | 0.3 mm | The gap kept between the middle / ring fingertips and the case beside the R2 / R4 openings, all the way down the trigger press. | needs refit |
| **Screw Wall** | 3.0 mm | The least wall kept between the skin and the two handle screw channels. Where GripV's own wall is thinner (near the screw exits), the fit keeps GripV's wall. | needs refit |
| **Thenar** | on | The slope behind the sticks for the ball of your thumb. | acts at once |

**View is no longer on the dashboard** (review round 2): its inspection meshes broke prints (View 1 was inside-out,
View 2 a coarse mesh that changed case_top). It is a hidden input of the 'hand fit' modifier, for inspection only,
and **must be 0** when you export. `export_check.py` and `check.py` refuse to run on a file whose View is not 0.

Advanced inputs, hidden on the 'hand fit' modifier at their defaults; leave them unless Claude asks (all need refit):
- **Touch Reach** (1 mm): how close a finger zone must be to touching before the shape grows toward it.
- **Middle PIP Pull** (0): extra pull toward the middle finger's PIP. Its skin lies in the zone the shape may not
  change, so it does nothing useful today.
- **Top Clear** (3 mm): the shape is exactly zero above z 38 - 3 = 35, which keeps case_top and the middle plate
  unchanged.

The v1 controls Smoothness, Press, Max Dig, Seat Pull and Seat Proud are gone, and so are the old seat empties.

## What touches, and what does not (measured, v_27 pose)

Gap = the distance between the hand and the case (<= 0.3 mm counts as touching; negative = the hand presses into
the case by that much, which soft tissue takes). "plain" = GripV with the hand fit switched off.

| contact zone | v3 | v2 | plain | |
|---|---|---|---|---|
| palm on the swell | -3.37 | -3.39 | -4.27 | touches; receded broadly to Palm Give |
| pinky PIP | -2.91 | -3.01 | -4.72 | touches; receded to Palm Give |
| pinky distal | -2.00 | -1.85 | -3.22 | touches |
| pinky root pad | -2.20 | -2.39 | -2.30 | touches |
| ring root pad | -1.44 | -2.48 | -2.27 | touches (less than v2: the palm recession reaches it) |
| middle root pad | **+0.87** | +0.87 | +0.87 | **does not touch**: its skin is in the zone the shape may not change |
| ring PIP | **+3.44** | +2.69 | +2.75 | **does not touch**: 3.2 mm away, beyond Touch Reach |
| middle PIP | **+5.51** | +5.55 | +5.67 | **does not touch**: its skin is under the top band, mostly frozen |

Gap of each finger segment to the case (mm; finger capsule surface to skin):

| segment | v3 | v2 | no fit |
|---|---|---|---|
| middle proximal / middle / distal | 6.78 / 2.74 / 0.52 (on R2) | 6.68 / 2.63 / 0.58 | 6.62 / 2.27 / -0.15 |
| ring proximal / middle | 6.68 / +0.24 | 5.31 / -0.19 | 4.93 / -0.44 |
| middle metacarpal | -1.77 | -1.80 | -1.79 |
| index metacarpal / proximal / middle / distal | -2.63 / -1.54 / 0.07 / -0.62 | the same | the same |

The index sits wholly in the frozen region, so the fit does not change it. How deep the soft tissue is pressed
(the Give rule; none is over its limit): ring metacarpal 3.37, pinky middle 3.43, pinky metacarpal 2.71, middle
metacarpal 1.77, ring middle 0.0 (limits 3.5 / 3.5 / 3.5 / 3.5 / 1.5).

## After you re-pose the hand (or change a "needs refit" control)

**Refit runs in Claude's environment only** (it needs Claude's Python with Blender 5.2 as a module and the solver
libraries; it does not run on your Windows machine). Pose 'hand rig' (or change a control), save, and send the
file. Claude runs:
```
/tmp/claude-0/venv/bin/python /tmp/claude-0/sdf/smooth/final/refit.py -- --blend <your file> --triggers
```
and sends it back with the check report. What happens inside:

1. **Trigger check, before anything is solved.** R2 and R4 must sit on your middle and ring fingertips: refit stops
   (exit code 2, file unchanged) when a press-face centre is more than 0.1 mm or 1 deg from its fingertip, when a
   trigger would travel more than 26 deg away from the way your fingertip presses, when a finger sinks more than
   0.55 mm into a trigger, when a value is missing or not a number, or when a lever is no longer under its switch
   (all 81 test rays from the switch must hit the lever, with the gap within 0.05 mm of today's 0.44-0.49 mm).
2. **The solve** (about 15 s) always starts from scratch, so the result does not depend on the old shape.
3. **Case check, before the file is replaced.** The result is saved beside your file and checked against the
   triggers: no case material between switch and lever, no trigger part inside the case. If that fails, refit
   stops (exit code 3) and **your file is not changed**; the rejected result is kept for Claude to look at. Only
   when it passes does it replace your file (the previous one is kept as `Grip_smooth.blend1`).

**Timing (measured on this machine):** a control change without `--triggers` takes about 100 s (trigger check
18 s, gather and solve 43 s, case check 25 s). With `--triggers` add the trigger fit, about 2 min (104-123 s in
the v2 tests), so about 3-4 min in all (*inferred*, not timed end to end in v3). The optional full check
(`final/check.py`) takes about 3 min more (186 s measured).

**How far you can re-pose** (measured on the solved shape with the relief rule; one bone moved, the rest as v_27):

| re-pose | refit | v3: dents >= 1 mm (largest) | v3: bumps >= 1 mm | v2: dents (largest) |
|---|---|---|---|---|
| as installed | runs | 0 (0.66) | 0 | 0 (0.90) |
| pinky proximal 5 deg into the case | runs | 0 (0.89) | 1 (1.19), within the limit of 2 | 1 (1.17) |
| pinky proximal 5 deg away | runs | 0 (0.60) | 0 | 0 (0.90) |
| middle proximal 5 deg (R2 re-fitted) | runs | 0 (0.68) | 0 | 0 (0.90) |
| ring proximal 5 deg (R4 re-fitted) | runs | 0 (0.69) | 0 | 0 (0.90) |
| ring knuckle curled 5 deg | **stops**: R4 would travel 28.3 deg off the press | 1 (1.46) | 2 | 1 (1.99) |
| middle proximal 5 deg the other way | **stops**: R2 face 1.02 mm off, travel 33.4 deg | 4 (4.77) | 4 | 1 (4.39) |

Notes:
- The two "stops" rows are what the shape would be without the trigger check; refit does not install them.
- With the pinky 5 deg into the case, its middle segment presses 0.66 mm deeper than Palm Give: the solve accepts
  that rather than dig a dent (the order is: hard limits, then no dent, then Give, then touch).
- A re-pose still changes the shape up to about 30 mm away from the moved finger (largest change 1.2-1.7 mm; v2
  1.8-2.7 mm), because the whole handle is one broad solve.

## Export

Use your presets **'Case standard'** and **'Triggers'** as always, **with View at 0** (it is hidden, so it stays
0 unless someone changes it on the modifier). The fit is part of the basemesh, so every part follows it. All parts
come out facing outward (positive volume).

- **Where the files go:** the export folder of the GripV scene is `//../orca_slicer/GripV`, relative to where the
  .blend sits. Put `Grip_smooth.blend` into your repo's `blender/` folder (or change the root folder) first.
- **How long:** 'Case standard' (case_bottom plus 4 case_top variants) took 305 s here, with another export
  running at the same time.

A ready-made check and export (refuses to run when View is not 0):
```
/tmp/claude-0/venv/bin/python /tmp/claude-0/sdf/smooth/final/export_check.py
```
It writes STLs (mm) of case_top, case_bottom, R2, R4, L2, L4 and the cable holder, plus `export_report.json`, to
`final/export/` (128 s measured) and does not change the file. Plain GripV through the same exporter is in
`final/export/plain_reference/`, the 'Case standard' preset output in `final/export/combi_case_standard/`.

**Look in the slicer before printing.** A few edges in the exported case parts are shared by more than two
triangles (Blender's own polygons are clean). Measured on 2026-10-20, with where they come from:
- **case_top: 15**, the same count as plain GripV through the same exporter (15). They sit at z 40, 43.8 and 55,
  on GripV's own internal features; none is on a trigger opening and none is in the area the fit changes.
- **case_bottom: 5** (plain GripV: 0). 4 are on the rims of the trigger openings (1 L2, 2 R2, 1 R4) and 1 at the
  right screw exit (65.9, -62.3, 5.0). They come with the hand-fit remesh: v2 had the same 5.
- **R2 / R4 / L2 / L4: 1 each**, the same as plain GripV. **Cable holder: none.**
- The 'Case standard' files, welded the way a slicer does: case_bottom 5, the case_top variants 12-19.

Orca normally repairs these. Check that it reports no errors.

The exported case_top differs from the design mesh in Blender by up to 9 mm at one place (-10.7, -41.8, 48.2).
Plain GripV through the same exporter differs by the same 9.04 mm at the same place, so that comes from the
export settings (very high subdivision, high voxels), not from the fit.

## Please do not touch

- The node groups 'HF Stage', 'HF shape', 'HF thenar' and 'hand fit values' (except its values), and the
  hidden inputs of the 'hand fit' modifier (View must stay 0 for export).
- The modifier order on 'basemesh GripV': MirrorX, subdivision surface, hand fit.
- The collection 'basemesh GripV': never put another object into it (the generator reads it).
- The R2/R4 trigger parameters by hand. If you do change them, ask Claude to run refit without `--triggers`.
- The body: body preset, design subdivision 'high', or basemesh edits. The shape is stored per vertex of
  the 'high' mesh. A body change needs a new solve and new reference runs (ask Claude).

## Known limits (details in /tmp/claude-0/sdf/smooth/DESIGN.md)

- **The middle finger touches nothing but R2** (table above). Its root pad and PIP lie in or under the zone the
  shape may not change (the top band that protects case_top). This needs your decision: curl the middle finger
  in the pose, allow the shape into part of the top band (case_top and middle-plate checks needed), or accept it.
- **Ring PIP 3.4 mm clear** (v2 2.7). Reaching it needs a finger-wide swell next to R4, which the relief rule and
  Touch Reach keep out. Your decision: allow such a swell, or accept.
- **Width:** the case is 0.6 mm narrower at its widest point, where the palm and ring metacarpal press (v2 0.43).
  A larger Palm Give (about 4.3, *inferred*) would keep the width, and the case would press the palm more.
- **Zebra stripes:** the v2 ridge under the top band is gone. What is left is one broad, shallow dish where the
  palm press is eased, and a small step in curvature where the dish fades out at about z 32.5 on the outer face
  (x 80, y -52.5): radius about 20 mm there over a few mm, where plain GripV has about 75 mm. In grazing light it
  looks close to plain.
- **A broad recession under the middle finger's middle segment** (gap 2.74 mm; no fit 2.27): it comes from keeping
  the fingertip clear over the R2 press, done broadly rather than as a local scallop.
- **A small bump on the inner back of the handle** remains (0.81 mm against no fit at (37.1, -58.9, 12.6); v2 0.96).
- **Relief margin:** the closest any feature comes to the 1 mm relief limit is a 0.86 mm bump on the inner side at
  (38.5, -35.5, 12.0) and 0.87 mm at the edge of the change underneath (64.2, -49.2, -14.9); the largest dent is
  0.64 mm (v2: dent 0.88, ridge 0.72, edge 0.97).
- **Moving a trigger moves the middle plate.** When refit with `--triggers` moves R2 / R4, their openings move, and
  case_middle_plate follows them (up to 0.97 mm in the v2 ring-curl test, all within 1 mm of the moved cutters).
  That is your generator following the triggers; the hand-fit shape is zero there.
- **Screw wall:** at least 3 mm wherever GripV has 3 mm or more (measured 3.0). Lower down, where the screw channel
  runs out through the skin, GripV's own wall is 0-2.9 mm (below z 12); there the fit leaves it as GripV has it
  (per 2 mm band, equal to no fit within 0.01 mm, measured).
- **GripV's own sharp edges** inside the changed area are kept: in a 2 mm neighbourhood the tightest crease is
  0.79 of its no-fit radius (concave) and 0.85 (convex); 0.02 % of crease points got more than 20 % tighter.
  Vertex by vertex (no neighbourhood) it is 0.22 % concave (worst 0.38, at (27.2, -35.8, 30.6) where the shape
  fades in next to the body) and 0.24 % convex (worst 0.77); v2 had 0.0 % / 0.36 %. Over 6 mm that worst spot
  is 13 % tighter than no fit (radius 8.7 vs 9.9 mm; v2 9.4).
