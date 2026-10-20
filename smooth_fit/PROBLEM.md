# Smooth hand-fit shape for GripV: problem statement for the prototypes

Everything here lives in `/tmp/claude-0/sdf/smooth/`. Units are mm, Blender scene coordinates (x to the right
handle, y forward = toward R1/L1, z up). Python: `PY=/tmp/claude-0/venv/bin/python` (Blender 5.2.2 as `bpy`, headless).
Numbers are **measured** unless marked *inferred*.

## 0. The direction (Olex, 2026-10-19 23:15, verbatim)

> "you trying to carve out sdfs with hand shape, that will not work well, we want build smooth shape that will fit our hand well"

So **no imprints or grooves** of the rigid capsule hand. An imprint fits only one exact pose, and the hand's
soft tissue does the conforming. Build **one smooth shape** over GripV as a signed offset: grow where the hand
is beyond the case, recede where the hand digs in. The case should touch the hand where Olex says it should,
and be smooth everywhere, with **no feature narrower than a finger**.

Olex's grip:
* The middle phalanges grab the handle front near their PIP joints and press it into the pads at the finger
  roots (palm over the knuckles + base of the proximal phalanges).
* The palm is cupped, and the case fills the palm cup (palm swell).
* The middle and ring fingertips press the R2 / R4 trigger pads.
* The pinky only wraps.
* The index sits on R1 (top region, frozen).
* The thenar rests on the slope behind the sticks (thenar module, done).

## 1. What a prototype delivers (the variables)

**One module file `mod_shape.py`** in your own folder (plus any helper files it imports), following the
module contract of `/tmp/claude-0/sdf/build/INTERFACE.md` section 3. It returns `Phi`, the signed distance
field (< 0 inside) of the new skin. The stage chain in `smooth/stage/build_stage.py` is:

```
input = subdivided GripV basemesh ('high')   ->  Base SDF (mirrored, exactly symmetric)
  -> HF shape  (YOUR module; pass-through in base.blend)
  -> HF fill   (old height-field fill: OFF, Fill = False)
  -> HF carve  (capsule imprint: OFF, Carve = False; never switch it on)
  -> HF seats  (old seats with hand re-carve: OFF, Seats = False; never switch it on)
  -> HF thenar (ON, accepted: back edge of the top face 6.58 mm forward, slope behind the sticks)
  -> coarse pass (Fill Voxel Size 1.0) + fine pass (Voxel Size 0.5) -> Grid to Mesh -> seam snap at x = 0
```

* The edit is a **signed offset of the base shape**: grow and recede, both smooth. Its R2/R4 part must also
  make the skin flush with the trigger seat planes (section 3.2), because the old seats module is off.
* **Parameters:** few and meaningful for Olex (he is not an expert and must be able to adjust them), with names
  unique across the whole modifier. `--set 'Name=value'` sets them in a run.
* **What a module may read:** the context sockets (`Phi`, `Base SDF`, `Mirrored Position`, `Base Grid`,
  `Base Wide Grid`, `Base Mesh`, `Hand Points`, `Cutouts`, `Voxel Size`, `Fill Voxel Size`, `Press`,
  `Edit Min X`, `Mirror To Left`). It may also use Object Info of objects that do **not** read the basemesh:
  'hand points', 'hand rig', 'R2', 'R4', 'R2 bool', 'R4 bool' (mod_seats.py does this), the seat empties and
  the static cutouts. **Never** read 'Stick R cutout', 'StickL cutout', the hole objects, 'home cutout' or any
  case part (dependency cycle).
* **Live is preferred:** re-posing 'hand rig' should re-fit the shape. An offline solve that writes data into
  your private copy (for example a point cloud read by Object Info) is allowed if you document it and hand over
  the script that regenerates it.
* Hand-related positions use `Mirrored Position` (the left handle copies the right one). Keep the field
  exactly symmetric for |x| < 1 mm (the seam snap welds mirrored vertex pairs).
* Machinery you can reuse from `stage/mod_fill.py`: Repeat-zone propagation along mesh edges, slope limit,
  pinned smoothing, ray limits, a height field along the case normal.

## 2. Files

| path | what |
|---|---|
| `base.blend` | Grip_hand.blend + solved R2/R4 trigger values (`trigger_solver.py --apply build/w_triggers/trigger_values.json`) + stage rebuilt from `smooth/stage/` (shape pass-through, fill/carve/seats off, thenar on). Active scene 'hf idle', so it opens in 0.55 s; switch to 'GripX' to look at it in the UI. **Never write it**: `eval_proto.py` makes a private copy. |
| `stage/` | copy of build/stage with: slot `shape` first in ORDER, `mod_shape.py` pass-through, `Fill`/`Carve`/`Seats` default False (a single-value switch, so an off module costs nothing), `Carve Max Z` 40, `eval_all.py` with the fixed press face, cutter export, triggers also in `--skin-only` runs, the 3 x 3 sheet, no-fit reference |
| `lib/gripmetrics.py` | measurement library + **smooth/ extension** (curvature, kink, smoothness stats, zone support, max dig, seats, frozen mask, change vs no-fit, acceptance) |
| `lib/measure_case.py` | report builder, new option `--nofit-run` |
| `lib/validate_curvature.py`, `lib/make_analytic.py`, `lib/validation/` | curvature validation on analytic shapes (section 4.3) |
| `targets.npz`, `targets.json` | shared problem data (section 7), made by `make_targets.py` |
| `baseline/nofit/` | the no-fit baseline run (report.json, sheet.jpg, curvature.npz, stage.npz, parts/...) |
| `baseline/base_base_high_low/` | plain GripV (stage Enable off), cached reference run |
| `eval_proto.py` | one-command evaluation of a prototype (section 6) |
| `scratch/test_bump/` | **setup check only, not a prototype**: a 3 mm Gaussian swell at the ring PIP (shows the chain works) |

## 3. Targets

### 3.1 Contact zones (gripmetrics zones; the capsules are rigid, so "penetration" means how far the rigid capsule sinks in)

| zone | definition | target | no-fit baseline: penetration (min gap) / support |
|---|---|---|---|
| middle root pad | 20 % of the metacarpal next to the knuckle + 35 % of the proximal phalanx | pen 0.5 | 0.00 (+0.86) / 8.7 % |
| middle PIP | 40 % of the middle phalanx next to the PIP joint | pen 0.5 | 0.00 (+5.60) / 0 % |
| ring root pad | as middle | pen 0.5 | 2.30 / 52.1 % |
| ring PIP | as middle | pen 0.5 | 0.00 (+2.72) / 0 % |
| pinky root pad | as middle | pen 0.5 | 2.32 / 69.5 % |
| pinky PIP | as middle | pen 0.5 | 4.75 / 100 % |
| pinky distal | whole pinky distal phalanx (wraps) | pen 0.5 | 3.21 / 100 % |
| palm on swell | middle, ring, pinky metacarpals | pen 0.5 | 4.26 / 52.4 % |
| R2 / R4 fingertip | middle / ring distal phalanx | on the trigger press face (3.2) | vs skin: 0.16 / 0.00 (+0.56) |
| index proximal / middle / distal | on R1, top region | frozen, information only | 1.54 / 0.00 / 0.62 |

* **Penetration** = the deepest point of the zone: max(0, -min gap), gap = signed distance of the capsule axis
  point minus r. Target 0.5 (= Press), **accept 0.2 to 1.0**, for each of the 8 contact zones.
* **Support** = the share of the zone's axis length whose gap is at most 1.5 mm, i.e. how much of the finger
  the case carries. Soft target: at least 50 % for the root pads, the PIPs and the palm.
* **No dig:** every hand capsule (all 20, palm heel and thumb included) sinks at most **1.0 mm** anywhere in
  the editable region.
  - No-fit baseline: 5.19 mm (pinky middle phalanx at (61.0, -35.3, -7.9)).
  - In the frozen region it is 2.63 mm (index metacarpal at (79.5, -58.8, 46.4), on the top). That one is
    information only, because the shape cannot change it.
* The PIP zones lie in front of the case. The middle phalanx reaches y up to +5, but the case may never pass
  y = -4.158, so the PIP contact has to come from below or behind, not by growing forward.

### 3.2 R2 / R4 trigger seats (triggers solved and fixed; do not change their parameters)

* **Press face** = the outer cap of the finger end (`trigger_solver.press_face()`). The old gripmetrics used the
  generator's 'top' faces, 1 mm lower (fixed in this copy).
  - R2: centre (39.147, -25.134, 19.753), normal (-0.70517, 0.67487, -0.21746).
  - R4: centre (41.459, -39.455, 4.487), normal (-0.74075, 0.55549, -0.37778).
  - Both are within 0.02 mm / 0.01 deg of the fingertip targets.
  - Fingertip pad to press-face centre: -0.50 / -0.50 (the pad sinks Press into the face).
  - Hand overlap with the trigger parts: 0.50 / 0.50.
* **Seat plane** = centre - 0.2 * normal. The skin around the opening lies in it, so the press face stands
  0.2 mm proud.
* **Opening** = the section of 'R2 bool' / 'R4 bool' (right half) with the seat plane. It reaches 9.5 / 9.7 mm
  from the press-face centre (`seat_<P>_opening_segments` in targets.npz).
* **Rim** = skin vertices within 8 mm of the plane whose projection lies outside the cutter and within
  **2.5 mm** of the opening outline.
* **Targets:**
  - skin at the press-face centre (signed distance of the centre to the skin): 0.2, accept 0.1 to 0.3;
  - rim max |deviation from the seat plane|: at most 0.3 mm;
  - keep at least 1.2 mm of wall to the cutter outside the opening (the lever channel).
* **No-fit baseline:**
  - skin at the press-face centre: R2 0.648, R4 1.520;
  - rim deviation: R2 -5.07 .. +1.64 mm, R4 -8.00 .. +1.41 mm.

  The skin at the R4 rim is up to 8 mm *below* the plane, so the seat has to raise the skin a lot there and
  still blend smoothly. That is a real conflict with smoothness: blending an 8 mm step with concave radius
  >= 12 mm needs room (the old seats module capped the move at 5 mm and left -7.96 mm at R4).

### 3.3 Smoothness (the new requirement)

**Judged region** = the edited region dilated by 3 mm (right half), minus unchanged design features:
* **Edited** = a skin vertex more than 0.2 mm from the no-fit skin.
* **Unchanged design features** = vertices that are not edited and lie within 2 mm of a vertex that already
  was a feature on the no-fit skin (concave R < 12, convex R < 3, concave turn > 10 or convex turn > 25 deg/mm).

  Such features include the body-handle crease along z ~29-31 (x 25-32) and the outline edges.

| metric (judged region) | limit | meaning |
|---|---|---|
| % area with concave radius < 8 mm | <= 0.5 % | a groove narrower than a finger (finger ~16-18 mm across) |
| % area with concave radius < 12 mm | <= 5 % | reads as a groove |
| concave radius p0.5 (99.5 % of the area is flatter) | >= 8 mm | robust minimum |
| convex radius p0.5 | >= 3 mm | no sharp convex ridge |
| max concave normal turn | <= 10 deg/mm | concave kink / crease (an R 8 groove reads 7.2, noise <= 4.5) |
| max convex normal turn | <= 25 deg/mm | convex kink (an R 3 rounding reads 19; the handle's own roundings 15-20) |

Also reported, not judged: the min concave and convex radius, the unsigned normal turn, the same statistics
on the raw dilated region and on the edited region alone, and the **no-fit baseline in the same footprint**.

**References (right handle, x >= 25, z < 40)** are dominated by the generator's own design features at the
body-handle junction, which is why the acceptance is on the judged region and not on the whole handle:

| | plain GripV (subdivided, not remeshed) | no-fit baseline (remeshed + thenar) |
|---|---|---|
| min concave radius / p0.5 | 0.88 / 1.62 mm at (27.8, -35.8, 30.6) | 1.44 / 1.90 mm at (27.8, -36.2, 30.3) |
| % area concave R < 12 / < 8 | 7.67 / 4.41 % | 8.76 / 5.22 % |
| min convex radius / p0.5 | 0.60 / 1.52 mm | 0.58 / 1.65 mm |
| max concave / convex normal turn | 62.3 / 75.9 deg/mm | 42.4 / 71.9 deg/mm |
| kink noise floor on smooth skin (>= 2.5 mm from any feature) | | p99 3.3, max 4.5 deg/mm |

### 3.4 Frozen region and invariants (hard)

* **Frozen** (`gripmetrics.frozen_mask(V)`; per no-fit vertex in `targets.npz: nofit_frozen`):
  - |x| < 25 (Edit Min X: body, mirror seam);
  - z > 38 (parting band 38-40 + case_top; the thenar module, which runs after the shape, owns z > 40);
  - y > -4.658 (a 0.5 mm front band holding max Y, verts at |x| < 28, z 33-48).

  Max change there vs the no-fit skin: **<= 0.05 mm**.
* **Top face** exactly flat at max Z 59.04002 with 0 verts above. Its vertex count (27,566) and back edge
  (Y -62.25) must equal the no-fit's (a 0.05 mm ripple collapses case_top).
* **max Y = -4.158** (it positions R1/L1); R1/L1 shift 0.
* **bbox min y / z** unchanged (<= 0.01 mm): -90.62 / -15.323.
* **bbox x extent** unchanged (|x| max 82.896): **OPEN QUESTION, see 3.5.**
* Z = 40 section vs no-fit: max deviation <= 0.05 mm.
* Skin closed manifold, 1 island, genus 0.
* Battery face: 442 verts at x = 0 exactly, flat face at z 26.56.
* case_top vs no-fit case_top: surface max deviation (two-sided) <= 0.05 mm and volume |d| <= 10 mm3 (no-fit:
  34,171.5 triangle / 34,172.2 bmesh). Noise floor, measured with the test swell (skin above z 38 bit-identical,
  only the vertex order changed): the generator re-triangulates case_top's parting band z 40-45, giving
  -3.2 mm3 and 0.018 mm.
* Stick cutout clearance >= 0.8 mm (no-fit: 2.25, dynamic 'StickL hole').
* Handle wall p1 >= 2 mm (no-fit: 11.7).
* Wall to each internal object >= min(its no-fit wall, 1.5 mm).
* No dependency cycle; symmetric (edits mirrored); subdivision 'high'; body preset 'Case standard'; hand pose v_27.

### 3.5 Open question: bbox x extent vs the palm contact (needs Olex's decision)

The outermost point of the case, (82.90, -70.75, 25.75), lies **4.0 mm inside the ring metacarpal capsule**,
and the palm's deepest dig (4.26 mm) is right there. Any palm fit recedes the skin at the x extreme and changes
bbox min/max x. For example, the old carve moved it 1.20 mm and the old fill+carve+seats stack moved it
0.58 mm.

The invariant list says "bbox min (screws)". I measured that 'screw bools' (the screw cutter object) does
**not** move at all (0.0000 mm) when the skin's bbox min x moves 1.2 mm (`scratch/screw_test.py`), so for x
the dependency is not confirmed. Other generator parts could still use it; I did not trace the screw node tree.

Until decided, the acceptance keeps the x extent as a hard check of its own kind (`bbox_x`) and reports both:
* `accepted` (all hard checks), and
* `accepted_except_bbox_x`.

## 4. How the metrics are computed (lib/gripmetrics.py, smooth/ section)

### 4.1 Curvature
Principal curvatures per vertex from a **robust weighted quadric fit over a 3 mm ball**:
* Gaussian weights with sigma 1.5.
* Neighbours whose normal faces away (< ~70 deg) are dropped, so the far side of a thin wall never counts.
* Local frame from the mean normal; h = au^2 + buv + cv^2 + du + ev + f.
* 3 Huber re-weightings (MAD scale), then refit on the fitted normal.
* Shape operator of the graph.

Sign: + convex, - concave (w.r.t. the outward normal; faces oriented by signed volume).
`curvature()` returns k1 >= k2 and dir2, the direction across a groove. About 0.1 ms per vertex:
65k vertices take ~7 s.

### 4.2 Kink detector
* Normals are averaged over 1 mm (area-weighted, Gaussian).
* Per vertex: the largest normal turn per mm towards any vertex 1-2 mm away on the same sheet.
* Split into **concave** (normals converging along the pair) and **convex** (diverging).

A smooth surface of radius R reads 57.3/R deg/mm (R 8: 7.2; R 12: 4.8). A 90 deg crease reads 48-65.

### 4.3 Validation (analytic shapes meshed exactly like the stage: analytic SDF -> Field to Grid 0.5 mm, off-lattice -> Grid to Mesh)
`lib/validate_curvature.py` gives these results (see `lib/validation/curvature_validation.json`):

| shape | quantity | reference | measured | radius error median / p95 / max |
|---|---|---|---|---|
| sphere r 20 | k1, k2 | 1/20 | 0.05027, 0.05023 | 0.54 / 0.63 / 0.98 %, 0.47 / 0.53 / 0.69 % |
| cylinder r 10 | k1 / k2 | 1/10 / 0 | 0.10152 / -0.00003 | 1.51 / 1.62 / 1.68 %; flat direction read as R > 11 m |
| torus groove (cylinder r 40 minus torus R 40, r 8) | k2 across the groove | -1/8 | -0.1277 | 2.14 / 2.39 / 2.64 %, sign never wrong |
| same | k1 along (cos t / (40 - 8 cos t)) | | | 0.22 / 0.78 / 1.63 % |
| same | min concave radius at the groove bottom | 8 | 7.79 | |
| kink | sphere / cylinder / groove bottom | 2.86 / 5.73 / 7.16 | 2.94 / 5.80 / 7.27 (median) | |
| kink | 90 deg convex crease (cylinder cap edge, groove rims) | | 65.5 / 47.8-62.0 | read as convex; min convex radius 0.97 / 1.19 mm |

The quadric over 3 mm reads curvature 0.5-2.5 % too high on tight radii (truncation bias). The limits
above have that margin.

### 4.4 Contacts
`zone_support()` samples the zone axis every 0.5 mm, with the sign by ray parity on the skin (never normals).
`max_dig()` runs over all capsules and splits the result into the editable and frozen regions by the location
of the nearest skin point. The fingertip pad gap uses the fixed press face.

## 5. Acceptance (report.json > `acceptance`, one line in `summary3`)

A prototype is **accepted** when every hard check passes:
* 8 contact-zone penetrations;
* max dig (editable region);
* 6 smoothness checks;
* frozen change, Z40 section, top face, bbox y/z, bbox x (3.5), case_top surface and volume;
* topology, top face Z, max Y, battery face;
* stick cutout clearance, handle wall, internal walls;
* 2 x 2 seat checks, 2 x 2 trigger checks (pad gap, hand overlap), R1/L1 shift.

Soft checks (support >= 50 %) are reported. Information items (frozen dig, unsigned turn, grow/recede maxima,
internal object walls) are not judged.

The checks discriminate as intended. Measured with the fast loop:

| run | contacts in 0.2..1.0 | max dig (editable) | smoothness checks failed (judged region) | other hard fails |
|---|---|---|---|---|
| no-fit baseline | 0 of 8 | 5.19 | (nothing edited) | seats 4 |
| test swell (3 mm Gaussian at the ring PIP, setup check) | 0 of 8 (ring PIP gap +2.72 -> -0.04) | 5.20 | **0 of 6** (min concave R 9.2 mm, 0 % < 8) | seats 4 |
| old carve (imprint) | 5 of 8 | 0.53 | **3 of 6** (concave p0.5 5.5 mm, 10.6 % < 8 mm) | frozen 0.34 mm, bbox x 1.20, seats |
| old fill + carve + seats | 8 of 8 | 0.55 | **6 of 6** (concave p0.5 2.1 mm, 16.2 % < 8 mm, concave kink 79 deg/mm) | frozen 6.46 mm, bbox x 0.58, seat rims |

The old stack meets every contact and fails every smoothness check, which is exactly Olex's objection.

## 6. How to evaluate a prototype (one command)

```
PY=/tmp/claude-0/venv/bin/python
# full: stage + 9 case parts + cutouts + internals, full report, 3 x 3 sheet
$PY /tmp/claude-0/sdf/smooth/eval_proto.py -- --proto /tmp/claude-0/sdf/<you>/proto1 --tag t1 [--set 'My Param=2.0']
# fast loop: stage + R2/R4 only, no wrap / internals, no render
$PY /tmp/claude-0/sdf/smooth/eval_proto.py -- --proto /tmp/claude-0/sdf/<you>/proto1 --tag f1 --fast
```

`--proto DIR` must contain `mod_shape.py`. On first use `DIR/proto.blend` is copied from base.blend.
Results go to `DIR/runs/<tag>/`:
* `report.json` with keys `skin`, `parts`, `smooth`, `acceptance`, `summary`, `summary2`, `summary3`;
* `sheet.jpg` + `view_*.png`;
* `stage.npz` (skin V, T, change_mm, hf_* masks);
* `curvature.npz` (V, k1, k2, kink, kink_concave, kink_convex, mask, edited, change_vs_nofit);
* `parts/` (with `press_tris` on R2/R4), `cutters.npz`, `cutouts_*.npz`, `internals.npz`, `parts.json`.

The command ends with the acceptance table. The plain-GripV reference run is made once per proto folder
(`DIR/runs/base_proto_high_low`, +10 s). The same thing without the wrapper:

```
$PY /tmp/claude-0/sdf/smooth/stage/eval_all.py -- --blend DIR/proto.blend --modules-dir DIR --out-root DIR/runs --tag t1 [--skin-only --quick --no-render]
```

**Sheet (3 x 3):**
* Row 1: case outside / front / inside, coloured by the skin change vs plain GripV (blue receded, red grown,
  full colour at 4 mm).
* Row 2: hand front (capsules shown), underside, **curvature inside view**.
* Row 3: **R2 close-up**, **R4 close-up** (camera along each press-face normal), **curvature front-below**.
* Curvature colours: concave radius grey > 20 mm, yellow 12-20, orange 8-12, red 5-8, magenta < 5; blue =
  convex radius < 3 mm; dark = not measured.

**Sweeps in one session:** `sys.path.insert(0, '/tmp/claude-0/sdf/smooth/stage'); import stage_api as SA;
mod = SA.open_copy('DIR/proto.blend', modules_dirs=['DIR'])`, then `V, T, chg, s = SA.eval_skin(mod, {...})`
(~2.5 s per evaluation with the pass-through) and `GM.smooth_report(V, T, caps, nofit=(Vn, Tn), ...)` for the
new metrics (~10 s).

**Measure only** (an existing run folder):

```
$PY lib/measure_case.py -- --run DIR/runs/t1 --base-run DIR/runs/base_proto_high_low --nofit-run /tmp/claude-0/sdf/smooth/baseline/nofit [--quick]
```

### Timings (measured, 4 shared CPUs)

| step | time |
|---|---|
| open base.blend / a private copy | 0.55 s |
| rebuild the stage node groups | 1.8-2.1 s |
| stage alone, pass-through shape (subdivision high, Voxel 0.5) | 2.3 s (your module adds its own cost) |
| stage + 9 case parts + cutouts + internals (one depsgraph evaluation) | 14.5-16 s |
| plain-GripV reference run (once per out-root) | 9.5 s |
| measure: full report incl. smooth/ extension / `--quick` | 27-29 s / 20 s |
| of which smooth/ extension (curvature + kink on ~65k verts, contacts, dig, seats) | ~10 s |
| render 9 views 640 x 480, 16 samples, 2 threads + sheet | 65 s |
| **full eval_proto run** | ~2 min (117 s no-fit baseline, 125 s test swell) |
| **fast loop** (`--fast`) | ~26 s |
| validate_curvature.py | 20 s |
| make_targets.py | 2-3 s |

## 7. Shared data: targets.npz / targets.json

`np.load('/tmp/claude-0/sdf/smooth/targets.npz')`. `targets.json` lists every array with its shape and meaning
(`arrays`, `arrays_meaning`), the constants, the zones, the seats, the frozen rule and the hand / rig names.
In short:

* **Zone samples** (1022 points, 13 zones; `zone_id` indexes `targets.json: zones`):
  - `zone_pts` axis points every 0.5 mm, `zone_r` radius, `zone_bone` (index into `caps_names`), `zone_w`
    axis length per point;
  - directions: `zone_dir_nearest` (to the nearest no-fit skin point), `zone_dir_palmar` (rig bone axis,
    perpendicular to the bone, sign from the chain flexion), `zone_dir_flexion` (chain only; differs from
    palmar by the bone roll, up to 37 deg on the pinky metacarpal);
  - `zone_target_*` = axis + dir * (r - 0.5): where the skin should be for the rigid finger to sink 0.5 mm;
  - `zone_gap_nofit`, `zone_surf_nofit`, `zone_axis_inside_nofit`, `zone_frozen`.
* **No-dig capsules:** `caps_a`, `caps_b`, `caps_r` (all 20 hand parts), `caps_palmar`, `bone_frames`
  (world 3x3 per pose bone); 'hand points' as evaluated: `hand_points`, `_radius`, `_bone_id`,
  `_fill_zones`.
* **Seats:** `seat_<P>_center`, `_normal`, `_plane_point` (= centre - 0.2 n), `_opening_segments` (cutter
  section in the seat plane), `_press_pts`; `trigger_<P>_V/T`, `cutter_<P>_V/T`, `switch_<P>_V/T`.
  `targets.json: seats` also has the seat radius 2.5, the suggested fade 5, the cutter wall 1.2, the pad axis
  point, the press direction, the trigger parameters and the no-fit skin at the centre.
* **No-fit skin:** `nofit_V`, `nofit_T`, `nofit_frozen`, `nofit_handle`.
* **Keep-outs near the handle:** `static_cutout_*_V/T`, `internal_*_V/T` (with their no-fit wall in
  `targets.json: keepouts`).
* **Live names:**
  - object 'hand points' (826 points; point attributes `radius`, `bone_id`, `seg_first`, `seg_last`;
    vertex group 'fill zones');
  - armature 'hand rig' (20 bones, order = `bone_id`);
  - collection 'hand fit'; seat empties 'R2 seat', 'R4 seat', 'R1 seat';
  - triggers 'R2', 'R4', cutters 'R2 bool', 'R4 bool', switches 'R2_pcb_button', 'R4_pcb_button';
  - trigger parameters in 'GripV R2/R4 parameters' > 'trigger parameters.001'.

## 8. Notes and caveats

* **Measured, not a stage issue:** with the solved trigger values R2/L2 and R4/L4 have 1 non-manifold edge
  each, also with the stage off (old values: R2 0, R4 2). This comes from the generator / trigger values.
* The no-fit skin differs from plain GripV only by the remesh (Z40 section max 0.083 mm, p99 0.026) and the
  thenar module (z > 40, change -2.12 .. +0.03 mm, case_top 34,383.7 -> 34,172.2 mm3 bmesh).
* Zone directions: the axis is outside the no-fit skin for every zone point except where the capsule digs past
  its axis (`zone_axis_inside_nofit`). There the nearest direction is flipped, so it still points from the axis
  to the case.
* Grow vs recede in `change_vs_nofit` is signed by ray parity on the no-fit skin, only for vertices more than
  0.2 mm away.
* The thickness metric casts one ray per vertex. Check `min_at` before trusting a minimum near 0.
* Run-to-run: with the old fill off, separate runs are bit-identical (build/INTERFACE.md section 4.5). A module
  that uses Sample Nearest on 'hand points' can tie on coincident joint points.
