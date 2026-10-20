# Smooth hand fit: design decision

Date 2026-10-20. All numbers are **measured** with the shared metrics (`lib/gripmetrics.py`, full
`eval_all` runs) unless marked *inferred*. Right handle.

## v3 (review round 2, 2026-10-20): fixes to v2; same representation

**Installed:** `final/Grip_smooth.blend` holds v3 (v2 kept as `final/Grip_smooth_v2.blend`, md5 4a9964967cdd...).
Numbers are **measured** on the installed file (`final/runs/v3_final`, `final/runs/v3_check`, exports in
`final/export/`) or with the solver harness `final/work/v3/harness.py` on the solved offset h ("h-proxy": the
SPEC v2 relief metric applied to h on the stage input mesh), unless marked *inferred*.

**What v3 is:** plain GripV with the pinky and palm press eased by up to about 2 mm (h -2.07..+0.82 mm). It is not
yet a grip the middle finger grabs: the middle finger rests on R2 and touches nothing else.

### Decisions
1. **View off the dashboard.** The dashboard has 7 controls (Fit, Shape Size, Palm Give, Finger Give, Stroke Clear,
   Screw Wall, Thenar). View is a hidden modifier input, inspection only; View 1 now shows the remesh output with
   outward winding. `export_check.py`, `check.py` and `stage/eval_all.py` refuse a file whose View is not 0.
2. **Refit gate.** Values must be finite numbers; the lever must be under its switch (81/81 rays, gap min / max
   within 0.05 mm of the accepted 0.438..0.484 (R2) / 0.470..0.491 (R4)); after the solve, the result goes to a
   side file and `trigger_solver --check-case` must report no blocked lever path and no trigger vertex in the case
   before the file is replaced (exit 2 = trigger check, 3 = case check; file and `solve/` arrays unchanged on a stop).
3. **Solver priorities (`final/solver_v2/lowfreq.py`, v2 copy `lowfreq_v2.py`).** Weights hard 1e7 > no-dent 3e5 >
   Give 3e4 > crease 1e4 > touch 3e3 / bump 3e3 (v2: hard 1e5, dent 3e3). A touch zone is pulled only when its
   nearest skin vertex has window >= 0.5 (`touch_wmin`) and it is within **Touch Reach** 1 mm of touching (the
   penalty is capped there, so a far zone no longer pushes a constant force into the shape). Each zone's status is
   written to `solve/refit_Grip_smooth.json` (`zones_status`).
4. Seat empties deleted (unused by v2/v3).

### Before / after per review issue
| issue | v2 (before) | v3 (after) | how measured |
|---|---|---|---|
| View 1 inside-out / View 2 changes case_top | View 1 skin -433,723 mm3 (inside-out); View 2 coarse 54,846 verts (case_top 2,339: make2's number, not re-measured) | View hidden; View 1 +433,723 (outward); export_check / check / eval_all exit 1 on View 2 | `work/v3/view_test.py`, refusal runs |
| gate passes NaN, ignores switch gap, no case check | NaN centre passed; 40/81 switch rays passed | 9/9 trigger-gate and 5/5 case-gate unit cases as expected; real refit: R2 / R4 lever path 81/81, 0 blocked, 0 trigger verts in case, case check 24.5 s | `work/v3/gate_unit.py`, `out/case_unit.json`, refit log |
| middle finger touches nothing | wording said "low frequency cannot reach it" | cause is the window and the pose: window at the nearest skin 0.004 (root pad) and 0.43 (PIP); gaps 0.87 / 5.51; not fixed (open question) | solver zone status |
| top-band ridge 0.75 mm, 8-11 mm wide | profile at y -52.5 (outer x vs no fit): +0.51 z23, +0.74 z27, +0.65 z29, +0.42 z31; kmax 3 / 6 mm at (80.6,-52.4,28): 0.057 / 0.039 | monotonic recession -1.11 z21 ... -0.17 z31, 0.0 z35; kmax 0.0064 / 0.0069 (no fit 0.0088 / 0.0089); grazing close-ups show no streak | `work/v3/skin_profile.py`, `renders/closeup/` |
| re-pose not robust | pinky -5 deg: 1 dent, 1.17; largest far change 1.78 mm | pinky -5: 0 dents (0.89), 1 bump 1.19, pinky middle 0.66 over Palm Give; far change 1.22 mm (31.6 mm from the pinky) | h-proxy, `work/v3/harness.py`, `nonlocal.py` |

Full re-pose table (h-proxy; one bone, the rest v_27): installed 0 dents (0.66, v2 0.90); pinky +5 0 (0.60; far
change 1.68 at 16 mm, v2 2.65); middle +5 with R2 refit 0 (0.68); ring +5 with R4 refit 0 (0.69); ring knuckle -5
**gate stops** (R4 travel 28.3 deg) 1 dent 1.46 / 2 bumps (v2 1.99); middle -5 **gate stops** (R2 centre 1.022 mm,
travel 33.4 deg) 4 dents 4.77 (v2 1 dent 4.39). The two stopped poses are not installable, by design.

### SPEC v2 table v3 (`final/runs/v3_final/spec_v2.json`; v2 column from the table below)
| row | v2 | **v3** |
|---|---|---|
| offset vs no-fit (mm) | -1.80..+0.96 | **-2.07..+0.82** |
| dents / bumps >= 1 mm | 0 / 0 | **0 / 0** (h-proxy max dent 0.90 -> 0.66, max bump 0.75 -> 0.78, area abs relief > 0.5: 1141 -> 1058 mm2) |
| relief margin on the real skin (largest sub-threshold prominence, `work/v3/out/relief_margin.json`) | dent 0.88 (81.7,-72.2,22.8); bump 0.97 (rim of the edit, (64.2,-49.2,-14.9)); edited bump 0.72 (80.6,-52.2,27.2: the ridge) | **dent 0.64 (63.2,-53.7,-10.8); bump 0.87 (same rim point); edited bump 0.86 (38.5,-35.5,12.0, inner side)** |
| 6 mm concave R<20 / convex R<8 vs plain | x1.06 / x0.78 | **x1.02 / x0.76** (11.65 vs 11.46 %; 6.91 vs 9.07 %) |
| concave creases, same vertex: min ratio / % > 20 % tighter | 0.85 / 0.0 | **0.38 / 0.22** (worst (27.2,-35.8,30.6); 6 mm median R there: no fit 9.9, v2 9.4, v3 8.7) |
| concave creases, 2 mm neighbourhood | 0.87 / 0.0 | **0.79 / 0.02** |
| convex creases, same vertex / 2 mm | 0.76 / 0.36, 0.85 / 0.0 | **0.77 / 0.24, 0.85 / 0.0** |
| touches (mm): middle root, middle PIP, ring root, ring PIP | +0.87, +5.55, -2.48, +2.69 | **+0.87, +5.51, -1.44, +3.44** |
| touches: pinky root, pinky PIP, pinky distal, palm | -2.39, -3.01, -1.85, -3.39 | **-2.20, -2.91, -2.00, -3.37** |
| Give over limit | none | **none** (ring metacarpal 3.37, pinky middle 3.43, pinky metacarpal 2.71, middle metacarpal 1.77, ring middle 0.0) |
| trigger stroke R2 / R4 rest / pressed | 0.96 / 0.49, 1.40 / 0.92 | **0.97 / 0.49, 1.30 / 0.85** |
| screw wall (where plain >= 3) | 3.02 | **3.0**; below z 12 (plain 0-2.9) equal to no fit within 0.01 per band |
| frozen / mirror | 0 / 0 | **0 / 0** |
| case_top / middle plate vs no-fit (max mm, verts > 0.05) | 0.014, 0 / 0.017, 0 | **0.012, 0 / 0.014, 0** |
| bbox x extent vs no-fit | -0.43 | **-0.596** |
| skin volume (mm3) | 429,490 | **426,603** (outward, genus 0) |
| change vs no-fit by z: 34-35 / 35-35.5 / 36-36.5 / >= 36.5 | 0.118 / 0.021 / 0.002 / 0.0 | **0.070 / 0.012 / 0.0014 / 0.0** |
| finger segments: middle middle, ring proximal, ring middle (gap) | 2.63, 5.31, -0.19 | **2.74, 6.68, +0.24** (no fit 2.27, 4.93, -0.44) |
| inner-back bump change vs no-fit | 0.96 | **0.81** |

Shared metrics (v3_final): top face 27,566 verts at 59.04002, nothing above; max Y -4.158; bbox min shift 0.0001;
Z40 section 0.0; 21/21 parts outward; 24/24 other parts equal to plain; cable holder +0.4 mm3; stick clearance 2.25;
handle wall p1 9.67; R1/L1 shift 0; case_top volume +0.1 mm3, middle plate -0.3 mm3 / bbox 0.004 vs no-fit.
The shared table's old tight-radius rows still fail (concave R<8 1.6 %, concave p0.5 2.98, convex p0.5 1.59, kinks
41.07 / 61.69; v2 1.26 / 4.12 / 1.63 / 39.5 / 60.8); the screw-bools row is a misfire on the channel that exits
through the skin (-12.26, v2 -12.22).

### Exports v3 (`final/export/`)
export_check 128 s: case_top 35,874.6 mm3 (15 tri non-manifold edges; plain reference 15), case_bottom 174,094.8
(5: 1 L2, 2 R2, 1 R4 cutter rim, 1 right screw exit at (65.89,-62.34,4.97); plain reference 0; v2 5), R2/L2 2,924.6,
R4/L4 5,276.7, cable holder 1,422.6; all outward. Skin vs design max 0.061 mm. STL vs design: case_bottom max
0.056; case_top max 9.04 at (-10.73,-41.76,48.16), and plain GripV through the same exporter has the same 9.04 at the
same point (p99 6.58), so it is the export settings. 'Case standard' preset (305 s, run alongside the plain export):
case_bottom 174,094.82 (welded non-manifold 5), case_top cap_knob 35,874.56 (17), cap_plate 35,887.84 (12),
dpad_stick_cap_knob 37,187.57 (19), dpad_stick_cap_plate 37,200.85 (14); all outward. case_top STL volume is
35,875 for v3 and 35,873 for v2 but 36,153 through the exporter with the hand fit off: that difference is the stage
remesh at any Fit (*inferred* from v2 = v3 there; not located spatially).

### Corrections to the v2 text below
- "any combination is C4-smooth" (Representation): h = W x (B c) is only C2 (the window is C2), and smoothness of
  the basis does not forbid narrow features: with centres 9.7 mm apart a combination can form an 8-11 mm wide ridge
  (v2 had one). Narrow features are kept out by the relief rows of the solve, not by the basis.
- "A broad forward swell cannot make the middle finger touch near its PIP": the cause is the window (the PIP's
  nearest skin has window 0.43, the root pad's 0.004) together with the pose, not the low frequency of the basis.
  Whether opening part of the top band would let it touch is not measured (it needs case_top / middle-plate checks).
- The crease rows reported only the 2 mm-neighbourhood value; the same-vertex value is now reported next to it (v2:
  concave 0.85 / 0.0 %, convex 0.76 / 0.36 %).
- "stays exactly zero above z 35": the shape is zero above z 35; the remeshed case changes by up to 0.021 (v2) /
  0.012 (v3) mm between z 35 and 36.5 and is unchanged above z 36.5.
- "keeps at least 3 mm of wall around the two handle screw channels": only where plain GripV has 3 mm; at the screw
  exits plain is 0-2.9 mm and the fit keeps that wall.
- Non-manifold attribution: case_top's 15-16 tri non-manifold edges are GripV's own (plain reference 15); only
  case_bottom's 5 come with the fit.

### Remaining open questions (Olex)
Middle finger (curl the pose / open part of the top band / accept R2 only); ring PIP (finger-wide swell / larger Touch
Reach / accept 3.4 mm); Palm Give 4.3 vs width -0.6 and the palm dish; taper of the top band (20 mm: shelf
curvature 0.0163 vs 0.0256 at (80,-52.5,32.5) but ring middle no longer touches, gap 0.52); a local stroke scallop
vs the broad recession under the middle finger; 26 deg trigger travel as the re-pose envelope; pinky press 0.66 mm
beyond Palm Give vs a dent; whether the shared `smooth/lib/gripmetrics.py` should drop the v1 rows; whether refit
should stay history-free or regularise toward the installed shape.

## v2 decision (2026-10-20): ONE low-frequency shape (SPEC v2) replaces v1 (superseded by v3 above; corrections listed there)

**Installed:** `final/Grip_smooth.blend` holds the v2 shape since 2026-10-20. The repaired v1 file is kept as
`final/Grip_smooth_v1.blend`. All numbers are **measured** on the installed file (runs `final/runs/v2_final`,
`final/runs/fix_after` for v1, re-pose `final/runs/repose_ringMCP5`) unless marked *inferred*.

Why v1 was replaced: its spec asked for 0.2-1.0 mm penetration of **rigid** capsules at 8 zones where plain GripV
already overlaps them by 2.3-4.8 mm, so the solver receded capsule-shaped pockets (5 dents / 7 bumps of >= 1 mm
prominence, offset -4.9..+7.7 mm). The independent check refuted it as a smooth shape.

**Representation (SPEC v2 item 1).** `h = W * (B c)` on the stage input mesh (subdivided GripV basemesh, 'high'):
- B: 144 Wendland C4 radial basis functions, support radius 34 mm (68 mm across, Shape Size), centres every 9.7 mm
  on the right handle. A single function is about 15 mm wide at half height; ~~any combination is C4-smooth~~
  [v3 correction: h is C2 and the basis does not forbid 8-11 mm features; see v3 above].
- W: a C2 window, 0 on the frozen region (|x| < 25, z > 35 = 38 - Top Clear 3, front band y > -4.658) and fading in
  over 10 mm; it also fades to 0 over 6 mm at the bbox min y / z planes (screws).
- c: the weights, minimising thin-plate bending + a small membrane term + |h|^2/22^4, with squared-hinge rows
  re-linearised at the moved skin: one-sided touches (gap <= 0.3 mm), Palm Give 3.5 / Finger Give 1.5 (only the
  excess recedes), the R2 / R4 distal capsules swept along the trigger travel keeping 0.3 mm from material outside
  the openings, screw wall >= 3 (never thinner than plain), the other internal and cutter walls, max Y, bbox and x
  extent limits, crease caps tapering over 10 mm, and the relief metric itself (|h - ring mean| <= 0.55).
- **The cage was not used.** Olex's cage edges in the handle are median 7.2 mm and 5th percentile 2.2 mm, so cage
  supports would fall below 25 mm (edge lengths measured by the v2 work; the support size is *inferred*).
- Solver: `final/solver_v2/lowfreq.py` (from `v2/solver`), called by `final/refit.py`. Solve 6.5 s; the whole
  refit after a control change 67 s. It reproduces the v2 work's `v2/solve/best.npz` exactly (h max diff 0.0,
  same 144 weights; the v2 report's "139 functions" was a slip).

**Controls (8 in 'GripV hand fit parameters'):** Fit, Shape Size 34, Palm Give 3.5, Finger Give 1.5, Stroke Clear
0.3, Screw Wall 3.0, Thenar, View. Hidden on the modifier: Middle PIP Pull 0, Top Clear 3. The v1 controls
Smoothness / Press / Max Dig / Seat Pull / Seat Proud are removed (v1's Smoothness 18 meant a minimum concave
radius; reusing it as a basis radius would have been wrong). Only Fit / Thenar / View act at once; the five solve
inputs act when `final/refit.py` runs.

### Table: plain GripV / no-fit / v1 / v2 (SPEC v2 metrics, `final/lib_v2/relief.py`)

Plain and no-fit columns are from the v2 work (`v2/solve/an/plain.json`, `nofit.json`; same meshes); v1 and v2 were
measured here (`final/runs/fix_after/relief_v1.json`, `final/runs/v2_final/relief.json`). Plain is the raw GripV mesh
(not remeshed, no thenar), so its curvature-vs-itself, crease and frozen rows are not comparable ('-').

| metric (right handle) | plain | no-fit | v1 | **v2** | SPEC v2 |
|---|---|---|---|---|---|
| offset vs no-fit (mm) | - | 0 | -4.93..+7.74 | **-1.80..+0.96** | - |
| dents / bumps, prominence >= 1 mm vs 12-16 mm ring | 0 / 0 | 0 / 0 | 5 / 7 | **0 / 0** | 0 / <= 2 |
| 6 mm concave R<20 %, vs plain same footprint | - | - | 14.37 vs 4.94 (x2.91) | **13.2 vs 12.51 (x1.06)** | <= 1.5x |
| 6 mm convex R<8 %, vs plain same footprint | - | - | 14.16 vs 6.81 (x2.08) | **7.59 vs 9.68 (x0.78)** | <= 1.5x |
| GripV concave creases: min R ratio / % > 20 % tighter (2 mm nbhd) | - | - | 0.48 / 7.56 | **0.87 / 0.0** | judged vs no-fit |
| GripV convex creases: same | - | - | 0.80 / 0.12 | **0.85 / 0.0** | judged vs no-fit |
| gap middle root pad (mm, <= 0.3 touches) | +0.87 | +0.87 | -0.51 | **+0.87** | touch |
| gap middle PIP | +5.67 | +5.69 | +2.61 | **+5.55** | touch (item 4) |
| gap ring root pad | -2.27 | -2.26 | -0.59 | **-2.48** | touch |
| gap ring PIP | +2.75 | +2.75 | -0.61 | **+2.69** | touch |
| gap pinky root pad | -2.30 | -2.30 | -0.81 | **-2.39** | touch |
| gap pinky PIP | -4.72 | -4.71 | -0.57 | **-3.01** | touch |
| gap pinky distal | -3.22 | -3.21 | -0.80 | **-1.85** | touch |
| gap palm on swell | -4.27 | -4.26 | -0.99 | **-3.39** | touch |
| overlap ring metacarpal (Palm Give 3.5) | 4.27 | 4.26 | 0.59 | **3.39** | <= 3.5 |
| overlap pinky middle (Palm Give 3.5) | 5.20 | 5.19 | 0.85 | **3.43** | <= 3.5 |
| overlap ring middle (Finger Give 1.5) | 0.44 | 0.44 | 0.82 | **0.19** | <= 1.5 |
| trigger stroke R2: fingertip gap rest / pressed | 0.28 / -0.19 | 0.28 / -0.19 | -0.51 / -0.93 | **0.96 / 0.49** | pressed >= 0.3 |
| trigger stroke R4: same | 0.56 / 0.08 | 0.56 / 0.08 | -0.38 / -0.77 | **1.40 / 0.92** | pressed >= 0.3 |
| screw-channel wall min (mm) | 3.0 | 3.0 | 2.24 | **3.02** | >= 3 |
| frozen change / mirror dev (mm) | - | 0 / 0 | 0 / 0 | **0 / 0** | <= 0.05 |
| case_top dev vs no-fit (max mm, verts > 0.05) | - | 0, 0 | 0.262, 1031 | **0.014, 0** | unchanged |
| case_middle_plate dev vs no-fit | - | 0, 0 | 0.275, 1927 | **0.017, 0** | unchanged |
| bbox x extent (mm) | 82.901 | 82.896 | 82.351 | **82.466** | open question |
| skin volume (mm3) | 433,766 | 432,691 | 437,995 | **429,490** | - |

Shared metrics on the same run (`final/runs/v2_final/report.json`, `final/check.py`): frozen 0.0, Z40 section 0.0, top
face 27,566 verts at 59.04 with nothing above, max Y -4.158, bbox min y / z shift 0.0, skin genus 0 / 1 island, skin
raw volume +429,490 (outward), all 21 parts outward, all 24 non-case parts and the cable holder equal to plain GripV,
case_top volume -1.8 mm3 vs no-fit, case_middle_plate +0.9 mm3 / bbox 0.008 vs no-fit, R1/L1 shift 0, stick clearance
2.25, handle wall p1 9.14, trigger pads -0.5 / overlap 0.5.

**Old tight-radius checks (kept, as SPEC v2 asks).** On the shared judged region (edited + 3 mm, minus unchanged
features) v2 fails them: % concave R < 8 = 1.26 (<= 0.5), concave R p0.5 4.12 (>= 8), convex R p0.5 1.63 (>= 3),
kinks 39.5 / 60.8 deg/mm (<= 10 / 25). v1 passed four of them (0.28, 10.59, 3.68; kinks 20.5 / 31.7). The reason
is measured: v2's footprint is broad and contains GripV's own creases. On the edited + 3 mm region against the no-fit
skin on the same footprint, v2 is 4.24 vs 3.94 %, 2.26 vs 2.08, 1.53 vs 1.55, 44.9 vs 42.4, 70.2 vs 71.9; and no
GripV crease got more than 20 % tighter (table). So these are GripV's edges, not new ones.

### Where v2 falls short (honest list)

1. **Three zones do not touch:** middle PIP +5.55, ring PIP +2.69, middle root pad +0.87 (plain: +5.67 / +2.75 /
   +0.87). The v2 work's sweeps (`v2/solve/*.npz`, measured with the solver's proxy metric):
   - middle PIP: it lies under the top band (nearest skin z 31.5) next to the R2 fingertip. With no relief or crease
     limits the best was gap 1.16, and only with a 5 mm bump (prominence 4.9) and a -3.8 mm dent. **A broad forward
     swell cannot make the middle finger touch near its PIP** (SPEC v2 item 4: not met). [v3 correction: the cause
     is the window and the pose, not the low frequency; see v3 above.]
   - ring PIP: touching needs a bump of 3.3 mm about 16 mm across (finger width), because the R4 stroke keep-out is
     9 mm away.
   - middle root pad: needs about +0.6 mm at z ~30.5, where the window is about 0.2 (zero band from z 35). A 6 mm
     taper reached 0.67 but left a visible step under the top band.
2. **Zebra calmness** (my look at `final/final_sheet.jpg`): much calmer than v1 and grazing light is close to plain,
   but zebra is busier than plain GripV: wavier broad stripes and a large flat-normal patch on the outer face, and a
   few small rings near the top band (outside / frontout) and underneath. No extremum reaches 1 mm prominence, so
   they are sub-mm broad undulations (*inferred* from the relief metric; whether the small rings near the top band are
   voxel-remesh aliasing is not measured).
3. **Width:** bbox x extent shrinks 0.43 mm (82.466 vs 82.896): the broad palm / ring-metacarpal recession reaches
   the outermost point. A larger Palm Give (~4.3) would keep it (*inferred* from the overlaps: ring metacarpal 4.26 in
   no-fit).
4. **Trigger tops stand proud:** the skin at the press-face centres is 1.44 / 1.51 mm below the R2 / R4 press faces
   (v1 0.22 / 0.15); the old flush-seat checks are v1 rules and are not part of SPEC v2. The stroke keep-out is why.
5. **Shared rule misfire:** 'internal objects: wall >= min(no-fit, 1.5)' fails on 'screw bools' (-12.22 vs -12.13 at
   (-73.8, -62.1, -16.9)). That point is below the case bottom (bbox min z -15.32), where the screw hole opens through
   the bottom: there is no wall there. The real screw-channel wall minimum is 3.02 mm.

### Re-pose test (ring MCP curled 5 deg, `final/tests/repose_ringMCP5.blend`)

- Pose: 'ring_proximal' rotated -5 deg about its local X (the same pose as the earlier 'ringflex5' tests). Measured
  as a curl: the ring fingertip moves 4.96 mm and comes 3.8 mm closer to the metacarpal's wrist end.
- `refit.py --triggers`: the trigger solver re-fits R4 (centre 0.002 mm, normal 0.0 deg, overlap 0.5) but its travel
  ends 28.3 deg off the press direction (> 26). **Refit stops with exit 2 before the shape solve** and does not save
  (97 s).
- With `--ignore-trigger-check`: 207 s in total (trigger fit 123 s, gather + solve 57 s, solve 10.0 s), slowed by
  the combi export running at the same time (the v2 work measured 104 s on a quiet machine).
- Result: offset -2.78..+0.92; **1 dent** (prominence 2.07 at (43.2, -40.7, 8.7), 1.3 mm from the ring distal
  capsule), forced by the hard R4 stroke keep-out because the curled fingertip sweeps into the case; 0 bumps;
  6 mm ratios 1.11 / 0.79; ring PIP gap +0.51 (closer, still not touching), middle PIP +5.51, middle root pad +0.86;
  allowances all within; stroke R2 0.95 / 0.49, R4 0.94 / 0.48; screw wall 3.0; frozen 0.0; case_top 0.018 mm.
  Concave creases: min ratio 0.62 and 0.51 % more than 20 % tighter, next to the dent.
- case_middle_plate changes by up to 0.97 mm (106 verts > 0.05 mm). All 104 vertices that move more than 0.5 mm lie
  within 1 mm of the re-fitted R4 cutter or its L4 mirror, which moved by up to 5.07 mm (bbox). So it is the
  generator following the moved triggers, not the shape (zero above z 35). The v2 work's guess "the thenar follows
  the hand" was wrong.

### Export (installed v2 file)

- `export_check.py` (export settings: subdivision 'very high', voxel 'high', tight tolerances): every part outward;
  case_top +35,872.7 mm3, case_bottom +177,014.3, cable holder +1,422.6. Triangle-level non-manifold edges: case_top
  16 (plain 15), case_bottom 5 (4 on the R2 / R4 rims; v1 24, plain 0), triggers 1 each (as plain). The exported
  case_bottom matches the design-level v2 case_bottom within 0.049 mm (p99 0.014; `final/export/export_vs_design_parts.json`).
- Olex's own exporter, preset 'Case standard', headless (`v_make/scripts/run_combi.py` on a copy):
  `final/export/combi_case_standard/case_standard/`, 291 s. All 5 STLs have positive volume: case_bottom 177,014,
  case_top variants 35,873 / 35,886 / 37,186 / 37,199 mm3 (`stl_check.json`). The earlier v1 export
  (`v_make/combi`, before the winding fix) was inside-out (-184,567 for case_bottom).

### Files

- `final/refit.py` (v2 solver wired in; trigger check as before), `final/solver_v2/` (lowfreq, gather, gather_np,
  store), `final/fit_opts.json` (v2 options), `final/modules/mod_shape.py` (v2 controls; the nodes are unchanged),
  `final/stage/build_stage.py` (CONTROL_NAMES), `final/check.py` (adds the SPEC v2 table: relief.json / spec_v2.json),
  `final/lib_v2/` (relief.py, vlib.py, render_relief.py), `final/scripts/sheets_v2.py`, `final/scripts/stl_volumes.py`.
- v1 kept: `final/Grip_smooth_v1.blend`, `final/solver/v1/` (refit, fit options, mod_shape), `final/export/v1/`.

---

# v1 record (superseded by v2 above)

## Decision

The winner is the **dense solve** (`p_dense`): one height h per vertex of the subdivided basemesh, solved
offline with thin-plate bending, soft contacts, no-dig and frozen limits, and curvature limits. The cage
result (`p_cage`) was not chosen: it is smooth, but its max dig is 1.83 mm (limit 1.0) and only 2 of 8
contact zones are in range. That fails the rule "cage only if max dig <= 1 mm".

One idea from the cage prototype is **grafted**, and it removed most of the dense solve's grooves: **soft
trigger seats**.
- The dense prototype held the skin around each R2/R4 opening in hard boxes, +-0.12 mm from the seat plane.
  Next to GripV's locked design creases and between the two openings, those boxes cannot all be met. The
  result was ripples and creases: 64 % of the R < 8 mm groove area and most of the kinks were within
  14 mm of a seat (measured with `final/scripts/where_bad.py`).
- The cage solve used soft seat targets. The final solve does the same: every rim point is a soft
  target with weight 'Seat Pull' 0.05. Only a 4 mm disc at the press-face centre stays a hard box, so the
  press face still stands about 0.2 mm proud.
- Cost: the seat rims are not flush (see the gaps below).

## Comparison (full runs; limits from PROBLEM.md)

| metric (judged region = edited + 3 mm, minus unchanged features) | cage c2_b | dense c8 | **final (dense + soft seats)** | limit |
|---|---|---|---|---|
| hard checks failed (of 38) | 13 | 12 | **7** | 0 |
| contact zones in 0.2-1.0 mm | 2 / 8 | 7 / 8 | **7 / 8** (middle PIP 0) | 8 / 8 |
| max dig, editable region (mm) | 1.83 | 0.99 | **0.99** | <= 1.0 |
| % area concave R < 8 mm | 0.0 | 2.46 | **0.28** | <= 0.5 |
| % area concave R < 12 mm | 5.1 | 7.00 | **1.06** | <= 5 |
| concave radius p0.5 (mm) | 10.86 | 3.73 | **10.59** | >= 8 |
| min concave radius (mm, info) | 10.0 | 2.42 | 3.92 | - |
| convex radius p0.5 (mm) | 3.54 | 2.73 | **3.68** | >= 3 |
| max concave kink (deg/mm) | 9.53 | 42.5 | 20.45 | <= 10 |
| max convex kink (deg/mm) | 41.56 | 55.3 | 31.71 | <= 25 |
| frozen region change (mm) | 0.0001 | 0.0 | **0.0** | <= 0.05 |
| case_top surface vs no-fit (mm) | 0.018 | 0.258 | 0.262 | <= 0.05 |
| bbox x extent shrink (mm, open question) | 0.59 | 0.535 | 0.545 | (Olex) |
| R2 seat: skin at centre / rim max dev (mm) | 0.203 / 4.31 | 0.229 / 4.39 | 0.218 / 4.70 | 0.1-0.3 / 0.3 |
| R4 seat: skin at centre / rim max dev (mm) | 0.073 / 4.08 | 0.309 / 1.70 | 0.148 / 8.00 | 0.1-0.3 / 0.3 |
| stage time per update (s) | 5.95 | 3.9 | 3.8 | - |

Sources:
- `p_cage/runs/final_c2b/report.json`
- `p_dense/runs/best_c8/report.json`
- `final/runs/final/report.json`

### Screening solves

These ran with the solver's quick metric on the moved input mesh, before the remesh. Warm start from c8 or d1.

| run | change | concave R < 8 / < 12 % | concave p0.5 | kinks cc / cv | R4 rim min (mm) |
|---|---|---|---|---|---|
| c8 | hard seat boxes (dense prototype) | 1.94 / 6.22 | 3.28 | 45.3 / 52.7 | -1.12 |
| **d1** | **all rims soft, Seat Pull 0.05** | **0.03 / 0.57** | **11.79** | **15.1 / 30.5** | -7.95 |
| d2 | d1 + curvature weight x3.3 | 0.02 / 0.89 | 11.41 | 16.9 / 34.6 | -7.85 |
| d3 / d4 | Seat Pull 0.3 / 1.0 | 0.03 / 0.61, 0.97 | 11.76 / 11.5 | 15.5 / 30.6 | -8.0 / -7.8 |
| e1 | hard boxes away from the R2-R4 corridor | concave excess back to 0.36 /mm | | | |
| f1 | d1 + bending x3 | 0.03 / 0.55 | 11.78 | 13.5 / 29.5 | -7.70 |
| f2 | d1 + min convex R 6 + contact sigma 2.5 | 0.03 / 0.69 | 11.73 | 23.4 / 36.9 | -7.61 |
| p075 / p150 | Gaussian post-smoothing of d1 (0.75 / 1.5 mm) | 0.38 / 1.23, 0.42 / 2.37 | 9.2 / 8.4 | 15.1 / 22.5, 9.9 / 19.0 | dig 1.21 at 1.5 mm |

- d1 was kept. f1 is marginally better in the quick metric but still fails both kink limits, so it was
  not worth another full run.
- The Seat Pull weight barely moves the rims: the rims are limited by smoothness, not by the weight.

## What was grafted, and what was not

Grafted:
- **Soft seats** from the cage, as described above.
- **The helper-collection lesson** from the cage run. The solved shape lives on the mesh object
  'shape offset' in collection 'hand fit', never in 'basemesh GripV' (the generator reads that collection),
  and `final/refit.py` always puts it there.
- **One 'Fit' slider** (the cage's 'Shape Strength'). It scales the stored shape live: 0 = no fit,
  1 = the solved fit.
- **The cage's list of Olex inputs**, as 8 named controls in the node group 'GripV hand fit parameters'.
  That group sits in the dashboard 'GripV controls', frame HAND FIT.

Not grafted:
- **Coarse cage carriers**: a second parameterisation, with no install path for level 2.
- **The SDF(shaped) - SDF(unshaped) trick**: the shape is the first module, where Phi = the base SDF, so it
  would be identical.

## Gaps against PROBLEM.md, and why

These are the 7 failed hard checks of `final/runs/final`.

1. **Middle PIP contact = 0** (gap +2.49 mm). The middle phalanx lies in front of the max-Y plane
   (y -4.158, which places R1/L1). Only a forward bump about 6 mm high and about 40 mm wide (*inferred*)
   could reach it with R >= 14-18. Fix: curl the middle finger tighter in the pose, or accept a local
   groove there.
2. **Concave kink 20.45 deg/mm** at (60.9, -16.1, 26.2), handle front:
   - the spot is an edited vertex (0.35 mm) 2-4 mm from one of GripV's own concave creases, which the
     solver locks (|h| <= 0.12);
   - the lock boundary reads as a crease (the no-fit value at that point is 2.9).
3. **Convex kink 31.71 deg/mm** at (47.2, -37.3, -3.3):
   - it is 1.4 mm outside the R4 cutter, on the lower rim, where the ring and pinky contacts pull;
   - these two kinks are 1-2 mm features. The 3 mm curvature limit of the solver does not see them, and
     bending x3 or a tighter convex limit did not remove them (f1, f2).
4. **R2 / R4 seat rims**: max deviation 4.70 / 8.00 mm against a limit of 0.3.
   - The skin at the press-face centre is in range (0.218 / 0.148).
   - Flush rims need an 8 mm step within 2.5 mm of the R4 opening, and the two openings are 8.6 mm
     apart with planes 11.7 deg apart. That cannot be smooth, so smoothness won, as Olex asked.
   - The trigger tops therefore stand out of the case on one side, like a normal trigger.
5. **case_top surface 0.262 mm** at (-77.3, -62.3, 40.0):
   - this point is 3.5-3.9 mm inside the skin, on case_top's inner shell wall (measured); the outer skin
     there is frozen and unchanged;
   - the edits below reach z 37.5 with up to 4.8 mm of recession (the palm and metacarpal no-dig);
   - *inferred*: the shell's inward offset sees them;
   - the volume change is +0.3 mm3 (passes);
   - stopping the edits further below z 38 would push the palm dig over 1.0 (dense run c1).
6. **bbox x extent -0.545 mm**: the palm contact sits on the case's outermost point. This is the open
   question for Olex (PROBLEM.md 3.5). 'screw bools' does not follow x.

Everything else passes:
- the 7 other contacts, max dig 0.99, and 4 of the 6 smoothness checks;
- the frozen region, top face, max Y, Z40 section, bbox y/z, topology, battery face;
- stick clearance, walls, triggers, R1/L1.

## Corrections and generator repairs (2026-10-20)

These repairs leave the v1 shape as it is. The numbers are **measured** unless marked *inferred*. Run folders:
`final/runs/fix_before` (the delivered v1 file, its saved stage) and `final/runs/fix_after` (the repaired file).
Both were measured with the same new checks. Scripts and logs are in `final/fix/`.

1. **Winding.** The stage output was inside-out from build/ onwards. The skin's raw signed volume was -437,995 mm3.
   case_top (-34,172) and case_bottom (-184,395) inherited it, and so did every STL Olex's exporter wrote.
   - The cause is Grid to Mesh in 'HF Stage'. It now ends in a Flip Faces node (`final/stage/build_stage.py`), and
     the inspection views 1 and 2 are flipped too. The Enable-off pass-through is unchanged.
   - After the fix the skin is +437,995 and all 21 non-empty parts are positive. No vertex moves: the skin is the
     same as the in-memory flip test, and at Fit 0 its vertex set is identical to `baseline/nofit`.
   - Rebuilding the stage from the scripts (which `check.py` does on every run) moves 7,395 left-handle vertices
     by at most 0.0016 mm against the saved v1 group. The old script does the same, so this is not the flip.
   - The fit-dependent case_top volume changes by -38 mm3 when the winding flips, and case_bottom by +248 mm3. The
     surfaces stay within 0.018 / 0.005 mm. The difference is how the large non-planar boolean n-gons get
     triangulated. Because of it, the no-fit reference for the repaired stage is `final/runs/nofit_fix` (Fit 0,
     outward), and `check.py` uses it by default. Measured against the old inside-out `baseline/nofit`, the case_top
     volume check would read -39.9 mm3.
2. **'cable holder'** (preset 'Parts'). Before the fix: 2,934.9 mm3 with Euler characteristic -6, and overlapping
   shells (plain GripV: 1,422.4 mm3, characteristic 0). After the fix: 1,422.8 mm3, characteristic 0. The volume is
   +0.4 mm3 against plain, inside the 1 mm3 limit.
   - `eval_all` now evaluates every object that Olex's GripV presets export and that reads 'basemesh variant',
     27 parts in all. `measure_case` checks each one against plain GripV: volume within 1 mm3 and the same Euler
     characteristic. case_top and case_bottom are excluded because they change by design.
   - case_middle_plate is judged against no-fit instead, because the stage remesh alone moves it by +1.9 mm3. That
     check fails for v1 (-1.9 mm3, bbox shift 0.23 mm), which matches the review: the fit changes the plate.
3. **Triangle-level non-manifold edges** (export settings). The v1 report said that case_bottom's 30 were 'new with
   the fit' and that the cause was untraced. Both claims were wrong. The stage alone, at Fit 0, already made 12 of
   them, and they sit on the trigger-cutter rims. After the winding fix:
   - case_bottom has 24 at Fit 1, and all 24 are within 0.05 mm of 'R2 bool' / 'R4 bool'. It has 11 at Fit 0 (8 of
     them on the rims) and 0 in plain GripV.
   - case_top has 16 at Fit 1, 15 at Fit 0 and 15 in plain GripV. They are internal (z 40 parting face, z 43.8,
     z 55), and none is on a trigger rim.
   - Before the fix the counts were 26 / 30 at Fit 1 and 24 / 12 at Fit 0. The triggers keep their 1 each, as in
     plain GripV.
4. **Re-pose time.** The 58 s in the v1 report is for an unchanged pose. A real re-pose with `refit.py --triggers`
   takes **about 4-4.5 min** (4.3 min measured by the review).
5. **refit.py** reads the trigger solver's residuals before solving the shape. It stops, without changing the
   file, when any of these limits is exceeded:
   - press-face centre more than 0.1 mm from the target;
   - normal more than 1 deg off;
   - travel more than **26 deg** from the press direction;
   - any hand overlap with a trigger part above 0.55 mm.
   The travel limit is 26 rather than 25 because the accepted v_27 triggers sit at 25.4 / 25.6 deg: the solver
   allows max(25, its starting angle). With 25 the delivered pose would always stop. All four limits are in
   `TRIGGER_GATE`. The shape solve is one function, `solve_shape_dense(ctx)`; another solver drops in through
   `"solver": "/path/file.py:function"` in `fit_opts.json`.
6. **File.** Collection 'hand fit' is now in scene GripV, the scene the file opens in, with 'shape offset' hidden.
   It is no longer in GripX. The empty scene 'hf idle', which nothing referenced, is removed. The text block
   'README hand fit' now holds GUIDE.md.
