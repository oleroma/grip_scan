# Rig handles and mock-up print

After the scan and the grip solve, the user corrects the grip by hand and prints a mock-up to try it on.

## Handles (armature `grip rig`, scene `GripV`)

Drag a ball with **G**, or use the sliders in the 3D view sidebar (**N**) > **Grip rig** tab (your own **Grip** tab with the case settings is untouched).

| handle | moves | slider names |
|---|---|---|
| `palm swell` (red ball + arrow) | the whole palm swell; every grip handle hangs off it | out / in, along A, along B |
| `fin middle handle`, `fin ring handle`, `fin pinky handle` | the lobe that finger curls round (the fin bone aims at its ball) | out / in, sideways |
| `stand handle` | the low end of the handle (it stands on this and the pinky fins) | out / in, sideways |
| `R1 hand` (double arrow + ring, on the centre of the R1 tab's outer face) | **index offset** slider (0 = fingertip on R1, 1 = R1 at **deepest contact**, default 5 mm before the middle joint = 38.8 mm from the tip; change it in the panel): the whole hand slides along the finger's own curve, like pulling it through a ring at R1. Turn = the hand about the X axis through R1 (+-15 deg). The palm swell follows only the palm's position (`palm follow` 0..1), so the blue `R1 palm link` stretches | index offset, deepest contact, turn about R1, palm follow |
| `thenar surface` (square) | the surface under the thumb base | out / in, along A, along B |
| `R2 button top`, `R4 button top` (square + arrow) | centre of the trigger's press face; the arrow is the face direction. G moves it (the arch is refitted to the switch), R rotates it | across the face, out toward the finger, up toward the face buttons, arch radius, tilt (= tip angle offset), sideways tilt (= tip shear) |

* The fin/stand bones are aimed at their handle with a Damped Track constraint, so dragging the ball swings
  the fin; the ball's move along the fin is locked (it would do nothing).
* `R2/R4 button top` drive the inputs `button top`, `arch radius`, `tip angle offset` and `tip shear` of the
  node `button top control` in `GripV R2/R4 parameters` (drivers with simple expressions, no Python needed).
  The bone's rest frame is the world turned by the trigger's `button rotation`; euler mode ZYX, X = tip angle
  offset (absolute), Z = shear angle psi with tan(psi) = tip shear * cos(a - tip angle offset). The arrow
  matches the built press face within 0.5 deg; at the defaults all triggers and cutters are bit-exact. They stay within 10 mm of the original top
  (Limit Location). L2/L4 mirror.
* Bone collections: `grip shape (drag these)`, `buttons (drag these)`, `mechanics (do not touch)` (hidden).
* Panel buttons: **Back to scan result** (the pose stored in `grip rig["scan_pose"]`), **Back to
  original GripV**, per-handle reset, select.
* Skin under the triggers follows them: hidden deform bones `R2 seat` / `R4 seat` copy the button handle's
  move; cage points within 6 mm of the trigger follow fully, fading to 0 at 20 mm (weight taken only from
  `grip root`, so the scan pose is unchanged). Moving a top 3 mm moves the skin up to 2.2 mm (R2) / 3 mm (R4):
  the subdivision and the exception zone hold some of it back.
* Press angle: the panel shows the angle between each press face and the direction the face travels when
  pressed (arch tangent); defaults R2 21 deg, R4 17 deg; above `grip rig["max press angle"]` (30) it says
  TOO STEEP.
* Everything protected by `grip rig pins` stays bit-exact: exception zone, flat top (Z 59.04), battery.

The panel is the text block `grip_controls_ui.py`. It registers when the file is opened with
"Allow Execution" (or "Trust"). Without that, open it in the Text Editor and press Run Script. The handles
and drivers work without it.

The file opens with the heavy case parts (`case`, `case middle plate`) excluded from the view layer;
only the basemesh and the buttons are in the scene. Tick them back on in the outliner only to check the case.

The slide is precomputed: `add_rig_handles.py` fits a smooth curve through the index joints, and for every
0.5 mm of contact keys the rigid move (curve point onto the scan contact, curve direction onto the old one)
into the action `R1 finger slide`; an Action constraint on `hand mover` plays it from the handle's X. The
hand objects ride on `hand mover`; `palm follow` copies the position of `palm target` (a point carried by the
hand) and is the parent of `palm swell`.

## Same file as Grip.blend, plus the rig

The output is Grip.blend with additions only: every scene, node group (apart from the trigger controls in
`GripV R2/R4 parameters`), export preset and setting is unchanged. `GripV_preset.json` and `GripX_preset.json`
written on save are byte-identical to the ones in the Grip repo. Everything added sits in one top-level
collection `grip rig` in scene `GripV` (with `grip rig widgets`, `hand reference (hold_upright)` and
`mock-up print` inside it). The variant is set to GripV and the file opens on the GripV scene.

`add_rig_handles.py` builds with Olex's add-ons on (combi_export, blender_tracking, grip_controls,
orca_3mf, from the Grip repo's `extensions/` folder next to grip_scan, or `GRIP_EXTENSIONS`), so their save
handlers stamp the file as in his Blender. Their side files (`*_preset.json`, `tracking.json`) land next to
the output in `build/`. Background Blender cannot store the active workspace tab, so the panel script
switches to the GripV workspace once when the file opens with scripts allowed.

## Mock-up print (collection `mock-up print`, hidden by default)

* `mock-up whole` = posed basemesh (`basemesh variant`) + ABXY, D-pad, Select and Home caps, R2/R4/L2/L4,
  R1/L1, both sticks and the scroll wheel, merged with one Mesh Boolean (Manifold solver, about 0.2 s).
  Shafts are not included.
* `mock-up top` / `mock-up bottom` cut it with the case's own node group `cut basemesh`, the same Z = 40
  split as `case_top` / `case_bottom`.
* It follows every handle live. Hide it while posing; show it to check; **Export mock-up STLs** in the
  panel writes `mockup_top.stl` and `mockup_bottom.stl`.
* Both halves are solid and watertight. Print with low infill; glue or tape the halves together.

## Build

```
pip install bpy==5.2.2 numpy pillow
python trigger_controls/build_trigger_controls.py grip_rig_posed_smooth.blend build/combined.blend
python rig_handles/add_rig_handles.py build/combined.blend build/grip_rig_handles.blend --stl-dir build/stl
python rig_handles/render_handles.py build/grip_rig_handles.blend build/handles_sheet.jpg
```

`add_rig_handles.py` keeps the input's pose as the scan result (moving it onto the handles changes the
grip by < 0.005 mm) and prints a report (mock-up volume, non-manifold edges).
The input .blend holds the scanned hand: keep it and the output out of this public repo.
