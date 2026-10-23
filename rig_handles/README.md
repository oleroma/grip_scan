# Rig handles and mock-up print

After the scan and the grip solve, the user corrects the grip by hand and prints a mock-up to try it on.

## Handles (armature `grip rig`, scene `GripV`)

Drag a ball with **G**, or use the sliders in the 3D view sidebar (**N**) > **Grip** tab.

| handle | moves | slider names |
|---|---|---|
| `palm swell` (red ball + arrow) | the whole palm swell; every grip handle hangs off it | out / in, along A, along B |
| `fin middle handle`, `fin ring handle`, `fin pinky handle` | the lobe that finger curls round (the fin bone aims at its ball) | out / in, sideways |
| `stand handle` | the low end of the handle (it stands on this and the pinky fins) | out / in, sideways |
| `thenar surface` (square) | the surface under the thumb base | out / in, along A, along B |
| `R2 button top`, `R4 button top` (crosshair) | centre of the trigger's press face; the arch is refitted to the switch | X sideways, Y toward bumpers, Z toward face, arch radius |

* The fin/stand bones are aimed at their handle with a Damped Track constraint, so dragging the ball swings
  the fin; the ball's move along the fin is locked (it would do nothing).
* `R2/R4 button top` drive the inputs `button top` and `arch radius` of the node `button top control` in
  `GripV R2/R4 parameters` (drivers, no Python needed). They stay within 10 mm of the original top
  (Limit Location). L2/L4 mirror.
* Bone collections: `grip shape (drag these)`, `buttons (drag these)`, `mechanics (do not touch)` (hidden).
* Panel buttons: **Back to scan result** (the pose stored in `grip rig["scan_pose"]`), **Back to
  original GripV**, per-handle reset, select.
* Everything protected by `grip rig pins` stays bit-exact: exception zone, flat top (Z 59.04), battery,
  skin around the triggers. The skin next to R2/R4 does not follow a moved trigger.

The panel is the text block `grip_controls_ui.py`. It registers when the file is opened with
"Allow Execution" (or "Trust"). Without that, open it in the Text Editor and press Run Script. The handles
and drivers work without it.

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
