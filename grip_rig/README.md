# Grip rig: pose the grip, let the hand fill in

The handle of GripV is rigged with a few bones and posed by a solver so that the (soft) hand in the
approved hold rests on a smooth handle with a light press. Nothing outside the handle changes.

## Bones (armature `grip rig`, in `basemesh GripV`'s modifier stack)

| bone | from -> to | what it does |
|---|---|---|
| `grip root` | fixed | holds everything that must not move |
| `palm swell` | the palm swell point, pointing out of the surface | moves the whole palm swell; every other bone hangs off it |
| `fin middle` / `fin ring` / `fin pinky` | palm swell -> the lobe each finger curls round | swings that lobe out/in and sideways |
| `stand` | palm swell -> the low end of the handle | the end the controller stands on (with the pinky fins) |
| `thenar surface` | under the thumb base | moves the surface the thumb-base pad rests on |
| `top back edge` | back edge of the flat top | may only slide in Y; locked in the solver (it would touch the exception zone) |

Frozen: every cage point inside Olex's `exception_zone.stl` (mirrored), every cage point whose
subdivision reaches into it, the R2/R4 trigger surroundings, the battery box, the mirror seam (X), and
the flat top (Z). The modifier `grip rig pins` puts frozen points back exactly, so the exception zone
and the top are bit-exact in any pose.

## Pipeline (bpy 5.2 as a Python module)

```
PY=/path/to/venv/bin/python          # pip install bpy==5.2.2 numpy scipy pillow
$PY build_grip_rig.py                # Grip.blend -> grip_rig.blend (rest pose == original GripV)
$PY extract_hand.py                  # hand in the approved hold -> hand_hold.npz
$PY export_bones.py                  # bone matrices + X subdivision matrix; checks numpy vs Blender
$PY solve_grip.py [--smooth 5]       # solve the bone pose -> solve.json
$PY pose_in_blender.py --solve solve.json   # pose the rig, check every case part, save posed .blend
$PY render_sheet.py                  # picture: original | posed | what changed
$PY check_demo.py                    # optional: each bone moved alone
```

## What the solver wants

* contact: each contact part of the hand (palm pads over the metacarpals, inner side of the middle,
  ring and pinky fingers, thumb base) pressed in by half its soft-tissue give at its closest spot
* give: no skin pressed deeper than palm 3.5 mm / finger 1.5 mm
* stand: still stands on the two pinky fins and the two handle ends (nothing else touches the table,
  centre of mass at least 8 mm inside the supports)
* calm + smooth: small bone moves, no bends in the change (umbrella Laplacian of the displacement)

Hand data and .blend files with the hand stay out of this public repo (project files only).
