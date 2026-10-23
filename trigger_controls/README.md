# Trigger controls: R2/R4 driven by the button top

The GripV R2 and R4 arched triggers can now be placed by **where the button top is** instead of by
rotation / shift z / length. The arch is fitted between the SMD switch and the button top.

## What you edit

In `GripV R2 parameters` and `GripV R4 parameters` there is a new node **button top control**
(it feeds `trigger parameters.001`):

| input | meaning |
|---|---|
| `button top` (x, y, z) | centre of the press face, on its outer (top-bevelled) surface, mm, right-hand trigger |
| `arch radius` | radius of the arch (the old `trigger radius`) |
| `SMD button` | the switch it presses (`R2_pcb_button` / `R4_pcb_button`) |
| top bevel, height, button radius offset, button pressed height, x offset, tip angle offset, tip shear | the offsets, same meaning as before, now set here |
| panel `original (do not edit)` | the values the trigger had before; keeps the defaults bit-exact |

The node works out `button rotation`, `shift z` and `trigger length`; every other trigger input
(width, bevels, stopper, tolerances, marks) stays on `trigger parameters.001` as before. The left
triggers (L2/L4) mirror as before. R1 is not touched.

Moving the top in x/y turns the trigger about the switch and changes its length; moving it in z or
changing the arch radius changes the arch height. The switch contact stays where it was
(surface under the switch moves < 0.05 mm in all tests).

## Defaults

| | button top (x, y, z) mm | arch radius |
|---|---|---|
| R2 | 37.341, -23.894, 20.325 | 30 |
| R4 | 38.227, -37.675, 7.898 | 80 |

## Checks (Blender 5.2)

* Defaults: R2, R4, L2, L4, R2 bool, R4 bool are identical to the original Grip.blend, 0.000 mm,
  same vertex count.
* 26 moved cases (top +-3 mm per axis, radius +-8 mm, random mixes): the built top face passes
  through the requested point every time (0.001 mm = the existing 0.001 overlap).

## Files

* `build_trigger_controls.py IN.blend OUT.blend [--lab]` adds the node groups `trigger top solve`
  and `trigger top control` and wires them in. `--lab` keeps only the trigger parts (small work file).
* `trigger_model.py` the same math in numpy (forward: parameters -> top; inverse: top -> parameters),
  for solvers that move the buttons without Blender.

```
pip install bpy==5.2.2 numpy
python trigger_controls/build_trigger_controls.py Grip.blend build/Grip_trigger_controls.blend
```
