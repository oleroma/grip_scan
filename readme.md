# grip_scan

Measure a hand and solve a custom controller handle for it, as an SDF optimisation.
The first target is the GripV case in [oleroma/Grip](https://github.com/oleroma/Grip).
See [PLAN.md](PLAN.md).

## Board data from Grip.blend

Blender 5.2+, run headless. The .blend is never saved.

```bash
blender -b D:/Grip/blender/Grip.blend --python blender/export_board.py -- --out data/board_gripv.json --ref-mesh data/reference_basemesh_gripv.obj
blender -b D:/Grip/blender/Grip.blend --python blender/preview_board.py -- --board data/board_gripv.json --out data/board_preview
```

`board_gripv.json` holds the fixed top face, the thumb targets (button and stick
tops), the R1 target, the R2/R4 press centres and normals, and the PCB keep-out.
The preview renders put those points on the basemesh so you can check them:
blue = thumb, yellow = R1, red = R2, green = R4.
