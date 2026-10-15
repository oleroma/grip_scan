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

## Hand capture (step 1: flat photo)

Setup, once (Python 3.12 via uv):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e .
```

The MediaPipe hand model goes in `models/hand_landmarker.task`
(from the MediaPipe model page, `hand_landmarker/float16/latest`).

1. `python -m gripscan.marker_sheet --out out/marker_sheet` prints an A4 sheet
   (PDF/PNG). Print at 100% and check the 100 mm bar with a ruler.
2. Calibrate the phone once: take 10-20 photos of the flat sheet from different
   angles, at 1x zoom, then
   `python -m gripscan.calibrate "photos/calib/*.jpg" --out data/camera.json`.
3. Lay the hand palm down on the sheet, fingers straight and slightly spread,
   and shoot from straight above at 40-50 cm. Then
   `python -m gripscan.measure_flat photos/flat.jpg --camera data/camera.json --out data/hand_flat.json --annotated out/flat_check.jpg`.

Test the geometry chain without photos: `python -m tests.test_geometry`.
