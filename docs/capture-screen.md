# Capture with the monitor as the marker board

For the ASUS VG27AQL5A (27", 2560×1440). It takes about 20 minutes. You need a
ruler (30 cm or longer, metal if you have one) and your phone. All commands run
in `D:\grip_scan`.

## 1. Prepare

1. Wipe the screen clean.
2. Set the monitor brightness to about 50%. Dim the room, but put a lamp to the
   side so your hand is lit. Avoid lamp reflections on the screen.
3. Set up the phone camera, and keep these settings for every photo:
   - **JPEG, not HEIC.** On iPhone: Settings › Camera › Formats › **Most Compatible**.
   - The **1x** main camera. Not 0.5x, not 2x, not Portrait mode.
   - Photo mode, highest resolution. Flash off.

## 2. Show the board and check its scale

```bash
.venv\Scripts\python -m gripscan.screen_board
```

The board fills the screen. If a warning about the window size appears in the
terminal, tell me: it means Windows scaled the image.

Put the ruler on the black bar under the board. Measure it end to end to the
nearest 0.5 mm. It should be about 298–299 mm. Press **Esc**, then rerun with
your value:

```bash
.venv\Scripts\python -m gripscan.screen_board --bar-mm 298.5
```

This sets the board size from your own measurement and writes
`data\board_screen.json`. Leave the board on screen for the photos.

## 3. Calibration photos (15–20)

Photograph the screen only. The whole board should be in the picture, sharp,
with no glare spots. Vary the shots:
- straight on, from 35, 50 and 70 cm
- tilted 20–40° from the left, the right, above and below
- the board near each corner and edge of the frame (part of it can be cut off
  in a few shots)

Copy them to `D:\grip_scan\photos\calib\`.

## 4. Hand photos (3)

1. Tilt the monitor back as far as its stand allows, or lay it on its back on a
   towel.
2. Place your **right hand** palm against the screen, lightly, with no
   pressing. Put it on the lower half so plenty of the board stays visible
   around it.
   - Fingers straight and slightly spread.
   - The thumb flat on the screen and out to the side.
   - Wrist in line with the forearm.
3. Hold the phone in your left hand, square to the screen, about 40–50 cm away.
   Tap on your hand so focus and exposure are set on it, then take the photo.
4. Take 3 photos. Lift your hand and place it again between shots.

Copy them to `D:\grip_scan\photos\flat\`.

## 5. Tell me

I'll run calibration and the measurement, and send back the lengths with marked-up
photos. To run them yourself:

```bash
.venv\Scripts\python -m gripscan.calibrate "photos\calib\*.jpg" --board-spec data\board_screen.json --out data\camera.json
.venv\Scripts\python -m gripscan.measure_flat photos\flat\IMG_0001.jpg --camera data\camera.json --board-spec data\board_screen.json --out data\hand_flat.json --annotated out\flat_check.jpg
```
