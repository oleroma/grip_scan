# Hand from scan: put your real hand on the grip

This folder turns a 3D scan of your open hand into a hand model that bends like a
real hand, puts it on the controller in your hold pose and draws a comparison
picture against the old cylinder hand. You run it with one command.

Right now there is no real scan yet, so everything was tested on a **stand-in**: a
smooth hand built from your measurements. When your scan exists, it drops in
unchanged.

## What you need

| File | What it is |
|---|---|
| your scan (`.ply`, `.obj` or `.glb`) | the 3D model COLMAP / OpenMVS (or other photogrammetry) makes from the video of your open hand, palm up. Any size units, any orientation. A textured OBJ/GLB straight out of the photogrammetry tool is fine (the texture is ignored). A bit of table or background in it is fine, also when the table surface is noisy or slightly curved. |
| `hold_rig.blend` | the Blender file with your posed cylinder hand ('hand rig') on 'grip (current GripV)'. Its pose is the **hold pose** the new hand copies. |

Nothing else. The scan does not need cleaning, rotating or scaling first.

**When you film the scan:** spread your fingers so there is at least about 5 mm of
air between neighbouring fingers and between the thumb and the index finger. Touching
fingers still work in most cases (the fit then says so in a warning), but it is less
reliable. If the back of the hand rests on the table, the part the camera never saw is
closed with a flat cap, and the fit warns how much of the skin that is.

## How to run it

On your PC, in this folder, with the Python that has Blender installed as a module
(here: `/tmp/claude-0/hs/venv/bin/python`):

```
python run_all.py my_hand_scan.ply --out my_result --hold-rig path/to/hold_rig.blend
```

It takes about 5 minutes. To re-run only part of it: `--steps hold,contacts,sheet`
(add `--force` to redo steps whose files already exist). The demo on the stand-in
is `python run_all.py --demo`.

## What each step does

1. **fit**: cleans the scan (removes the table, fills holes, makes one closed
   surface), works out its size in millimetres, finds the fingertips and the
   finger joints, and puts a skeleton inside the hand (the "rig": one bone per
   finger segment, same names as your cylinder rig).
2. **skin**: decides how strongly each bone pulls each bit of skin ("weight
   maps") and adds the effects that keep bent joints looking right: knuckles stay
   round, the pads bulge a little on the palm side, and creases are smoothed. The
   settings come from `skin_params.json`.
3. **hold**: bends the scan hand's joints to the same angles as your cylinder hand,
   then moves and turns the whole hand so its joints land on the cylinder hand's
   joints as closely as possible. It also measures where skin goes through skin
   in that pose (fingers pressing on each other); see `hold_placement.json`.
4. **contacts**: measures both hands against the grip: how deep skin goes into
   the plastic, the gap at the palm swell, and how far the fingertips are from
   R1, R2 and R4.
5. **sheet**: renders `demo_sheet.jpg`, with the old hand on the top row and the
   new hand below, seen from outside, front and top.

## What you get (in `--out`)

- `hand_on_grip_scan.blend`: open this in Blender. It holds the grip, the new hand
  in the hold pose and the old cylinder hand (hidden; click the eye next to the
  collection **old cylinder hand** to show it).
  To change the pose, select **hand rig**, press Ctrl+Tab (Pose Mode), click a bone
  and rotate it. Rotating about the bone's X axis bends the finger; positive values
  curl it toward the palm. The skin follows straight away, including the pad
  bulges (driven by the bone rotations). Alt+R puts a bone back to the open scan
  pose. One thing Blender does not do live: where two fingers press together, the
  skin of one goes into the other. The Python posing (`skin.py pose`) flattens
  such contacts; in Blender's Pose Mode you see the overlap.
- `demo_sheet.jpg`: the comparison picture.
- `contacts.json`: the contact numbers for both hands.
- `hold_placement.json`: how closely the new hand's joints match the cylinder hand's.
- `fit/`: the cleaned scan, the joints that were found, and `check.png` showing the
  skeleton drawn inside the scan. Look at `check.png` first: if a joint is
  clearly in the wrong place, nothing after it will be right. Also read the
  `warnings` in `fit/joints.json` (and printed at the end of the run): touching
  fingers, a skin that could not be closed cleanly, or a large flat cap are
  reported there.

## Re-placing the hand so the real skin sits on the grip

The **hold** step copies the cylinder hand's joints, so a hand with a different
skin thickness can float off the grip or sink into it. `hold_solve.py` moves and
turns the whole hand and bends its joints (staying close to your hold pose and
inside normal joint limits) until the palm rests on the palm swell, the index
touches R1, the middle and ring fingertips sit on R2 / R4, the pinky pad touches
the grip, the thumb stays where it was and no skin goes more than about half a
millimetre into the plastic:

```
python hold_solve.py solve --params skin_params.json   # about 3 minutes
python hold_solve.py export                            # .blend, pose.json, contacts.json, metrics.json
python hold_solve.py render --solve <solve json> ; python hold_solve.py sheet   # before / after picture
```

`metrics.json` lists before and after side by side (palm gap, deepest skin in the
plastic, fingertip-to-button distances, flipped skin triangles, index-to-thumb
weight leak). The file locations at the top of `hold_solve.py` need the same
change as the other scripts.

## Palm upright, in the web hand's pose

`hold_v2` folded the pinky side of the palm about 19 deg in around the grip (palm arch 15.5 deg), so
the hand looked rolled inward. `hold_upright.py` puts the scanned hand into the pose of the web
(MakeHuman) hand in `web_hand_rig/` instead: flat palm, both halves of the palm upright "like holding
a box", then moves it only as much as the real skin needs to sit on the palm swell, R1, R2 and R4:

```
python hold_upright.py ref ; python hold_upright.py start      # take over the web hand's joints
python hold_upright.py solve --tag a                            # about 2 minutes
python hold_upright.py export --tag a                           # .blend, pose.json, contacts.json, metrics.json
python hold_upright.py render --tag a ; python hold_upright.py sheet
```

`metrics.json` compares hold_v2 and the new pose (palm tilt, how far the pinky side is folded, palm
arch, contacts, flipped triangles) and lists every joint angle against the web hand's.

## Things to know

- The new hand only matches your hand once the real scan is in. Until then it is
  the stand-in, made from your finger lengths, with typical finger widths.
- The bending settings were tuned on the stand-in. After the scan, `compare.py`
  can tune some of them against your board videos (see that file's header). It
  only changes the settings the videos can actually measure (mainly how wide each
  joint's bending zone is) and says how sure it is about each; offsets, tilts and
  most bulges cannot be seen in a silhouette and stay at their defaults.
- For those videos: turn the phone's video stabilisation off (it changes the
  picture crop from frame to frame, which breaks the lens model), and keep the
  hand and the board well lit. `compare.py` cuts the hand out of each frame using
  the scan hand placed at the landmark pose as a guide, and drops frames where the
  cut-out and the hand model disagree; it reports how many it dropped.
- A pose file can give the thumb either the SPEC angles or, better,
  `thumb_cmc_azimuth` (signed angle of the thumb metacarpal from the index
  metacarpal in the palm plane, + toward the thumb side) together with
  `thumb_cmc_elev`. The SPEC `thumb_cmc` alone has two mirror solutions; the
  solver then takes the one on the side of the open scan pose, which is wrong for a
  thumb opposed across the palm (it warns when the angles cannot be met).
- Before the first run on your PC, the file locations written at the top of
  `run_all.py`, `fit_rig.py` and `skin.py` (they point at this test machine,
  `/tmp/claude-0/...` and `/mnt/project-files/...`) need changing to your folders.
  The code also uses the `gripscan` package from the capture project.
