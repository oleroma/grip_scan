# web_hand_rig

Rigged MakeHuman (CC0) hand rescaled to measured joints and finger sections, and a hold-pose solver on GripV. Code only: the measured hand data and the .blend outputs stay out of this public repo. Paths at the top of each script point to the cloud work folder; change them to run elsewhere.

## Source and licence
MakeHuman base mesh `hm08`, default rig and its skin weights, from github.com/makehumancommunity/mpfb2 (`src/mpfb/data`). Assets are CC0 1.0 (LICENSE.ASSETS.md in that repo), so commercial use is fine and no attribution is required. The add-on code is GPL but none of it is used here.

Considered and not used: WebXR generic-hand (MIT, but low-poly, ~95 KB); Blender Studio Human Base Meshes (CC0, but no skin weights); paid Superhive/CGTrader hands (some have corrective shape keys, cost money, licences vary).
No free commercial-licence hand with corrective shape keys was found. The MakeHuman hand has skin weights only; candy-wrapper at the knuckles stays small at grip angles.

## How it was scaled
1. Similarity fit of MakeHuman joints to Olex's fitted joints (hand_scan_rig/fitted/joints.json).
2. Every bone mapped onto Olex's bone (length exactly), blended by the MakeHuman weights.
3. Finger cross-sections scaled per segment to the fitted widths/thicknesses (fitted_params.json sections): all within 1 mm. Palm already 86 mm across the knuckle row and ~28 mm thick (targets 86.1 / 29.4), left as is.
4. Forearm stub made straight with the middle metacarpal, cut ~60 mm behind the wrist and capped.

## Hold pose
Start = the hold_v2 pose transferred bone by bone, then a least-squares solve (`code/hold.py`):
flat palm (metacarpals locked to the open-hand shape, no cupping), palm plane vertical facing the other hand,
index (middle or distal segment) on R1, middle pad on R2, ring pad on R4, palm on the palm swell, pinky pad near the grip, thumb kept where it was, skin may press 0.4 mm into the plastic for free.
R1's marker is 4.6 mm under the face, so the nearest face point is the touch target.

Result: palm 1.9 deg from vertical (old hold: palm arch ~15 deg), R1 0.35 mm, R2 0.49 mm, R4 0.58 mm, palm swell 0.49 mm, deepest skin in plastic 1.25 mm on 7 vertices (palm on the swell and ring middle segment), pinky pad 1.4 mm off the grip.
To roll the hand yourself: rotate bone `lowerarm01.R` (root) in Pose mode about its own Y axis.
