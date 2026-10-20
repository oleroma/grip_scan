GRIP CASE RIG
=============

What is in this file
- "hand rig": your measured hand. Its rest pose is your hand-posed hold;
  the pose you see is what the solver computed.
- "basemesh GripV": the GripV case cage from Grip.blend with an extra
  Armature modifier at the top of the stack ("case rig").
- "case rig": 9 handles that reshape the case (grab them in Pose Mode):
    bump              palm bump; moves with your palm pocket (G to move)
    middle_proximal   wall under the middle finger, proximal phalanx
    middle_middle     wall under the middle finger, middle phalanx
    ring_proximal     wall under the ring finger, proximal phalanx
    ring_middle       wall under the ring finger, middle phalanx
    pinky_proximal    wall under the pinky, proximal phalanx
    pinky_middle      wall under the pinky, middle phalanx
    pinky_distal      wall under the pinky, fingertip
    top_back          back edge of the top face; slides forward towards
                      the sticks so your thenar rests on a smooth slope
  The wall handles and top_back only slide along their local Y.
  How the walls are fitted: the middle phalanges grab the front of the
  handle next to their PIP joint, and the handle is pressed into the
  pads at the finger roots (palm over the knuckles). Those zones touch
  the case; everywhere else the hand may only clear it.
  The palm is cupped: the ring and pinky knuckles sit deeper towards the
  palm (palm_cup_deg), so the knuckle arch wraps around the palm swell.
  The pinky has no button. It is turned to curl right beside the ring
  finger (same plane, so it wraps the handle instead of passing under
  it) and keeps your posed pinky curl, changed by as much as the ring
  finger's curl changed. Its walls move until all three phalanges touch.
  Everything above Z 38 (top part, index wrap) and the R2/R4 button
  areas never move, except the back edge of the top face (top_back).
- "stick cutouts" collection (hidden): the stick and home button cutouts
  baked from Grip.blend. top_back stops when the case gets within
  stick_clearance_mm of any of them. Re-bake them if the sticks move.
- "fitted variants" collection (hidden): ready cages for 4 R1 overlaps.
  Any of them can replace "basemesh GripV" in Grip.blend.

Re-solving with other settings
1. Select "case rig" > Object properties > Custom Properties.
2. Change a value, e.g. r1_overlap_mm.
3. Text Editor > open "grip_case_rig.py" > Run Script (Alt+P).
   It re-places the hand, re-curls the fingers and re-fits the walls
   in a few seconds. Results go to the text block "case_rig_report".

Settings
  r1_overlap_mm    where the R1 centre sits on the index, measured from
                   the fingertip end: ~8 fingertip, ~15 distal pad,
                   ~27 DIP crease (your current hold), ~38 middle phalanx
  dip_release_deg  how bent the middle/ring distal joints are on R2/R4
  tip_pad_mm       which point of the fingertip pad lands on the button
  finger_press_mm  how far proximal/middle phalanges press into the case
  knuckle_z        index knuckle height (your hold: 58.4)
  fixed_above_z / free_below_z   frozen zone of the case
  wall_sigma_mm / bump_sigma_mm  how wide each handle's influence is
  button_keep_mm   radius kept fixed around R2/R4
  pinky_extra_curl_deg  pinky curls this much more (+) or less (-) than
                   your posed pinky curl; the case follows it
  min_thickness_mm walls may not make the handle thinner than this
  root_pad_frac    part of the proximal phalanx (from the knuckle) that
                   touches, together with the palm over the knuckle
  pip_pad_frac     part of the middle phalanx (from the PIP joint) that
                   touches the front of the handle
  palm_cup_deg     how far the pinky metacarpal flexes towards the palm
                   (ring finger half as much); 0 = your scanned arch
  top_back_max_mm  how far the top face back edge may move forward
                   (0 = leave it where it is)
  stick_clearance_mm  closest the case may come to the stick cutouts
