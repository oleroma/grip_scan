# hand_scan_rig: shared spec (read fully before coding)

Goal (Olex): replace the cylinder hand with Olex's scanned hand. Map the existing rig
into the scan mesh, simulate the finger fold with tunable deform/weight maps, compare
the folded hand against the photogrammetry data (board videos), tune for best match.
The real scan does not exist yet (Olex will film a palm-up open hand, OnePlus 15
4K60 video, COLMAP photogrammetry). Everything is built and tested now on a STAND-IN
hand made from Olex's measurements, so the real scan is a drop-in later.

## Environment
- Python venv with Blender-as-module: `/tmp/claude-0/hs/venv/bin/python` (bpy 5.2.2,
  numpy, scipy, opencv-python-headless, scikit-image, trimesh, cma, mediapipe).
  Never name a script after a stdlib module (inspect.py, json.py ... crash bpy).
- Code goes in `/home/claude/grip_scan/hand_scan_rig/` (git repo, branch
  claude/project-thread-i3yp76; do NOT commit/push, the lead does).
- Outputs (meshes, .blend, json, images) go in `/mnt/project-files/hand_scan_rig/`
  (network mount shared with other sessions; write into your own subfolder,
  e.g. standin/, fit/, skin/, compare/, demo/). Scratch in `/tmp/claude-0/hs/work/<module>/`.
- The container can restart; save progress to files as you go, keep runs < 10 min each.
- Do not call any mcp__hearthbot__ tools. Do not install other packages without need;
  if you must, only well-known permissive-licence ones via the venv's pip.
- Licences: commercial-friendly only (no MANO, no SMPL-family, no NC models/data).
- Existing capture code for reference (formats, angle conventions):
  `/tmp/claude-0/hs/pg/gripscan/` (relaxed_pose.py, handmodel.py, landmarks.py,
  triangulate.py, board.py) and `/tmp/claude-0/hs/pg/blender/make_rig.py`,
  `read_rig.py`, `data/camera.json`, `data/board_screen.json`.

## Units, frames, names
- 1 Blender unit = 1 mm. Right hand.
- Landmark names = MediaPipe 21 (gripscan/landmarks.py NAMES): wrist, thumb_cmc,
  thumb_mcp, thumb_ip, thumb_tip, index_mcp, index_pip, index_dip, index_tip, ...
  pinky_tip. In OUR data these are JOINT CENTRES (inside the finger), tips are the
  fingertip end of the distal bone (~ 2-3 mm inside the skin tip).
- Bone names/hierarchy = existing hand rig (`/tmp/claude-0/hs/hold_rig.blend`,
  object 'hand rig'): palm (wrist -> centre of knuckle row, a handle only);
  {index,middle,ring,pinky}_metacarpal (wrist -> MCP), _proximal (MCP->PIP),
  _middle (PIP->DIP), _distal (DIP->tip); thumb_metacarpal (CMC->MCP),
  thumb_proximal (MCP->IP), thumb_distal (IP->tip). Bone local X = flexion axis,
  positive rotation about X curls toward the palm (verify against hold_rig.blend and
  state the convention in your code).
- Angles: as relaxed_pose.py docstring: flexion 0 = straight, + toward palm;
  MCP measured vs metacarpal; spread between neighbouring proximal segments.
- Olex's measured lengths (mm, flat photos): index 42.7/24.4/19.8, middle
  46.2/26.5/20.4, ring 42.7/26.5/20.3, pinky 32.8/19.7/17.7, thumb 36.5(meta)/33.1/23.9;
  palm 63 between index and pinky MCP, 88 wrist->middle MCP. The rig's bone lengths
  (hold_rig.blend) are the authoritative values used so far.
- The hold pose = rest pose of 'hand rig' in hold_rig.blend (posed by Olex on the
  grip 'grip (current GripV)'). Finger widths were never measured: typical values.

## File formats (interfaces between modules)
- `joints.json`: {"side":"right","units":"mm","pose":"open"|..., "landmarks":
  {name:[x,y,z]} (21), "palm_normal":[x,y,z] (unit, pointing out of the palm),
  "lengths": {finger:{segment:mm}}}
- Rigged hand .blend: objects 'hand scan' (mesh, the skin), 'hand rig' (armature,
  same bone names as above, rest pose = the scan's open pose, mesh parented with an
  Armature modifier named 'Armature'), vertex groups named after bones.
- `skin_params.json`: per-joint deform-map parameters (see skin module), plus global.
- `pose.json`: {"angles": {"index_mcp":deg, "index_pip":..., "spread_index_middle":...,
  "thumb_cmc":..., ...}, "root": {"location":[..], "rotation_quat":[..]}}.
