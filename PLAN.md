# grip_scan: Plan (draft v2)

Goal: measure a hand, then generate the handle of a fully 3D-printed controller
case (Alpakka 1.0 board first, any custom board later). The handle is solved as
an **SDF optimisation**: it fills the hollow of the hand in its *relaxed* pose,
while the board's controls stay reachable.

## Decisions so far
- **First case: a single target**, your custom semi-pistol grip:
  - more vertical and thicker than a gamepad handle, less than a pistol grip
  - the palm swell sits in the middle of the palm, like an Olympic pistol grip
  - classic and other shapes come later, reusing the same solver
- **Capture:** the recommended path:
  - flat photos with a printed marker for metric lengths
  - finger-bend photos for the true joint pivots
  - a video of the relaxed pose, triangulated over the marker board
    (MediaPipe + OpenCV, no MANO)
- **Board:** the top buttons are fixed. The thumb must reach all of them.
- **Printing:** FDM first.
- **Blender:** latest stable. No calipers.
- **Finger mapping:** index → R1 (bumper), middle → R2, ring → R4.
  Pinky has no button and only wraps the grip.
- **Contact:** exact touch at rest (clearance = 0).
- **Licence:** only permissive dependencies (Apache/BSD), so everything is
  usable commercially. **No MANO, HaMeR or WiLoR** in the pipeline (decided
  2026-10-15, "least friction").
- **Compute:** NVIDIA GPU available and fine to use: PyTorch CUDA for the
  solver, falling back to CPU.
- **iPhone (Pro, with LiDAR) as an optional depth source:** record an RGB-D
  sequence (e.g. Record3D export: colour + depth + ARKit camera poses).
  - **ARKit poses** give metric camera tracking, so the marker board becomes a
    scale check rather than a requirement.
  - **Depth** gives the palm-pad and finger-thickness surface, which photos
    can't.
  - **Limits:** the rear LiDAR is coarse (low-res depth, roughly several mm on
    small objects, adjacent fingers blur together), so joint positions still
    come from MediaPipe.
  - **Higher-detail option:** the front TrueDepth camera gives finer depth at
    close range. It is worth testing for the palm surface.
- **Reference grip:** your current hand-made grip/case will be added as a repo.
  It is the source of button positions and the R2/R4 arch geometry, and the
  baseline the solver is compared against.

```
capture ──► hand.json (lengths, radii, pivots, RELAXED pose θ0)
                     │
board.json ──────────┤   (fixed top buttons, bumpers, R2/R4 arcs, body volume)
                     ▼
        SDF optimisation (hand placement + small pose deltas + grip field)
                     ▼
        grip SDF ∪ case body  →  mesh (marching cubes)  →  Blender  →  STL/3MF
```

---

## 1. The relaxed pose (what the grip is shaped from)

Your definition:
1. Stand with your arms hanging and relaxed. The pinky is the most curled, and
   the curl decreases slightly toward the index.
2. Raise your forearms by bending the elbows.
3. The palm keeps almost the same shape, and the curl gradient toward the pinky
   gets slightly steeper.

That pose, **θ0**, is *measured*, not assumed:
- **Capture:** hold the elbow-bent relaxed pose about 10–20 cm above the
  printed ChArUco board, and film it while moving the phone in an arc around
  the hand: palm side, both sides, from below. This takes about 10 s and is
  repeated 3 times.
- **Fit (multi-view triangulation, no learned hand model):**
  1. The board gives the camera pose of each frame (OpenCV ChArUco).
  2. MediaPipe gives the 21 landmarks per frame.
  3. Robust triangulation (RANSAC over frames) gives metric 3D landmarks.
  4. Our capsule skeleton (lengths from the flat photos, pivots from the bend
     photos) is fitted to the 3D landmarks → per-joint flexion angles.

  Hidden joints (curled pinky) are bridged by constraining each finger's
  joints to one flexion plane.
- **Stored:** in `hand.json` as `relaxed_pose`, with a per-joint spread over
  the repeats. That spread becomes the "how much can this joint move from
  relaxed" tolerance in the solver.

## 2. Hand as an SDF

- **Fingers:** chains of capsules (segment length + radius per segment), posed
  by joint angles about the measured pivots.
- **Palm:** a smooth union of a rounded slab plus two ellipsoids for the thenar
  pad (thumb base) and hypothenar pad (pinky side). The palm *hollow* between
  them is an explicit named region, used as the palm-swell target.
- **H(x; T, θ):** the signed distance to the hand surface, where
  T = hand placement (6 DOF) and θ = joint angles.

  All of these are analytic and differentiable (smooth-min for the unions).

## 3. The optimisation

**Variables:**
- **T:** hand placement relative to the board (6 DOF)
- **Δθ:** small deviations from the relaxed pose θ0 (about 15 finger and palm
  angles; the thumb is handled separately)
- **p:** grip shape parameters (see 3.1)
- **s₂, s₄:** positions of the R2/R4 buttons along their allowed arc ranges

**Grip field (the result):**
```
Grip(x) = smooth_union( Core(x; p), Hollow(x; T, θ) ) ∩ Envelope(x)  −  dilate(H, clearance)
```
- **Hollow:** the morphological *closing* of the hand volume (dilate by R, then
  erode by R), minus the hand itself. This is exactly "the space the fingers wrap around".
- **Core:** a guaranteed minimum cross-section along the handle axis (a
  super-ellipse sweep, minimum thickness `t_min`). This is your rule that even
  a fully curled pinky must still wrap around something. Where the hollow is
  thinner than the core, the pinky is pushed *open*, and that costs pose deviation.
- **Envelope:** the space the case is allowed to occupy. It joins the body
  around the board.
- **clearance:** the contact offset (0 = exact touch, negative = slight squeeze).

**Cost (minimised):**
| Term | Meaning |
|---|---|
| Σ w_j (Δθ_j / σ_j)² | stay at the relaxed pose; σ_j is the measured spread |
| contact | palmar surface samples of the fingers + palm sit on Grip = 0 (no gaps) |
| penetration | no hand sample inside the grip (hard penalty) |
| palm swell | the grip surface reaches into the palm-hollow region, centred between thenar and hypothenar |
| smoothness | penalise curvature spikes and undercuts (FDM printability, no pressure ridges) |
| volume / size | mild preference for a compact handle |

**Constraints:**
| Constraint | Formulation |
|---|---|
| **Top buttons (fixed)** | For *each* top button b, a thumb IK solve reaches b with joint angles in a comfortable range, while the rest of the hand stays gripping. Cost = thumb effort, hard limit = range of motion. |
| **Bumpers (R1/L1)** | Some point on the chosen finger chain lies within d of the bumper surface. It doesn't have to be the fingertip, so a slight wrap is fine. |
| **R2 / R4 (arched buttons)** | Fingertip of the chosen finger, projected onto the grip, lies on the button's allowed arc (s ∈ [s_min, s_max], plus an in/out and up/down range). The button position is an *output*. |
| **Board** | Grip ∩ PCB keep-out = ∅; the case body contains the board with its wall thickness. |
| **Minimum thickness** | Core ≥ t_min everywhere along the handle. |

**Solver:**
1. **Coarse:** CMA-ES (gradient-free) over T, Δθ and p, evaluated on a
   2 mm grid. This is robust to the non-smooth closing step and runs in seconds
   to minutes on a CPU.
2. **Fine:** gradient refinement with PyTorch autodiff on the analytic SDF
   terms, at a 0.5 mm grid.
3. **Output:** grip SDF → marching cubes → mesh, plus a report: per-joint pose
   deviation, thumb effort per button, min thickness, R2/R4 positions.

### 3.1 Grip shape parametrisation (p)

The grip is mostly *derived* (hollow ∪ core), so p stays small:
- handle axis curve (rake angle, length)
- core super-ellipse width/depth at about 5 stations
- palm-swell bulge amplitude/height
- smooth-union radius

Low-dimensional means the optimisation is stable, and the result can be
explained and tweaked by hand in Blender.

## 4. Inputs needed

- **`hand.json`:** lengths, radii, pivots (finger-bend photos), relaxed pose θ0
  + spread, palm pad geometry.
- **`board.json`:** the Alpakka 1.0 data:
  - PCB outline and keep-outs, mounting holes
  - top button and stick positions (fixed)
  - bumper positions
  - R2/R4 arch geometry (arc centre, radius, allowed range)
  - which finger operates each control (index R1, middle R2, ring R4)
- **Your current hand-made grip mesh:** the baseline the solver output is
  compared against (validation).

## 4a. Integration with the Grip repo (github.com/oleroma/Grip, `D:\Grip`)

From reading `docs/blender/develop/03-case-pipeline.md` and the part docs:
- **Only one thing in Grip.blend is hand-modelled: the outer shape.**
  - It's a basemesh: `basemesh GripV`, a half model.
  - It's mirrored at X = 0 by `MirrorX`, subdivided, and split into top/bottom
    at the fixed Z = 40 mm.
  - Openings, shells, ribs, plate, triggers and bumpers are all generated from
    presets and cut from it.
- **The solver's output is therefore a new `basemesh GripV` half**, not a whole
  case.
  - These are export rules for Grip.blend compatibility, not solver terms:
    - **Z = 40:** the handle may cross Z = 40 freely, because Grip splits there
      and shapes the plate from that outline. The one added check is a minimum
      wall thickness near Z = 40 for the ledge/plate.
    - **Half model:** one (right) hand is solved and Grip's `MirrorX` builds
      the left. Asymmetric hands would mean disabling `MirrorX` and solving
      both.
    - **Top face:** untouched (fixed envelope).
  - Everything downstream (hollow/lattice/indent bodies, middle plate,
    openings) keeps working unchanged.
  - The solver only reshapes the handle region. The top/body region is a fixed
    "envelope" taken from the current basemesh.
- **R2/R4 positions are output as trigger preset values**, not geometry.
  - Grip places each arch trigger relative to `R2_pcb_button` / `R4_pcb_button`
    with `shift z` (along the arc), `button radius offset`, `x offset` and
    `button rotation`.
  - The solver searches exactly those parameters (with `trigger radius`) to put
    the middle fingertip on R2 and the ring fingertip on R4.
- **Bumper:** the R1 lever (`R1L1 lever parameters`, flex design on GripV) gives
  the target surface for the index finger wrap.
- **Board data (`board.json`): minimal, built around the fixed top face**
  (decided 2026-10-15):
  - **Fixed top face:** taken from the current `basemesh GripV`. It is the
    single flat n-gon at Z = 59 mm, spanning x 0–73.2 and y −69.1 to −10.3 on
    the half mesh. It's a hard constraint: the solver never moves it.
  - **Button area:** the stick, ABXY, D-pad and select/home top positions on
    that face are the thumb targets.
  - **R1:** one point plus a normal on the bumper surface (index target).
  - **R2/R4:** an allowed region per fingertip press point only. The solver
    outputs the desired press points. A Blender-side step fits the trigger
    settings (`shift z`, `button radius offset`, `x offset`,
    `button rotation`) so the press centres land there. Pressed-vs-released
    evaluation is already proven on Grip.blend (variant must be set to GripV).
  - **Keep-out:** a rough PCB box plus the fixed body region.
  - All of this is roughly 20 numbers, exported by a ~30-line `bpy` script (or
    hand-written).
- **Fit:** this replaces the roadmap's "Next: ergonomics for your hand"
  (the "My hand" layer) with a measured-hand version. The results could be
  written into the same presets the Grip tab already exposes.
- **Licence:** Grip is CC BY-NC-SA, and its add-ons are GPL. grip_scan only
  reads the .blend and writes a basemesh and preset values, so it can carry
  its own licence.

## 5. Tech stack

- **`gripsolve` (standalone Python):**
  - numpy, scipy, scikit-image (marching cubes), PyTorch (autodiff), `cma`
  - OpenCV + MediaPipe for capture
  - relaxed pose by ChArUco multi-view triangulation (no MANO; all
    dependencies Apache/BSD)
- **Blender add-on:**
  - imports hand/board/result
  - shows the posed hand, grip and board
  - sliders for p and weights → re-runs the solver as a subprocess
  - unions the grip with the case body
  - splits into printable halves, exports STL/3MF
- The solver runs outside Blender's Python, so we aren't limited to its
  bundled packages.

## 6. Milestones

1. **Capture → `hand.json`:**
   - marker sheet
   - flat photos → lengths
   - bend photos → pivots
   - relaxed video → θ0
2. **Minimal `board.json` from `Grip.blend`:** fixed top face, button tops,
   R1 point, R2/R4 regions, keep-out. Also the current basemesh as the
   reference shape.
3. **Hand SDF + visual check in Blender:** the relaxed hand placed on your existing
   grip, to show how well the current manual shape already matches.
4. **Optimiser v1:** placement + pose + core + hollow, with the thumb/top-button
   constraint only.
5. **Add constraints:** bumpers, R2/R4 arcs, palm swell, printability → first
   FDM test handle.
6. Compare with your manual grip, tune weights, iterate.
