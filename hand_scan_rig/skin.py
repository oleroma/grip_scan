"""Module C: parametric deform (weight) maps, posing and fold simulation for the hand scan.

    python skin.py build                 # test rig(s) from the stand-in + skin_params.json defaults
    python skin.py pose --pose pose.json [--blend B] [--params P] [--backend blender|numpy] --out x.ply
    python skin.py verify                # numpy backend vs Blender, timing
    python skin.py check [--configs ...] # quality metrics -> skin/quality.json
    python skin.py render                # side views 0/45/90 deg, default vs plain skinning

From other modules (no GUI):

    from skin import pose_mesh, Skinner
    verts, faces = pose_mesh(blend_path, params, pose)          # params: dict | path | None
    sk = Skinner(blend_path, params); v = sk.pose(pose)          # keep it for many poses

Rigged hand .blend (SPEC): mesh 'hand scan', armature 'hand rig' (bones as SPEC, rest = open
pose of the scan), Armature modifier 'Armature', vertex groups named after bones. This module
adds/overwrites the vertex groups (parametric weights), a 'cs_mask' group + 'CorrectiveSmooth'
modifier, and one shape key 'bulge_<joint>' per finger joint.

Conventions (stated here because they are load-bearing):
- 1 unit = 1 mm, right hand.  Palm frame as relaxed_pose.angles: along = wrist->middle MCP,
  across = pinky MCP->index MCP (orthogonalised), palmar normal n = across x along
  (points out of the palm for a right hand).
- pose.json "angles": absolute angles in the relaxed_pose.py definitions (0 = straight,
  + toward the palm for finger flexion).  Missing keys keep the scan's rest value.
  THUMB SIGN: relaxed_pose.angles() returns thumb_mcp/thumb_ip NEGATIVE when the thumb bends
  toward the palm for a right hand (its axis is cross(meta, normal)).  pose.json follows the
  SPEC (+ = toward the palm), i.e. pose thumb_mcp = -relaxed_pose thumb_mcp.  spec_angles()
  does this conversion.
  Optional extra keys (DOFs the SPEC angles do not fix): arch_index / arch_ring / arch_pinky =
  elevation of that metacarpal (wrist->MCP) toward the palm side, asin(dir . n) in degrees;
  thumb_cmc_elev = elevation of the thumb metacarpal toward the palm side, same definition.
  Optional "twist": {bone: deg} axial rotation of a bone about its own length (tests).
  "root": {"location":[x,y,z], "rotation_quat":[w,x,y,z]}: rigid motion of the whole hand,
  rotation about the wrist then translation (armature space).
- Posing is independent of bone roll: the pose is solved as joint positions (forward
  kinematics on the scan's joints, rotations about anatomical axes, solved so that
  relaxed_pose.angles() of the posed joints equals the requested angles to <0.01 deg), then
  each pose bone gets matrix_basis = Rest^-1 Mparent^-1 M Rest.  So any roll works.
  Bone local X convention: in the rigs this module BUILDS, bone Z points to the palm side and
  X = Y x Z, so a positive rotation about local X curls toward the palm (SPEC).  In
  hold_rig.blend the rolls were aligned to cross(along, across) = the BACK of the hand, so
  there positive X curls toward the back for most bones, and some flexed bones have the
  opposite roll (see the report).  Nothing here relies on it.
- Flexion hinge axes: fingers k_f = unit(metacarpal x n) for MCP, PIP and DIP (the same axis
  relaxed_pose measures about), carried by the parent bone; MCP spread about n.  Thumb: hinge
  = normal of the thumb chain's own bending plane (rest), signed so + bends toward the palm;
  CMC = rotation about n (changes thumb_cmc) and about unit(thumb_meta x n) (elevation).

Weights (parametric, no bone heat):
- chain membership mu_c (thumb, index, middle, ring, pinky): soft-min of the distance to each
  chain's joint polyline (wrist-MCP-PIP-DIP-tip; thumb wrist-CMC-MCP-IP-tip), temperature
  palm_softness_mm on the palm (smooth weights between metacarpals = palm arch) and
  chain_softness_mm on the fingers.
- along each chain, every joint has a smooth (cosine) transition parent->child in the
  coordinate s = (v - J) . unit(d_parent + d_child) (bisector plane).  Centre of the zone =
  offset_mm + 0.5 * tilt_mm * p, where p = cosine of the vertex's angle around the joint axis
  to the palm side (+1 palm side, -1 back side): tilt moves the palm-side crease distally
  relative to the back-side knuckle.  Zone length = width_mm.  Weights cascade
  (palm|metacarpal|proximal|middle|distal), get weight_smooth_iters Laplacian passes, are cut
  to max_influences per vertex and normalised.  Thenar flesh goes to thumb_metacarpal via the
  wide thumb_cmc zone.
- cs_mask = transition zones (bump over cs_zone_scale * width), times the joint's 'smooth'.
- bulge_<joint> shape key: displacement along the rest normal, amplitude bulge_mm at
  bulge_ref_deg of flexion beyond the rest angle (zero at the scan's rest angle), Gaussian along
  the bone around the palm-side crease centre (sigma = bulge_sigma_scale * width), palm side
  only (max(p,0)^1.5).
Deformation: shape keys -> Armature (LBS, or DQS with dqs=true) -> Corrective Smooth (SIMPLE,
factor cs_factor, cs_iterations, vertex group cs_mask, rest = original mesh).  The numpy
backend implements the same three stages and is checked against Blender (verify).
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/tmp/claude-0/hs/pg")
from gripscan.relaxed_pose import angles as relaxed_angles  # noqa: E402
from gripscan.landmarks import NAMES  # noqa: E402

OUT = "/mnt/project-files/hand_scan_rig/skin"
WORK = "/tmp/claude-0/hs/work/skin"
STANDIN = "/mnt/project-files/hand_scan_rig/standin"
HOLD_RIG = "/tmp/claude-0/hs/hold_rig.blend"
TEST_BLEND = os.path.join(OUT, "standin_rigged.blend")
AUTO_BLEND = os.path.join(OUT, "standin_rigged_autoweights.blend")
PARAMS_PATH = os.path.join(OUT, "skin_params.json")

QUANT = 0   # weight quantisation (0 = exact floats)
F4 = ["index", "middle", "ring", "pinky"]
CHAINS = ["thumb"] + F4
SEGS = {"thumb": ["metacarpal", "proximal", "distal"]}
for _f in F4:
    SEGS[_f] = ["metacarpal", "proximal", "middle", "distal"]
LMK = {"thumb": ["thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"]}
for _f in F4:
    LMK[_f] = ["wrist", f"{_f}_mcp", f"{_f}_pip", f"{_f}_dip", f"{_f}_tip"]

BONES = ["palm"]
PARENT = {"palm": None}
HEADTAIL = {"palm": ("wrist", "knuckle")}
for _f in CHAINS:
    prev = "palm"
    for _i, _s in enumerate(SEGS[_f]):
        b = f"{_f}_{_s}"
        BONES.append(b)
        PARENT[b] = prev
        HEADTAIL[b] = (LMK[_f][_i], LMK[_f][_i + 1])
        prev = b
BI = {b: i for i, b in enumerate(BONES)}

# joints of every chain: (key, param name, chain, landmark, parent bone, child bone)
JOINTS = []
for _f in F4:
    JOINTS.append((f"{_f}_wrist", "wrist", _f, "wrist", "palm", f"{_f}_metacarpal"))
    for _j, (a, b) in zip(["mcp", "pip", "dip"], [("metacarpal", "proximal"), ("proximal", "middle"),
                                                 ("middle", "distal")]):
        JOINTS.append((f"{_f}_{_j}", f"{_f}_{_j}", _f, f"{_f}_{_j}", f"{_f}_{a}", f"{_f}_{b}"))
JOINTS += [("thumb_cmc", "thumb_cmc", "thumb", "thumb_cmc", "palm", "thumb_metacarpal"),
           ("thumb_mcp", "thumb_mcp", "thumb", "thumb_mcp", "thumb_metacarpal", "thumb_proximal"),
           ("thumb_ip", "thumb_ip", "thumb", "thumb_ip", "thumb_proximal", "thumb_distal")]
FLEX_JOINTS = [f"{f}_{j}" for f in F4 for j in ("mcp", "pip", "dip")] + ["thumb_mcp", "thumb_ip"]
# joints that may carry a pad bulge shape key: the hinges + the thumb CMC (thenar corrective,
# driven by the thumb metacarpal's local X swing = its flexion toward the palm / opposition)
BULGE_JOINTS = FLEX_JOINTS + ["thumb_cmc"]

# measurement / unknown layout of the pose solver (square system, 22 x 22)
MEAS = (["arch_imr", "arch_mrp", "spread_index_middle", "spread_middle_ring",
         "spread_ring_pinky"] + [f"{f}_{j}" for f in F4 for j in ("mcp", "pip", "dip")]
        + ["thumb_cmc", "thumb_cmc_elev", "thumb_mcp", "thumb_ip"])
# keys a pose.json should carry to pin the pose down completely: MEAS + the signed thumb
# CMC azimuth (thumb_cmc is an unsigned angle with a mirror solution) + thumb pronation
POSE_KEYS = MEAS + ["thumb_cmc_azimuth", "thumb_pronation"]
I_PRON = len(MEAS)          # extra solver unknown: thumb metacarpal axial rotation (pronation)
MAX_ANGLE_ERR_WARN = 0.5    # deg; a pose request off by more than this is impossible / ambiguous


# ------------------------------------------------------------------ defaults

def default_params():
    """Defaults: offsets/tilts from anatomy (see PARAM_DOC['anatomy']); zone widths, DQS,
    smoothing and softness chosen by measurement on the stand-in (skin/tuning.json)."""
    g = {
        "dqs": True,
        "cs_factor": 0.5, "cs_iterations": 10, "cs_zone_scale": 1.6,
        "chain_softness_mm": 2.0, "palm_softness_mm": 8.0, "thumb_softness_mm": 36.0,
        "thumb_blend": "hybrid", "thumb_pure": False, "thumb_gate_mm": 10.0,
        "web_smooth": 1.0, "web_smooth_power": 1.0, "thumb_web_mu": [0.0, 0.0],
        "weight_smooth_iters": 5, "max_influences": 8,
        "bulge_sigma_scale": 0.3, "bulge_ref_deg": 90.0, "bulge_two_lobes": True,
        "weights": "parametric",
        "contact_resolve": True, "contact_rest_sep_mm": 8.0, "contact_bulge": 0.3,
    }
    J = {
        "wrist": dict(width_mm=40.0, offset_mm=10.0, tilt_mm=0.0, smooth=0.3, bulge_mm=0.0),
        "thumb_cmc": dict(width_mm=36.0, offset_mm=0.0, tilt_mm=6.0, smooth=0.6, bulge_mm=4.0),
        "thumb_mcp": dict(width_mm=54.0, offset_mm=1.0, tilt_mm=8.0, smooth=1.0, bulge_mm=7.5),
        "thumb_ip": dict(width_mm=35.0, offset_mm=0.0, tilt_mm=-2.0, smooth=1.0, bulge_mm=0.9),
    }
    for f in F4:
        J[f"{f}_mcp"] = dict(width_mm=64.0, offset_mm=4.0, tilt_mm=10.0, smooth=1.0, bulge_mm=2.5)
        J[f"{f}_pip"] = dict(width_mm=42.0, offset_mm=0.5, tilt_mm=-2.0, smooth=1.0, bulge_mm=0.9)
        J[f"{f}_dip"] = dict(width_mm=32.0, offset_mm=0.5, tilt_mm=-2.0, smooth=1.0, bulge_mm=0.6)
    return {"global": g, "joints": J}


PARAM_DOC = {
    "global": {
        "dqs": "Armature modifier 'Preserve Volume' (dual quaternion skinning) instead of linear blend.",
        "cs_factor": "Corrective Smooth factor (Blender 'Factor', lambda per iteration, 0 = off).",
        "cs_iterations": "Corrective Smooth repeat count.",
        "cs_zone_scale": "Corrective Smooth mask covers this many times each joint's width_mm.",
        "chain_softness_mm": "How softly a finger vertex is shared with the neighbouring finger (web).",
        "palm_softness_mm": "Same on the palm: smooth weights between metacarpals (palm arch).",
        "thumb_softness_mm": "Width of the thumb / rest-of-hand blend over the thenar and first web.",
        "thumb_blend": ("How skin is shared between the thumb chain and the finger chains, by dt = distance to "
                        "the thumb minus distance to the nearest finger: 'logistic' 1/(1+exp(dt/thumb_softness)) "
                        "(old; never reaches 0, so with 24 mm the index PIP moved 4.3 mm when only the thumb bent, "
                        "10.7 mm with 36 mm); 'hybrid' (default) the same logistic over palm / thenar / first web "
                        "but zero on finger phalanges past their MCP zone (index PIP moves 0.18 mm); 'compact' a "
                        "cosine band +-thumb_softness (no leak, but 2-3x the opposition fold-overs)."),
        "thumb_pure": "hybrid only: also force the thumb phalanges to pure thumb (true gave 61 hold-pose folds; off).",
        "thumb_gate_mm": "hybrid only: dt range over which the finger-phalanx cut-off fades in.",
        "web_smooth": ("Extra Corrective Smooth in the thumb/rest-of-hand blend band (strength 4*m*(1-m) of the "
                       "thumb mix m); 1.0 removes most first-web folds in opposition, 0 = off."),
        "web_smooth_power": "Exponent on that band mask (>1 narrows it).",
        "thumb_web_mu": ("[lo, hi] thumb-mix range over which the thumb MCP/IP zones fade out in the first web; "
                         "[0, 0] = off (measured: little effect on the stand-in)."),
        "weight_smooth_iters": "Laplacian smoothing passes over the weight maps (removes kinks).",
        "max_influences": ("Bones per vertex after pruning (weights renormalised).  8, not 4: on the palm five "
                           "bones (index/middle/ring/thumb metacarpal + palm) carry nearly equal weight, and "
                           "pruning to 4 cut a diagonal groove across the palm in the hold pose."),
        "bulge_sigma_scale": "Bulge Gaussian sigma along the bone = this x the joint's width_mm.",
        "bulge_ref_deg": "Flexion (beyond the rest angle) at which the bulge reaches bulge_mm.",
        "bulge_two_lobes": "true: two pad lobes either side of the palm-side crease (zero on it); false: one bump on it.",
        "weights": "'parametric' (this module) or 'blend' (keep the vertex groups in the .blend).",
        "contact_resolve": ("After skinning, push skin that went through other skin (neighbouring fingers, "
                            "fingertip into the palm, deep creases) back to the contact surface (skin.resolve_contacts). "
                            "Applied by Skinner.pose / pose_mesh for both backends; NOT live in Blender's Pose Mode."),
        "contact_rest_sep_mm": "Only skin that was at least this far apart at rest can be in contact.",
        "contact_bulge": "Fraction of the volume pushed in at a contact that comes back as a swell around it.",
    },
    "joints": {
        "width_mm": "Length along the bone of the parent->child blend zone.",
        "offset_mm": "Shift of the zone centre along the bone (+ = distal), at the sides of the finger.",
        "tilt_mm": "Palm-side centre minus back-side centre (+ = palm crease lies distal of the knuckle).",
        "smooth": "Corrective Smooth mask strength in this joint's zone (0..1).",
        "bulge_mm": "Palm-side pad bulge along the normal at bulge_ref_deg of flexion.",
    },
    "anatomy": {
        "mcp": ("Back side: knuckle skin lies right over the MCP head (zone centre offset-tilt/2 = -1 mm). "
                "Palm side: the palmar digital crease is 15-20 mm distal of the MCP centre and the distal "
                "palmar crease lies over the joint, so the palm-side zone centre is offset+tilt/2 = +9 mm."),
        "pip": "Middle digital crease ~1-2 mm proximal of the PIP centre (palm-side centre -0.5 mm), back skin over the joint (+1.5 mm).",
        "dip": "Distal digital crease ~2 mm proximal of the DIP centre; back skin over the joint.",
        "thumb": "Thumb MCP crease distal on the palm side (+5 mm, tilt 8), IP crease slightly proximal; CMC zone wide (36 mm) so the thenar eminence is shared between palm and thumb metacarpal.",
        "wrist": "Metacarpals pivot at the wrist landmark; wide zone (40 mm) centred 10 mm distal so the palm arch bends the palm smoothly.",
        "bulge": ("Pad flesh bunches on the palm side when flexed, in two lobes either side of the crease.  Amplitudes at 90 deg (MCP 2.5, PIP 0.9, DIP 0.6, thumb MCP 7.5, IP 0.9 mm) set so that, together with DQS, the local volume at 90 deg stays within about +-10% (DQS alone keeps the PIP/DIP; MCP and thumb lose more).  The thumb MCP needs the largest bulge because the thenar/first-web flesh is thick: with 3.0 mm it lost 36% at 90 deg, 6 mm -15%, 7.5 mm +1%, 9 mm overshot into self-intersection."),
        "widths": ("Zone lengths are long (MCP 64, PIP 42, DIP 32, thumb MCP 54, IP 35 mm): skin is "
                   "carried by the whole segment, not just the joint.  Measured on the stand-in, "
                   "1.6x the first anatomical guess (MCP 40 / PIP 26 / DIP 20) halved the fold-overs "
                   "and crease sharpness at 90 deg (see skin/tuning.json).  Neighbouring zones overlap; "
                   "the cascade keeps weights normalised."),
        "dqs": "Dual quaternion skinning keeps finger cross-sections round (LBS loses 25-36% local volume at 90 deg and 7.6% radius under a 45 deg twist).",
        "softness": ("thumb_blend hybrid + thumb_softness 36 mm + web_smooth 1.0 + thumb CMC bulge 4 mm + thumb IP "
                     "bulge 0.9 (was logistic / 24 / 0 / 0 / 2.4).  Thumb opposition (thumb tip to pinky tip), "
                     "no contact pass: thenar/first-web fold-overs 158 -> 46, crease p99 145 -> 14 deg; hold pose "
                     "thumb MCP region folds 8 -> 0.  The old logistic also moved the index PIP 4.3 mm when only "
                     "the thumb bent; now 0.18 mm.  Thumb IP bulge lowered because the thumb no longer leaks onto "
                     "the index chain (2.4 mm then gave +16% local volume at 90 deg).  Tuned on the stand-in only."),
    },
}


def load_params(params=None):
    """dict | path | None -> full params (defaults filled in)."""
    p = default_params()
    if params is None and os.path.exists(PARAMS_PATH):
        params = PARAMS_PATH
    if isinstance(params, str):
        params = json.load(open(params))
    if params:
        for k, v in params.get("global", {}).items():
            p["global"][k] = v
        for j, d in params.get("joints", {}).items():
            p["joints"].setdefault(j, {}).update(d)
    return p


def params_key(p):
    return hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ math

def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def rot(axis, deg):
    a = unit(axis)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def quat_to_mat(q):
    w, x, y, z = unit(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def spec_angles(W, side="right"):
    """relaxed_pose.angles() with the thumb flexion sign turned to the SPEC (+ toward palm)."""
    a = relaxed_angles(np.asarray(W, float))
    if side == "right":
        a["thumb_mcp"], a["thumb_ip"] = -a["thumb_mcp"], -a["thumb_ip"]
    return a


def palm_frame(L):
    wrist = L["wrist"]
    along = unit(L["middle_mcp"] - wrist)
    c0 = L["index_mcp"] - L["pinky_mcp"]
    across = unit(c0 - (c0 @ along) * along)
    return along, across, np.cross(across, along)


def _dihedral(L, axis_end, a, b, n):
    e = unit(L[axis_end] - L["wrist"])
    u1 = L[a] - L["wrist"]
    u2 = L[b] - L["wrist"]
    u1 = unit(u1 - (u1 @ e) * e)
    u2 = unit(u2 - (u2 @ e) * e)
    ang = 180.0 - np.degrees(np.arccos(np.clip(u1 @ u2, -1, 1)))
    return float(ang if (u1 + u2) @ n >= 0 else -ang)


def measurements(L):
    """SPEC angles (relaxed_pose definitions, thumb sign to SPEC) + the extra DOFs:
    arch_imr / arch_mrp: palm arch, 0 = flat, + = cupped toward the palm; dihedral angle at the
    wrist->middle MCP (resp. wrist->ring MCP) edge between the triangles wrist-index-middle and
    wrist-middle-ring (resp. wrist-middle-ring and wrist-ring-pinky) MCPs.
    thumb_cmc_elev: asin(unit(thumb MCP - CMC) . n), elevation of the thumb metacarpal to the
    palm side."""
    W = np.array([L[n] for n in NAMES])
    a = spec_angles(W)
    along, across, n = palm_frame(L)
    a["arch_imr"] = _dihedral(L, "middle_mcp", "index_mcp", "ring_mcp", n)
    a["arch_mrp"] = _dihedral(L, "ring_mcp", "middle_mcp", "pinky_mcp", n)
    a["thumb_cmc_elev"] = float(np.degrees(np.arcsin(np.clip(unit(L["thumb_mcp"] - L["thumb_cmc"]) @ n, -1, 1))))
    # signed azimuth of the thumb metacarpal in the palm plane, from the index metacarpal:
    # + = radial (index / thumb side, the open hand), - = swung across the palm (opposition)
    ti = L["index_mcp"] - L["wrist"]
    tt = L["thumb_mcp"] - L["thumb_cmc"]
    ti, tt = ti - (ti @ n) * n, tt - (tt @ n) * n
    a["thumb_cmc_azimuth"] = float(np.degrees(np.arctan2(np.cross(tt, ti) @ n, ti @ tt)))
    flat = {}
    for f in F4:
        v = L[f + "_pip"] - L[f + "_mcp"]
        flat[f] = v - (v @ n) * n
    for x, y in [("index", "middle"), ("middle", "ring"), ("ring", "pinky")]:
        u, w = flat[x], flat[y]
        ang = np.degrees(np.arctan2(np.cross(u, w) @ n, u @ w))
        a[f"_sspread_{x}_{y}"] = float(ang)
        a[f"_inplane_{x}_{y}"] = float(min(np.linalg.norm(u) / np.linalg.norm(L[x + "_pip"] - L[x + "_mcp"]),
                                           np.linalg.norm(w) / np.linalg.norm(L[y + "_pip"] - L[y + "_mcp"])))
    return a


# ------------------------------------------------------------------ rig / forward kinematics

class Rig:
    """Rest joints of the scan + FK that turns SPEC angles into bone transforms."""

    def __init__(self, landmarks):
        L = {k: np.asarray(v, float) for k, v in landmarks.items() if k in NAMES}
        L["knuckle"] = np.mean([L[f + "_mcp"] for f in F4], axis=0)
        self.L = L
        self.along, self.across, self.n = palm_frame(L)
        n = self.n
        self.head = {b: L[HEADTAIL[b][0]] for b in BONES}
        self.tail = {b: L[HEADTAIL[b][1]] for b in BONES}
        self.dir = {b: unit(self.tail[b] - self.head[b]) for b in BONES}
        self.k = {f: unit(np.cross(self.dir[f + "_metacarpal"], n)) for f in F4}
        s0, s1, s2 = (L["thumb_mcp"] - L["thumb_cmc"], L["thumb_ip"] - L["thumb_mcp"],
                      L["thumb_tip"] - L["thumb_ip"])
        hc = np.cross(s0, s1) + np.cross(s1, s2)
        bend = np.degrees(np.arcsin(min(1.0, np.linalg.norm(hc) / (np.linalg.norm(s1) * (np.linalg.norm(s0) + np.linalg.norm(s2))))))
        h = unit(hc) if bend > 6 else unit(np.cross(s1, n))
        if np.cross(h, s1) @ n < 0:
            h = -h
        if bend > 6 and np.cross(h, unit(s1)) @ n < 0.2:   # bending plane not toward the palm
            h = unit(np.cross(s1, n))
        self.h_thumb = h
        self.e_thumb = unit(np.cross(self.dir["thumb_metacarpal"], n))
        # palm-side direction per bone (perpendicular to the bone)
        self.pal = {}
        for b in BONES:
            d = self.dir[b]
            ref = np.cross(self.h_thumb, d) if b.startswith("thumb") else n - (n @ d) * d
            self.pal[b] = unit(ref - (ref @ d) * d)
        self.rest_meas = measurements(L)
        self._x0 = np.zeros(len(MEAS) + 1)

    # x layout: alpha ring/pinky, phi index/ring/pinky, 12 finger flexions, beta1, beta2, t_mcp, t_ip
    def fk(self, x, twist=None, root=None):
        twist = twist or {}
        L, n = self.L, self.n
        if root is not None:
            R0 = quat_to_mat(root.get("rotation_quat", [1, 0, 0, 0]))
            loc = np.asarray(root.get("location", [0, 0, 0]), float)
        else:
            R0, loc = np.eye(3), np.zeros(3)
        Q = {b: np.eye(3) for b in BONES}
        for i, f in enumerate(("ring", "pinky")):
            Q[f + "_metacarpal"] = rot(self.k[f], x[i])
        phi = {"index": x[2], "middle": 0.0, "ring": x[3], "pinky": x[4]}
        for i, f in enumerate(F4):
            Q[f + "_proximal"] = rot(self.k[f], x[5 + 3 * i]) @ rot(n, phi[f])
            Q[f + "_middle"] = rot(self.k[f], x[6 + 3 * i])
            Q[f + "_distal"] = rot(self.k[f], x[7 + 3 * i])
        pron = float(x[I_PRON]) if len(x) > I_PRON else 0.0
        Q["thumb_metacarpal"] = rot(self.e_thumb, x[18]) @ rot(n, x[17]) @ rot(self.dir["thumb_metacarpal"], pron)
        Q["thumb_proximal"] = rot(self.h_thumb, x[19])
        Q["thumb_distal"] = rot(self.h_thumb, x[20])
        for b, deg in twist.items():
            Q[b] = Q[b] @ rot(self.dir[b], deg)
        Rw, Hp = {}, {}
        for b in BONES:
            p = PARENT[b]
            if p is None:
                Rw[b] = R0 @ Q[b]
                Hp[b] = self.head[b] + loc
            else:
                Rw[b] = Rw[p] @ Q[b]
                Hp[b] = Rw[p] @ (self.head[b] - self.head[p]) + Hp[p]
        M = {b: (Rw[b], Hp[b] - Rw[b] @ self.head[b]) for b in BONES}
        P = {"wrist": Hp["palm"]}
        for b in BONES:
            P[HEADTAIL[b][1]] = Rw[b] @ (self.tail[b] - self.head[b]) + Hp[b]
        P["thumb_cmc"] = Hp["thumb_metacarpal"]
        return M, P

    def solve(self, pose):
        """pose dict -> (bone transforms {b: (R, t)}, posed landmarks, solve info).

        Unknowns: the 21 rig DOFs (skin.MEAS order) + thumb metacarpal pronation.
        Thumb CMC: with "thumb_cmc_azimuth" in the request the CMC direction is solved from
        (azimuth, thumb_cmc_elev) and is unique; with only the SPEC "thumb_cmc" (an unsigned
        angle, two mirror solutions) the branch on the side of the scan's rest pose is taken
        and the info says so.  "thumb_pronation" (deg, + = about the metacarpal's own
        wrist->tip axis) is a real DOF; without it pronation stays at the rest value.
        "twist" ({bone: deg}) is applied inside the solve, so the requested angles still hold.
        info["warnings"] lists impossible / ambiguous / ignored requests; a request missed by
        more than MAX_ANGLE_ERR_WARN deg is also sent to warnings.warn."""
        import warnings as _warnings
        from scipy.optimize import least_squares
        pose = pose or {}
        ang = pose.get("angles", {})
        rm = self.rest_meas
        twist, root = pose.get("twist"), pose.get("root")
        use_az = "thumb_cmc_azimuth" in ang
        keys = [("thumb_cmc_azimuth" if (k == "thumb_cmc" and use_az) else k) for k in MEAS]
        tgt = np.array([float(ang.get(k, rm[k])) for k in keys])
        # spreads are solved signed (the unsigned relaxed_pose value flips through 0)
        sp_idx = [keys.index(k) for k in keys if k.startswith("spread")]
        for i in sp_idx:
            tgt[i] *= np.sign(rm["_s" + keys[i]]) or 1.0
        x0 = self._x0.copy()
        for i, k in enumerate(keys):
            if k in ang and not k.startswith(("spread", "thumb_cmc")):
                x0[i] = tgt[i] - rm[k]
        if use_az:
            x0[17] = -(tgt[keys.index("thumb_cmc_azimuth")] - rm["thumb_cmc_azimuth"])
        pron_req = "thumb_pronation" in ang
        if pron_req:
            x0[I_PRON] = float(ang["thumb_pronation"])

        def meas_vec(P):
            m = measurements(P)
            v = np.array([m[k] for k in keys])
            w = np.ones(len(keys))
            for i in sp_idx:
                v[i] = m["_s" + keys[i]]
                # spread is undefined when a proximal phalanx points along the palm normal
                # (weight 1 while >= 50% of the segment is in the palm plane, 0 below 15%)
                w[i] = np.clip((m["_inplane" + keys[i][6:]] - 0.15) / 0.35, 0, 1) ** 2
            return v, w, m

        reg = np.full(len(x0), 1e-3)
        # pronation is a direct DOF: held at the request (or at the rest value 0).  Left free it
        # would trade against thumb_mcp, whose relaxed_pose definition (fixed axis) is not the
        # hinge angle.
        reg[I_PRON] = 10.0

        def res(x):
            v, w, _ = meas_vec(self.fk(x, twist)[1])
            return np.r_[(v - tgt) * w, reg * (x - x0)]

        r0 = res(x0)[:len(keys)]
        best = None
        seeds = [x0]
        if use_az:      # the azimuth sign picks the branch; also try the mirrored start
            x1 = x0.copy()
            x1[17] = -x0[17]
            seeds.append(x1)
        for xs in seeds:
            sol = least_squares(res, xs, method="lm", xtol=1e-10, ftol=1e-12)
            if best is None or sol.cost < best.cost:
                best = sol
        sol = best
        v, w, m = meas_vec(self.fk(sol.x, twist)[1])
        err_vec = np.abs(v - tgt) * (w > 0.99)
        err = float(np.max(err_vec))
        warns = []
        worst = keys[int(np.argmax(err_vec))]
        if err > MAX_ANGLE_ERR_WARN:
            warns.append(f"pose not reachable: {worst} off by {err:.1f} deg (request impossible "
                         f"for this rig or inconsistent with the other angles)")
        for i in sp_idx:
            if keys[i] in ang and w[i] < 0.99:
                warns.append(f"{keys[i]} request ignored ({100 * w[i]:.0f}% weight): the proximal "
                             f"phalanges point out of the palm plane, spread is undefined there")
        if use_az and "thumb_cmc" in ang and abs(m["thumb_cmc"] - float(ang["thumb_cmc"])) > 1.0:
            warns.append(f"thumb_cmc {ang['thumb_cmc']:.1f} inconsistent with thumb_cmc_azimuth/elev "
                         f"(gives {m['thumb_cmc']:.1f}); azimuth + elevation used")
        if not use_az and "thumb_cmc" in ang:
            if np.sign(m["thumb_cmc_azimuth"]) != np.sign(rm["thumb_cmc_azimuth"]) and abs(m["thumb_cmc_azimuth"]) > 2:
                warns.append("thumb_cmc solved on the opposite branch to the rest pose; give "
                             "thumb_cmc_azimuth to pick the branch")
            note = "thumb CMC branch: rest side (no thumb_cmc_azimuth given)"
        else:
            note = None
        for wmsg in warns:
            _warnings.warn("skin.Rig.solve: " + wmsg, stacklevel=2)
        M, P = self.fk(sol.x, twist, root)
        return M, P, {"max_angle_err_deg": err, "worst": worst, "start_err": float(np.max(np.abs(r0))),
                      "x": sol.x, "warnings": warns, "branch_note": note,
                      "thumb_pronation": float(sol.x[I_PRON] + (twist or {}).get("thumb_metacarpal", 0.0)),
                      "thumb_cmc_azimuth": float(m["thumb_cmc_azimuth"])}


# ------------------------------------------------------------------ mesh helpers

def edges_of(F):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def adjacency(F, n):
    import scipy.sparse as sp
    e = edges_of(F)
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    return A, e


def vertex_normals(V, F):
    """Corner-angle weighted vertex normals (as Blender)."""
    t = V[F]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20)
    N = np.zeros_like(V)
    for c in range(3):
        a = unit(t[:, (c + 1) % 3] - t[:, c])
        b = unit(t[:, (c + 2) % 3] - t[:, c])
        ang = np.arccos(np.clip(np.sum(a * b, 1), -1, 1))
        np.add.at(N, F[:, c], fn * ang[:, None])
    return unit(N + 1e-30)


def seg_dist(V, a, b):
    d = b - a
    t = np.clip((V - a) @ d / (d @ d), 0, 1)
    return np.linalg.norm(V - (a + t[:, None] * d), axis=1)


def cosstep(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


# ------------------------------------------------------------------ parametric weights

def compute_weights(V, F, rig, params):
    """Parametric weight maps. Returns dict with W (n x len(BONES)), cs_mask, bulge basis, fields."""
    g, Jp = params["global"], params["joints"]
    n = len(V)
    L = rig.L
    # chain membership
    D = np.zeros((n, len(CHAINS)))
    for ci, c in enumerate(CHAINS):
        pts = [L["wrist"]] + [L[k] for k in LMK[c]] if c == "thumb" else [L[k] for k in LMK[c]]
        D[:, ci] = np.min([seg_dist(V, pts[i], pts[i + 1]) for i in range(len(pts) - 1)], axis=0)
    near = np.argmin(D, axis=1)
    geo = {}
    S = {}
    for key, pk, c, lm, pb, cb in JOINTS:
        J = L[lm]
        dp, dc = rig.dir[pb], rig.dir[cb]
        nj = unit(dp + dc)
        rel = V - J
        s = rel @ nj
        r = rel - s[:, None] * nj
        palv = rig.pal[pb] + rig.pal[cb]
        palv = unit(palv - (palv @ nj) * nj)
        p = (r @ palv) / np.maximum(np.linalg.norm(r, axis=1), 1e-9)
        q = Jp[pk]
        centre = q["offset_mm"] + 0.5 * q["tilt_mm"] * p
        S[key] = cosstep((s - centre) / q["width_mm"] + 0.5)
        geo[key] = dict(s=s, p=p, centre=centre, nj=nj, J=J, r=np.linalg.norm(r, axis=1))
    # finger vs palm softness: blend by how far past its MCP the vertex is, averaged over the
    # chains with a fixed soft-min (no hard switch -> no seams)
    first = {c: (f"{c}_mcp" if c != "thumb" else "thumb_mcp") for c in CHAINS}
    pre = np.exp(-(D - D.min(axis=1, keepdims=True)) / g["chain_softness_mm"])
    pre /= pre.sum(axis=1, keepdims=True)
    Sfirst = sum(pre[:, ci] * S[first[c]] for ci, c in enumerate(CHAINS))
    tau = g["palm_softness_mm"] + (g["chain_softness_mm"] - g["palm_softness_mm"]) * Sfirst
    # two levels: thumb vs the four fingers with thumb_softness_mm (the thenar/first web moves
    # with a 2-DOF CMC, so it needs a wide blend), then among the fingers with tau
    Df = D[:, 1:]
    mf = np.exp(-(Df - Df.min(axis=1, keepdims=True)) / tau[:, None])
    mf /= mf.sum(axis=1, keepdims=True)
    tt = g.get("thumb_softness_mm", g["palm_softness_mm"])   # constant: thumb and index are far apart
    dt = D[:, 0] - Df.min(axis=1)
    if g.get("thumb_blend", "logistic") == "hybrid":
        # wide logistic over the palm / thenar / first web (the palm gives way under an opposing
        # thumb), but the finger phalanges (past their MCP zone, nearer a finger) are never on
        # the thumb chain and the thumb phalanges never on a finger chain.  The plain logistic
        # never reaches 0 or 1: the index PIP moved 4.3 mm (thumb_softness 24) / 10.7 mm (36)
        # when only the thumb bent
        mt0 = 1.0 / (1.0 + np.exp(np.clip(dt / tt, -50, 50)))
        pf = pre[:, 1:] / np.maximum(pre[:, 1:].sum(axis=1, keepdims=True), 1e-12)
        s_f = sum(pf[:, ci - 1] * S[first[c]] for ci, c in enumerate(CHAINS) if c != "thumb")
        gw = float(g.get("thumb_gate_mm", 10.0))
        mt = mt0 * (1.0 - cosstep(dt / gw) * s_f)
        if g.get("thumb_pure", True):
            mt = mt + (1.0 - mt) * cosstep(-dt / gw) * S["thumb_mcp"]
    elif g.get("thumb_blend", "logistic") == "pure_phalanges":
        # wide logistic over the thenar / first web, but skin past the thumb MCP zone (and
        # nearer the thumb than any finger) is pure thumb: a wide logistic alone never reaches 1
        # and left the thumb phalanges 10-20% on the index chain (thumb IP flexion lost up to
        # 48% local volume)
        mt0 = 1.0 / (1.0 + np.exp(np.clip(dt / tt, -50, 50)))
        gate = cosstep(-dt / 10.0)
        mt = mt0 + (1.0 - mt0) * S["thumb_mcp"] * gate
    elif g.get("thumb_blend", "logistic") == "compact":
        # compact band: exactly thumb for dt < -tt, exactly hand for dt > tt (the logistic never
        # reaches 1, so a wide one left the thumb phalanges 10-20% on the index chain and
        # flexing the thumb IP lost up to 48% local volume)
        mt = cosstep(0.5 - 0.5 * dt / tt)
    else:
        mt = 1.0 / (1.0 + np.exp(np.clip(dt / tt, -50, 50)))
    mu = np.c_[mt, (1 - mt)[:, None] * mf]
    # cascaded bone weights per chain
    W = np.zeros((n, len(BONES)))
    # first web: skin shared between thumb and index (mu_thumb ~ 0.5) belongs to the thumb
    # METACARPAL, not to the thumb phalanges (it does not travel with the thumb MCP/IP when the
    # thumb opposes): the thumb MCP/IP transitions only act where mu_thumb > web_lo .. web_hi
    wlo, whi = g.get("thumb_web_mu", [0.0, 0.0])
    web_ramp = cosstep((mu[:, 0] - wlo) / max(whi - wlo, 1e-6)) if whi > wlo else np.ones(n)
    for ci, c in enumerate(CHAINS):
        js = [j for j in JOINTS if j[2] == c]
        bones = [js[0][4]] + [j[5] for j in js]
        acc = np.ones(n)
        for i, b in enumerate(bones):
            if i < len(js):
                sj = S[js[i][0]]
                if c == "thumb" and js[i][0] in ("thumb_mcp", "thumb_ip"):
                    sj = sj * web_ramp
                W[:, BI[b]] += mu[:, ci] * acc * (1 - sj)
                acc = acc * sj
            else:
                W[:, BI[b]] += mu[:, ci] * acc
    A, E = adjacency(F, n)
    deg = np.asarray(A.sum(axis=1)).ravel()
    for _ in range(int(g["weight_smooth_iters"])):
        W = W + 0.5 * ((A @ W) / deg[:, None] - W)
    # continuous pruning to max_influences: subtract the (k+1)-th largest weight and clip
    # (hard zeroing leaves seams where the 4th and 5th bone swap rank)
    k = int(g["max_influences"])
    if k < len(BONES):
        kth = -np.partition(-W, k, axis=1)[:, k]
        W = np.clip(W - kth[:, None], 0, None)
    W = np.clip(W, 0, None)
    W /= W.sum(axis=1, keepdims=True)
    W = np.round(W * QUANT) / QUANT if QUANT else W
    # corrective-smooth mask and bulge basis
    N0 = vertex_normals(V, F)
    mask = np.zeros(n)
    bulge = {}
    for key, pk, c, lm, pb, cb in JOINTS:
        q = Jp[pk]
        G = geo[key]
        ci = CHAINS.index(c)
        t = (G["s"] - G["centre"]) / (q["width_mm"] * g["cs_zone_scale"]) + 0.5
        bump = np.where((t > 0) & (t < 1), np.sin(np.pi * np.clip(t, 0, 1)), 0.0)
        mask = np.maximum(mask, mu[:, ci] * bump * q.get("smooth", 1.0))
        if key in BULGE_JOINTS and q.get("bulge_mm", 0) > 0:
            cpal = q["offset_mm"] + 0.5 * q["tilt_mm"]
            sig = g["bulge_sigma_scale"] * q["width_mm"]
            # two pad lobes either side of the crease (zero on the crease line itself, peak 1 at
            # +-sqrt(2) sigma): the pad flesh bunches up beside the fold instead of filling it
            u = ((G["s"] - cpal) / sig) ** 2
            lobe = 0.5 * np.e * u * np.exp(-0.5 * u) if g.get("bulge_two_lobes", True) else np.exp(-0.5 * u)
            amp = q["bulge_mm"] * mu[:, ci] * np.clip(G["p"], 0, 1) ** 1.5 * lobe
            amp[amp < 1e-3] = 0
            bulge[key] = amp[:, None] * N0
    # first web / thenar: the band where skin is shared between the thumb and the rest of the
    # hand (thumb membership ~ 0.5) blends bones that turn 40-60 deg apart when the thumb
    # opposes; Corrective Smooth there keeps the web from folding
    ws = float(g.get("web_smooth", 0.0))
    if ws > 0:
        web = np.clip(4.0 * mt * (1.0 - mt), 0, 1) ** float(g.get("web_smooth_power", 1.0))
        mask = np.maximum(mask, ws * web)
    mask = np.round(np.clip(mask, 0, 1) * 2048) / 2048
    return dict(W=W, cs_mask=mask, bulge=bulge, mu=mu, near=near, geo=geo, S=S, edges=E)


# ------------------------------------------------------------------ numpy deformation

def skin_lbs(V, W, M):
    out = np.zeros_like(V)
    for b, (R, t) in M.items():
        w = W[:, BI[b]]
        m = w > 0
        if m.any():
            out[m] += w[m, None] * (V[m] @ R.T + t)
    return out / W.sum(axis=1, keepdims=True)


def mat_to_quat(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q


def qmul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def skin_dqs(V, W, M, order=None):
    """Dual-quaternion skinning as Blender's armature deform (hemisphere of the running sum)."""
    n = len(V)
    q0 = np.zeros((n, 4))
    qe = np.zeros((n, 4))
    if order is None:
        # Blender accumulates in the order of the vertex's deform-group list; for groups written
        # by this module that is bone order (it matters when blended bones differ by > 180 deg in
        # quaternion space, e.g. a wide zone spanning MCP+PIP+DIP at 90 deg each)
        order = np.argsort(np.where(W > 0, 0, 1) * len(BONES) + np.arange(W.shape[1])[None, :], axis=1, kind="stable")
    bones = list(M)
    Qr = np.array([mat_to_quat(M[b][0]) for b in BONES])
    Qd = np.array([0.5 * qmul(np.r_[0, M[b][1]], Qr[i]) for i, b in enumerate(BONES)])
    for col in range(order.shape[1]):
        bi = order[:, col]
        w = W[np.arange(n), bi]
        if not (w > 0).any():
            break
        qr, qd = Qr[bi], Qd[bi]
        sgn = np.where(np.sum(qr * q0, 1) < 0, -1.0, 1.0)
        q0 += (w * sgn)[:, None] * qr
        qe += (w * sgn)[:, None] * qd
    ln = np.linalg.norm(q0, axis=1, keepdims=True)
    q0, qe = q0 / ln, qe / ln
    w0, v = q0[:, :1], q0[:, 1:]
    rv = V + 2 * np.cross(v, np.cross(v, V) + w0 * V)
    t = 2 * (q0[:, :1] * qe[:, 1:] - qe[:, :1] * q0[:, 1:] + np.cross(q0[:, 1:], qe[:, 1:]))
    del bones
    return rv + t


def contact_labels(W):
    """Per-vertex part label for contact search: 0 = palm (palm + finger metacarpal bones),
    1..5 = thumb, index, middle, ring, pinky (phalanges; the thumb metacarpal / thenar counts
    as thumb)."""
    lab_b = np.zeros(len(BONES), int)
    for b, i in BI.items():
        if b == "palm" or (b.endswith("_metacarpal") and not b.startswith("thumb")):
            lab_b[i] = 0
        else:
            lab_b[i] = 1 + CHAINS.index(b.split("_")[0])
    return lab_b[np.argmax(W, axis=1)]


def _nearest_other(X, F, C, fn, V0, C0, vlab, flab, reach_mm, rest_sep_mm, k=6):
    """For every vertex: nearest posed face of ANOTHER part whose rest position was more than
    rest_sep_mm away.  Returns (has, face index, signed distance along that face's normal)."""
    from scipy.spatial import cKDTree
    n = len(X)
    best_d = np.full(n, np.inf)
    best_f = np.zeros(n, int)
    for L in np.unique(flab):
        fs = np.flatnonzero(flab == L)
        vs = np.flatnonzero(vlab != L)
        if not len(fs) or not len(vs):
            continue
        kk = min(k, len(fs))
        d, idx = cKDTree(C[fs]).query(X[vs], k=kk, distance_upper_bound=reach_mm)
        d, idx = d.reshape(len(vs), kk), idx.reshape(len(vs), kk)
        valid = np.isfinite(d)
        fi = fs[np.where(valid, idx, 0)]
        far = np.linalg.norm(C0[fi] - V0[vs, None, :], axis=2) > rest_sep_mm
        dd = np.where(valid & far, d, np.inf)
        j = np.argmin(dd, axis=1)
        dj = dd[np.arange(len(vs)), j]
        upd = dj < best_d[vs]
        best_d[vs[upd]] = dj[upd]
        best_f[vs[upd]] = fi[np.arange(len(vs)), j][upd]
    has = np.isfinite(best_d)
    sd = np.einsum("ij,ij->i", X - C[best_f], fn[best_f])
    return has, best_f, sd


def resolve_contacts(V, F, V0, A=None, labels=None, iters=40, rest_sep_mm=8.0, reach_mm=8.0,
                     smooth_iters=4, clearance_mm=0.05, bulge=0.6, tol_mm=0.2):
    """Skin-on-skin contact after skinning: where skin of one part has gone into another part
    (a finger into its neighbour, a fingertip into the palm), push both surfaces back to their
    common contact surface so they flatten against each other like real skin, and put part of
    the displaced volume back as a soft swell around the contact patch.

    Parts: contact_labels(W) (palm, thumb, index, middle, ring, pinky).  Penetration test per
    vertex v: nearest posed face of another part within reach_mm whose REST position was more
    than rest_sep_mm from v's rest position (so the shared skin of neighbouring parts never
    counts); signed distance s = (v - c) . n_face < 0  ->  v is inside by -s.  Each pass moves
    every penetrating vertex half way out along that face normal (the other side moves the
    other half), the push is spread over a few rings (Laplacian), repeated until clear.
    Returns (V corrected, info)."""
    from scipy.spatial import cKDTree
    n = len(V)
    if A is None:
        A, _ = adjacency(F, n)
    if labels is None:
        labels = np.zeros(n, int)
    deg = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1)
    X = np.array(V, float, copy=True)
    C0 = V0[F].mean(axis=1)
    flab = labels[F[:, 0]]
    first_depth, touched = None, np.zeros(n, bool)
    lost = 0.0

    def faces(X):
        t = X[F]
        fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        fa = 0.5 * np.linalg.norm(fn, axis=1)
        return t.mean(axis=1), fn / np.maximum(2 * fa[:, None], 1e-20), fa

    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    Fl = F.tolist()

    def inside(X, C, fn, cand):
        """Ray test: a vertex is inside other skin when the first surface hit along its own
        outward normal (within reach) is a back face (we are leaving a solid) of skin that
        was far from it at rest."""
        N = np.zeros_like(X)
        for c in range(3):
            np.add.at(N, F[:, c], fn)
        N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-20)
        bvh = BVHTree.FromPolygons(X.tolist(), Fl, epsilon=0.0)
        ok = np.zeros(len(X), bool)
        for i in np.flatnonzero(cand):
            o = X[i] + 1e-3 * N[i]
            loc, nrm, idx, dist = bvh.ray_cast(Vector(o), Vector(N[i]), 2.0 * reach_mm)
            if idx is not None and fn[idx] @ N[i] > 0 and np.linalg.norm(C0[idx] - V0[i]) > rest_sep_mm:
                ok[i] = True
        return ok

    after = 0.0
    for it in range(iters + 1):
        C, fn, fa = faces(X)
        has, fi, sd = _nearest_other(X, F, C, fn, V0, C0, labels, flab, reach_mm, rest_sep_mm)
        pen = has & (sd < 0)
        if pen.any():
            pen &= inside(X, C, fn, pen)
        depth = np.where(pen, -sd, 0.0)
        if first_depth is None:
            first_depth = depth.copy()
        after = float(depth.max())
        if not pen.any() or it == iters or after < tol_mm:
            break
        touched |= pen
        disp = np.zeros_like(X)
        disp[pen] = (0.5 * depth[pen] + clearance_mm)[:, None] * fn[fi[pen]]
        need = np.linalg.norm(disp, axis=1)
        for _ in range(smooth_iters):
            disp = 0.5 * disp + 0.5 * (A @ disp) / deg[:, None]
        got = np.linalg.norm(disp, axis=1)
        sc = np.where(pen & (got > 1e-9), need / np.maximum(got, 1e-9), 1.0)
        disp[pen] *= sc[pen, None]          # penetrating vertices get their full push
        va = np.zeros(n)
        np.add.at(va, F.ravel(), np.repeat(fa / 3.0, 3))
        X += disp
        lost += float((va * np.linalg.norm(disp, axis=1)).sum())
    info = {"contact_depth_max_mm": float(first_depth.max()), "contact_vertices": int(touched.sum()),
            "contact_depth_after_mm": after}
    if touched.any() and bulge > 0 and lost > 0:
        # put `bulge` x the displaced volume back as a smooth outward swell on the ring
        # 1-8 mm (rest) around each contact patch (flesh pushed aside)
        r, _ = cKDTree(V0[touched]).query(V0, distance_upper_bound=8.0)
        ring = np.isfinite(r) & ~touched & (r > 1.0)
        wgt = np.where(ring, np.sin(np.pi * np.clip((r - 1.0) / 7.0, 0, 1)), 0.0)
        if wgt.sum() > 0:
            C, fn, fa = faces(X)
            N = np.zeros_like(X)
            for c in range(3):
                np.add.at(N, F[:, c], fn * fa[:, None])
            N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-20)
            va = np.zeros(n)
            np.add.at(va, F.ravel(), np.repeat(fa / 3.0, 3))
            h = min(bulge * 0.5 * lost / max((va * wgt).sum(), 1e-9), 1.5)   # both sides counted
            X += (h * wgt)[:, None] * N
            info["bulge_mm"] = float(h)
    return X, info


def bulge_curve(r):
    """Bulge amount for r = flexion / bulge_ref_deg: linear up to 1, then saturating smoothly
    (C1) toward 1.5 (pad flesh cannot keep growing past 90 deg); 0 for extension.  The
    Blender drivers use the same formula (BULGE_EXPR)."""
    r = max(float(r), 0.0)
    u = max(r - 1.0, 0.0) / 0.5
    return min(r, 1.0 + 0.5 * u / (1.0 + u))


BULGE_EXPR = ("max(0.0, min(flex / {R:.6f}, 1.0 + 0.5 * (max(flex / {R:.6f} - 1.0, 0.0) / 0.5) / "
              "(1.0 + max(flex / {R:.6f} - 1.0, 0.0) / 0.5)))")


def swing_x_deg(Rl):
    """X angle of the swing part of rotation Rl (bone frame, twist axis Y), degrees,
    as Blender's driver transform channel ROT_X in SWING_TWIST_Y mode."""
    q = mat_to_quat(Rl)                       # w, x, y, z
    w, x, y, z = q
    tw = np.array([w, 0.0, y, 0.0])
    nt = np.linalg.norm(tw)
    tw = tw / nt if nt > 1e-12 else np.array([1.0, 0, 0, 0])
    sw = qmul(q, tw * np.array([1, -1, -1, -1]))     # swing = q * conj(twist)
    if sw[0] < 0:
        sw = -sw
    return float(np.degrees(2.0 * np.arcsin(np.clip(sw[1], -1, 1))))   # = Blender (checked)


class CorrectiveSmooth:
    """Blender's Corrective Smooth (SIMPLE smoothing, rest = original mesh), in numpy."""

    def __init__(self, V0, F, mask, factor, iterations):
        import scipy.sparse as sp
        n = len(V0)
        self.F = F
        A, E = adjacency(F, n)
        cnt = np.asarray(A.sum(axis=1)).ravel()
        self.A, self.cnt = A, cnt
        self.div = mask * factor * np.where(cnt > 0, 1.0 / np.maximum(cnt, 1), 1.0)
        self.it = int(iterations)
        self.active = factor > 0 and self.it > 0 and mask.max() > 0
        self.corner_v = F.reshape(-1)
        if self.active:
            sm = self.smooth(V0)
            T, _ = self.tangents(sm)
            d = (V0 - sm)[self.corner_v]
            self.deltas = np.einsum("nij,nj->ni", T, d)   # components along the frame axes
        del sp

    def smooth(self, X):
        X = X.copy()
        for _ in range(self.it):
            X += self.div[:, None] * (self.A @ X - self.cnt[:, None] * X)
        return X

    def tangents(self, X):
        F = self.F
        cur, prv, nxt = F, np.roll(F, 1, axis=1), np.roll(F, -1, axis=1)
        dprev = unit(X[prv] - X[cur] + 1e-30)
        dnext = unit(X[cur] - X[nxt] + 1e-30)
        # dir_prev = prev - curr, dir_next = curr - next for every corner (matches Blender 5.2
        # to 1e-4 mm, see verify.json)
        dprev, dnext = dprev.reshape(-1, 3), dnext.reshape(-1, 3)
        z = unit(np.cross(dprev, dnext) + 1e-30)
        x = unit(dprev + dnext + 1e-30)
        y = np.cross(z, x)
        T = np.stack([x, y, z], axis=1)   # rows = axes
        wgt = np.abs(np.arccos(np.clip(np.sum(dnext * dprev, 1), -1, 1)))
        return T, wgt

    def __call__(self, X):
        if not self.active:
            return X
        sm = self.smooth(X)
        T, wgt = self.tangents(sm)
        wv = np.zeros(len(X))
        np.add.at(wv, self.corner_v, wgt)
        w = wgt / wv[self.corner_v]
        delta = np.einsum("nij,ni->nj", T, self.deltas) * w[:, None]
        out = sm.copy()
        np.add.at(out, self.corner_v, delta)
        return out


# ------------------------------------------------------------------ Blender side

def _bpy():
    import bpy
    return bpy


def read_ply(path):
    import trimesh
    m = trimesh.load(path, process=False)
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


def write_ply(path, V, F):
    import trimesh
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    trimesh.Trimesh(V, F, process=False).export(path)


def build_test_rig(ply, joints_json, out_blend, params=None, auto_weights=False):
    """Tiny stand-in for module B: mesh + armature from ground-truth joints, SPEC layout."""
    bpy = _bpy()
    from mathutils import Matrix, Vector
    V, F = read_ply(ply)
    jj = json.load(open(joints_json))
    rig = Rig(jj["landmarks"])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 0.001
    sc.unit_settings.length_unit = "MILLIMETERS"
    me = bpy.data.meshes.new("hand scan")
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.astype(np.float32).ravel())
    me.loops.add(F.size)
    me.loops.foreach_set("vertex_index", F.astype(np.int32).ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, F.size, 3, dtype=np.int32))
    me.update(calc_edges=True)
    me.validate()
    ob = bpy.data.objects.new("hand scan", me)
    sc.collection.objects.link(ob)
    ad = bpy.data.armatures.new("hand rig")
    arm = bpy.data.objects.new("hand rig", ad)
    sc.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for b in BONES:
        e = ad.edit_bones.new(b)
        e.head, e.tail = Vector(rig.head[b]), Vector(rig.tail[b])
        e.align_roll(Vector(rig.pal[b]))      # Z -> palm side, so +X rotation curls to the palm
    for b in BONES:
        if PARENT[b]:
            e = ad.edit_bones[b]
            e.parent = ad.edit_bones[PARENT[b]]
            e.use_connect = np.allclose(rig.head[b], rig.tail[PARENT[b]], atol=1e-4)
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    arm["joints_json"] = json.dumps({k: list(map(float, v)) for k, v in jj["landmarks"].items()})
    ob.parent = arm
    if auto_weights:
        for o in bpy.context.selected_objects:
            o.select_set(False)
        ob.select_set(True)
        arm.select_set(True)
        bpy.context.view_layer.objects.active = arm
        t0 = time.time()
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        print(f"bone heat weights: {time.time() - t0:.1f} s, groups {len(ob.vertex_groups)}")
    else:
        mod = ob.modifiers.new("Armature", "ARMATURE")
        mod.object = arm
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out_blend))
    if not auto_weights:
        sk = Skinner(out_blend, params)
        sk.apply_to_blend()
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out_blend))
    print("WROTE", out_blend)


def _landmarks_from_armature(arm):
    from mathutils import Matrix  # noqa: F401
    mw = np.array(arm.matrix_world)
    bones = arm.data.bones

    def P(v):
        return (mw @ np.r_[np.array(v), 1.0])[:3]

    L = {"wrist": P(bones["palm"].head_local)}
    for b in BONES[1:]:
        L[HEADTAIL[b][1]] = P(bones[b].tail_local)
    L["thumb_cmc"] = P(bones["thumb_metacarpal"].head_local)
    return L


def add_bulge_drivers(ob, arm, params, rest_mat=None):
    """Drivers so the bulge shape keys follow the bones live in Pose Mode:
    bulge_<joint> = bulge_curve(child bone local X rotation / bulge_ref_deg) (BULGE_EXPR,
    a Blender 'simple expression', so it runs without enabling Python scripts).  Same
    measure and curve as the numpy backend (Skinner.local_flex / bulge_curve)."""
    keys = ob.data.shape_keys
    if not keys:
        return []
    child = {j[0]: j[5] for j in JOINTS}
    ref = np.radians(params["global"]["bulge_ref_deg"])
    made = []
    for kb in keys.key_blocks[1:]:
        j = kb.name[len("bulge_"):]
        if j not in child:
            continue
        kb.driver_remove("value")
        fc = kb.driver_add("value")
        drv = fc.driver
        drv.type = "SCRIPTED"
        v = drv.variables.new()
        v.name, v.type = "flex", "TRANSFORMS"
        tg = v.targets[0]
        tg.id, tg.bone_target = arm, child[j]
        tg.transform_type, tg.transform_space, tg.rotation_mode = "ROT_X", "LOCAL_SPACE", "SWING_TWIST_Y"
        drv.expression = BULGE_EXPR.format(R=ref)
        for m in list(fc.modifiers):
            fc.modifiers.remove(m)
        made.append((kb.name, child[j], drv.is_simple_expression))
    return made


class Skinner:
    """Holds a rigged hand and poses it.  backend 'blender' (Armature + Corrective Smooth
    modifiers evaluated by Blender) or 'numpy' (same algorithm, verified against Blender)."""

    def __init__(self, blend_path, params=None):
        bpy = _bpy()
        self.blend_path = os.path.abspath(blend_path)
        bpy.ops.wm.open_mainfile(filepath=self.blend_path)
        self.ob = bpy.data.objects["hand scan"]
        self.arm = bpy.data.objects["hand rig"]
        me = self.ob.data
        n = len(me.vertices)
        co = np.zeros(n * 3, np.float32)
        me.vertices.foreach_get("co", co)
        mw = np.array(self.ob.matrix_world)
        self.mesh_mw = mw
        self.V0 = co.reshape(-1, 3).astype(float) @ mw[:3, :3].T + mw[:3, 3]
        lt = np.zeros(len(me.loop_triangles) * 3, np.int32) if False else None
        me.calc_loop_triangles()
        if any(len(p.vertices) != 3 for p in me.polygons):
            raise ValueError("'hand scan' must be a triangle mesh")
        vi = np.zeros(len(me.polygons) * 3, np.int32)
        me.polygons.foreach_get("vertices", vi)
        self.F = vi.reshape(-1, 3).astype(np.int64)
        del lt
        self.rig = Rig(_landmarks_from_armature(self.arm))
        self.amw = np.array(self.arm.matrix_world)
        self.rest_mat = {b: np.array(self.arm.data.bones[b].matrix_local) for b in BONES}
        self.applied_key = None
        self.set_params(params)

    # -- parameters / weights
    def set_params(self, params):
        self.params = load_params(params)
        g = self.params["global"]
        self.order = None
        if g["weights"] == "blend":
            self.W = self._read_groups()
            self.cs_mask = np.zeros(len(self.V0))
            self.bulge = {}
            self.fields = None
        else:
            r = compute_weights(self.V0, self.F, self.rig, self.params)
            self.W, self.cs_mask, self.bulge, self.fields = r["W"], r["cs_mask"], r["bulge"], r
        self.cs = CorrectiveSmooth(self.V0, self.F, self.cs_mask, g["cs_factor"], g["cs_iterations"])
        self._clabels = None

    def _read_groups(self):
        names = {vg.index: vg.name for vg in self.ob.vertex_groups}
        W = np.zeros((len(self.V0), len(BONES)))
        order = np.full((len(self.V0), len(BONES)), -1, int)
        for v in self.ob.data.vertices:
            k = 0
            for gr in v.groups:
                b = names.get(gr.group)
                if b in BI and gr.weight > 0:
                    W[v.index, BI[b]] = gr.weight
                    order[v.index, k] = BI[b]
                    k += 1
        none = W.sum(1) == 0
        W[none, BI["palm"]] = 1.0
        order[none, 0] = BI["palm"]
        # pad with unused bones (weight 0) so every row is a full permutation
        for i in range(len(order)):
            used = order[i][order[i] >= 0]
            order[i] = np.r_[used, np.setdiff1d(np.arange(len(BONES)), used)]
        self.order = order
        return W

    def apply_to_blend(self):
        """Write vertex groups, cs_mask, bulge shape keys and modifiers into the open .blend."""
        key = params_key(self.params)
        if self.applied_key == key:
            return
        ob, g = self.ob, self.params["global"]
        if g["weights"] != "blend":
            for vg in list(ob.vertex_groups):
                if vg.name in BI or vg.name == "cs_mask":
                    ob.vertex_groups.remove(vg)
            for b in BONES:
                self._write_group(b, self.W[:, BI[b]])
            self._write_group("cs_mask", self.cs_mask)
            if ob.data.shape_keys:
                ob.shape_key_clear()
            if self.bulge:
                ob.shape_key_add(name="Basis", from_mix=False)
                inv = np.linalg.inv(self.mesh_mw[:3, :3])
                for j, d in self.bulge.items():
                    k = ob.shape_key_add(name=f"bulge_{j}", from_mix=False)
                    k.slider_max = 3.0
                    co = np.zeros(len(self.V0) * 3, np.float32)
                    k.data.foreach_get("co", co)
                    k.data.foreach_set("co", (co.reshape(-1, 3) + d @ inv.T).astype(np.float32).ravel())
                    k.value = 0.0
        mod = ob.modifiers.get("Armature") or ob.modifiers.new("Armature", "ARMATURE")
        mod.object = self.arm
        mod.use_vertex_groups, mod.use_bone_envelopes = True, False
        mod.use_deform_preserve_volume = bool(g["dqs"])
        cs = ob.modifiers.get("CorrectiveSmooth")
        if g["cs_factor"] > 0 and g["cs_iterations"] > 0 and self.cs_mask.max() > 0:
            cs = cs or ob.modifiers.new("CorrectiveSmooth", "CORRECTIVE_SMOOTH")
            cs.factor, cs.iterations, cs.scale = g["cs_factor"], int(g["cs_iterations"]), 1.0
            cs.smooth_type, cs.rest_source = "SIMPLE", "ORCO"
            cs.vertex_group, cs.use_only_smooth, cs.use_pin_boundary = "cs_mask", False, False
        elif cs:
            ob.modifiers.remove(cs)
        self.drivers = add_bulge_drivers(ob, self.arm, self.params, self.rest_mat)
        ob["skin_params"] = json.dumps(self.params)
        self.applied_key = key

    def _write_group(self, name, w):
        vg = self.ob.vertex_groups.new(name=name)
        idx = np.nonzero(w > 0)[0]
        if not len(idx):
            return
        vals, inv = np.unique(w[idx], return_inverse=True)
        order = np.argsort(inv, kind="stable")
        bounds = np.searchsorted(inv[order], np.arange(len(vals) + 1))
        for i, u in enumerate(vals):
            vg.add(idx[order[bounds[i]:bounds[i + 1]]].tolist(), float(u), "REPLACE")

    # -- posing
    def local_flex(self, M):
        """Flexion of every FLEX_JOINT as the child bone's own local X rotation relative to its
        parent (swing-twist about the bone's Y, X component of the swing: what the bulge
        drivers read in Blender, SWING_TWIST_Y / ROT_X, local space), degrees, 0 at rest.
        This is the hinge angle; for the thumb it differs from relaxed_pose's thumb_mcp/ip
        (measured about a fixed axis) whenever the thumb metacarpal is pronated."""
        out = {}
        A3 = self.amw[:3, :3]
        for key, pk, c, lm, pb, cb in JOINTS:
            if key not in BULGE_JOINTS:
                continue
            B = A3 @ self.rest_mat[cb][:3, :3]
            B = B / np.linalg.norm(B, axis=0, keepdims=True)
            Rrel = M[pb][0].T @ M[cb][0]
            out[key] = swing_x_deg(B.T @ Rrel @ B)
        return out

    def bulge_values(self, P=None, M=None):
        """Bulge shape-key values.  From the bone transforms M (preferred: the hinge flexion
        the Blender drivers read) or, without M, from relaxed_pose angles of the landmarks P."""
        g = self.params["global"]
        if M is not None:
            fl = self.local_flex(M)
            return {j: bulge_curve(fl[j] / g["bulge_ref_deg"]) for j in self.bulge}
        m = measurements(P)
        return {j: bulge_curve((m[j] - self.rig.rest_meas[j]) / g["bulge_ref_deg"]) for j in self.bulge}

    def pose(self, pose=None, backend="blender", return_info=False):
        M, P, info = self.rig.solve(pose)
        bv = self.bulge_values(M=M)
        if backend == "numpy":
            V = self._pose_numpy(M, bv)
        else:
            V = self._pose_blender(M, bv)
        g = self.params["global"]
        if g.get("contact_resolve", False):
            if getattr(self, "_clabels", None) is None:
                self._clabels = contact_labels(self.W)
            V, cinfo = resolve_contacts(V, self.F, self.V0, self.cs.A, labels=self._clabels,
                                        rest_sep_mm=g.get("contact_rest_sep_mm", 8.0),
                                        bulge=g.get("contact_bulge", 0.6))
            info["contacts"] = cinfo
        info.update(landmarks=P, transforms=M, bulge=bv)
        return (V, info) if return_info else V

    def _pose_numpy(self, M, bv):
        X = self.V0.copy()
        for j, a in bv.items():
            if a > 0:
                X += a * self.bulge[j]
        X = skin_dqs(X, self.W, M, self.order) if self.params["global"]["dqs"] else skin_lbs(X, self.W, M)
        return self.cs(X)

    def _pose_blender(self, M, bv):
        bpy = _bpy()
        from mathutils import Matrix
        self.apply_to_blend()
        A, Ai = self.amw, np.linalg.inv(self.amw)
        Marm = {}
        for b, (R, t) in M.items():
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, t
            Marm[b] = Ai @ T @ A
        for b in BONES:
            Rb = self.rest_mat[b]
            Mp = Marm[PARENT[b]] if PARENT[b] else np.eye(4)
            B = np.linalg.inv(Rb) @ np.linalg.inv(Mp) @ Marm[b] @ Rb
            self.arm.pose.bones[b].matrix_basis = Matrix(B.tolist())
        keys = self.ob.data.shape_keys
        if keys:
            for kb in keys.key_blocks[1:]:
                kb.value = bv.get(kb.name[6:], 0.0)
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        me = self.ob.evaluated_get(dg).data
        co = np.zeros(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        mw = self.mesh_mw
        return co.reshape(-1, 3).astype(float) @ mw[:3, :3].T + mw[:3, 3]


_CACHE = {}


def pose_mesh(blend_path, params=None, pose=None, backend="blender", ply_out=None):
    """Pose the rigged hand: returns (verts n x 3 mm, faces m x 3).  pose: dict or pose.json path."""
    if isinstance(pose, str):
        pose = json.load(open(pose))
    key = os.path.abspath(blend_path)
    sk = _CACHE.get(key)
    if sk is None or os.path.getmtime(key) != sk._mtime:
        sk = Skinner(key, params)
        sk._mtime = os.path.getmtime(key)
        _CACHE[key] = sk
    elif params_key(load_params(params)) != params_key(sk.params):
        sk.set_params(params)
    V = sk.pose(pose, backend)
    if ply_out:
        write_ply(ply_out, V, sk.F)
    return V, sk.F


# ------------------------------------------------------------------ CLI

def cmd_build(a):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    p = load_params(a.params) if a.params else default_params()
    doc = {"_doc": PARAM_DOC, **p}
    if not a.params:
        json.dump(doc, open(PARAMS_PATH, "w"), indent=1)
        print("WROTE", PARAMS_PATH)
    t0 = time.time()
    build_test_rig(os.path.join(STANDIN, "standin_open.ply"), os.path.join(STANDIN, "joints.json"), TEST_BLEND, p)
    print(f"param rig {time.time() - t0:.1f} s")
    if a.auto:
        t0 = time.time()
        build_test_rig(os.path.join(STANDIN, "standin_open.ply"), os.path.join(STANDIN, "joints.json"), AUTO_BLEND,
                       auto_weights=True)
        print(f"auto rig {time.time() - t0:.1f} s")


def cmd_pose(a):
    V, F = pose_mesh(a.blend, a.params, a.pose, a.backend, a.out)
    print("WROTE", a.out, V.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--params")
    b.add_argument("--auto", action="store_true", help="also build the bone-heat (plain) rig")
    q = sp.add_parser("pose")
    q.add_argument("--blend", default=TEST_BLEND)
    q.add_argument("--params")
    q.add_argument("--pose", required=True)
    q.add_argument("--backend", default="blender", choices=["blender", "numpy"])
    q.add_argument("--out", required=True)
    for name in ("verify", "check", "render"):
        s = sp.add_parser(name)
        s.add_argument("--blend", default=TEST_BLEND)
        s.add_argument("--params")
        s.add_argument("--configs", default="plain,param_lbs,default")
        s.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        cmd_build(a)
    elif a.cmd == "pose":
        cmd_pose(a)
    else:
        import skin_checks
        getattr(skin_checks, "cmd_" + a.cmd)(a)


if __name__ == "__main__":
    main()
