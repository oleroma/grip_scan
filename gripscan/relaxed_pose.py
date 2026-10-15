"""Joint angles of the relaxed hand, triangulated from videos shot in front of the board.

    python -m gripscan.relaxed_pose "photos/relaxed/*.mp4" --board-spec data/board_screen.json \
        --flat data/hand_flat.json --out data/hand_relaxed.json

The right hand floats, relaxed and still, in front of the screen board while
the phone moves around it. In every frame the board gives the camera pose and
MediaPipe gives the 2D joints; the 2D joints from many viewpoints are
triangulated into 3D (mm). MediaPipe's own 3D estimate is not used: tested on
flat-hand photos it reported 4-28 degrees of curl where the truth is 0.

The video's lens model is calibrated from the same frames (the board is in all
of them), because video modes crop and scale the sensor differently from photos.
Segment lengths of the triangulated hand are compared with the flat-photo
lengths as an accuracy check.

Angle conventions (degrees):
- flexion: 0 = straight, positive = bent towards the palm. MCP is measured
  against the metacarpal (wrist -> MCP), PIP/DIP against the previous segment,
  all in the finger's bending plane (sideways spread does not leak in).
  Thumb: MCP and IP the same way; CMC is the angle between the thumb and
  index metacarpals.
- spread: angle between neighbouring proximal segments in the palm plane.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from .board import BoardSpec, Camera, Pose, detect, make_board
from .landmarks import DEFAULT_MODEL, FINGERS, MODEL_URL, NAMES, SEGMENTS
from .handmodel import HandModel, degrees, fit, offsets
from .triangulate import baseline_deg, normalize, projection, robust_point

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".3gp"}
MIN_CORNERS = 12
MAX_POSE_RMS = 3.0


# ---------------------------------------------------------------- angles

def unit(v):
    return v / np.linalg.norm(v)


def signed_angle(a, b, axis):
    """Angle from a to b about `axis`, after projecting both onto the plane
    perpendicular to it (so sideways deviation does not count)."""
    axis = unit(axis)
    a = unit(a - (a @ axis) * axis)
    b = unit(b - (b @ axis) * axis)
    ang = np.degrees(np.arccos(np.clip(a @ b, -1, 1)))
    return float(ang if np.cross(a, b) @ axis >= 0 else -ang)


def angles(W):
    wrist, idx_mcp, mid_mcp, pky_mcp = W[0], W[5], W[9], W[17]
    across = unit(idx_mcp - pky_mcp)
    along = unit(mid_mcp - wrist)
    normal = unit(np.cross(along, across))
    out = {}
    for f, (a, b, c, d) in FINGERS.items():
        seg = [W[b] - W[a], W[c] - W[b], W[d] - W[c]]
        if f == "thumb":
            meta = W[b] - W[a]
            ax = unit(np.cross(meta, normal))
            out["thumb_cmc"] = float(np.degrees(np.arccos(np.clip(unit(meta) @ unit(idx_mcp - wrist), -1, 1))))
            out["thumb_mcp"] = signed_angle(seg[0], seg[1], ax)
            out["thumb_ip"] = signed_angle(seg[1], seg[2], ax)
            continue
        meta = W[a] - wrist
        ax = unit(np.cross(normal, unit(meta)))
        out[f"{f}_mcp"] = signed_angle(meta, seg[0], ax)
        out[f"{f}_pip"] = signed_angle(seg[0], seg[1], ax)
        out[f"{f}_dip"] = signed_angle(seg[1], seg[2], ax)
    prox = {f: W[ids[1]] - W[ids[0]] for f, ids in FINGERS.items() if f != "thumb"}
    flat = {f: unit(v - (v @ normal) * normal) for f, v in prox.items()}
    for a, b in [("index", "middle"), ("middle", "ring"), ("ring", "pinky")]:
        out[f"spread_{a}_{b}"] = float(np.degrees(np.arccos(np.clip(flat[a] @ flat[b], -1, 1))))
    return out


def segment_lengths(W):
    return {f: {n: round(float(np.linalg.norm(W[ids[i + 1]] - W[ids[i]])), 2)
                for i, n in enumerate(SEGMENTS[f])} for f, ids in FINGERS.items()}


def make_curl_positive(takes):
    """The palm-normal side depends on handedness; make the dominant curl positive."""
    pip = np.median([t["angles"][k] for t in takes for k in t["angles"] if k.endswith("_pip")])
    if pip < 0:
        for t in takes:
            for k in t["angles"]:
                if not (k.startswith("spread") or k == "thumb_cmc"):
                    t["angles"][k] = -t["angles"][k]


# ---------------------------------------------------------------- frames

def read_frames(path, fps):
    if os.path.splitext(path)[1].lower() not in VIDEO_EXT:
        raise SystemExit(f"{path}: expected a video")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path} (try H.264 / 'most compatible')")
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src / fps)))
    out, i = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            out.append((int(i * 1000 / src), img))
        i += 1
    cap.release()
    return out


def landmarker(model_path):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    model_path = os.path.abspath(model_path or DEFAULT_MODEL)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MediaPipe model not found at {model_path}; download {MODEL_URL}")
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
    return vision.HandLandmarker.create_from_options(opts)


def calibrate_from_frames(dets, size, max_views=40):
    views = [d for d in dets if d is not None and len(d[0]) >= 20]
    if len(views) < 8:
        return None, None
    pick = np.linspace(0, len(views) - 1, min(max_views, len(views))).astype(int)
    objs = [views[i][0].astype(np.float32) for i in pick]
    imgs = [views[i][1].astype(np.float32) for i in pick]
    flags = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None, flags=flags)
    return Camera(K, dist, size), float(rms)


def pose_of(det, cam):
    obj, img = det
    ok, rvec, tvec = cv2.solvePnP(obj, img, cam.K, cam.dist)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, cam.K, cam.dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    return Pose(rvec, tvec, rms, len(obj))


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("videos", nargs="+")
    p.add_argument("--board-spec", required=True)
    p.add_argument("--flat", required=True, help="hand_flat.json: lengths and palm layout of the model")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--model", default=None)
    p.add_argument("--out", default="data/hand_relaxed.json")
    a = p.parse_args()
    paths = [f for g in a.videos for f in (sorted(glob.glob(g)) or [g])]
    spec = BoardSpec.load(a.board_spec)
    board = make_board(spec)
    flat_json = json.load(open(a.flat))
    model = HandModel(flat_json["per_photo"][0]["landmarks_board_mm"])

    # pass 1: frames, board corners, 2D joints
    import mediapipe as mp
    raw = []
    for path in paths:
        frames = read_frames(path, a.fps)
        lm = landmarker(a.model)
        recs = []
        for ts, img in frames:
            det = detect(img, board)
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                               data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), ts)
            h, w = img.shape[:2]
            px = (np.array([[q.x * w, q.y * h] for q in res.hand_landmarks[0]])
                  if res.hand_landmarks else None)
            recs.append({"det": det, "px": px})
        lm.close()
        size = (frames[0][1].shape[1], frames[0][1].shape[0]) if frames else None
        raw.append({"path": path, "size": size, "recs": recs})
        print(f"{os.path.basename(path)}: {len(recs)} frames, board in "
              f"{sum(r['det'] is not None for r in recs)}, hand in {sum(r['px'] is not None for r in recs)}")

    # lens model of the video mode, from all takes with the same frame size
    sizes = {t["size"] for t in raw if t["size"]}
    cams = {}
    for size in sizes:
        dets = [r["det"] for t in raw if t["size"] == size for r in t["recs"]]
        cam, rms = calibrate_from_frames(dets, size)
        if cam is None:
            raise SystemExit(f"not enough clear board views to calibrate the {size} video mode")
        cams[size] = cam
        print(f"video lens {size[0]}x{size[1]}: f={cam.K[0, 0]:.0f}px, calibration RMS {rms:.2f}px")

    # pass 2: poses and triangulation per take
    takes = []
    for t in raw:
        cam = cams.get(t["size"])
        Ps, poses, xs = [], [], []
        for r in t["recs"]:
            if r["det"] is None or r["px"] is None or len(r["det"][0]) < MIN_CORNERS:
                continue
            pose = pose_of(r["det"], cam)
            if pose is None or pose.reproj_px > MAX_POSE_RMS:
                continue
            Ps.append(projection(cam, pose))
            poses.append(pose)
            xs.append(normalize(r["px"], cam))
        if len(Ps) < 5:
            print(f"{os.path.basename(t['path'])}: only {len(Ps)} usable frames, skipped")
            continue
        f = float(cam.K[0, 0])
        W, resid, used, base = np.zeros((21, 3)), [], [], []
        for j in range(21):
            X, keep, rms = robust_point(Ps, [x[j] for x in xs], f)
            W[j] = X
            resid.append(rms)
            used.append(int(keep.sum()))
            base.append(baseline_deg(poses, keep))
        v, err, Xfit, sway = fit(model, Ps, xs, f, W[[0, 1, 5, 9, 13, 17]])
        joint_err = np.median(err, axis=0)
        takes.append({
            "input": t["path"], "frames_used": len(Ps),
            "model_angles_deg": degrees(v),
            "model_offsets_mm": offsets(v),
            "sway_mm": {"median": round(float(np.median(np.linalg.norm(sway, axis=1))), 1),
                        "max": round(float(np.max(np.linalg.norm(sway, axis=1))), 1)},
            "model_fit_px": {"median": round(float(np.median(err)), 1),
                             "per_joint_median": {n: round(float(e), 1) for n, e in zip(NAMES, joint_err)}},
            "model_landmarks_board_mm": {n: [round(float(c), 2) for c in Xfit[i]] for i, n in enumerate(NAMES)},
            "baseline_deg_min": round(min(base), 1),
            "reprojection_px": {n: round(r, 1) for n, r in zip(NAMES, resid)},
            "views_per_joint_min": min(used),
            "landmarks_board_mm": {n: [round(float(v), 2) for v in W[i]] for i, n in enumerate(NAMES)},
            "segments_mm": segment_lengths(W),
            "angles": angles(W),
        })
        print(f"{os.path.basename(t['path'])}: model fit median {np.median(err):.1f}px "
              f"(free triangulation {np.median(resid):.1f}px)")
        print(f"{os.path.basename(t['path'])}: {len(Ps)} views, widest baseline >= {min(base):.0f} deg, "
              f"reprojection median {np.median(resid):.1f}px, worst {max(resid):.1f}px ({NAMES[int(np.argmax(resid))]})")
    if not takes:
        raise SystemExit("no usable takes")
    make_curl_positive(takes)

    joints = {}
    for k in takes[0]["model_angles_deg"]:
        vals = [t["model_angles_deg"][k] for t in takes]
        joints[k] = {"deg": round(float(np.median(vals)), 1),
                     "range_between_takes": round(float(np.ptp(vals)), 1),
                     "per_take": [round(v, 1) for v in vals]}
    free = {}
    for k in takes[0]["angles"]:
        vals = [t["angles"][k] for t in takes]
        free[k] = {"deg": round(float(np.median(vals)), 1), "per_take": [round(v, 1) for v in vals]}

    check = None
    if a.flat and os.path.exists(a.flat):
        flat = json.load(open(a.flat))["fingers"]
        check = {}
        for f, segs in flat.items():
            for s, v in segs.items():
                tri = float(np.median([t["segments_mm"][f][s] for t in takes]))
                check[f"{f}_{s}"] = {"flat": v["mean"], "triangulated": round(tri, 2),
                                     "diff": round(tri - v["mean"], 2)}

    result = {
        "units": "degrees (angles), mm (positions)",
        "method": ("board pose per video frame + MediaPipe 2D joints; articulated hand model "
                   "(lengths and palm from the flat photos) fitted robustly to all views"),
        "joints": joints,
        "free_triangulation_angles": free,
        "length_check_vs_flat": check,
        "takes": takes,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print()
    for k, v in joints.items():
        print(f"{k:22s} {v['deg']:7.1f}   between takes {v['range_between_takes']:5.1f}   {v['per_take']}")
    if check:
        d = np.array([c["diff"] for c in check.values()])
        print(f"\nlength check vs flat photos: median |diff| {np.median(np.abs(d)):.1f} mm, "
              f"worst {d[np.argmax(np.abs(d))]:+.1f} mm "
              f"({list(check)[int(np.argmax(np.abs(d)))]})")
    print("->", a.out)


if __name__ == "__main__":
    main()
