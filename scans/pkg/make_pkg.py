import json, os, sys
import cv2
import numpy as np

ROOT = r"D:\dev\grip_scan"
sys.path.insert(0, ROOT)
from gripscan.board import BoardSpec, make_board
from gripscan.landmarks import Landmarker

VID = os.path.join(ROOT, "scans", "VID20261020144853.mp4")
OUT = os.path.join(ROOT, "scans", "pkg")
STEP = 4
QUAL = int(os.environ.get("QUAL", 92))

spec = BoardSpec.load(os.path.join(ROOT, "data", "board_screen.json"))
board = make_board(spec)
charuco = cv2.aruco.CharucoDetector(board)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))


def charuco_detect(gray):
    corners, ids, _, _ = charuco.detectBoard(gray)
    if ids is None:
        return [], np.zeros((0, 2))
    return ids.ravel().tolist(), corners.reshape(-1, 2)


def detect_board(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ids, px = charuco_detect(gray)
    method = "plain" if len(ids) else None
    if len(ids) < 12:
        eq = clahe.apply(gray)
        for g in (1.5, 2.0, 2.5):
            lut = (((np.arange(256) / 255.0) ** g) * 255).astype(np.uint8)
            ids2, px2 = charuco_detect(cv2.LUT(eq, lut))
            if len(ids2) > len(ids):
                ids, px, method = ids2, px2, f"enhanced_g{g}"
    return ids, px, method


def hand_box(lm, w, h, margin=0.35, min_side=1400):
    x0, y0 = lm.min(0)
    x1, y1 = lm.max(0)
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    side = int(min(max(side, min_side), w, h))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ox = int(np.clip(cx - side / 2, 0, w - side))
    oy = int(np.clip(cy - side / 2, 0, h - side))
    return ox, oy, side


def frames(indices):
    cap = cv2.VideoCapture(VID)
    want = set(indices)
    last = max(indices)
    i = 0
    while i <= last:
        if i in want:
            ok, img = cap.read()
            if not ok:
                break
            yield i, img
        else:
            if not cap.grab():
                break
        i += 1
    cap.release()


def main():
    os.makedirs(os.path.join(OUT, "crops"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "full"), exist_ok=True)
    cap = cv2.VideoCapture(VID)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    lmk = Landmarker(os.path.join(ROOT, "models", "hand_landmarker.task"))

    recs = []
    for i, img in frames(list(range(0, n, STEP))):
        h, w = img.shape[:2]
        ids, px, method = detect_board(img)
        r = {"index": i, "time_s": round(i / fps, 4), "board_method": method,
             "n_corners": len(ids), "corner_ids": ids,
             "corner_px": np.round(px, 2).tolist()}
        res = lmk.detect(img)
        if res is not None:
            lm, label, score = res
            r["landmarks_px"] = np.round(lm, 2).tolist()
            r["mp_handedness_label_raw"] = label
            r["mp_score"] = round(score, 3)
            x0, y0 = np.clip(lm.min(0), 0, [w - 1, h - 1]).astype(int)
            x1, y1 = np.clip(lm.max(0), 0, [w - 1, h - 1]).astype(int)
            g = cv2.cvtColor(img[y0:y1 + 1, x0:x1 + 1], cv2.COLOR_BGR2GRAY)
            r["sharpness"] = round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 2) if g.size else 0.0
        else:
            r["landmarks_px"] = None
            r["sharpness"] = None
        recs.append(r)
        print(i, method, len(ids), "hand" if res is not None else "-", r["sharpness"], flush=True)

    # selection: all frames with >=12 corners + sharpest hand frame per time bin
    sel = {r["index"] for r in recs if r["n_corners"] >= 12 and r["landmarks_px"]}
    hand = [r for r in recs if r["landmarks_px"]]
    target = 120
    nbins = max(target - len(sel), 0)
    if nbins:
        edges = np.linspace(0, n, nbins + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            cand = [r for r in hand if a <= r["index"] < b and r["index"] not in sel]
            if cand:
                sel.add(max(cand, key=lambda r: r["sharpness"])["index"])
    sel = sorted(sel)
    full_set = set(sel[::10])
    by = {r["index"]: r for r in recs}
    for r in recs:
        r["selected"] = r["index"] in sel
    for i, img in frames(sel):
        r = by[i]
        lm = np.array(r["landmarks_px"])
        ox, oy, side = hand_box(lm, W, H)
        name = f"crops/f{i:05d}.jpg"
        cv2.imwrite(os.path.join(OUT, name), img[oy:oy + side, ox:ox + side],
                    [cv2.IMWRITE_JPEG_QUALITY, QUAL])
        r["crop"] = {"file": name, "x0": ox, "y0": oy, "w": side, "h": side}
        if i in full_set:
            name = f"full/f{i:05d}.jpg"
            cv2.imwrite(os.path.join(OUT, name), cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            r["full"] = {"file": name, "scale": 1920 / W}
        print("saved", i, flush=True)

    meta = {"video": os.path.basename(VID), "fps": fps, "n_frames": n, "size": [W, H], "step": STEP,
            "board_spec": json.load(open(os.path.join(ROOT, "data", "board_screen.json"))),
            "crop_jpeg_quality": QUAL, "full_jpeg_quality": 85,
            "notes": "corner_px / landmarks_px are full-resolution pixels; crop px = full px - (x0,y0). "
                     "mp_handedness_label_raw is gripscan's swapped label, unreliable for back camera; ignore.",
            "frames": recs}
    json.dump(meta, open(os.path.join(OUT, "frames.json"), "w"))


if __name__ == "__main__":
    main()
