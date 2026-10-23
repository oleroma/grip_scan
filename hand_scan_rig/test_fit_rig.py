"""Tests for fit_rig.py on the stand-in hands (module A) with synthetic scan defects.

    /tmp/claude-0/hs/venv/bin/python test_fit_rig.py [case ...]     # default: all cases

Each case builds a perturbed copy of a stand-in mesh (rotation, translation, units,
noise, table plane, holes, background blobs, long forearm), runs fit_rig.py on it in
a subprocess, and compares the fitted joints with the stand-in's ground-truth
joints.json mapped through the same transform. Errors are in mm in the stand-in's
true frame, so a wrong scale estimate counts in full.
Writes /mnt/project-files/hand_scan_rig/fit/test_summary.json (updated after every case).
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/tmp/claude-0/hs/venv/bin/python"
STANDIN = "/mnt/project-files/hand_scan_rig/standin"
OUT = "/mnt/project-files/hand_scan_rig/fit"
CASES_DIR = "/tmp/claude-0/hs/work/fit/cases"
NAMES = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_mcp", "index_pip",
         "index_dip", "index_tip", "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
         "ring_mcp", "ring_pip", "ring_dip", "ring_tip", "pinky_mcp", "pinky_pip", "pinky_dip",
         "pinky_tip"]


def load_standin(variant=None):
    d = STANDIN if variant is None else os.path.join(STANDIN, f"variant_{variant}")
    m = trimesh.load(os.path.join(d, "standin_open.ply"), process=False)
    gt = json.load(open(os.path.join(d, "joints.json")))["landmarks"]
    return m, {k: np.array(v, float) for k, v in gt.items()}


def add_noise(m, sigma, rng):
    m = m.copy()
    m.vertices = m.vertices + rng.normal(0, sigma, m.vertices.shape)
    return m


def punch_holes(m, rng, n=8, r=(3.0, 6.0)):
    """Remove the faces around n random surface points (holes of radius r mm)."""
    m = m.copy()
    keep = np.ones(len(m.faces), bool)
    C = m.triangles_center
    centres = []
    for _ in range(n):
        c = m.vertices[rng.integers(len(m.vertices))]
        rr = rng.uniform(*r)
        keep &= np.linalg.norm(C - c, axis=1) > rr
        centres.append((c.round(1).tolist(), round(rr, 1)))
    m.update_faces(keep)
    m.remove_unreferenced_vertices()
    return m, centres


def long_forearm(m, y0=-10.0, k=5.0):
    """Stretch everything proximal of y0 by k (forearm 40 -> ~160 mm long)."""
    m = m.copy()
    V = m.vertices.copy()
    s = V[:, 1] < y0
    V[s, 1] = y0 + (V[s, 1] - y0) * k
    m.vertices = V
    return m


def add_table(m, rng, sink=1.5, size=420.0, step=4.0, sigma=0.3):
    """Hand lying on its back on a table (photogrammetry sees only the table top):
    clip the hand below the plane z = zmin + sink and add a noisy table sheet with
    a cut-out where the hand meets it (open, touching edges, roughly fused)."""
    V = m.vertices
    z0 = V[:, 2].min() + sink
    keep = ~(V[m.faces][:, :, 2] < z0).all(1)
    hand = m.copy()
    hand.update_faces(keep)
    hand.remove_unreferenced_vertices()
    # table grid
    c = V.mean(0)
    xs = np.arange(c[0] - size / 2, c[0] + size / 2 + 1e-9, step)
    ys = np.arange(c[1] - size / 2, c[1] + size / 2 + 1e-9, step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z0)], 1)
    P[:, 2] += rng.normal(0, sigma, len(P))
    nx, ny = len(xs), len(ys)
    idx = np.arange(nx * ny).reshape(nx, ny)
    a, b, cc, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    F = np.concatenate([np.stack([a, b, cc], 1), np.stack([a, cc, d], 1)])
    # cut-out: grid cells under the hand where it touches the plane
    near = V[np.abs(V[:, 2] - z0) < 2.0]
    from scipy.spatial import cKDTree
    t = cKDTree(near[:, :2])
    fc = P[F].mean(1)
    dd, _ = t.query(fc[:, :2])
    F = F[dd > 3.0]
    table = trimesh.Trimesh(P, F, process=False)
    table.remove_unreferenced_vertices()
    return trimesh.util.concatenate([hand, table]), float(z0)


def add_blobs(m, rng, n=4):
    parts = [m]
    c = m.vertices.mean(0)
    for _ in range(n):
        s = trimesh.creation.icosphere(subdivisions=2, radius=rng.uniform(4, 14))
        s.vertices *= rng.uniform(0.6, 1.4, 3)
        d = rng.normal(size=3)
        d[2] = abs(d[2])
        s.vertices += c + d / np.linalg.norm(d) * rng.uniform(150, 220)
        parts.append(s)
    return trimesh.util.concatenate(parts)


def transform(m, gt, R, t, unit_scale):
    m = m.copy()
    m.vertices = (m.vertices @ R.T + t) * unit_scale
    gt2 = {k: (R @ v + t) * unit_scale for k, v in gt.items()}
    return m, gt2


CASES = {
    "base": dict(),
    "variant_1": dict(variant=1),
    "variant_2": dict(variant=2),
    "variant_1_units_mm": dict(variant=1, units="mm"),
    "variant_2_units_mm": dict(variant=2, units="mm"),
    "rotated": dict(rot=True, ext="obj"),
    "metres": dict(rot=True, unit=0.001),
    "noise": dict(noise=0.3),
    "table": dict(table=True, blobs=True),
    "holes": dict(holes=True),
    "long_forearm": dict(forearm=True, rot=True),
    "all_v2": dict(variant=2, rot=True, unit=0.001, noise=0.3, table=True, holes=True, blobs=True,
                   forearm=True, ext="glb"),
    "all_v1": dict(variant=1, rot=True, unit=0.01, noise=0.3, table=True, holes=True, blobs=True,
                   ext="ply"),
}


def make_case(name, spec, seed):
    rng = np.random.default_rng(seed)
    m, gt = load_standin(spec.get("variant"))
    notes = {}
    if spec.get("forearm"):
        m = long_forearm(m)
    if spec.get("holes"):
        m, notes["holes"] = punch_holes(m, rng)
    if spec.get("table"):
        m, notes["table_z"] = add_table(m, rng)
    if spec.get("blobs"):
        m = add_blobs(m, rng)
    if spec.get("noise"):
        m = add_noise(m, spec["noise"], rng)
    R = Rotation.random(random_state=seed).as_matrix() if spec.get("rot") else np.eye(3)
    t = rng.uniform(-300, 300, 3) if spec.get("rot") else np.zeros(3)
    u = spec.get("unit", 1.0)
    m, gt2 = transform(m, gt, R, t, u)
    os.makedirs(CASES_DIR, exist_ok=True)
    path = os.path.join(CASES_DIR, f"{name}.{spec.get('ext', 'ply')}")
    if spec.get("ext") == "glb":
        m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.tile([200, 160, 140, 255], (len(m.vertices), 1)))
    m.export(path)
    return path, gt2, u, notes


def run_case(name, seed=7):
    spec = CASES[name]
    path, gt, u, notes = make_case(name, spec, seed)
    out = os.path.join(OUT, f"test_{name}")
    t0 = time.time()
    p = subprocess.run([PY, os.path.join(HERE, "fit_rig.py"), path, "--out", out, "--name", f"test {name}"]
                       + (["--units", spec["units"]] if spec.get("units") else []),
                       capture_output=True, text=True)
    dt = time.time() - t0
    log = p.stdout[-4000:] + p.stderr[-4000:]
    open(os.path.join(CASES_DIR, f"{name}.log"), "w").write(p.stdout + p.stderr)
    if p.returncode != 0:
        return {"case": name, "ok": False, "error": log[-1500:], "time_s": round(dt, 1)}
    jj = json.load(open(os.path.join(out, "joints.json")))
    k = jj["fit"]["scale_raw_to_mm"]
    err = {}
    for nm in NAMES:
        fit_raw = np.array(jj["landmarks"][nm]) / k          # back to the scan's raw units
        err[nm] = float(np.linalg.norm(fit_raw - gt[nm]) / u)  # mm in the true frame
    e = np.array(list(err.values()))
    return {"case": name, "ok": True, "spec": spec, "time_s": round(dt, 1),
            "median_mm": round(float(np.median(e)), 2), "max_mm": round(float(e.max()), 2),
            "worst": max(err, key=err.get),
            "scale_error_pct": round((k * u - 1.0) * 100, 3) if True else None,
            "seg_rms_mm": jj["fit"]["scale_fit"].get("rms_mm"),
            "mesh": jj["fit"]["mesh"], "rig_check": jj["fit"]["rig_check"],
            "planes_removed": jj["fit"].get("planes_removed"),
            "errors_mm": {kk: round(v, 2) for kk, v in err.items()}, "notes": notes}


def main():
    names = sys.argv[1:] or list(CASES)
    summ_path = os.path.join(OUT, "test_summary.json")
    summ = json.load(open(summ_path)) if os.path.exists(summ_path) else {}
    for nm in names:
        print(f"=== {nm}", flush=True)
        r = run_case(nm)
        summ[nm] = r
        if r["ok"]:
            print(f"    median {r['median_mm']} mm  max {r['max_mm']} mm ({r['worst']})  "
                  f"scale err {r['scale_error_pct']}%  t {r['time_s']} s  mesh {r['mesh']['vertices']} v "
                  f"watertight {r['mesh']['watertight']}", flush=True)
        else:
            print("    FAILED", r["error"][-800:], flush=True)
        os.makedirs(OUT, exist_ok=True)
        json.dump(summ, open(summ_path, "w"), indent=1)


if __name__ == "__main__":
    main()
