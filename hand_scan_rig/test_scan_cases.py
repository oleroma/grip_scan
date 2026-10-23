"""Realistic photogrammetry test cases for fit_rig.py (from the review of 2026-10-21).

    /tmp/claude-0/hs/venv/bin/python test_scan_cases.py make [case ...]   # build the case meshes
    /tmp/claude-0/hs/venv/bin/python test_scan_cases.py run  [case ...]   # fit + grade (default: all)

What a COLMAP / OpenMVS mesh of a palm-up open hand on a table looks like, built from
the stand-in SDF (standin.py) so the true joints are known:
  - table fused to the hand (smooth union with the half-space below the table plane;
    only the top surface exists, so the back of the hand is MISSING), table noise and
    bowl, hand pressed in deeper, finger backs on a tilted table,
  - forearm running off the reconstructed area (open at the grid edge),
  - fingers touching / nearly touching, thumb adducted, back of the hand deleted,
  - textured OBJ (OpenMVS style: positions once, one 'vt' per corner) and textured GLB,
  - arbitrary scale (x0.0237) and rotation.
Cases go to /tmp/claude-0/hs/work/fit/scan_cases/, fits to .../scan_cases/out/<case>/,
the graded summary (joint errors in mm in the true frame, mesh genus, warnings) to
/mnt/project-files/hand_scan_rig/fit/scan_cases_summary.json.
Targets: median < 2 mm, max < 4 mm, output skin genus 0 (Euler 2), or a clear failure
with advice for inputs that cannot be fitted.
"""
import copy
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import standin as SI  # noqa: E402

PY = "/tmp/claude-0/hs/venv/bin/python"
OUT = "/tmp/claude-0/hs/work/fit/scan_cases"
SUMMARY = "/mnt/project-files/hand_scan_rig/fit/scan_cases_summary.json"
STANDIN = "/mnt/project-files/hand_scan_rig/standin"
os.makedirs(OUT, exist_ok=True)


def make_textured(name, spec):
    """Stand-in with ~200 UV charts and a texture, vertices split per chart.
    spec['textured'] = 'glb' or 'obj_shared' (OpenMVS: 'v' once, per-corner 'vt')."""
    from PIL import Image
    m = trimesh.load(os.path.join(STANDIN, "standin_open.ply"), process=False)
    gt = json.load(open(os.path.join(STANDIN, "joints.json")))["landmarks"]
    rng = np.random.default_rng(5)
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    N = m.face_normals
    ax = np.argmax(np.abs(N), 1) * 2 + (N[np.arange(len(N)), np.argmax(np.abs(N), 1)] > 0)
    cell = np.floor(V[F].mean(1) / 25.0).astype(int)
    key = ax * 1000000 + (cell[:, 0] + 50) * 10000 + (cell[:, 1] + 50) * 100 + (cell[:, 2] + 50)
    _, chart = np.unique(key, return_inverse=True)
    chart = chart.ravel()
    nch = chart.max() + 1
    pairs = chart[:, None] * len(V) + F
    uniq, inv = np.unique(pairs.ravel(), return_inverse=True)
    V2 = V[uniq % len(V)]
    F2 = inv.reshape(-1, 3)
    uv = np.zeros((len(V2), 2))
    vch = uniq // len(V)
    g = int(np.ceil(np.sqrt(nch)))
    for c in range(nch):
        s = vch == c
        a = (np.unique(ax[chart == c])[0]) // 2
        P = V2[s][:, [i for i in range(3) if i != a]]
        P = (P - P.min(0)) / max(np.ptp(P, 0).max(), 1e-9)
        uv[s] = (P * 0.9 + 0.05 + [c % g, c // g]) / g
    R = Rotation.random(random_state=3).as_matrix()
    t = rng.uniform(-300, 300, 3)
    u = 0.0237
    V2 = (V2 @ R.T + t) * u
    img = Image.fromarray((rng.uniform(120, 220, (256, 256, 3))).astype(np.uint8))
    tm = trimesh.Trimesh(V2, F2, process=False, visual=trimesh.visual.TextureVisuals(uv=uv, image=img))
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    if spec["textured"] == "glb":
        tm.export(os.path.join(OUT, name + ".glb"))
    else:
        Vu, inv2 = np.unique(np.round(V2, 9), axis=0, return_inverse=True)
        inv2 = inv2.ravel()
        img.save(os.path.join(d, "material_0.png"))
        with open(os.path.join(d, name + ".obj"), "w") as f:
            f.write(f"mtllib {name}.mtl\n")
            for p in Vu:
                f.write("v %.9f %.9f %.9f\n" % tuple(p))
            for q in uv:
                f.write("vt %.6f %.6f\n" % tuple(q))
            f.write("usemtl material_0\n")
            for tri in F2:
                a, b, c = inv2[tri]
                f.write(f"f {a + 1}/{tri[0] + 1} {b + 1}/{tri[1] + 1} {c + 1}/{tri[2] + 1}\n")
        open(os.path.join(d, name + ".mtl"), "w").write("newmtl material_0\nmap_Kd material_0.png\n")
    gtj = {"landmarks": {k: ((R @ np.array(v) + t) * u).tolist() for k, v in gt.items()}, "unit": u,
           "pose": None, "spec": spec}
    json.dump(gtj, open(os.path.join(OUT, f"{name}_gt.json"), "w"))
    print(name, "charts", nch, "verts", len(V2), flush=True)


def pose(**kw):
    p = copy.deepcopy(SI.OPEN_POSE)
    for k, v in kw.items():
        p[k] = v
    return p


def build(spec, res=0.7):
    lay = SI.palm_layout(SI.read_rig())
    J, tbend = SI.build_joints(lay, spec.get("pose", SI.OPEN_POSE))
    prims = SI.build_prims(J, tbend, None, 1.0)
    if spec.get("long_arm"):
        prims.append(SI.round_cone([0, -40, 0], [0, -260, 0], 30.5, 26.0, np.array([0, 0, 1.0]),
                                   squash=0.62, k=16, name="arm2"))
    pts = np.array(list(J.values()))
    cut_y = spec.get("cut_y", -40.0)
    ylo = cut_y - 3 if not spec.get("long_arm") else -150.0
    marg = spec.get("table_margin", 0.0)
    lo = np.floor(np.minimum(pts.min(0) - 26 - [marg, 0, 0], [-45 - marg, ylo, -30]))
    hi = np.ceil(np.maximum(pts.max(0) + 26 + [marg, marg, 0], [45 + marg, 0, 35]))
    if spec.get("long_arm"):
        lo[1] = ylo
    xs, ys, zs = (np.arange(lo[i], hi[i] + res, res) for i in range(3))
    vol = np.empty((len(xs), len(ys), len(zs)), np.float32)
    XX, ZZ = np.meshgrid(xs, zs, indexing="ij")
    tab = spec.get("table")
    z0 = None
    if tab:
        z0 = -20.0 + tab["sink"]      # stand-in dorsal skin min is about z = -20
    step = max(1, int(6 / res))
    cy = cut_y if not spec.get("long_arm") else -1e6
    for j0 in range(0, len(ys), step):
        yy = ys[j0:j0 + step]
        Pc = np.stack(np.broadcast_arrays(XX[:, None, :], yy[None, :, None], ZZ[:, None, :]), -1)
        sh = Pc.shape[:3]
        P = Pc.reshape(-1, 3)
        d = SI.eval_sdf(P, prims, cy)
        if tab:
            zt = z0 + tab.get("tilt", 0.0) * P[:, 1] + tab.get("warp", 0.0) * ((P[:, 0] / 150.0) ** 2 + ((P[:, 1] - 60) / 150.0) ** 2)
            d = SI.smin(d, P[:, 2] - zt, tab.get("fillet", 2.0))
        vol[:, j0:j0 + step, :] = d.reshape(sh)
    from skimage.measure import marching_cubes
    V, F, _, _ = marching_cubes(vol, 0.0, spacing=(res, res, res))
    V += np.array([xs[0], ys[0], zs[0]])
    m = trimesh.Trimesh(V, F, process=False)
    return m, J, z0


def make(name, spec, seed=11):
    if spec.get("textured"):
        return make_textured(name, spec)
    t0 = time.time()
    rng = np.random.default_rng(seed)
    m, J, z0 = build(spec)
    V = m.vertices.copy()
    F = m.faces.copy()
    if spec.get("delete_back"):          # back of hand missing, no table (holes)
        Fc = V[F].mean(1)
        N = m.face_normals
        keep = ~((N[:, 2] < -0.3) & (Fc[:, 2] < spec["delete_back"]))
        F = F[keep]
    if spec.get("noise"):
        V = V + rng.normal(0, spec["noise"], V.shape)
    if spec.get("table_noise") and z0 is not None:
        flat = np.abs(V[:, 2] - z0) < 1.0
        V[flat, 2] += rng.normal(0, spec["table_noise"], flat.sum())
    m = trimesh.Trimesh(V, F, process=False)
    m.remove_unreferenced_vertices()
    R = Rotation.random(random_state=seed).as_matrix() if spec.get("rot", True) else np.eye(3)
    t = rng.uniform(-3, 3, 3) * 100
    u = spec.get("unit", 1.0)
    m.vertices = (m.vertices @ R.T + t) * u
    gt = {k: ((R @ v + t) * u).tolist() for k, v in J.items()}
    ext = spec.get("ext", "ply")
    path = os.path.join(OUT, f"{name}.{ext}")
    col = np.tile([205, 160, 140, 255], (len(m.vertices), 1))
    m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=col)
    m.export(path)
    json.dump({"landmarks": gt, "unit": u, "z0": z0, "spec": {k: v for k, v in spec.items() if k != "pose"},
               "pose": spec.get("pose")}, open(os.path.join(OUT, f"{name}_gt.json"), "w"), indent=1)
    print(name, "verts", len(m.vertices), "faces", len(m.faces), "t", round(time.time() - t0, 1), flush=True)


TABLE = dict(sink=3.0, fillet=2.0)
TOGETHER = pose(fan={"index": 2.5, "middle": 0.8, "ring": -0.8, "pinky": -2.5}, thumb_cmc=18.0,
                thumb_elev=5.0)
CASES = {
    # fused table, back of the hand below the table plane missing, long arm off the edge
    "fused_table": dict(table=TABLE, long_arm=True, table_margin=40, noise=0.3, unit=0.0237),
    # same + fingers touching side by side + thumb adducted next to the index
    "fused_together": dict(table=TABLE, long_arm=True, table_margin=40, noise=0.3, unit=0.0237,
                           pose=TOGETHER),
    # textureless, noisy, slightly bowled table (typical COLMAP)
    "fused_noisytable": dict(table=dict(sink=3.0, fillet=2.0, warp=2.0), long_arm=True, table_margin=40,
                             noise=0.3, table_noise=0.8, unit=0.0237),
    # hand pressed deeper: more of the back missing
    "fused_deep": dict(table=dict(sink=8.0, fillet=3.0), long_arm=True, table_margin=40, noise=0.3,
                       unit=0.0237),
    "together_mild": dict(pose=pose(fan={"index": 5.0, "middle": 1.7, "ring": -1.7, "pinky": -5.0},
                                    thumb_cmc=25.0, thumb_elev=8.0), unit=0.0237),
    "thumb_adducted": dict(pose=pose(thumb_cmc=18.0, thumb_elev=5.0), unit=0.0237),
    # palm-up hand flat on the table: finger backs fused with the table as well
    "fused_fingers_down": dict(table=dict(sink=3.0, fillet=2.0, tilt=0.06), long_arm=True, table_margin=40,
                               noise=0.3, unit=0.0237),
    "tnoise08": dict(table=TABLE, long_arm=True, table_margin=40, noise=0.3, table_noise=0.8, unit=0.0237),
    "tnoise05": dict(table=TABLE, long_arm=True, table_margin=40, noise=0.3, table_noise=0.5, unit=0.0237),
    "twarp2": dict(table=dict(sink=3.0, fillet=2.0, warp=2.0), long_arm=True, table_margin=40, noise=0.3,
                   unit=0.0237),
    # fingers together only (no table)
    "together_only": dict(pose=TOGETHER, unit=0.0237),
    # wrist cut short (just proximal of the wrist joint)
    "short_cut": dict(cut_y=-6.0, unit=0.0237),
    # back of the hand missing (hole), no table
    "no_back": dict(delete_back=-8.0, noise=0.3, unit=0.0237),
    # relaxed curl palm up
    "relaxed_curl_table": dict(table=TABLE, long_arm=True, table_margin=40, noise=0.3, unit=0.0237,
                               pose=pose(mcp_flex=20.0, pip_flex=25.0, dip_flex=12.0)),
}


CASES.update({
    "textured_obj": dict(textured="obj_shared"),
    "textured_glb": dict(textured="glb"),
})


def case_path(name):
    c = [f for f in glob.glob(os.path.join(OUT, name + ".*")) + glob.glob(os.path.join(OUT, name, name + ".obj"))
         if not f.endswith(".json")]
    return c[0] if c else None


def run(name, extra=()):
    path = case_path(name)
    if path is None:
        make(name, CASES[name])
        path = case_path(name)
    gt = json.load(open(os.path.join(OUT, name + "_gt.json")))
    out = os.path.join(OUT, "out", name)
    t0 = time.time()
    p = subprocess.run([PY, os.path.join(HERE, "fit_rig.py"), path, "--out", out, "--name", name] + list(extra),
                       capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="4"))
    dt = time.time() - t0
    os.makedirs(os.path.join(OUT, "out"), exist_ok=True)
    open(os.path.join(OUT, "out", name + ".log"), "w").write(p.stdout + p.stderr)
    res = {"case": name, "ok": p.returncode == 0, "time_s": round(dt, 1)}
    if p.returncode != 0:
        err = [ln for ln in (p.stdout + p.stderr).splitlines() if "Error" in ln]
        res["error"] = err[-1] if err else (p.stdout + p.stderr)[-600:]
    else:
        jj = json.load(open(os.path.join(out, "joints.json")))
        k = jj["fit"]["scale_raw_to_mm"]
        u = gt["unit"]
        err = {nm: float(np.linalg.norm(np.array(jj["landmarks"][nm]) / k - np.array(v)) / u)
               for nm, v in gt["landmarks"].items()}
        e = np.array(list(err.values()))
        res.update(median_mm=round(float(np.median(e)), 2), max_mm=round(float(e.max()), 2),
                   worst=max(err, key=err.get), scale_err_pct=round((k * u - 1) * 100, 2),
                   euler=jj["fit"]["mesh"]["euler"], volume_ml=jj["fit"]["mesh"]["volume_ml"],
                   tip_test=jj["fit"].get("tip_test"), warnings=jj["fit"].get("warnings", []),
                   planes=[{kk: pl[kk] for kk in ("area_frac", "rule", "plan_cover")}
                           for pl in jj["fit"].get("planes_removed", [])],
                   filled_area_frac=jj["fit"].get("rebuild", {}).get("filled_area_frac"),
                   wrist_from=jj["fit"].get("wrist_from"),
                   errors_mm={a: round(b, 2) for a, b in err.items()})
        res["pass"] = bool(res["median_mm"] < 2 and res["max_mm"] < 4 and res["euler"] == 2)
    S = json.load(open(SUMMARY)) if os.path.exists(SUMMARY) else {}
    S[name] = res
    json.dump(S, open(SUMMARY, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("errors_mm", "planes")}), flush=True)
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    names = sys.argv[2:] or list(CASES)
    for n in names:
        if cmd == "make":
            make(n, CASES[n])
        else:
            run(n)
