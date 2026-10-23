"""Hold pose of the fitted (scan) hand with an UPRIGHT palm, in the web hand's pose.

    P=/tmp/claude-0/hs/venv/bin/python
    $P hold_upright.py ref                  # web hand pose -> WORK/web_landmarks.json (+ its skin for the sheet)
    $P hold_upright.py start                # web pose transferred onto the scan rig -> WORK/z_start.npy
    $P hold_upright.py solve [--w J] [--start S] [--nfev N] [--tag T]   # -> WORK/solve_<tag>.json
    $P hold_upright.py export --tag T       # .blend, pose.json, contacts.json, metrics.json -> OUT
    $P hold_upright.py render --tag T ; $P hold_upright.py sheet          # hold_upright_sheet.jpg

Why.  Olex (2026-10-22): in hold_v2 (hold_solve.py, fitted_hold/) "the hand rolls too much inward,
palm planes should be almost perpendicular like we holding a box if they were open", and the pose
of the MakeHuman web hand (web_hand_rig/, solved with a flat palm and a vertical palm plane) is
better, but its surface lacks the thenar eminence.  So the scan hand gets the web hand's pose.

Transfer.  The web hand's bones were mapped onto the same measured skeleton (fitted/joints.json,
bone lengths identical), so its posed bone heads / tails ARE posed joint centres of our skeleton:
wrist = wrist.R head, <finger>_mcp/pip/dip = finger<k>-1/2/3.R heads, tip = finger<k>-3.R tail,
thumb_cmc/mcp/ip/tip = finger1-1/2/3.R heads + tail.  Our rig is fitted to them by angles taken from
world-space joint positions (skin.measurements, the SPEC definitions) and then a joint least
squares (run_all.refine_on_joints), never by copying Euler numbers between different rest frames.

Solve.  hold_solve.Objective with the web joints as the reference hold (S.H / S.ref), tightened
reference sigmas (ref_scale), plus two terms: 'upright' (n_z of the palm-side skin plane, the
web_hand_rig definition, hold_solve.palm_plane) and 'arch' (palm arch angles held at the open-hand
values: flat palm, no cupping).  Contacts as hold_v2: palm on the swell, index on R1's surface point,
middle on R2, ring on R4, pinky pad on the grip, <= 0.4 mm free squash into the plastic.
Skin parameters = hold_v2's (fitted_hold/skin_params_hold_v2.json).
"""
import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hold_solve as hs  # noqa: E402

WEB_BLEND = "/mnt/project-files/web_hand_rig/hand_on_grip_web.blend"
WEB_METRICS = "/mnt/project-files/web_hand_rig/hold_metrics.json"
V2_DIR = "/mnt/project-files/hand_scan_rig/fitted_hold"
V2_SOLVE = "/tmp/claude-0/hs/work/hold_v2/solve_v2.json"
PARAMS = os.path.join(V2_DIR, "skin_params_hold_v2.json")
OUT = "/mnt/project-files/hand_scan_rig/hold_upright"
WORK = "/tmp/claude-0/hs/work/hold_up"
BLEND_NAME = "hand_on_grip_upright.blend"
SHEET_NAME = "hold_upright_sheet.jpg"
# default weights of the upright solve (hold_solve.Objective keys)
W_DEFAULT = dict(upright=40.0, arch=3.0, arch_sig=1.5, ref_scale=0.5, thumb=3.0, swell=6.0, flips=6.0)

log = hs.log
jdump = hs.jdump


def web_landmarks_from_blend(path=WEB_BLEND):
    """Posed joint centres of the web hand (world mm, the grip frame of hold_rig.blend) and its
    evaluated skin (world)."""
    import bpy
    bpy.ops.wm.open_mainfile(filepath=path)
    a = bpy.data.objects["hand rig"]
    mw = np.array(a.matrix_world)

    def w(p):
        return (mw @ np.r_[np.array(p), 1.0])[:3]
    pb = a.pose.bones
    H = {"wrist": w(pb["wrist.R"].head)}
    for k, f in enumerate(("thumb", "index", "middle", "ring", "pinky")):
        names = ("cmc", "mcp", "ip") if f == "thumb" else ("mcp", "pip", "dip")
        for j, nm in enumerate(names):
            H[f"{f}_{nm}"] = w(pb[f"finger{k + 1}-{j + 1}.R"].head)
        H[f"{f}_tip"] = w(pb[f"finger{k + 1}-3.R"].tail)
    dg = bpy.context.evaluated_depsgraph_get()
    ob = bpy.data.objects["hand"]
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    M = np.array(ob.matrix_world)
    V = np.array([M[:3, :3] @ np.array(v.co) + M[:3, 3] for v in me.vertices])
    F = []
    for p in me.polygons:
        vs = list(p.vertices)
        F += [[vs[0], vs[i], vs[i + 1]] for i in range(1, len(vs) - 1)]
    ev.to_mesh_clear()
    return H, V, np.array(F)


def cmd_ref(a):
    os.makedirs(WORK, exist_ok=True)
    H, V, F = web_landmarks_from_blend()
    jdump({k: v for k, v in H.items()}, os.path.join(WORK, "web_landmarks.json"))
    np.savez_compressed(os.path.join(WORK, "web_skin.npz"), V=V, F=F)
    log("web landmarks", len(H), "skin", V.shape, F.shape)


def load_web():
    return {k: np.array(v, float) for k, v in json.load(open(os.path.join(WORK, "web_landmarks.json"))).items()}


def scene():
    """hold_solve Scene with hold_v2's skin params and placement frame, reference = the web pose."""
    import skin
    S, z_old = hs.setup(PARAMS)
    S.H_cyl, S.ref_cyl = S.H, S.ref
    S.H = load_web()
    S.ref = skin.measurements(S.H)
    return S, z_old


def z_v2():
    return np.array(json.load(open(V2_SOLVE))["z"])


def transfer(S):
    """Web joints -> rig DOFs + placement (z, rotation relative to S.R0)."""
    from scipy.spatial.transform import Rotation
    from run_all import kabsch, refine_on_joints
    from skin import MEAS
    pkeys = list(MEAS) + ["thumb_cmc_azimuth"]
    pose = {"angles": {k: float(S.ref[k]) for k in pkeys}}
    Hs = np.array([S.H[n] for n in S.names])
    M0, P0, info = S.rig.solve(pose)
    R0, t0 = kabsch(np.array([P0[n] for n in S.names]), Hs)
    x, R, t = refine_on_joints(S.rig, info["x"], R0, t0, Hs, S.names, 0.05)
    rv = Rotation.from_matrix(R @ S.R0.T).as_rotvec()
    # refine_on_joints places armature points as R p + t; hold_solve.forward does the same with
    # R = rot(rotvec) @ R0, so t carries over unchanged
    return np.r_[x, rv, t]


def joint_err(S, z):
    o = S.forward(z, verts=False)
    return {n: round(float(np.linalg.norm(o["P"][n] - S.H[n])), 2) for n in S.names}


def cmd_start(a):
    S, _ = scene()
    z = transfer(S)
    np.save(os.path.join(WORK, "z_start.npy"), z)
    obj = hs.Objective(S, z, W_DEFAULT)
    je = joint_err(S, z)
    log("start: joint error vs web (mm) max", max(je.values()), je)
    log("start:", json.dumps(hs.summary(obj, z)))
    log("start palm plane", palm_report(S, z))


def palm_report(S, z):
    o = S.forward(z)
    n, tilt, plan = hs.palm_plane(S, o)
    import skin
    m = skin.measurements(o["P"])
    return {"palm_normal": n.round(3).tolist(), "palm_tilt_from_vertical_deg": round(tilt, 2),
            "palm_faces_deg_from_minusX_in_plan": round(plan, 2),
            "arch_deg": {"index_middle_ring (arch_imr)": round(float(m["arch_imr"]), 2),
                         "middle_ring_pinky (arch_mrp)": round(float(m["arch_mrp"]), 2)},
            **hs.half_palm_rolls(o["P"])}


def cmd_solve(a):
    S, _ = scene()
    z_start = np.load(os.path.join(WORK, "z_start.npy"))
    w = dict(W_DEFAULT)
    w.update(json.loads(a.w) if a.w else {})
    obj = hs.Objective(S, z_start, w)
    start = np.array(json.load(open(a.start))["z"]) if a.start else z_start
    log("weights", w)
    log("start", json.dumps(hs.summary(obj, start)), palm_report(S, start))
    t0 = time.time()
    sol = hs.solve(obj, start, max_nfev=a.nfev, workers=a.workers)
    s = hs.summary(obj, sol.x)
    s["palm"] = palm_report(S, sol.x)
    log(f"done {time.time() - t0:.0f} s, nfev {sol.nfev}:", json.dumps(s))
    jdump({"z": sol.x, "R0": S.R0, "summary": s, "params": PARAMS, "weights": w, "nfev": sol.nfev,
           "start": a.start or "z_start.npy"}, os.path.join(WORK, f"solve_{a.tag}.json"))


# ------------------------------------------------------------------ export

def cmd_blend(a):
    import skin
    from run_all import write_hold_blend
    d = json.load(open(a.solve))
    sk = skin.Skinner(hs.FIT_BLEND, d["params"])
    sk.arm.data.bones["palm"].use_deform = True
    sk.apply_to_blend()
    x, tw, R, t = hs.unpack(d["z"], np.array(d["R0"]))
    M, P = sk.rig.fk(x, tw)
    pose = hs.pose_json(sk.rig, d["z"], np.array(d["R0"]), "hold_upright.py (scan hand, web hand's pose, upright palm)")
    Vb, Vn, check = write_hold_blend(sk, M, R, t, pose["angles"], hs.HOLD_RIG, a.out_blend)
    jdump({"live_check": check}, os.path.join(WORK, "blend_check.json"))
    log("blend written", a.out_blend, "blender vs numpy", check["blender_vs_numpy_same_bulge_mm"])


def cmd_export(a):
    import subprocess
    os.makedirs(OUT, exist_ok=True)
    sol = os.path.join(WORK, f"solve_{a.tag}.json")
    blend = os.path.join(OUT, BLEND_NAME)
    py = sys.executable
    subprocess.run([py, os.path.join(HERE, "hold_upright.py"), "blend", "--solve", sol, "--out-blend", blend], check=True)
    subprocess.run([py, os.path.join(HERE, "hold_solve.py"), "contacts", "--blend", blend,
                    "--out-json", os.path.join(OUT, "contacts.json")], check=True)
    subprocess.run([py, os.path.join(HERE, "hold_upright.py"), "metrics", "--tag", a.tag], check=True)


def angle_table(S, z):
    """SPEC angles (deg) of a pose vs the web pose (S.ref), per joint."""
    import skin
    o = S.forward(z, verts=False)
    m = skin.measurements(o["P"])
    keys = [k for k in skin.MEAS if k != "thumb_cmc"] + ["thumb_cmc_azimuth"]
    g = (lambda d, k: d["_s" + k] if k.startswith("spread") else d[k])
    return {k: {"web": round(float(g(S.ref, k)), 1), "scan_hand": round(float(g(m, k)), 1),
                "diff": round(float(g(m, k) - g(S.ref, k)), 1)} for k in keys}


def cmd_metrics(a):
    from skin import contact_labels, resolve_contacts
    d = json.load(open(os.path.join(WORK, f"solve_{a.tag}.json")))
    S, _ = scene()
    rows = {}
    for which, z, cpath in (("before", z_v2(), os.path.join(V2_DIR, "contacts.json")),
                            ("after", np.array(d["z"]), os.path.join(OUT, "contacts.json"))):
        con = json.load(open(cpath))["new_scan_hand"]
        o = S.forward(z)
        fl = hs.flip_report(S, z)
        _, cinf = resolve_contacts(o["V"], S.F, S.sk.V0, S.sk.cs.A, labels=contact_labels(S.sk.W), iters=0)
        sd = S.grip_sd(o["V"])
        je = joint_err(S, z)
        rows[which] = {
            **palm_report(S, z),
            "palm_swell_gap_mm": con["palm_swell_gap_mm"],
            "max_penetration_into_grip_mm": round(con["penetration_max_mm"], 2),
            "penetration_p95_mm": round(con["penetration_p95_mm"], 2),
            "penetrating_area_cm2": round(con["penetrating_area_cm2"], 2),
            "deepest_at": con["deepest_at"],
            "penetration_max_by_part_mm": con["penetration_max_by_part_mm"],
            "fingertip_to_button_mm": {"R1_index (marker centre, 4.6 mm inside the plastic)": con["R1_index_finger_mm"],
                                       **hs.r1_surface_gap(S, o["V"]),
                                       "R2_middle": con["R2_middle_fingertip_mm"],
                                       "R4_ring": con["R4_ring_fingertip_mm"]},
            "pinky_pad_gap_to_grip_mm": round(float(sd[S.sel["pinky_distal"]].min()), 2),
            "thumb_tip_gap_to_grip_mm": round(float(sd[S.sel["thumb_distal"]].min()), 2),
            "flipped_triangles": fl["flipped_faces"],
            "flipped_by_part": fl["flipped_by_part"],
            "faces_turned_over_60deg": fl["faces_turned_over_60deg"],
            "skin_into_skin_max_mm": round(cinf["contact_depth_max_mm"], 2),
            "angles_vs_web_pose_deg": angle_table(S, z),
            "joint_distance_from_web_pose_mm": je,
            "joint_distance_from_web_pose_max_mm": max(je.values()),
        }
        if which == "after":
            jdump(hs.pose_json(S.rig, z, S.R0, "hold_upright.py solve (scan hand in the web hand's pose, upright palm)"),
                  os.path.join(OUT, "pose.json"))
    web = json.load(open(WEB_METRICS))["report"]
    bc = json.load(open(os.path.join(WORK, "blend_check.json")))
    met = {"before": rows["before"], "after": rows["after"],
           "web_hand_for_comparison": {k: web[k] for k in web if k != "angles_deg"} | {
               "halves_from_its_joints": hs.half_palm_rolls(S.H)},
           "definitions": {
               "before": "hold_v2 (fitted_hold/, hold_solve.py): scan hand, cylinder-rig reference, palm arch 15.5 deg",
               "after": "scan hand in the web hand's pose (web_hand_rig/hand_on_grip_web.blend), re-solved for the contacts",
               "palm_tilt_from_vertical_deg": ("plane fitted to the palm-side skin (15-75 mm from the wrist, palm / finger "
                                               "metacarpal skin, thenar left out), asin(n_z), + = palm faces up; same as "
                                               "web_hand_rig hold_metrics.json (which reports |.|)"),
               "palm_faces_deg_from_minusX_in_plan": "atan2(n_y, -n_x) of that normal: 0 = facing straight at the other hand",
               "radial/ulnar_half_tilt_deg": ("tilt of the planes (wrist, index MCP, middle MCP) and (wrist, ring MCP, pinky "
                                              "MCP); fold = ulnar - radial = how far the pinky side of the palm is rolled "
                                              "in around the grip"),
               "arch_deg": "SPEC palm arch angles (skin.measurements); open scan hand: %.1f / %.1f" % (
                   S.rig.rest_meas["arch_imr"], S.rig.rest_meas["arch_mrp"]),
               "palm_swell_gap_mm": "palm swell marker (on the grip surface) to the nearest skin (contacts.json)",
               "max_penetration_into_grip_mm": "deepest skin inside the grip (0.5 mm skin samples, contacts.json)",
               "pinky_pad_gap_to_grip_mm": "signed distance of the pinky distal skin to the grip (- = inside), 1 mm grid",
               "flipped_triangles": "skin triangles turned inside out against their bone's rotation",
               "skin_into_skin_max_mm": "deepest skin of one part inside another (finger into finger), not corrected",
               "angles_vs_web_pose_deg": ("SPEC angles from world joint positions: 'web' = the web hand's posed joints "
                                          "(same skeleton), 'scan_hand' = this pose"),
           },
           "blender_vs_numpy_after_mm": bc["live_check"]["blender_vs_numpy_same_bulge_mm"],
           "solve": {"file": f"solve_{a.tag}.json", "weights": d["weights"], "summary": d["summary"], "nfev": d["nfev"]}}
    jdump(met, os.path.join(OUT, "metrics.json"))
    log("metrics written")


# ------------------------------------------------------------------ picture

# key, title, camera looks FROM this direction, up, mode ('solid' = hand + grip, 'hand' = hand only, close-up on the thenar)
VIEWS = [("along", "from behind, along the grip", (0.0, -1.0, 0.45), (0, 0, 1), "solid"),
         ("side", "from the side (back of the hand)", (1.0, -0.15, 0.25), (0, 0, 1), "solid"),
         ("front", "from the front (trigger side)", (0.25, 1.0, 0.2), (0, 0, 1), "solid"),
         ("thenar", "palm side, grip hidden", (-1.0, -0.25, 0.3), (0, 0, 1), "hand")]
RES = (900, 760)


def thenar_height(S, o):
    """How far the thenar (thumb-base pad: skin with thumb-metacarpal weight > 0.5 on the palm side) stands out of
    the palm-side skin plane, mm (95th percentile): the pad is intact when this stays at the open-hand value."""
    import skin
    n, _, _ = hs.palm_plane(S, o)
    c = o["V"][S.sel["palm_side"]].mean(0)
    th = (S.sk.W[:, skin.BI["thumb_metacarpal"]] > 0.5)
    h = (o["V"][th] - c) @ n
    return float(np.percentile(h, 95))


def cmd_render(a):
    import bpy
    from mathutils import Matrix, Vector
    from bpy_extras.object_utils import world_to_camera_view
    met = json.load(open(os.path.join(OUT, "metrics.json")))
    d = json.load(open(os.path.join(WORK, f"solve_{a.tag}.json")))
    S, _ = scene()
    out = os.path.join(WORK, "sheet")
    os.makedirs(out, exist_ok=True)
    hands, extra = {}, {}
    z_open = np.zeros_like(z_v2())
    for which, z in (("before", z_v2()), ("after", np.array(d["z"]))):
        o = S.forward(z)
        fl, _ = S.flips(o)
        dep = hs.inside_depth(S.Vg, S.Fg, o["V"])
        n, tilt, plan = hs.palm_plane(S, o)
        hands[which] = dict(V=o["V"], F=S.F, dep=dep, spots=hs.problem_spots(S, o["V"], fl, dep, met[which]),
                            palm_c=o["V"][S.sel["palm_side"]].mean(0), n=n, P=o["P"])
        th_faces = np.isin(S.lab[S.F[:, 0]], ["thumb_metacarpal"])
        extra[which] = {"thenar_height_mm": thenar_height(S, o), "tilt": tilt,
                        "thenar_flipped_triangles": int((fl & th_faces).sum())}
    # open hand (rest pose) thenar height, for the caption
    o_open = S.forward(np.r_[z_open[:24], z_v2()[24:]])
    extra["open_thenar_height_mm"] = thenar_height(S, o_open)
    wsk = np.load(os.path.join(WORK, "web_skin.npz"))
    hands["web"] = dict(V=wsk["V"], F=wsk["F"], dep=np.zeros(len(wsk["V"])), spots=[], palm_c=None, n=None, P=S.H)
    F, Vg, Fg = S.F, S.Vg, S.Fg

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene

    def mk(name, V, Ff, col):
        me = bpy.data.meshes.new(name)
        me.from_pydata(V.tolist(), [], Ff.tolist())
        for p in me.polygons:
            p.use_smooth = True
        ca = me.color_attributes.new("col", "FLOAT_COLOR", "POINT")
        ca.data.foreach_set("color", col.ravel().tolist())
        ob = bpy.data.objects.new(name, me)
        sc.collection.objects.link(ob)
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Roughness"].default_value = 0.55
        at = m.node_tree.nodes.new("ShaderNodeAttribute")
        at.attribute_name = "col"
        m.node_tree.links.new(at.outputs["Color"], b.inputs["Base Color"])
        ob.data.materials.append(m)
        return ob
    objs = {"grip": mk("grip", Vg, Fg, np.tile([[0.42, 0.45, 0.50, 1.0]], (len(Vg), 1)))}
    for which, h in hands.items():
        red = h["dep"] > 1.0
        skin_col = [0.30, 0.42, 0.62, 1.0] if which == "web" else [0.80, 0.55, 0.42, 1.0]
        col = np.where(red[:, None], [[0.9, 0.02, 0.02, 1.0]], [skin_col])
        objs[which] = mk(f"hand {which}", h["V"], h["F"], col)
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = a.samples
    try:
        sc.cycles.use_denoising = True
    except Exception:
        pass
    sc.render.resolution_x, sc.render.resolution_y = RES
    sc.view_settings.view_transform = "Standard"
    w = bpy.data.worlds.new("w")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.97, 0.97, 0.98, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.8
    for name, e, en in (("key", (0.6, -0.3, 0.9), 2.6), ("fill", (-0.7, -0.5, 0.4), 1.2), ("rim", (0.2, 0.9, -0.2), 0.8)):
        ld = bpy.data.lights.new(name, "SUN")
        ld.energy = en
        lo = bpy.data.objects.new(name, ld)
        sc.collection.objects.link(lo)
        lo.rotation_euler = Vector(e).to_track_quat("Z", "Y").to_euler()
    cam_d = bpy.data.cameras.new("cam")
    cam_d.type = "ORTHO"
    cam_d.clip_start, cam_d.clip_end = 1, 5000
    cam = bpy.data.objects.new("cam", cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam
    aspect = RES[0] / RES[1]
    meta = {}
    for key, title, dv, up, mode in VIEWS:
        dv = np.array(dv, float) / np.linalg.norm(dv)
        up = np.array(up, float)
        up = up - (up @ dv) * dv
        up /= np.linalg.norm(up)
        right = np.cross(up, dv)
        if mode == "hand":     # the palm side of the hand alone, both scan poses framed alike
            pts = np.concatenate([hands[k]["V"] for k in ("before", "after")])
            Pa = hands["after"]["P"]
            ax = (Pa["middle_mcp"] - Pa["wrist"]) / np.linalg.norm(Pa["middle_mcp"] - Pa["wrist"])
            pts = pts[(pts - Pa["wrist"]) @ ax > -20.0]       # leave most of the forearm stub out of the frame
            pad_mm = 6.0
        else:
            pts = np.concatenate([Vg, hands["before"]["V"], hands["after"]["V"], hands["web"]["V"]])
            pad_mm = 4.0
        cc = (pts.min(0) + pts.max(0)) / 2
        Q = pts - cc
        qr, qu = Q @ right, Q @ up
        cam_d.ortho_scale = max(np.ptp(qr), np.ptp(qu) * aspect) + 2 * pad_mm
        c2 = cc + right * 0.5 * (qr.max() + qr.min()) + up * 0.5 * (qu.max() + qu.min())
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = np.column_stack([right, up, dv]), c2 + dv * 800
        cam.matrix_world = Matrix(M.tolist())
        bpy.context.view_layer.update()

        def px(q):
            c = world_to_camera_view(sc, cam, Vector(q))
            return [float(c.x * RES[0]), float((1 - c.y) * RES[1])]
        for which in ("before", "after", "web"):
            if which == "web" and key not in ("along", "side"):
                continue
            for k, ob in objs.items():
                ob.hide_render = not (k == which or (k == "grip" and mode == "solid"))
            p = os.path.join(out, f"{which}_{key}.png")
            sc.render.filepath = p
            t0 = time.time()
            bpy.ops.render.render(write_still=True)
            log("rendered", p, f"{time.time() - t0:.0f} s")
            h = hands[which]
            m = {"spots": {k: {"px": px(q), "problem": bool(bad)} for k, q, bad in h["spots"]},
                 "mm_per_px": cam_d.ortho_scale / RES[0]}
            if key == "along" and which != "web":
                # the two halves of the palm seen edge-on: index side (wrist, index MCP, middle MCP) drawn
                # upward from the palm centre, pinky side (wrist, ring MCP, pinky MCP) downward, each along
                # its own plane; dashed vertical reference beside them
                P = h["P"]
                c = h["palm_c"]
                ez = np.array([0, 0, 1.0])
                for nm, (b1, b2), sgn in (("index_side", ("index_mcp", "middle_mcp"), 1), ("pinky_side", ("ring_mcp", "pinky_mcp"), -1)):
                    mu = np.cross(P[b1] - P["wrist"], P[b2] - P["wrist"])
                    mu /= np.linalg.norm(mu)
                    uu = ez - mu[2] * mu
                    uu /= np.linalg.norm(uu)
                    m[nm + "_line"] = [px(c), px(c + sgn * 48 * uu)]
                m["vertical_line"] = [px(c - 48 * ez - 9 * right), px(c + 48 * ez - 9 * right)]
                m["pinky_side_center"] = px(c - 30 * ez)
            if key == "thenar" and which != "web":
                import skin
                th = (S.sk.W[:, skin.BI["thumb_metacarpal"]] > 0.5)
                q = h["V"][th]
                m["thenar_center"] = px(q[np.argsort((q - h["palm_c"]) @ h["n"])[-200:]].mean(0))
                m["thenar_r_px"] = 20.0 / (cam_d.ortho_scale / RES[0])
            meta[f"{which}_{key}"] = m
    meta["extra"] = extra
    jdump(meta, os.path.join(out, "meta.json"))


def cmd_sheet(a):
    from PIL import Image, ImageDraw, ImageFont
    met = json.load(open(os.path.join(OUT, "metrics.json")))
    wd = os.path.join(WORK, "sheet")
    meta = json.load(open(os.path.join(wd, "meta.json")))
    ex = meta["extra"]
    FONT, FONT_B = hs.FONT, hs.FONT_B

    def font(sz, bold=False):
        return ImageFont.truetype(FONT_B if bold else FONT, sz)
    f_title, f_row, f_view, f_cap, f_big = font(56, True), font(48, True), font(32, True), font(30), font(40, True)
    pw = 640
    ph = int(RES[1] * pw / RES[0])
    sc_ = pw / RES[0]
    wpw = 470
    wph = int(RES[1] * wpw / RES[0])
    pad, top, rowh, viewh, caph = 26, 108, 74, 46, 96
    W = 4 * pw + 5 * pad + wpw + pad
    H = top + 2 * (rowh + viewh + ph + caph) + 130
    sh = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(sh)
    dr.text((pad, 22), "Your scanned hand on the grip: palm rolled in vs. palm upright", fill=(15, 15, 15), font=f_title)

    def cap_lines(txt, ft, width):
        words, lines = txt.split(), [""]
        for w_ in words:
            t = (lines[-1] + " " + w_).strip()
            if dr.textlength(t, font=ft) <= width:
                lines[-1] = t
            else:
                lines.append(w_)
        return lines

    def fmt_btn(r):
        f = r["fingertip_to_button_mm"]
        return (f"index {f['R1_surface_point_to_index_finger_mm']:.1f} mm from R1, middle {abs(f['R2_middle']):.1f} "
                f"from R2, ring {abs(f['R4_ring']):.1f} from R4")
    b, af = met["before"], met["after"]
    caps = {}
    for which, r in (("before", b), ("after", af)):
        caps[which] = {
            "along": (f"Palm cupped: the pinky side is rolled in {r['fold_ulnar_minus_radial_deg']:.0f} deg "
                      f"against the index side (red ring)") if which == "before" else
                     (f"Palm flat and upright: both sides within {max(abs(r['radial_half_tilt_deg']), abs(r['ulnar_half_tilt_deg'])):.0f} "
                      f"deg of vertical (green ring)"),
            "side": (f"Deepest skin in the plastic {r['max_penetration_into_grip_mm']:.1f} mm; palm on the palm swell: "
                     f"gap {r['palm_swell_gap_mm']:.1f} mm"),
            "front": f"On the buttons: {fmt_btn(r)}",
            "thenar": (f"Thumb-base pad (green ring) full and smooth: {ex[which]['thenar_flipped_triangles']} "
                       f"folded skin triangles there"),
        }
    rows = [("before", "BEFORE: hand rolled inward", (170, 20, 20)),
            ("after", "AFTER: your scanned hand in the web hand's pose (palm upright, like holding a box)", (20, 120, 40))]
    y = top
    row_y = {}
    for which, label, colr in rows:
        dr.text((pad, y + 8), label, fill=colr, font=f_row)
        y += rowh
        row_y[which] = y
        for vi, (key, title, _, _, _) in enumerate(VIEWS):
            x = pad + vi * (pw + pad)
            dr.text((x, y + 4), title, fill=(60, 60, 60), font=f_view)
            im = Image.open(os.path.join(wd, f"{which}_{key}.png")).convert("RGB").resize((pw, ph), Image.LANCZOS)
            di = ImageDraw.Draw(im)
            m = meta[f"{which}_{key}"]
            ok = (0, 165, 60)
            if key == "along":
                (x0, y0), (x1, y1) = [[v * sc_ for v in p] for p in m["vertical_line"]]
                for t in np.linspace(0, 1, 14)[::2]:          # dashed vertical reference
                    t2 = t + 1 / 13
                    di.line([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, x0 + (x1 - x0) * t2, y0 + (y1 - y0) * t2],
                            fill=(20, 20, 20), width=5)
                for nm in ("index_side", "pinky_side"):
                    (x0, y0), (x1, y1) = [[v * sc_ for v in p] for p in m[nm + "_line"]]
                    di.line([x0, y0, x1, y1], fill=(20, 80, 220), width=8)
                r = met[which]
                di.rectangle([8, 8, 400, 150], fill=(255, 255, 255))
                di.text((16, 10), f"index side {abs(r['radial_half_tilt_deg']):.0f} deg", fill=(20, 80, 220), font=font(36, True))
                di.text((16, 56), f"pinky side {abs(r['ulnar_half_tilt_deg']):.0f} deg", fill=(20, 80, 220), font=font(36, True))
                di.text((16, 104), "from upright (dashed)", fill=(20, 20, 20), font=font(30))
                cx, cy = [v * sc_ for v in m["pinky_side_center"]]
                folded = met[which]["fold_ulnar_minus_radial_deg"] > 10
                r_ = 46
                di.ellipse([cx - r_, cy - r_, cx + r_, cy + r_], outline=(225, 0, 0) if folded else ok, width=7)
            if key == "thenar":
                cx, cy = [v * sc_ for v in m["thenar_center"]]
                r_ = m["thenar_r_px"] * sc_
                di.ellipse([cx - r_, cy - r_, cx + r_, cy + r_], outline=ok, width=7)
            if key in ("side", "front"):
                for k, sp in m["spots"].items():
                    if hs.SPOT_VIEW.get(k) not in ((0,) if key == "side" else (1,)):
                        continue
                    bad = sp["problem"]
                    was_bad = meta.get(f"before_{key}", {}).get("spots", {}).get(k, {}).get("problem")
                    if not bad and not (which == "after" and was_bad):
                        continue
                    cx, cy = sp["px"][0] * sc_, sp["px"][1] * sc_
                    r_ = 34
                    di.ellipse([cx - r_, cy - r_, cx + r_, cy + r_], outline=(225, 0, 0) if bad else ok, width=7)
            sh.paste(im, (x, y + viewh))
            dr.rectangle([x, y + viewh, x + pw - 1, y + viewh + ph - 1], outline=(200, 200, 200), width=2)
            for li, t in enumerate(cap_lines(caps[which][key], f_cap, pw)[:3]):
                dr.text((x, y + viewh + ph + 6 + li * 33), t, fill=(30, 30, 30), font=f_cap)
        y += viewh + ph + caph
    # web hand panel (right column)
    xw = pad + 4 * (pw + pad)
    yw = row_y["before"]
    dr.rectangle([xw - 12, yw - 60, W - 8, yw + 2 * wph + 230], outline=(120, 120, 140), width=3)
    dr.text((xw, yw - 52), "For comparison:", fill=(60, 60, 60), font=f_view)
    dr.text((xw, yw - 14), "the web hand's pose", fill=(60, 60, 60), font=f_view)
    for i, key in enumerate(("along", "side")):
        im = Image.open(os.path.join(wd, f"web_{key}.png")).convert("RGB").resize((wpw, wph), Image.LANCZOS)
        sh.paste(im, (xw, yw + 30 + i * (wph + 50)))
        dr.text((xw, yw + 30 + i * (wph + 50) + wph + 4), VIEWS[i][1], fill=(60, 60, 60), font=font(28))
    wm = met["web_hand_for_comparison"]
    for li, t in enumerate(cap_lines(f"Blue hand = web hand (only its pose is used, not its surface). Palm "
                                     f"{wm['palm_tilt_from_vertical_deg']:.0f} deg from upright.", font(28), wpw)):
        dr.text((xw, yw + 30 + 2 * (wph + 50) + li * 32), t, fill=(30, 30, 30), font=font(28))
    leg = ["Blue lines = the two halves of the palm seen edge-on (index side up, pinky side down); dashed black line = straight up.  Red ring = problem spot, "
           "green ring = the same spot now fine.",
           "Red skin = more than 1 mm inside the plastic.  R1 / R2 / R4 = the buttons for the index, middle and ring finger.  Grey = the grip."]
    for i, t in enumerate(leg):
        dr.text((pad, H - 112 + i * 44), t, fill=(40, 40, 40), font=font(30))
    p = os.path.join(OUT, SHEET_NAME)
    sh.save(p, quality=90)
    log("WROTE", p, sh.size)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("ref")
    sp.add_parser("start")
    s = sp.add_parser("solve")
    s.add_argument("--tag", default="up")
    s.add_argument("--start", default=None)
    s.add_argument("--nfev", type=int, default=40)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--w", default=None)
    b = sp.add_parser("blend")
    b.add_argument("--solve", required=True)
    b.add_argument("--out-blend", required=True)
    for n in ("export", "metrics", "render"):
        e = sp.add_parser(n)
        e.add_argument("--tag", default="up")
        if n == "render":
            e.add_argument("--samples", type=int, default=24)
    sp.add_parser("sheet")
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
