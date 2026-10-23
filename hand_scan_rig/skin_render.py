"""Rendering for skin.py: the finger at 0/45/90 deg from the side, plain vs default skinning."""
import json
import os
import time

import numpy as np

import skin
from skin import OUT, WORK

SKIN_RGB = (0.80, 0.55, 0.42, 1.0)
BAD_RGB = (0.85, 0.15, 0.12, 1.0)


def _scene(res_x, res_y, samples=24):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = False
    sc.render.resolution_x, sc.render.resolution_y = res_x, res_y
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.6
    sc.world = w
    return sc


def _material(name, rgba):
    import bpy
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = 0.6
    return m


def _add_mesh(name, V, F, mat, colors=None):
    import bpy
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.astype(np.float32).ravel())
    me.loops.add(F.size)
    me.loops.foreach_set("vertex_index", F.astype(np.int32).ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, F.size, 3, dtype=np.int32))
    me.update(calc_edges=True)
    me.polygons.foreach_set("use_smooth", np.ones(len(F), bool))
    me.materials.append(mat)
    if colors is not None:
        attr = me.color_attributes.new("col", "FLOAT_COLOR", "POINT")
        attr.data.foreach_set("color", colors.astype(np.float32).ravel())
        m = mat
        nt = m.node_tree
        a = nt.nodes.new("ShaderNodeVertexColor")
        a.layer_name = "col"
        nt.links.new(a.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def _camera(loc, look_dir, up, ortho_scale):
    import bpy
    from mathutils import Matrix, Vector
    cam = bpy.data.cameras.new("cam")
    cam.type = "ORTHO"
    cam.ortho_scale = ortho_scale
    cam.clip_end = 5000
    ob = bpy.data.objects.new("cam", cam)
    bpy.context.scene.collection.objects.link(ob)
    f = Vector(look_dir).normalized()
    u = Vector(up).normalized()
    r = f.cross(u).normalized()
    u = r.cross(f)
    M = Matrix((r, u, -f)).transposed().to_4x4()
    M.translation = Vector(loc)
    ob.matrix_world = M
    bpy.context.scene.camera = ob
    return ob


def _lights(key_dir, centre):
    import bpy
    from mathutils import Vector
    for name, d, e in [("key", key_dir, 3.5), ("fill", (-key_dir[0], key_dir[1], -key_dir[2]), 1.2)]:
        L = bpy.data.lights.new(name, "SUN")
        L.energy = e
        L.angle = 0.2
        ob = bpy.data.objects.new(name, L)
        bpy.context.scene.collection.objects.link(ob)
        ob.rotation_euler = Vector(d).normalized().to_track_quat("Z", "Y").to_euler()
        ob.location = Vector(centre)


def render_panel(V, F, out_png, colors=None, centre=(0.0, 120.0, 25.0), span=170.0, cell=(560, 520),
                 view="side", samples=24):
    """One mesh, own orthographic camera.  view 'side': from +X (thumb side), palm down
    (camera up = -Z) so fingers point left and curl down.  view 'palm': from +Z."""
    import bpy
    sc = _scene(cell[0], cell[1], samples)
    mat = _material("skin", SKIN_RGB)
    _add_mesh("hand", V, F, mat, colors)
    c = np.asarray(centre, float)
    if view == "side":
        _camera(c + [400.0, 0, 0], (-1, 0, 0), (0, 0, -1), span)
        _lights((0.6, 0.3, -0.75), c)
    elif view == "back":
        _camera(c + [0, 0, -400.0], (0, 0, 1), (0, 1, 0), span)
        _lights((0.3, 0.4, 0.8), c)
    else:
        _camera(c + [0, 0, 400.0], (0, 0, -1), (0, 1, 0), span)
        _lights((0.3, 0.4, -0.8), c)
    sc.render.filepath = out_png
    bpy.ops.render.render(write_still=True)
    return out_png


def compose(pngs, ncols, out_png):
    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in pngs]
    w, h = ims[0].size
    nrows = (len(ims) + ncols - 1) // ncols
    can = Image.new("RGB", (w * ncols, h * nrows), "white")
    for k, im in enumerate(ims):
        r, c = divmod(k, ncols)
        can.paste(im, (c * w, r * h))
    can.save(out_png)
    return out_png


def label_grid(png, out_png, ncols, nrows, row_labels, col_labels, title, caption, cell):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(png).convert("RGB")
    left, top, bottom = 420, 150, 90
    W = im.width + left + 20
    H = im.height + top + bottom
    can = Image.new("RGB", (W, H), "white")
    can.paste(im, (left, top))
    d = ImageDraw.Draw(can)

    def font(sz, bold=False):
        for p in (["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else []) + \
                ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            if os.path.exists(p):
                return ImageFont.truetype(p, sz)
        return ImageFont.load_default(size=sz)

    d.text((20, 18), title, fill="black", font=font(46, True))
    for c, t in enumerate(col_labels):
        x = left + c * cell[0] + cell[0] // 2
        d.text((x, top - 40), t, fill="black", font=font(44, True), anchor="mm")
    for r, t in enumerate(row_labels):
        y = top + r * cell[1] + cell[1] // 2
        for i, line in enumerate(t.split("\n")):
            d.text((20, y - 30 + 50 * i), line, fill="black", font=font(40 if i == 0 else 28, i == 0))
        if r:
            d.line([(20, top + r * cell[1]), (W - 20, top + r * cell[1])], fill=(180, 180, 180), width=2)
    d.text((20, H - bottom + 25), caption, fill=(40, 40, 40), font=font(30))
    can.save(out_png)
    print("WROTE", out_png)


def render(a):
    import skin_checks as sc_
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    angs = [0, 45, 90]
    rows = [("plain", "Plain skinning", "(Blender automatic weights)"),
            ("default", "Our skin model", "(default settings)")]
    cell = (620, 560)
    pngs = []
    for cname, _, _ in rows:
        sk = skin.Skinner(sc_.CONFIGS[cname]["blend"], sc_.config_params(cname))
        for ang in angs:
            # the four fingers bend together (a fist closing), thumb moved out of the line of sight
            angles = {"thumb_cmc": 62.0}
            for f in skin.F4:
                angles.update({f"{f}_mcp": ang, f"{f}_pip": ang, f"{f}_dip": ang * 2 / 3})
            V, info = sk.pose({"angles": angles}, "numpy", return_info=True)
            np.save(os.path.join(WORK, f"render_{cname}_{ang}.npy"), V)
            c = info["landmarks"]["index_mcp"] + np.array([0.0, 30.0, 25.0])
            pngs.append(render_panel(V, sk.F, os.path.join(WORK, f"panel_{cname}_{ang}.png"),
                                     centre=c, span=175.0, cell=cell, view="side", samples=32))
    raw = compose(pngs, 3, os.path.join(WORK, "side_raw.png"))
    label_grid(raw, os.path.join(OUT, "finger_fold_compare.png"), 3, 2,
               [f"{r[1]}\n{r[2]}" for r in rows], [f"bent {x}°" for x in angs],
               "Fingers bending, seen from the thumb side",
               "Same hand, same bend. Watch the knuckle (top) and the inside of each bend.", cell)
