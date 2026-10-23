"""Render 'hand on the controller' views for Olex: the grip opaque, the skin see-through, skin
that goes into the plastic painted red, plus the projected 2D positions of points to circle.

    $P grip_views.py <hand_on_grip.blend> <outdir> [--samples 24]

Writes <outdir>/{old,new}_{view}.png and <outdir>/grip_views.json with, per image, the pixel
positions of: the deepest skin point inside the grip and the palm-swell marker.
Used by fit_standin_to_scan.py sheet.
"""
import json
import os
import sys

import numpy as np

GRIP = "grip (current GripV)"
OLD_COLL = "old cylinder hand"
NEW_COLL = "new hand from scan"
# title, direction the camera looks FROM, up
VIEWS = [("outside", (1.0, -0.25, 0.35), (0, 0, 1)),
         ("from above", (0.35, -0.45, 1.0), (0, 1, 0))]
RES = (900, 760)


def world_mesh(o, dg):
    e = o.evaluated_get(dg)
    me = e.to_mesh()
    me.calc_loop_triangles()
    M = np.array(o.matrix_world)
    V = np.array([v.co[:] for v in me.vertices]) @ M[:3, :3].T + M[:3, 3]
    F = np.array([t.vertices[:] for t in me.loop_triangles])
    e.to_mesh_clear()
    return V, F


def main():
    import bpy
    from mathutils import Matrix, Vector
    from mathutils.bvhtree import BVHTree
    from bpy_extras.object_utils import world_to_camera_view
    blend, out = sys.argv[1], sys.argv[2]
    samples = int(sys.argv[sys.argv.index("--samples") + 1]) if "--samples" in sys.argv else 24
    os.makedirs(out, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=blend)
    sc = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    grip = bpy.data.objects[GRIP]
    Vg, Fg = world_mesh(grip, dg)
    bvh = BVHTree.FromPolygons(Vg.tolist(), Fg.tolist())

    def inside_depth(V):
        """depth (mm) of each point inside the closed grip (0 outside): nearest-face normal test."""
        d = np.zeros(len(V))
        for i, p in enumerate(V):
            loc, nrm, _, dist = bvh.find_nearest(Vector(p))
            if loc is not None and (np.array(p) - np.array(loc)) @ np.array(nrm) < 0:
                d[i] = dist
        return d

    # bake each hand mesh (evaluated, posed) into a plain mesh with a red/skin colour attribute
    hands = {"old": [o for o in bpy.data.collections[OLD_COLL].objects if o.type == "MESH"],
             "new": [bpy.data.objects["hand scan"]]}
    baked, deepest = {}, {}
    for row, objs in hands.items():
        baked[row] = []
        best = (0.0, None)
        for o in objs:
            V, F = world_mesh(o, dg)
            dep = inside_depth(V)
            if dep.max() > best[0]:
                best = (float(dep.max()), V[int(np.argmax(dep))].tolist())
            me = bpy.data.meshes.new(f"bake {row} {o.name}")
            me.from_pydata(V.tolist(), [], F.tolist())
            col = me.color_attributes.new("pen", "FLOAT_COLOR", "POINT")
            c = np.where((dep > 0.3)[:, None], [[0.85, 0.02, 0.02, 1.0]], [[0.80, 0.55, 0.42, 1.0]])
            col.data.foreach_set("color", c.ravel().tolist())
            me.attributes.new("solid", "FLOAT", "POINT").data.foreach_set("value", (dep > 0.3).astype(float).tolist())
            for p in me.polygons:
                p.use_smooth = True
            ob = bpy.data.objects.new(f"bake {row} {o.name}", me)
            sc.collection.objects.link(ob)
            baked[row].append(ob)
        deepest[row] = best
    # hide every original object except the grip
    for o in bpy.data.objects:
        if o.type == "MESH" and o not in sum(baked.values(), []) and o.name != GRIP:
            o.hide_render = True
    for c in bpy.data.collections:
        c.hide_render = False

    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.transparent_max_bounces = 16
    try:
        sc.cycles.use_denoising = True
    except Exception:
        pass
    sc.render.resolution_x, sc.render.resolution_y = RES
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    w = sc.world or bpy.data.worlds.new("w")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.95, 0.95, 0.96, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.7

    gm = bpy.data.materials.new("grip mat")
    gm.use_nodes = True
    b = gm.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.20, 0.22, 0.26, 1)
    b.inputs["Roughness"].default_value = 0.5
    b.inputs["Alpha"].default_value = 0.45      # see-through, so skin inside the plastic shows red
    grip.data.materials.clear()
    grip.data.materials.append(gm)
    sm = bpy.data.materials.new("skin see-through")
    sm.use_nodes = True
    nt = sm.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.55
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_name = "pen"
    nt.links.new(at.outputs["Color"], b.inputs["Base Color"])
    # red skin also glows a little, so it reads through the see-through plastic
    em = [i for i in b.inputs if i.name in ("Emission Color", "Emission")][0]
    nt.links.new(at.outputs["Color"], em)
    at3 = nt.nodes.new("ShaderNodeAttribute")
    at3.attribute_name = "solid"
    nt.links.new(at3.outputs["Fac"], b.inputs["Emission Strength"])
    at2 = nt.nodes.new("ShaderNodeAttribute")
    at2.attribute_name = "solid"
    mx = nt.nodes.new("ShaderNodeMapRange")          # alpha 0.45 normal skin, 1.0 red skin
    mx.inputs["To Min"].default_value = 0.45
    mx.inputs["To Max"].default_value = 1.0
    nt.links.new(at2.outputs["Fac"], mx.inputs["Value"])
    nt.links.new(mx.outputs["Result"], b.inputs["Alpha"])
    try:
        sm.blend_method = "BLEND"
    except Exception:
        pass
    for ob in sum(baked.values(), []):
        ob.data.materials.append(sm)
    for name, e, en in (("key", (0.6, -0.3, 0.9), 2.6), ("fill", (-0.7, -0.5, 0.4), 1.0), ("rim", (0.2, 0.9, -0.2), 0.7)):
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
    allp = np.concatenate([Vg] + [world_mesh(o, dg)[0] for o in sum(baked.values(), [])])
    lo_, hi_ = allp.min(0), allp.max(0)
    ctr = (lo_ + hi_) / 2
    aspect = RES[0] / RES[1]
    swell = np.array(bpy.data.objects["palm swell"].matrix_world.translation)
    meta = {}
    for row in ("old", "new"):
        for o in sum(baked.values(), []):
            o.hide_render = o not in baked[row]
        for vi, (title, d, up) in enumerate(VIEWS):
            d = np.array(d, float) / np.linalg.norm(d)
            up = np.array(up, float)
            up = up - (up @ d) * d
            up /= np.linalg.norm(up)
            right = np.cross(up, d)
            Q = allp - ctr
            qr, qu = Q @ right, Q @ up
            wspan, hspan = np.ptp(qr), np.ptp(qu)
            cam_d.ortho_scale = 1.06 * max(wspan, hspan * aspect)
            c2 = ctr + right * 0.5 * (qr.max() + qr.min()) + up * 0.5 * (qu.max() + qu.min())
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = np.column_stack([right, up, d]), c2 + d * 800
            cam.matrix_world = Matrix(M.tolist())
            bpy.context.view_layer.update()
            p = os.path.join(out, f"{row}_{vi}.png")
            sc.render.filepath = p
            bpy.ops.render.render(write_still=True)

            def px(q):
                c = world_to_camera_view(sc, cam, Vector(q))
                return [float(c.x * RES[0]), float((1 - c.y) * RES[1])]
            meta[f"{row}_{vi}"] = {"view": title, "deepest_px": px(deepest[row][1]) if deepest[row][1] else None,
                                   "deepest_mm": deepest[row][0], "palm_swell_px": px(swell),
                                   "mm_per_px": cam_d.ortho_scale / RES[0]}
            print("rendered", p, flush=True)
    with open(os.path.join(out, "grip_views.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
