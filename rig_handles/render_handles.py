"""render_handles.py: a readable picture of the handles and the mock-up halves.

    $PY rig_handles/render_handles.py grip_rig_handles.blend OUT.jpg

Left: the grip from the palm side and the finger side with every handle marked and named.
Right: the two mock-up halves (top lifted off) as they come out of 'mock-up top' / 'mock-up bottom'.
"""
import os
import sys

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FONT_B = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
RES = 760
HANDLES = [  # bone, label, colour, use tail
    ("palm swell", "palm swell", (230, 60, 60), False),
    ("fin middle handle", "middle finger fin", (60, 160, 60), False),
    ("fin ring handle", "ring finger fin", (40, 110, 220), False),
    ("fin pinky handle", "pinky fin", (160, 70, 200), False),
    ("stand handle", "handle end (stand)", (230, 140, 30), False),
    ("thenar surface", "thumb base", (20, 170, 170), True),
    ("R2 button top", "R2 button top", (200, 170, 0), False),
    ("R4 button top", "R4 button top", (200, 170, 0), False),
]
OFFSETS = {("finger side (front)", "fin middle handle"): (-70, 16),
           ("finger side (front)", "fin ring handle"): (15, 4),
           ("finger side (front)", "stand handle"): (-205, -12)}
GRIP_PARTS = ["basemesh GripV", "R2", "R4", "L2", "L4", "R1", "L1"]


def mesh_copy(name, sc, dz=0.0, color=(0.8, 0.8, 0.82, 1)):
    dg = bpy.context.evaluated_depsgraph_get()
    src = bpy.data.objects[name]
    me = bpy.data.meshes.new_from_object(src.evaluated_get(dg), depsgraph=dg)
    ob = bpy.data.objects.new("pic " + name, me)
    ob.matrix_world = src.matrix_world
    ob.location.z += dz
    me.materials.clear()
    mat = bpy.data.materials.new("pic " + name)
    mat.diffuse_color = color
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = color
    mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.6
    me.materials.append(mat)
    for p in me.polygons:
        p.use_smooth = True
    sc.collection.objects.link(ob)
    return ob


def setup_scene():
    sc = bpy.data.scenes.new("handle picture")
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 24
    sc.cycles.use_denoising = False
    sc.render.resolution_x = RES
    sc.render.resolution_y = RES
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    w = bpy.data.worlds.new("pic world")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.92, 0.92, 0.94, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.35
    sc.world = w
    sun = bpy.data.objects.new("pic sun", bpy.data.lights.new("pic sun", "SUN"))
    sun.data.energy = 4.0
    sun.rotation_euler = (0.7, 0.3, 0.6)
    sc.collection.objects.link(sun)
    cam = bpy.data.objects.new("pic cam", bpy.data.cameras.new("pic cam"))
    cam.data.type = "ORTHO"
    sc.collection.objects.link(cam)
    sc.camera = cam
    return sc, cam


def aim(cam, target, direction, up, scale, dist=400.0):
    d = Vector(direction).normalized()
    cam.location = Vector(target) + d * dist
    upv = Vector(up)
    right = (-d).cross(upv).normalized()
    true_up = right.cross(-d)
    from mathutils import Matrix
    M = Matrix((right, true_up, d)).transposed()
    cam.rotation_euler = M.to_euler()
    cam.data.ortho_scale = scale
    cam.data.clip_end = 2000


def render(sc, path):
    sc.render.filepath = path
    bpy.context.window.scene = sc
    bpy.ops.render.render(write_still=True)
    return Image.open(path).convert("RGB")


def main():
    src, out = sys.argv[-2], sys.argv[-1]
    bpy.ops.wm.open_mainfile(filepath=src)
    bpy.data.collections["mock-up print"].hide_viewport = False
    bpy.data.objects["mock-up whole"].hide_viewport = False
    rig = bpy.data.objects["grip rig"]
    bpy.context.view_layer.update()
    pts = {}
    for b, _, _, tail in HANDLES:
        pb = rig.pose.bones[b]
        pts[b] = rig.matrix_world @ (pb.tail if tail else pb.head)
    sc, cam = setup_scene()
    grip = [mesh_copy(n, sc, color=(0.85, 0.85, 0.87, 1) if n == "basemesh GripV" else (0.35, 0.35, 0.4, 1))
            for n in GRIP_PARTS]
    halves = [mesh_copy("mock-up top", sc, dz=35.0, color=(0.95, 0.75, 0.45, 1)),
              mesh_copy("mock-up bottom", sc, color=(0.55, 0.75, 0.95, 1))]
    for o in halves:
        o.hide_render = True
    tmp = os.path.dirname(os.path.abspath(out))
    font = ImageFont.truetype(FONT, 22)
    fontb = ImageFont.truetype(FONT_B, 30)

    views = [("palm side", (1.0, -0.25, 0.1), (0, 0, 1), (60, -50, 20), 150),
             ("finger side (front)", (0.45, 1.0, -0.2), (0, 0, 1), (55, -45, 20), 150)]
    tiles = []
    for title, d, up, tgt, scale in views:
        aim(cam, tgt, d, up, scale)
        bpy.context.view_layer.update()
        im = render(sc, os.path.join(tmp, "_pic.png"))
        dr = ImageDraw.Draw(im)
        dirv = Vector(d).normalized()
        for b, label, col, _ in HANDLES:
            p = pts[b]
            if (p - Vector(tgt)).dot(dirv) < -25:     # far behind the grip in this view
                continue
            u = world_to_camera_view(sc, cam, p)
            x, y = u.x * RES, (1 - u.y) * RES
            dr.ellipse([x - 11, y - 11, x + 11, y + 11], fill=col, outline=(0, 0, 0), width=2)
            dx, dy = OFFSETS.get((title, b), (15, -12))
            dr.text((x + dx, y + dy), label, font=font, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
        dr.text((16, 12), "Handles, " + title, font=fontb, fill=(0, 0, 0))
        tiles.append(im)
    for o in grip:
        o.hide_render = True
    for o in halves:
        o.hide_render = False
    for title, d in (("Mock-up print, top half lifted", (0.9, -0.6, 0.55)),
                     ("Mock-up print, from below", (0.6, 0.5, -0.8))):
        aim(cam, (0, -45, 35), d, (0, 0, 1), 250)
        im = render(sc, os.path.join(tmp, "_pic.png"))
        dr = ImageDraw.Draw(im)
        dr.text((16, 12), title, font=fontb, fill=(0, 0, 0))
        tiles.append(im)
    os.remove(os.path.join(tmp, "_pic.png"))

    W = RES * 2
    sheet = Image.new("RGB", (W, RES * 2 + 80), (255, 255, 255))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % 2) * RES, (i // 2) * RES))
    ImageDraw.Draw(sheet).text(
        (16, RES * 2 + 10),
        "Top row: the balls you drag (G) or set with sliders in the N panel > Grip tab.\n"
        "Bottom row: caps, triggers, bumpers, sticks merged into the grip, cut at Z = 40 like the real case.",
        font=ImageFont.truetype(FONT, 24), fill=(0, 0, 0))
    sheet.save(out, quality=90)
    print("saved", out)


if __name__ == "__main__":
    main()
