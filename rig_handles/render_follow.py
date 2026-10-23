"""render_follow.py: picture of R2/R4 following the fingertips: index offset 0 | scan | 1, finger side.

    $PY rig_handles/render_follow.py grip_rig_handles.blend OUT.jpg
"""
import os
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_handles as rh  # noqa: E402

RES = 640
PARTS = ["basemesh GripV", "R1", "R2", "R4"]


def main():
    src, out = sys.argv[-2], sys.argv[-1]
    bpy.context.preferences.filepaths.use_scripts_auto_execute = True   # the arch-tilt drivers
    bpy.ops.wm.open_mainfile(filepath=src)
    ns = {"__name__": "ui"}
    t = bpy.data.texts["grip_controls_ui.py"]
    exec(compile(t.as_string(), t.name, "exec"), ns)
    rig = bpy.data.objects["grip rig"]
    h = rig.pose.bones["R1 hand"]
    u0, m = h["index offset"], h["deepest contact"]
    poses = [(0.0, "index offset 0: fingertip on R1"), (u0, "scan result"), (1.0, "index offset 1: deepest contact")]
    sc, cam = rh.setup_scene()
    rh.RES = RES
    sc.render.resolution_x = sc.render.resolution_y = RES
    win = bpy.context.window.scene
    tiles = []
    font = ImageFont.truetype(rh.FONT_B, 28)
    small = ImageFont.truetype(rh.FONT, 22)
    tmp = os.path.join(os.path.dirname(os.path.abspath(out)), "_follow.png")
    for u, title in poses:
        bpy.context.window.scene = win
        h["index offset"] = u
        rig.update_tag()
        bpy.context.view_layer.update()
        angles = {p: ns["press_angle"](p) for p in ("R2", "R4")}
        tops = {p: rig.matrix_world @ rig.pose.bones[p + " button top"].head for p in ("R2", "R4")}
        objs = [rh.mesh_copy(n, sc, color=(0.85, 0.85, 0.87, 1) if n == "basemesh GripV" else (0.3, 0.3, 0.35, 1))
                for n in PARTS]
        hand = rh.mesh_copy("hand scan", sc, color=(0.9, 0.62, 0.5, 1))
        rh.aim(cam, (45, -35, 25), (0.55, 1.0, -0.15), (0, 0, 1), 150)
        im = rh.render(sc, tmp)
        hand.hide_render = True
        im2 = rh.render(sc, tmp)
        bpy.data.objects.remove(hand)
        dr = ImageDraw.Draw(im)
        dr2 = ImageDraw.Draw(im2)
        for p, col in (("R2", (230, 190, 0)), ("R4", (40, 110, 220))):
            q = world_to_camera_view(sc, cam, tops[p])
            px, py = q.x * RES, (1 - q.y) * RES
            for d in (dr, dr2):
                d.ellipse([px - 8, py - 8, px + 8, py + 8], fill=col, outline=(0, 0, 0), width=2)
                d.text((px + 12, py - 12), f"{p} press angle {angles[p]:.0f}°", font=small, fill=(0, 0, 0),
                       stroke_width=3, stroke_fill=(255, 255, 255))
        dr.text((14, 10), title, font=font, fill=(0, 0, 0))
        dr.text((14, 46), f"R1 touches {u * m:.0f} mm from the index tip", font=small, fill=(0, 0, 0))
        dr2.text((14, 10), "same, hand hidden", font=font, fill=(0, 0, 0))
        tiles.append((im, im2))
        for o in objs:
            bpy.data.objects.remove(o)
    os.remove(tmp)
    sheet = Image.new("RGB", (RES * 3, 2 * RES + 50), (255, 255, 255))
    for i, (im, im2) in enumerate(tiles):
        sheet.paste(im, (i * RES, 0))
        sheet.paste(im2, (i * RES, RES))
    ImageDraw.Draw(sheet).text((14, 2 * RES + 12), "R2 and R4 move with the middle and ring fingertips and turn with "
                               "their arch, so the press angle stays as designed (R2 21°, R4 17°).",
                               font=ImageFont.truetype(rh.FONT, 22), fill=(0, 0, 0))
    sheet.save(out, quality=90)
    print("saved", out)


if __name__ == "__main__":
    main()
