"""render_r1_slide.py: picture of the index offset: 0 (fingertip on R1) | scan | 1 (deepest contact).

    $PY rig_handles/render_r1_slide.py grip_rig_handles.blend OUT.jpg
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
    bpy.ops.wm.open_mainfile(filepath=src)
    rig = bpy.data.objects["grip rig"]
    h = rig.pose.bones["R1 hand"]
    u0, m = h["index offset"], h["deepest contact"]
    poses = [(0.0, "index offset 0: fingertip on R1"), (u0, "scan result"),
             (1.0, "index offset 1: deepest contact")]
    sc, cam = rh.setup_scene()
    rh.RES = RES
    sc.render.resolution_x = sc.render.resolution_y = RES
    win = bpy.context.window.scene
    c = rig.matrix_world @ rig.data.bones["R1 hand"].head_local
    tiles = []
    font = ImageFont.truetype(rh.FONT_B, 28)
    small = ImageFont.truetype(rh.FONT, 22)
    for x, title in poses:
        bpy.context.window.scene = win
        h["index offset"] = x
        rig.update_tag()
        bpy.context.view_layer.update()
        objs = [rh.mesh_copy(n, sc, color=(0.85, 0.85, 0.87, 1) if n == "basemesh GripV" else (0.3, 0.3, 0.35, 1))
                for n in PARTS]
        objs.append(rh.mesh_copy("hand scan", sc, color=(0.9, 0.62, 0.5, 1)))
        ps = rig.matrix_world @ rig.pose.bones["palm swell"].head
        rh.aim(cam, (60, -45, 30), (0.15, 0.35, 1.0), (0, 1, 0), 230)
        im = rh.render(sc, os.path.join(os.path.dirname(os.path.abspath(out)), "_r1.png"))
        dr = ImageDraw.Draw(im)
        for p, col, lab in ((c, (220, 30, 30), "R1 handle"), (ps, (40, 180, 40), "palm swell")):
            u = world_to_camera_view(sc, cam, p)
            px, py = u.x * RES, (1 - u.y) * RES
            dr.ellipse([px - 9, py - 9, px + 9, py + 9], fill=col, outline=(0, 0, 0), width=2)
            dr.text((px + 12, py - 10), lab, font=small, fill=(0, 0, 0), stroke_width=3, stroke_fill=(255, 255, 255))
        uc, up = world_to_camera_view(sc, cam, c), world_to_camera_view(sc, cam, ps)
        dr.line([uc.x * RES, (1 - uc.y) * RES, up.x * RES, (1 - up.y) * RES], fill=(40, 120, 230), width=3)
        dr.text((14, 10), title, font=font, fill=(0, 0, 0))
        dr.text((14, 46), f"index offset {x:.2f}: R1 touches {x * m:.0f} mm from the tip", font=small, fill=(0, 0, 0))
        tiles.append(im)
        for o in objs:
            bpy.data.objects.remove(o)
    os.remove(os.path.join(os.path.dirname(os.path.abspath(out)), "_r1.png"))
    sheet = Image.new("RGB", (RES * 3, RES + 50), (255, 255, 255))
    for i, im in enumerate(tiles):
        sheet.paste(im, (i * RES, 0))
    ImageDraw.Draw(sheet).text((14, RES + 12), "The index offset slides the hand along the index finger; "
                               "the palm swell (green) follows the palm, so the blue link stretches.",
                               font=ImageFont.truetype(rh.FONT, 22), fill=(0, 0, 0))
    sheet.save(out, quality=90)
    print("saved", out)


if __name__ == "__main__":
    main()
