"""Compose final/final_sheet.jpg and final/compare_sheet.jpg from final/renders/{view}_{mode}_{label}.png
(made by lib_v2/render_relief.py) and the check run's sheet.jpg.   python3 sheets_v2.py CHECK_RUN_DIR"""
import os, sys
from PIL import Image, ImageDraw
R = '/tmp/claude-0/sdf/smooth/final/renders'
F = '/tmp/claude-0/sdf/smooth/final'
VIEWS = ('outside', 'front', 'frontout', 'inside', 'under')
tw, th = 400, 300


def grid(cols, title, out, extra_top=None):
    rows = [(v, cols) for v in VIEWS]
    top = 44
    ex = None
    if extra_top:
        ex = Image.open(extra_top).convert('RGB')
        ex = ex.resize((tw * len(cols), int(ex.height * tw * len(cols) / ex.width)))
    H = top + (ex.height + 10 if ex else 0) + th * len(rows)
    S = Image.new('RGB', (tw * len(cols), H), (30, 30, 30))
    d = ImageDraw.Draw(S)
    d.text((8, 4), title, fill=(255, 255, 255))
    y0 = top
    if ex:
        S.paste(ex, (0, y0 - 20)); y0 += ex.height + 10
    for ci, (lab, mode, head) in enumerate(cols):
        d.text((ci * tw + 8, y0 - 18 if ex else 24), head, fill=(255, 255, 0))
    for ri, v in enumerate(VIEWS):
        for ci, (lab, mode, head) in enumerate(cols):
            f = os.path.join(R, f'{v}_{mode}_{lab}.png')
            if os.path.exists(f):
                S.paste(Image.open(f).convert('RGB').resize((tw, th)), (ci * tw, y0 + ri * th))
            d.text((ci * tw + 6, y0 + ri * th + 4), f'{head} | {v}', fill=(255, 90, 90))
    S.save(out, quality=88)
    print('SHEET', out, S.size)


run = sys.argv[1] if len(sys.argv) > 1 else F + '/runs/v2_final'
grid([('v2', 'graze', 'v2 (installed) grazing light'), ('v2', 'zebra', 'v2 zebra'), ('plain', 'zebra', 'plain GripV zebra (reference)')],
     'final_sheet: Grip_smooth.blend v2 (SPEC v2 low-frequency fit), right handle, z <= 41; top: check.py views of the same run',
     F + '/final_sheet.jpg', extra_top=os.path.join(run, 'sheet.jpg'))
grid([('plain', 'graze', 'plain GripV'), ('v1', 'graze', 'v1 (Grip_smooth_v1.blend)'), ('v2', 'graze', 'v2 (Grip_smooth.blend)')],
     'compare_sheet: grazing light, same views and light; plain GripV vs v1 vs v2', F + '/compare_sheet.jpg')
