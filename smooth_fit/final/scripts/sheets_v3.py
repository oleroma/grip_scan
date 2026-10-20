"""Compose final/final_sheet.jpg and final/compare_sheet.jpg (review round 2, v3 shape) from final/renders/
{view}_{mode}_{label}.png (lib_v2/render_relief.py) and the close-ups in final/renders/closeup (work/v3/render_closeup.py,
the independent verifier's top-lit cameras).   python3 sheets_v3.py [CHECK_RUN_DIR]
Reference column = no-fit (plain GripV through the same remesh, Fit 0), as the verifier asked: the remesh itself
changes zebra, so 'calm as plain' is judged against no-fit."""
import os, sys
from PIL import Image, ImageDraw
R = '/tmp/claude-0/sdf/smooth/final/renders'
F = '/tmp/claude-0/sdf/smooth/final'
VIEWS = ('outside', 'front', 'frontout', 'inside', 'under')
tw, th = 400, 300


def closeup_band(width):
    D = R + '/closeup'
    names = [('c_nofit', 'no-fit'), ('c_v2', 'v2 (before)'), ('c_v3', 'v3 (installed)')]
    crops = [(('graze', 'outA'), (185, 80, 465, 260)), (('graze', 'outB'), (185, 80, 465, 330)), (('zebra', 'outA'), (185, 80, 465, 260))]
    cw = width // 3; ch = int(cw * 0.62)
    B = Image.new('RGB', (cw * 3, ch * len(crops) + 22), (30, 30, 30))
    d = ImageDraw.Draw(B)
    d.text((8, 4), 'Close-ups, outer face, light from above (the v2 top-band ridge shows as a horizontal streak in v2 graze outA)', fill=(255, 255, 255))
    for r, ((m, v), box) in enumerate(crops):
        for c, (n, lab) in enumerate(names):
            f = f'{D}/{n}_{m}_{v}.png'
            if os.path.exists(f):
                B.paste(Image.open(f).convert('RGB').crop(box).resize((cw, ch)), (c * cw, 22 + r * ch))
            d.text((c * cw + 6, 22 + r * ch + 4), f'{lab} | {m} {v}', fill=(255, 90, 90))
    return B


def grid(cols, title, out, extra_top=None, extra_bottom=None):
    top = 44
    ex = None
    if extra_top:
        ex = Image.open(extra_top).convert('RGB')
        ex = ex.resize((tw * len(cols), int(ex.height * tw * len(cols) / ex.width)))
    H = top + (ex.height + 10 if ex else 0) + th * len(VIEWS) + (extra_bottom.height + 10 if extra_bottom else 0)
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
    if extra_bottom:
        S.paste(extra_bottom, (0, y0 + th * len(VIEWS) + 10))
    S.save(out, quality=88)
    print('SHEET', out, S.size)


run = sys.argv[1] if len(sys.argv) > 1 else F + '/runs/v3_final'
grid([('v3', 'graze', 'v3 (installed) grazing light'), ('nofit', 'graze', 'no-fit grazing (reference)'),
      ('v3', 'zebra', 'v3 zebra'), ('nofit', 'zebra', 'no-fit zebra (reference)')],
     'final_sheet: Grip_smooth.blend v3 (SPEC v2 low-frequency fit, review round 2), right handle, z <= 41; reference = no-fit '
     '(plain GripV through the same remesh); top: check.py views of the same run', F + '/final_sheet.jpg', extra_top=os.path.join(run, 'sheet.jpg'))
grid([('nofit', 'graze', 'no-fit (plain GripV, same remesh)'), ('v1', 'graze', 'v1 (Grip_smooth_v1.blend)'),
      ('v2', 'graze', 'v2 (Grip_smooth_v2.blend)'), ('v3', 'graze', 'v3 (Grip_smooth.blend)')],
     'compare_sheet: grazing light, same views and light; no-fit vs v1 vs v2 vs v3; bottom: top-lit close-ups of the outer face',
     F + '/compare_sheet.jpg', extra_bottom=closeup_band(tw * 4))
