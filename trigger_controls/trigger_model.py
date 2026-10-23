"""Numpy copy of the GripV R2/R4 arched trigger placement (no Blender needed).

The trigger is an arch: a ring of outer radius R about an axis that lies along
the trigger-local X. In the trigger frame the SMD contact point is at arc
angle -tal/R, the top (press) face at angle a = (L - top bevel - tal) / R,
where tal = (pi/2 - asin(z/R)) * R. 'trigger align' then moves the frame so the
axis sits at (x offset, sqrt(R^2 - z^2) - button radius offset, z) above the
switch (lowered by 1.6 + pressed height) and turns it about Z by
'button rotation'.

Button top centre = the middle of the top face (half the trigger height in from
the outer arc, at x = 0) pushed out by the top bevel along the face normal. The
face normal comes from the 'tip angle offset' / 'tip shear' cut in
'trigger cut'.

forward(): original parameters -> button top centre.
inverse(): button top centre + arch radius -> button rotation, shift z,
           trigger length. Same math as the 'trigger top solve' node group.

Parameter dict keys: R z rot L tb h bro ph xoff o sh
(trigger radius, shift z, button rotation [rad], trigger length, top bevel,
height, button radius offset, button pressed height, x offset,
tip angle offset [rad], tip shear).
"""
import numpy as np

SMD_DROP = 1.6


def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def face_normal(a, o, sh):
    v = np.array([-sh, 1.0, -np.tan(a - o)])
    return rx(a) @ (v / np.linalg.norm(v))


def forward(p, smd, drop=SMD_DROP):
    """-> (button top centre, top face normal, tip angle), world mm."""
    R, z, rot, L = p["R"], p["z"], p["rot"], p["L"]
    tb, h, bro, ph, xo = p["tb"], p["h"], p["bro"], p["ph"], p["xoff"]
    tal = (np.pi / 2 - np.arcsin(z / R)) * R
    a = (L - tb - tal) / R
    n = face_normal(a, p["o"], p["sh"])
    local = rx(a) @ np.array([0, 0, -(R - h / 2)]) + tb * n
    base = np.asarray(smd, float) - np.array([0, 0, drop + ph])
    axis = np.array([xo, np.sqrt(R * R - z * z) - bro, z])
    return base + rz(rot) @ (axis + local), rz(rot) @ n, a


def inverse(top, R, smd, p, iters=8, drop=SMD_DROP):
    """button top centre + arch radius -> dict(rot, z, L, a).

    The arch axis C lies on the circle of radius R round the switch contact
    and on the circle of radius R - h/2 round the (bevel-corrected) top.
    """
    tb, h, bro, ph, xo = p["tb"], p["h"], p["bro"], p["ph"], p["xoff"]
    base = np.asarray(smd, float) - np.array([0, 0, drop + ph])
    d = np.asarray(top, float) - base
    rho = R - h / 2
    a = 0.0
    for _ in range(iters):
        n = face_normal(a, p["o"], p["sh"])
        qx = xo + tb * n[0]
        qy = np.sqrt(max(d[0] ** 2 + d[1] ** 2 - qx ** 2, 0.0))
        rot = np.arctan2(d[1], d[0]) - np.arctan2(qy, qx)
        U = qy - tb * n[1] + bro
        V = d[2] - tb * n[2]
        W = max(np.hypot(U, V), 1e-6)
        xx = (R * R - rho * rho + W * W) / (2 * W)
        yy = np.sqrt(max(R * R - xx * xx, 0.0))
        cy = (xx * U - yy * V) / W
        cz = (xx * V + yy * U) / W
        a = np.arctan2(U - cy, cz - V)
    tal = (np.pi / 2 - np.arcsin(np.clip(cz / R, -1, 1))) * R
    return dict(rot=rot, z=cz, L=tal + R * a + tb, a=a)
