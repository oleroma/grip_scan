"""solve_core.py: the posable grip in numpy (no bpy): bone pose -> basemesh GripV surface.

The pose of the grip rig is a short vector x (one entry per free bone channel, see CHANNELS).
Model.surface(x) reproduces what Blender evaluates (armature with the cage's vertex groups, then the
'grip rig pins' mask, then MirrorX + subdivision, which is linear: out = A @ cage), so the solver can
run thousands of poses a second without Blender. export_bones.py checks it against Blender.
"""
import os

import numpy as np

import rig_core as rc

CAGE = "basemesh GripV"
RIG = "grip rig"

# (bone, channel, scale): the scale is a "typical" amount, used to normalise the solver variables
#   loc x/y/z: bone-local translation in mm; rot x/z: bone-local rotation in radians
#   (rot x swings a fin tip along its outward surface normal, rot z swings it sideways)
CHANNELS = [
    (rc.PALM, "loc", 0, 4.0), (rc.PALM, "loc", 1, 4.0), (rc.PALM, "loc", 2, 4.0),
    ("fin middle", "rot", 0, None), ("fin middle", "rot", 2, None),
    ("fin ring", "rot", 0, None), ("fin ring", "rot", 2, None),
    ("fin pinky", "rot", 0, None), ("fin pinky", "rot", 2, None),
    (rc.STAND, "rot", 0, None), (rc.STAND, "rot", 2, None),
    (rc.THENAR, "loc", 0, 3.0), (rc.THENAR, "loc", 1, 3.0), (rc.THENAR, "loc", 2, 3.0),
]
TIP_MM = 4.0      # rotation channels are scaled so one unit swings the bone tip by about this much


def euler_xyz(r):
    cx, cy, cz = np.cos(r)
    sx, sy, sz = np.sin(r)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class Model:
    def __init__(self, work):
        st = np.load(os.path.join(work, "rig_state.npz"))
        self.A, self.S0, self.T, self.co = st["A"], st["S0"], st["T"], st["co"]
        self.mask = st["mask"]
        self.Ax = np.load(os.path.join(work, "Ax.npy"))
        self.W = {k[2:]: st[k] for k in st.files if k.startswith("W|")}
        self.out_in_zone = st["out_in_zone"]
        self.out_free = st["out_free"]
        b = np.load(os.path.join(work, "bones.npz"))
        self.names = list(b["names"])
        self.parent = list(b["parent"])
        self.B = {n: b["B"][i] for i, n in enumerate(self.names)}
        self.length = {n: float(b["length"][i]) for i, n in enumerate(self.names)}
        self.Binv = {n: np.linalg.inv(m) for n, m in self.B.items()}
        assert np.allclose(b["rig_world"], np.eye(4)), "rig object must sit at the origin"
        self.n = len(CHANNELS)
        sc = []
        for bone, kind, ax, s in CHANNELS:
            if kind == "rot":
                s = TIP_MM / self.length[bone]
            sc.append(s)
        self.scale = np.array(sc)

    def basis(self, x):
        """Pose basis (location, euler) per bone from the pose vector x (real units)."""
        P = {n: [np.zeros(3), np.zeros(3)] for n in self.names}
        for v, (bone, kind, ax, _) in zip(x, CHANNELS):
            P[bone][0 if kind == "loc" else 1][ax] = v
        return P

    def deform_matrices(self, x):
        P = self.basis(x)
        M = {}
        for i, n in enumerate(self.names):          # bones are stored parents first
            loc, rot = P[n]
            Pb = np.eye(4)
            Pb[:3, :3] = euler_xyz(rot)
            Pb[:3, 3] = loc
            p = self.parent[i]
            if p < 0:
                M[n] = self.B[n] @ Pb
            else:
                pn = self.names[p]
                M[n] = M[pn] @ self.Binv[pn] @ self.B[n] @ Pb
        return {n: M[n] @ self.Binv[n] for n in self.names}

    def cage(self, x):
        posed = rc.lbs(self.co, self.W, self.deform_matrices(x))
        d = (posed - self.co) * self.mask
        d[np.linalg.norm(d, axis=1) < 1e-4] = 0.0
        return self.co + d

    def surface(self, x):
        c = self.cage(x)
        S = self.A @ c
        S[:, 0] = self.Ax @ c[:, 0]
        return S

    def apply_to_rig(self, rig, x):
        """Put pose x on the Blender armature object."""
        import bpy
        for pb in rig.pose.bones:
            pb.location = (0, 0, 0)
            pb.rotation_euler = (0, 0, 0)
        for n, (loc, rot) in self.basis(x).items():
            pb = rig.pose.bones[n]
            pb.rotation_mode = "XYZ"
            pb.location = tuple(float(v) for v in loc)
            pb.rotation_euler = tuple(float(v) for v in rot)
        bpy.context.view_layer.update()
