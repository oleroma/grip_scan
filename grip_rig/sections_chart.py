"""sections_chart.py: outlines of several solved grips on top of each other, so millimetre differences show.

    $PY sections_chart.py --out sections.png  LABEL=WORKDIR [LABEL=WORKDIR ...]

Each WORKDIR holds posed_meshes.npz (pose_in_blender.py) and optionally hand_factors.json.
Panels: side profile of the right handle (cut at x = 65 mm, through the palm side) and three
horizontal cuts across the handle. Original GripV = dashed grey.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY, INK, INK2 = "#8a8985", "#0b0b0b", "#52514e"
CUTS = [  # title, plane axis, value, (u axis, v axis), u range, v range
    ("Side profile (cut through the palm side, x = 65 mm)", 0, 65.0, (1, 2), (-100, -5), (-25, 60)),
    ("Horizontal cut, high (z = 15 mm)", 2, 15.0, (0, 1), (25, 95), (-100, -5)),
    ("Horizontal cut, middle (z = 5 mm)", 2, 5.0, (0, 1), (25, 95), (-100, -5)),
    ("Horizontal cut, low (z = -5 mm)", 2, -5.0, (0, 1), (25, 95), (-100, -5)),
]


def slice_mesh(V, T, axis, val):
    d = V[:, axis] - val
    segs = []
    for tri in T:
        dd = d[tri]
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = dd[i], dd[j]
            if (a < 0) != (b < 0):
                t = a / (a - b)
                pts.append(V[tri[i]] + t * (V[tri[j]] - V[tri[i]]))
        if len(pts) == 2:
            segs.append(pts)
    return np.array(segs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    runs = []
    for r in a.runs:
        lab, w = r.split("=", 1)
        pm = np.load(os.path.join(w, "posed_meshes.npz"))
        runs.append((lab, pm["posed"], pm["T"], pm["rest"]))
    rest, T = runs[0][3], runs[0][2]
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 15})
    fig, axs = plt.subplots(2, 2, figsize=(16, 15), facecolor="#fcfcfb")
    for ax, (title, axis, val, (iu, iv), ur, vr) in zip(axs.ravel(), CUTS):
        ax.set_facecolor("#fcfcfb")
        s = slice_mesh(rest, T, axis, val)
        if len(s):
            ax.add_collection(LineCollection(s[:, :, [iu, iv]], colors=GREY, linewidths=3.0, linestyles=(0, (4, 3))))
        for k, (lab, V, Tk, _) in enumerate(runs):
            s = slice_mesh(V, Tk, axis, val)
            if len(s):
                ax.add_collection(LineCollection(s[:, :, [iu, iv]], colors=SERIES[k], linewidths=2.0))
        ax.set_xlim(*ur)
        ax.set_ylim(*vr)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=18, color=INK, loc="left")
        names = "xyz"
        ax.set_xlabel(f"{names[iu]} (mm)" + ("   front of the controller is to the right" if iu == 1 else
                                                  "   outside of the handle (palm) is to the right"), color=INK2)
        ax.set_ylabel(f"{names[iv]} (mm)", color=INK2)
        ax.grid(color="#e4e3df", linewidth=0.8)
        for sp in ax.spines.values():
            sp.set_color("#c3c2b7")
        ax.tick_params(colors=INK2)
    handles = [plt.Line2D([], [], color=GREY, lw=3, ls=(0, (4, 3)), label="Original GripV")]
    handles += [plt.Line2D([], [], color=SERIES[k], lw=3, label=lab) for k, (lab, *_) in enumerate(runs)]
    fig.legend(handles=handles, loc="upper center", ncol=min(2, len(handles)), fontsize=15, frameon=False,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Where each solved grip's surface sits, cut through the right handle "
                 "(lines further out = grip grew there)", y=0.895, fontsize=19, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(a.out, dpi=110, facecolor=fig.get_facecolor())
    print("[chart] wrote", a.out)


if __name__ == "__main__":
    main()
