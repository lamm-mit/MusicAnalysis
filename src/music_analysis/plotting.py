from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_pub_style():
    mpl.rcParams.update({
        "figure.figsize": (6.0, 4.0),
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.6,
        "savefig.bbox": "tight",
        "svg.fonttype": "none",  # keep text as text in SVG
    })


def save_svg(path: str):
    if not path.lower().endswith(".svg"):
        raise ValueError("Output path must be an .svg file")
    plt.savefig(path)
    plt.close()

