"""Matplotlib helpers (Agg backend, print-friendly)."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#1b3a5b", "#b5341f", "#2f7d32", "#c89b2c", "#5c457f", "#777777"]


def style():
    plt.rcParams.update({
        "figure.figsize": (5.0, 3.2), "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3,
        "savefig.bbox": "tight", "savefig.dpi": 200,
    })


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
