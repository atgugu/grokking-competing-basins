"""Publication-quality plot styling."""

import matplotlib
import matplotlib.pyplot as plt

# Color palette
COLORS = {
    "train": "#2196F3",       # blue
    "val": "#F44336",         # red
    "llc": "#4CAF50",         # green
    "K600": "#1976D2",        # dark blue
    "K1000": "#F57C00",       # orange
    "K1400": "#388E3C",       # dark green
    "p53": "#1976D2",
    "p61": "#F57C00",
    "p71": "#388E3C",
    "accent1": "#9C27B0",     # purple
    "accent2": "#FF9800",     # amber
    "accent3": "#009688",     # teal
    "gray": "#757575",
}

# Markers for different series
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p"]


def setup_style():
    """Set up publication-quality matplotlib style."""
    matplotlib.use("Agg")
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })
