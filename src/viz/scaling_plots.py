"""Scaling law plots (Figures 1-2)."""

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.scaling_laws import LinearFit
from .style import COLORS, MARKERS


def plot_llc_vs_p(
    fits: dict[int, LinearFit],
    title: str = "Final LLC vs Prime Number p",
    save_path: str | None = None,
) -> plt.Figure:
    """Figure 1: LLC vs p for different K values.

    Args:
        fits: {K_value: LinearFit} mapping
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    color_keys = ["K600", "K1000", "K1400"]

    for i, (K, fit) in enumerate(sorted(fits.items())):
        color = COLORS.get(f"K{K}", COLORS.get(color_keys[i % len(color_keys)]))
        marker = MARKERS[i % len(MARKERS)]

        # Data points
        ax.scatter(fit.x, fit.y, color=color, marker=marker, s=60, zorder=5,
                   label=f"K={K}")

        # Regression line
        x_line = np.linspace(fit.x.min() - 2, fit.x.max() + 2, 100)
        y_line = fit.predict(x_line)
        ax.plot(x_line, y_line, color=color, linestyle="--", alpha=0.7,
                label=f"  y={fit.slope:.2f}x{fit.intercept:+.2f}, R²={fit.r_squared:.3f}")

    ax.set_xlabel("Prime Number (p)")
    ax.set_ylabel("Final LLC")
    ax.legend(fontsize=10)
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_llc_vs_K(
    fits: dict[int, LinearFit],
    title: str = "Final LLC vs Hidden Dimension K",
    save_path: str | None = None,
) -> plt.Figure:
    """Figure 2: LLC vs K for different primes.

    Args:
        fits: {p_value: LinearFit} mapping
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    color_keys = ["p53", "p61", "p71"]

    for i, (p, fit) in enumerate(sorted(fits.items())):
        color = COLORS.get(f"p{p}", COLORS.get(color_keys[i % len(color_keys)]))
        marker = MARKERS[i % len(MARKERS)]

        ax.scatter(fit.x, fit.y, color=color, marker=marker, s=60, zorder=5,
                   label=f"p={p}")

        x_line = np.linspace(fit.x.min() - 50, fit.x.max() + 50, 100)
        y_line = fit.predict(x_line)
        ax.plot(x_line, y_line, color=color, linestyle="--", alpha=0.7,
                label=f"  y={fit.slope:.2f}x{fit.intercept:+.2f}, R²={fit.r_squared:.3f}")

    ax.set_xlabel("Hidden Dimension (K)")
    ax.set_ylabel("Final LLC")
    ax.legend(fontsize=10)
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig
