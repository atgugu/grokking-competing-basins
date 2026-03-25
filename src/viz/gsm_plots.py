"""GSM and related plots (Figures 4, 11-12)."""

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, MARKERS


def plot_gsm_vs_lr(
    lr_values: list[float],
    gsm_values: list[float],
    title: str = "GSM vs Learning Rate",
    save_path: str | None = None,
) -> plt.Figure:
    """Figure 4: GSM vs learning rate."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(lr_values, gsm_values, color=COLORS["train"], s=60, zorder=5)
    ax.plot(lr_values, gsm_values, color=COLORS["train"], alpha=0.5)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("GSM")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_gsm_vs_lr_multi_wd(
    results: dict[float, tuple[list[float], list[float]]],
    title: str = "GSM vs Learning Rate",
    save_path: str | None = None,
) -> plt.Figure:
    """Figure 11: GSM vs lr for multiple weight decay values.

    Args:
        results: {wd: (lr_values, gsm_values)} mapping
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS["train"], COLORS["val"], COLORS["llc"]]

    for i, (wd, (lrs, gsms)) in enumerate(sorted(results.items())):
        c = colors[i % len(colors)]
        m = MARKERS[i % len(MARKERS)]
        label = f"wd={wd}" if wd > 0 else "wd=0"
        ax.scatter(lrs, gsms, color=c, marker=m, s=60, zorder=5, label=label)
        ax.plot(lrs, gsms, color=c, alpha=0.5)

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("GSM")
    ax.legend()
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_max_llc_vs_lr(
    results: dict[float, tuple[list[float], list[float]]],
    title: str = "Max LLC vs Learning Rate",
    save_path: str | None = None,
) -> plt.Figure:
    """Figure 12: Max LLC vs lr for multiple weight decay values.

    Args:
        results: {wd: (lr_values, max_llc_values)} mapping
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS["train"], COLORS["val"], COLORS["llc"]]

    for i, (wd, (lrs, llcs)) in enumerate(sorted(results.items())):
        c = colors[i % len(colors)]
        m = MARKERS[i % len(MARKERS)]
        label = f"wd={wd}" if wd > 0 else "wd=0"
        ax.scatter(lrs, llcs, color=c, marker=m, s=60, zorder=5, label=label)
        ax.plot(lrs, llcs, color=c, alpha=0.5)

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Max LLC")
    ax.legend()
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig
