"""Training curve plots (Figure 3 and Figures 7-10)."""

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS


def plot_loss_and_llc(
    epochs: list[int],
    train_loss: list[float],
    val_loss: list[float],
    llc_epochs: list[int] | None = None,
    llc_values: list[float] | None = None,
    title: str = "",
    save_path: str | None = None,
    log_scale_loss: bool = True,
) -> plt.Figure:
    """Dual-axis plot: loss (left) and LLC (right).

    Reproduces Figure 3 and Figures 7-10.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Loss curves
    ax1.plot(epochs, train_loss, color=COLORS["train"], label="Train Loss", alpha=0.8)
    ax1.plot(epochs, val_loss, color=COLORS["val"], label="Val Loss", alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    if log_scale_loss:
        ax1.set_yscale("log")
    ax1.tick_params(axis="y")

    # LLC on right axis
    if llc_epochs is not None and llc_values is not None:
        ax2 = ax1.twinx()
        ax2.plot(llc_epochs, llc_values, color=COLORS["llc"], label="LLC",
                 linewidth=2.5, alpha=0.9)
        ax2.set_ylabel("LLC", color=COLORS["llc"])
        ax2.tick_params(axis="y", labelcolor=COLORS["llc"])
        ax2.spines["right"].set_visible(True)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend()

    if title:
        ax1.set_title(title)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_accuracy_curves(
    epochs: list[int],
    train_acc: list[float],
    val_acc: list[float],
    title: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot training and validation accuracy."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_acc, color=COLORS["train"], label="Train Acc")
    ax.plot(epochs, val_acc, color=COLORS["val"], label="Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.95, color=COLORS["gray"], linestyle="--", alpha=0.5, label="95%")
    ax.legend()
    if title:
        ax.set_title(title)
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_multi_loss_llc(
    results: list[dict],
    labels: list[str],
    title: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot multiple runs on same axes (for robustness figures 7-10).

    Each result dict should have: epochs, train_loss, val_loss, llc_epochs, llc_values.
    """
    colors = list(COLORS.values())
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    for i, (res, label) in enumerate(zip(results, labels)):
        c = colors[i % len(colors)]
        ax1.plot(res["epochs"], res["train_loss"], color=c, linestyle="-",
                 alpha=0.6, label=f"{label} Train")
        ax1.plot(res["epochs"], res["val_loss"], color=c, linestyle="--",
                 alpha=0.6, label=f"{label} Val")
        if "llc_epochs" in res and res["llc_epochs"]:
            ax2.plot(res["llc_epochs"], res["llc_values"], color=c,
                     linestyle=":", linewidth=2.5, alpha=0.9, label=f"{label} LLC")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax2.set_ylabel("LLC")
    ax2.spines["right"].set_visible(True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    if title:
        ax1.set_title(title)
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig
