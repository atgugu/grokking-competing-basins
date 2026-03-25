"""Animated GIF of training progress."""

import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS


def create_training_gif(
    epochs: list[int],
    train_loss: list[float],
    val_loss: list[float],
    train_acc: list[float],
    val_acc: list[float],
    llc_epochs: list[int] | None = None,
    llc_values: list[float] | None = None,
    save_path: str = "results/training_animation.gif",
    n_frames: int = 60,
    duration_ms: int = 100,
):
    """Create an animated GIF showing training progress.

    Uses PIL to avoid heavy ffmpeg dependency.
    """
    from PIL import Image

    total_pts = len(epochs)
    frame_indices = np.linspace(10, total_pts - 1, n_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left panel: Loss
        ax1 = axes[0]
        ax1.plot(epochs[:idx+1], train_loss[:idx+1], color=COLORS["train"],
                 label="Train Loss")
        ax1.plot(epochs[:idx+1], val_loss[:idx+1], color=COLORS["val"],
                 label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_yscale("log")
        ax1.set_xlim(0, epochs[-1])
        ax1.set_ylim(min(min(train_loss), min(val_loss)) * 0.5,
                     max(max(train_loss), max(val_loss)) * 2)
        ax1.legend(loc="upper right")
        ax1.set_title(f"Epoch {epochs[idx]}")

        if llc_epochs is not None and llc_values is not None:
            # Find LLC points up to current epoch
            llc_mask = [i for i, e in enumerate(llc_epochs) if e <= epochs[idx]]
            if llc_mask:
                ax1_r = ax1.twinx()
                llc_e = [llc_epochs[i] for i in llc_mask]
                llc_v = [llc_values[i] for i in llc_mask]
                ax1_r.plot(llc_e, llc_v, color=COLORS["llc"], linewidth=2.5,
                           label="LLC")
                ax1_r.set_ylabel("LLC", color=COLORS["llc"])
                ax1_r.set_ylim(min(llc_values) * 0.9, max(llc_values) * 1.1)
                ax1_r.spines["right"].set_visible(True)

        # Right panel: Accuracy
        ax2 = axes[1]
        ax2.plot(epochs[:idx+1], train_acc[:idx+1], color=COLORS["train"],
                 label="Train Acc")
        ax2.plot(epochs[:idx+1], val_acc[:idx+1], color=COLORS["val"],
                 label="Val Acc")
        ax2.axhline(y=0.95, color=COLORS["gray"], linestyle="--", alpha=0.5)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_xlim(0, epochs[-1])
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(loc="lower right")
        ax2.set_title("Accuracy")

        fig.suptitle("Grokking in Modular Addition", fontsize=14)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Animation saved to {save_path}")
