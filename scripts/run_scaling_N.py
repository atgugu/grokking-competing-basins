"""Figure 14: Scaling N/(M log M) — regime classification.

M (group sizes): {53, 67, 79, 83, 101, 103}
Training fractions: {0.1, 0.15, 0.3, 0.4, 0.9, 0.95}
Classify as: no_generalization, grokking, immediate_generalization
Critical transition at N/(M log M) ~ 2.5-3
"""

import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import Trainer, TrainConfig
from src.analysis.grokking_severity import classify_regime
from src.viz.style import setup_style, COLORS
from src.utils import get_device


def main():
    setup_style()
    device = str(get_device())
    output_dir = "results/scaling_N"
    os.makedirs(output_dir, exist_ok=True)

    M_values = [53, 67, 79, 83, 101, 103]
    train_fractions = [0.1, 0.15, 0.3, 0.4, 0.9, 0.95]
    K = 1024

    results = []

    for M in M_values:
        for tf in train_fractions:
            N = int(tf * M * M)
            ratio = N / (M * math.log(M))

            print(f"\nM={M}, tf={tf}, N={N}, N/(M log M) = {ratio:.2f}")

            config = TrainConfig(
                p=M, K=K, lr=0.001, weight_decay=1e-5,
                batch_size=128, max_epochs=50_000,
                train_fraction=tf, seed=0,
                checkpoint_interval=100,
                checkpoint_dir=os.path.join(output_dir, f"M{M}_tf{tf}", "ckpt"),
                device=device,
            )
            trainer = Trainer(config)
            history = trainer.train(verbose=False)

            regime = classify_regime(history.train_acc, history.val_acc)
            print(f"  Regime: {regime}, Final val acc: {history.val_acc[-1]:.4f}")

            results.append({
                "M": M, "tf": tf, "N": N, "ratio": ratio,
                "regime": regime,
                "final_val_acc": history.val_acc[-1],
                "final_train_acc": history.train_acc[-1],
            })

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot Figure 14
    regime_colors = {
        "no_generalization": "#F44336",
        "grokking": "#FF9800",
        "immediate_generalization": "#4CAF50",
    }
    regime_markers = {
        "no_generalization": "x",
        "grokking": "o",
        "immediate_generalization": "s",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for regime_type in ["no_generalization", "grokking", "immediate_generalization"]:
        pts = [r for r in results if r["regime"] == regime_type]
        if pts:
            ratios = [r["ratio"] for r in pts]
            M_vals = [r["M"] for r in pts]
            label = regime_type.replace("_", " ").title()
            ax.scatter(ratios, M_vals,
                       c=regime_colors[regime_type],
                       marker=regime_markers[regime_type],
                       s=100, label=label, zorder=5)

    # Critical line
    ax.axvline(x=2.75, color=COLORS["gray"], linestyle="--", alpha=0.7,
               label="N/(M log M) ~ 2.75")

    ax.set_xlabel("N / (M log M)")
    ax.set_ylabel("Group Size M")
    ax.legend()
    ax.set_title("Regime Classification: N/(M log M) Scaling")

    fig.savefig(os.path.join(output_dir, "figure14_scaling_N.png"))
    plt.close(fig)
    print(f"\nFigure 14 saved to {output_dir}/figure14_scaling_N.png")


if __name__ == "__main__":
    main()
