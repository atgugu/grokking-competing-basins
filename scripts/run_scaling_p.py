"""Figure 1: LLC vs p for different K values.

Primes: {53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
K values: {600, 1000, 1400}
Uses H.1 baseline: wd=1e-5, tf=0.4, bs=128, lr=0.0001
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import Trainer, TrainConfig
from src.analysis.llc_estimation import estimate_llc
from src.analysis.scaling_laws import fit_linear
from src.viz.scaling_plots import plot_llc_vs_p
from src.viz.style import setup_style
from src.utils import get_device


def main():
    setup_style()
    device = str(get_device())
    output_dir = "results/scaling_p"
    os.makedirs(output_dir, exist_ok=True)

    primes = [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    K_values = [600, 1000, 1400]

    all_results = {}

    for K in K_values:
        llc_values = []
        for p in primes:
            run_key = f"p{p}_K{K}"
            print(f"\n{'='*50}")
            print(f"Training p={p}, K={K}")
            print(f"{'='*50}")

            config = TrainConfig(
                p=p, K=K, lr=0.001, weight_decay=1e-5,
                batch_size=128, max_epochs=100_000,
                train_fraction=0.4, seed=0,
                checkpoint_interval=100,
                checkpoint_dir=os.path.join(output_dir, run_key, "checkpoints"),
                device=device,
            )
            trainer = Trainer(config)
            trainer.train(verbose=True)

            # Estimate LLC at final checkpoint
            print(f"Estimating LLC for p={p}, K={K}...")
            x_train, y_train = trainer.dataset.full_train_batch(device)
            result = estimate_llc(
                trainer.model, x_train, y_train,
                device=device, localization=5.0,
            )
            llc_values.append(result["llc_mean"])
            all_results[run_key] = {
                "p": p, "K": K,
                "llc_mean": result["llc_mean"],
                "llc_std": result["llc_std"],
            }
            print(f"  LLC = {result['llc_mean']:.2f} ± {result['llc_std']:.2f}")

        # Fit linear regression
        fit = fit_linear(primes, llc_values)
        print(f"\nK={K}: {fit}")
        all_results[f"fit_K{K}"] = {
            "slope": fit.slope, "intercept": fit.intercept,
            "r_squared": fit.r_squared,
        }

    # Save results
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot
    fits = {}
    for K in K_values:
        p_vals = [all_results[f"p{p}_K{K}"]["p"] for p in primes]
        llc_vals = [all_results[f"p{p}_K{K}"]["llc_mean"] for p in primes]
        fits[K] = fit_linear(p_vals, llc_vals)

    plot_llc_vs_p(fits, save_path=os.path.join(output_dir, "figure1_llc_vs_p.png"))
    print(f"\nFigure saved to {output_dir}/figure1_llc_vs_p.png")


if __name__ == "__main__":
    main()
