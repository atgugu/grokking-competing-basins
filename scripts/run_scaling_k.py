"""Figure 2: LLC vs K for different primes.

K values: {200, 400, 600, 800, 1000, 1200, 1400, 1600}
Primes: {53, 61, 71}
Uses H.1 baseline: wd=1e-5, tf=0.4, bs=128, lr=0.0001
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import Trainer, TrainConfig
from src.analysis.llc_estimation import estimate_llc
from src.analysis.scaling_laws import fit_linear
from src.viz.scaling_plots import plot_llc_vs_K
from src.viz.style import setup_style
from src.utils import get_device


def main():
    setup_style()
    device = str(get_device())
    output_dir = "results/scaling_K"
    os.makedirs(output_dir, exist_ok=True)

    K_values = [200, 400, 600, 800, 1000, 1200, 1400, 1600]
    primes = [53, 61, 71]

    all_results = {}

    for p in primes:
        llc_values = []
        for K in K_values:
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

        fit = fit_linear(K_values, llc_values)
        print(f"\np={p}: {fit}")
        all_results[f"fit_p{p}"] = {
            "slope": fit.slope, "intercept": fit.intercept,
            "r_squared": fit.r_squared,
        }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    fits = {}
    for p in primes:
        k_vals = [all_results[f"p{p}_K{K}"]["K"] for K in K_values]
        llc_vals = [all_results[f"p{p}_K{K}"]["llc_mean"] for K in K_values]
        fits[p] = fit_linear(k_vals, llc_vals)

    plot_llc_vs_K(fits, save_path=os.path.join(output_dir, "figure2_llc_vs_K.png"))
    print(f"\nFigure saved to {output_dir}/figure2_llc_vs_K.png")


if __name__ == "__main__":
    main()
