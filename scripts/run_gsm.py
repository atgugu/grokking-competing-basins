"""Figure 4: GSM vs learning rate.
Figures 11-12: GSM and Max LLC vs LR for multiple weight decays.

p=53, K=1024, wd=0.0001 (Fig 4)
LR values: 0.0001 to 0.01
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import Trainer, TrainConfig
from src.analysis.grokking_severity import compute_gsm
from src.viz.gsm_plots import plot_gsm_vs_lr, plot_gsm_vs_lr_multi_wd, plot_max_llc_vs_lr
from src.viz.style import setup_style
from src.utils import get_device

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=100_000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--figure", choices=["4", "11", "12", "all"], default="all")
    args = parser.parse_args()

    setup_style()
    device = args.device or str(get_device())
    output_dir = "results/gsm"
    os.makedirs(output_dir, exist_ok=True)

    p, K = 53, 1024

    # Figure 4: single wd
    if args.figure in ("4", "all"):
        lr_values = [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005,
                     0.006, 0.007, 0.008, 0.009, 0.01]
        wd = 1e-5
        gsm_values = []

        for lr in lr_values:
            print(f"\nTraining lr={lr}, wd={wd}...")
            config = TrainConfig(
                p=p, K=K, lr=lr, weight_decay=wd,
                batch_size=128, max_epochs=args.max_epochs,
                train_fraction=0.4, seed=0,
                checkpoint_interval=100,
                checkpoint_dir=os.path.join(output_dir, f"lr{lr}_wd{wd}", "ckpt"),
                device=device,
            )
            trainer = Trainer(config)
            history = trainer.train(verbose=False)
            gsm = compute_gsm(history.train_acc, history.val_acc)
            gsm_values.append(gsm)
            print(f"  GSM = {gsm:.4f}, Final val acc = {history.val_acc[-1]:.4f}")

        plot_gsm_vs_lr(lr_values, gsm_values,
                       title=f"GSM vs Learning Rate (p={p}, K={K}, wd={wd})",
                       save_path=os.path.join(output_dir, "figure4_gsm_vs_lr.png"))

        with open(os.path.join(output_dir, "figure4_results.json"), "w") as f:
            json.dump({"lr_values": lr_values, "gsm_values": gsm_values, "wd": wd}, f)

    # Figures 11-12: multiple wd values
    if args.figure in ("11", "12", "all"):
        wd_values = [0, 1e-6, 1e-5]
        lr_values = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]

        gsm_results = {}
        max_llc_results = {}

        for wd in wd_values:
            lrs, gsms = [], []
            for lr in lr_values:
                print(f"\nTraining lr={lr}, wd={wd}...")
                config = TrainConfig(
                    p=p, K=K, lr=lr, weight_decay=wd,
                    batch_size=128, max_epochs=args.max_epochs,
                    train_fraction=0.4, seed=0,
                    checkpoint_interval=100,
                    checkpoint_dir=os.path.join(output_dir, f"lr{lr}_wd{wd}", "ckpt"),
                    device=device,
                )
                trainer = Trainer(config)
                history = trainer.train(verbose=False)
                gsm = compute_gsm(history.train_acc, history.val_acc)
                lrs.append(lr)
                gsms.append(gsm)
                print(f"  GSM = {gsm:.4f}")

            gsm_results[wd] = (lrs, gsms)
            # Max LLC would require LLC computation at each checkpoint
            # Placeholder: proportional to GSM for now
            max_llc_results[wd] = (lrs, gsms)

        plot_gsm_vs_lr_multi_wd(gsm_results,
                                title="GSM vs LR (multiple weight decays)",
                                save_path=os.path.join(output_dir, "figure11_gsm_multi_wd.png"))


if __name__ == "__main__":
    main()
