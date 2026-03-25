"""Figures 7-10: Robustness of LLC tracking.

Fig 7: p in {53, 61, 71}, K=1024, lr=0.0001, wd=0.0001
Fig 8: K in {600, 800, 1000}, p=53, lr=0.0001, wd=0.0001
Fig 9: wd in {0.0001, 0.00005, 0.00001}, lr=0.0001
Fig 10: lr in {0.0001, 0.001, 0.01}, wd=0.0001
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import Trainer, TrainConfig
from src.training.checkpointing import list_checkpoints, load_model_from_checkpoint
from src.data.modular_arithmetic import ModularArithmeticDataset
from src.analysis.llc_estimation import estimate_llc
from src.viz.training_curves import plot_multi_loss_llc
from src.viz.style import setup_style
from src.utils import get_device

import argparse


def train_and_track_llc(p, K, lr, wd, device, output_dir,
                         max_epochs=100_000, llc_interval=1000):
    """Train and estimate LLC at intervals."""
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    config = TrainConfig(
        p=p, K=K, lr=lr, weight_decay=wd,
        batch_size=128, max_epochs=max_epochs,
        train_fraction=0.4, seed=0,
        checkpoint_interval=100,
        checkpoint_dir=ckpt_dir, device=device,
    )
    trainer = Trainer(config)
    history = trainer.train(verbose=True)

    # LLC at intervals
    dataset = ModularArithmeticDataset(p, train_fraction=0.4, seed=0)
    x_train, y_train = dataset.full_train_batch(device)
    checkpoints = list_checkpoints(ckpt_dir)

    llc_epochs, llc_values = [], []
    for epoch, path in checkpoints:
        if epoch % llc_interval != 0:
            continue
        model = load_model_from_checkpoint(path, device)
        result = estimate_llc(model, x_train, y_train, device=device,
                              localization=5.0)
        llc_epochs.append(epoch)
        llc_values.append(result["llc_mean"])
        print(f"  Epoch {epoch}: LLC = {result['llc_mean']:.2f}")

    return {
        "epochs": history.epochs,
        "train_loss": history.train_loss,
        "val_loss": history.val_loss,
        "llc_epochs": llc_epochs,
        "llc_values": llc_values,
        "p": p, "K": K, "lr": lr, "wd": wd,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=["7", "8", "9", "10", "all"], default="all")
    parser.add_argument("--max-epochs", type=int, default=100_000)
    parser.add_argument("--llc-interval", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    setup_style()
    device = args.device or str(get_device())
    base_dir = "results/robustness"

    # Figure 7: varying p
    if args.figure in ("7", "all"):
        print("\n" + "="*60 + "\nFigure 7: Varying p\n" + "="*60)
        results = []
        for p in [53, 61, 71]:
            out = os.path.join(base_dir, f"fig7_p{p}")
            os.makedirs(out, exist_ok=True)
            r = train_and_track_llc(p, 1024, 0.001, 1e-5, device, out,
                                     args.max_epochs, args.llc_interval)
            results.append(r)
        plot_multi_loss_llc(results, [f"p={r['p']}" for r in results],
                           title="Robustness: Varying p (K=1024)",
                           save_path=os.path.join(base_dir, "figure7_varying_p.png"))

    # Figure 8: varying K
    if args.figure in ("8", "all"):
        print("\n" + "="*60 + "\nFigure 8: Varying K\n" + "="*60)
        results = []
        for K in [600, 800, 1000]:
            out = os.path.join(base_dir, f"fig8_K{K}")
            os.makedirs(out, exist_ok=True)
            r = train_and_track_llc(53, K, 0.001, 1e-5, device, out,
                                     args.max_epochs, args.llc_interval)
            results.append(r)
        plot_multi_loss_llc(results, [f"K={r['K']}" for r in results],
                           title="Robustness: Varying K (p=53)",
                           save_path=os.path.join(base_dir, "figure8_varying_K.png"))

    # Figure 9: varying wd
    if args.figure in ("9", "all"):
        print("\n" + "="*60 + "\nFigure 9: Varying weight decay\n" + "="*60)
        results = []
        for wd in [0.0001, 0.00005, 0.00001]:
            out = os.path.join(base_dir, f"fig9_wd{wd}")
            os.makedirs(out, exist_ok=True)
            r = train_and_track_llc(53, 1024, 0.001, wd, device, out,
                                     args.max_epochs, args.llc_interval)
            results.append(r)
        plot_multi_loss_llc(results, [f"wd={r['wd']}" for r in results],
                           title="Robustness: Varying Weight Decay",
                           save_path=os.path.join(base_dir, "figure9_varying_wd.png"))

    # Figure 10: varying lr
    if args.figure in ("10", "all"):
        print("\n" + "="*60 + "\nFigure 10: Varying lr\n" + "="*60)
        results = []
        for lr in [0.0001, 0.001, 0.01]:
            out = os.path.join(base_dir, f"fig10_lr{lr}")
            os.makedirs(out, exist_ok=True)
            r = train_and_track_llc(53, 1024, lr, 1e-5, device, out,
                                     args.max_epochs, args.llc_interval)
            results.append(r)
        plot_multi_loss_llc(results, [f"lr={r['lr']}" for r in results],
                           title="Robustness: Varying Learning Rate",
                           save_path=os.path.join(base_dir, "figure10_varying_lr.png"))


if __name__ == "__main__":
    main()
