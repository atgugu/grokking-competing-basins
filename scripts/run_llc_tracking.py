"""Figure 3: LLC tracks generalization during training.

p=53, K=1024, lr=0.0001, wd=0.0001, bs=128
Train model, then estimate LLC at periodic checkpoints.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import Trainer, TrainConfig
from src.training.checkpointing import list_checkpoints, load_model_from_checkpoint
from src.data.modular_arithmetic import ModularArithmeticDataset
from src.analysis.llc_estimation import estimate_llc
from src.viz.training_curves import plot_loss_and_llc
from src.viz.animation import create_training_gif
from src.viz.style import setup_style
from src.utils import get_device

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llc-interval", type=int, default=500,
                        help="Estimate LLC every N epochs")
    parser.add_argument("--max-epochs", type=int, default=100_000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only compute LLC from existing checkpoints")
    args = parser.parse_args()

    setup_style()
    device = args.device or str(get_device())
    output_dir = "results/llc_tracking"
    os.makedirs(output_dir, exist_ok=True)

    p, K = 53, 1024

    # Step 1: Train
    if not args.skip_training:
        config = TrainConfig(
            p=p, K=K, lr=0.001, weight_decay=1e-5,
            batch_size=128, max_epochs=args.max_epochs,
            train_fraction=0.4, seed=0,
            checkpoint_interval=100,
            checkpoint_dir=os.path.join(output_dir, "checkpoints"),
            device=device,
        )
        print("Step 1: Training...")
        trainer = Trainer(config)
        history = trainer.train(verbose=True)

        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({
                "epochs": history.epochs,
                "train_loss": history.train_loss,
                "val_loss": history.val_loss,
                "train_acc": history.train_acc,
                "val_acc": history.val_acc,
            }, f)
    else:
        with open(os.path.join(output_dir, "history.json")) as f:
            hist_data = json.load(f)
        class H:
            pass
        history = H()
        for k, v in hist_data.items():
            setattr(history, k, v)

    # Step 2: Estimate LLC at periodic checkpoints
    print("\nStep 2: Estimating LLC at checkpoints...")
    dataset = ModularArithmeticDataset(p, train_fraction=0.4, seed=0)
    x_train, y_train = dataset.full_train_batch(device)

    checkpoints = list_checkpoints(os.path.join(output_dir, "checkpoints"))
    llc_epochs = []
    llc_values = []
    llc_stds = []

    for epoch, ckpt_path in checkpoints:
        if epoch % args.llc_interval != 0:
            continue
        print(f"  Epoch {epoch}...", end=" ", flush=True)
        model = load_model_from_checkpoint(ckpt_path, device)
        result = estimate_llc(
            model, x_train, y_train,
            device=device, localization=5.0,
        )
        llc_epochs.append(epoch)
        llc_values.append(result["llc_mean"])
        llc_stds.append(result["llc_std"])
        print(f"LLC = {result['llc_mean']:.2f} ± {result['llc_std']:.2f}")

    # Save LLC results
    with open(os.path.join(output_dir, "llc_results.json"), "w") as f:
        json.dump({
            "llc_epochs": llc_epochs,
            "llc_values": llc_values,
            "llc_stds": llc_stds,
        }, f)

    # Step 3: Plot Figure 3
    fig = plot_loss_and_llc(
        history.epochs, history.train_loss, history.val_loss,
        llc_epochs, llc_values,
        title=f"LLC Tracks Generalization (p={p}, K={K})",
        save_path=os.path.join(output_dir, "figure3_llc_tracking.png"),
    )
    print(f"\nFigure 3 saved to {output_dir}/figure3_llc_tracking.png")

    # Step 4: Create animation
    try:
        create_training_gif(
            history.epochs, history.train_loss, history.val_loss,
            history.train_acc, history.val_acc,
            llc_epochs, llc_values,
            save_path=os.path.join(output_dir, "training_animation.gif"),
        )
    except ImportError:
        print("Pillow not installed, skipping animation")


if __name__ == "__main__":
    main()
