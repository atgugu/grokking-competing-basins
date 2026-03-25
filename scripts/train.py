"""Train a single quadratic network on modular addition."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import Trainer, TrainConfig
from src.utils import get_device


def main():
    parser = argparse.ArgumentParser(description="Train QuadraticNet on modular addition")
    parser.add_argument("--p", type=int, default=53)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", "--weight-decay", type=float, default=1e-5, dest="wd")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=100_000)
    parser.add_argument("--train-fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    device = args.device or str(get_device())
    output_dir = args.output_dir or f"results/train_p{args.p}_K{args.K}_lr{args.lr}_wd{args.wd}"
    ckpt_dir = os.path.join(output_dir, "checkpoints")

    config = TrainConfig(
        p=args.p,
        K=args.K,
        lr=args.lr,
        weight_decay=args.wd,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        train_fraction=args.train_fraction,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=ckpt_dir,
        device=device,
    )

    print(f"Training QuadraticNet: p={args.p}, K={args.K}, lr={args.lr}, wd={args.wd}")
    print(f"Device: {device}, Output: {output_dir}")
    print(f"Params: 3*{args.p}*{args.K} = {3*args.p*args.K}")

    trainer = Trainer(config)
    history = trainer.train(verbose=not args.quiet)

    # Save history
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump({
            "epochs": history.epochs,
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "train_acc": history.train_acc,
            "val_acc": history.val_acc,
        }, f)
    print(f"\nHistory saved to {history_path}")
    print(f"Final: Train Loss={history.train_loss[-1]:.6f}, "
          f"Val Loss={history.val_loss[-1]:.6f}, "
          f"Train Acc={history.train_acc[-1]:.4f}, "
          f"Val Acc={history.val_acc[-1]:.4f}")


if __name__ == "__main__":
    main()
