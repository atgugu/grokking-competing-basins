"""Estimate LLC for a single checkpoint."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.llc_estimation import estimate_llc_from_checkpoint
from src.utils import get_device


def main():
    parser = argparse.ArgumentParser(description="Estimate LLC for a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--p", type=int, default=53)
    parser.add_argument("--train-fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-chains", type=int, default=3)
    parser.add_argument("--num-draws", type=int, default=500)
    parser.add_argument("--nbeta", type=float, default=26.4)
    parser.add_argument("--localization", type=float, default=5.0)
    parser.add_argument("--sgld-lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = args.device or str(get_device())

    print(f"Estimating LLC for {args.checkpoint}")
    result = estimate_llc_from_checkpoint(
        args.checkpoint,
        p=args.p,
        train_fraction=args.train_fraction,
        data_seed=args.seed,
        device=device,
        num_chains=args.num_chains,
        num_draws=args.num_draws,
        nbeta=args.nbeta,
        localization=args.localization,
        sgld_lr=args.sgld_lr,
    )

    print(f"LLC = {result['llc_mean']:.4f} ± {result['llc_std']:.4f}")
    print(f"Init loss = {result['init_loss']:.6f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
