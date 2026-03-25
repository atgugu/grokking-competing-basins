"""Checkpoint loading utilities."""

import os
import re

import torch

from src.models.quadratic_net import QuadraticNet


def load_checkpoint(path: str, device: str = "cpu") -> dict:
    """Load a checkpoint and return the full dict."""
    return torch.load(path, map_location=device, weights_only=False)


def load_model_from_checkpoint(path: str, device: str = "cpu") -> QuadraticNet:
    """Load a model from checkpoint."""
    ckpt = load_checkpoint(path, device)
    config = ckpt["config"]
    model = QuadraticNet(config.p, config.K).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def list_checkpoints(checkpoint_dir: str) -> list[tuple[int, str]]:
    """List all checkpoints sorted by epoch. Returns [(epoch, path), ...]."""
    results = []
    for fname in os.listdir(checkpoint_dir):
        m = re.match(r"epoch_(\d+)\.pt", fname)
        if m:
            epoch = int(m.group(1))
            results.append((epoch, os.path.join(checkpoint_dir, fname)))
    results.sort()
    return results
