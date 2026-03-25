"""Modular addition dataset with one-hot encoding."""

import torch
from torch.utils.data import DataLoader, TensorDataset


class ModularArithmeticDataset:
    """Dataset for (a + b) mod p with one-hot encoding.

    Input: concatenation of one-hot(a) and one-hot(b), shape (2p,)
    Target: one-hot((a+b) mod p), shape (p,)
    """

    def __init__(self, p: int, train_fraction: float = 0.4, seed: int = 0):
        self.p = p
        self.train_fraction = train_fraction
        self.seed = seed

        # Generate all p^2 pairs sorted by (a, b)
        all_a = torch.arange(p).repeat_interleave(p)  # 0,0,...,0,1,1,...
        all_b = torch.arange(p).repeat(p)              # 0,1,...,p-1,0,1,...
        all_c = (all_a + all_b) % p

        # One-hot encode
        x_a = torch.nn.functional.one_hot(all_a, p).float()  # (p^2, p)
        x_b = torch.nn.functional.one_hot(all_b, p).float()  # (p^2, p)
        self.x_all = torch.cat([x_a, x_b], dim=1)            # (p^2, 2p)
        self.y_all = torch.nn.functional.one_hot(all_c, p).float()  # (p^2, p)

        # Deterministic split: first floor(tf * p^2) are train
        n_total = p * p
        n_train = int(train_fraction * n_total)

        # Shuffle with seed for the split
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_total, generator=gen)

        self.train_idx = perm[:n_train]
        self.val_idx = perm[n_train:]

        self.x_train = self.x_all[self.train_idx]
        self.y_train = self.y_all[self.train_idx]
        self.x_val = self.x_all[self.val_idx]
        self.y_val = self.y_all[self.val_idx]

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_val(self) -> int:
        return len(self.val_idx)

    def train_dataset(self) -> TensorDataset:
        return TensorDataset(self.x_train, self.y_train)

    def val_dataset(self) -> TensorDataset:
        return TensorDataset(self.x_val, self.y_val)

    def full_train_batch(self, device: str = "cpu"):
        return self.x_train.to(device), self.y_train.to(device)

    def full_val_batch(self, device: str = "cpu"):
        return self.x_val.to(device), self.y_val.to(device)


def make_dataloaders(
    p: int,
    train_fraction: float = 0.4,
    batch_size: int = 128,
    seed: int = 0,
) -> tuple[ModularArithmeticDataset, DataLoader, DataLoader]:
    """Create dataset and dataloaders."""
    dataset = ModularArithmeticDataset(p, train_fraction, seed)
    train_loader = DataLoader(
        dataset.train_dataset(),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        dataset.val_dataset(),
        batch_size=len(dataset.val_idx),
        shuffle=False,
    )
    return dataset, train_loader, val_loader
