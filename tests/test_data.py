"""Tests for modular arithmetic dataset."""

import torch
import pytest
from src.data.modular_arithmetic import ModularArithmeticDataset, make_dataloaders


def test_dataset_shapes():
    p = 7
    ds = ModularArithmeticDataset(p, train_fraction=0.4, seed=0)
    assert ds.x_all.shape == (p * p, 2 * p)
    assert ds.y_all.shape == (p * p, p)
    assert ds.n_train + ds.n_val == p * p
    assert ds.n_train == int(0.4 * p * p)


def test_one_hot_encoding():
    p = 5
    ds = ModularArithmeticDataset(p)
    # Each row of x should have exactly 2 ones (one-hot a + one-hot b)
    assert (ds.x_all.sum(dim=1) == 2.0).all()
    # Each row of y should have exactly 1 one
    assert (ds.y_all.sum(dim=1) == 1.0).all()


def test_addition_correctness():
    p = 5
    ds = ModularArithmeticDataset(p)
    for i in range(p * p):
        a = ds.x_all[i, :p].argmax().item()
        b = ds.x_all[i, p:].argmax().item()
        c = ds.y_all[i].argmax().item()
        assert c == (a + b) % p, f"Failed: {a}+{b} mod {p} = {c}"


def test_train_val_disjoint():
    p = 7
    ds = ModularArithmeticDataset(p, train_fraction=0.4, seed=0)
    train_set = set(ds.train_idx.tolist())
    val_set = set(ds.val_idx.tolist())
    assert len(train_set & val_set) == 0
    assert len(train_set | val_set) == p * p


def test_seed_reproducibility():
    p = 7
    ds1 = ModularArithmeticDataset(p, seed=42)
    ds2 = ModularArithmeticDataset(p, seed=42)
    assert torch.equal(ds1.train_idx, ds2.train_idx)
    assert torch.equal(ds1.val_idx, ds2.val_idx)


def test_make_dataloaders():
    p = 7
    ds, train_dl, val_dl = make_dataloaders(p, batch_size=16)
    batch_x, batch_y = next(iter(train_dl))
    assert batch_x.shape[1] == 2 * p
    assert batch_y.shape[1] == p


def test_p53_sizes():
    """Verify sizes for the main experiment prime."""
    p = 53
    ds = ModularArithmeticDataset(p, train_fraction=0.4, seed=0)
    assert ds.n_train == int(0.4 * 53 * 53)  # 1123
    assert ds.n_val == 53 * 53 - ds.n_train
    assert ds.x_train.shape == (ds.n_train, 106)
    assert ds.y_train.shape == (ds.n_train, 53)
