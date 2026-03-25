"""Tests for centered MSE loss."""

import torch
import pytest
from src.training.trainer import centered_mse, centered_mse_perelement, centered_mse_sum, accuracy


def test_centered_mse_zero_residual():
    """Perfect predictions should give zero loss."""
    y = torch.randn(10, 5)
    loss = centered_mse(y, y)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_centered_mse_is_persample():
    """centered_mse is per-sample: 0.5 * sum(cR²) / N (sums over p, averages over N)."""
    N, p = 20, 5
    y_true = torch.randn(N, p)
    y_pred = torch.zeros(N, p)
    loss = centered_mse(y_pred, y_true)
    # Per-sample = per-element * p
    expected = centered_mse_perelement(y_pred, y_true) * p
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_centered_mse_centering():
    """Adding a constant to predictions shouldn't change the centered loss."""
    N, p = 20, 5
    y_true = torch.randn(N, p)
    y_pred = torch.randn(N, p)
    loss1 = centered_mse(y_pred, y_true)
    offset = torch.randn(1, p)
    loss2 = centered_mse(y_pred + offset, y_true)
    assert loss1.item() == pytest.approx(loss2.item(), rel=1e-5)


def test_init_loss_scale():
    """With Kaiming init and p=53, per-sample init loss should be ~0.5."""
    from src.models.quadratic_net import QuadraticNet
    from src.data.modular_arithmetic import ModularArithmeticDataset

    torch.manual_seed(0)
    p = 53
    model = QuadraticNet(p, 1024)
    ds = ModularArithmeticDataset(p, train_fraction=0.4, seed=0)
    x, y = ds.full_train_batch()

    with torch.no_grad():
        y_pred = model(x)
        loss = centered_mse(y_pred, y)

    # Per-sample loss ≈ p * per-element ≈ 53 * 0.009 ≈ 0.48
    assert 0.2 < loss.item() < 1.0, f"Init loss {loss.item()} outside expected range"


def test_sum_vs_persample_loss():
    """Sum loss should equal per-sample loss * N."""
    N, p = 20, 5
    y_true = torch.randn(N, p)
    y_pred = torch.randn(N, p)
    sum_loss = centered_mse_sum(y_pred, y_true)
    persample_loss = centered_mse(y_pred, y_true)
    assert sum_loss.item() == pytest.approx(persample_loss.item() * N, rel=1e-5)


def test_accuracy_perfect():
    y_pred = torch.eye(5)
    y_true = torch.eye(5)
    assert accuracy(y_pred, y_true) == 1.0


def test_accuracy_random():
    torch.manual_seed(0)
    y_pred = torch.randn(100, 10)
    y_true = torch.eye(10)[torch.randint(0, 10, (100,))]
    acc = accuracy(y_pred, y_true)
    assert 0.0 <= acc <= 1.0
