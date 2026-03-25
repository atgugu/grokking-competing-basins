"""Tests for QuadraticNet model."""

import torch
import pytest
from src.models.quadratic_net import QuadraticNet


def test_output_shape():
    p, K = 7, 32
    model = QuadraticNet(p, K)
    x = torch.randn(16, 2 * p)
    y = model(x)
    assert y.shape == (16, p)


def test_no_bias():
    model = QuadraticNet(7, 32)
    assert model.W.bias is None
    assert model.V.bias is None


def test_param_count():
    p, K = 53, 1024
    model = QuadraticNet(p, K)
    expected = 3 * p * K  # 2pK + Kp
    assert model.num_params == expected


def test_quadratic_activation():
    """Verify that activation is x^2."""
    p, K = 5, 3
    model = QuadraticNet(p, K)

    # Set W to identity-like, V to identity-like for testing
    with torch.no_grad():
        model.W.weight.fill_(1.0)
        model.V.weight.fill_(1.0)

    x = torch.ones(1, 2 * p)
    y = model(x)
    # W(x) = [sum(x)] * K = [2p] * K  (each row dot all-1 input)
    # W(x)^2 = [(2p)^2] * K
    # V(W(x)^2) = [K * (2p)^2] * p
    expected_val = K * (2 * p) ** 2
    assert torch.allclose(y, torch.full((1, p), float(expected_val)))


def test_kaiming_init():
    """Check that default init gives reasonable scale."""
    p, K = 53, 1024
    model = QuadraticNet(p, K)
    # Kaiming init for linear: std ≈ 1/sqrt(fan_in)
    w_std = model.W.weight.std().item()
    expected_std = 1.0 / (2 * p) ** 0.5  # fan_in = 2p
    assert abs(w_std - expected_std) < 0.05, f"W std {w_std} far from {expected_std}"


def test_gradient_flow():
    """Ensure gradients flow through quadratic activation."""
    model = QuadraticNet(7, 16)
    x = torch.randn(4, 14, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    assert model.W.weight.grad is not None
    assert model.V.weight.grad is not None
    assert x.grad is not None
