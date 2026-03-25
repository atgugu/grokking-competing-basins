"""Training smoke tests."""

import torch
import pytest
from src.training.trainer import Trainer, TrainConfig


def test_training_smoke():
    """Short training run should decrease loss."""
    config = TrainConfig(
        p=7, K=32, lr=0.001, weight_decay=0.0001,
        batch_size=32, max_epochs=100,
        train_fraction=0.4, seed=0,
        checkpoint_interval=50,
        checkpoint_dir="/tmp/test_ckpt",
        device="cpu",
    )
    trainer = Trainer(config)
    history = trainer.train(verbose=False)

    assert len(history.epochs) >= 2
    # Loss should decrease
    assert history.train_loss[-1] < history.train_loss[0]


def test_optimizer_config():
    """Verify optimizer is Adam with coupled L2 weight decay."""
    config = TrainConfig(p=7, K=16, device="cpu")
    trainer = Trainer(config)

    assert isinstance(trainer.optimizer, torch.optim.Adam)
    for group in trainer.optimizer.param_groups:
        assert group["lr"] == config.lr
        assert group["weight_decay"] == config.weight_decay  # coupled L2


def test_checkpoint_save_load():
    """Test checkpoint round-trip."""
    import os
    import tempfile
    from src.training.checkpointing import load_model_from_checkpoint

    config = TrainConfig(
        p=7, K=16, lr=0.001, max_epochs=10,
        checkpoint_interval=5, device="cpu",
        checkpoint_dir=tempfile.mkdtemp(),
    )
    trainer = Trainer(config)
    trainer.train(verbose=False)

    # Save and reload
    path = trainer.save_checkpoint(10)
    model = load_model_from_checkpoint(path, "cpu")
    assert model.p == 7
    assert model.K == 16

    # Check weights match
    for (n1, p1), (n2, p2) in zip(
        trainer.model.named_parameters(), model.named_parameters()
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2)
