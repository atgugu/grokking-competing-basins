"""Training loop for the quadratic network."""

import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.quadratic_net import QuadraticNet
from src.data.modular_arithmetic import ModularArithmeticDataset


def centered_mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Per-sample centered MSE matching paper's centred_loss.

    J = (1/2)||P_⊥(Y - Ŷ)||²_F / N  (sum over p outputs, average over N samples).
    Used for both training and LLC estimation.
    """
    R = y_true - y_pred
    centered_R = R - R.mean(dim=0, keepdim=True)
    n = y_pred.shape[0]
    return 0.5 * centered_R.pow(2).sum() / n


def centered_mse_perelement(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Per-element centered MSE for reporting only (divides by N*p)."""
    R = y_true - y_pred
    centered_R = R - R.mean(dim=0, keepdim=True)
    return 0.5 * centered_R.pow(2).mean()


def centered_mse_sum(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Raw Frobenius norm loss: J = (1/2)||P_⊥(Y - Ŷ)||²_F (unnormalized)."""
    R = y_true - y_pred
    centered_R = R - R.mean(dim=0, keepdim=True)
    return 0.5 * centered_R.pow(2).sum()


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute classification accuracy from continuous predictions vs one-hot targets."""
    pred_classes = y_pred.argmax(dim=1)
    true_classes = y_true.argmax(dim=1)
    return (pred_classes == true_classes).float().mean().item()


@dataclass
class TrainConfig:
    p: int = 53
    K: int = 1024
    lr: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 100_000
    train_fraction: float = 0.4
    seed: int = 0
    checkpoint_interval: int = 100
    checkpoint_dir: str = "results/checkpoints"
    device: str = "cpu"
    eval_interval: int | None = None  # None => same as checkpoint_interval
    save_checkpoints: bool = True


@dataclass
class TrainHistory:
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


class Trainer:
    """Trainer matching the paper's anonymous code repo.

    Key settings (from https://anonymous.4open.science/r/geom_phase_transitions-DF59/):
    - Adam with coupled L2 weight decay (weight_decay kwarg, NOT decoupled)
    - lr=0.001, weight_decay=1e-5
    - Per-sample centered MSE loss: 0.5 * ||P_⊥(Y-Ŷ)||²_F / N
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)

        torch.manual_seed(config.seed)

        self.dataset = ModularArithmeticDataset(
            config.p, config.train_fraction, config.seed
        )
        self.train_loader = DataLoader(
            self.dataset.train_dataset(),
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed),
        )
        self.model = QuadraticNet(config.p, config.K).to(self.device)
        # Adam with coupled L2 weight decay (matching paper's code)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.history = TrainHistory()

        # Pre-compute full batches for eval
        self.x_train, self.y_train = self.dataset.full_train_batch(str(self.device))
        self.x_val, self.y_val = self.dataset.full_val_batch(str(self.device))

    def train_epoch(self) -> float:
        """Train for one epoch over mini-batches. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in self.train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)
            loss = centered_mse(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float, float, float]:
        """Evaluate on full train and val sets. Returns (train_loss, val_loss, train_acc, val_acc)."""
        self.model.eval()
        y_pred_train = self.model(self.x_train)
        y_pred_val = self.model(self.x_val)
        tl = centered_mse(y_pred_train, self.y_train).item()
        vl = centered_mse(y_pred_val, self.y_val).item()
        ta = accuracy(y_pred_train, self.y_train)
        va = accuracy(y_pred_val, self.y_val)
        return tl, vl, ta, va

    def save_checkpoint(self, epoch: int, path: str | None = None):
        if path is None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            path = os.path.join(self.config.checkpoint_dir, f"epoch_{epoch:06d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "history": self.history,
        }, path)
        return path

    def train(self, verbose: bool = True) -> TrainHistory:
        """Full training loop."""
        from tqdm import tqdm

        eval_int = self.config.eval_interval or self.config.checkpoint_interval
        pbar = tqdm(range(1, self.config.max_epochs + 1),
                    desc="Training", disable=not verbose,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        for epoch in pbar:
            self.train_epoch()

            if epoch % eval_int == 0 or epoch == 1:
                tl, vl, ta, va = self.evaluate()
                self.history.epochs.append(epoch)
                self.history.train_loss.append(tl)
                self.history.val_loss.append(vl)
                self.history.train_acc.append(ta)
                self.history.val_acc.append(va)

                pbar.set_postfix_str(
                    f"loss={tl:.4f} val={vl:.4f} acc={ta:.2f}/{va:.2f}"
                )

            if self.config.save_checkpoints and epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint(epoch)

        pbar.close()
        return self.history
