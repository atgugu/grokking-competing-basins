"""LLC estimation using devinterp SGLD sampling.

Parameters match the paper's anonymous code repo:
- Loss: per-sample centered MSE = 0.5 * ||P_⊥(yᵢ - ŷᵢ)||².sum(p).mean(N)
- SGLD lr: 5e-4
- nbeta: default_nbeta(batch_size=128) ≈ 26.4
- localization: 5.0
- num_chains: 3, num_draws: 500, num_burnin: 100
"""

import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.quadratic_net import QuadraticNet
from src.data.modular_arithmetic import ModularArithmeticDataset
from src.training.checkpointing import load_checkpoint


def default_nbeta(batch_size: int) -> float:
    """Compute default nbeta matching devinterp's convention: batch_size / log(batch_size)."""
    if batch_size <= 1:
        return 1.0
    return batch_size / math.log(batch_size)


def _make_evaluate_fn(device: str):
    """Create per-sample centered MSE evaluate function for SGLD sampling.

    Returns 0.5 * ||P_⊥(Y - Ŷ)||²_F / N  (sum over p outputs, average over N samples).
    Matches paper's centred_loss: 0.5 * (E*E).sum() / n.
    """
    def evaluate(model, batch):
        x = batch[0].to(device)
        y_true = batch[1].to(device)
        y_pred = model(x)
        R = y_true - y_pred
        centered_R = R - R.mean(dim=0, keepdim=True)
        return 0.5 * centered_R.pow(2).sum(dim=1).mean(dim=0)
    return evaluate


# Default SGLD parameters matching paper's anonymous code repo
_DEFAULT_SGLD_LR = 5e-4
_DEFAULT_NBETA = default_nbeta(128)  # ≈ 26.4
_DEFAULT_NUM_DRAWS = 500


def estimate_llc(
    model: QuadraticNet,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    device: str = "cpu",
    num_chains: int = 3,
    num_draws: int = _DEFAULT_NUM_DRAWS,
    num_burnin_steps: int = 100,
    sgld_lr: float = _DEFAULT_SGLD_LR,
    nbeta: float = _DEFAULT_NBETA,
    localization: float = 5.0,
    seed: int = 42,
    batch_size: int = 128,
) -> dict:
    """Estimate LLC for a model at current parameters.

    Uses mini-batch SGLD matching the paper's code (batch_size=128).
    Returns dict with 'llc_mean', 'llc_std', 'init_loss'.
    """
    from devinterp.slt.llc import LLCEstimator
    from devinterp.slt.sampler import sample

    evaluate = _make_evaluate_fn(device)

    # Compute init_loss at current params (full dataset)
    model.eval()
    with torch.no_grad():
        init_loss = evaluate(model, (x_train.to(device), y_train.to(device))).item()

    # Create dataloader for SGLD (mini-batch, matching paper's code)
    dataset = TensorDataset(x_train.cpu(), y_train.cpu())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    llc_estimator = LLCEstimator(
        num_chains=num_chains,
        num_draws=num_draws,
        init_loss=torch.tensor(init_loss),
        nbeta=nbeta,
        device=str(device),
    )

    sample(
        model,
        loader,
        evaluate=evaluate,
        callbacks=[llc_estimator],
        num_draws=num_draws,
        num_chains=num_chains,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=1,
        optimizer_kwargs={
            "lr": sgld_lr,
            "nbeta": nbeta,
            "localization": localization,
        },
        device=str(device),
        verbose=False,
        seed=seed,
    )

    results = llc_estimator.get_results()
    return {
        "llc_mean": float(results["llc/mean"]),
        "llc_std": float(results["llc/std"]),
        "init_loss": init_loss,
    }


def estimate_llc_from_checkpoint(
    checkpoint_path: str,
    p: int,
    train_fraction: float = 0.4,
    data_seed: int = 0,
    device: str = "cpu",
    **llc_kwargs,
) -> dict:
    """Estimate LLC from a saved checkpoint."""
    ckpt = load_checkpoint(checkpoint_path, device)
    config = ckpt["config"]
    model = QuadraticNet(config.p, config.K).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    dataset = ModularArithmeticDataset(p, train_fraction, data_seed)
    x_train, y_train = dataset.full_train_batch(device)

    return estimate_llc(model, x_train, y_train, device=device, **llc_kwargs)
