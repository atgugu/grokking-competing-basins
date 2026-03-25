"""Robustness plots (Figures 7-10) — wrapper around training_curves."""

from .training_curves import plot_multi_loss_llc


def plot_robustness_p(results, save_path=None):
    """Figure 7: Varying p with fixed K=1024."""
    labels = [f"p={r['p']}" for r in results]
    return plot_multi_loss_llc(results, labels, title="Robustness: Varying p",
                               save_path=save_path)


def plot_robustness_K(results, save_path=None):
    """Figure 8: Varying K with fixed p=53."""
    labels = [f"K={r['K']}" for r in results]
    return plot_multi_loss_llc(results, labels, title="Robustness: Varying K",
                               save_path=save_path)


def plot_robustness_wd(results, save_path=None):
    """Figure 9: Varying weight decay."""
    labels = [f"wd={r['wd']}" for r in results]
    return plot_multi_loss_llc(results, labels, title="Robustness: Varying Weight Decay",
                               save_path=save_path)


def plot_robustness_lr(results, save_path=None):
    """Figure 10: Varying learning rate."""
    labels = [f"lr={r['lr']}" for r in results]
    return plot_multi_loss_llc(results, labels, title="Robustness: Varying Learning Rate",
                               save_path=save_path)
