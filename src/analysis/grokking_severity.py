"""Grokking Severity Metric (GSM) from Cullen et al. (2026).

GSM = (1_{aV(T)≥0.95} / T) × Σ_{x=1}^T |aT(x) - aV(x)|

Where aT(x) and aV(x) are train and validation accuracies at epoch x,
and T is the total number of epochs.
"""

import numpy as np


def compute_gsm(
    train_acc: list[float] | np.ndarray,
    val_acc: list[float] | np.ndarray,
    threshold: float = 0.95,
) -> float:
    """Compute the Grokking Severity Metric.

    Args:
        train_acc: Training accuracies over time
        val_acc: Validation accuracies over time
        threshold: Final validation accuracy threshold (default 0.95)

    Returns:
        GSM value. 0 if model never reaches threshold.
    """
    train_acc = np.asarray(train_acc)
    val_acc = np.asarray(val_acc)
    T = len(train_acc)

    # Indicator: does final val accuracy reach threshold?
    if val_acc[-1] < threshold:
        return 0.0

    # Sum of absolute differences
    diff_sum = np.sum(np.abs(train_acc - val_acc))
    return diff_sum / T


def classify_regime(
    train_acc: list[float] | np.ndarray,
    val_acc: list[float] | np.ndarray,
    threshold: float = 0.95,
    gsm_grokking_threshold: float = 0.1,
) -> str:
    """Classify training regime as one of three types.

    Returns:
        'no_generalization': val acc never reaches threshold
        'immediate_generalization': val acc reaches threshold with low GSM
        'grokking': val acc reaches threshold with high GSM
    """
    val_acc = np.asarray(val_acc)
    if val_acc[-1] < threshold:
        return "no_generalization"

    gsm = compute_gsm(train_acc, val_acc, threshold)
    if gsm < gsm_grokking_threshold:
        return "immediate_generalization"
    return "grokking"
