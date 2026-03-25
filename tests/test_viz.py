"""Tests for visualization functions."""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.viz.style import setup_style
from src.viz.training_curves import plot_loss_and_llc, plot_accuracy_curves
from src.viz.scaling_plots import plot_llc_vs_p, plot_llc_vs_K
from src.viz.gsm_plots import plot_gsm_vs_lr
from src.analysis.scaling_laws import fit_linear


@pytest.fixture(autouse=True)
def _setup():
    setup_style()
    yield
    plt.close("all")


def test_plot_loss_and_llc():
    epochs = list(range(0, 100, 10))
    train_loss = np.exp(-np.linspace(0, 3, len(epochs))).tolist()
    val_loss = np.exp(-np.linspace(0, 2, len(epochs))).tolist()
    llc_epochs = epochs[::2]
    llc_values = np.linspace(100, 200, len(llc_epochs)).tolist()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig = plot_loss_and_llc(epochs, train_loss, val_loss,
                                llc_epochs, llc_values, save_path=f.name)
        assert os.path.exists(f.name)
        os.unlink(f.name)


def test_plot_accuracy_curves():
    epochs = list(range(100))
    train_acc = np.linspace(0.1, 1.0, 100).tolist()
    val_acc = np.linspace(0.1, 0.95, 100).tolist()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig = plot_accuracy_curves(epochs, train_acc, val_acc, save_path=f.name)
        assert os.path.exists(f.name)
        os.unlink(f.name)


def test_plot_llc_vs_p():
    fits = {
        600: fit_linear([53, 61, 71], [2800, 3200, 3700]),
        1000: fit_linear([53, 61, 71], [4200, 4900, 5600]),
    }
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig = plot_llc_vs_p(fits, save_path=f.name)
        assert os.path.exists(f.name)
        os.unlink(f.name)


def test_plot_gsm_vs_lr():
    lrs = [0.001, 0.005, 0.01]
    gsms = [0.5, 0.2, 0.05]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig = plot_gsm_vs_lr(lrs, gsms, save_path=f.name)
        assert os.path.exists(f.name)
        os.unlink(f.name)
