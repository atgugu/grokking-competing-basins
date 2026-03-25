"""Generate all figures from saved results.

Run this after all experiments have completed to regenerate plots
without retraining.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.scaling_laws import fit_linear
from src.viz.scaling_plots import plot_llc_vs_p, plot_llc_vs_K
from src.viz.training_curves import plot_loss_and_llc, plot_multi_loss_llc
from src.viz.gsm_plots import plot_gsm_vs_lr
from src.viz.animation import create_training_gif
from src.viz.style import setup_style, COLORS


def generate_figure1():
    """Regenerate Figure 1 from saved results."""
    path = "results/scaling_p/results.json"
    if not os.path.exists(path):
        print("Figure 1: No results found, skipping")
        return
    with open(path) as f:
        data = json.load(f)

    primes = [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    K_values = [600, 1000, 1400]
    fits = {}
    for K in K_values:
        p_vals = [data[f"p{p}_K{K}"]["p"] for p in primes if f"p{p}_K{K}" in data]
        llc_vals = [data[f"p{p}_K{K}"]["llc_mean"] for p in primes if f"p{p}_K{K}" in data]
        if p_vals:
            fits[K] = fit_linear(p_vals, llc_vals)
    if fits:
        plot_llc_vs_p(fits, save_path="results/scaling_p/figure1_llc_vs_p.png")
        print("Figure 1 generated")


def generate_figure1_lowwd():
    """Regenerate Figure 1 (wd=1e-5) from saved results."""
    path = "results/scaling_p_wd1e5/results.json"
    if not os.path.exists(path):
        print("Figure 1 (wd=1e-5): No results found, skipping")
        return
    with open(path) as f:
        data = json.load(f)

    all_p = sorted(set(v["p"] for k, v in data.items()
                       if not k.startswith("fit_") and isinstance(v, dict) and "p" in v))
    all_K = sorted(set(v["K"] for k, v in data.items()
                       if not k.startswith("fit_") and isinstance(v, dict) and "K" in v))
    fits = {}
    for K in all_K:
        p_vals = [data[f"p{p}_K{K}"]["p"] for p in all_p if f"p{p}_K{K}" in data]
        llc_vals = [data[f"p{p}_K{K}"]["llc_mean"] for p in all_p if f"p{p}_K{K}" in data]
        if len(p_vals) >= 2:
            fits[K] = fit_linear(p_vals, llc_vals)
    if fits:
        plot_llc_vs_p(fits, save_path="results/scaling_p_wd1e5/figure1_llc_vs_p_wd1e5.png")
        print("Figure 1 (wd=1e-5) generated")


def generate_figure2():
    """Regenerate Figure 2 from saved results."""
    path = "results/scaling_K/results.json"
    if not os.path.exists(path):
        print("Figure 2: No results found, skipping")
        return
    with open(path) as f:
        data = json.load(f)

    K_values = [200, 400, 600, 800, 1000, 1200, 1400, 1600]
    primes = [53, 61, 71]
    fits = {}
    for p in primes:
        k_vals = [data[f"p{p}_K{K}"]["K"] for K in K_values if f"p{p}_K{K}" in data]
        llc_vals = [data[f"p{p}_K{K}"]["llc_mean"] for K in K_values if f"p{p}_K{K}" in data]
        if k_vals:
            fits[p] = fit_linear(k_vals, llc_vals)
    if fits:
        plot_llc_vs_K(fits, save_path="results/scaling_K/figure2_llc_vs_K.png")
        print("Figure 2 generated")


def generate_figure3():
    """Regenerate Figure 3 from saved results."""
    base = "results/llc_tracking"
    hist_path = os.path.join(base, "history.json")
    llc_path = os.path.join(base, "llc_results.json")
    if not os.path.exists(hist_path) or not os.path.exists(llc_path):
        print("Figure 3: No results found, skipping")
        return
    with open(hist_path) as f:
        hist = json.load(f)
    with open(llc_path) as f:
        llc = json.load(f)

    plot_loss_and_llc(
        hist["epochs"], hist["train_loss"], hist["val_loss"],
        llc["llc_epochs"], llc["llc_values"],
        title="LLC Tracks Generalization (p=53, K=1024)",
        save_path=os.path.join(base, "figure3_llc_tracking.png"),
    )
    print("Figure 3 generated")


def generate_figure4():
    """Regenerate Figure 4 from saved results."""
    path = "results/gsm/figure4_results.json"
    if not os.path.exists(path):
        print("Figure 4: No results found, skipping")
        return
    with open(path) as f:
        data = json.load(f)
    plot_gsm_vs_lr(
        data["lr_values"], data["gsm_values"],
        title=f"GSM vs Learning Rate (wd={data['wd']})",
        save_path="results/gsm/figure4_gsm_vs_lr.png",
    )
    print("Figure 4 generated")


def generate_figure14():
    """Regenerate Figure 14 from saved results."""
    path = "results/scaling_N/results.json"
    if not os.path.exists(path):
        print("Figure 14: No results found, skipping")
        return
    with open(path) as f:
        results = json.load(f)

    regime_colors = {
        "no_generalization": "#F44336",
        "grokking": "#FF9800",
        "immediate_generalization": "#4CAF50",
    }
    regime_markers = {
        "no_generalization": "x",
        "grokking": "o",
        "immediate_generalization": "s",
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for rt in ["no_generalization", "grokking", "immediate_generalization"]:
        pts = [r for r in results if r["regime"] == rt]
        if pts:
            ax.scatter([r["ratio"] for r in pts], [r["M"] for r in pts],
                       c=regime_colors[rt], marker=regime_markers[rt],
                       s=100, label=rt.replace("_", " ").title(), zorder=5)
    ax.axvline(x=2.75, color=COLORS["gray"], linestyle="--", alpha=0.7,
               label="N/(M log M) ~ 2.75")
    ax.set_xlabel("N / (M log M)")
    ax.set_ylabel("Group Size M")
    ax.legend()
    ax.set_title("Regime Classification: N/(M log M) Scaling")
    fig.savefig("results/scaling_N/figure14_scaling_N.png")
    plt.close(fig)
    print("Figure 14 generated")


def generate_robustness():
    """Regenerate Figures 7-10 from saved results."""
    base = "results/robustness"
    names = {
        "7": ("Varying p", lambda rs: [f"p={r['p']}" for r in rs]),
        "8": ("Varying K", lambda rs: [f"K={r['K']}" for r in rs]),
        "9": ("Varying wd", lambda rs: [f"wd={r['wd']}" for r in rs]),
        "10": ("Varying lr", lambda rs: [f"lr={r['lr']}" for r in rs]),
    }
    for fig_num, (title_suffix, label_fn) in names.items():
        path = os.path.join(base, f"figure{fig_num}_results.json")
        if not os.path.exists(path):
            print(f"Figure {fig_num}: No results found, skipping")
            continue
        with open(path) as f:
            results = json.load(f)
        fname = title_suffix.split()[-1].lower()
        plot_multi_loss_llc(results, label_fn(results),
                            title=f"Robustness: {title_suffix}",
                            save_path=os.path.join(base, f"figure{fig_num}_varying_{fname}.png"))
        print(f"Figure {fig_num} generated")


def generate_animation():
    """Regenerate training animation from Figure 3 data."""
    base = "results/llc_tracking"
    hist_path = os.path.join(base, "history.json")
    llc_path = os.path.join(base, "llc_results.json")
    if not os.path.exists(hist_path) or not os.path.exists(llc_path):
        print("Animation: No results found, skipping")
        return
    with open(hist_path) as f:
        hist = json.load(f)
    with open(llc_path) as f:
        llc = json.load(f)
    try:
        create_training_gif(
            hist["epochs"], hist["train_loss"], hist["val_loss"],
            hist["train_acc"], hist["val_acc"],
            llc["llc_epochs"], llc["llc_values"],
            save_path=os.path.join(base, "training_animation.gif"),
            n_frames=80, duration_ms=80,
        )
        print("Animation generated")
    except Exception as e:
        print(f"Animation failed: {e}")


def main():
    setup_style()
    os.makedirs("results", exist_ok=True)

    print("Regenerating all figures from saved results...\n")
    generate_figure1()
    generate_figure1_lowwd()
    generate_figure2()
    generate_figure3()
    generate_figure4()
    generate_figure14()
    generate_robustness()
    generate_animation()
    print("\nDone!")


if __name__ == "__main__":
    main()
