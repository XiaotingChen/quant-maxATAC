# -*- coding: utf-8 -*-
"""
Quick result checks for model_interpreting NPZ output.

Usage:
    python check_model_interpreting_results.py \
        --npz ./test_model_interpreting_output/test_CTCF_Jurkat_chr1_model_interpreting.npz \
        --output_dir ./test_model_interpreting_output/plots
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

CATEGORIES = ("TP", "FP", "FN", "TN")
SIGNALS = ("prediction", "gs", "motif", "atac")
SIGNAL_COLORS = {"prediction": "#1f77b4", "gs": "#ff7f0e", "motif": "#2ca02c", "atac": "#d62728"}


def plot_avg_signal_profile(data, category, output_dir):
    """Average per-base signal across all windows in the category - one subplot per signal type."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"{category}: average signal profile (n={len(data[f'{category}_{SIGNALS[0]}'])} windows)", fontsize=13)

    for ax, signal in zip(axes, SIGNALS):
        key = f"{category}_{signal}"
        if key not in data or len(data[key]) == 0:
            ax.set_ylabel(signal)
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            continue
        arr = data[key]  # shape (n, 1024)
        mean_trace = arr.mean(axis=0)
        x = np.arange(len(mean_trace))
        ax.plot(x, mean_trace, color=SIGNAL_COLORS[signal], linewidth=0.8)
        ax.set_ylabel(signal)
        ax.set_xlim(0, len(mean_trace) - 1)
        if (signal != "motif") and (signal != "prediction"):
            ax.set_ylim(0, 1.0)

    axes[-1].set_xlabel("Position in 1024 bp window (bp)")
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{category}_avg_signal_profile.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_motif_max_boxplot(data, output_dir):
    """Box plot of motif_max distribution for all four categories on one figure."""
    values = []
    labels = []
    for cat in CATEGORIES:
        key = f"{cat}_motif_max"
        if key in data and len(data[key]) > 0:
            values.append(data[key])
            labels.append(f"{cat}\n(n={len(data[key])})")

    if not values:
        print("No motif_max data found - skipping box plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(values, labels=labels, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 1.5})

    category_colors = {"TP": "#2ca02c", "FP": "#d62728", "FN": "#ff7f0e", "TN": "#1f77b4"}
    for patch, lbl in zip(bp["boxes"], labels):
        cat = lbl.split("\n")[0]
        patch.set_facecolor(category_colors.get(cat, "#888888"))
        patch.set_alpha(0.7)

    ax.set_ylabel("Motif max score (1024 bp window)")
    ax.set_title("Motif max score distribution by classification category")
    fig.tight_layout()
    out_path = os.path.join(output_dir, "motif_max_boxplot_all_categories.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Check model_interpreting NPZ output.")
    parser.add_argument("--npz", required=True, help="Path to the _model_interpreting.npz file")
    parser.add_argument("--output_dir", default=".", help="Directory for output PNG files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = dict(np.load(args.npz, allow_pickle=False))
    print(f"Loaded {args.npz}: keys = {sorted(data.keys())}")

    for cat in CATEGORIES:
        n_key = f"{cat}_prediction"
        n = len(data[n_key]) if n_key in data else 0
        print(f"  {cat}: {n} windows")
        if n > 0:
            plot_avg_signal_profile(data, cat, args.output_dir)

    plot_motif_max_boxplot(data, args.output_dir)


if __name__ == "__main__":
    main()
