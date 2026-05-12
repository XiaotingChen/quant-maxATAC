# -*- coding: utf-8 -*-
"""
Single-figure heatmap view of model_interpreting NPZ output.

Each signal (prediction, gs, motif, atac) gets one row of subplots:
  - a narrow category-color strip (rows = windows, colour = category)
  - a full heatmap (rows = windows, columns = 1024-bp positions)

Windows are stacked in category order: TP → FP → FN → TN.

Usage:
    python model_interpreting_data2figure_heatmap.py \
        --npz ./test_model_interpreting_output/test_CTCF_Jurkat_chr1_model_interpreting.npz \
        --output_dir ./test_model_interpreting_output/plots
"""

import argparse
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

CATEGORIES = ("TP", "FP", "FN", "TN")
SIGNALS = ("prediction", "gs", "motif", "atac")

CATEGORY_COLORS = {
    "TP": "#2ca02c",
    "FP": "#d62728",
    "FN": "#ff7f0e",
    "TN": "#1f77b4",
}


def _hex_to_rgb01(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def plot_combined_heatmap(data, output_dir):
    """One figure: 4 signal rows, each with a category-color strip and a heatmap."""
    n_signals = len(SIGNALS)
    strip_width_ratio = 0.02

    fig = plt.figure(figsize=(14, 3 * n_signals))
    gs = GridSpec(
        n_signals, 2,
        figure=fig,
        width_ratios=[strip_width_ratio, 1 - strip_width_ratio],
        hspace=0.35,
        wspace=0.02,
    )

    for row_idx, signal in enumerate(SIGNALS):
        arrays, cat_labels = [], []
        for cat in CATEGORIES:
            key = f"{cat}_{signal}"
            if key in data and len(data[key]) > 0:
                arr = data[key]
                arrays.append(arr)
                cat_labels.extend([cat] * len(arr))

        ax_strip = fig.add_subplot(gs[row_idx, 0])
        ax_heat = fig.add_subplot(gs[row_idx, 1])

        if not arrays:
            for ax in (ax_strip, ax_heat):
                ax.set_visible(False)
            continue

        matrix = np.vstack(arrays)

        rgb_strip = np.array(
            [_hex_to_rgb01(CATEGORY_COLORS[c]) for c in cat_labels],
            dtype=np.float32,
        ).reshape(-1, 1, 3)

        ax_strip.imshow(rgb_strip, aspect="auto", interpolation="nearest")
        ax_strip.set_xticks([])
        ax_strip.set_yticks([])
        ax_strip.set_ylabel(signal, fontsize=10, labelpad=4)

        im = ax_heat.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            origin="upper",
        )
        ax_heat.set_xticks([])
        ax_heat.set_yticks([])

        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.015, pad=0.01)
        cbar.ax.tick_params(labelsize=7)

    ax_heat_last = fig.add_subplot(gs[-1, 1])
    ax_heat_last.set_xlabel("Position in 1024 bp window (bp)", fontsize=10)
    ax_heat_last.set_visible(False)

    legend_patches = [
        mpatches.Patch(facecolor=CATEGORY_COLORS[cat], label=cat)
        for cat in CATEGORIES
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(CATEGORIES),
        fontsize=9,
        title="Category",
        title_fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle("Signal heatmaps by classification category", fontsize=13, y=1.01)

    out_path = os.path.join(output_dir, "all_categories_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Single-figure heatmap of model_interpreting NPZ output.")
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

    plot_combined_heatmap(data, args.output_dir)


if __name__ == "__main__":
    main()
