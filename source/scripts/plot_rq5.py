#!/usr/bin/env python3
"""
RQ5 -- Cross-dataset / cross-region generalization.

2x2 grid: Panel A (line plot) x 2 test types (synthetic, soundscape);
Panel B (heatmap) x 2 test types.

Panel A: x = 7 regional datasets, ordered by species count ascending (tick
labels include species count). "Region-specific head" (archive/Pooled-
Embeddings/, 11 backbones -- each dataset's own single-dataset-trained
model) vs. "XCM head" (archive/XCM-Generalization/, same 11 backbones,
XCM-trained head tested on each dataset), both mean +/- SD across the 11
backbones at each dataset.

Panel B: rows = the 5 XCM-fine-tuned backbones (archive/XCM-Generalization-
fine-tune/), columns = same 7 datasets/order as Panel A, cell = single MAE
value (no whiskers), shared sequential color scale (light = low MAE) across
both sub-panels. Missing cells (e.g. a backbone/dataset combo that was never
run) are masked rather than shown as 0.

Usage:
    complete-venv/bin/python source/scripts/plot_rq5.py [--out-dir plots/figures/rq5]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import BACKBONE_META, FINETUNE_NAME_TO_CANONICAL
from plot_data import REGIONAL_DATASETS, filter_long, load_study, mean_sd_n
from plot_style import draw_dot_whisker, mask_missing, save_fig

SYN_SOURCE = "synthetic_mixture_test"
SCAPE_SOURCE = "soundscape_test"
STATS_JSON = Path(__file__).resolve().parents[2] / "plots" / "data" / "polybirdmix_soundscape_stats.json"
ACCENT_COLOR = "#d62728"


def species_ordered_datasets():
    stats = json.loads(STATS_JSON.read_text())["subsets"]
    ordered = sorted(REGIONAL_DATASETS, key=lambda d: stats[d]["n_unique_species"])
    labels = [f"{d} (n={stats[d]['n_unique_species']})" for d in ordered]
    return ordered, labels


# ---------------------------------------------------------------------------
# Panel A -- line plot
# ---------------------------------------------------------------------------

def build_panel_a(ax, pooled_long, xg_long, source, metric, dataset_order, title):
    region_specific = mean_sd_n(
        filter_long(pooled_long, source=source, metric=metric, head="reg", dataset=dataset_order), ["dataset"]
    ).reindex(dataset_order)
    xcm_head = mean_sd_n(
        filter_long(xg_long, source=source, metric=metric, head="reg", dataset=dataset_order), ["dataset"]
    ).reindex(dataset_order)

    xs = range(len(dataset_order))
    ax.plot(xs, region_specific["mean"], color="0.5", linestyle="--", linewidth=1.5, zorder=1, label="Region-specific head")
    ax.errorbar(xs, region_specific["mean"], yerr=region_specific["std"], fmt="none", ecolor="0.5", capsize=4, zorder=1)
    ax.scatter(xs, region_specific["mean"], facecolor="0.5", edgecolor="0.5", s=50, zorder=2)

    ax.plot(xs, xcm_head["mean"], color=ACCENT_COLOR, linewidth=2.5, zorder=3, label="XCM head")
    ax.errorbar(xs, xcm_head["mean"], yerr=xcm_head["std"], fmt="none", ecolor=ACCENT_COLOR, capsize=4, zorder=3)
    ax.scatter(xs, xcm_head["mean"], facecolor=ACCENT_COLOR, edgecolor=ACCENT_COLOR, s=60, zorder=4)

    ax.set_title(title)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    return xs


# ---------------------------------------------------------------------------
# Panel B -- heatmap
# ---------------------------------------------------------------------------

def build_heatmap_matrix(ft_long, source, metric, dataset_order, row_order):
    sub = filter_long(ft_long, source=source, metric=metric, head="reg", dataset=dataset_order)
    sub = sub.copy()
    sub["model"] = sub["model"].map(FINETUNE_NAME_TO_CANONICAL)
    matrix = sub.pivot_table(index="model", columns="dataset", values="value")
    return matrix.reindex(index=row_order, columns=dataset_order)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq5"))
    args = parser.parse_args()

    xg_models = mrt.discover_models(Path("archive/XCM-Generalization"))
    pooled_long = load_study("pooled", models=xg_models)
    xg_long = load_study("xcm_gen", models=xg_models)
    ft_long = load_study("xcm_gen_ft")

    dataset_order, tick_labels = species_ordered_datasets()
    ft_row_order = [m for m in BACKBONE_META if m in FINETUNE_NAME_TO_CANONICAL.values()]

    syn_matrix = build_heatmap_matrix(ft_long, SYN_SOURCE, "mae", dataset_order, ft_row_order)
    scape_matrix = build_heatmap_matrix(ft_long, SCAPE_SOURCE, "range_mae", dataset_order, ft_row_order)
    vmin = min(np.nanmin(syn_matrix.values), np.nanmin(scape_matrix.values))
    vmax = max(np.nanmax(syn_matrix.values), np.nanmax(scape_matrix.values))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("0.85")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), gridspec_kw={"height_ratios": [1, 1.15]},
                              constrained_layout=True)

    xs = build_panel_a(axes[0, 0], pooled_long, xg_long, SYN_SOURCE, "mae", dataset_order,
                       "Synthetic test (frozen backbone)")
    build_panel_a(axes[0, 1], pooled_long, xg_long, SCAPE_SOURCE, "range_mae", dataset_order,
                  "Soundscape (frozen backbone)")
    axes[0, 0].set_ylabel("MAE (lower is better)")
    for ax in axes[0]:
        ax.set_xticks(list(xs))
        ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    axes[0, 0].legend(loc="upper left", frameon=False)

    # Shared y-scale across the two Panel A sub-panels (per the RQ5 spec).
    lo = min(axes[0, 0].get_ylim()[0], axes[0, 1].get_ylim()[0])
    hi = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
    axes[0, 0].set_ylim(lo, hi)
    axes[0, 1].set_ylim(lo, hi)

    ims = []
    for ax, matrix, title in [(axes[1, 0], syn_matrix, "Synthetic test (fine-tuned backbone, XCM → cross-region)"),
                               (axes[1, 1], scape_matrix, "Soundscape (fine-tuned backbone, XCM → cross-region)")]:
        im = ax.imshow(mask_missing(matrix), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ims.append(im)
        ax.set_xticks(range(len(dataset_order)))
        ax.set_xticklabels(tick_labels, rotation=30, ha="right")
        ax.set_yticks(range(len(ft_row_order)))
        ax.set_yticklabels([BACKBONE_META[m]["display"] for m in ft_row_order])
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if val > (vmin + vmax) / 2 else "black")

    fig.colorbar(ims[0], ax=axes[1], shrink=0.85, label="MAE (lower is better)", location="right")

    fig.suptitle("RQ5 -- Cross-dataset / cross-region generalization", fontsize=14)
    save_fig(fig, args.out_dir / "rq5_grid")
    plt.close(fig)
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
