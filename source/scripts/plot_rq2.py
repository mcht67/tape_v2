#!/usr/bin/env python3
"""
RQ2 -- Multi-task / prediction head.

Single panel, from archive/Spatial-Embeddings/ (5 backbones, TemporalCNN
architecture) paired against archive/Pooled-Embeddings/ (same 5 backbones,
plain pooled-embedding MLP) as the "pre-TC-head" baseline.

x = 4 TC-head configs (+TC-head, +TC-head+frame-level polyphony,
+TC-head+frame-level call activity, +TC-head+both-aux). Per backbone: a jittered point
at each config, y = mean paired ΔMAE (config MAE minus that backbone's own
baseline MAE, paired by dataset) across the 8 datasets, with a ±1 SD inner
whisker. Per config: a black diamond = mean ΔMAE across the 5 backbones,
with a gray SD outer whisker (heterogeneous models -> SD), visually flagged
if it excludes zero.

Pairing is exact: baseline and config MAE are joined on (model, dataset),
which is a literal 1:1 key shared between the two studies.

Usage:
    complete-venv/bin/python source/scripts/plot_rq2.py [--out-dir plots/figures/rq2]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backbone_meta import BACKBONE_META
from plot_data import ALL_DATASETS, filter_long, load_study, mean_sd_n, paired_join
from plot_style import build_encoding_legend, draw_dot_whisker, save_fig, zero_ref_line

SOURCE = "synthetic_mixture_test"
SPATIAL_BACKBONES = ["AudioProtoPNet-20-BirdSet-XCL", "Bird-MAE-Huge",
                      "EfficientNet-B1-BirdSet-XCL", "NatureLMBEATs", "perch_v2_cpu"]

CONFIG_LABELS = ["+TC-head", "+TC-head+frame-level polyphony", "+TC-head+frame-level call activity", "+TC-head+both-aux"]
CONFIG_HEAD = {
    "+TC-head": "reg_reg",
    "+TC-head+frame-level polyphony": "reg_and_frame_reg_reg",
    "+TC-head+frame-level call activity": "reg_and_events_reg",
    "+TC-head+both-aux": "reg_and_events_and_frame_reg_reg",
}


def compute_baseline(pooled_long: pd.DataFrame) -> pd.DataFrame:
    return filter_long(pooled_long, source=SOURCE, metric="mae", head="reg",
                        model=SPATIAL_BACKBONES, dataset=ALL_DATASETS)


def compute_deltas(spatial_long: pd.DataFrame, baseline_indexed: pd.Series) -> pd.DataFrame:
    """Per (config, model, dataset): paired delta = config MAE - baseline MAE."""
    frames = []
    for label, head in CONFIG_HEAD.items():
        config_df = filter_long(spatial_long, source=SOURCE, metric="mae", head=head,
                                 model=SPATIAL_BACKBONES, dataset=ALL_DATASETS)
        merged = paired_join(config_df, baseline_indexed, on=("model", "dataset"))
        merged["config"] = label
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def build_figure(deltas: pd.DataFrame, out_dir: Path):
    backbone_order = [m for m in BACKBONE_META if m in SPATIAL_BACKBONES]
    # Slot the grand-mean diamond in among the backbones (roughly central)
    # rather than overlaying it at the config tick, so all 6 markers --
    # 5 backbones + mean -- sit at evenly spaced x-offsets with no overlap.
    half = len(backbone_order) // 2
    slot_order = backbone_order[:half] + ["__MEAN__"] + backbone_order[half:]
    jitter = dict(zip(slot_order, np.linspace(-0.2, 0.2, len(slot_order))))

    fig, ax = plt.subplots(figsize=(10, 5.5))

    per_bm = mean_sd_n(deltas, ["config", "model"], value_col="delta")

    zero_ref_line(ax)

    # Connecting line per backbone (its mean ΔMAE across the 4 configs),
    # drawn first so the dot+whisker points sit on top of it.
    for model in backbone_order:
        color = BACKBONE_META[model]["identity_color"]
        xs = [cx + jitter[model] for cx in range(len(CONFIG_LABELS))]
        means = [per_bm.loc[(label, model), "mean"] for label in CONFIG_LABELS]
        ax.plot(xs, means, color=color, linewidth=1.5, alpha=0.7, zorder=1)

    for cx, label in enumerate(CONFIG_LABELS):
        backbone_means = []
        for model in backbone_order:
            mean, sd, n = per_bm.loc[(label, model)]
            color = BACKBONE_META[model]["identity_color"]
            draw_dot_whisker(ax, cx + jitter[model], mean, sd,
                              facecolor=color, edgecolor=color, marker="o", s=60)
            backbone_means.append(mean)

        grand_mean = float(np.mean(backbone_means))
        grand_sd = float(np.std(backbone_means, ddof=1))
        mx = cx + jitter["__MEAN__"]
        ax.errorbar(mx, grand_mean, yerr=grand_sd, fmt="none", ecolor="0.4", capsize=5,
                    linewidth=1.2, zorder=4)
        ax.scatter([mx], [grand_mean], marker="D", facecolor="black", edgecolor="0.3",
                   linewidth=1.2, s=110, zorder=5)

    ax.set_xticks(range(len(CONFIG_LABELS)))
    ax.set_xticklabels(CONFIG_LABELS, rotation=12, ha="right")
    ax.set_ylabel(r"$\Delta$MAE vs. own baseline (negative = improvement)")
    ax.set_title("RQ2 -- Multi-task / prediction head (paired ΔMAE across 8 datasets)")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)

    legend_entries = [(BACKBONE_META[m]["display"], dict(marker="o", markerfacecolor=BACKBONE_META[m]["identity_color"],
                       markeredgecolor=BACKBONE_META[m]["identity_color"], color=BACKBONE_META[m]["identity_color"]))
                      for m in backbone_order]
    legend_entries.append(("Mean across 5 backbones", dict(marker="D", markerfacecolor="black", markeredgecolor="0.3", color="0.3")))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    build_encoding_legend(fig, [("Backbone", legend_entries)], bbox_to_anchor=(0.5, -0.02), entries_ncol=3)
    save_fig(fig, out_dir / "rq2_multitask")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq2"))
    args = parser.parse_args()

    pooled_long = load_study("pooled", models=SPATIAL_BACKBONES)
    spatial_long = load_study("spatial", models=SPATIAL_BACKBONES)

    baseline = compute_baseline(pooled_long)
    baseline_indexed = baseline.set_index(["model", "dataset"])["value"]

    deltas = compute_deltas(spatial_long, baseline_indexed)

    build_figure(deltas, args.out_dir)
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
