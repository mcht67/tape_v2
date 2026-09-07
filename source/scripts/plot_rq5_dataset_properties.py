#!/usr/bin/env python3
"""
RQ5 (additional) -- Panel A (region-specific head vs. XCM head, mean +/- SD
across 11 backbones) reproduced once per dataset property from
plots/data/polybirdmix_soundscape_stats.json, instead of just species count.

Grid: one row per property (num_species, num_segments, ratio_polyp,
mean_polyp), 2 columns (synthetic test, soundscape) -- same line/whisker
styling as rq5_grid's Panel A, with the x-axis in each row ordered
ascending by that row's property and tick labels showing its value. All
8 sub-panels share one y-scale so the underlying MAE values (which don't
change, only their left-right order does) stay directly comparable.

Usage:
    complete-venv/bin/python source/scripts/plot_rq5_dataset_properties.py [--out-dir plots/figures/rq5]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from plot_data import REGIONAL_DATASETS, load_study
from plot_rq5 import SCAPE_SOURCE, STATS_JSON, SYN_SOURCE, build_panel_a
from plot_style import save_fig

PROPERTIES = {
    "num_species": ("#Species", lambda s: s["n_unique_species"], "{:d}"),
    "num_segments": ("#Segments", lambda s: s["n_soundscape_segments"], "{:d}"),
    "ratio_polyp": ("Ratio polyphonic", lambda s: s["polyphony_metric_summary"]["ratio_polyp"]["mean"], "{:.2f}"),
    "mean_polyp": ("Mean polyphony", lambda s: s["polyphony_metric_summary"]["mean_polyp"]["mean"], "{:.2f}"),
}


def property_ordered_datasets(stats, prop_key):
    label, getter, fmt = PROPERTIES[prop_key]
    ordered = sorted(REGIONAL_DATASETS, key=lambda d: getter(stats[d]))
    labels = [f"{d} ({fmt.format(getter(stats[d]))})" for d in ordered]
    return ordered, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq5"))
    args = parser.parse_args()

    stats = json.loads(STATS_JSON.read_text())["subsets"]

    xg_models = mrt.discover_models(Path("archive/XCM-Generalization"))
    pooled_long = load_study("pooled", models=xg_models)
    xg_long = load_study("xcm_gen", models=xg_models)

    prop_keys = list(PROPERTIES.keys())
    fig, axes = plt.subplots(len(prop_keys), 2, figsize=(13, 4.2 * len(prop_keys)), constrained_layout=True)

    for row, prop_key in enumerate(prop_keys):
        prop_label = PROPERTIES[prop_key][0]
        dataset_order, tick_labels = property_ordered_datasets(stats, prop_key)

        xs = build_panel_a(axes[row, 0], pooled_long, xg_long, SYN_SOURCE, "mae", dataset_order,
                            f"Synthetic test -- ordered by {prop_label}")
        build_panel_a(axes[row, 1], pooled_long, xg_long, SCAPE_SOURCE, "range_mae", dataset_order,
                      f"Soundscape -- ordered by {prop_label}")
        axes[row, 0].set_ylabel("MAE (lower is better)")
        for ax in axes[row]:
            ax.set_xticks(list(xs))
            ax.set_xticklabels(tick_labels, rotation=30, ha="right")
        if row == 0:
            axes[row, 0].legend(loc="upper left", frameon=False)

    # Shared y-scale across all 8 sub-panels.
    lo = min(ax.get_ylim()[0] for ax in axes.flat)
    hi = max(ax.get_ylim()[1] for ax in axes.flat)
    for ax in axes.flat:
        ax.set_ylim(lo, hi)

    fig.suptitle("RQ5 -- Region-specific vs. XCM head, ordered by each dataset property", fontsize=14)
    save_fig(fig, args.out_dir / "rq5_dataset_properties")
    plt.close(fig)
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
