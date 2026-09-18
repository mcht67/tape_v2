#!/usr/bin/env python3
"""
RQ5 (additional) -- Panel A (region-specific head vs. XCM head, mean +/- SD
across 11 backbones) reproduced once per dataset property from
plots/data/polybirdmix_soundscape_stats.json, instead of just species count.

Grid: one row per property (num_species, num_segments, n_train_segments,
n_species, n_detections, min_polyphony, max_polyphony, ratio_polyp,
mean_polyp, max_polyp, species_per_detection -- see PROPERTIES), 2 columns
(synthetic test, soundscape) -- same line/whisker styling as rq5_grid's
Panel A, with the x-axis in each row ordered ascending by that row's
property and tick labels showing its value. All sub-panels share one
y-scale so the underlying MAE values (which don't change, only their
left-right order does) stay directly comparable.

The soundscape column additionally carries the "XCM fine-tune head" line
plus backbone-matched "Region-specific head (4 backbones)" and "XCM head
(4 backbones)" variants of the first two lines (see plot_rq5.py's
build_panel_a()/rq5_panel_a_soundscape) -- all three restricted to the 5
backbones that were actually fine-tuned, so both frozen baselines are
backbone-matched to the fine-tune line instead of averaged over all 11.
The synthetic column stays at its original 2 lines.

A second figure, rq5_dataset_properties_matched, is the same 11x2 grid but
with every panel (both columns) showing only the 3 backbone-matched lines
(build_panel_a(show_full=False, ...)) -- a fully symmetric, backbone-matched
counterpart to the mixed 2-lines/5-lines figure above.

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
from backbone_meta import FINETUNE_NAME_TO_CANONICAL
from plot_data import REGIONAL_DATASETS, load_study
from plot_rq5 import SCAPE_SOURCE, STATS_JSON, SYN_SOURCE, build_panel_a
from plot_style import save_fig, set_rcparams, TWO_COL_WIDTH_IN

PROPERTIES = {
    "num_species": ("#Species", lambda s: s["n_unique_species"], "{:d}"),
    "num_segments": ("#Segments", lambda s: s["n_soundscape_segments"], "{:d}"),
    "n_train_segments": ("#Train segments", lambda s: s["n_train_segments"], "{:d}"),
    "n_species": ("Mean #species/segment", lambda s: s["distributions"]["n_species"]["mean"], "{:.2f}"),
    "n_detections": ("Mean #detections/segment", lambda s: s["distributions"]["n_detections"]["mean"], "{:.2f}"),
    "min_polyphony": ("Mean min polyphony", lambda s: s["distributions"]["min_polyphony"]["mean"], "{:.2f}"),
    "max_polyphony": ("Mean max polyphony", lambda s: s["distributions"]["max_polyphony"]["mean"], "{:.2f}"),
    "ratio_polyp": ("Ratio polyphonic", lambda s: s["polyphony_metric_summary"]["ratio_polyp"]["mean"], "{:.2f}"),
    "mean_polyp": ("Mean polyphony", lambda s: s["polyphony_metric_summary"]["mean_polyp"]["mean"], "{:.2f}"),
    # max-polyp = max_{i=1,...,n} p_i -- per-segment max over sources of the
    # polyphony metric p_i, averaged across segments (distinct from
    # max_polyphony above, which is the mean *count* of detections/segment).
    "max_polyp": ("Mean max-polyp", lambda s: s["polyphony_metric_summary"]["max_polyp"]["mean"], "{:.2f}"),
    # #species / mean #detections-per-segment: a combined acoustic-sparsity-
    # relative-to-diversity measure -- high when a region has many distinct
    # species but each is rarely detected per segment. Added because it
    # correlates with the cross-region generalization gap (r=0.82, p=0.023,
    # soundscape) much more strongly than either #species or mean
    # #detections/segment does on its own (see compute_rq5_diff_correlations.py).
    "species_per_detection": ("Species per detection", lambda s: s["n_unique_species"] / s["distributions"]["n_detections"]["mean"], "{:.1f}"),
}


def property_ordered_datasets(stats, prop_key):
    label, getter, fmt = PROPERTIES[prop_key]
    ordered = sorted(REGIONAL_DATASETS, key=lambda d: getter(stats[d]))
    labels = [f"{d} ({fmt.format(getter(stats[d]))})" for d in ordered]
    return ordered, labels


def build_matched_grid(pooled_long, xg_long, ft_long, stats, matched_backbones, out_dir):
    """Same 11x2 grid as main()'s rq5_dataset_properties, but every panel
    (both columns, not just soundscape) shows only the 3 lines averaged over
    the 4 backbones also fine-tuned (build_panel_a(show_full=False, ...)) --
    a fully backbone-matched counterpart, instead of the original figure's
    asymmetric 2-lines-synthetic/5-lines-soundscape panels."""
    prop_keys = list(PROPERTIES.keys())
    fig, axes = plt.subplots(len(prop_keys), 2, figsize=(TWO_COL_WIDTH_IN, 2.15 * len(prop_keys)),
                              constrained_layout=True)

    for row, prop_key in enumerate(prop_keys):
        prop_label = PROPERTIES[prop_key][0]
        dataset_order, tick_labels = property_ordered_datasets(stats, prop_key)

        xs = build_panel_a(axes[row, 0], pooled_long, xg_long, SYN_SOURCE, "mae", dataset_order,
                            "Synthetic test" if row == 0 else "", ft_long=ft_long,
                            matched_backbones=matched_backbones, show_full=False)
        build_panel_a(axes[row, 1], pooled_long, xg_long, SCAPE_SOURCE, "range_mae", dataset_order,
                      "Soundscape" if row == 0 else "", ft_long=ft_long,
                      matched_backbones=matched_backbones, show_full=False)
        axes[row, 0].set_ylabel(f"MAE\n(ordered by {prop_label})")
        for ax in axes[row]:
            ax.set_xticks(list(xs))
            ax.set_xticklabels(tick_labels, rotation=75, ha="right")
        if row == 0:
            axes[row, 0].legend(loc="upper left", frameon=False, fontsize=7)

    # Shared y-scale across all 22 sub-panels.
    lo = min(ax.get_ylim()[0] for ax in axes.flat)
    hi = max(ax.get_ylim()[1] for ax in axes.flat)
    for ax in axes.flat:
        ax.set_ylim(lo, hi)

    # No in-figure heading -- the caption describes this figure already.
    save_fig(fig, out_dir / "rq5_dataset_properties_matched")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq5"))
    args = parser.parse_args()

    set_rcparams()
    stats = json.loads(STATS_JSON.read_text())["subsets"]

    xg_models = mrt.discover_models(Path("archive/XCM-Generalization"))
    pooled_long = load_study("pooled", models=xg_models)
    xg_long = load_study("xcm_gen", models=xg_models)
    # models= excludes perch_v2's raw folder (still on disk but not a
    # genuine fine-tuning result -- see backbone_meta.py's comment).
    ft_long = load_study("xcm_gen_ft", models=list(FINETUNE_NAME_TO_CANONICAL.keys()))
    matched_backbones = list(FINETUNE_NAME_TO_CANONICAL.values())

    prop_keys = list(PROPERTIES.keys())
    fig, axes = plt.subplots(len(prop_keys), 2, figsize=(TWO_COL_WIDTH_IN, 2.15 * len(prop_keys)),
                              constrained_layout=True)

    for row, prop_key in enumerate(prop_keys):
        prop_label = PROPERTIES[prop_key][0]
        dataset_order, tick_labels = property_ordered_datasets(stats, prop_key)

        # Column titles ("Synthetic test"/"Soundscape") only on the top row;
        # the ordering property is given by the shared y-axis label instead,
        # keeping each panel title short enough to fit at this column width.
        xs = build_panel_a(axes[row, 0], pooled_long, xg_long, SYN_SOURCE, "mae", dataset_order,
                            "Synthetic test" if row == 0 else "")
        build_panel_a(axes[row, 1], pooled_long, xg_long, SCAPE_SOURCE, "range_mae", dataset_order,
                      "Soundscape" if row == 0 else "", ft_long=ft_long, matched_backbones=matched_backbones)
        axes[row, 0].set_ylabel(f"MAE\n(ordered by {prop_label})")
        for ax in axes[row]:
            ax.set_xticks(list(xs))
            ax.set_xticklabels(tick_labels, rotation=75, ha="right")
        if row == 0:
            axes[row, 0].legend(loc="upper left", frameon=False)
            # Soundscape's own legend -- it has 3 extra lines (XCM fine-tune
            # head, Region-specific head (4 backbones), XCM head (4
            # backbones)) the synthetic column's legend doesn't carry.
            axes[row, 1].legend(loc="upper left", frameon=False, fontsize=6)

    # Shared y-scale across all 8 sub-panels.
    lo = min(ax.get_ylim()[0] for ax in axes.flat)
    hi = max(ax.get_ylim()[1] for ax in axes.flat)
    for ax in axes.flat:
        ax.set_ylim(lo, hi)

    # No in-figure heading -- the caption describes this figure already.
    save_fig(fig, args.out_dir / "rq5_dataset_properties")
    plt.close(fig)

    build_matched_grid(pooled_long, xg_long, ft_long, stats, matched_backbones, args.out_dir)
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
