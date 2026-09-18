#!/usr/bin/env python3
"""RQ2 -- macro-MAE(1-6) variant of rq2_multitask_soundscape (single panel,
soundscape only), plus a mixed-metric combined figure (synthetic in ordinary
MAE, soundscape in macro-MAE(1-6)).

Reuses compute_rq2_per_level_accuracy.py's already-recovered-and-validated
macro_mae_1_6 per (backbone, config) cell (5 SPATIAL_BACKBONES x 5 TC-head
configs, pooled across the 7 REGIONAL_DATASETS and restricted to the
unambiguous-label subset -- see that script's docstring for the full
per-level/data-recovery methodology) as the y-axis of the same "ΔMAE vs.
TC-head-alone" dot+whisker chart plot_rq2.py's build_figure()/build_figure_
combined() draw, in place of the ordinary pooled range_mae.

There is no synthetic-domain macro-MAE, and so no true (single-metric)
combined figure: synthetic's raw per-clip predictions for the 4 recovered
TC-head configs are not preserved anywhere the same recovery technique can
reach -- each training run's own logs/test_results.pkl turns out to be
domain-specific to that run's dataset.subset (regional-subset runs only
ever wrote soundscape range-metrics there; XCM-subset runs only wrote
synthetic point-metrics), so there is no per-run backup of a regional
subset's synthetic predictions to recover a 4th/5th config's predictions
from. build_figure_combined_mixed() below therefore pairs synthetic's
*ordinary* ΔMAE (plot_rq2.py's own compute_tc_alone_indexed()/
compute_deltas(), unchanged) with soundscape's macro-MAE(1-6) -- two
different metrics side by side, not one metric across two domains -- and is
labeled accordingly (separate y-axis labels and panel titles per side,
unlike build_figure_combined()'s single shared label).

Soundscape's macro-MAE(1-6) is computed per region (compute_rq2_per_level_
accuracy.py's build_region_macro_mae_table(), mirroring plot_rq1.py's own
_region_macro_mae_1_6()/compute_soundscape_macro_mae_stats() construction
exactly: one macro-MAE(1-6) value per (backbone, config, region), reduced
to mean +/- SD across the 7 regions), not pooled across regions into a
single value -- so the soundscape panel's per-backbone dots get a genuine
per-region whisker, the same as the synthetic panel's, rather than only the
grand-mean diamond having one. The red "improves over pooled-MLP baseline"
ring build_figure_combined()
draws (via its `improved_deltas` argument) is also soundscape-macro-MAE-only
out of scope here -- it would need the same per-level recovery run against
archive/Pooled-Embeddings/ too, not just archive/Spatial-Embeddings/ -- so
the soundscape panel below carries no ring, only the synthetic panel does.

Usage:
    complete-venv/bin/python source/scripts/plot_rq2_macro_mae.py [--out-dir plots/figures/rq2]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backbone_meta import BACKBONE_META
from compute_rq2_per_level_accuracy import build_region_macro_mae_table
from plot_data import load_study
from plot_rq2 import (COMBINED_ERRORBAR_LIGHTEN, COMBINED_WIDTH_IN, CONFIG_LABELS, FIG_HEIGHT, LABEL_AREA_HEIGHT,
                       LEGENDS_VERTICAL_ANCHOR, LEGEND_ROW_SPACING, SINGLE_LABEL_AREA_HEIGHT, SINGLE_LEGEND_NCOL,
                       SINGLE_OMITTED_LABEL, SINGLE_XTICK_FONT_PT, SOURCE, SPATIAL_BACKBONES, _backbone_jitter,
                       _baseline_footnote, _legend_entries, compute_baseline, compute_deltas,
                       compute_tc_alone_indexed, draw_config_panel)
from plot_style import ONE_COL_WIDTH_IN, build_encoding_legend, save_fig, set_rcparams

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_region_deltas(region_table: pd.DataFrame) -> pd.DataFrame:
    """Same (config, model, dataset, delta) shape as plot_rq2.py's compute_
    deltas(), but delta = config's per-region macro-MAE(1-6) - bare-TC-
    head's macro-MAE(1-6), paired by (backbone, region) -- mirrors plot_rq1.
    py's per-region macro-MAE construction (mean +/- SD across the 7
    regions) instead of a single value pooled across all 7 regions, so
    draw_config_panel() gets a genuine per-backbone whisker here too."""
    tc_alone = region_table[region_table["config"] == "+TC-head"].set_index(["backbone", "region"])["macro_mae_1_6"]
    rows = []
    for label in CONFIG_LABELS:
        sub = region_table[region_table["config"] == label]
        for _, r in sub.iterrows():
            baseline = tc_alone.loc[(r["backbone"], r["region"])]
            rows.append({
                "config": label, "model": r["backbone"], "dataset": r["region"],
                "delta": r["macro_mae_1_6"] - baseline,
            })
    return pd.DataFrame(rows)


def build_figure_macro_mae(deltas: pd.DataFrame, out_dir: Path, filename: str = "rq2_multitask_soundscape_macro_mae"):
    """Same layout as plot_rq2.py's build_figure(), reused via draw_config_
    panel() (only the axis-label text and y-values differ)."""
    backbone_order = [m for m in BACKBONE_META if m in SPATIAL_BACKBONES]
    jitter = _backbone_jitter(backbone_order)

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_IN, FIG_HEIGHT))
    draw_config_panel(ax, deltas, backbone_order, jitter, omit_label=SINGLE_OMITTED_LABEL)
    ax.tick_params(axis="x", labelsize=SINGLE_XTICK_FONT_PT)
    ax.set_ylabel("")
    ax.text(0.0, 1.015, r"$\Delta$ macro-MAE(1-6) vs. TC-head (negative = improvement)",
            transform=ax.transAxes, ha="left", va="bottom")

    fig.tight_layout(rect=(0, SINGLE_LABEL_AREA_HEIGHT, 1, 1))
    build_encoding_legend(fig, [(None, _legend_entries(backbone_order))], bbox_to_anchor=(0.5, LEGENDS_VERTICAL_ANCHOR),
                          entries_ncol=SINGLE_LEGEND_NCOL, labelspacing=LEGEND_ROW_SPACING)
    _baseline_footnote(fig)
    save_fig(fig, out_dir / filename)
    plt.close(fig)


def build_figure_combined_mixed(synthetic_deltas: pd.DataFrame, synthetic_improved: pd.DataFrame,
                                 macro_mae_deltas: pd.DataFrame, out_dir: Path,
                                 filename: str = "rq2_multitask_combined_macro_mae"):
    """Mixed-metric combined figure: left panel = synthetic ΔMAE (plot_rq2.
    py's ordinary per-dataset deltas, unchanged, including the pooled-MLP-
    baseline red ring via `synthetic_improved`); right panel = soundscape
    Δmacro-MAE(1-6) (this module's build_deltas(), no red ring -- see module
    docstring). Same two-panel layout as plot_rq2.py's build_figure_
    combined(), but with independent y-axis labels per panel since the two
    sides are not the same metric."""
    backbone_order = [m for m in BACKBONE_META if m in SPATIAL_BACKBONES]
    jitter = _backbone_jitter(backbone_order)

    fig, axes = plt.subplots(1, 2, figsize=(COMBINED_WIDTH_IN, FIG_HEIGHT))
    draw_config_panel(axes[0], synthetic_deltas, backbone_order, jitter,
                      omit_label=SINGLE_OMITTED_LABEL, show_errorbars=True,
                      errorbar_lighten=COMBINED_ERRORBAR_LIGHTEN, connect_means=True, scale_by_dots=True,
                      improved_deltas=synthetic_improved)
    draw_config_panel(axes[1], macro_mae_deltas, backbone_order, jitter,
                      omit_label=SINGLE_OMITTED_LABEL, show_errorbars=True,
                      errorbar_lighten=COMBINED_ERRORBAR_LIGHTEN, connect_means=True, scale_by_dots=True,
                      improved_deltas=None)
    axes[0].set_title("Synthetic")
    axes[1].set_title("Soundscape (macro-MAE 1-6)")
    axes[0].set_ylabel(r"$\Delta$MAE vs. TC-head (negative = improvement)")
    axes[1].set_ylabel(r"$\Delta$ macro-MAE(1-6) vs. TC-head (negative = improvement)")

    fig.tight_layout(rect=(0, LABEL_AREA_HEIGHT, 1, 1))
    build_encoding_legend(fig, [(None, _legend_entries(backbone_order))], bbox_to_anchor=(0.5, LEGENDS_VERTICAL_ANCHOR),
                          entries_ncol=3, labelspacing=LEGEND_ROW_SPACING)
    _baseline_footnote(fig)
    save_fig(fig, out_dir / filename)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq2"))
    args = parser.parse_args()

    set_rcparams()
    region_table = build_region_macro_mae_table()
    macro_mae_deltas = build_region_deltas(region_table)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    build_figure_macro_mae(macro_mae_deltas, args.out_dir)
    print(f"Wrote {args.out_dir / 'rq2_multitask_soundscape_macro_mae'}.(png|pdf)")

    # Synthetic side: plot_rq2.py's own ordinary ΔMAE machinery, unchanged.
    spatial_long = load_study("spatial", models=SPATIAL_BACKBONES)
    pooled_long = load_study("pooled", models=SPATIAL_BACKBONES)
    tc_alone_indexed = compute_tc_alone_indexed(spatial_long, SOURCE)
    synthetic_deltas = compute_deltas(spatial_long, tc_alone_indexed, SOURCE)
    baseline_indexed = compute_baseline(pooled_long, SOURCE).set_index(["model", "dataset"])["value"]
    synthetic_improved = compute_deltas(spatial_long, baseline_indexed, SOURCE)

    build_figure_combined_mixed(synthetic_deltas, synthetic_improved, macro_mae_deltas, args.out_dir)
    print(f"Wrote {args.out_dir / 'rq2_multitask_combined_macro_mae'}.(png|pdf)")


if __name__ == "__main__":
    main()
