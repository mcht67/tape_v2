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

Panel B: rows = the 4 XCM-fine-tuned backbones (archive/XCM-Generalization-
fine-tune/), columns = same 7 datasets/order as Panel A, cell = single MAE
value (no whiskers), shared sequential color scale (light = low MAE) across
both sub-panels. Missing cells (e.g. a backbone/dataset combo that was never
run) are masked rather than shown as 0.

perch_v2 is excluded from every "fine-tuned"/"5 backbones" set below (via
backbone_meta.FINETUNE_NAME_TO_CANONICAL, and via an explicit models=
restriction on every load_study("xcm_gen_ft") call in this file, since that
archive's raw folder for perch_v2 still exists on disk): its SavedModel
backbone cannot receive gradients at all (jax2tf export without
with_gradient=True), so its archive data under XCM-Generalization-fine-tune
only ever trained the head on frozen embeddings, identical to the frozen
baseline, not a genuine fine-tuning result -- see backbone_meta.py's comment
for the full explanation.

Two standalone single-column figures reproduce the soundscape sub-panels of
each row (rq5_panel_a_soundscape, rq5_panel_b_soundscape) -- the heatmap one
(Panel B) unchanged (still the 4 fine-tuned backbones only; see the two
further per-study heatmaps below for the other 11-backbone studies), but the
line chart (Panel A) gains 3 more lines: "XCM
fine-tune head" (mean +/- SD across the same 4 fine-tuned backbones as
Panel B, evaluated per-dataset rather than pooled), plus backbone-matched
"Region-specific head (4 backbones)" and "XCM head (4 backbones)" variants
of the first two lines, restricted to that same 4-backbone set. The matched
pair isolates fine-tuning's effect from the backbone-selection difference
between the studies (only 4 of the 11 Panel-A backbones were ever
fine-tuned), so all three of the fine-tuned backbones' reference points
(their own region-specific ceiling, their frozen cross-region baseline, and
their fine-tuned result) are visible on one chart, not just per-backbone in
Panel B's heatmap. rq5_grid's own Panel A is unchanged (still 2 lines) to
keep it consistent with the synthetic sub-panel next to it.

A further standalone figure, rq5_panel_a_soundscape_matched, is the same
line chart but with the two 11-backbone lines dropped -- only the three
4-backbone-matched lines remain, for a like-for-like 3-way comparison with
no backbone-population mismatch anywhere on the chart.

rq5_panel_a_soundscape_matched_macro_mae is that same 3-line, 4-backbone-
matched chart again, but plotting macro-MAE(1-6) (unweighted mean of
per-true-level MAE, levels 1-6, on the unambiguous-label subset -- plot_
rq1.py's _region_macro_mae_1_6(), see compute_macro_mae_long()) instead of
range_mae in each of the three per-dataset backbone means, addressing the
same level-1-dominance concern as plot_rq1.py's build_figure_macro_mae() and
compute_soundscape_per_level_accuracy.py's macro-accuracy columns, for this
backbone-matched 3-way comparison rather than RQ1's single-study backbone
ranking.

Two further standalone heatmaps, rq5_panel_b_soundscape_xcm_head and
rq5_panel_b_soundscape_region_specific, are Panel B's soundscape heatmap
reproduced for the other two 11-backbone studies ("XCM head" =
archive/XCM-Generalization/, frozen backbone; "Region-specific head" =
archive/Pooled-Embeddings/) instead of the 4 fine-tuned backbones -- same
row-per-backbone/column-per-dataset layout and own (not cross-figure-shared)
color scale, figure height scaled up for the larger row count.

Two more, rq5_panel_b_soundscape_xcm_head_matched and rq5_panel_b_soundscape_
region_specific_matched, are those same two heatmaps again but restricted to
the 4 backbones that were actually fine-tuned (row_order=ft_row_order instead
of xg_row_order) -- backbone-matched counterparts to rq5_panel_b_soundscape
(the fine-tuned heatmap), so all three studies can be compared row-for-row
over the identical 4 backbones.

rq5_panel_b_soundscape_matched_grid combines those three backbone-matched
heatmaps (region-specific, XCM head frozen, XCM fine-tune head) into one
figure, stacked with a single shared color scale, so the identical 4x7 grid
can be compared cell-for-cell across all three studies directly instead of
via 3 separately-scaled standalone figures.
rq5_panel_b_soundscape_matched_grid_macro_mae is that same grid with
macro-MAE(1-6) in each cell instead of range_mae (the per-cell values are
also written out by compute_rq5_macro_mae_summary.py).

Two further standalone figures, rq5_confusion_matrix_xcm_head_soundscape
and rq5_confusion_matrix_region_specific_soundscape, are confusion matrices
(reusing plot_confusion_matrix.py) for the "XCM head" and "Region-specific
head" respectively on the soundscape test, each pooled across all 11 Panel-A
backbones and all 7 datasets -- the study-level error distribution behind
that line's mean +/- SD, e.g. to see whether it's systematically biased
rather than just noisy.

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
from plot_confusion_matrix import ARCHIVE as CM_ARCHIVE
from plot_confusion_matrix import SPINE_COLOR as CM_SPINE_COLOR
from plot_confusion_matrix import draw_confusion_matrix, load_soundscape_multi, prune_confusion_matrix
from plot_data import REGIONAL_DATASETS, filter_long, load_study, mean_sd_n
from plot_rq1 import _region_macro_mae_1_6
from plot_style import draw_dot_whisker, mask_missing, save_fig, set_rcparams, ONE_COL_WIDTH_IN, TWO_COL_WIDTH_IN

SYN_SOURCE = "synthetic_mixture_test"
SCAPE_SOURCE = "soundscape_test"
STATS_JSON = Path(__file__).resolve().parents[2] / "plots" / "data" / "polybirdmix_soundscape_stats.json"
ACCENT_COLOR = "#d62728"
FT_ACCENT_COLOR = "#2ca02c"


def species_ordered_datasets():
    stats = json.loads(STATS_JSON.read_text())["subsets"]
    ordered = sorted(REGIONAL_DATASETS, key=lambda d: stats[d]["n_unique_species"])
    labels = [f"{d} (n={stats[d]['n_unique_species']})" for d in ordered]
    return ordered, labels


# ---------------------------------------------------------------------------
# Panel A -- line plot
# ---------------------------------------------------------------------------

def build_panel_a(ax, pooled_long, xg_long, source, metric, dataset_order, title, ft_long=None,
                   matched_backbones=None, show_full=True):
    """ft_long: optional archive/XCM-Generalization-fine-tune long df -- when
    given, adds a 3rd line (mean +/- SD across its 4 fine-tuned backbones per
    dataset, same style as the other two) so fine-tuning's effect on
    cross-region generalization can be read directly off this chart, instead
    of only per-backbone in Panel B's heatmap. Only build_panel_a_single()
    (the standalone soundscape line chart) passes this; rq5_grid's Panel A
    stays at its original 2 lines.

    matched_backbones: optional list of canonical model names -- when given
    (together with ft_long), adds two more lines: "Region-specific head" and
    "XCM head" (same pooled_long/xg_long/metric as the 1st and 2nd lines)
    but each restricted to just this backbone set, so both are directly
    backbone-matched to the fine-tune line instead of averaged over all 11
    Panel-A backbones. Isolates fine-tuning's effect from the backbone-
    selection difference between the studies (only 4 of the 11 Panel-A
    backbones were ever fine-tuned).

    show_full: set False to skip the 11-backbone "Region-specific head" /
    "XCM head" lines entirely, leaving only the matched_backbones/ft_long
    lines (all three then averaged over the same 4 backbones) -- see
    build_panel_a_soundscape_matched()."""
    xs = range(len(dataset_order))

    if show_full:
        region_specific = mean_sd_n(
            filter_long(pooled_long, source=source, metric=metric, head="reg", dataset=dataset_order), ["dataset"]
        ).reindex(dataset_order)
        xcm_head = mean_sd_n(
            filter_long(xg_long, source=source, metric=metric, head="reg", dataset=dataset_order), ["dataset"]
        ).reindex(dataset_order)

        ax.plot(xs, region_specific["mean"], color="0.5", linestyle="--", linewidth=1.5, zorder=1, label="Region-specific head")
        ax.errorbar(xs, region_specific["mean"], yerr=region_specific["std"], fmt="none", ecolor="0.5", capsize=4, zorder=1)
        ax.scatter(xs, region_specific["mean"], facecolor="0.5", edgecolor="0.5", s=50, zorder=2)

        ax.plot(xs, xcm_head["mean"], color=ACCENT_COLOR, linewidth=2.5, zorder=3, label="XCM head")
        ax.errorbar(xs, xcm_head["mean"], yerr=xcm_head["std"], fmt="none", ecolor=ACCENT_COLOR, capsize=4, zorder=3)
        ax.scatter(xs, xcm_head["mean"], facecolor=ACCENT_COLOR, edgecolor=ACCENT_COLOR, s=60, zorder=4)

    if matched_backbones is not None:
        region_specific_matched = mean_sd_n(
            filter_long(pooled_long, source=source, metric=metric, head="reg", dataset=dataset_order,
                        model=matched_backbones), ["dataset"]
        ).reindex(dataset_order)
        ax.plot(xs, region_specific_matched["mean"], color="0.5", linestyle=":", linewidth=1.5, zorder=2.5,
                label="Region-specific head (4 backbones)")
        ax.errorbar(xs, region_specific_matched["mean"], yerr=region_specific_matched["std"], fmt="none",
                     ecolor="0.5", capsize=4, zorder=2.5)
        ax.scatter(xs, region_specific_matched["mean"], facecolor="none", edgecolor="0.5", marker="^", s=50, zorder=2.5)

        xcm_head_matched = mean_sd_n(
            filter_long(xg_long, source=source, metric=metric, head="reg", dataset=dataset_order,
                        model=matched_backbones), ["dataset"]
        ).reindex(dataset_order)
        ax.plot(xs, xcm_head_matched["mean"], color=ACCENT_COLOR, linestyle=":", linewidth=1.5, zorder=3.5,
                label="XCM head (4 backbones)")
        ax.errorbar(xs, xcm_head_matched["mean"], yerr=xcm_head_matched["std"], fmt="none",
                     ecolor=ACCENT_COLOR, capsize=4, zorder=3.5)
        ax.scatter(xs, xcm_head_matched["mean"], facecolor="none", edgecolor=ACCENT_COLOR, marker="^", s=50, zorder=3.5)

    if ft_long is not None:
        ft_head = mean_sd_n(
            filter_long(ft_long, source=source, metric=metric, head="reg", dataset=dataset_order), ["dataset"]
        ).reindex(dataset_order)
        ax.plot(xs, ft_head["mean"], color=FT_ACCENT_COLOR, linewidth=2.5, zorder=5, label="XCM fine-tune head")
        ax.errorbar(xs, ft_head["mean"], yerr=ft_head["std"], fmt="none", ecolor=FT_ACCENT_COLOR, capsize=4, zorder=5)
        ax.scatter(xs, ft_head["mean"], facecolor=FT_ACCENT_COLOR, edgecolor=FT_ACCENT_COLOR, marker="s", s=50, zorder=6)

    ax.set_title(title)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    return xs


# ---------------------------------------------------------------------------
# Panel B -- heatmap
# ---------------------------------------------------------------------------

def build_heatmap_matrix(long_df, source, metric, dataset_order, row_order, model_map=None):
    """model_map: optional {raw model name: canonical name} -- only the
    fine-tune study's model dirs need this (their folder names, e.g.
    "BirdSetBirdMAE", aren't the canonical BACKBONE_META keys); Pooled-
    Embeddings/XCM-Generalization model dirs already are canonical, so
    callers pass None for those."""
    sub = filter_long(long_df, source=source, metric=metric, head="reg", dataset=dataset_order)
    sub = sub.copy()
    if model_map is not None:
        sub["model"] = sub["model"].map(model_map)
    matrix = sub.pivot_table(index="model", columns="dataset", values="value")
    return matrix.reindex(index=row_order, columns=dataset_order)


def draw_heatmap_panel(ax, matrix, dataset_order, tick_labels, row_order, cmap, vmin, vmax, title=None):
    """Shared by rq5_grid's Panel B and the standalone single-column heatmap
    figure below."""
    im = ax.imshow(mask_missing(matrix), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(dataset_order)))
    ax.set_xticklabels(tick_labels, rotation=75, ha="right")
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels([BACKBONE_META[m]["display"] for m in row_order])
    if title:
        ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if val > (vmin + vmax) / 2 else "black")
    return im


# ---------------------------------------------------------------------------
# Standalone single-column soundscape figures
# ---------------------------------------------------------------------------

def build_panel_a_soundscape_single(pooled_long, xg_long, ft_long, dataset_order, tick_labels, out_dir):
    """Panel A's soundscape line chart as its own one-column figure -- with a
    3rd "XCM fine-tune head" line and a 4th "Region-specific head (5
    backbones)" line (see build_panel_a()'s ft_long/matched_backbones) that
    rq5_grid's combined Panel A does not have."""
    matched_backbones = list(FINETUNE_NAME_TO_CANONICAL.values())
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_IN, 3.3))
    xs = build_panel_a(ax, pooled_long, xg_long, SCAPE_SOURCE, "range_mae", dataset_order, title=None,
                        ft_long=ft_long, matched_backbones=matched_backbones)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(tick_labels, rotation=75, ha="right")
    ax.set_ylabel("MAE (lower is better)")
    ax.legend(loc="upper left", frameon=False, fontsize=7)

    fig.tight_layout()
    save_fig(fig, out_dir / "rq5_panel_a_soundscape")
    plt.close(fig)


def build_panel_a_soundscape_matched(pooled_long, xg_long, ft_long, dataset_order, tick_labels, out_dir):
    """Same 3 backbone-matched lines as build_panel_a_soundscape_single()
    (all averaged over the same 4 fine-tuned backbones), but as this
    figure's only content -- the two 11-backbone lines are dropped rather
    than just added alongside. Drawn directly (not via build_panel_a())
    since its legend/line-style convention is bespoke to this figure:
    "Region-specific" is the solid line here (build_panel_a()'s matched
    lines are dotted, to stay visually secondary to its own 11-backbone
    solid lines -- with those absent, region-specific becomes this chart's
    own reference line instead)."""
    matched_backbones = list(FINETUNE_NAME_TO_CANONICAL.values())
    region_specific = mean_sd_n(
        filter_long(pooled_long, source=SCAPE_SOURCE, metric="range_mae", head="reg", dataset=dataset_order,
                    model=matched_backbones), ["dataset"]
    ).reindex(dataset_order)
    xcm_head = mean_sd_n(
        filter_long(xg_long, source=SCAPE_SOURCE, metric="range_mae", head="reg", dataset=dataset_order,
                    model=matched_backbones), ["dataset"]
    ).reindex(dataset_order)
    ft_head = mean_sd_n(
        filter_long(ft_long, source=SCAPE_SOURCE, metric="range_mae", head="reg", dataset=dataset_order), ["dataset"]
    ).reindex(dataset_order)

    xs = range(len(dataset_order))
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_IN, 3.3))

    ax.plot(xs, region_specific["mean"], color="0.5", linewidth=2.5, zorder=4, label="Region-specific")
    ax.errorbar(xs, region_specific["mean"], yerr=region_specific["std"], fmt="none", ecolor="0.5", capsize=4, zorder=4)
    ax.scatter(xs, region_specific["mean"], facecolor="0.5", edgecolor="0.5", s=60, zorder=5)

    ax.plot(xs, xcm_head["mean"], color=ACCENT_COLOR, linestyle=":", linewidth=1.5, zorder=2, label="XCM frozen backbone")
    ax.errorbar(xs, xcm_head["mean"], yerr=xcm_head["std"], fmt="none", ecolor=ACCENT_COLOR, capsize=4, zorder=2)
    ax.scatter(xs, xcm_head["mean"], facecolor="none", edgecolor=ACCENT_COLOR, marker="^", s=50, zorder=3)

    ax.plot(xs, ft_head["mean"], color=FT_ACCENT_COLOR, linestyle="--", linewidth=1.5, zorder=2, label="XCM fine-tuned")
    ax.errorbar(xs, ft_head["mean"], yerr=ft_head["std"], fmt="none", ecolor=FT_ACCENT_COLOR, capsize=4, zorder=2)
    ax.scatter(xs, ft_head["mean"], facecolor=FT_ACCENT_COLOR, edgecolor=FT_ACCENT_COLOR, marker="s", s=50, zorder=3)

    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(tick_labels, rotation=75, ha="right")
    ax.set_ylabel("MAE (lower is better)")
    ax.legend(loc="upper left", frameon=False, fontsize=7)

    fig.tight_layout()
    save_fig(fig, out_dir / "rq5_panel_a_soundscape_matched")
    plt.close(fig)


def compute_macro_mae_long(archive_dir, models, dataset_order, model_map=None):
    """Per-(model, dataset) macro-MAE(1-6) for one archive/{study}/ dir --
    plot_rq1.py's _region_macro_mae_1_6() (same levels-1-6,
    unambiguous-label-only definition, same per-region NaN/thin-support
    caveats -- see that function's docstring) applied per dataset instead of
    reduced to a single per-model mean across all 7 regions, so it slots
    into build_panel_a_soundscape_matched_macro_mae() the same way
    range_mae's pooled_long/xg_long/ft_long do: a tidy (model, dataset,
    value) long table that mean_sd_n() reduces across backbones per dataset.

    model_map: optional {raw model dir name: canonical name}, same as
    build_heatmap_matrix()'s -- only the fine-tune study's archive needs it."""
    rows = []
    for model in models:
        model_dir = archive_dir / model
        canonical = model_map[model] if model_map is not None else model
        for dataset in dataset_order:
            macro_mae, n_levels = _region_macro_mae_1_6(model_dir, dataset)
            rows.append({"model": canonical, "dataset": dataset, "value": macro_mae, "n_levels_used": n_levels})
    long = pd.DataFrame(rows)
    thin = long[long["n_levels_used"] < 6]
    if len(thin):
        print(f"compute_macro_mae_long({archive_dir.name}): {len(thin)}/{len(long)} (model, dataset) cells used "
              f"fewer than 6 levels for their macro-MAE(1-6) average.")
    return long


def build_panel_a_soundscape_matched_macro_mae(pooled_macro_long, xg_macro_long, ft_macro_long, dataset_order,
                                                tick_labels, out_dir):
    """Same figure as build_panel_a_soundscape_matched() (3 lines, all mean
    +/- SD across the 4 fine-tuned backbones per dataset, same styling), but
    plotting macro-MAE(1-6) (see compute_macro_mae_long()) instead of
    range_mae -- addresses the same level-1-dominance concern documented for
    plot_rq1.py's build_figure_macro_mae() and compute_soundscape_per_level_
    accuracy.py's macro-accuracy columns, for this backbone-matched 3-way
    comparison rather than RQ1's single-study backbone ranking."""
    region_specific = mean_sd_n(pooled_macro_long, ["dataset"]).reindex(dataset_order)
    xcm_head = mean_sd_n(xg_macro_long, ["dataset"]).reindex(dataset_order)
    ft_head = mean_sd_n(ft_macro_long, ["dataset"]).reindex(dataset_order)

    xs = range(len(dataset_order))
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_IN, 3.3))

    ax.plot(xs, region_specific["mean"], color="0.5", linewidth=2.5, zorder=4, label="Region-specific")
    ax.errorbar(xs, region_specific["mean"], yerr=region_specific["std"], fmt="none", ecolor="0.5", capsize=4, zorder=4)
    ax.scatter(xs, region_specific["mean"], facecolor="0.5", edgecolor="0.5", s=60, zorder=5)

    ax.plot(xs, xcm_head["mean"], color=ACCENT_COLOR, linestyle=":", linewidth=1.5, zorder=2, label="XCM frozen backbone")
    ax.errorbar(xs, xcm_head["mean"], yerr=xcm_head["std"], fmt="none", ecolor=ACCENT_COLOR, capsize=4, zorder=2)
    ax.scatter(xs, xcm_head["mean"], facecolor="none", edgecolor=ACCENT_COLOR, marker="^", s=50, zorder=3)

    ax.plot(xs, ft_head["mean"], color=FT_ACCENT_COLOR, linestyle="--", linewidth=1.5, zorder=2, label="XCM fine-tuned")
    ax.errorbar(xs, ft_head["mean"], yerr=ft_head["std"], fmt="none", ecolor=FT_ACCENT_COLOR, capsize=4, zorder=2)
    ax.scatter(xs, ft_head["mean"], facecolor=FT_ACCENT_COLOR, edgecolor=FT_ACCENT_COLOR, marker="s", s=50, zorder=3)

    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(tick_labels, rotation=75, ha="right")
    ax.set_ylabel("Macro-MAE(1-6) (lower is better)")
    ax.legend(loc="upper left", frameon=False, fontsize=7)

    fig.tight_layout()
    save_fig(fig, out_dir / "rq5_panel_a_soundscape_matched_macro_mae")
    plt.close(fig)


def build_heatmap_soundscape_single(long_df, dataset_order, tick_labels, row_order, out_dir, filename,
                                     model_map=None):
    """Panel B's soundscape heatmap as its own one-column figure, for any of
    the 3 studies (long_df/row_order/model_map picks which). Uses its own
    vmin/vmax (not shared with the synthetic panel, unlike rq5_grid) since
    there is no neighboring synthetic panel here to stay comparable with.
    Figure height scales with row count, so this also fits the 11-backbone
    studies (XCM head, region-specific), not just the fine-tune study's 5."""
    matrix = build_heatmap_matrix(long_df, SCAPE_SOURCE, "range_mae", dataset_order, row_order, model_map=model_map)
    vmin, vmax = np.nanmin(matrix.values), np.nanmax(matrix.values)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("0.85")

    # Colorbar goes below (not to the right, unlike rq5_grid) -- an
    # ax-attached right-side colorbar shrinks the heatmap's own width
    # further, and at one-column width the 7 rotated dataset tick labels are
    # already tight; horizontal below leaves the full column width for them.
    height = 4.1 * len(row_order) / 5
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_IN, height))
    im = draw_heatmap_panel(ax, matrix, dataset_order, tick_labels, row_order, cmap, vmin, vmax)
    # pad is a fraction of the axes' own height, not an absolute distance --
    # keep the actual (inches) gap roughly constant across row counts by
    # scaling it down as the figure (and so the axes) gets taller, instead
    # of the tuned-for-5-rows 0.32 blowing up into a huge empty gap at 11.
    fig.colorbar(im, ax=ax, orientation="horizontal", location="bottom", shrink=0.9, pad=1.3 / height,
                 label="MAE (lower is better)")

    fig.tight_layout()
    save_fig(fig, out_dir / filename)
    plt.close(fig)


def build_panel_b_soundscape_matched_grid(pooled_long, xg_long, ft_long, dataset_order, tick_labels,
                                           ft_row_order, out_dir):
    """The 3 backbone-matched Panel B soundscape heatmaps (fine-tune, XCM
    head frozen, region-specific -- all restricted to the same 4 fine-tuned
    backbones) stacked into one figure with a single shared color scale, so
    the identical 4x7 grid can be compared row-for-row and cell-for-cell
    across the three studies directly, not just via 3 separately-scaled
    standalone figures."""
    ft_matrix = build_heatmap_matrix(ft_long, SCAPE_SOURCE, "range_mae", dataset_order, ft_row_order,
                                      model_map=FINETUNE_NAME_TO_CANONICAL)
    xcm_matrix = build_heatmap_matrix(xg_long, SCAPE_SOURCE, "range_mae", dataset_order, ft_row_order)
    region_matrix = build_heatmap_matrix(pooled_long, SCAPE_SOURCE, "range_mae", dataset_order, ft_row_order)

    all_vals = np.concatenate([m.values.flatten() for m in (ft_matrix, xcm_matrix, region_matrix)])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("0.85")

    fig, axes = plt.subplots(3, 1, figsize=(TWO_COL_WIDTH_IN, 7.4), constrained_layout=True)
    panels = [
        (axes[0], region_matrix, "Region-specific head"),
        (axes[1], xcm_matrix, "XCM head (frozen backbone)"),
        (axes[2], ft_matrix, "XCM fine-tune head"),
    ]
    ims = [draw_heatmap_panel(ax, matrix, dataset_order, tick_labels, ft_row_order, cmap, vmin, vmax, title=title)
           for ax, matrix, title in panels]

    fig.colorbar(ims[0], ax=axes, shrink=0.85, label="MAE (lower is better)", location="right")

    # No in-figure heading -- the caption describes this figure already.
    save_fig(fig, out_dir / "rq5_panel_b_soundscape_matched_grid")
    plt.close(fig)


def macro_long_to_matrix(macro_long, dataset_order, row_order):
    """(model, dataset, value) long table from compute_macro_mae_long() ->
    model x dataset matrix, same shape build_heatmap_matrix() returns."""
    return macro_long.pivot_table(index="model", columns="dataset", values="value").reindex(
        index=row_order, columns=dataset_order)


def build_panel_b_soundscape_matched_grid_macro_mae(pooled_macro_long, xg_macro_long, ft_macro_long, dataset_order,
                                                     tick_labels, ft_row_order, out_dir):
    """Same figure as build_panel_b_soundscape_matched_grid() (3 stacked
    4x7 heatmaps, one shared color scale), but each cell is macro-MAE(1-6)
    (see compute_macro_mae_long()) instead of range_mae."""
    ft_matrix = macro_long_to_matrix(ft_macro_long, dataset_order, ft_row_order)
    xcm_matrix = macro_long_to_matrix(xg_macro_long, dataset_order, ft_row_order)
    region_matrix = macro_long_to_matrix(pooled_macro_long, dataset_order, ft_row_order)

    all_vals = np.concatenate([m.values.flatten() for m in (ft_matrix, xcm_matrix, region_matrix)])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("0.85")

    fig, axes = plt.subplots(3, 1, figsize=(TWO_COL_WIDTH_IN, 7.4), constrained_layout=True)
    panels = [
        (axes[0], region_matrix, "Region-specific head"),
        (axes[1], xcm_matrix, "XCM head (frozen backbone)"),
        (axes[2], ft_matrix, "XCM fine-tune head"),
    ]
    ims = [draw_heatmap_panel(ax, matrix, dataset_order, tick_labels, ft_row_order, cmap, vmin, vmax, title=title)
           for ax, matrix, title in panels]

    fig.colorbar(ims[0], ax=axes, shrink=0.85, label="Macro-MAE(1-6) (lower is better)", location="right")

    save_fig(fig, out_dir / "rq5_panel_b_soundscape_matched_grid_macro_mae")
    plt.close(fig)


PRED_CLIP_MAX = 14  # see build_confusion_matrix_pooled_soundscape()


def build_confusion_matrix_pooled_soundscape(study: str, xg_models, out_dir, out_name):
    """Confusion matrix for one archive/{study}/ study on the soundscape
    test, pooled across all 11 Panel-A backbones and all 7 regional datasets
    into one matrix -- the study-level counterpart to a Panel A line (which
    only shows mean +/- SD MAE, not the actual error distribution/bias).
    Used for both "XCM head" (study="XCM-Generalization") and
    "Region-specific head" (study="Pooled-Embeddings"). Reuses
    plot_confusion_matrix.py's loader/figure-drawing helpers unchanged.

    Pooling 11 architecturally different backbones (rather than one
    fine-tuned checkpoint, as plot_confusion_matrix.py's usual subjects are)
    surfaces a handful of wildly miscalibrated predictions from the weaker
    backbones -- e.g. isolated predictions up to 62, against a true range of
    0-11 -- that are under 0.1% of the ~2.5M pooled predictions (true for
    both studies) but would otherwise balloon the matrix to 60+ sparse
    columns. Predictions are clipped to [0, PRED_CLIP_MAX] (negative
    predictions floored to 0, since polyphony can't be negative; true values
    never exceed 11) so those extremely rare outliers saturate into the
    boundary bins instead of each claiming their own column -- the
    systematic bias itself (the actual finding) is entirely within this
    range and unaffected.

    Even clipped, pooling 11 backbones' worth of data lands on a denser
    grid than plot_confusion_matrix.py's usual single-backbone subjects, so
    this is built at TWO_COL_WIDTH_IN (like rq5_grid) rather than reusing
    that module's one-column build_figure_single() -- at one column the
    cell text overlapped."""
    model_dirs = [CM_ARCHIVE / study / m for m in xg_models]
    y_true, y_pred = load_soundscape_multi(model_dirs, regions=None)
    y_pred = np.clip(y_pred, 0, PRED_CLIP_MAX)

    cm, row_labels, col_labels = prune_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_IN, 1.02 * TWO_COL_WIDTH_IN * len(row_labels) / len(col_labels)))
    im, n = draw_confusion_matrix(ax, cm, row_labels, col_labels, "True polyphony degree")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=25, pad=0.02)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels([f"{v}%" for v in [0, 20, 40, 60, 80, 100]])
    cbar.outline.set_edgecolor(CM_SPINE_COLOR)
    cbar.outline.set_linewidth(0.8)

    save_fig(fig, out_dir / out_name)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq5"))
    args = parser.parse_args()

    set_rcparams()
    xg_models = mrt.discover_models(Path("archive/XCM-Generalization"))
    pooled_long = load_study("pooled", models=xg_models)
    xg_long = load_study("xcm_gen", models=xg_models)
    # models= restricts to the archive's raw folder names, i.e.
    # FINETUNE_NAME_TO_CANONICAL's keys -- excludes perch_v2's raw "perch_v2"
    # folder, which still exists on disk (see module docstring).
    ft_long = load_study("xcm_gen_ft", models=list(FINETUNE_NAME_TO_CANONICAL.keys()))

    dataset_order, tick_labels = species_ordered_datasets()
    ft_row_order = [m for m in BACKBONE_META if m in FINETUNE_NAME_TO_CANONICAL.values()]
    xg_row_order = [m for m in BACKBONE_META if m in xg_models]

    syn_matrix = build_heatmap_matrix(ft_long, SYN_SOURCE, "mae", dataset_order, ft_row_order,
                                       model_map=FINETUNE_NAME_TO_CANONICAL)
    scape_matrix = build_heatmap_matrix(ft_long, SCAPE_SOURCE, "range_mae", dataset_order, ft_row_order,
                                         model_map=FINETUNE_NAME_TO_CANONICAL)
    vmin = min(np.nanmin(syn_matrix.values), np.nanmin(scape_matrix.values))
    vmax = max(np.nanmax(syn_matrix.values), np.nanmax(scape_matrix.values))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("0.85")

    fig, axes = plt.subplots(2, 2, figsize=(TWO_COL_WIDTH_IN, 8.0), gridspec_kw={"height_ratios": [1, 1.15]},
                              constrained_layout=True)

    xs = build_panel_a(axes[0, 0], pooled_long, xg_long, SYN_SOURCE, "mae", dataset_order,
                       "Synthetic test (frozen backbone)")
    build_panel_a(axes[0, 1], pooled_long, xg_long, SCAPE_SOURCE, "range_mae", dataset_order,
                  "Soundscape (frozen backbone)")
    axes[0, 0].set_ylabel("MAE (lower is better)")
    for ax in axes[0]:
        ax.set_xticks(list(xs))
        ax.set_xticklabels(tick_labels, rotation=75, ha="right")
    axes[0, 0].legend(loc="upper left", frameon=False)

    # Shared y-scale across the two Panel A sub-panels (per the RQ5 spec).
    lo = min(axes[0, 0].get_ylim()[0], axes[0, 1].get_ylim()[0])
    hi = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
    axes[0, 0].set_ylim(lo, hi)
    axes[0, 1].set_ylim(lo, hi)

    # Titles just name the test type (matching Panel A's style); the shared
    # "fine-tuned backbone, XCM -> cross-region" qualifier belongs in the
    # caption -- at this column's width the full phrase collided with its
    # neighbor's title.
    ims = [
        draw_heatmap_panel(axes[1, 0], syn_matrix, dataset_order, tick_labels, ft_row_order, cmap, vmin, vmax,
                            title="Synthetic test"),
        draw_heatmap_panel(axes[1, 1], scape_matrix, dataset_order, tick_labels, ft_row_order, cmap, vmin, vmax,
                            title="Soundscape"),
    ]

    fig.colorbar(ims[0], ax=axes[1], shrink=0.85, label="MAE (lower is better)", location="right")

    # No in-figure heading -- the caption describes this figure already.
    save_fig(fig, args.out_dir / "rq5_grid")
    plt.close(fig)

    build_panel_a_soundscape_single(pooled_long, xg_long, ft_long, dataset_order, tick_labels, args.out_dir)
    build_panel_a_soundscape_matched(pooled_long, xg_long, ft_long, dataset_order, tick_labels, args.out_dir)

    matched_backbones = list(FINETUNE_NAME_TO_CANONICAL.values())
    pooled_macro_long = compute_macro_mae_long(CM_ARCHIVE / "Pooled-Embeddings", matched_backbones, dataset_order)
    xg_macro_long = compute_macro_mae_long(CM_ARCHIVE / "XCM-Generalization", matched_backbones, dataset_order)
    ft_macro_long = compute_macro_mae_long(CM_ARCHIVE / "XCM-Generalization-fine-tune",
                                            list(FINETUNE_NAME_TO_CANONICAL.keys()), dataset_order,
                                            model_map=FINETUNE_NAME_TO_CANONICAL)
    build_panel_a_soundscape_matched_macro_mae(pooled_macro_long, xg_macro_long, ft_macro_long, dataset_order,
                                                tick_labels, args.out_dir)
    build_heatmap_soundscape_single(ft_long, dataset_order, tick_labels, ft_row_order, args.out_dir,
                                     "rq5_panel_b_soundscape", model_map=FINETUNE_NAME_TO_CANONICAL)
    build_heatmap_soundscape_single(xg_long, dataset_order, tick_labels, xg_row_order, args.out_dir,
                                     "rq5_panel_b_soundscape_xcm_head")
    build_heatmap_soundscape_single(pooled_long, dataset_order, tick_labels, xg_row_order, args.out_dir,
                                     "rq5_panel_b_soundscape_region_specific")
    build_heatmap_soundscape_single(xg_long, dataset_order, tick_labels, ft_row_order, args.out_dir,
                                     "rq5_panel_b_soundscape_xcm_head_matched")
    build_heatmap_soundscape_single(pooled_long, dataset_order, tick_labels, ft_row_order, args.out_dir,
                                     "rq5_panel_b_soundscape_region_specific_matched")
    build_panel_b_soundscape_matched_grid(pooled_long, xg_long, ft_long, dataset_order, tick_labels,
                                           ft_row_order, args.out_dir)
    build_panel_b_soundscape_matched_grid_macro_mae(pooled_macro_long, xg_macro_long, ft_macro_long, dataset_order,
                                                     tick_labels, ft_row_order, args.out_dir)
    build_confusion_matrix_pooled_soundscape("XCM-Generalization", xg_models, args.out_dir,
                                              "rq5_confusion_matrix_xcm_head_soundscape")
    build_confusion_matrix_pooled_soundscape("Pooled-Embeddings", xg_models, args.out_dir,
                                              "rq5_confusion_matrix_region_specific_soundscape")

    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
