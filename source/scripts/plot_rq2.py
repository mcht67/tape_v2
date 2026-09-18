#!/usr/bin/env python3
"""
RQ2 -- Multi-task / prediction head.

Single panel, from archive/Spatial-Embeddings/ (5 backbones, TemporalCNN
architecture). Two different reference points are used across this file's
figures, both against the same Spatial-Embeddings config MAE:
  - build_figure() (rq2_multitask / rq2_multitask_soundscape, one column
    wide) and build_figure_combined() (rq2_multitask_combined, two columns
    wide): ΔMAE relative to the bare TC-head config alone (no auxiliary),
    paired by (model, dataset) -- so y=0 reads as "no different from
    TC-head with no auxiliary", and the TC-head config itself is left out
    of the x-axis (its own delta is trivially 0 for every backbone).
  - the dual heatmap (rq2_dual_heatmap): ΔMAE relative to archive/Pooled-
    Embeddings/ (same 5 backbones, plain pooled-embedding MLP) as the
    "pre-TC-head" baseline -- the original reference point, kept here since
    it answers "does the whole TC-head configuration still beat not having
    a TC-head at all", a different question from build_figure()'s (the
    dual heatmap shows both references side by side per cell, see below).

x = TC-head configs (+TC-head, +TC-head+frame-level polyphony,
+TC-head+polyphony classification, +TC-head+frame-level call activity,
+TC-head+both-aux; build_figure() drops +TC-head itself, see above). Per
backbone: a jittered point at each config, y = mean paired ΔMAE (paired by
dataset) across the 8 datasets, with a ±1 SD inner whisker. Per config: a
black diamond = mean ΔMAE across the 5 backbones, with a gray SD outer
whisker (heterogeneous models -> SD), visually flagged if it excludes zero.

"+TC-head+polyphony classification" is a whole-clip (not frame-level)
classification auxiliary alongside the main regression head -- evaluated at
the regression head's own output (head "reg_and_class_reg"), same as every
other config, so it stays directly comparable on the MAE-based y-axis.

Pairing is exact: baseline and config MAE are joined on (model, dataset),
which is a literal 1:1 key shared between the two studies.

A second figure (rq2_multitask_soundscape) reuses the exact same structure
on the soundscape test set instead of synthetic: metric is range_mae (no
exact ground truth there) and the dataset scope drops to the 7 region-
matched datasets (soundscape has no XCM counterpart), same restriction used
elsewhere (RQ1/RQ4) whenever synthetic and soundscape are compared.

A third figure (rq2_multitask_combined) puts the two side by side (synthetic
left, soundscape right) in one two-column-wide figure, sharing y-scale so
the ΔMAE values stay directly comparable across panels. Same TC-head-alone
baseline and omitted +TC-head column as build_figure() (see above), not the
pooled-MLP baseline -- only the dual heatmap still uses that one.

A fourth and fifth figure (rq2_backbone_bars_soundscape, rq2_backbone_bars_
combined) are a third alternative to the dual heatmap for the same pooled-
MLP-baseline question, in absolute MAE rather than ΔMAE: one panel per
backbone, x = {pooled-MLP, +TC-head, +TC-head+polyphony classification,
+TC-head+frame-level polyphony, +TC-head+frame-level call activity,
+TC-head+both-aux}, y = mean raw MAE +/-1 SD across the region-matched
datasets, with a dashed horizontal line at the pooled-MLP value so each
config's bar/dot is visually read directly against that reference rather
than a computed delta. rq2_backbone_bars_soundscape is soundscape-only (one
row of 5 panels); rq2_backbone_bars_combined stacks synthetic on top of
soundscape (2 rows x 5 panels), sharing one y-scale across all 10 panels so
backbones and sources stay directly comparable, same sharing discipline as
build_figure_combined() and RQ4's Figure 6 (source/scripts/plot_rq4.py).

A sixth figure (rq2_pooled_vs_tc_soundscape_slope) narrows Figures 4-5's
question to just the plain TC-head config (no auxiliary) vs. pooled-MLP,
soundscape only, as a single-panel slope chart: x = {Pooled MLP, +TC-head},
y = mean range_mae +/-1 SD across the 7 region-matched soundscape datasets,
one line per backbone (BACKBONE_META identity color) instead of per config
-- so it reads as "does the bare TC-head architecture change (no auxiliary
task) help or hurt each backbone on soundscape, relative to pooled-MLP".

Usage:
    complete-venv/bin/python source/scripts/plot_rq2.py [--out-dir plots/figures/rq2]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backbone_meta import BACKBONE_META
from plot_data import ALL_DATASETS, REGIONAL_DATASETS, filter_long, load_study, mean_sd_n, paired_join
from plot_style import build_encoding_legend, draw_dot_whisker, lighten_color, save_fig, set_rcparams, zero_ref_line, ONE_COL_WIDTH_IN, TWO_COL_WIDTH_IN

SOURCE = "synthetic_mixture_test"
SCAPE_SOURCE = "soundscape_test"
SOURCE_METRIC = {SOURCE: "mae", SCAPE_SOURCE: "range_mae"}
SOURCE_DATASETS = {SOURCE: ALL_DATASETS, SCAPE_SOURCE: REGIONAL_DATASETS}
SPATIAL_BACKBONES = ["AudioProtoPNet-20-BirdSet-XCL", "Bird-MAE-Huge",
                      "EfficientNet-B1-BirdSet-XCL", "NatureLMBEATs", "perch_v2_cpu"]

CONFIG_LABELS = ["+TC-head", "+TC-head+polyphony classification", "+TC-head+frame-level polyphony",
                  "+TC-head+frame-level call activity", "+TC-head+both-aux"]
# Shared with plot_rq4.py (imported from here) so a given config reads as
# the same color in every figure across both files.
CONFIG_COLOR = dict(zip(CONFIG_LABELS, ["#4c72b0", "#dd8452", "#8172b2", "#55a868", "#c44e52"]))
CONFIG_HEAD = {
    "+TC-head": "reg_reg",
    "+TC-head+polyphony classification": "reg_and_class_reg",
    "+TC-head+frame-level polyphony": "reg_and_frame_reg_reg",
    "+TC-head+frame-level call activity": "reg_and_events_reg",
    "+TC-head+both-aux": "reg_and_events_and_frame_reg_reg",
}
# Horizontal, multi-line tick labels (left-to-right, wrapped onto their own
# rows) instead of one steeply rotated line -- easier to read than text
# running bottom-to-top. Only used for the x-axis; CONFIG_LABELS itself stays
# single-line since it also doubles as the "config" column value and (via
# plot_rq4.py) a legend label elsewhere, where wrapping isn't needed.
CONFIG_TICK_LABEL = {
    "+TC-head": "TC",
    "+TC-head+polyphony classification": "TC\n+pol-class",
    "+TC-head+frame-level polyphony": "TC\n+frame-pol",
    "+TC-head+frame-level call activity": "TC\n+frame-act",
    "+TC-head+both-aux": "TC\n+both-aux",
}

FIG_HEIGHT = 3.5
LABEL_AREA_HEIGHT = 0.18 # figure fraction reserved (bottom) for tick labels + legend + footnote
# build_figure() (one column wide) needs its own taller reserved band: its
# legend is 2 columns/3 rows (SINGLE_LEGEND_NCOL below) instead of
# build_figure_combined()'s 3 columns/2 rows, since 3 columns of this
# legend's entries don't fit ONE_COL_WIDTH_IN without overflowing the canvas
# -- confirmed via check_figure_layout: at entries_ncol=3, the legend
# overflows the authored canvas by 0.34in on each side, which is exactly
# what made save_fig's bbox_inches="tight" pad the saved PNG out to fit the
# legend, leaving the (narrower, unchanged) axes visibly inset from the
# image edges instead of spanning it.
SINGLE_LABEL_AREA_HEIGHT = 0.24
SINGLE_LEGEND_NCOL = 2
# 5 wrapped 2-line x-tick labels ("TC\n+frame-pol" etc.) collide with each
# other at ONE_COL_WIDTH_IN and FIGURE_FONT_PT (confirmed via
# check_figure_layout, not visible at a quick glance -- the overlap is a few
# points at each label's edge); build_figure_combined() doesn't need this,
# its axes are roughly twice as wide (TWO_COL_WIDTH_IN, one axes per source).
SINGLE_XTICK_FONT_PT = 7
# bbox_to_anchor's y (below) is a figure-fraction measured from the CANVAS
# bottom, not from the top of the reserved label area -- tight_layout(rect=
# (0, LABEL_AREA_HEIGHT, 1, 1)) only constrains the axes+tick-labels to sit
# above LABEL_AREA_HEIGHT, it does not shrink the legend's own placement
# range. So this must stay a small value, well below LABEL_AREA_HEIGHT
# (whose reserved band sits at the BOTTOM of the canvas): a large value
# (e.g. 0.73) pushes the legend up into the axes and it ends up drawn on
# top of the plotted data instead of below it.
LEGENDS_VERTICAL_ANCHOR = 0.055 # figure fraction
LEGEND_ROW_SPACING = 0.7 # font-size units, RQ2-only override of matplotlib's legend.labelspacing default

# build_figure_combined()-only: a bit wider than TWO_COL_WIDTH_IN, since it
# drops the right panel's own y-axis label (redundant with the left panel's)
# and gives that reclaimed width back to both axes instead of leaving it as
# unused margin.
COMBINED_WIDTH_IN = TWO_COL_WIDTH_IN * 1.12
# How far each whisker's color is lightened toward white relative to its own
# dot's color (0 = same as the dot, 1 = white) -- keeps the whiskers legible
# without competing with the dots/diamonds/mean-connecting line for
# attention, while still reading as "that backbone's whisker" by hue.
COMBINED_ERRORBAR_LIGHTEN = 0.75
# The y-axis is scaled to fit only the dots (backbone means + grand means,
# see draw_config_panel()'s scale_by_dots) -- whiskers that extend past the
# resulting limits are simply clipped (matplotlib's default clip_on=True for
# errorbar/scatter artists) rather than growing the axes to fit them.


def compute_baseline(pooled_long: pd.DataFrame, source: str = SOURCE) -> pd.DataFrame:
    return filter_long(pooled_long, source=source, metric=SOURCE_METRIC[source], head="reg",
                        model=SPATIAL_BACKBONES, dataset=SOURCE_DATASETS[source])


def compute_deltas(spatial_long: pd.DataFrame, baseline_indexed: pd.Series, source: str = SOURCE) -> pd.DataFrame:
    """Per (config, model, dataset): paired delta = config MAE - baseline MAE.
    baseline_indexed can be either reference point used in this file -- the
    pooled-MLP (pre-TC-head) baseline (compute_baseline()) or the bare
    TC-head config alone (compute_tc_alone_indexed()) -- whichever the
    caller passes in; the pairing/averaging logic is identical either way."""
    frames = []
    for label, head in CONFIG_HEAD.items():
        config_df = filter_long(spatial_long, source=source, metric=SOURCE_METRIC[source], head=head,
                                 model=SPATIAL_BACKBONES, dataset=SOURCE_DATASETS[source])
        merged = paired_join(config_df, baseline_indexed, on=("model", "dataset"))
        merged["config"] = label
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def compute_tc_alone_indexed(spatial_long: pd.DataFrame, source: str = SOURCE) -> pd.Series:
    """The bare TC-head config's own MAE, indexed by (model, dataset) --
    usable as compute_deltas()'s baseline_indexed to get ΔMAE relative to
    TC-head-alone instead of the pooled-MLP baseline (build_figure() below;
    also used by compute_dual_heatmap_data())."""
    tc_alone_df = filter_long(spatial_long, source=source, metric=SOURCE_METRIC[source],
                               head=CONFIG_HEAD["+TC-head"], model=SPATIAL_BACKBONES, dataset=SOURCE_DATASETS[source])
    return tc_alone_df.set_index(["model", "dataset"])["value"]


def _backbone_jitter(backbone_order: list[str]) -> dict:
    # Slot the grand-mean diamond in among the backbones (roughly central)
    # rather than overlaying it at the config tick, so all 6 markers --
    # 5 backbones + mean -- sit at evenly spaced x-offsets with no overlap.
    half = len(backbone_order) // 2
    slot_order = backbone_order[:half] + ["__MEAN__"] + backbone_order[half:]
    return dict(zip(slot_order, np.linspace(-0.2, 0.2, len(slot_order))))


def draw_config_panel(ax, deltas: pd.DataFrame, backbone_order: list[str], jitter: dict, *,
                       omit_label=None, show_errorbars=True, errorbar_lighten=None,
                       connect_means=False, scale_by_dots=False, improved_deltas=None):
    """Draw one source's config-vs-ΔMAE panel (points/whiskers/mean diamonds
    + x-axis) onto ax. Shared by build_figure() (one source) and
    build_figure_combined() (synthetic + soundscape side by side).

    omit_label: an entry of CONFIG_LABELS to leave out of the x-axis
    columns entirely -- e.g. build_figure() passes "+TC-head" when `deltas`
    is itself computed relative to TC-head-alone (compute_tc_alone_indexed()),
    since that config's own delta is then trivially 0 for every backbone and
    not worth its own column. None (default, used by build_figure_combined())
    keeps every CONFIG_LABELS entry as its own column, unchanged.

    show_errorbars: draw the per-backbone ±1 SD inner whisker and the mean
    diamond's gray SD outer whisker (default True).

    errorbar_lighten: None (default, build_figure()'s case) draws each
    whisker in its dot's own color (identity color for a backbone, "0.4" gray
    for the mean diamond) at full strength; a 0-1 amount instead lightens
    each whisker toward white by that much off its own dot's color (via
    plot_style.lighten_color()) -- build_figure_combined()'s case, so the
    whiskers stay identifiable by hue without competing with the dots.

    connect_means: also draw a black line through the per-config grand-mean
    diamonds, the same way each backbone's own points are already connected
    (default False, build_figure_combined()'s case).

    scale_by_dots: set the y-axis limits from only the plotted dots
    (backbone means + grand means), ignoring the whiskers' extent, with a
    15% margin -- whiskers that reach past those limits are then simply
    clipped (default False, matplotlib's own autoscale over every artist
    including whiskers).

    improved_deltas: optional DataFrame in the same (config, model, delta)
    shape as `deltas` (e.g. computed against the pooled-MLP baseline via
    compute_baseline()/compute_deltas(), independent of whatever baseline
    `deltas` itself is relative to) -- any per-backbone dot whose mean delta
    here is negative (an improvement against that reference) is drawn with
    an added red ring, regardless of what `deltas` plots it against. None
    (default) draws no rings."""
    per_bm = mean_sd_n(deltas, ["config", "model"], value_col="delta")
    per_bm_improved = mean_sd_n(improved_deltas, ["config", "model"], value_col="delta") \
        if improved_deltas is not None else None

    zero_ref_line(ax)

    config_labels = [c for c in CONFIG_LABELS if c != omit_label] if omit_label else CONFIG_LABELS

    # Connecting line per backbone (its mean ΔMAE across the configs), drawn
    # first so the dot+whisker points sit on top of it.
    for model in backbone_order:
        color = BACKBONE_META[model]["identity_color"]
        xs = [cx + jitter[model] for cx in range(len(config_labels))]
        means = [per_bm.loc[(label, model), "mean"] for label in config_labels]
        ax.plot(xs, means, color=color, linewidth=1.5, alpha=0.7, zorder=1)

    dot_values = []
    grand_means = []
    for cx, label in enumerate(config_labels):
        backbone_means = []
        for model in backbone_order:
            mean, sd, n = per_bm.loc[(label, model)]
            color = BACKBONE_META[model]["identity_color"]
            ecolor = lighten_color(color, errorbar_lighten) if errorbar_lighten is not None else color
            draw_dot_whisker(ax, cx + jitter[model], mean, sd if show_errorbars else None,
                              facecolor=color, edgecolor=color, marker="o", s=20,
                              ecolor=ecolor)
            if per_bm_improved is not None and per_bm_improved.loc[(label, model), "mean"] < 0:
                ax.scatter([cx + jitter[model]], [mean], marker="o", facecolor="none",
                           edgecolor="red", linewidth=1.3, s=50, zorder=3.5)
            backbone_means.append(mean)
            dot_values.append(mean)

        grand_mean = float(np.mean(backbone_means))
        grand_means.append(grand_mean)
        dot_values.append(grand_mean)
        mx = cx + jitter["__MEAN__"]
        if show_errorbars:
            grand_sd = float(np.std(backbone_means, ddof=1))
            mean_ecolor = lighten_color("0.4", errorbar_lighten) if errorbar_lighten is not None else "0.4"
            ax.errorbar(mx, grand_mean, yerr=grand_sd, fmt="none", ecolor=mean_ecolor, capsize=5,
                        linewidth=1.2, zorder=4)
        ax.scatter([mx], [grand_mean], marker="D", facecolor="black", edgecolor="0.3",
                   linewidth=1.2, s=50, zorder=5)

    if connect_means:
        mean_xs = [cx + jitter["__MEAN__"] for cx in range(len(config_labels))]
        ax.plot(mean_xs, grand_means, color="black", linewidth=1.5, zorder=4)

    if scale_by_dots:
        lo, hi = min(dot_values), max(dot_values)
        pad = (hi - lo) * 0.15 if hi > lo else 0.01
        ax.set_ylim(lo - pad, hi + pad)

    # Omitted-column configs drop the "TC\n" tick-label prefix (e.g. "TC\n
    # +pol-class" -> "+pol-class") -- redundant once TC-head-alone isn't a
    # column at all, and shorter labels ease the crowding at ONE_COL_WIDTH_IN
    # (see SINGLE_XTICK_FONT_PT).
    tick_labels = [CONFIG_TICK_LABEL[label].removeprefix("TC\n") if omit_label else CONFIG_TICK_LABEL[label]
                   for label in config_labels]
    ax.set_xticks(range(len(config_labels)))
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax.grid(axis="y", linewidth=1.0, alpha=0.4)


def _legend_entries(backbone_order: list[str]) -> list:
    entries = [(BACKBONE_META[m]["display"], dict(marker="o", markerfacecolor=BACKBONE_META[m]["identity_color"],
               markeredgecolor=BACKBONE_META[m]["identity_color"], color=BACKBONE_META[m]["identity_color"]))
               for m in backbone_order]
    entries.append(("Mean across 5 backbones", dict(marker="D", markerfacecolor="black", markeredgecolor="0.3", color="0.3")))
    return entries


def _baseline_footnote(fig):
    # What "own baseline" means (the y-axis label only names the quantity,
    # not the reference point) -- kept as a small footnote rather than back
    # in the axis label, matching how a caption footnote would read.
    # fig.text(0.5, 0.01, "Baseline: plain pooled-embedding MLP head, without the TC-head.",
    #           ha="center", va="bottom", fontsize=7, style="italic", color="0.3")
    pass


# The config build_figure()'s `deltas` is computed relative to (via
# compute_tc_alone_indexed() in main(), not the pooled-MLP baseline) --
# its own delta is then trivially 0 for every backbone, so it's left out of
# the x-axis columns entirely rather than shown as a column of zeros.
SINGLE_OMITTED_LABEL = "+TC-head"


def build_figure(deltas: pd.DataFrame, out_dir: Path, filename: str = "rq2_multitask"):
    """deltas must already be relative to TC-head-alone (compute_tc_alone_
    indexed(), not compute_baseline()'s pooled-MLP baseline) -- so the y=0
    line here reads directly as "TC-head alone", and every point is "this
    auxiliary vs. TC-head with no auxiliary", not vs. the pre-TC-head
    baseline (that comparison is build_figure_combined()'s / the dual
    heatmap's, both still on the pooled-MLP baseline)."""
    backbone_order = [m for m in BACKBONE_META if m in SPATIAL_BACKBONES]
    jitter = _backbone_jitter(backbone_order)

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_IN, FIG_HEIGHT))
    draw_config_panel(ax, deltas, backbone_order, jitter, omit_label=SINGLE_OMITTED_LABEL)
    # Smaller than the shared FIGURE_FONT_PT -- see SINGLE_XTICK_FONT_PT's
    # comment: at this column width, 4 wrapped 2-line labels still collide
    # with each other at the default size.
    ax.tick_params(axis="x", labelsize=SINGLE_XTICK_FONT_PT)
    # Unrotated, above the axes (like a small title) instead of matplotlib's
    # default rotated side label -- reclaims the rotated label's own width
    # for the axes (the actual ask behind "make the plot a little wider"),
    # and sits directly over the y-tick numbers it describes.
    ax.set_ylabel("")
    # "vs. TC-head" (not "vs. TC-head alone") -- confirmed via check_figure_
    # layout that the longer wording overflows the canvas by ~0.02in at this
    # column width.
    ax.text(0.0, 1.015, r"$\Delta$MAE vs. TC-head (negative = improvement)", transform=ax.transAxes,
            ha="left", va="bottom")

    # bbox_to_anchor's y is figure-fraction, not axes-relative -- keep it
    # positive (a negative value places the legend's anchor point below the
    # canvas edge by construction) and size tight_layout's bottom margin to
    # the legend's actual rendered height.
    fig.tight_layout(rect=(0, SINGLE_LABEL_AREA_HEIGHT, 1, 1))
    build_encoding_legend(fig, [(None, _legend_entries(backbone_order))], bbox_to_anchor=(0.5, LEGENDS_VERTICAL_ANCHOR),
                          entries_ncol=SINGLE_LEGEND_NCOL, labelspacing=LEGEND_ROW_SPACING)
    _baseline_footnote(fig)
    save_fig(fig, out_dir / filename)
    plt.close(fig)


def build_figure_combined(source_deltas: dict[str, pd.DataFrame], pooled_deltas: dict[str, pd.DataFrame],
                           out_dir: Path, filename: str = "rq2_multitask_combined"):
    """Synthetic (left) and soundscape (right) side by side -- same
    per-source panel as build_figure(), but each panel keeps its own y-scale,
    fit to only its dots rather than matplotlib's default of autoscaling to
    every artist including the whiskers (soundscape's whiskers are much
    wider than synthetic's, so autoscaling by them would flatten the
    synthetic panel's dots down to a sliver); the per-config grand means are
    additionally connected by a black line; the whiskers are drawn in each
    dot's own color, heavily lightened, rather than build_figure()'s
    full-strength per-backbone-color/"0.4"; and any per-backbone dot that
    improves over that backbone's pooled-MLP baseline (pre-TC-head) gets a
    red ring, independent of the TC-head-alone reference the dot's own
    position is plotted against (see draw_config_panel()'s scale_by_dots/
    connect_means/errorbar_lighten/improved_deltas). Slightly wider than
    TWO_COL_WIDTH_IN -- there's no per-panel y-axis label competing for width
    on the right panel (dropped, redundant with the left panel's), so that
    reclaimed width goes to widening both axes instead.

    source_deltas must already be relative to TC-head-alone (compute_tc_
    alone_indexed(), same as build_figure()'s `deltas` argument), not the
    pooled-MLP baseline -- so the +TC-head column is omitted here too, same
    as build_figure(). pooled_deltas is the pooled-MLP-baseline-relative
    counterpart (compute_baseline()/compute_deltas()), used only to decide
    the red rings above."""
    backbone_order = [m for m in BACKBONE_META if m in SPATIAL_BACKBONES]
    jitter = _backbone_jitter(backbone_order)

    fig, axes = plt.subplots(1, 2, figsize=(COMBINED_WIDTH_IN, FIG_HEIGHT))
    draw_config_panel(axes[0], source_deltas["synthetic"], backbone_order, jitter,
                      omit_label=SINGLE_OMITTED_LABEL, show_errorbars=True,
                      errorbar_lighten=COMBINED_ERRORBAR_LIGHTEN, connect_means=True, scale_by_dots=True,
                      improved_deltas=pooled_deltas["synthetic"])
    draw_config_panel(axes[1], source_deltas["soundscape"], backbone_order, jitter,
                      omit_label=SINGLE_OMITTED_LABEL, show_errorbars=True,
                      errorbar_lighten=COMBINED_ERRORBAR_LIGHTEN, connect_means=True, scale_by_dots=True,
                      improved_deltas=pooled_deltas["soundscape"])
    axes[0].set_title("Synthetic")
    axes[1].set_title("Soundscape")
    axes[0].set_ylabel(r"$\Delta$MAE vs. TC-head (negative = improvement)")

    fig.tight_layout(rect=(0, LABEL_AREA_HEIGHT, 1, 1))
    build_encoding_legend(fig, [(None, _legend_entries(backbone_order))], bbox_to_anchor=(0.5, LEGENDS_VERTICAL_ANCHOR),
                          entries_ncol=3, labelspacing=LEGEND_ROW_SPACING)
    _baseline_footnote(fig)
    save_fig(fig, out_dir / filename)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Alternative figure: dual-encoded (diagonally-split-cell) heatmap -- per
# backbone x config cell, one triangle shows ΔMAE vs. the bare TC-head (no
# auxiliary) and the other shows ΔMAE vs. the original pooled-MLP baseline,
# so both "does this auxiliary help over TC-head alone" and "does the whole
# configuration still beat the pre-TC-head baseline" are visible in one grid
# instead of two separate comparisons. Not a replacement for build_figure()/
# build_figure_combined() above -- an alternative encoding of the same
# underlying data, written alongside them.
# ---------------------------------------------------------------------------

# Explicit alphabetical order (not BACKBONE_META's domain-grouped order used
# elsewhere) -- this figure's own row order.
DUAL_ROW_ORDER = ["AudioProtoPNet-20-BirdSet-XCL", "Bird-MAE-Huge", "EfficientNet-B1-BirdSet-XCL",
                   "NatureLMBEATs", "perch_v2_cpu"]
DUAL_COL_LABEL = {
    "+TC-head": "TC-alone",
    "+TC-head+polyphony classification": "+event-logits",
    "+TC-head+frame-level polyphony": "+frame-poly",
    "+TC-head+frame-level call activity": "+frame-act",
    "+TC-head+both-aux": "+both",
}
DUAL_FIG_HEIGHT = 3.4
# 10 columns' worth of tick labels (5 configs x 2 panels) at TWO_COL_WIDTH_IN
# collide at shallower rotation/larger font -- confirmed via
# check_figure_layout, this combination is the first clean one found.
DUAL_XTICK_FONT_PT = 6.5
DUAL_XTICK_ROTATION = 60
DUAL_CELL_FONT_PT = 6.5


def compute_dual_heatmap_data(pooled_long: pd.DataFrame, spatial_long: pd.DataFrame,
                               source: str = SOURCE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (backbone, config): mean paired ΔMAE against two references -- the
    bare TC-head (no auxiliary) and the pooled-MLP baseline (pre-TC-head) --
    for build_dual_heatmap_figure()'s diagonally-split cells. Paired by
    (model, dataset) throughout, same discipline as compute_deltas(); "vs.
    TC-head" comes out exactly 0 for the "+TC-head" config itself (compared
    to itself), matching the spec's "always neutral for the TC-alone column"
    without special-casing it.

    Returns (vs_tc_head, vs_baseline), each a model x CONFIG_LABELS
    DataFrame of mean deltas."""
    baseline = compute_baseline(pooled_long, source)
    baseline_indexed = baseline.set_index(["model", "dataset"])["value"]
    tc_alone_indexed = compute_tc_alone_indexed(spatial_long, source)

    vs_tc, vs_baseline = {}, {}
    for label, head in CONFIG_HEAD.items():
        config_df = filter_long(spatial_long, source=source, metric=SOURCE_METRIC[source], head=head,
                                 model=SPATIAL_BACKBONES, dataset=SOURCE_DATASETS[source])
        vs_tc[label] = paired_join(config_df, tc_alone_indexed, on=("model", "dataset")).groupby("model")["delta"].mean()
        vs_baseline[label] = paired_join(config_df, baseline_indexed, on=("model", "dataset")).groupby("model")["delta"].mean()

    return pd.DataFrame(vs_tc), pd.DataFrame(vs_baseline)


def _legible_text_color(cmap, norm, value: float) -> str:
    """Black or white, whichever reads legibly against cmap(norm(value))'s
    background -- standard relative-luminance threshold."""
    r, g, b = cmap(norm(value))[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.5 else "white"


def draw_dual_triangle_panel(ax, vs_tc: pd.DataFrame, vs_baseline: pd.DataFrame, row_order: list[str],
                              col_order: list[str], cmap, norm):
    """One panel: an n_backbones x n_configs grid of unit cells, each split
    into an upper-left triangle (vs_tc's value) and a lower-right triangle
    (vs_baseline's value), filled from the shared diverging cmap/norm and
    annotated with its own ΔMAE. Row 0 (row_order[0]) is drawn at the top."""
    n_rows, n_cols = len(row_order), len(col_order)

    for r, model in enumerate(row_order):
        y0, y1 = n_rows - r - 1, n_rows - r
        for c, label in enumerate(col_order):
            x0, x1 = c, c + 1
            v_tc, v_bl = vs_tc.loc[model, label], vs_baseline.loc[model, label]

            ax.add_patch(plt.Polygon([(x0, y1), (x1, y1), (x0, y0)], facecolor=cmap(norm(v_tc)),
                                      edgecolor="white", linewidth=0.6, zorder=1))
            ax.add_patch(plt.Polygon([(x1, y1), (x1, y0), (x0, y0)], facecolor=cmap(norm(v_bl)),
                                      edgecolor="white", linewidth=0.6, zorder=1))

            ax.text(x0 + (x1 - x0) / 3, y0 + 2 * (y1 - y0) / 3, f"{v_tc:+.2f}", ha="center", va="center",
                    fontsize=DUAL_CELL_FONT_PT, color=_legible_text_color(cmap, norm, v_tc), zorder=2)
            ax.text(x0 + 2 * (x1 - x0) / 3, y0 + (y1 - y0) / 3, f"{v_bl:+.2f}", ha="center", va="center",
                    fontsize=DUAL_CELL_FONT_PT, color=_legible_text_color(cmap, norm, v_bl), zorder=2)

    for c in range(n_cols + 1):
        ax.plot([c, c], [0, n_rows], color="0.3", linewidth=0.8, zorder=3)
    for r in range(n_rows + 1):
        ax.plot([0, n_cols], [r, r], color="0.3", linewidth=0.8, zorder=3)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([c + 0.5 for c in range(n_cols)])
    ax.set_xticklabels([DUAL_COL_LABEL[label] for label in col_order], rotation=DUAL_XTICK_ROTATION, ha="right",
                        rotation_mode="anchor", fontsize=DUAL_XTICK_FONT_PT)
    ax.set_yticks([n_rows - r - 0.5 for r in range(n_rows)])
    ax.set_yticklabels([BACKBONE_META[m]["display"] for m in row_order])
    ax.set_aspect("equal")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_dual_heatmap_figure(pooled_long: pd.DataFrame, spatial_long: pd.DataFrame, out_dir: Path,
                               filename: str = "rq2_dual_heatmap"):
    """Alternative to build_figure_combined(): same underlying MAE values,
    encoded as a diagonally-split-cell heatmap (backbone x config) instead of
    a dot+whisker line chart, so both references (TC-head alone, pooled-MLP
    baseline) are visible per cell instead of only one at a time."""
    vs_tc_syn, vs_bl_syn = compute_dual_heatmap_data(pooled_long, spatial_long, SOURCE)
    vs_tc_sc, vs_bl_sc = compute_dual_heatmap_data(pooled_long, spatial_long, SCAPE_SOURCE)

    # Single shared, zero-centered scale across both triangle types and both
    # panels, per spec, so color intensity is directly comparable anywhere
    # in the figure.
    vmax = float(np.nanmax(np.abs(np.concatenate(
        [d.values.flatten() for d in (vs_tc_syn, vs_bl_syn, vs_tc_sc, vs_bl_sc)]))))
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL_WIDTH_IN, DUAL_FIG_HEIGHT))
    draw_dual_triangle_panel(axes[0], vs_tc_syn, vs_bl_syn, DUAL_ROW_ORDER, CONFIG_LABELS, cmap, norm)
    draw_dual_triangle_panel(axes[1], vs_tc_sc, vs_bl_sc, DUAL_ROW_ORDER, CONFIG_LABELS, cmap, norm)
    axes[0].set_title("Synthetic")
    axes[1].set_title("Soundscape")

    fig.tight_layout(rect=(0, 0.09, 0.88, 1))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.8, pad=0.02, fraction=0.05)
    cbar.set_label(r"$\Delta$MAE")

    fig.text(0.5, 0.015, "Upper-left triangle: vs. TC-head alone.  Lower-right triangle: vs. pooled-MLP baseline.",
              ha="center", va="bottom", fontsize=7)

    save_fig(fig, out_dir / filename)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures 4-5 -- per-backbone absolute-MAE bars vs. pooled-MLP baseline
# ---------------------------------------------------------------------------

BAR_BASELINE_COLOR = "0.5"


def _backbone_bar_stats(pooled_long: pd.DataFrame, spatial_long: pd.DataFrame, model: str, source: str):
    """Per x-axis category for one backbone: (mean, sd, color) of raw MAE
    across that source's region-matched datasets -- pooled-MLP baseline
    first, then the 5 TC-head configs in CONFIG_LABELS order. Unlike
    compute_baseline()/compute_deltas() (used by the delta-based figures
    above), no join against a second study is needed here: each category
    reads its own model's MAE directly, so plain groupby-free mean/std over
    the already-filtered rows suffices."""
    metric = SOURCE_METRIC[source]
    dataset = SOURCE_DATASETS[source]

    baseline_df = filter_long(pooled_long, source=source, metric=metric, head="reg", model=model, dataset=dataset)
    stats = [(baseline_df["value"].mean(), baseline_df["value"].std(ddof=1), BAR_BASELINE_COLOR)]
    for label in CONFIG_LABELS:
        config_df = filter_long(spatial_long, source=source, metric=metric, head=CONFIG_HEAD[label],
                                 model=model, dataset=dataset)
        stats.append((config_df["value"].mean(), config_df["value"].std(ddof=1), CONFIG_COLOR[label]))
    return stats


def draw_backbone_bar_panel(ax, pooled_long: pd.DataFrame, spatial_long: pd.DataFrame, model: str, source: str, *,
                             title: str | None):
    """One panel: dot+whisker at each of the 6 categories (pooled-MLP, then
    the 5 TC-head configs, left to right) for a single backbone, plus a
    dashed reference line at the pooled-MLP mean so every config's dot is
    read directly against it instead of requiring a remembered baseline
    value. No per-category x-tick text -- 6 labels don't fit legibly at this
    panel width (5 panels/row); category identity is carried by color +
    the shared legend (_config_legend_handles()) instead, same left-to-right
    order as the dots themselves."""
    stats = _backbone_bar_stats(pooled_long, spatial_long, model, source)
    for x, (mean, sd, color) in enumerate(stats):
        draw_dot_whisker(ax, x, mean, sd, facecolor=color, edgecolor=color, marker="o", s=45, zorder=2)
    ax.axhline(stats[0][0], color=BAR_BASELINE_COLOR, linewidth=0.8, linestyle="--", zorder=1)

    ax.set_xlim(-0.6, len(stats) - 0.4)
    ax.set_xticks([])
    if title is not None:
        ax.set_title(title, fontsize=SINGLE_XTICK_FONT_PT)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)


def _config_legend_handles():
    """Proxy handles for the shared bottom legend: pooled-MLP baseline
    (gray) + the 5 TC-head configs (CONFIG_COLOR), same left-to-right order
    as each panel's own dots."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor=BAR_BASELINE_COLOR,
                       markeredgecolor=BAR_BASELINE_COLOR, markersize=7, label="Pooled MLP")]
    handles += [Line2D([], [], marker="o", linestyle="none", markerfacecolor=CONFIG_COLOR[label],
                        markeredgecolor=CONFIG_COLOR[label], markersize=7, label=label) for label in CONFIG_LABELS]
    return handles


def build_backbone_bars(pooled_long: pd.DataFrame, spatial_long: pd.DataFrame, out_dir: Path, *,
                         source: str = SCAPE_SOURCE, filename: str = "rq2_backbone_bars_soundscape"):
    """Soundscape-only: one row of 5 panels, one per backbone. No in-figure
    heading -- the caption describes this figure already."""
    fig, axes = plt.subplots(1, len(SPATIAL_BACKBONES), figsize=(TWO_COL_WIDTH_IN, 3.2), sharey=True)
    for ax, model in zip(axes, SPATIAL_BACKBONES):
        draw_backbone_bar_panel(ax, pooled_long, spatial_long, model, source, title=BACKBONE_META[model]["display"])
    axes[0].set_ylabel("MAE (lower is better)")

    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.legend(handles=_config_legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=3,
               frameon=False, title="Config")
    fig.text(0.5, 0.01, "Dashed line: pooled-MLP baseline (no TC-head).", ha="center", va="bottom", fontsize=7)
    save_fig(fig, out_dir / filename)
    plt.close(fig)


def build_backbone_bars_combined(pooled_long: pd.DataFrame, spatial_long: pd.DataFrame, out_dir: Path,
                                  filename: str = "rq2_backbone_bars_combined"):
    """Synthetic + soundscape: 2 rows (synthetic on top) x 5 backbone
    columns, sharing one y-scale across all 10 panels -- see docstring."""
    fig, axes = plt.subplots(2, len(SPATIAL_BACKBONES), figsize=(TWO_COL_WIDTH_IN, 5.8), sharey=True)
    for row, source in enumerate([SOURCE, SCAPE_SOURCE]):
        for col, model in enumerate(SPATIAL_BACKBONES):
            title = BACKBONE_META[model]["display"] if row == 0 else None
            draw_backbone_bar_panel(axes[row, col], pooled_long, spatial_long, model, source, title=title)
    axes[0, 0].set_ylabel("Synthetic MAE")
    axes[1, 0].set_ylabel("Soundscape MAE")

    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(handles=_config_legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=3,
               frameon=False, title="Config")
    fig.text(0.5, 0.005, "Dashed line: pooled-MLP baseline (no TC-head).", ha="center", va="bottom", fontsize=7)
    save_fig(fig, out_dir / filename)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 -- plain TC-head vs. pooled-MLP slope chart, soundscape only
# ---------------------------------------------------------------------------

def build_pooled_vs_tc_slope(pooled_long: pd.DataFrame, spatial_long: pd.DataFrame, out_dir: Path, *,
                              source: str = SCAPE_SOURCE, filename: str = "rq2_pooled_vs_tc_soundscape_slope"):
    """No in-figure heading -- the caption describes this figure already.
    Same slope-chart structure as build_fig2() in plot_rq4.py, but x = the
    two heads being compared (pooled-MLP, plain TC-head) instead of the two
    sources, and one line per backbone (identity color) instead of one line
    per config."""
    metric = SOURCE_METRIC[source]
    dataset = SOURCE_DATASETS[source]

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_IN, 4.0))
    for model in SPATIAL_BACKBONES:
        color = BACKBONE_META[model]["identity_color"]
        pooled_df = filter_long(pooled_long, source=source, metric=metric, head="reg", model=model, dataset=dataset)
        tc_df = filter_long(spatial_long, source=source, metric=metric, head=CONFIG_HEAD["+TC-head"],
                             model=model, dataset=dataset)
        means = [pooled_df["value"].mean(), tc_df["value"].mean()]
        sds = [pooled_df["value"].std(ddof=1), tc_df["value"].std(ddof=1)]

        xs = [0, 1]
        ax.plot(xs, means, color=color, marker="o", markersize=9, linewidth=2,
                 label=BACKBONE_META[model]["display"], zorder=2)
        for x, mean, sd in zip(xs, means, sds):
            ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor=color, capsize=5, zorder=1)

    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pooled MLP", "+TC-head"])
    ax.set_ylabel("MAE (lower is better)")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, title="Backbone")

    fig.tight_layout()
    save_fig(fig, out_dir / filename)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq2"))
    args = parser.parse_args()

    set_rcparams()
    pooled_long = load_study("pooled", models=SPATIAL_BACKBONES)
    spatial_long = load_study("spatial", models=SPATIAL_BACKBONES)

    source_deltas = {}
    pooled_deltas = {}
    for source, source_key, filename in [(SOURCE, "synthetic", "rq2_multitask"),
                                          (SCAPE_SOURCE, "soundscape", "rq2_multitask_soundscape")]:
        # vs. TC-head alone -- build_figure() and build_figure_combined(); see their docstrings.
        tc_alone_indexed = compute_tc_alone_indexed(spatial_long, source)
        tc_deltas = compute_deltas(spatial_long, tc_alone_indexed, source)
        source_deltas[source_key] = tc_deltas
        build_figure(tc_deltas, args.out_dir, filename)

        # vs. pooled-MLP baseline -- build_figure_combined()'s red-ring "improves
        # over pooled baseline" flag only, see its docstring.
        baseline = compute_baseline(pooled_long, source)
        baseline_indexed = baseline.set_index(["model", "dataset"])["value"]
        pooled_deltas[source_key] = compute_deltas(spatial_long, baseline_indexed, source)

    build_figure_combined(source_deltas, pooled_deltas, args.out_dir)
    build_dual_heatmap_figure(pooled_long, spatial_long, args.out_dir)
    build_backbone_bars(pooled_long, spatial_long, args.out_dir)
    build_backbone_bars_combined(pooled_long, spatial_long, args.out_dir)
    build_pooled_vs_tc_slope(pooled_long, spatial_long, args.out_dir)

    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
