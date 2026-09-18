#!/usr/bin/env python3
"""
RQ1 -- Backbone comparison, synthetic vs. soundscape.

Two side-by-side dot+whisker panels (synthetic left, soundscape right),
sharing y-axis scale and backbone x-order, from archive/Pooled-Embeddings/,
regression formulation only. Per backbone: mean +/- SD MAE across the 7
region-matched datasets (UHH/HSN/PER/NES/POW/SSW/SNE) -- XCM is excluded
throughout since it has no soundscape counterpart to compare against (same
scope restriction as RQ4's synthetic-vs-soundscape figures). Soundscape has
no exact polyphony-degree ground truth, so its "MAE" is actually range_mae
(distance to the nearest edge of the plausible [min, max] range) -- both are
plotted on the same axis and captioned as "MAE (lower is better)" for visual
comparability, but a caption discussing exact numbers should name the
soundscape one range_mae. Point color = domain, shape = architecture,
fill = paradigm.

build_figure_macro_mae() is a further variant of build_figure() itself:
same two-panel synthetic-vs-soundscape layout and full encoding, but the
soundscape panel plots macro-MAE(1-6) (unweighted mean of per-true-level
MAE, levels 1-6, on the unambiguous-label subset -- see
compute_soundscape_macro_mae_stats()) instead of range_mae, addressing the
same level-1-dominance concern documented for compute_soundscape_per_level_
accuracy.py's macro-accuracy columns. Not a drop-in swap of the soundscape
panel alone -- see that function's docstring for why the two panels no
longer share a y-axis. build_figure_macro_mae_single_panel() is its
single-panel jittered-overlay counterpart, mirroring build_figure_single_
panel() -- necessarily still shares one y-axis between the two series, per
that function's own docstring caveat.

A single-panel alternative (build_figure_single_panel) drops the
architecture/paradigm encoding and instead overlays both test sources at
each backbone's x-position (circle = synthetic, square = soundscape,
jittered apart), so their gap is visible directly rather than split across
two panels.

Further variants split by formulation instead of test source, synthetic
test, all 8 datasets: build_figure_reg_vs_class() reuses the full
architecture/domain/paradigm encoding and two-panel layout (regression vs.
classification); build_figure_reg_vs_class_single() is its single-panel,
jittered-overlay counterpart (mirroring build_figure_single_panel(), marker
shape = formulation instead of architecture); build_figure_synthetic_reg()
drops the comparison entirely (regression only, one panel).

Also emits two companion LaTeX top-5 tables (synthetic MAE across all 8
datasets, and soundscape range_mae across the 7 region-matched datasets) and
the full supplementary table (reusing make_result_tables.py's existing table
builder unchanged) -- the top-5 tables' ranking formulation is independent of
the synthetic-vs-soundscape scope used by the main figure above.

Usage:
    complete-venv/bin/python source/scripts/plot_rq1.py [--out-dir plots/figures/rq1]
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import BACKBONE_META, DOMAIN_COLOR, MARKER_BY_ARCH, ordered_backbones
from plot_data import ALL_DATASETS, REGIONAL_DATASETS, filter_long, load_study, mean_sd_n
from plot_style import build_encoding_legend, draw_dot_whisker, lighten_color, save_fig, set_rcparams, TWO_COL_WIDTH_IN

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = REPO_ROOT / "archive"
MACRO_MAE_LEVELS = range(1, 7)  # levels 1-6 -- excludes level 0 (presence/absence,
# not a polyphony-count discrimination question) and 7/8 (n<130 pooled across all
# regions, per compute_soundscape_per_level_accuracy.py's caution)

SOURCE = "synthetic_mixture_test"
SCAPE_SOURCE = "soundscape_test"
SOURCE_PANEL_TITLE = {"synthetic": "Synthetic", "soundscape": "Soundscape"}
SOURCE_KEY_BY_SOURCE = {SOURCE: "synthetic", SCAPE_SOURCE: "soundscape"}
HEAD_PANEL_TITLE = {"reg": "Regression", "class": "Classification"}
MARKER_BY_SOURCE = {"synthetic": "o", "soundscape": "s"}
SOURCE_JITTER = {"synthetic": -0.15, "soundscape": 0.15}
MARKER_BY_HEAD = {"reg": "o", "class": "s"}
HEAD_JITTER = {"reg": -0.15, "class": 0.15}
SINGLE_PANEL_HEIGHT = 4.0
# How far a whisker's color is lightened toward white off its own dot's
# color (0 = same as the dot, 1 = white) -- build_figure() and
# build_figure_single_panel() only (rq1_backbone_comparison[_single_panel]),
# so the whiskers read as slightly secondary to the dots without losing
# their domain-color identity.
ERRORBAR_LIGHTEN = 0.35


def compute_source_stats(long_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Regression-only, per-model mean/SD/n across the 7 region-matched
    datasets, one stats table per panel of build_figure() ({"synthetic",
    "soundscape"} -> DataFrame indexed by model)."""
    syn = filter_long(long_df, source=SOURCE, metric="mae", head="reg", dataset=REGIONAL_DATASETS)
    scape = filter_long(long_df, source=SCAPE_SOURCE, metric="range_mae", head="reg", dataset=REGIONAL_DATASETS)
    return {
        "synthetic": mean_sd_n(syn, ["model"]),
        "soundscape": mean_sd_n(scape, ["model"]),
    }


def _region_macro_mae_1_6(model_dir: Path, region: str) -> tuple[float, int]:
    """Per-(model, region) macro-MAE(1-6): unweighted mean of per-level MAE
    (round(pred) vs. true level) over whichever of levels 1-6 have at least
    one unambiguous-label (min_polyphony == max_polyphony) clip in this
    region. Some regions have essentially no support at the higher levels
    (e.g. HSN/NES/UHH have zero unambiguous clips at level 6 -- see
    compute_soundscape_macro_mae_stats()'s docstring), so a region's average
    can silently be taken over fewer than 6 levels; the count of levels
    actually used is returned so that can be surfaced rather than hidden.
    Returns (nan, 0) if the region file is missing or has zero unambiguous
    clips at every level in MACRO_MAE_LEVELS."""
    path = model_dir / "scape_eval_results" / f"{region}_reg_scape_test_results.pkl"
    if not path.exists():
        return float("nan"), 0
    df = pickle.load(open(path, "rb"))
    unamb = df[df["y_true.min_polyphony"] == df["y_true.max_polyphony"]].copy()
    unamb["true_level"] = unamb["y_true.min_polyphony"].round().astype(int)
    unamb["pred_level"] = unamb["predictions.polyphony_reg"].round().astype(int)

    level_maes = []
    for level in MACRO_MAE_LEVELS:
        sub = unamb[unamb["true_level"] == level]
        if len(sub):
            level_maes.append(float((sub["true_level"] - sub["pred_level"]).abs().mean()))
    if not level_maes:
        return float("nan"), 0
    return float(np.mean(level_maes)), len(level_maes)


def compute_soundscape_macro_mae_stats(models: list[str]) -> pd.DataFrame:
    """Soundscape macro-MAE(1-6) per backbone: _region_macro_mae_1_6(),
    computed separately per region (UHH/HSN/PER/NES/POW/SSW/SNE) then
    reduced to mean/SD/n across regions -- mirrors compute_source_stats()'s
    output shape (DataFrame indexed by model, columns mean/std/count) so it
    slots into draw_source_panel() unchanged.

    Unlike range_mae (compute_source_stats()'s soundscape column), this is
    an unweighted average across true polyphony levels within each region
    first, not a single pooled distance-to-range number -- see
    plot_rq1.py's compute_soundscape_per_level_accuracy.py sibling script
    for why range_mae's aggregate can mask genuinely poor per-level
    discrimination. CAVEAT: several regions have zero unambiguous clips at
    the higher levels within MACRO_MAE_LEVELS (empirically: HSN has none at
    levels 4-6, NES none at level 6, UHH none at level 6 -- see that
    script's printed n-per-level table, which is pooled across all 7
    regions and already thin at level 6 there), so those regions'
    macro-MAE is silently averaged over fewer than 6 levels rather than
    being NaN'd out entirely. This makes the per-region numbers feeding
    this plot's whiskers not perfectly like-for-like across regions -- a
    caveat worth restating in any caption/text that cites exact whisker
    widths."""
    archive_dir = ARCHIVE / "Pooled-Embeddings"
    rows = []
    for model in models:
        model_dir = archive_dir / model
        for region in REGIONAL_DATASETS:
            macro_mae, n_levels = _region_macro_mae_1_6(model_dir, region)
            rows.append({"model": model, "region": region, "macro_mae_1_6": macro_mae, "n_levels_used": n_levels})
    long = pd.DataFrame(rows)
    stats = long.groupby("model")["macro_mae_1_6"].agg(["mean", "std", "count"])
    thin = long[long["n_levels_used"] < len(list(MACRO_MAE_LEVELS))]
    if len(thin):
        print(f"compute_soundscape_macro_mae_stats: {len(thin)}/{len(long)} (model, region) cells used fewer "
              f"than {len(list(MACRO_MAE_LEVELS))} levels for their macro-MAE(1-6) average (zero unambiguous "
              f"clips at the missing level(s) in that region) -- see function docstring.")
    return stats


def _region_synthetic_macro_mae_1_6(model_dir: Path, region: str) -> tuple[float, int]:
    """Synthetic counterpart to _region_macro_mae_1_6(): unweighted mean of
    per-level MAE over levels 1-6, this time from eval_results/{region}_reg_
    test_results.pkl (y_true.polyphony is an exact label here, not a
    [min, max] range, so there is no unambiguous-subset restriction to
    apply -- every row is used as-is)."""
    path = model_dir / "eval_results" / f"{region}_reg_test_results.pkl"
    if not path.exists():
        return float("nan"), 0
    df = pickle.load(open(path, "rb"))
    true_level = df["y_true.polyphony"].round().astype(int)
    pred_level = df["predictions.polyphony_reg"].round().astype(int)

    level_maes = []
    for level in MACRO_MAE_LEVELS:
        mask = true_level == level
        if mask.any():
            level_maes.append(float((true_level[mask] - pred_level[mask]).abs().mean()))
    if not level_maes:
        return float("nan"), 0
    return float(np.mean(level_maes)), len(level_maes)


def compute_synthetic_macro_mae_stats(models: list[str]) -> pd.DataFrame:
    """Synthetic macro-MAE(1-6) per backbone, mirroring
    compute_soundscape_macro_mae_stats()'s construction (per-region
    macro-MAE first, then mean/SD/n across the 7 region-matched datasets) so
    both panels of build_figure_macro_mae() are the same quantity computed
    the same way, just on different data.

    Unlike soundscape, the synthetic mixture is close to uniformly
    distributed across polyphony levels 0-8 by construction (each region has
    roughly len(region)/9 clips per level, confirmed empirically -- e.g.
    HSN's 263 synthetic clips split ~23-34 per level, vs. its soundscape
    counterpart's real-world skew toward 0/1), so every region has ample
    support at every level and there is no missing-level thinning to flag
    here the way there is for soundscape."""
    archive_dir = ARCHIVE / "Pooled-Embeddings"
    rows = []
    for model in models:
        model_dir = archive_dir / model
        for region in REGIONAL_DATASETS:
            macro_mae, n_levels = _region_synthetic_macro_mae_1_6(model_dir, region)
            rows.append({"model": model, "region": region, "macro_mae_1_6": macro_mae, "n_levels_used": n_levels})
    long = pd.DataFrame(rows)
    return long.groupby("model")["macro_mae_1_6"].agg(["mean", "std", "count"])


def compute_head_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """Synthetic-only, per-model mean/SD/n across all 8 datasets, separately
    for the regression and classification formulations (-> DataFrame indexed
    by (model, head)) -- feeds build_figure_reg_vs_class()."""
    sub = filter_long(long_df, source=SOURCE, metric="mae", head=["reg", "class"], dataset=ALL_DATASETS)
    return sub.groupby(["model", "head"])["value"].agg(["mean", "std", "count"])


def _full_encoding_legend_groups():
    """Architecture (shape) + Domain (color) + Paradigm (fill) legend, shared
    by every figure below that uses the full 3-way point encoding. Domain
    goes in the middle, not first: build_encoding_legend centers each group
    at an evenly-spaced x regardless of its own width, and Domain (6 entries)
    is by far the widest -- centered in the leftmost third its left edge runs
    past the canvas; centered in the middle it has room to expand either
    direction."""
    return [
        ("Architecture", [(a, dict(marker=m, markerfacecolor="0.3", markeredgecolor="0.3", color="0.3")) for a, m in MARKER_BY_ARCH.items()]),
        ("Domain", [(d, dict(marker="o", markerfacecolor=c, markeredgecolor=c, color=c)) for d, c in DOMAIN_COLOR.items()]),
        ("Paradigm", [
            ("Supervised (SL)", dict(marker="o", markerfacecolor="0.3", markeredgecolor="0.3", color="0.3")),
            ("Self-supervised (SSL)", dict(marker="o", markerfacecolor="none", markeredgecolor="0.3", color="0.3")),
        ]),
    ]


def _draw_encoded_points(ax, stats: pd.DataFrame, x_order: list[str], *, n_total: int, dot_values=None,
                          errorbar_lighten=None):
    """Draw one dot+whisker per backbone at stats.loc[model] = (mean, sd, n),
    using the full architecture/domain/paradigm encoding. Shared by
    draw_source_panel(), draw_head_panel(), and build_figure_synthetic_reg().

    dot_values: an optional list to append each drawn point's mean into --
    lets a caller scale the y-axis from only the dots (see build_figure()),
    ignoring the whiskers' extent, without this function needing to know
    anything about how the axis will ultimately be scaled.

    errorbar_lighten: None (default) draws each whisker at full-strength
    domain color, same as its dot; a 0-1 amount instead lightens the whisker
    toward white by that much (via plot_style.lighten_color()) -- see
    ERRORBAR_LIGHTEN / build_figure()."""
    for x, model in enumerate(x_order):
        if model not in stats.index:
            continue
        mean, sd, n = stats.loc[model]
        meta = BACKBONE_META[model]
        color = DOMAIN_COLOR[meta["domain"]]
        ecolor = lighten_color(color, errorbar_lighten) if errorbar_lighten is not None else color
        draw_dot_whisker(
            ax, x, mean, sd,
            facecolor=color,
            edgecolor=color,
            ecolor=ecolor,
            marker=MARKER_BY_ARCH[meta["architecture"]],
            filled=(meta["paradigm"] != "SSL"),
        )
        if dot_values is not None:
            dot_values.append(mean)
        if n < n_total:
            ax.annotate(f"n={int(n)}", (x, mean), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, color="0.4")
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels([BACKBONE_META[m]["display"] for m in x_order], rotation=90, ha="center")
    ax.set_xlim(-0.5, len(x_order) - 0.5)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)


def draw_source_panel(ax, stats: pd.DataFrame, source_key: str, x_order: list[str], *, dot_values=None,
                       errorbar_lighten=None):
    """One point per backbone -- the source/metric split is already applied
    when stats was computed, in compute_source_stats()."""
    _draw_encoded_points(ax, stats, x_order, n_total=len(REGIONAL_DATASETS), dot_values=dot_values,
                         errorbar_lighten=errorbar_lighten)
    ax.set_title(SOURCE_PANEL_TITLE[source_key])


def draw_head_panel(ax, stats: pd.DataFrame, head: str, x_order: list[str]):
    """One point per backbone -- stats is compute_head_stats() indexed by
    (model, head); this slices out one head's rows first."""
    _draw_encoded_points(ax, stats.xs(head, level="head"), x_order, n_total=len(ALL_DATASETS))
    ax.set_title(HEAD_PANEL_TITLE[head])


def build_figure(source_stats: dict[str, pd.DataFrame], out_dir: Path):
    """No in-figure heading -- the caption describes this figure already."""
    x_order = ordered_backbones()
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL_WIDTH_IN, 4.8), sharey=True)
    dot_values = []
    draw_source_panel(axes[0], source_stats["synthetic"], "synthetic", x_order, dot_values=dot_values,
                       errorbar_lighten=ERRORBAR_LIGHTEN)
    draw_source_panel(axes[1], source_stats["soundscape"], "soundscape", x_order, dot_values=dot_values,
                       errorbar_lighten=ERRORBAR_LIGHTEN)
    axes[0].set_ylabel("MAE (lower is better)")

    # Scale the (shared) y-axis to fit only the dots, not the whiskers --
    # sharey=True links axes[0]/axes[1], so setting it on one applies to
    # both; a whisker that reaches past the resulting limits is simply
    # clipped (matplotlib's default clip_on=True for errorbar artists)
    # rather than being allowed to shrink every dot down to fit it in.
    lo, hi = min(dot_values), max(dot_values)
    pad = (hi - lo) * 0.15 if hi > lo else 0.01
    axes[0].set_ylim(lo - pad, hi + pad)

    # bbox_to_anchor's y is in *figure* fraction, not axes-relative -- a
    # negative value (the previous convention here) places the legend's
    # anchor point literally below the canvas edge, which guarantees
    # overflow regardless of tight_layout's rect margin. Keep it positive
    # and small instead, with rect's bottom margin sized to the legend's
    # actual rendered height (entries_ncol=3 keeps Domain's 6 entries to 2
    # rows instead of 6, the tallest group and thus the height driver).
    build_encoding_legend(fig, _full_encoding_legend_groups(), bbox_to_anchor=(0.5, 0.22), entries_ncol=[1, 2, 1])

    fig.tight_layout(rect=(0, 0.36, 1, 1))
    save_fig(fig, out_dir / "rq1_backbone_comparison")
    plt.close(fig)


def build_figure_macro_mae(synthetic_macro_mae_stats: pd.DataFrame, soundscape_macro_mae_stats: pd.DataFrame,
                            out_dir: Path):
    """Variant of build_figure(): both panels now plot macro-MAE(1-6)
    (unweighted mean of per-true-level MAE, levels 1-6, mean +/- SD across
    the 7 region-matched datasets) instead of plain MAE (synthetic) /
    range_mae (soundscape) -- compute_synthetic_macro_mae_stats() and
    compute_soundscape_macro_mae_stats() respectively, same construction on
    both sides so the two panels are now a like-for-like metric, just on
    different data.

    The synthetic mixture is close to uniformly distributed across
    polyphony levels by construction (see compute_synthetic_macro_mae_
    stats()'s docstring), so this barely moves its panel relative to plain
    MAE (~0.6-1.0 vs. ~0.6-1.15 before) -- soundscape macro-MAE(1-6) remains
    several times larger (~1.4-2.8) because soundscape prediction is
    genuinely harder (see compute_soundscape_macro_mae_stats()'s docstring),
    not because of a metric artifact anymore. The two panels still do NOT
    share a y-axis for that reason -- sharing one would still squash the
    synthetic panel into a sliver. Each panel is scaled to its own dots
    instead, same padding convention as build_figure()."""
    x_order = ordered_backbones()
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL_WIDTH_IN, 4.8), sharey=False)

    dot_values_syn = []
    draw_source_panel(axes[0], synthetic_macro_mae_stats, "synthetic", x_order, dot_values=dot_values_syn,
                       errorbar_lighten=ERRORBAR_LIGHTEN)
    axes[0].set_title("Synthetic (macro-MAE, levels 1-6)")
    axes[0].set_ylabel("Macro-MAE, levels 1-6 (lower is better)")
    lo, hi = min(dot_values_syn), max(dot_values_syn)
    pad = (hi - lo) * 0.15 if hi > lo else 0.01
    axes[0].set_ylim(lo - pad, hi + pad)

    dot_values_sc = []
    draw_source_panel(axes[1], soundscape_macro_mae_stats, "soundscape", x_order, dot_values=dot_values_sc,
                       errorbar_lighten=ERRORBAR_LIGHTEN)
    axes[1].set_title("Soundscape (macro-MAE, levels 1-6)")
    axes[1].set_ylabel("Macro-MAE, levels 1-6 (lower is better)")
    lo, hi = min(dot_values_sc), max(dot_values_sc)
    pad = (hi - lo) * 0.15 if hi > lo else 0.01
    axes[1].set_ylim(lo - pad, hi + pad)

    build_encoding_legend(fig, _full_encoding_legend_groups(), bbox_to_anchor=(0.5, 0.22), entries_ncol=[1, 2, 1])

    fig.tight_layout(rect=(0, 0.36, 1, 1))
    save_fig(fig, out_dir / "rq1_backbone_comparison_macro_mae")
    plt.close(fig)


def build_figure_reg_vs_class(head_stats: pd.DataFrame, out_dir: Path):
    """Formulation-comparison variant of build_figure(): same two-panel
    layout and full architecture/domain/paradigm encoding, but the panels
    split regression vs. classification (synthetic test, all 8 datasets)
    instead of synthetic vs. soundscape."""
    x_order = ordered_backbones()
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL_WIDTH_IN, 4.8), sharey=True)
    draw_head_panel(axes[0], head_stats, "reg", x_order)
    draw_head_panel(axes[1], head_stats, "class", x_order)
    axes[0].set_ylabel("MAE (lower is better)")

    build_encoding_legend(fig, _full_encoding_legend_groups(), bbox_to_anchor=(0.5, 0.22), entries_ncol=[1, 2, 1])

    fig.tight_layout(rect=(0, 0.36, 1, 1))
    save_fig(fig, out_dir / "rq1_reg_vs_class")
    plt.close(fig)


def build_figure_synthetic_reg(long_df: pd.DataFrame, out_dir: Path):
    """Single panel, no comparison: synthetic test, regression formulation
    only, across all 8 datasets -- the plain backbone-comparison figure with
    the full architecture/domain/paradigm encoding."""
    x_order = ordered_backbones()
    stats = mean_sd_n(filter_long(long_df, source=SOURCE, metric="mae", head="reg", dataset=ALL_DATASETS), ["model"])

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_IN, 4.8))
    _draw_encoded_points(ax, stats, x_order, n_total=len(ALL_DATASETS))
    ax.set_ylabel("MAE (lower is better)")

    build_encoding_legend(fig, _full_encoding_legend_groups(), bbox_to_anchor=(0.5, 0.22), entries_ncol=[1, 2, 1])

    fig.tight_layout(rect=(0, 0.36, 1, 1))
    save_fig(fig, out_dir / "rq1_synthetic_regression")
    plt.close(fig)


def _draw_jittered_pair(ax, x_order: list[str], keys: tuple[str, str], jitter: dict[str, float],
                         marker_by_key: dict[str, str], stats_lookup, n_total: int, lighten_keys=frozenset(),
                         dot_values=None, errorbar_lighten=None):
    """Draw two dot+whiskers per backbone, side by side (jittered) at the
    same x-position -- domain still colors the point, but marker shape now
    distinguishes `keys` instead of architecture. stats_lookup(key, model)
    returns (mean, sd, n) or None if that combination has no data. Shared by
    build_figure_single_panel() and build_figure_reg_vs_class_single().

    lighten_keys: keys whose domain color is lightened (blended toward
    white) instead of drawn at full saturation -- e.g. build_figure_single_
    panel() lightens "soundscape" so it reads as visually secondary to
    "synthetic" at the same x-position, on top of the marker-shape
    distinction. Empty by default (build_figure_reg_vs_class_single()'s
    "reg"/"class" keys aren't a primary/secondary pair, so neither is
    lightened there).

    dot_values: an optional list to append each drawn point's mean into --
    lets a caller scale the y-axis from only the dots (see
    build_figure_single_panel()), ignoring the whiskers' extent.

    errorbar_lighten: None (default) draws each whisker at the same color as
    its own dot (already lightened or not, per lighten_keys); a 0-1 amount
    instead lightens the whisker further toward white by that much on top of
    whatever the dot's own color is -- see ERRORBAR_LIGHTEN /
    build_figure_single_panel()."""
    for x, model in enumerate(x_order):
        meta = BACKBONE_META[model]
        for key in keys:
            row = stats_lookup(key, model)
            if row is None:
                continue
            mean, sd, n = row
            color = lighten_color(DOMAIN_COLOR[meta["domain"]], 0.5) if key in lighten_keys \
                else DOMAIN_COLOR[meta["domain"]]
            ecolor = lighten_color(color, errorbar_lighten) if errorbar_lighten is not None else color
            draw_dot_whisker(
                ax, x + jitter[key], mean, sd,
                facecolor=color,
                edgecolor=color,
                ecolor=ecolor,
                marker=marker_by_key[key],
                filled=True,
            )
            if dot_values is not None:
                dot_values.append(mean)
            if n < n_total:
                ax.annotate(f"n={int(n)}", (x + jitter[key], mean), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7, color="0.4")
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels([BACKBONE_META[m]["display"] for m in x_order], rotation=90, ha="center")
    ax.set_xlim(-0.5, len(x_order) - 0.5)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.set_ylabel("MAE (lower is better)")


def build_figure_single_panel(source_stats: dict[str, pd.DataFrame], out_dir: Path):
    """Alternative to build_figure(): one panel instead of two, dropping the
    architecture/paradigm shape+fill encoding entirely and using marker
    shape for synthetic vs. soundscape instead, so both test sources sit
    side by side at the same backbone x-position (jittered slightly so
    neither point hides the other)."""
    x_order = ordered_backbones()
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_IN, SINGLE_PANEL_HEIGHT))

    def lookup(source_key, model):
        stats = source_stats[source_key]
        return stats.loc[model] if model in stats.index else None

    dot_values = []
    _draw_jittered_pair(ax, x_order, ("synthetic", "soundscape"), SOURCE_JITTER, MARKER_BY_SOURCE,
                         lookup, n_total=len(REGIONAL_DATASETS), lighten_keys={"soundscape"}, dot_values=dot_values,
                         errorbar_lighten=ERRORBAR_LIGHTEN)

    # Scale the y-axis to fit only the dots, not the whiskers -- same
    # rationale as build_figure()'s: a whisker reaching past the resulting
    # limits is simply clipped rather than being allowed to shrink every dot
    # down to fit it in.
    lo, hi = min(dot_values), max(dot_values)
    pad = (hi - lo) * 0.15 if hi > lo else 0.01
    ax.set_ylim(lo - pad, hi + pad)

    legend_groups = [
        ("Domain", [(d, dict(marker="o", markerfacecolor=c, markeredgecolor=c, color=c)) for d, c in DOMAIN_COLOR.items()]),
        ("Test source", [(SOURCE_PANEL_TITLE[s], dict(marker=m, markerfacecolor="0.3", markeredgecolor="0.3", color="0.3")) for s, m in MARKER_BY_SOURCE.items()]),
    ]
    # See build_figure()'s comment: bbox_to_anchor's y is figure-fraction, so
    # it must stay positive (a negative value overflows the canvas by
    # construction), and entries_ncol=3 for Domain overflows sideways instead
    # (6 entries across 3 columns is wide) -- ncol=2 balances the two. rect's
    # bottom margin is sized to the legend's actual rendered height (measured
    # via check_figure_layout) so there is no leftover gap above it.
    build_encoding_legend(fig, legend_groups, bbox_to_anchor=(0.5, 0.02), entries_ncol=[2, 1])

    fig.tight_layout(rect=(0, 0.17, 1, 1))
    save_fig(fig, out_dir / "rq1_backbone_comparison_single_panel")
    plt.close(fig)


def build_figure_macro_mae_single_panel(synthetic_macro_mae_stats: pd.DataFrame,
                                         soundscape_macro_mae_stats: pd.DataFrame, out_dir: Path):
    """Single-panel counterpart to build_figure_macro_mae(), in the same
    style as build_figure_single_panel(): synthetic macro-MAE(1-6) (circle)
    and soundscape macro-MAE(1-6) (square) jittered side by side at each
    backbone's x-position instead of split across two panels -- both series
    are now the same metric (see build_figure_macro_mae()'s docstring),
    unlike this figure's earlier plain-MAE-vs-macro-MAE version.

    CAVEAT this variant still inherits from being single-panel: there is
    only one y-axis here, so it still shares one between the two series even
    though soundscape macro-MAE(1-6) (~1.4-2.8) remains several times larger
    than synthetic's (~0.6-1.0) -- that gap is now a genuine difficulty
    difference between the two test sources rather than a metric artifact
    (see build_figure_macro_mae()'s docstring), but it still means the axis,
    scaled to fit every dot from both series (same "fit dots, clip whiskers"
    rule as build_figure_single_panel()), compresses the synthetic circles
    into a narrower band than the soundscape squares. Read backbone
    *ordering* within each series, not the absolute visual gap between a
    circle and its own square."""
    x_order = ordered_backbones()
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_IN, SINGLE_PANEL_HEIGHT))

    def lookup(source_key, model):
        stats = synthetic_macro_mae_stats if source_key == "synthetic" else soundscape_macro_mae_stats
        return stats.loc[model] if model in stats.index else None

    dot_values = []
    _draw_jittered_pair(ax, x_order, ("synthetic", "soundscape"), SOURCE_JITTER, MARKER_BY_SOURCE,
                         lookup, n_total=len(REGIONAL_DATASETS), lighten_keys={"soundscape"}, dot_values=dot_values,
                         errorbar_lighten=ERRORBAR_LIGHTEN)
    ax.set_ylabel("Macro-MAE, levels 1-6 (lower is better)")

    lo, hi = min(dot_values), max(dot_values)
    pad = (hi - lo) * 0.15 if hi > lo else 0.01
    ax.set_ylim(lo - pad, hi + pad)

    legend_groups = [
        ("Domain", [(d, dict(marker="o", markerfacecolor=c, markeredgecolor=c, color=c)) for d, c in DOMAIN_COLOR.items()]),
        ("Test source", [(SOURCE_PANEL_TITLE[s], dict(marker=m, markerfacecolor="0.3", markeredgecolor="0.3", color="0.3")) for s, m in MARKER_BY_SOURCE.items()]),
    ]
    build_encoding_legend(fig, legend_groups, bbox_to_anchor=(0.5, 0.02), entries_ncol=[2, 1])

    fig.tight_layout(rect=(0, 0.17, 1, 1))
    save_fig(fig, out_dir / "rq1_backbone_comparison_macro_mae_single_panel")
    plt.close(fig)


def build_figure_reg_vs_class_single(head_stats: pd.DataFrame, out_dir: Path):
    """Single-panel counterpart to build_figure_reg_vs_class(), in the same
    style as build_figure_single_panel(): regression and classification
    overlaid at each backbone's x-position (jittered) instead of split
    across two panels."""
    x_order = ordered_backbones()
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_IN, 5.1))

    def lookup(head, model):
        return head_stats.loc[(model, head)] if (model, head) in head_stats.index else None

    _draw_jittered_pair(ax, x_order, ("reg", "class"), HEAD_JITTER, MARKER_BY_HEAD,
                         lookup, n_total=len(ALL_DATASETS))

    legend_groups = [
        ("Domain", [(d, dict(marker="o", markerfacecolor=c, markeredgecolor=c, color=c)) for d, c in DOMAIN_COLOR.items()]),
        ("Formulation", [(HEAD_PANEL_TITLE[h], dict(marker=m, markerfacecolor="0.3", markeredgecolor="0.3", color="0.3")) for h, m in MARKER_BY_HEAD.items()]),
    ]
    build_encoding_legend(fig, legend_groups, bbox_to_anchor=(0.5, 0.02), entries_ncol=[2, 1])

    fig.tight_layout(rect=(0, 0.17, 1, 1))
    save_fig(fig, out_dir / "rq1_reg_vs_class_single")
    plt.close(fig)


SOURCE_TABLE_DATASETS = {SOURCE: ALL_DATASETS, SCAPE_SOURCE: REGIONAL_DATASETS}


def build_top_n_table(long_df: pd.DataFrame, source: str, head: str, n: int = 5) -> str:
    """Top-n backbones by the source's primary MAE-like metric (overview_metrics[0]
    -- "mae" for synthetic, "range_mae" for soundscape, since soundscape has no
    exact ground truth), mean +/- SD across that source's datasets (all 8 for
    synthetic; the 7 region-matched ones for soundscape, which has no XCM
    counterpart)."""
    cfg = mrt.SOURCES[source]
    metric_info = cfg["metric_info"]
    metrics = cfg["overview_metrics"]
    rank_metric = metrics[0]
    datasets = SOURCE_TABLE_DATASETS[source]

    sub = filter_long(long_df, source=source, metric=metrics, head=head, dataset=datasets)
    grouped = sub.groupby(["model", "metric"])["value"]
    avg, std = grouped.mean().unstack("metric"), grouped.std().unstack("metric")
    top_models = avg[rank_metric].nsmallest(n).index.tolist()

    metric_label = metric_info[rank_metric][0]
    source_label = SOURCE_PANEL_TITLE[SOURCE_KEY_BY_SOURCE[source]]
    header = "Backbone & " + " & ".join(mrt.metric_header_cells(metric_info, metrics)) + r" \\"
    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{Top {n} backbones by {metric_label} ({HEAD_PANEL_TITLE[head]} formulation, "
        rf"{source_label.lower()} test), mean $\pm$ SD across {len(datasets)} datasets. "
        rf"Full results for all {len(avg)} backbones in the supplementary table.}}",
        rf"\label{{tab:rq1_top{n}_{source}_{head}}}",
        rf"\begin{{tabular}}{{l{'c' * len(metrics)}}}", r"\toprule", header, r"\midrule",
    ]
    for model in top_models:
        row = [mrt.fmt_mean_std(avg.loc[model, m], std.loc[model, m]) for m in metrics]
        lines.append(f"{mrt.escape_latex(BACKBONE_META[model]['display'])} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq1"))
    parser.add_argument("--rank-head", choices=["reg", "class"], default="reg",
                         help="Which formulation's MAE ranks the top-5 table (default: reg)")
    args = parser.parse_args()

    set_rcparams()
    long_df = load_study("pooled")
    source_stats = compute_source_stats(long_df)
    head_stats = compute_head_stats(long_df)

    build_figure(source_stats, args.out_dir)
    build_figure_single_panel(source_stats, args.out_dir)
    build_figure_reg_vs_class(head_stats, args.out_dir)
    build_figure_reg_vs_class_single(head_stats, args.out_dir)
    build_figure_synthetic_reg(long_df, args.out_dir)

    models = sorted(long_df["model"].unique())
    synthetic_macro_mae_stats = compute_synthetic_macro_mae_stats(models)
    soundscape_macro_mae_stats = compute_soundscape_macro_mae_stats(models)
    build_figure_macro_mae(synthetic_macro_mae_stats, soundscape_macro_mae_stats, args.out_dir)
    build_figure_macro_mae_single_panel(synthetic_macro_mae_stats, soundscape_macro_mae_stats, args.out_dir)

    top5_tex = build_top_n_table(long_df, SOURCE, args.rank_head, n=5)
    (args.out_dir / "rq1_top5_table.tex").write_text(top5_tex)

    top5_scape_tex = build_top_n_table(long_df, SCAPE_SOURCE, args.rank_head, n=5)
    (args.out_dir / "rq1_top5_soundscape_table.tex").write_text(top5_scape_tex)

    full_tex = mrt.build_overview_by_model_table(long_df, SOURCE, models)
    (args.out_dir / "rq1_full_table.tex").write_text(full_tex)

    print(f"Wrote figures/tables to {args.out_dir}")


if __name__ == "__main__":
    main()
