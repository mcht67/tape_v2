#!/usr/bin/env python3
"""
RQ4 -- Synthetic-to-real generalization. XCM excluded throughout (no
soundscape counterpart to compare against).

Figure 1 -- region-specific dumbbell plot (archive/Pooled-Embeddings/, all
13 backbones): per backbone, a synthetic dot (mean MAE across the 7
region-matched synthetic-test subsets) and a soundscape dot (mean MAE
across the same 7 soundscape datasets), each with a +/-1 SD whisker across
those same 7 regions -- so the two whiskers are directly comparable spreads.
Rows sorted by mean soundscape MAE ascending (best at top).

Figure 2 -- multi-task variant slope/bump chart (archive/Spatial-Embeddings/,
the same 5 backbones x 4 TC-head configs as RQ2): one line per config,
x = {synthetic, soundscape}, y = mean MAE across the 5 backbones (each
backbone's own MAE first averaged across its 7 regions), with a +/-1 SD
whisker across the 5 backbones at each endpoint.

Figure 6 -- multi-task variant slope/bump chart, per backbone (same data as
Figure 2, not averaged across backbones): one panel per of the 5
Spatial-Embeddings backbones, sharing a y-scale so cross-backbone MAE is
still comparable; within each panel, one line per TC-head config,
x = {synthetic, soundscape}, y = mean MAE across the 7 region-matched
datasets, with a +/-1 SD whisker across those datasets at each endpoint.
Answers "does a config's synthetic->soundscape generalization gap hold up
per backbone, or is Figure 2's average hiding backbone-specific behavior".

Figure 3 -- soundscape-difficulty LaTeX table: per dataset, #species,
#segments, ratio_polyp, mean_polyp (from plots/data/polybirdmix_soundscape_stats.json)
alongside MAE/Accuracy averaged across all 13 Pooled-Embeddings backbones,
ordered by increasing MAE, with Pearson r/p (vs. MAE) for each characteristic
in the caption.

Figure 4 -- easy-vs-hard backbone-ranking scatter (archive/Pooled-Embeddings/,
all 13 backbones): per backbone, mean soundscape range_mae across the easy
subsets (HSN, NES, SSW) on x vs. the hard subsets (PER, POW) on y, +/-1 SD
whiskers across the datasets within each subset, colored by backbone
identity, with a y=x reference line and the Pearson r/p (easy vs. hard MAE)
annotated in-panel.

Figure 5 -- easy-vs-hard backbone-ranking slope/bump chart: same two subsets
as Figure 4, but plotting each backbone's *rank* (1 = lowest/best MAE) within
the easy subset against its rank within the hard subset, one line per
backbone directly labeled at both ends (no legend needed -- rank is a
permutation of 1..13, so labels never collide within a side) plus the
Spearman rank correlation (easy vs. hard rank) annotated in-panel.

Figure 7 -- 8 more rank-bump-chart variants of Figure 5's idea, one figure,
one panel per variant (all 13 Pooled-Embeddings backbones, all always
head="reg"): soundscape easy vs. hard (Figure 5 itself, repeated here for
side-by-side comparison), soundscape all-7-regions vs. easy, soundscape
all-7-regions vs. hard, synthetic-vs-soundscape (both all-7-regions),
synthetic all-7-regions vs. soundscape easy, synthetic all-7-regions vs.
soundscape hard, synthetic all-7-regions vs. soundscape all-but-hard (the 5
regions left after dropping PER/POW), and synthetic easy vs. hard. Unlike
Figure 5, backbones aren't labeled by name at the line endpoints (8 panels
side by side leaves no room) -- color (BACKBONE_META identity color) plus
one shared legend across all panels carries backbone identity instead, and
each panel is annotated with its own Spearman rho/p.

Figure 8 -- synthetic/easy/hard/all 4-column rank-bump chart (same 13
Pooled-Embeddings backbones, head="reg"): each backbone's rank (1 =
lowest/best MAE) within synthetic all-7-regions, soundscape easy (HSN, NES,
SSW), soundscape hard (PER, POW), and soundscape all-7-regions, plotted as
one 4-point line per backbone. Unlike Figure 5, a backbone's name is
labeled once, to the left of the synthetic column, rather than repeated at
every column. All 6 pairwise Spearman rank correlations between the 4
columns are computed but not drawn in-figure -- written instead to
rq4_fig8_correlations.md alongside the figure.

Usage:
    complete-venv/bin/python source/scripts/plot_rq4.py [--out-dir plots/figures/rq4]
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
import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import BACKBONE_META, ordered_backbones
from plot_data import REGIONAL_DATASETS, filter_long, load_study, mean_sd_n
from plot_rq2 import CONFIG_COLOR, CONFIG_HEAD, CONFIG_LABELS, SINGLE_XTICK_FONT_PT, SPATIAL_BACKBONES
from plot_style import build_encoding_legend, draw_dot_whisker, save_fig, set_rcparams, FIGURE_FONT_PT, ONE_COL_WIDTH_IN, TWO_COL_WIDTH_IN

SYN_SOURCE = "synthetic_mixture_test"
SCAPE_SOURCE = "soundscape_test"
SOURCE_METRIC = {SYN_SOURCE: "mae", SCAPE_SOURCE: "range_mae"}
STATS_JSON = Path(__file__).resolve().parents[2] / "plots" / "data" / "polybirdmix_soundscape_stats.json"

# Easy/hard soundscape split used by Figures 4-5 -- same criterion as the
# user's own "backbone ranking is largely uncorrelated across difficulty"
# table: HSN/NES/SSW are the subsets every backbone scores best on (lowest
# mean range_mae across the 13 Pooled-Embeddings backbones), PER/POW the
# worst, per Figure 3's per-dataset MAE ordering. UHH and SNE sit in between
# and are left out so the two groups stay maximally separated.
EASY_DATASETS = ["HSN", "NES", "SSW"]
HARD_DATASETS = ["PER", "POW"]
# All 7 regional datasets except the hard ones (i.e. easy + the two
# in-between datasets, UHH/SNE) -- used by Figure 7's "All-but-hard" panel.
ALL_BUT_HARD_DATASETS = [d for d in REGIONAL_DATASETS if d not in HARD_DATASETS]


# ---------------------------------------------------------------------------
# Figure 1 -- region-specific dumbbell
# ---------------------------------------------------------------------------

def build_fig1(pooled_long: pd.DataFrame, out_dir: Path):
    """No in-figure heading -- the caption describes this figure already."""
    syn = filter_long(pooled_long, source=SYN_SOURCE, metric="mae", head="reg", dataset=REGIONAL_DATASETS)
    scape = filter_long(pooled_long, source=SCAPE_SOURCE, metric="range_mae", head="reg", dataset=REGIONAL_DATASETS)
    syn_stats = mean_sd_n(syn, ["model"])
    scape_stats = mean_sd_n(scape, ["model"])

    # Descending sort -> worst (highest MAE) at y=0 (bottom), best (lowest MAE) at the top row.
    order = scape_stats["mean"].sort_values(ascending=False).index.tolist()

    row_offset = 0.16  # separates the synthetic row from the soundscape row within each backbone's slot

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_IN, 5.7))
    for y, model in enumerate(order):
        syn_mean, syn_sd, _ = syn_stats.loc[model]
        scape_mean, scape_sd, _ = scape_stats.loc[model]
        color = BACKBONE_META[model]["identity_color"]
        syn_y, scape_y = y + row_offset, y

        draw_dot_whisker(ax, syn_y, syn_mean, syn_sd, facecolor="0.85", edgecolor="0.5",
                          marker="o", s=70, orientation="horizontal", zorder=2)
        draw_dot_whisker(ax, scape_y, scape_mean, scape_sd, facecolor=color, edgecolor=color,
                          marker="o", s=70, orientation="horizontal", zorder=3)

    ax.set_yticks([y + row_offset / 2 for y in range(len(order))])
    ax.set_yticklabels([BACKBONE_META[m]["display"] for m in order])
    ax.set_xlabel("MAE (lower is better)")
    ax.grid(axis="x", linewidth=0.5, alpha=0.4)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="0.85", markeredgecolor="0.5", markersize=9, label="Synthetic"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="0.3", markeredgecolor="0.3", markersize=9, label="Soundscape"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)

    fig.tight_layout()
    save_fig(fig, out_dir / "rq4_fig1_dumbbell")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 -- multi-task variant slope/bump chart
# ---------------------------------------------------------------------------

def build_fig2(spatial_long: pd.DataFrame, out_dir: Path):
    """No in-figure heading -- the caption describes this figure already."""
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_IN, 4.0))

    for label, head in CONFIG_HEAD.items():
        color = CONFIG_COLOR[label]
        endpoint_means, endpoint_sds = [], []
        for source, metric in [(SYN_SOURCE, "mae"), (SCAPE_SOURCE, "range_mae")]:
            sub = filter_long(spatial_long, source=source, metric=metric, head=head,
                               model=SPATIAL_BACKBONES, dataset=REGIONAL_DATASETS)
            per_backbone = sub.groupby("model")["value"].mean()  # average across 7 regions first
            endpoint_means.append(per_backbone.mean())
            endpoint_sds.append(per_backbone.std(ddof=1))

        xs = [0, 1]
        ax.plot(xs, endpoint_means, color=color, marker="o", markersize=9, linewidth=2, label=label, zorder=2)
        for x, mean, sd in zip(xs, endpoint_means, endpoint_sds):
            ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor=color, capsize=5, zorder=1)

    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Synthetic", "Soundscape"])
    ax.set_ylabel("MAE (lower is better)")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, title="Config")

    fig.tight_layout()
    save_fig(fig, out_dir / "rq4_fig2_multitask_slope")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 -- multi-task variant slope/bump chart, per backbone
# ---------------------------------------------------------------------------

FIG6_TICK_LABEL = {SYN_SOURCE: "Synth", SCAPE_SOURCE: "Scape"}


def build_fig6(spatial_long: pd.DataFrame, out_dir: Path):
    """No in-figure heading -- the caption describes this figure already.
    Same per-config lines as build_fig2(), one panel per backbone instead of
    averaged across backbones -- panels share a y-axis so magnitude stays
    comparable across backbones, not just within one."""
    fig, axes = plt.subplots(1, len(SPATIAL_BACKBONES), figsize=(TWO_COL_WIDTH_IN, 3.4), sharey=True)

    for ax, model in zip(axes, SPATIAL_BACKBONES):
        for label, head in CONFIG_HEAD.items():
            color = CONFIG_COLOR[label]
            means, sds = [], []
            for source, metric in [(SYN_SOURCE, "mae"), (SCAPE_SOURCE, "range_mae")]:
                sub = filter_long(spatial_long, source=source, metric=metric, head=head,
                                   model=model, dataset=REGIONAL_DATASETS)
                means.append(sub["value"].mean())
                sds.append(sub["value"].std(ddof=1))

            xs = [0, 1]
            ax.plot(xs, means, color=color, marker="o", markersize=5, linewidth=1.4, label=label, zorder=2)
            for x, mean, sd in zip(xs, means, sds):
                ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor=color, capsize=3, zorder=1)

        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([FIG6_TICK_LABEL[SYN_SOURCE], FIG6_TICK_LABEL[SCAPE_SOURCE]], fontsize=SINGLE_XTICK_FONT_PT)
        ax.set_title(BACKBONE_META[model]["display"], fontsize=SINGLE_XTICK_FONT_PT)
        ax.grid(axis="y", linewidth=0.5, alpha=0.4)

    axes[0].set_ylabel("MAE (lower is better)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, title="Config")

    fig.tight_layout(rect=(0, 0.22, 1, 1))
    save_fig(fig, out_dir / "rq4_fig6_multitask_slope_per_backbone")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 -- soundscape-difficulty LaTeX table
# ---------------------------------------------------------------------------

def build_fig3(pooled_long: pd.DataFrame, out_dir: Path) -> str:
    stats = json.loads(STATS_JSON.read_text())["subsets"]

    scape = filter_long(pooled_long, source=SCAPE_SOURCE, metric=["range_mae", "range_accuracy"],
                         head="reg", dataset=REGIONAL_DATASETS)
    per_dataset = scape.groupby(["dataset", "metric"])["value"].mean().unstack("metric")

    rows = []
    for ds in REGIONAL_DATASETS:
        s = stats[ds]
        rows.append(dict(
            dataset=ds,
            num_species=s["n_unique_species"],
            num_segments=s["n_soundscape_segments"],
            ratio_polyp=s["polyphony_metric_summary"]["ratio_polyp"]["mean"],
            mean_polyp=s["polyphony_metric_summary"]["mean_polyp"]["mean"],
            mae=per_dataset.loc[ds, "range_mae"],
            accuracy=per_dataset.loc[ds, "range_accuracy"],
        ))
    table_df = pd.DataFrame(rows).sort_values("mae")

    correlates = ["num_species", "num_segments", "ratio_polyp", "mean_polyp"]
    corr = {c: scipy.stats.pearsonr(table_df[c], table_df["mae"]) for c in correlates}
    corr_str = "; ".join(f"{c}: $r={r.statistic:.2f}$, $p={r.pvalue:.3f}$" for c, r in corr.items())

    header = r"Dataset & \#Species & \#Segments & Ratio polyp. & Mean polyp. & MAE $\downarrow$ & Accuracy $\uparrow$ \\"
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Soundscape difficulty by dataset (region-specific Pooled-Embeddings models, "
        r"mean across 13 backbones), ordered by increasing MAE. Pearson correlation with MAE: "
        + corr_str + ".}",
        r"\label{tab:rq4_fig3_difficulty}",
        r"\begin{tabular}{lccccc c}", r"\toprule", header, r"\midrule",
    ]
    for _, row in table_df.iterrows():
        lines.append(
            f"{mrt.escape_latex(row['dataset'])} & {int(row['num_species'])} & {int(row['num_segments'])} & "
            f"{row['ratio_polyp']:.2f} & {row['mean_polyp']:.2f} & {row['mae']:.4f} & {row['accuracy']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    (out_dir / "rq4_fig3_difficulty_table.tex").write_text(tex)
    return tex


# ---------------------------------------------------------------------------
# Figure 4 -- easy-vs-hard backbone-ranking scatter
# ---------------------------------------------------------------------------

def _easy_hard_stats(pooled_long: pd.DataFrame) -> pd.DataFrame:
    """Per-backbone mean +/- SD of soundscape range_mae within the easy
    (HSN, NES, SSW) and hard (PER, POW) subsets -- shared by Figures 4-5."""
    easy = filter_long(pooled_long, source=SCAPE_SOURCE, metric="range_mae", head="reg", dataset=EASY_DATASETS)
    hard = filter_long(pooled_long, source=SCAPE_SOURCE, metric="range_mae", head="reg", dataset=HARD_DATASETS)
    easy_stats = mean_sd_n(easy, ["model"])
    hard_stats = mean_sd_n(hard, ["model"])
    return pd.DataFrame({
        "easy_mean": easy_stats["mean"], "easy_sd": easy_stats["std"],
        "hard_mean": hard_stats["mean"], "hard_sd": hard_stats["std"],
    })


def build_fig4(pooled_long: pd.DataFrame, out_dir: Path):
    """No in-figure heading -- the caption describes this figure already."""
    stats = _easy_hard_stats(pooled_long)
    backbone_order = ordered_backbones(subset=stats.index)
    r, p = scipy.stats.pearsonr(stats["easy_mean"], stats["hard_mean"])

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_IN, 5.3))

    lo = min(stats["easy_mean"].min(), stats["hard_mean"].min()) - 0.05
    hi = max(stats["easy_mean"].max(), stats["hard_mean"].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color="0.6", linestyle="--", linewidth=1, zorder=1)

    for model in backbone_order:
        row = stats.loc[model]
        color = BACKBONE_META[model]["identity_color"]
        ax.errorbar(row["easy_mean"], row["hard_mean"], xerr=row["easy_sd"], yerr=row["hard_sd"],
                     fmt="none", ecolor=color, alpha=0.5, capsize=3, zorder=2)
        ax.scatter(row["easy_mean"], row["hard_mean"], facecolor=color, edgecolor=color, s=70, zorder=3)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Easy (HSN, NES, SSW) MAE $\\downarrow$")
    ax.set_ylabel("Hard (PER, POW) MAE $\\downarrow$")
    ax.grid(linewidth=0.5, alpha=0.4)
    ax.text(0.97, 0.03, f"Pearson $r={r:.2f}$, $p={p:.2f}$", transform=ax.transAxes,
            ha="right", va="bottom")

    fig.tight_layout(rect=(0, 0.22, 1, 1))
    build_encoding_legend(fig, [(None, [
        (BACKBONE_META[m]["display"], dict(marker="o", markerfacecolor=BACKBONE_META[m]["identity_color"],
         markeredgecolor=BACKBONE_META[m]["identity_color"], color=BACKBONE_META[m]["identity_color"]))
        for m in backbone_order
    ])], bbox_to_anchor=(0.5, 0.02), entries_ncol=4)

    save_fig(fig, out_dir / "rq4_fig4_easy_hard_scatter")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 -- easy-vs-hard backbone-ranking slope/bump chart
# ---------------------------------------------------------------------------

# Widest backbone display name ("BEANS baseline"/"NatureLM-audio") rendered
# at FIGURE_FONT_PT measures ~0.85in (measured via get_window_extent, the
# same technique check_figure_layout uses) -- ROW_LABEL_MAX_W adds a small
# buffer over that so every label (all 13 appear on both sides, at
# different rows) clears its anchor point with room to spare.
ROW_LABEL_MAX_W = 0.86  # inches, at FIGURE_FONT_PT
ROW_LABEL_GAP = 0.05  # inches, between a label's near edge and the line endpoint
ROW_LINE_SPAN = 1.2  # inches, horizontal distance between the two line endpoints


def build_fig5(pooled_long: pd.DataFrame, out_dir: Path):
    """No in-figure heading -- the caption describes this figure already.

    The axes are placed via add_axes to span the figure's full width (not
    plt.subplots + tight_layout's own auto-sized margins), with xlim set to
    [0, ONE_COL_WIDTH_IN] -- so 1 data-x unit is exactly 1 inch and the
    label/line placement below (ROW_LABEL_MAX_W/GAP/LINE_SPAN, all in
    inches) can be reasoned about, and verified, directly against the
    column width, rather than guessing at axes-fraction margins. Confirmed
    via check_figure_layout + get_tightbbox that the full rendered figure
    (13 two-sided FIGURE_FONT_PT labels included) lands within
    ONE_COL_WIDTH_IN itself, rather than relying on save_fig's
    bbox_inches="tight" to silently grow the canvas past that width (which
    would make \\includegraphics[width=\\columnwidth] downscale every label
    below its nominal point size -- see plot_style.py's ONE_COL_WIDTH_IN/
    TWO_COL_WIDTH_IN comment)."""
    stats = _easy_hard_stats(pooled_long)
    easy_rank = stats["easy_mean"].rank(method="first").astype(int)
    hard_rank = stats["hard_mean"].rank(method="first").astype(int)
    rho, p = scipy.stats.spearmanr(stats["easy_mean"], stats["hard_mean"])
    n = len(stats)

    anchor_left = ROW_LABEL_MAX_W + ROW_LABEL_GAP
    anchor_right = anchor_left + ROW_LINE_SPAN

    fig = plt.figure(figsize=(ONE_COL_WIDTH_IN, 3.5))
    ax = fig.add_axes((0, 0.09, 1, 0.85))

    for model in stats.index:
        color = BACKBONE_META[model]["identity_color"]
        display = BACKBONE_META[model]["display"]
        ey, hy = easy_rank[model], hard_rank[model]
        ax.plot([anchor_left, anchor_right], [ey, hy], color=color, marker="o", markersize=5, linewidth=1.4, zorder=2)
        ax.text(anchor_left - ROW_LABEL_GAP, ey, display, color=color, ha="right", va="center", clip_on=False)
        ax.text(anchor_right + ROW_LABEL_GAP, hy, display, color=color, ha="left", va="center", clip_on=False)

    ax.set_xlim(0, ONE_COL_WIDTH_IN)
    ax.set_ylim(n + 0.6, 0.4)  # inverted: rank 1 (best) at the top
    ax.set_yticks([])
    ax.set_xticks([anchor_left, anchor_right])
    ax.set_xticklabels(["Easy\n(HSN, NES, SSW)", "Hard\n(PER, POW)"])
    ax.spines[["left", "right", "top"]].set_visible(False)
    for y in range(1, n + 1):
        ax.axhline(y, color="0.85", linewidth=0.4, zorder=0)
    # fig.text, not ax.set_title -- add_axes' manual rect (above) isn't
    # followed by tight_layout, so a title would sit at the axes' own top
    # edge (inside the 0.85 height fraction) rather than in the margin.
    fig.text(0.5, 0.97, f"Spearman $\\rho={rho:.2f}$, $p={p:.2f}$", fontsize=FIGURE_FONT_PT, ha="center", va="top")

    save_fig(fig, out_dir / "rq4_fig5_easy_hard_bump")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7 -- 5 more rank-bump-chart variants of Figure 5's idea
# ---------------------------------------------------------------------------

# Each entry: (panel title, left side, right side), each side
# (source, dataset subset, short endpoint label). Order matches the request
# this figure was built for.
FIG7_VARIANTS = [
    ("Soundscape\nEasy vs Hard",
     (SCAPE_SOURCE, EASY_DATASETS, "Scape\nEasy"), (SCAPE_SOURCE, HARD_DATASETS, "Scape\nHard")),
    ("Soundscape\nAll vs Easy",
     (SCAPE_SOURCE, REGIONAL_DATASETS, "Scape\nAll"), (SCAPE_SOURCE, EASY_DATASETS, "Scape\nEasy")),
    ("Soundscape\nAll vs Hard",
     (SCAPE_SOURCE, REGIONAL_DATASETS, "Scape\nAll"), (SCAPE_SOURCE, HARD_DATASETS, "Scape\nHard")),
    ("Synthetic vs\nSoundscape",
     (SYN_SOURCE, REGIONAL_DATASETS, "Synth\nAll"), (SCAPE_SOURCE, REGIONAL_DATASETS, "Scape\nAll")),
    ("Synthetic All vs\nSoundscape Easy",
     (SYN_SOURCE, REGIONAL_DATASETS, "Synth\nAll"), (SCAPE_SOURCE, EASY_DATASETS, "Scape\nEasy")),
    ("Synthetic All vs\nSoundscape Hard",
     (SYN_SOURCE, REGIONAL_DATASETS, "Synth\nAll"), (SCAPE_SOURCE, HARD_DATASETS, "Scape\nHard")),
    ("Synthetic All vs\nSoundscape All-Hard",
     (SYN_SOURCE, REGIONAL_DATASETS, "Synth\nAll"), (SCAPE_SOURCE, ALL_BUT_HARD_DATASETS, "Scape\nAll−Hard")),
    ("Synthetic\nEasy vs Hard",
     (SYN_SOURCE, EASY_DATASETS, "Synth\nEasy"), (SYN_SOURCE, HARD_DATASETS, "Synth\nHard")),
]


def _rank_source_stats(pooled_long: pd.DataFrame, source: str, dataset_subset: list[str]) -> pd.Series:
    """Per-model mean MAE (lower is better) over dataset_subset for one
    source -- one side of a Figure 7 panel, before ranking."""
    sub = filter_long(pooled_long, source=source, metric=SOURCE_METRIC[source], head="reg", dataset=dataset_subset)
    return sub.groupby("model")["value"].mean()


def draw_rank_bump_panel(ax, pooled_long: pd.DataFrame, left: tuple, right: tuple, backbone_order: list[str]):
    """One Figure 7 panel: rank-1..13 bump chart between `left` and `right`
    (each a (source, dataset_subset, endpoint_label) tuple), colored by
    backbone identity, no per-line text (see module docstring). Returns
    (rho, p) so the caller can annotate it."""
    left_source, left_datasets, left_label = left
    right_source, right_datasets, right_label = right
    left_vals = _rank_source_stats(pooled_long, left_source, left_datasets)
    right_vals = _rank_source_stats(pooled_long, right_source, right_datasets)
    models = [m for m in backbone_order if m in left_vals.index and m in right_vals.index]
    left_rank = left_vals[models].rank(method="first")
    right_rank = right_vals[models].rank(method="first")
    rho, p = scipy.stats.spearmanr(left_vals[models], right_vals[models])
    n = len(models)

    for model in models:
        color = BACKBONE_META[model]["identity_color"]
        ax.plot([0, 1], [left_rank[model], right_rank[model]], color=color, marker="o", markersize=4,
                 linewidth=1.2, zorder=2)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(n + 0.6, 0.4)  # inverted: rank 1 (best) at the top, same convention as Figure 5
    ax.set_yticks([])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([left_label, right_label], fontsize=SINGLE_XTICK_FONT_PT)
    for y in range(1, n + 1):
        ax.axhline(y, color="0.9", linewidth=0.4, zorder=0)
    ax.spines[["left", "right", "top"]].set_visible(False)
    # Own row below the two-line xtick labels (rather than sharing their
    # baseline) -- at 7 panels the endpoint labels ("Scape\nEasy" etc.) eat
    # most of a narrow panel's width, so a centered annotation level with
    # them collides with both; stacking it underneath keeps it clear
    # regardless of panel count/width.
    ax.text(0.5, -0.22, f"$\\rho={rho:.2f}$", transform=ax.transAxes, ha="center", va="top", fontsize=SINGLE_XTICK_FONT_PT)
    return rho, p


def build_fig7(pooled_long: pd.DataFrame, out_dir: Path, filename: str = "rq4_fig7_bump_variants"):
    """No in-figure heading -- the caption describes this figure already."""
    backbone_order = ordered_backbones(subset=set(pooled_long["model"]))

    fig, axes = plt.subplots(1, len(FIG7_VARIANTS), figsize=(TWO_COL_WIDTH_IN, 3.6))
    for ax, (title, left, right) in zip(axes, FIG7_VARIANTS):
        draw_rank_bump_panel(ax, pooled_long, left, right, backbone_order)
        ax.set_title(title, fontsize=SINGLE_XTICK_FONT_PT)

    fig.tight_layout(rect=(0, 0.22, 1, 1))
    build_encoding_legend(fig, [(None, [
        (BACKBONE_META[m]["display"], dict(marker="o", markerfacecolor=BACKBONE_META[m]["identity_color"],
         markeredgecolor=BACKBONE_META[m]["identity_color"], color=BACKBONE_META[m]["identity_color"]))
        for m in backbone_order
    ])], bbox_to_anchor=(0.5, 0.02), entries_ncol=4)

    save_fig(fig, out_dir / filename)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8 -- synthetic/easy/hard 3-column rank-bump chart
# ---------------------------------------------------------------------------

FIG8_RIGHT_MARGIN_IN = 0.15  # inches, clearance past the Hard column's dots
FIG8_WIDTH_IN = ONE_COL_WIDTH_IN


# Each entry: (short pair name, first column key, second column key) --
# drives both the 6 pairwise Spearman correlations and their .md report.
FIG8_COLUMNS = ["Synthetic", "Easy", "Hard", "All"]
FIG8_CORR_PAIRS = [
    ("Synthetic-Easy", "Synthetic", "Easy"),
    ("Synthetic-Hard", "Synthetic", "Hard"),
    ("Synthetic-All", "Synthetic", "All"),
    ("Easy-Hard", "Easy", "Hard"),
    ("Easy-All", "Easy", "All"),
    ("Hard-All", "Hard", "All"),
]


def build_fig8(pooled_long: pd.DataFrame, out_dir: Path):
    """No in-figure heading -- the caption describes this figure already.

    Same rank-bump idea as Figure 5, but 4 columns (synthetic all-7-regions,
    soundscape easy, soundscape hard, soundscape all-7-regions) instead of 2,
    and -- since a backbone's name would otherwise repeat 4x per row --
    labeled only once, to the left of the synthetic column, rather than at
    every column like Figure 5 labels both its endpoints. Axes placement
    follows Figure 5's inches-as-data-units trick (see its docstring) so the
    label/column spacing is reasoned about directly in inches rather than
    axes-fraction guesswork.

    All 6 pairwise Spearman rank correlations between the 4 columns are
    computed here but not drawn in-figure (no room at one-column width with
    4 columns already using it) -- see build_fig8_correlations_md() below,
    written alongside the figure instead."""
    backbone_order = ordered_backbones(subset=set(pooled_long["model"]))
    vals = {
        "Synthetic": _rank_source_stats(pooled_long, SYN_SOURCE, REGIONAL_DATASETS),
        "Easy": _rank_source_stats(pooled_long, SCAPE_SOURCE, EASY_DATASETS),
        "Hard": _rank_source_stats(pooled_long, SCAPE_SOURCE, HARD_DATASETS),
        "All": _rank_source_stats(pooled_long, SCAPE_SOURCE, REGIONAL_DATASETS),
    }
    models = [m for m in backbone_order if all(m in v.index for v in vals.values())]
    ranks = {col: vals[col][models].rank(method="first") for col in FIG8_COLUMNS}
    n = len(models)

    correlations = [
        (label, *scipy.stats.spearmanr(vals[a][models], vals[b][models]), n)
        for label, a, b in FIG8_CORR_PAIRS
    ]
    build_fig8_correlations_md(correlations, out_dir)

    anchor_syn = ROW_LABEL_MAX_W + ROW_LABEL_GAP
    col_span = (FIG8_WIDTH_IN - anchor_syn - FIG8_RIGHT_MARGIN_IN) / 3
    anchors = [anchor_syn + i * col_span for i in range(4)]

    fig = plt.figure(figsize=(FIG8_WIDTH_IN, 3.0))
    ax = fig.add_axes((0, 0.08, 1, 0.86))

    # Lines first, then labels on top (in their own pass) -- so each label's
    # white box sits above every line, including ones that would otherwise
    # cross behind it, not just the line it belongs to (see rq3_finetuning).
    for model in models:
        color = BACKBONE_META[model]["identity_color"]
        ys = [ranks[col][model] for col in FIG8_COLUMNS]
        ax.plot(anchors, ys, color=color, marker="o", markersize=5, linewidth=1.4, zorder=2)

    for model in models:
        color = BACKBONE_META[model]["identity_color"]
        display = BACKBONE_META[model]["display"]
        ax.annotate(display, (anchor_syn, ranks["Synthetic"][model]), textcoords="offset points", xytext=(-6, 0),
                    ha="right", va="center", fontsize=9, color=color,
                    bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor=color, linewidth=0.6),
                    zorder=5, clip_on=False)

    ax.set_xlim(0, FIG8_WIDTH_IN)
    ax.set_ylim(n + 0.6, 0.4)  # inverted: rank 1 (best) at the top
    ax.set_yticks([])
    ax.set_xticks(anchors)
    ax.set_xticklabels(["Synthetic", "Soundscape\nEasy\n(HSN, NES, SSW)", "Soundscape\nHard\n(PER, POW)",
                         "Soundscape\nAll"], fontsize=SINGLE_XTICK_FONT_PT)
    ax.spines[["left", "right", "top"]].set_visible(False)
    for y in range(1, n + 1):
        ax.axhline(y, color="0.85", linewidth=0.4, zorder=0)

    save_fig(fig, out_dir / "rq4_fig8_synth_easy_hard_bump")
    plt.close(fig)


def build_fig8_correlations_md(correlations: list[tuple], out_dir: Path) -> None:
    """correlations: list of (pair label, rho, p, n) -- the 6 pairwise
    Spearman rank correlations between Figure 8's 4 columns, computed in
    build_fig8() but reported here instead of in-figure."""
    lines = [
        "# RQ4 Figure 8 -- pairwise rank correlations",
        "",
        "Spearman rank correlation (backbone MAE ranks, Pooled-Embeddings, head=\"reg\") between each "
        "pair of Figure 8's 4 columns: Synthetic (all 7 regions), Soundscape Easy (HSN, NES, SSW), "
        "Soundscape Hard (PER, POW), Soundscape All (all 7 regions).",
        "",
        "| Pair | Spearman rho | p | n |",
        "|---|---|---|---|",
    ]
    for label, rho, p, n in correlations:
        lines.append(f"| {label} | {rho:+.3f} | {p:.4f} | {n} |")
    lines.append("")
    out_md = out_dir / "rq4_fig8_correlations.md"
    out_md.write_text("\n".join(lines))
    print(f"Wrote {out_md}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq4"))
    args = parser.parse_args()

    set_rcparams()
    pooled_long = load_study("pooled")
    spatial_long = load_study("spatial", models=SPATIAL_BACKBONES)

    build_fig1(pooled_long, args.out_dir)
    build_fig2(spatial_long, args.out_dir)
    build_fig3(pooled_long, args.out_dir)
    build_fig4(pooled_long, args.out_dir)
    build_fig5(pooled_long, args.out_dir)
    build_fig6(spatial_long, args.out_dir)
    build_fig7(pooled_long, args.out_dir)
    build_fig8(pooled_long, args.out_dir)

    print(f"Wrote figures/tables to {args.out_dir}")


if __name__ == "__main__":
    main()
