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

Figure 3 -- soundscape-difficulty LaTeX table: per dataset, #species,
#segments, ratio_polyp, mean_polyp (from plots/data/polybirdmix_soundscape_stats.json)
alongside MAE/Accuracy averaged across all 13 Pooled-Embeddings backbones,
ordered by increasing MAE, with Pearson r/p (vs. MAE) for each characteristic
in the caption.

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
from backbone_meta import BACKBONE_META
from plot_data import REGIONAL_DATASETS, filter_long, load_study, mean_sd_n
from plot_rq2 import CONFIG_HEAD, CONFIG_LABELS, SPATIAL_BACKBONES
from plot_style import draw_dot_whisker, save_fig

SYN_SOURCE = "synthetic_mixture_test"
SCAPE_SOURCE = "soundscape_test"
STATS_JSON = Path(__file__).resolve().parents[2] / "plots" / "data" / "polybirdmix_soundscape_stats.json"
CONFIG_COLOR = dict(zip(CONFIG_LABELS, ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]))


# ---------------------------------------------------------------------------
# Figure 1 -- region-specific dumbbell
# ---------------------------------------------------------------------------

def build_fig1(pooled_long: pd.DataFrame, out_dir: Path):
    syn = filter_long(pooled_long, source=SYN_SOURCE, metric="mae", head="reg", dataset=REGIONAL_DATASETS)
    scape = filter_long(pooled_long, source=SCAPE_SOURCE, metric="range_mae", head="reg", dataset=REGIONAL_DATASETS)
    syn_stats = mean_sd_n(syn, ["model"])
    scape_stats = mean_sd_n(scape, ["model"])

    # Descending sort -> worst (highest MAE) at y=0 (bottom), best (lowest MAE) at the top row.
    order = scape_stats["mean"].sort_values(ascending=False).index.tolist()

    row_offset = 0.16  # separates the synthetic row from the soundscape row within each backbone's slot

    fig, ax = plt.subplots(figsize=(8, 7))
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
    ax.set_title("RQ4 Fig 1 -- Synthetic vs. soundscape MAE (Pooled-Embeddings, mean ± SD across 7 regions)")
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
    fig, ax = plt.subplots(figsize=(7, 5.5))

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
    ax.set_title("RQ4 Fig 2 -- Multi-task variants: synthetic vs. soundscape\n(mean ± SD across 5 backbones; XCM excluded)")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, title="Config")

    fig.tight_layout()
    save_fig(fig, out_dir / "rq4_fig2_multitask_slope")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq4"))
    args = parser.parse_args()

    pooled_long = load_study("pooled")
    spatial_long = load_study("spatial", models=SPATIAL_BACKBONES)

    build_fig1(pooled_long, args.out_dir)
    build_fig2(spatial_long, args.out_dir)
    build_fig3(pooled_long, args.out_dir)

    print(f"Wrote figures/tables to {args.out_dir}")


if __name__ == "__main__":
    main()
