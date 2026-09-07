#!/usr/bin/env python3
"""
RQ1 -- Backbone comparison (+ RQ1.b formulation comparison).

Two side-by-side dot+whisker panels (regression left, classification right),
sharing y-axis scale and backbone x-order, from archive/Pooled-Embeddings/.
Per backbone: mean +/- SD MAE across all 8 datasets (UHH/HSN/PER/NES/POW/
SSW/SNE/XCM). Point color = domain, shape = architecture, fill = paradigm.

Also emits a companion LaTeX top-5 table and the full supplementary table
(reusing make_result_tables.py's existing table builder unchanged).

Usage:
    complete-venv/bin/python source/scripts/plot_rq1.py [--out-dir plots/figures/rq1]
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

import make_result_tables as mrt
from backbone_meta import BACKBONE_META, DOMAIN_COLOR, MARKER_BY_ARCH, ordered_backbones
from plot_data import filter_long, load_study
from plot_style import build_encoding_legend, draw_dot_whisker, save_fig

SOURCE = "synthetic_mixture_test"
HEAD_PANEL_TITLE = {"reg": "Regression", "class": "Classification"}
MARKER_BY_HEAD = {"reg": "o", "class": "s"}
HEAD_JITTER = {"reg": -0.15, "class": 0.15}


def compute_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    sub = filter_long(long_df, source=SOURCE, metric="mae", head=["reg", "class"])
    return sub.groupby(["model", "head"])["value"].agg(["mean", "std", "count"])


def draw_panel(ax, stats: pd.DataFrame, head: str, x_order: list[str]):
    for x, model in enumerate(x_order):
        if (model, head) not in stats.index:
            continue
        mean, sd, n = stats.loc[(model, head)]
        meta = BACKBONE_META[model]
        draw_dot_whisker(
            ax, x, mean, sd,
            facecolor=DOMAIN_COLOR[meta["domain"]],
            edgecolor=DOMAIN_COLOR[meta["domain"]],
            marker=MARKER_BY_ARCH[meta["architecture"]],
            filled=(meta["paradigm"] != "SSL"),
        )
        if n < 8:
            ax.annotate(f"n={int(n)}", (x, mean), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, color="0.4")
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels([BACKBONE_META[m]["display"] for m in x_order], rotation=45, ha="right")
    ax.set_title(f"{HEAD_PANEL_TITLE[head]}")
    ax.set_xlim(-0.5, len(x_order) - 0.5)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)


def build_figure(stats: pd.DataFrame, out_dir: Path):
    x_order = ordered_backbones()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    draw_panel(axes[0], stats, "reg", x_order)
    draw_panel(axes[1], stats, "class", x_order)
    axes[0].set_ylabel("MAE (lower is better)")

    legend_groups = [
        ("Domain", [(d, dict(marker="o", markerfacecolor=c, markeredgecolor=c, color=c)) for d, c in DOMAIN_COLOR.items()]),
        ("Architecture", [(a, dict(marker=m, markerfacecolor="0.3", markeredgecolor="0.3", color="0.3")) for a, m in MARKER_BY_ARCH.items()]),
        ("Paradigm", [
            ("Supervised (SL)", dict(marker="o", markerfacecolor="0.3", markeredgecolor="0.3", color="0.3")),
            ("Self-supervised (SSL)", dict(marker="o", markerfacecolor="none", markeredgecolor="0.3", color="0.3")),
        ]),
    ]
    build_encoding_legend(fig, legend_groups, bbox_to_anchor=(0.5, -0.18))

    fig.suptitle("RQ1 -- Backbone comparison (Pooled-Embeddings, mean ± SD MAE across 8 datasets)")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_fig(fig, out_dir / "rq1_backbone_comparison")
    plt.close(fig)


def build_figure_single_panel(stats: pd.DataFrame, out_dir: Path):
    """Alternative to build_figure(): one panel instead of two, dropping the
    architecture/paradigm shape+fill encoding entirely and using marker
    shape for regression vs. classification instead, so both formulations
    sit side by side at the same backbone x-position (jittered slightly so
    neither point hides the other)."""
    x_order = ordered_backbones()
    fig, ax = plt.subplots(figsize=(11, 5.5))

    for x, model in enumerate(x_order):
        meta = BACKBONE_META[model]
        for head in ("reg", "class"):
            if (model, head) not in stats.index:
                continue
            mean, sd, n = stats.loc[(model, head)]
            draw_dot_whisker(
                ax, x + HEAD_JITTER[head], mean, sd,
                facecolor=DOMAIN_COLOR[meta["domain"]],
                edgecolor=DOMAIN_COLOR[meta["domain"]],
                marker=MARKER_BY_HEAD[head],
                filled=True,
            )
            if n < 8:
                ax.annotate(f"n={int(n)}", (x + HEAD_JITTER[head], mean), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7, color="0.4")

    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels([BACKBONE_META[m]["display"] for m in x_order], rotation=45, ha="right")
    ax.set_xlim(-0.5, len(x_order) - 0.5)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.set_ylabel("MAE (lower is better)")

    legend_groups = [
        ("Domain", [(d, dict(marker="o", markerfacecolor=c, markeredgecolor=c, color=c)) for d, c in DOMAIN_COLOR.items()]),
        ("Formulation", [(HEAD_PANEL_TITLE[h], dict(marker=m, markerfacecolor="0.3", markeredgecolor="0.3", color="0.3")) for h, m in MARKER_BY_HEAD.items()]),
    ]
    build_encoding_legend(fig, legend_groups, bbox_to_anchor=(0.5, -0.16))

    fig.suptitle("RQ1 -- Backbone comparison (Pooled-Embeddings, mean ± SD MAE across 8 datasets)")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save_fig(fig, out_dir / "rq1_backbone_comparison_single_panel")
    plt.close(fig)


def build_top_n_table(long_df: pd.DataFrame, head: str, n: int = 5) -> str:
    cfg = mrt.SOURCES[SOURCE]
    metric_info = cfg["metric_info"]
    metrics = cfg["overview_metrics"]  # mae, accuracy, off_by_one_accuracy, qwk

    sub = filter_long(long_df, source=SOURCE, metric=metrics, head=head)
    grouped = sub.groupby(["model", "metric"])["value"]
    avg, std = grouped.mean().unstack("metric"), grouped.std().unstack("metric")
    top_models = avg["mae"].nsmallest(n).index.tolist()

    header = "Backbone & " + " & ".join(mrt.metric_header_cells(metric_info, metrics)) + r" \\"
    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{Top {n} backbones by MAE ({HEAD_PANEL_TITLE[head]} formulation), "
        rf"mean $\pm$ SD across 8 datasets. Full results for all {len(avg)} backbones in the supplementary table.}}",
        rf"\label{{tab:rq1_top{n}_{head}}}",
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

    long_df = load_study("pooled")
    stats = compute_stats(long_df)

    build_figure(stats, args.out_dir)
    build_figure_single_panel(stats, args.out_dir)

    top5_tex = build_top_n_table(long_df, args.rank_head, n=5)
    (args.out_dir / "rq1_top5_table.tex").write_text(top5_tex)

    models = sorted(long_df["model"].unique())
    full_tex = mrt.build_overview_by_model_table(long_df, SOURCE, models)
    (args.out_dir / "rq1_full_table.tex").write_text(full_tex)

    print(f"Wrote figures/tables to {args.out_dir}")


if __name__ == "__main__":
    main()
