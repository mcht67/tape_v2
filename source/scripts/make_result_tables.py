#!/usr/bin/env python3
"""
Build a LaTeX document with overview + detailed result tables from a folder
of per-model evaluation CSVs.

Expected folder structure (under --target_dir):

    {target_dir}/{model_name}/eval_results/test_metrics.csv
    {target_dir}/{model_name}/scape_eval_results/soundscape_test_metrics.csv

Each CSV has metric names as the (unnamed) first column / index, and one
column per "{dataset}_{head}" combination, e.g. "HSN_reg", "HSN_class".

NOTE: the two sources use *different* metric sets, because they evaluate
different scenarios:
  - "synthetic": ground truth is a single point value -> point-estimate
    metrics (mae, rmse, accuracy, ...)
  - "soundscape": ground truth is a range -> range-aware metrics
    (range_mae, range_mse, qwk_vs_midpoint, overlap_f1, ...)
Metric config is therefore defined per-source below, not globally.

Usage:
    python make_result_tables.py --target_dir /path/to/results --out tables.tex
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HEAD_ORDER = ["reg", "class"]
HEAD_LABEL = {"reg": "Regression", "class": "Classification"}

COL_RE = re.compile(r"^(?P<dataset>.+)_(?P<head>reg|class)$")

# Per-source configuration. Each source defines:
#   csv_rel_path : where to find the CSV under a model dir
#   table_desc   : human-readable description used in captions
#   metric_info  : raw metric name (as it appears in the CSV index) ->
#                  (display name, "higher"/"lower"/None direction)
#   overview_metrics : metric keys (raw names) shown in the compact overview
#                       table, averaged across datasets
#   detailed_metrics : metric keys (raw names) shown in the per-dataset
#                       detailed tables
SOURCES = {
    "synthetic": {
        "csv_rel_path": Path("eval_results") / "test_metrics.csv",
        "table_desc": "synthetic mixture",
        "metric_info": {
            "mae": ("MAE", "lower"),
            "rmse": ("RMSE", "lower"),
            "accuracy": ("Accuracy", "higher"),
            "off_by_one_accuracy": ("Off-by-one Acc.", "higher"),
            "macro_f1": ("Macro F1", "higher"),
            "weighted_f1": ("Weighted F1", "higher"),
            "qwk": ("QWK", "higher"),
            "pearson_r": ("Pearson $r$", "higher"),
            "support": ("Support", None),
        },
        "overview_metrics": ["mae", "accuracy", "off_by_one_accuracy", "qwk"],
        "detailed_metrics": [
            "mae",
            "rmse",
            "accuracy",
            "off_by_one_accuracy",
            "qwk",
            "support",
        ],
    },
    "soundscape": {
        "csv_rel_path": Path("scape_eval_results") / "soundscape_test_metrics.csv",
        "table_desc": "soundscape",
        "metric_info": {
            "range_mae": ("MAE", "lower"),
            "range_mse": ("MSE", "lower"),
            "range_accuracy": ("Accuracy", "higher"),
            "off_by_one_range_accuracy": ("Off-by-one Acc.", "higher"),
            "qwk_vs_midpoint": ("QWK", "higher"),
            "qwk_vs_min": ("QWK_min", "higher"),
            "qwk_vs_max": ("QWK_max", "higher"),
            "pearson_r_vs_midpoint": ("Pearson $r$", "higher"),
            "overlap_precision": ("Overlap Prec.", "higher"),
            "overlap_recall": ("Overlap Rec.", "higher"),
            "overlap_f1": ("Overlap F1", "higher"),
            "mean_interval_width": ("Mean Interval Width", "lower"),
            "support": ("Support", None),
        },
        "overview_metrics": [
            "range_mae",
            "range_accuracy",
            "off_by_one_range_accuracy",
            "qwk_vs_midpoint",
        ],
        "detailed_metrics": [
            "range_mae",
            "range_mse",
            "range_accuracy",
            "off_by_one_range_accuracy",
            "qwk_vs_midpoint",
            "mean_interval_width",
            "support",
        ],
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_models(target_dir: Path):
    """A model dir is any subdirectory of target_dir that contains at least
    one of the two expected result CSVs."""
    models = []
    for child in sorted(target_dir.iterdir()):
        if not child.is_dir():
            continue
        has_any = any((child / cfg["csv_rel_path"]).is_file() for cfg in SOURCES.values())
        if has_any:
            models.append(child.name)
    return models


def load_long_df(target_dir: Path, models):
    """Return one tidy long dataframe with columns:
    model, source, dataset, head, metric, value

    `metric` values are the *raw* metric names as they appear in each
    source's CSV; interpretation (display name/direction) is looked up
    per-source via SOURCES[source]["metric_info"].
    """
    rows = []
    for model in models:
        for source, cfg in SOURCES.items():
            csv_path = target_dir / model / cfg["csv_rel_path"]
            if not csv_path.is_file():
                continue
            df = pd.read_csv(csv_path, index_col=0)
            df.index = df.index.map(lambda m: str(m).strip())
            for col in df.columns:
                m = COL_RE.match(col.strip())
                if not m:
                    print(f"  [warn] skipping unrecognized column '{col}' in {csv_path}")
                    continue
                dataset = m.group("dataset")
                head = m.group("head")
                for metric, value in df[col].items():
                    if metric not in cfg["metric_info"]:
                        print(f"  [warn] unrecognized metric '{metric}' for source '{source}' in {csv_path}")
                        continue
                    rows.append(
                        {
                            "model": model,
                            "source": source,
                            "dataset": dataset,
                            "head": head,
                            "metric": metric,
                            "value": value,
                        }
                    )
    if not rows:
        raise RuntimeError(f"No result CSVs found under {target_dir}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def escape_latex(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def fmt(value, decimals=2):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    if float(value).is_integer() and abs(value) >= 100:
        # e.g. support counts
        return f"{int(value)}"
    return f"{value:.{decimals}f}"


def fmt_mean_std(mean_value, std_value, decimals=2):
    """Format a mean value with an optional '$\\pm$ std' suffix.

    The std suffix is omitted when it is unavailable (e.g. NaN, which
    happens when there is only a single dataset to average over).
    """
    mean_str = fmt(mean_value, decimals)
    if mean_str == "--":
        return mean_str
    if std_value is None or (isinstance(std_value, float) and np.isnan(std_value)):
        return mean_str
    return f"{mean_str} $\\pm$ {fmt(std_value, decimals)}"


def bold(text):
    return rf"\textbf{{{text}}}"


def best_mask(series: pd.Series, direction: str):
    """Return a boolean mask marking the best value(s) in a series."""
    if direction == "higher":
        best = series.max()
    elif direction == "lower":
        best = series.min()
    else:
        return pd.Series(False, index=series.index)
    return series == best


# ---------------------------------------------------------------------------
# Overview table (averaged over datasets, one table per source)
# ---------------------------------------------------------------------------

def build_overview_table(long_df: pd.DataFrame, source: str, models) -> str:
    cfg = SOURCES[source]
    metric_info = cfg["metric_info"]
    overview_metrics = cfg["overview_metrics"]

    sub = long_df[(long_df["source"] == source) & (long_df["metric"].isin(overview_metrics))]
    if sub.empty:
        return ""

    # average (and std) over datasets -> index (model, head), columns metric
    grouped = sub.groupby(["model", "head", "metric"])["value"]
    avg = grouped.mean().unstack("metric").reindex(columns=overview_metrics)
    std = grouped.std().unstack("metric").reindex(columns=overview_metrics)

    n_cols = len(overview_metrics)
    col_spec = "ll" + "c" * n_cols
    header_cells = []
    for metric in overview_metrics:
        label, direction = metric_info[metric]
        arrow = r"$\uparrow$" if direction == "higher" else r"$\downarrow$" if direction == "lower" else ""
        header_cells.append(f"{label} {arrow}".strip())
    header = "Backbone & Head & " + " & ".join(header_cells) + r" \\"

    # figure out best value per column across the whole table (model+head combos)
    # "best" is based on the mean, not the std
    best_masks = {}
    for metric in overview_metrics:
        _, direction = metric_info[metric]
        best_masks[metric] = best_mask(avg[metric], direction) if metric in avg else None

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    src_desc = cfg["table_desc"]
    lines.append(
        rf"\caption{{Results on {src_desc} data, averaged (mean $\pm$ std) across evaluation subsets, "
        rf"for each model and head type.}}"
    )
    lines.append(rf"\label{{tab:overview_{source}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")

    for model in models:
        heads_present = [h for h in HEAD_ORDER if (model, h) in avg.index]
        if not heads_present:
            continue
        for i, head in enumerate(heads_present):
            row_vals = []
            for metric in overview_metrics:
                mean_val = avg.loc[(model, head), metric] if metric in avg.columns else np.nan
                std_val = std.loc[(model, head), metric] if metric in std.columns else np.nan
                cell = fmt_mean_std(mean_val, std_val)
                mask = best_masks.get(metric)
                if mask is not None and (model, head) in mask.index and mask.loc[(model, head)]:
                    cell = bold(cell)
                row_vals.append(cell)
            model_cell = (
                rf"\multirow{{{len(heads_present)}}}{{*}}{{{escape_latex(model)}}}" if i == 0 else ""
            )
            lines.append(f"{model_cell} & {HEAD_LABEL[head]} & " + " & ".join(row_vals) + r" \\")
        lines.append(r"\cmidrule(lr){1-2}")
    if lines[-1] == r"\cmidrule(lr){1-2}":
        lines.pop()  # no trailing rule before bottomrule
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detailed table (per model, per source: rows = head x dataset)
# ---------------------------------------------------------------------------

def build_detailed_table(long_df: pd.DataFrame, source: str, model: str) -> str:
    cfg = SOURCES[source]
    metric_info = cfg["metric_info"]
    detailed_metrics = cfg["detailed_metrics"]

    sub = long_df[
        (long_df["source"] == source)
        & (long_df["model"] == model)
        & (long_df["metric"].isin(detailed_metrics))
    ]
    if sub.empty:
        return ""

    pivot = (
        sub.groupby(["head", "dataset", "metric"])["value"]
        .mean()
        .unstack("metric")
        .reindex(columns=detailed_metrics)
    )

    n_cols = len(detailed_metrics)
    col_spec = "ll" + "c" * n_cols
    header_cells = []
    for metric in detailed_metrics:
        label, direction = metric_info[metric]
        arrow = r"$\uparrow$" if direction == "higher" else r"$\downarrow$" if direction == "lower" else ""
        header_cells.append(f"{label} {arrow}".strip())
    header = "Head & Dataset & " + " & ".join(header_cells) + r" \\"

    src_desc = cfg["table_desc"]
    label_safe = re.sub(r"[^a-zA-Z0-9]+", "_", model.lower())

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(
        rf"\caption{{Results on {src_desc} data for {escape_latex(model)}, per dataset and head type.}}"
    )
    lines.append(rf"\label{{tab:supp_{source}_{label_safe}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(header)

    for head in HEAD_ORDER:
        if head not in pivot.index.get_level_values("head"):
            continue
        lines.append(r"\midrule")
        datasets = sorted(pivot.loc[head].index)
        for i, dataset in enumerate(datasets):
            row_vals = [fmt(pivot.loc[(head, dataset), metric]) for metric in detailed_metrics]
            head_cell = (
                rf"\multirow{{{len(datasets)}}}{{*}}{{{HEAD_LABEL[head]}}}" if i == 0 else ""
            )
            lines.append(f"{head_cell} & {escape_latex(dataset)} & " + " & ".join(row_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------

DOC_PREAMBLE = r"""\documentclass{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage{amsmath}
\begin{document}
"""

DOC_END = r"""
\end{document}
"""


def build_document(target_dir: Path) -> str:
    models = discover_models(target_dir)
    if not models:
        raise RuntimeError(f"No model subdirectories with result CSVs found under {target_dir}")
    print(f"Found {len(models)} model(s): {models}")

    long_df = load_long_df(target_dir, models)

    parts = [DOC_PREAMBLE]

    parts.append("\\section*{Overview}\n")
    for source in SOURCES:
        table = build_overview_table(long_df, source, models)
        if table:
            parts.append(table)
            parts.append("\n")

    parts.append("\\clearpage\n\\section*{Detailed results}\n")
    for source in SOURCES:
        for model in models:
            table = build_detailed_table(long_df, source, model)
            if table:
                parts.append(table)
                parts.append("\n")

    parts.append(DOC_END)
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_dir", type=Path, required=True, help="Root folder containing per-model result subfolders")
    parser.add_argument("--out", type=Path, default=Path("tables.tex"), help="Output .tex file path")
    args = parser.parse_args()

    doc = build_document(args.target_dir)
    args.out.write_text(doc)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()