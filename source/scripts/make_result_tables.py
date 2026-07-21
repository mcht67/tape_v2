#!/usr/bin/env python3
"""
Build a LaTeX document with overview + detailed result tables from a folder
of per-model evaluation CSVs.

Expected folder structure (under --target_dir):

    {target_dir}/{model_name}/eval_results/test_metrics.csv
    {target_dir}/{model_name}/scape_eval_results/soundscape_test_metrics.csv

Each CSV has metric names as the (unnamed) first column / index, and one
column per "{dataset}_{head}" combination, e.g. "HSN_reg", "HSN_class".

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

# raw metric row name -> (display name, "higher"/"lower" is better)
METRIC_INFO = {
    "range_mae": ("MAE", "lower"),
    "range_mse": ("MSE", "lower"),
    "range_accuracy": ("Accuracy", "higher"),
    "off_by_one_range_accuracy": ("Off-by-one Acc.", "higher"),
    "qwk_vs_midpoint": ("QWK", "higher"),
    "pearson_r_vs_midpoint": ("Pearson $r$", "higher"),
    "overlap_precision": ("Overlap Prec.", "higher"),
    "overlap_recall": ("Overlap Rec.", "higher"),
    "overlap_f1": ("Overlap F1", "higher"),
    "support": ("Support", None),
}

# Metrics shown in the compact overview table (averaged across datasets)
OVERVIEW_METRICS = ["range_mae", "range_accuracy", "off_by_one_range_accuracy", "qwk_vs_midpoint"]

# Metrics shown in the detailed per-dataset tables
DETAILED_METRICS = [
    "range_mae",
    "range_mse",
    "range_accuracy",
    "off_by_one_range_accuracy",
    "qwk_vs_midpoint",
    "support",
]

HEAD_ORDER = ["reg", "class"]
HEAD_LABEL = {"reg": "Regression", "class": "Classification"}

SOURCES = {
    "synthetic": {
        "csv_rel_path": Path("eval_results") / "test_metrics.csv",
        "table_desc": "synthetic mixture",
    },
    "soundscape": {
        "csv_rel_path": Path("scape_eval_results") / "soundscape_test_metrics.csv",
        "table_desc": "soundscape",
    },
}

COL_RE = re.compile(r"^(?P<dataset>.+)_(?P<head>reg|class)$")


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
    """
    rows = []
    for model in models:
        for source, cfg in SOURCES.items():
            csv_path = target_dir / model / cfg["csv_rel_path"]
            if not csv_path.is_file():
                continue
            df = pd.read_csv(csv_path, index_col=0)
            for col in df.columns:
                m = COL_RE.match(col.strip())
                if not m:
                    print(f"  [warn] skipping unrecognized column '{col}' in {csv_path}")
                    continue
                dataset = m.group("dataset")
                head = m.group("head")
                for metric, value in df[col].items():
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


def fmt(value, decimals=4):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    if float(value).is_integer() and abs(value) >= 100:
        # e.g. support counts
        return f"{int(value)}"
    return f"{value:.{decimals}f}"


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
    sub = long_df[(long_df["source"] == source) & (long_df["metric"].isin(OVERVIEW_METRICS))]
    if sub.empty:
        return ""

    # average over datasets -> index (model, head), columns metric
    avg = (
        sub.groupby(["model", "head", "metric"])["value"]
        .mean()
        .unstack("metric")
        .reindex(columns=OVERVIEW_METRICS)
    )

    n_cols = len(OVERVIEW_METRICS)
    col_spec = "ll" + "c" * n_cols
    header_cells = []
    for metric in OVERVIEW_METRICS:
        label, direction = METRIC_INFO[metric]
        arrow = r"$\uparrow$" if direction == "higher" else r"$\downarrow$" if direction == "lower" else ""
        header_cells.append(f"{label} {arrow}".strip())
    header = "Backbone & Head & " + " & ".join(header_cells) + r" \\"

    # figure out best value per column across the whole table (model+head combos)
    best_masks = {}
    for metric in OVERVIEW_METRICS:
        _, direction = METRIC_INFO[metric]
        best_masks[metric] = best_mask(avg[metric], direction) if metric in avg else None

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    src_desc = SOURCES[source]["table_desc"]
    lines.append(
        rf"\caption{{Results on {src_desc} data, averaged across evaluation subsets, for each model and head type.}}"
    )
    lines.append(rf"\label{{tab:overview_{source}}}")
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
            for metric in OVERVIEW_METRICS:
                val = avg.loc[(model, head), metric] if metric in avg.columns else np.nan
                cell = fmt(val)
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
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detailed table (per model, per source: rows = head x dataset)
# ---------------------------------------------------------------------------

def build_detailed_table(long_df: pd.DataFrame, source: str, model: str) -> str:
    sub = long_df[
        (long_df["source"] == source)
        & (long_df["model"] == model)
        & (long_df["metric"].isin(DETAILED_METRICS))
    ]
    if sub.empty:
        return ""

    pivot = (
        sub.groupby(["head", "dataset", "metric"])["value"]
        .mean()
        .unstack("metric")
        .reindex(columns=DETAILED_METRICS)
    )

    n_cols = len(DETAILED_METRICS)
    col_spec = "ll" + "c" * n_cols
    header_cells = []
    for metric in DETAILED_METRICS:
        label, direction = METRIC_INFO[metric]
        arrow = r"$\uparrow$" if direction == "higher" else r"$\downarrow$" if direction == "lower" else ""
        header_cells.append(f"{label} {arrow}".strip())
    header = "Head & Dataset & " + " & ".join(header_cells) + r" \\"

    src_desc = SOURCES[source]["table_desc"]
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
            row_vals = [fmt(pivot.loc[(head, dataset), metric]) for metric in DETAILED_METRICS]
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
