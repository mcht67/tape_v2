#!/usr/bin/env python3
"""
Build a LaTeX document with overview + detailed result tables from a folder
of per-model evaluation CSVs.

Expected folder structure (under --target_dir):

    {target_dir}/{model_name}/eval_results/test_metrics.csv                   (synthetic mixture, test split)
    {target_dir}/{model_name}/val_eval_results/val_metrics.csv                (synthetic mixture, val split)
    {target_dir}/{model_name}/scape_eval_results/soundscape_test_metrics.csv  (soundscape, test split only)

Each CSV has metric names as the (unnamed) first column / index, and one
column per dataset/task combination.

There are two *experiments*, each with its own metric set (ground truth is a
single point value for synthetic mixtures -> point-estimate metrics; ground
truth is a range for soundscapes -> range-aware metrics). The synthetic
mixture experiment has both a test and a validation split; soundscape
currently has a test split only.

Column naming convention (applies to ALL sources, since the underlying
experiments can be multi-task for any of them):

    {dataset}_{task}

where `dataset` is assumed to never contain an underscore (e.g. "HSN",
"XCM"), and `task` is everything after the first underscore. `task` is
either just "reg"/"class" for single-task experiments, or an arbitrary
string ending in "_reg"/"_class" for multi-task experiments (e.g.
"polyphony_reg", "pitch_class"). We therefore split each column on its
*first* underscore to get dataset, and treat the remainder as the task
label as a whole (mirroring what used to be just the "head").

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

# Known task labels that should be sorted first (in this order) whenever
# present; any other task label (e.g. multi-task labels like "polyphony_reg")
# is appended afterwards, sorted alphabetically.
HEAD_ORDER = ["reg", "class"]
HEAD_LABEL = {"reg": "Regression", "class": "Classification"}

# Column convention "{dataset}_{task}": dataset is a single underscore-free
# token, task is everything else (task must still end in "reg" or "class").
COL_RE = re.compile(r"^(?P<dataset>[^_]+)_(?P<head>.+)$")

POINT_METRIC_INFO = {
    "mae": ("MAE", "lower"),
    "rmse": ("RMSE", "lower"),
    "accuracy": ("Accuracy", "higher"),
    "off_by_one_accuracy": ("Off-by-one Acc.", "higher"),
    "macro_f1": ("Macro F1", "higher"),
    "weighted_f1": ("Weighted F1", "higher"),
    "qwk": ("QWK", "higher"),
    "pearson_r": ("Pearson $r$", "higher"),
    "support": ("Support", None),
}
POINT_OVERVIEW_METRICS = ["mae", "accuracy", "off_by_one_accuracy", "qwk"]
POINT_DETAILED_METRICS = ["mae", "rmse", "accuracy", "off_by_one_accuracy", "qwk", "support"]

RANGE_METRIC_INFO = {
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
}
RANGE_OVERVIEW_METRICS = ["range_mae", "range_accuracy", "off_by_one_range_accuracy"]
RANGE_DETAILED_METRICS = [
    "range_mae",
    "range_mse",
    "range_accuracy",
    "off_by_one_range_accuracy",
    "mean_interval_width",
    "support",
]

# Experiments: each has its own metric set, and one or more splits (test/val).
# `source` keys used throughout the rest of the script are "{experiment}_{split}".
EXPERIMENTS = {
    "synthetic_mixture": {
        "exp_desc": "synthetic mixture",
        "metric_info": POINT_METRIC_INFO,
        "overview_metrics": POINT_OVERVIEW_METRICS,
        "detailed_metrics": POINT_DETAILED_METRICS,
        "splits": {
            "test": Path("eval_results") / "test_metrics.csv",
            "val": Path("val_eval_results") / "val_metrics.csv",
        },
    },
    "soundscape": {
        "exp_desc": "soundscape",
        "metric_info": RANGE_METRIC_INFO,
        "overview_metrics": RANGE_OVERVIEW_METRICS,
        "detailed_metrics": RANGE_DETAILED_METRICS,
        "splits": {
            "test": Path("scape_eval_results") / "soundscape_test_metrics.csv",
        },
    },
}

SPLIT_LABEL = {"test": "test", "val": "validation"}


def _build_sources():
    """Flatten EXPERIMENTS into a per-(experiment, split) SOURCES dict keyed
    by "{experiment}_{split}", each carrying its experiment's metric config
    plus its own csv path / description."""
    sources = {}
    for exp_name, exp_cfg in EXPERIMENTS.items():
        for split, rel_path in exp_cfg["splits"].items():
            key = f"{exp_name}_{split}"
            sources[key] = {
                "experiment": exp_name,
                "split": split,
                "csv_rel_path": rel_path,
                "table_desc": f"{exp_cfg['exp_desc']} ({SPLIT_LABEL[split]} split)",
                "metric_info": exp_cfg["metric_info"],
                "overview_metrics": exp_cfg["overview_metrics"],
                "detailed_metrics": exp_cfg["detailed_metrics"],
            }
    return sources


SOURCES = _build_sources()


# ---------------------------------------------------------------------------
# Head/task ordering & labeling helpers
# ---------------------------------------------------------------------------

def order_heads(heads):
    """Order head/task labels: known reg/class first (in HEAD_ORDER order),
    then any other (e.g. multi-task) labels sorted alphabetically."""
    heads = set(heads)
    known = [h for h in HEAD_ORDER if h in heads]
    other = sorted(h for h in heads if h not in HEAD_ORDER)
    return known + other


def head_label(head: str) -> str:
    if head in HEAD_LABEL:
        return HEAD_LABEL[head]
    return head.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_models(target_dir: Path):
    """A model dir is any subdirectory of target_dir that contains at least
    one of the expected result CSVs."""
    models = []
    for child in sorted(target_dir.iterdir()):
        if not child.is_dir():
            continue
        has_any = any((child / cfg["csv_rel_path"]).is_file() for cfg in SOURCES.values())
        if has_any:
            models.append(child.name)
    return models


def parse_column(col: str):
    """Return (dataset, head) for a column name, or None if unrecognized."""
    col = col.strip()
    m = COL_RE.match(col)
    if not m:
        return None
    dataset, head = m.group("dataset"), m.group("head")
    if not (head == "reg" or head == "class" or head.endswith("_reg") or head.endswith("_class")):
        return None
    return dataset, head


def load_long_df(target_dir: Path, models):
    """Return one tidy long dataframe with columns:
    model, source, dataset, head, metric, value

    `source` is "{experiment}_{split}" (e.g. "synthetic_mixture_val").
    `metric` and `head` values are the *raw* names as they appear in each
    source's CSV; interpretation (display name/direction) is looked up
    per-source via SOURCES[source]["metric_info"] / head_label().
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
                parsed = parse_column(col)
                if parsed is None:
                    print(f"  [warn] skipping unrecognized column '{col}' in {csv_path}")
                    continue
                dataset, head = parsed
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
    return text.replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")


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
    happens when there is only a single unit to average over).
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


def metric_header_cells(metric_info: dict, metrics: list) -> list:
    cells = []
    for metric in metrics:
        label, direction = metric_info[metric]
        arrow = r"$\uparrow$" if direction == "higher" else r"$\downarrow$" if direction == "lower" else ""
        cells.append(f"{label} {arrow}".strip())
    return cells


# ---------------------------------------------------------------------------
# Overview table #1: averaged over datasets, one row per (model, head)
# ---------------------------------------------------------------------------

def build_overview_by_model_table(long_df: pd.DataFrame, source: str, models) -> str:
    cfg = SOURCES[source]
    metric_info = cfg["metric_info"]
    overview_metrics = cfg["overview_metrics"]

    sub = long_df[(long_df["source"] == source) & (long_df["metric"].isin(overview_metrics))]
    if sub.empty:
        return ""

    grouped = sub.groupby(["model", "head", "metric"])["value"]
    avg = grouped.mean().unstack("metric").reindex(columns=overview_metrics)
    std = grouped.std().unstack("metric").reindex(columns=overview_metrics)

    heads_present = order_heads(sub["head"].unique())

    n_cols = len(overview_metrics)
    col_spec = "ll" + "c" * n_cols
    header = "Backbone & Head & " + " & ".join(metric_header_cells(metric_info, overview_metrics)) + r" \\"

    best_masks = {
        metric: best_mask(avg[metric], metric_info[metric][1]) if metric in avg else None
        for metric in overview_metrics
    }

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    src_desc = cfg["table_desc"]
    lines.append(
        rf"\caption{{Results on {src_desc} data, averaged (mean $\pm$ std) across evaluation subsets, "
        rf"for each model and head/task type.}}"
    )
    lines.append(rf"\label{{tab:overview_by_model_{source}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")

    for model in models:
        model_heads = [h for h in heads_present if (model, h) in avg.index]
        if not model_heads:
            continue
        for i, head in enumerate(model_heads):
            row_vals = []
            for metric in overview_metrics:
                mean_val = avg.loc[(model, head), metric] if metric in avg.columns else np.nan
                std_val = std.loc[(model, head), metric] if metric in std.columns else np.nan
                cell = fmt_mean_std(mean_val, std_val)
                mask = best_masks.get(metric)
                if mask is not None and (model, head) in mask.index and mask.loc[(model, head)]:
                    cell = bold(cell)
                row_vals.append(cell)
            model_cell = rf"\multirow{{{len(model_heads)}}}{{*}}{{{escape_latex(model)}}}" if i == 0 else ""
            lines.append(f"{model_cell} & {escape_latex(head_label(head))} & " + " & ".join(row_vals) + r" \\")
        lines.append(r"\cmidrule(lr){1-2}")
    if lines[-1] == r"\cmidrule(lr){1-2}":
        lines.pop()
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Overview table #2: averaged over models, one row per (dataset, head)
# ---------------------------------------------------------------------------

def build_overview_by_dataset_table(long_df: pd.DataFrame, source: str) -> str:
    cfg = SOURCES[source]
    metric_info = cfg["metric_info"]
    overview_metrics = cfg["overview_metrics"]

    sub = long_df[(long_df["source"] == source) & (long_df["metric"].isin(overview_metrics))]
    if sub.empty:
        return ""

    grouped = sub.groupby(["dataset", "head", "metric"])["value"]
    avg = grouped.mean().unstack("metric").reindex(columns=overview_metrics)
    std = grouped.std().unstack("metric").reindex(columns=overview_metrics)

    heads_present = order_heads(sub["head"].unique())
    datasets = sorted(sub["dataset"].unique())

    n_cols = len(overview_metrics)
    col_spec = "ll" + "c" * n_cols
    header = "Head & Dataset & " + " & ".join(metric_header_cells(metric_info, overview_metrics)) + r" \\"

    src_desc = cfg["table_desc"]

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Results on {src_desc} data, averaged (mean $\pm$ std) across models, "
        rf"for each dataset and head/task type.}}"
    )
    lines.append(rf"\label{{tab:overview_by_dataset_{source}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(header)

    for head in heads_present:
        head_datasets = [d for d in datasets if (d, head) in avg.index]
        if not head_datasets:
            continue
        lines.append(r"\midrule")
        for i, dataset in enumerate(head_datasets):
            row_vals = []
            for metric in overview_metrics:
                mean_val = avg.loc[(dataset, head), metric] if metric in avg.columns else np.nan
                std_val = std.loc[(dataset, head), metric] if metric in std.columns else np.nan
                row_vals.append(fmt_mean_std(mean_val, std_val))
            head_cell = rf"\multirow{{{len(head_datasets)}}}{{*}}{{{escape_latex(head_label(head))}}}" if i == 0 else ""
            lines.append(f"{head_cell} & {escape_latex(dataset)} & " + " & ".join(row_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Overview table #2b: averaged over both models and heads, one row per dataset
# ---------------------------------------------------------------------------

def build_overview_by_dataset_all_heads_table(long_df: pd.DataFrame, source: str) -> str:
    cfg = SOURCES[source]
    metric_info = cfg["metric_info"]
    overview_metrics = cfg["overview_metrics"]

    sub = long_df[(long_df["source"] == source) & (long_df["metric"].isin(overview_metrics))]
    if sub.empty:
        return ""

    grouped = sub.groupby(["dataset", "metric"])["value"]
    avg = grouped.mean().unstack("metric").reindex(columns=overview_metrics)
    std = grouped.std().unstack("metric").reindex(columns=overview_metrics)

    datasets = sorted(sub["dataset"].unique())

    n_cols = len(overview_metrics)
    col_spec = "l" + "c" * n_cols
    header = "Dataset & " + " & ".join(metric_header_cells(metric_info, overview_metrics)) + r" \\"

    best_masks = {
        metric: best_mask(avg[metric], metric_info[metric][1]) if metric in avg else None
        for metric in overview_metrics
    }

    src_desc = cfg["table_desc"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Results on {src_desc} data, averaged (mean $\pm$ std) across both models and "
        rf"head/task types, for each dataset.}}"
    )
    lines.append(rf"\label{{tab:overview_by_dataset_all_heads_{source}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")

    for dataset in datasets:
        if dataset not in avg.index:
            continue
        row_vals = []
        for metric in overview_metrics:
            mean_val = avg.loc[dataset, metric] if metric in avg.columns else np.nan
            std_val = std.loc[dataset, metric] if metric in std.columns else np.nan
            cell = fmt_mean_std(mean_val, std_val)
            mask = best_masks.get(metric)
            if mask is not None and dataset in mask.index and mask.loc[dataset]:
                cell = bold(cell)
            row_vals.append(cell)
        lines.append(f"{escape_latex(dataset)} & " + " & ".join(row_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Overview table #3: averaged over both models and datasets, one row per head
# ---------------------------------------------------------------------------

def build_overview_grand_table(long_df: pd.DataFrame, source: str) -> str:
    cfg = SOURCES[source]
    metric_info = cfg["metric_info"]
    overview_metrics = cfg["overview_metrics"]

    sub = long_df[(long_df["source"] == source) & (long_df["metric"].isin(overview_metrics))]
    if sub.empty:
        return ""

    grouped = sub.groupby(["head", "metric"])["value"]
    avg = grouped.mean().unstack("metric").reindex(columns=overview_metrics)
    std = grouped.std().unstack("metric").reindex(columns=overview_metrics)

    heads_present = order_heads(sub["head"].unique())

    n_cols = len(overview_metrics)
    col_spec = "l" + "c" * n_cols
    header = "Head & " + " & ".join(metric_header_cells(metric_info, overview_metrics)) + r" \\"

    best_masks = {
        metric: best_mask(avg[metric], metric_info[metric][1]) if metric in avg else None
        for metric in overview_metrics
    }

    src_desc = cfg["table_desc"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Results on {src_desc} data, averaged (mean $\pm$ std) across both models and "
        rf"evaluation subsets, for each head/task type.}}"
    )
    lines.append(rf"\label{{tab:overview_grand_{source}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")

    for head in heads_present:
        if head not in avg.index:
            continue
        row_vals = []
        for metric in overview_metrics:
            mean_val = avg.loc[head, metric] if metric in avg.columns else np.nan
            std_val = std.loc[head, metric] if metric in std.columns else np.nan
            cell = fmt_mean_std(mean_val, std_val)
            mask = best_masks.get(metric)
            if mask is not None and head in mask.index and mask.loc[head]:
                cell = bold(cell)
            row_vals.append(cell)
        lines.append(f"{escape_latex(head_label(head))} & " + " & ".join(row_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detailed table (per model, per source: rows = head x dataset, single model)
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

    heads_present = order_heads(sub["head"].unique())

    n_cols = len(detailed_metrics)
    col_spec = "ll" + "c" * n_cols
    header = "Head & Dataset & " + " & ".join(metric_header_cells(metric_info, detailed_metrics)) + r" \\"

    src_desc = cfg["table_desc"]
    label_safe = re.sub(r"[^a-zA-Z0-9]+", "_", model.lower())

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(
        rf"\caption{{Results on {src_desc} data for {escape_latex(model)}, per dataset and head/task type.}}"
    )
    lines.append(rf"\label{{tab:supp_{source}_{label_safe}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(header)

    for head in heads_present:
        if head not in pivot.index.get_level_values("head"):
            continue
        lines.append(r"\midrule")
        datasets = sorted(pivot.loc[head].index)
        for i, dataset in enumerate(datasets):
            row_vals = [fmt(pivot.loc[(head, dataset), metric]) for metric in detailed_metrics]
            head_cell = rf"\multirow{{{len(datasets)}}}{{*}}{{{escape_latex(head_label(head))}}}" if i == 0 else ""
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


def source_section_title(source: str) -> str:
    cfg = SOURCES[source]
    exp_title = cfg["experiment"].replace("_", " ").title()
    return f"{exp_title} \u2014 {SPLIT_LABEL[cfg['split']].capitalize()} split"


def build_document(target_dir: Path) -> str:
    models = discover_models(target_dir)
    if not models:
        raise RuntimeError(f"No model subdirectories with result CSVs found under {target_dir}")
    print(f"Found {len(models)} model(s): {models}")

    long_df = load_long_df(target_dir, models)

    parts = [DOC_PREAMBLE]

    parts.append("\\section*{Overview}\n")
    for source in SOURCES:
        if long_df[long_df["source"] == source].empty:
            continue
        parts.append(f"\\subsection*{{{source_section_title(source)}}}\n")

        t1 = build_overview_by_model_table(long_df, source, models)
        if t1:
            parts.append(t1)
            parts.append("\n")

        t2 = build_overview_by_dataset_table(long_df, source)
        if t2:
            parts.append(t2)
            parts.append("\n")

        t2b = build_overview_by_dataset_all_heads_table(long_df, source)
        if t2b:
            parts.append(t2b)
            parts.append("\n")

        t3 = build_overview_grand_table(long_df, source)
        if t3:
            parts.append(t3)
            parts.append("\n")

    parts.append("\\clearpage\n\\section*{Detailed results}\n")
    for source in SOURCES:
        if long_df[long_df["source"] == source].empty:
            continue
        parts.append(f"\\subsection*{{{source_section_title(source)}}}\n")
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