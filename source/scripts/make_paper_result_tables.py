#!/usr/bin/env python3
"""
Build the six paper-ready LaTeX result-table files from archive/ data:

    results/result_tables_synthetic_test.tex     (Pooled-Embeddings, synthetic mixture test split)
    results/results_tables_synthetic_val.tex     (Pooled-Embeddings, synthetic mixture val split)
    results/result_tables_soundscape.tex         (Pooled-Embeddings, soundscape test split)
    results/result_tables_multi_task.tex         (Spatial-Embeddings, synthetic test + soundscape test)
    results/result_tables_fine_tune.tex          (Fine-Tuning, synthetic test + soundscape test)
    results/results_tables_XCM_generalization.tex (XCM-Generalization vs. fine-tune vs. Pooled-
                                                    Embeddings "region-specific" baseline, soundscape
                                                    test only)

This reuses make_result_tables.py's CSV discovery/parsing (SOURCES,
load_long_df) and backbone_meta.py's canonical backbone display names /
fine-tuning name mapping as the single source of truth for which backbones
appear and what they're called. QWK is dropped from every table; best (and
tied-best) values are bolded per metric column in every averaged/summary
table (not in raw per-model/per-grouping detail dumps, where there's no
single meaningful "best" to highlight across heterogeneous rows).

Usage:
    complete-venv/bin/python source/scripts/make_paper_result_tables.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import BACKBONE_META, FINETUNE_NAME_TO_CANONICAL, ordered_backbones
from plot_data import STUDY_DIRS, REGIONAL_DATASETS, ALL_DATASETS

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"

DOC_PREAMBLE = mrt.DOC_PREAMBLE
DOC_END = mrt.DOC_END

fmt = mrt.fmt
fmt_mean_std = mrt.fmt_mean_std
bold = mrt.bold
best_mask = mrt.best_mask
escape_latex = mrt.escape_latex
metric_header_cells = mrt.metric_header_cells


# ---------------------------------------------------------------------------
# Backbone name resolution (backbone_meta.py is the source of truth)
# ---------------------------------------------------------------------------

def canonical_key(raw_name):
    """Map an archive folder name to its canonical BACKBONE_META key, or
    None if it isn't a backbone we report (e.g. perch_v2/perch_v2_old_lr,
    which backbone_meta.py intentionally excludes from fine-tuning)."""
    if raw_name in BACKBONE_META:
        return raw_name
    return FINETUNE_NAME_TO_CANONICAL.get(raw_name)


def display_name(raw_name):
    key = canonical_key(raw_name)
    return None if key is None else BACKBONE_META[key]["display"]


_BACKBONE_ORDER = ordered_backbones()
FINETUNE_CANONICAL = set(FINETUNE_NAME_TO_CANONICAL.values())  # the 4 fine-tuned backbones


def filter_and_order_models(raw_models, allowed_canonical=None):
    """Drop any raw model name with no canonical backbone_meta.py entry
    (e.g. the stray perch_v2 variants), optionally restrict to a given set
    of canonical keys, and order by backbone_meta's fixed domain-based
    backbone order (reused across every figure in this project)."""
    kept = []
    for m in raw_models:
        key = canonical_key(m)
        if key is None:
            continue
        if allowed_canonical is not None and key not in allowed_canonical:
            continue
        kept.append(m)
    return sorted(kept, key=lambda m: _BACKBONE_ORDER.index(canonical_key(m)))


# ---------------------------------------------------------------------------
# Metric config (same as make_result_tables.py's SOURCES, minus QWK)
# ---------------------------------------------------------------------------

def cfg_no_qwk(source_key):
    cfg = dict(mrt.SOURCES[source_key])
    cfg["overview_metrics"] = [m for m in cfg["overview_metrics"] if m != "qwk"]
    cfg["detailed_metrics"] = [m for m in cfg["detailed_metrics"] if m != "qwk"]
    return cfg


# ---------------------------------------------------------------------------
# Simple head spec (Regression / Classification) -- pooled, fine-tune, soundscape
# ---------------------------------------------------------------------------

class HeadSpec:
    def __init__(self, order_fn, label_fn, column_label="Head"):
        self.order_fn = order_fn
        self.label_fn = label_fn
        self.column_label = column_label


SIMPLE_HEAD_SPEC = HeadSpec(order_fn=mrt.order_heads, label_fn=mrt.head_label, column_label="Head")


# ---------------------------------------------------------------------------
# Multi-task head spec (main task x auxiliary objective) -- spatial embeddings
# ---------------------------------------------------------------------------
# Column task suffixes as they appear in archive/Spatial-Embeddings CSVs
# (see plot_rq2.py's CONFIG_HEAD, which selects the same "main-task-output"
# columns for its own figures): (main_order, aux_order, main_label, aux_label)

MULTITASK_HEAD_INFO = {
    "reg_reg": (0, 0, "Regression", "---"),
    "reg_and_class_reg": (0, 1, "Regression", "polyphony classification"),
    "reg_and_events_reg": (0, 2, "Regression", "event logits"),
    "reg_and_frame_reg_reg": (0, 3, "Regression", "framewise polyphony"),
    "reg_and_events_and_frame_reg_reg": (0, 4, "Regression", "event logits, framewise polyphony"),
    "class_class": (1, 0, "Classification", "---"),
    "class_and_events_class": (1, 2, "Classification", "event logits"),
    "class_and_frame_class_class": (1, 3, "Classification", "framewise polyphony"),
    "class_and_events_and_frame_class_class": (1, 4, "Classification", "event logits, framewise polyphony"),
}


def multitask_order_heads(heads):
    known = [h for h in MULTITASK_HEAD_INFO if h in set(heads)]
    return sorted(known, key=lambda h: MULTITASK_HEAD_INFO[h][:2])


def multitask_combined_label(h):
    info = MULTITASK_HEAD_INFO.get(h)
    if info is None:
        return h
    main, aux = info[2], info[3]
    return main if aux == "---" else f"{main} + {aux}"


MULTITASK_HEAD_SPEC = HeadSpec(order_fn=multitask_order_heads, label_fn=multitask_combined_label,
                                column_label="Head/Task")


# ---------------------------------------------------------------------------
# Generic table builders
# ---------------------------------------------------------------------------

def _filter(long_df, source, models, metrics, dataset_filter=None, head_whitelist=None):
    sub = long_df[
        (long_df["source"] == source)
        & long_df["metric"].isin(metrics)
        & long_df["model"].isin(models)
    ]
    if dataset_filter is not None:
        sub = sub[sub["dataset"].isin(dataset_filter)]
    if head_whitelist is not None:
        sub = sub[sub["head"].isin(head_whitelist)]
    return sub


def build_by_model_table(long_df, source, cfg, models, caption, label, head_spec=SIMPLE_HEAD_SPEC,
                          dataset_filter=None):
    """Averaged across evaluation subsets, one row per (model, head)."""
    metric_info, overview_metrics = cfg["metric_info"], cfg["overview_metrics"]
    head_whitelist = set(MULTITASK_HEAD_INFO) if head_spec is MULTITASK_HEAD_SPEC else None
    sub = _filter(long_df, source, models, overview_metrics, dataset_filter, head_whitelist)
    if sub.empty:
        return ""

    grouped = sub.groupby(["model", "head", "metric"])["value"]
    avg = grouped.mean().unstack("metric").reindex(columns=overview_metrics)
    std = grouped.std().unstack("metric").reindex(columns=overview_metrics)
    best_masks = {m: best_mask(avg[m], metric_info[m][1]) if m in avg else None for m in overview_metrics}

    n_cols = len(overview_metrics)
    col_spec = "ll" + "c" * n_cols
    header = f"Backbone & {head_spec.column_label} & " + " & ".join(metric_header_cells(metric_info, overview_metrics)) + r" \\"

    lines = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
              r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule", header, r"\midrule"]

    for model in models:
        model_sub = sub[sub["model"] == model]
        if model_sub.empty:
            continue
        heads_here = head_spec.order_fn(model_sub["head"].unique())
        for i, head in enumerate(heads_here):
            row_vals = []
            for metric in overview_metrics:
                mean_val = avg.loc[(model, head), metric] if (model, head) in avg.index and metric in avg.columns else np.nan
                std_val = std.loc[(model, head), metric] if (model, head) in std.index and metric in std.columns else np.nan
                cell = fmt_mean_std(mean_val, std_val)
                mask = best_masks.get(metric)
                if mask is not None and (model, head) in mask.index and mask.loc[(model, head)]:
                    cell = bold(cell)
                row_vals.append(cell)
            model_cell = rf"\multirow{{{len(heads_here)}}}{{*}}{{{escape_latex(display_name(model))}}}" if i == 0 else ""
            lines.append(f"{model_cell} & {escape_latex(head_spec.label_fn(head))} & " + " & ".join(row_vals) + r" \\")
        lines.append(r"\cmidrule(lr){1-2}")
    if lines[-1] == r"\cmidrule(lr){1-2}":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"]
    return "\n".join(lines)


def build_by_dataset_grand_table(long_df, source, cfg, models, caption, label, dataset_filter=None):
    """Averaged across both models and head/task types, one row per dataset."""
    metric_info, overview_metrics = cfg["metric_info"], cfg["overview_metrics"]
    sub = _filter(long_df, source, models, overview_metrics, dataset_filter)
    if sub.empty:
        return ""

    grouped = sub.groupby(["dataset", "metric"])["value"]
    avg = grouped.mean().unstack("metric").reindex(columns=overview_metrics)
    std = grouped.std().unstack("metric").reindex(columns=overview_metrics)
    datasets = sorted(sub["dataset"].unique())
    best_masks = {m: best_mask(avg[m], metric_info[m][1]) if m in avg else None for m in overview_metrics}

    n_cols = len(overview_metrics)
    col_spec = "l" + "c" * n_cols
    header = "Dataset & " + " & ".join(metric_header_cells(metric_info, overview_metrics)) + r" \\"

    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
              rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule", header, r"\midrule"]
    for dataset in datasets:
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
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_detailed_tables(long_df, source, cfg, models, caption_fn, label_fn, head_spec=SIMPLE_HEAD_SPEC,
                           dataset_filter=None):
    """One table per model: rows = head/task type x dataset, raw values (no averaging across
    models, no bolding -- there is no single meaningful "best" across heterogeneous rows here)."""
    metric_info, detailed_metrics = cfg["metric_info"], cfg["detailed_metrics"]
    head_whitelist = set(MULTITASK_HEAD_INFO) if head_spec is MULTITASK_HEAD_SPEC else None
    tables = []
    for model in models:
        sub = _filter(long_df, source, [model], detailed_metrics, dataset_filter, head_whitelist)
        if sub.empty:
            continue
        pivot = (
            sub.groupby(["head", "dataset", "metric"])["value"]
            .mean()
            .unstack("metric")
            .reindex(columns=detailed_metrics)
        )
        heads_here = head_spec.order_fn(sub["head"].unique())

        n_cols = len(detailed_metrics)
        col_spec = "ll" + "c" * n_cols
        header = f"{head_spec.column_label} & Dataset & " + " & ".join(metric_header_cells(metric_info, detailed_metrics)) + r" \\"

        lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", rf"\caption{{{caption_fn(model)}}}",
                  rf"\label{{{label_fn(model)}}}", r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{col_spec}}}",
                  r"\toprule", header]
        for head in heads_here:
            if head not in pivot.index.get_level_values("head"):
                continue
            lines.append(r"\midrule")
            datasets = sorted(pivot.loc[head].index)
            for i, dataset in enumerate(datasets):
                row_vals = [fmt(pivot.loc[(head, dataset), metric]) for metric in detailed_metrics]
                head_cell = rf"\multirow{{{len(datasets)}}}{{*}}{{{escape_latex(head_spec.label_fn(head))}}}" if i == 0 else ""
                lines.append(f"{head_cell} & {escape_latex(dataset)} & " + " & ".join(row_vals) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"]
        tables.append("\n".join(lines))
    return tables


def build_multitask_aux_grand_table(long_df, source, cfg, models, caption, label, dataset_filter=None):
    """Averaged across both models and evaluation subsets, one row per (main head, auxiliary
    objective) -- the "Additionally" table for spatial-embeddings, matching the reference layout."""
    metric_info, overview_metrics = cfg["metric_info"], cfg["overview_metrics"]
    sub = _filter(long_df, source, models, overview_metrics, dataset_filter, set(MULTITASK_HEAD_INFO))
    if sub.empty:
        return ""

    grouped = sub.groupby(["head", "metric"])["value"]
    avg = grouped.mean().unstack("metric").reindex(columns=overview_metrics)
    std = grouped.std().unstack("metric").reindex(columns=overview_metrics)
    heads_here = multitask_order_heads(sub["head"].unique())
    best_masks = {m: best_mask(avg[m], metric_info[m][1]) if m in avg else None for m in overview_metrics}

    n_cols = len(overview_metrics)
    col_spec = "ll" + "c" * n_cols
    header = "Head & Auxiliary objectives & " + " & ".join(metric_header_cells(metric_info, overview_metrics)) + r" \\"

    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
              rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule", header, r"\midrule"]

    from itertools import groupby
    for main_order, group in groupby(heads_here, key=lambda h: MULTITASK_HEAD_INFO[h][0]):
        group = list(group)
        for i, head in enumerate(group):
            _, _, main_label, aux_label = MULTITASK_HEAD_INFO[head]
            row_vals = []
            for metric in overview_metrics:
                mean_val = avg.loc[head, metric] if head in avg.index and metric in avg.columns else np.nan
                std_val = std.loc[head, metric] if head in std.index and metric in std.columns else np.nan
                cell = fmt_mean_std(mean_val, std_val)
                mask = best_masks.get(metric)
                if mask is not None and head in mask.index and mask.loc[head]:
                    cell = bold(cell)
                row_vals.append(cell)
            main_cell = rf"\multirow{{{len(group)}}}{{*}}{{{main_label.lower()}}}" if i == 0 else ""
            lines.append(f"{main_cell} & {escape_latex(aux_label)} & " + " & ".join(row_vals) + r" \\")
        lines.append(r"\cmidrule(lr){1-2}")
    if lines[-1] == r"\cmidrule(lr){1-2}":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File 1-3: Pooled-Embeddings (synthetic test / synthetic val / soundscape)
# ---------------------------------------------------------------------------

def build_pooled_section(long_df, source, cfg, models, split_desc, tag):
    parts = []
    t1 = build_by_model_table(
        long_df, source, cfg, models,
        caption=rf"Results on {split_desc} data, averaged (mean $\pm$ std) across evaluation subsets, "
                rf"for each model and head/task type.",
        label=rf"tab:overview_by_model_{tag}",
    )
    if t1:
        parts.append(t1)
    t2 = build_by_dataset_grand_table(
        long_df, source, cfg, models,
        caption=rf"Results on {split_desc} data, averaged (mean $\pm$ std) across both models and "
                rf"head/task types, for each dataset.",
        label=rf"tab:overview_by_dataset_{tag}",
    )
    if t2:
        parts.append(t2)
    return parts


def build_pooled_detailed(long_df, source, cfg, models, split_desc, tag):
    return build_detailed_tables(
        long_df, source, cfg, models,
        caption_fn=lambda m: rf"Results on {split_desc} data for {escape_latex(display_name(m))}, "
                              rf"per dataset and head/task type.",
        label_fn=lambda m: rf"tab:supp_{tag}_{canonical_key(m).lower().replace('-', '_')}",
    )


def write_pooled_file(long_df, source, split_desc, section_title, tag, out_name):
    models = filter_and_order_models(mrt.discover_models(STUDY_DIRS["pooled"]))
    cfg = cfg_no_qwk(source)

    doc = [DOC_PREAMBLE, r"\section*{Overview}", "", rf"\subsection*{{{section_title}}}", ""]
    doc += [p + "\n" for p in build_pooled_section(long_df, source, cfg, models, split_desc, tag)]
    doc.append(r"\clearpage" + "\n" + r"\section*{Detailed results}" + "\n")
    doc.append(rf"\subsection*{{{section_title}}}" + "\n")
    doc += [p + "\n" for p in build_pooled_detailed(long_df, source, cfg, models, split_desc, tag)]
    doc.append(DOC_END)

    out_path = RESULTS_DIR / out_name
    out_path.write_text("\n".join(doc))
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# File 4: Spatial-Embeddings (multi-task) -- synthetic test + soundscape test
# ---------------------------------------------------------------------------

def write_multitask_file():
    models = filter_and_order_models(mrt.discover_models(STUDY_DIRS["spatial"]))
    long_df = mrt.load_long_df(STUDY_DIRS["spatial"], models)

    sections = [
        ("synthetic_mixture_test", "synthetic mixture (test split)", "Synthetic Mixture — Test split",
         "multitask_synthetic_test", None),
        ("soundscape_test", "soundscape (test split)", "Soundscape — Test split",
         "multitask_soundscape_test", REGIONAL_DATASETS),
    ]

    overview_parts, detailed_parts = [], []
    for source, split_desc, section_title, tag, dataset_filter in sections:
        cfg = cfg_no_qwk(source)
        overview_parts.append(f"\\subsection*{{{section_title}}}\n")

        t1 = build_by_model_table(
            long_df, source, cfg, models,
            caption=rf"Results on {split_desc} data, averaged (mean $\pm$ std) across evaluation subsets, "
                    rf"for each model and head/task type.",
            label=rf"tab:overview_by_model_{tag}",
            head_spec=MULTITASK_HEAD_SPEC, dataset_filter=dataset_filter,
        )
        if t1:
            overview_parts.append(t1 + "\n")

        t2 = build_by_dataset_grand_table(
            long_df, source, cfg, models,
            caption=rf"Results on {split_desc} data, averaged (mean $\pm$ std) across both models and "
                    rf"head/task types, for each dataset.",
            label=rf"tab:overview_by_dataset_{tag}",
            dataset_filter=dataset_filter,
        )
        if t2:
            overview_parts.append(t2 + "\n")

        t3 = build_multitask_aux_grand_table(
            long_df, source, cfg, models,
            caption=rf"Results on {split_desc} data, averaged (mean $\pm$ std) across both models and "
                    rf"evaluation subsets, for each head/task type.",
            label=rf"tab:overview_grand_{tag}",
            dataset_filter=dataset_filter,
        )
        if t3:
            overview_parts.append(t3 + "\n")

        detailed_parts.append(f"\\subsection*{{{section_title}}}\n")
        detailed = build_detailed_tables(
            long_df, source, cfg, models,
            caption_fn=lambda m, split_desc=split_desc: rf"Results on {split_desc} data for "
                        rf"{escape_latex(display_name(m))}, per dataset and head/task type.",
            label_fn=lambda m, tag=tag: rf"tab:supp_{tag}_{canonical_key(m).lower().replace('-', '_')}",
            head_spec=MULTITASK_HEAD_SPEC, dataset_filter=dataset_filter,
        )
        detailed_parts += [p + "\n" for p in detailed]

    doc = [DOC_PREAMBLE, r"\section*{Overview}", ""] + overview_parts
    doc.append(r"\clearpage" + "\n" + r"\section*{Detailed results}" + "\n")
    doc += detailed_parts
    doc.append(DOC_END)

    out_path = RESULTS_DIR / "result_tables_multi_task.tex"
    out_path.write_text("\n".join(doc))
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# File 5: Fine-Tuning -- synthetic test + soundscape test
# ---------------------------------------------------------------------------

def write_fine_tune_file():
    raw_models = mrt.discover_models(STUDY_DIRS["finetune"])
    models = filter_and_order_models(raw_models, allowed_canonical=FINETUNE_CANONICAL)
    long_df = mrt.load_long_df(STUDY_DIRS["finetune"], models)

    sections = [
        ("synthetic_mixture_test", "synthetic mixture (test split)", "Synthetic Mixture — Test split",
         "finetune_synthetic_test"),
        ("soundscape_test", "soundscape (test split)", "Soundscape — Test split", "finetune_soundscape_test"),
    ]

    overview_parts, detailed_parts = [], []
    for source, split_desc, section_title, tag in sections:
        cfg = cfg_no_qwk(source)
        overview_parts.append(f"\\subsection*{{{section_title}}}\n")
        section_tables = build_pooled_section(long_df, source, cfg, models, split_desc, tag)
        if section_tables:
            overview_parts += [p + "\n" for p in section_tables]
        else:
            overview_parts.append(
                r"\textit{No " + split_desc + " evaluation data is available for the fine-tuned "
                r"backbones (AudioProtoPNet, Bird-MAE, EfficientNet-B1, NatureLM-audio); only the "
                r"excluded perch\_v2 variants have results for this split (see backbone\_meta.py).}"
                "\n"
            )

        detailed_parts.append(f"\\subsection*{{{section_title}}}\n")
        detailed = build_pooled_detailed(long_df, source, cfg, models, split_desc, tag)
        detailed_parts += [p + "\n" for p in detailed]

    doc = [DOC_PREAMBLE, r"\section*{Overview}", ""] + overview_parts
    doc.append(r"\clearpage" + "\n" + r"\section*{Detailed results}" + "\n")
    doc += detailed_parts
    doc.append(DOC_END)

    out_path = RESULTS_DIR / "result_tables_fine_tune.tex"
    out_path.write_text("\n".join(doc))
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# File 6: XCM-Generalization -- frozen vs. fine-tuned vs. region-specific,
# soundscape test only, restricted to the 4 fine-tuning backbones.
# ---------------------------------------------------------------------------

GROUPINGS = ["Frozen", "Fine-tuned", "Region-specific"]


def load_xcm_grouped_long_df():
    frames = []

    frozen_models = [m for m in mrt.discover_models(STUDY_DIRS["xcm_gen"]) if canonical_key(m) in FINETUNE_CANONICAL]
    df = mrt.load_long_df(STUDY_DIRS["xcm_gen"], frozen_models)
    df = df[df["source"] == "soundscape_test"].copy()
    df["grouping"] = "Frozen"
    df["canon"] = df["model"].map(canonical_key)
    frames.append(df)

    ft_models = [m for m in mrt.discover_models(STUDY_DIRS["xcm_gen_ft"]) if canonical_key(m) in FINETUNE_CANONICAL]
    df = mrt.load_long_df(STUDY_DIRS["xcm_gen_ft"], ft_models)
    df = df[df["source"] == "soundscape_test"].copy()
    df["grouping"] = "Fine-tuned"
    df["canon"] = df["model"].map(canonical_key)
    frames.append(df)

    region_models = [m for m in mrt.discover_models(STUDY_DIRS["pooled"]) if canonical_key(m) in FINETUNE_CANONICAL]
    df = mrt.load_long_df(STUDY_DIRS["pooled"], region_models)
    df = df[(df["source"] == "soundscape_test") & (df["dataset"].isin(REGIONAL_DATASETS)) & (df["head"] == "reg")].copy()
    df["grouping"] = "Region-specific"
    df["canon"] = df["model"].map(canonical_key)
    frames.append(df)

    return pd.concat(frames, ignore_index=True)


def build_xcm_grouping_avg_table(long_df, cfg, caption, label):
    metric_info, overview_metrics = cfg["metric_info"], cfg["overview_metrics"]
    sub = long_df[long_df["metric"].isin(overview_metrics)]
    grouped = sub.groupby(["grouping", "metric"])["value"]
    avg = grouped.mean().unstack("metric").reindex(columns=overview_metrics)
    std = grouped.std().unstack("metric").reindex(columns=overview_metrics)
    best_masks = {m: best_mask(avg[m], metric_info[m][1]) if m in avg else None for m in overview_metrics}

    n_cols = len(overview_metrics)
    col_spec = "l" + "c" * n_cols
    header = "Model grouping & " + " & ".join(metric_header_cells(metric_info, overview_metrics)) + r" \\"

    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
              rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule", header, r"\midrule"]
    for grouping in GROUPINGS:
        if grouping not in avg.index:
            continue
        row_vals = []
        for metric in overview_metrics:
            mean_val = avg.loc[grouping, metric] if metric in avg.columns else np.nan
            std_val = std.loc[grouping, metric] if metric in std.columns else np.nan
            cell = fmt_mean_std(mean_val, std_val)
            mask = best_masks.get(metric)
            if mask is not None and grouping in mask.index and mask.loc[grouping]:
                cell = bold(cell)
            row_vals.append(cell)
        lines.append(f"{grouping} & " + " & ".join(row_vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_xcm_detailed_table(long_df, cfg, grouping, caption, label):
    metric_info, detailed_metrics = cfg["metric_info"], cfg["detailed_metrics"]
    sub = long_df[(long_df["grouping"] == grouping) & long_df["metric"].isin(detailed_metrics)]
    if sub.empty:
        return ""
    pivot = (
        sub.groupby(["canon", "dataset", "metric"])["value"]
        .mean()
        .unstack("metric")
        .reindex(columns=detailed_metrics)
    )
    canons = [c for c in _BACKBONE_ORDER if c in pivot.index.get_level_values("canon")]

    n_cols = len(detailed_metrics)
    col_spec = "ll" + "c" * n_cols
    header = "Backbone & Dataset & " + " & ".join(metric_header_cells(metric_info, detailed_metrics)) + r" \\"

    lines = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
              r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule", header, r"\midrule"]
    for canon in canons:
        datasets = sorted(pivot.loc[canon].index)
        for i, dataset in enumerate(datasets):
            row_vals = [fmt(pivot.loc[(canon, dataset), metric]) for metric in detailed_metrics]
            model_cell = rf"\multirow{{{len(datasets)}}}{{*}}{{{escape_latex(BACKBONE_META[canon]['display'])}}}" if i == 0 else ""
            lines.append(f"{model_cell} & {escape_latex(dataset)} & " + " & ".join(row_vals) + r" \\")
        lines.append(r"\cmidrule(lr){1-2}")
    if lines[-1] == r"\cmidrule(lr){1-2}":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"]
    return "\n".join(lines)


def write_xcm_generalization_file():
    long_df = load_xcm_grouped_long_df()
    cfg = cfg_no_qwk("soundscape_test")

    doc = [DOC_PREAMBLE, r"\section*{Overview}", "", r"\subsection*{Soundscape — Test split}", ""]
    doc.append(build_xcm_grouping_avg_table(
        long_df, cfg,
        caption=r"Results on soundscape (test split) data, averaged (mean $\pm$ std) across evaluation "
                r"datasets and models, for each model grouping (frozen, fine-tuned, region-specific).",
        label="tab:xcm_grouping_avg_soundscape_test",
    ) + "\n")

    doc.append(r"\clearpage" + "\n" + r"\section*{Detailed results}" + "\n")
    doc.append(r"\subsection*{Soundscape — Test split}" + "\n")
    grouping_slug = {"Frozen": "frozen", "Fine-tuned": "fine_tuned", "Region-specific": "region_specific"}
    for grouping in GROUPINGS:
        table = build_xcm_detailed_table(
            long_df, cfg, grouping,
            caption=rf"Results on soundscape (test split) data for the {grouping.lower()} models, "
                    rf"per backbone and evaluation dataset.",
            label=rf"tab:xcm_detailed_{grouping_slug[grouping]}_soundscape_test",
        )
        if table:
            doc.append(table + "\n")
    doc.append(DOC_END)

    out_path = RESULTS_DIR / "results_tables_XCM_generalization.tex"
    out_path.write_text("\n".join(doc))
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pooled_models = filter_and_order_models(mrt.discover_models(STUDY_DIRS["pooled"]))
    pooled_long_df = mrt.load_long_df(STUDY_DIRS["pooled"], pooled_models)

    write_pooled_file(pooled_long_df, "synthetic_mixture_test", "synthetic mixture (test split)",
                       "Synthetic Mixture — Test split", "synthetic_test", "result_tables_synthetic_test.tex")
    write_pooled_file(pooled_long_df, "synthetic_mixture_val", "synthetic mixture (validation split)",
                       "Synthetic Mixture — Validation split", "synthetic_val", "results_tables_synthetic_val.tex")
    write_pooled_file(pooled_long_df, "soundscape_test", "soundscape (test split)",
                       "Soundscape — Test split", "soundscape_test", "result_tables_soundscape.tex")

    write_multitask_file()
    write_fine_tune_file()
    write_xcm_generalization_file()


if __name__ == "__main__":
    main()
