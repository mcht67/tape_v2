#!/usr/bin/env python3
"""
RQ5 (additional) -- Macro-MAE(1-6) summary for the backbone-matched
soundscape comparison (region-specific vs. XCM head frozen vs. XCM head
fine-tuned), 4 backbones (EfficientNet-B1, AudioProtoPNet, Bird-MAE,
NatureLM-audio; Perch v2 excluded -- see FINETUNE_NAME_TO_CANONICAL's
comment) x 7 REGIONAL_DATASETS.

This is the checked-in source for two things that previously existed only
as an ad hoc, unsaved computation: (1) plot_rq5.py's
rq5_panel_a_soundscape_matched_macro_mae figure's per-dataset lines, via
the same compute_macro_mae_long() (plot_rq1.py's _region_macro_mae_1_6()
applied per dataset instead of averaged across all 7 regions into one
per-model number, see that function's docstring for the macro-MAE(1-6)
definition and its unambiguous-label/level-1-6 restriction), and (2) the
pooled (28-cell) macro-MAE(1-6) mean +/- SD per configuration cited in
results/rq5_results.tex's Table 1 -- both are recomputed here from
scratch rather than assumed, so both are reproducible from this script
alone.

Also runs the same cell-level paired comparisons (paired t-test, Wilcoxon
signed-rank, exact sign test) already established for range_mae in
compute_rq5_frozen_vs_region_4backbone.py and
compute_rq5_finetune_vs_frozen.py (reusing those functions directly), but
on macro-MAE(1-6) instead -- filling the gap flagged in
results/rq5_results.tex ("neither difference has been tested for
statistical significance on macro-MAE specifically").

Purpose of running this at all: rq5_kurtosis.md already reports the
equivalent range_mae pooled numbers (region-specific 0.903 +/- 0.567,
XCM frozen 1.022 +/- 0.699, XCM fine-tuned 0.938 +/- 0.488, n=28 throughout)
and finds region-specific best, frozen-XCM worst. Under macro-MAE(1-6) --
which weights all 6 polyphony levels equally rather than being dominated
by the easy, low-polyphony majority of soundscape clips that range_mae's
aggregate mostly reflects -- the ordering reverses: this script exists to
make that reversal, and whether it is statistically distinguishable from
noise at n=28, a reproducible, checked-in result rather than a claim
resting on an unsaved scratch computation.

**Caveat (same as plot_rq5.py's compute_macro_mae_long())**: several
(model, dataset) cells' macro-MAE(1-6) is silently averaged over fewer
than 6 levels (zero unambiguous-label clips at the missing level(s) in
that region) -- the count of thin cells per study is reported below.

**Caveat (same as elsewhere in RQ5)**: n=28 pooled (n=24 NES excluded) is
small for the paired significance tests below -- treat them as
exploratory, same caveat as compute_rq5_kurtosis.py,
compute_rq5_finetune_vs_frozen.py, and
compute_rq5_frozen_vs_region_4backbone.py.

Writes rq5_macro_mae_summary.json and rq5_macro_mae_summary.md to
--out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq5_macro_mae_summary.py [--out-dir plots/figures/rq5]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import BACKBONE_META, FINETUNE_NAME_TO_CANONICAL
from compute_rq5_frozen_vs_region_4backbone import cell_diffs, fmt_mean_std, full_entry
from plot_confusion_matrix import ARCHIVE as CM_ARCHIVE
from plot_data import load_study, mean_sd_n
from plot_rq5 import SCAPE_SOURCE, compute_macro_mae_long, species_ordered_datasets

# rq5_kurtosis.md's "Overall (all 28 cells per study)" table -- the range_mae
# counterpart to this script's macro-MAE(1-6) pooled numbers, reported
# alongside them below rather than recomputed (already checked in there).
RANGE_MAE_POOLED_REFERENCE = {
    "region_specific": {"mean": 0.903, "std": 0.567, "n": 28},
    "xcm_frozen": {"mean": 1.022, "std": 0.699, "n": 28},
    "xcm_finetuned": {"mean": 0.938, "std": 0.488, "n": 28},
}

STUDY_LABELS = [("region_specific", "Region-specific head"), ("xcm_frozen", "XCM head (frozen backbone)"),
                ("xcm_finetuned", "XCM head (fine-tuned)")]


def pivot_macro_long(long_df: pd.DataFrame, row_order: list, dataset_order: list) -> pd.DataFrame:
    """(model, dataset, value) long table -> model x dataset matrix, same
    shape/orientation as plot_rq5.build_heatmap_matrix()'s output, so
    cell_diffs() (written against that shape) works unchanged."""
    return long_df.pivot_table(index="model", columns="dataset", values="value").reindex(
        index=row_order, columns=dataset_order)


def per_dataset_table(long_df: pd.DataFrame, dataset_order: list) -> pd.DataFrame:
    return mean_sd_n(long_df, ["dataset"]).reindex(dataset_order)


def per_backbone_table(matrix: pd.DataFrame) -> pd.DataFrame:
    """mean/SD/n across each backbone's own datasets (row-wise over the
    model x dataset matrix; NaN cells skipped, n surfaces any gap)."""
    return pd.DataFrame({"mean": matrix.mean(axis=1), "std": matrix.std(axis=1, ddof=1),
                         "count": matrix.count(axis=1)})


def build_markdown(per_dataset: dict, pooled: dict, row_order: list, dataset_order: list,
                    thin_counts: dict, deltas: dict, matrices: dict, per_backbone: dict) -> str:
    backbone_list = ", ".join(BACKBONE_META[m]["display"] for m in row_order)

    lines = [
        "# RQ5 -- Macro-MAE(1-6) summary, backbone-matched 3-way soundscape comparison",
        "",
        "Checked-in source for `rq5_panel_a_soundscape_matched_macro_mae`'s per-dataset lines and for "
        "`results/rq5_results.tex`'s pooled macro-MAE(1-6) summary table -- both were previously computed "
        "in an unsaved scratch script. See module docstring for the macro-MAE(1-6) definition (plot_rq1.py's "
        "`_region_macro_mae_1_6()`, applied per dataset here instead of averaged across all 7 regions).",
        "",
        f"4 backbones ({backbone_list}) x 7 datasets ({', '.join(dataset_order)}) = 28 cells per study. "
        "Region-specific: `archive/Pooled-Embeddings/`; XCM frozen: `archive/XCM-Generalization/`; "
        "XCM fine-tuned: `archive/XCM-Generalization-fine-tune/`.",
        "",
    ]

    thin_note = "; ".join(f"{label}: {thin_counts[key]}/28" for key, label in STUDY_LABELS)
    lines += [
        f"**Thin-cell caveat**: cells using fewer than 6 levels for their macro-MAE(1-6) average -- {thin_note}. "
        "Same caveat as `plot_rq1.py`'s `compute_soundscape_macro_mae_stats()`: some regions have zero "
        "unambiguous-label clips at the higher polyphony levels, so those cells are silently averaged over "
        "fewer than 6 levels rather than being NaN'd out.",
        "",
        "## 1. Per-dataset macro-MAE(1-6) (mean +/- SD across the 4 backbones)",
        "",
        "Same data as the figure's lines -- reproduced here as a table for direct citation.",
        "",
        "| Dataset | Region-specific | XCM frozen | XCM fine-tuned |",
        "|---|---|---|---|",
    ]
    for dataset in dataset_order:
        row = [dataset]
        for key, _ in STUDY_LABELS:
            r = per_dataset[key].loc[dataset]
            row.append(fmt_mean_std(r["mean"], r["std"], decimals=3) + f" (n={int(r['count'])})")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## 2. Pooled overall (all 28 cells), macro-MAE(1-6) vs. range_mae",
        "",
        "The range_mae column is `rq5_kurtosis.md`'s already-reported \"Overall (all 28 cells per study)\" "
        "table, reproduced here (not recomputed) for direct side-by-side comparison -- the two metrics "
        "disagree on which configuration is best.",
        "",
        "| Configuration | Macro-MAE(1-6) (mean +/- SD) | range_mae (mean +/- SD) |",
        "|---|---|---|",
    ]
    for key, label in STUDY_LABELS:
        m = pooled[key]
        r = RANGE_MAE_POOLED_REFERENCE[key]
        macro_str = fmt_mean_std(m["mean"], m["std"]) + f" (n={m['n']})"
        range_str = fmt_mean_std(r["mean"], r["std"]) + f" (n={r['n']})"
        lines.append(f"| {label} | {macro_str} | {range_str} |")

    best_macro = min(STUDY_LABELS, key=lambda kl: pooled[kl[0]]["mean"])[1]
    best_range = min(STUDY_LABELS, key=lambda kl: RANGE_MAE_POOLED_REFERENCE[kl[0]]["mean"])[1]
    lines += [
        "",
        f"Best (lowest pooled mean) configuration under macro-MAE(1-6): **{best_macro}**. "
        f"Best under range_mae: **{best_range}**." + (
            " These agree." if best_macro == best_range else
            " **These disagree** -- the metric choice changes which configuration looks best."),
        "",
        "## 3. Cell-level paired comparisons, macro-MAE(1-6)",
        "",
        "Delta = (other config) - (baseline config), matched by (backbone, dataset) cell. Negative = other "
        "config beats baseline. Sign test: win = other beats baseline (Delta<0), loss = Delta>0, ties "
        "dropped before the test -- same convention as `compute_rq5_finetune_vs_frozen.py` and "
        "`compute_rq5_frozen_vs_region_4backbone.py`, applied here to macro-MAE(1-6) instead of range_mae.",
        "",
        "| Comparison (other - baseline) | Mean Delta (+/- SD) | n | Paired t-test | Wilcoxon | Sign test (W-L-T) | Sign test p |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, entry in deltas.items():
        t_p = f"{entry['t_p']:.4f}" if entry["t_p"] is not None else "n/a"
        w_p = f"{entry['wilcoxon_p']:.4f}" if entry["wilcoxon_p"] is not None else "n/a"
        st = entry["sign_test"]
        wlt = f"{st['wins']}-{st['losses']}-{st['ties']}"
        st_p = f"{st['p_two_sided']:.4f}" if st["p_two_sided"] is not None else "n/a"
        lines.append(f"| {label} | {fmt_mean_std(entry['mean'], entry['std'])} | {entry['n']} "
                     f"| p={t_p} | p={w_p} | {wlt} | p={st_p} |")

    lines += [
        "",
        "## 4. Per-cell macro-MAE(1-6) (backbone x dataset)",
        "",
        "Exact cell values behind `rq5_panel_b_soundscape_matched_grid_macro_mae`. `n/a` = no value for that "
        "cell.",
    ]
    for key, label in STUDY_LABELS:
        matrix = matrices[key]
        lines += [
            "",
            f"### {label}",
            "",
            "| Backbone | " + " | ".join(dataset_order) + " |",
            "|---|" + "---|" * len(dataset_order),
        ]
        for model in row_order:
            cells = ["n/a" if np.isnan(v) else f"{v:.3f}" for v in matrix.loc[model].values]
            lines.append(f"| {BACKBONE_META[model]['display']} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 5. Per-backbone macro-MAE(1-6) (mean +/- SD across the 7 datasets)",
        "",
        "Row-wise reduction of Section 4's matrices: each backbone's own mean and SD over its 7 datasets "
        "(the SD is spread across regions for one backbone, not disagreement between backbones).",
        "",
        "| Backbone | Region-specific | XCM frozen | XCM fine-tuned |",
        "|---|---|---|---|",
    ]
    for model in row_order:
        row = [BACKBONE_META[model]["display"]]
        for key, _ in STUDY_LABELS:
            r = per_backbone[key].loc[model]
            row.append(fmt_mean_std(r["mean"], r["std"], decimals=3) + f" (n={int(r['count'])})")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Reading this",
        "",
        "- Section 1 is the exact data behind `rq5_panel_a_soundscape_matched_macro_mae`'s three lines.",
        "- Section 4 is the exact data behind `rq5_panel_b_soundscape_matched_grid_macro_mae`'s three heatmaps; "
        "Section 5 is its per-backbone (row-wise) mean +/- SD.",
        "- Section 2 is the headline pooled comparison cited in `results/rq5_results.tex` -- the "
        "macro-MAE(1-6)-vs-range_mae disagreement on which configuration generalizes best is the central "
        "finding this script exists to make reproducible.",
        "- Section 3 formally tests the deltas underlying Section 2's ordering. Per the caveat above, n=28 "
        "is small -- read non-significant results as \"not distinguishable from chance at this sample size\", "
        "not as \"no real difference\".",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq5"))
    args = parser.parse_args()

    dataset_order, _ = species_ordered_datasets()
    matched_backbones = list(FINETUNE_NAME_TO_CANONICAL.values())
    row_order = [m for m in BACKBONE_META if m in matched_backbones]

    pooled_macro_long = compute_macro_mae_long(CM_ARCHIVE / "Pooled-Embeddings", matched_backbones, dataset_order)
    xg_macro_long = compute_macro_mae_long(CM_ARCHIVE / "XCM-Generalization", matched_backbones, dataset_order)
    ft_macro_long = compute_macro_mae_long(CM_ARCHIVE / "XCM-Generalization-fine-tune",
                                            list(FINETUNE_NAME_TO_CANONICAL.keys()), dataset_order,
                                            model_map=FINETUNE_NAME_TO_CANONICAL)
    long_by_key = {"region_specific": pooled_macro_long, "xcm_frozen": xg_macro_long,
                   "xcm_finetuned": ft_macro_long}

    per_dataset = {key: per_dataset_table(df, dataset_order) for key, df in long_by_key.items()}
    pooled = {key: {"mean": float(df["value"].mean()), "std": float(df["value"].std()), "n": len(df)}
              for key, df in long_by_key.items()}
    thin_counts = {key: int((df["n_levels_used"] < 6).sum()) for key, df in long_by_key.items()}

    region_matrix = pivot_macro_long(pooled_macro_long, row_order, dataset_order)
    xcm_matrix = pivot_macro_long(xg_macro_long, row_order, dataset_order)
    ft_matrix = pivot_macro_long(ft_macro_long, row_order, dataset_order)

    matrices = {"region_specific": region_matrix, "xcm_frozen": xcm_matrix, "xcm_finetuned": ft_matrix}
    per_backbone = {key: per_backbone_table(m) for key, m in matrices.items()}

    deltas = {
        "XCM frozen - Region-specific": full_entry(cell_diffs(xcm_matrix, region_matrix, dataset_order, row_order)),
        "XCM fine-tuned - Region-specific": full_entry(cell_diffs(ft_matrix, region_matrix, dataset_order, row_order)),
        "XCM fine-tuned - XCM frozen": full_entry(cell_diffs(ft_matrix, xcm_matrix, dataset_order, row_order)),
    }

    results = {
        "per_dataset": {key: json.loads(df.to_json(orient="index")) for key, df in per_dataset.items()},
        "pooled_macro_mae": pooled,
        "pooled_range_mae_reference": RANGE_MAE_POOLED_REFERENCE,
        "per_cell": {key: json.loads(m.to_json(orient="index")) for key, m in matrices.items()},
        "per_backbone": {key: json.loads(t.to_json(orient="index")) for key, t in per_backbone.items()},
        "thin_cell_counts": thin_counts,
        "cell_level_deltas": deltas,
        "backbone_order": row_order,
        "dataset_order": dataset_order,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq5_macro_mae_summary.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")

    md = build_markdown(per_dataset, pooled, row_order, dataset_order, thin_counts, deltas, matrices, per_backbone)
    out_md = args.out_dir / "rq5_macro_mae_summary.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
