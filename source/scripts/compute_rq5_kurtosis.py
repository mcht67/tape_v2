#!/usr/bin/env python3
"""
RQ5 (additional) -- Excess kurtosis (and mean) of the soundscape MAE
distribution underlying rq5_panel_b_soundscape_matched_grid (plot_rq5.py's
3 backbone-matched Panel B heatmaps: region-specific, XCM head frozen, XCM
fine-tune head; 4 backbones x 7 regional datasets each -- perch_v2 excluded,
see backbone_meta.py's comment on FINETUNE_NAME_TO_CANONICAL). Quantifies
how heavy-tailed / outlier-driven each study's error distribution is, as a
complement to the correlation and leave-one-out analyses elsewhere in RQ5
(e.g. NES's large residual in the XCM head vs. #species correlation --
excess kurtosis is a way to check whether that kind of single-point spike
is typical of these matrices or specific to that one case).

Excess kurtosis (Fisher's definition: 0 for a normal distribution, positive
"leptokurtic" for a heavy-tailed/peaked distribution with a few extreme
values, negative "platykurtic" for flatter-than-normal) is reported
alongside the mean and its standard deviation (so a high-kurtosis, low-mean
cell -- an occasional spike on an otherwise-easy set -- reads differently
from a high-kurtosis, high-mean one, and the SD gives a sense of spread even
where kurtosis itself is unstable at these sample sizes), computed 3 ways
per study:
  - overall: all cells (4 backbones x however many datasets) pooled.
  - per-dataset: across the 4 backbones for each dataset (n=4) -- "does this
    dataset have one backbone that's way off from the other 3".
  - per-backbone: across the datasets for each of the 4 backbones (n=7, or
    n=6 in the PER-excluded analysis) -- "does this backbone have one
    dataset that's way off from its others".

n=4/6/7/24/28 are all small for a 4th-moment statistic (kurtosis needs a
much larger sample to be a stable estimate) -- these are reported as
directional/exploratory diagnostics, not confirmatory hypothesis tests, and
the bias-corrected estimator (bias=False) is used throughout for that
reason.

Two analyses are run and reported side by side in the same JSON/markdown:
the full 7-dataset one, and a 2nd with PER excluded (n=6 datasets) -- PER is
the most polyphony-dense regional dataset and the intrinsically hardest one
for region-specific/fine-tuned heads throughout RQ1/RQ4/RQ5, so excluding it
checks how much of the heavy-tailedness above is specifically a PER effect
versus general across the other 6 datasets (the same kind of check already
done for NES elsewhere in RQ5's correlation analysis).

A third addition, alongside (not folded into) the per-study kurtosis/mean/
SD above: paired dataset-level differences of the XCM head and XCM
fine-tune head against the region-specific head -- same "gap" convention as
compute_rq5_diff_correlations.py (other head's per-dataset mean MAE minus
region-specific head's, each side first averaged across backbones per
dataset, then paired by dataset), reported as mean +/- SD of that gap across
datasets (n=7, or n=6 PER excluded). This SD is a different quantity from
the per-study SD above: it reflects how consistent the *gap* is dataset to
dataset, not how spread out either study's raw MAE is.

A fourth addition goes one level finer than the third: paired differences
at the individual (backbone, dataset) cell level, not first averaged across
backbones per dataset. Region-specific results exist per backbone (verified
against archive/Pooled-Embeddings/'s per-backbone folders, same data
build_fig1's dumbbell plot already uses per-backbone), so this finest-level
pairing is possible. Delta_frozen = XCM head MAE - region-specific MAE,
same (backbone, dataset) cell, over the 5 backbones common to both studies
(SPATIAL_BACKBONES, incl. Perch v2 -- frozen embeddings don't have Perch
v2's gradient problem) x 7 datasets = 35 cells (30 PER excluded).
Delta_finetuned = XCM fine-tune head MAE - region-specific MAE, same cell,
restricted to the 4 backbones with genuine fine-tuning results (Perch v2
excluded, see FINETUNE_NAME_TO_CANONICAL's comment) x 7 datasets = 28 cells
(24 PER excluded). Both are reported overall (mean +/- SD across all cells)
and Delta_finetuned is additionally broken down per backbone (mean +/- SD
across its own 7/6 datasets) -- the question motivating this: does
fine-tuning close the gap to region-specific consistently across backbones,
or reliably for some and not others. A paired t-test and Wilcoxon
signed-rank test against zero are reported alongside every mean +/- SD here
(pooled and, where n allows, per-backbone) -- the per-backbone tests (n=7 or
6) are exploratory/underpowered, same caveat as the kurtosis values above.

Writes rq5_kurtosis.json and rq5_kurtosis.md to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq5_kurtosis.py [--out-dir plots/figures/rq5]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import BACKBONE_META, FINETUNE_NAME_TO_CANONICAL
from plot_data import load_study
from plot_rq2 import SPATIAL_BACKBONES
from plot_rq5 import SCAPE_SOURCE, build_heatmap_matrix, species_ordered_datasets

STUDY_LABEL = {
    "region_specific": "Region-specific head",
    "xcm_head": "XCM head (frozen backbone)",
    "xcm_finetune_head": "XCM fine-tune head",
}


def excess_kurtosis(values) -> float | None:
    """Bias-corrected Fisher excess kurtosis; None if too few finite values
    (kurtosis needs at least 4 points, and is meaningless below that)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 4:
        return None
    return float(scipy.stats.kurtosis(values, fisher=True, bias=False))


def mean_value(values) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return None
    return float(values.mean())


def std_value(values) -> float | None:
    """Sample SD (ddof=1, matching plot_data.mean_sd_n's convention);
    None below n=2, where sample SD is undefined."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return None
    return float(values.std(ddof=1))


def stat_pair(values) -> dict:
    return {"kurtosis": excess_kurtosis(values), "mean": mean_value(values), "std": std_value(values),
            "n": int(len(np.asarray(values)))}


def fmt_mean_std(mean: float | None, std: float | None, decimals: int = 3) -> str:
    if mean is None:
        return "--"
    if std is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def analyze(matrices: dict, dataset_order: list, row_order: list) -> dict:
    """Overall/per-dataset/per-backbone kurtosis+mean for each study,
    restricted to dataset_order's columns (the full 7, or PER excluded)."""
    results = {}
    for study_key, matrix in matrices.items():
        sub = matrix[dataset_order]
        results[study_key] = {
            "overall": stat_pair(sub.values.flatten()),
            "per_dataset": {d: stat_pair(sub[d].values) for d in dataset_order},
            "per_backbone": {m: stat_pair(sub.loc[m].values) for m in row_order},
        }
    return results


def build_section(results: dict, dataset_order: list, row_order: list, heading: str) -> list:
    n_datasets = len(dataset_order)
    n_backbones = len(row_order)
    n_cells = n_datasets * n_backbones
    lines = [
        f"## {heading}",
        "",
        f"### Overall (all {n_cells} cells per study)",
        "",
        "| Study | Excess kurtosis | Mean MAE (±SD) | n |",
        "|---|---|---|---|",
    ]
    for study_key, label in STUDY_LABEL.items():
        s = results[study_key]["overall"]
        lines.append(f"| {label} | {s['kurtosis']:.2f} | {fmt_mean_std(s['mean'], s['std'])} | {s['n']} |")

    lines += [
        "",
        f"### Per-dataset (across the {n_backbones} backbones, n={n_backbones})",
        "",
        "| Dataset | " + " | ".join(f"{l} (kurt / mean±sd)" for l in STUDY_LABEL.values()) + " |",
        "|---|" + "---|" * len(STUDY_LABEL) + "",
    ]
    for d in dataset_order:
        row = []
        for s in STUDY_LABEL:
            stat = results[s]["per_dataset"][d]
            row.append(f"{stat['kurtosis']:.2f} / {fmt_mean_std(stat['mean'], stat['std'])}"
                       if stat["kurtosis"] is not None else "--")
        lines.append(f"| {d} | " + " | ".join(row) + " |")

    lines += [
        "",
        f"### Per-backbone (across the {n_datasets} datasets, n={n_datasets})",
        "",
        "| Backbone | " + " | ".join(f"{l} (kurt / mean±sd)" for l in STUDY_LABEL.values()) + " |",
        "|---|" + "---|" * len(STUDY_LABEL) + "",
    ]
    for m in row_order:
        display = BACKBONE_META[m]["display"]
        row = []
        for s in STUDY_LABEL:
            stat = results[s]["per_backbone"][m]
            row.append(f"{stat['kurtosis']:.2f} / {fmt_mean_std(stat['mean'], stat['std'])}"
                       if stat["kurtosis"] is not None else "--")
        lines.append(f"| {display} | " + " | ".join(row) + " |")

    return lines


# ---------------------------------------------------------------------------
# Paired dataset-level differences: (other head - region-specific head),
# paired by dataset after first averaging each study's MAE across backbones
# per dataset (i.e. diffing the "per_dataset" means computed above) -- same
# "gap" convention as compute_rq5_diff_correlations.py ("cross-region
# generalization gap" = cross-region head's per-dataset mean MAE minus the
# region-specific head's, paired by dataset).
# ---------------------------------------------------------------------------

GAP_BASELINE = "region_specific"
GAP_KEYS = ["xcm_head", "xcm_finetune_head"]


def paired_diff_stats(results: dict, dataset_order: list, other_key: str) -> dict:
    """Per-dataset gap (other_key's per-dataset mean minus GAP_BASELINE's),
    then mean+std across those len(dataset_order) per-dataset gaps -- 'n' is
    the number of datasets, not the number of underlying (backbone,
    dataset) cells."""
    diffs = np.array([
        results[other_key]["per_dataset"][d]["mean"] - results[GAP_BASELINE]["per_dataset"][d]["mean"]
        for d in dataset_order
    ], dtype=float)
    return {
        "mean": float(diffs.mean()),
        "std": float(diffs.std(ddof=1)) if len(diffs) > 1 else None,
        "n": len(diffs),
    }


def build_paired_diff_section(results_full: dict, results_no_per: dict, dataset_order: list,
                               no_per_order: list) -> list:
    lines = [
        "## Paired dataset-level differences vs. region-specific head",
        "",
        f"Gap = (other head's per-dataset mean MAE) - ({STUDY_LABEL[GAP_BASELINE]}'s per-dataset mean MAE), "
        "each side first averaged across backbones per dataset, then paired by dataset (not by individual "
        "(backbone, dataset) cell). Positive = other head is worse than the region-specific head on "
        "average. Mean ± SD is across the per-dataset gaps themselves (n=7 or n=6, PER excluded), so "
        "the SD reflects how consistent the gap is across datasets, not backbone-level spread.",
        "",
        "| Gap | Analysis 1 (7 datasets) | Analysis 2 (PER excluded, 6 datasets) |",
        "|---|---|---|",
    ]
    for key in GAP_KEYS:
        full = paired_diff_stats(results_full, dataset_order, key)
        no_per = paired_diff_stats(results_no_per, no_per_order, key)
        label = f"{STUDY_LABEL[key]} - {STUDY_LABEL[GAP_BASELINE]}"
        lines.append(f"| {label} | {fmt_mean_std(full['mean'], full['std'])} (n={full['n']}) "
                      f"| {fmt_mean_std(no_per['mean'], no_per['std'])} (n={no_per['n']}) |")
    return lines


# ---------------------------------------------------------------------------
# Cell-level (backbone x dataset) paired differences: finer-grained than the
# dataset-level gap above -- no averaging across backbones first, so the
# comparison is paired at the finest level the data supports (same backbone,
# same dataset).
# ---------------------------------------------------------------------------

FROZEN_LABEL = f"{STUDY_LABEL['xcm_head']} - {STUDY_LABEL[GAP_BASELINE]}"
FINETUNE_LABEL = f"{STUDY_LABEL['xcm_finetune_head']} - {STUDY_LABEL[GAP_BASELINE]}"


def cell_diffs(other_matrix, baseline_matrix, dataset_order: list, row_order: list) -> np.ndarray:
    """Flattened (backbone, dataset) cell-level differences (other -
    baseline), both matrices reindexed to the same row_order/dataset_order
    first so cells line up 1:1; NaN cells (missing data) are dropped."""
    diff = (other_matrix.loc[row_order, dataset_order] - baseline_matrix.loc[row_order, dataset_order]).values
    diff = diff.flatten().astype(float)
    return diff[~np.isnan(diff)]


def cell_diff_stats(diff: np.ndarray) -> dict:
    n = len(diff)
    return {
        "mean": float(diff.mean()) if n else None,
        "std": float(diff.std(ddof=1)) if n > 1 else None,
        "n": n,
    }


def paired_tests(diff: np.ndarray) -> dict:
    """Paired t-test and Wilcoxon signed-rank test of diff against zero.
    None for either test below the sample size it needs, or if Wilcoxon
    raises (e.g. all-zero differences)."""
    if len(diff) < 2:
        return {"t_stat": None, "t_p": None, "wilcoxon_stat": None, "wilcoxon_p": None}
    t_stat, t_p = scipy.stats.ttest_1samp(diff, popmean=0.0)
    try:
        w_stat, w_p = scipy.stats.wilcoxon(diff)
    except ValueError:
        w_stat, w_p = None, None
    return {
        "t_stat": float(t_stat), "t_p": float(t_p),
        "wilcoxon_stat": float(w_stat) if w_stat is not None else None,
        "wilcoxon_p": float(w_p) if w_p is not None else None,
    }


def _fmt_test(tests: dict) -> str:
    if tests["t_p"] is None:
        return "--"
    return f"t-test p={tests['t_p']:.3f}; Wilcoxon p={tests['wilcoxon_p']:.3f}" if tests["wilcoxon_p"] is not None \
        else f"t-test p={tests['t_p']:.3f}; Wilcoxon n/a"


def build_cell_diff_section(region_5, xcm_head_5, region_4, xcm_finetune_4,
                             dataset_order: list, no_per_order: list, ft_row_order: list) -> list:
    lines = [
        "## Paired (backbone x dataset) cell-level differences vs. region-specific head",
        "",
        "Finer-grained than the dataset-level gap above: no averaging across backbones first -- each "
        "cell is (backbone, dataset)-matched directly. Delta_frozen uses the 5 backbones common to the "
        f"region-specific and frozen-XCM studies ({', '.join(BACKBONE_META[m]['display'] for m in SPATIAL_BACKBONES)}"
        f"); Delta_finetuned uses the 4 with genuine fine-tuning results "
        f"({', '.join(BACKBONE_META[m]['display'] for m in ft_row_order)}; Perch v2 excluded -- see "
        "FINETUNE_NAME_TO_CANONICAL's comment). p-values are for a paired t-test and Wilcoxon "
        "signed-rank test of the cell-level differences against zero.",
        "",
        "### Overall",
        "",
        "| Delta | Analysis 1 (all datasets) | Test | Analysis 2 (PER excluded) | Test |",
        "|---|---|---|---|---|",
    ]

    frozen_full = cell_diffs(xcm_head_5, region_5, dataset_order, SPATIAL_BACKBONES)
    frozen_no_per = cell_diffs(xcm_head_5, region_5, no_per_order, SPATIAL_BACKBONES)
    finetune_full = cell_diffs(xcm_finetune_4, region_4, dataset_order, ft_row_order)
    finetune_no_per = cell_diffs(xcm_finetune_4, region_4, no_per_order, ft_row_order)

    for label, full, no_per in [(FROZEN_LABEL, frozen_full, frozen_no_per),
                                 (FINETUNE_LABEL, finetune_full, finetune_no_per)]:
        fs, ns = cell_diff_stats(full), cell_diff_stats(no_per)
        ft, nt = paired_tests(full), paired_tests(no_per)
        lines.append(f"| {label} | {fmt_mean_std(fs['mean'], fs['std'])} (n={fs['n']}) | {_fmt_test(ft)} "
                      f"| {fmt_mean_std(ns['mean'], ns['std'])} (n={ns['n']}) | {_fmt_test(nt)} |")

    lines += [
        "",
        "### Delta_finetuned by backbone",
        "",
        "Tests here are exploratory/underpowered (n=7 or 6 per backbone), same caveat as the kurtosis "
        "values above.",
        "",
        "| Backbone | Analysis 1 (7 datasets) | Test | Analysis 2 (PER excluded, 6 datasets) | Test |",
        "|---|---|---|---|---|",
    ]
    for m in ft_row_order:
        full = cell_diffs(xcm_finetune_4, region_4, dataset_order, [m])
        no_per = cell_diffs(xcm_finetune_4, region_4, no_per_order, [m])
        fs, ns = cell_diff_stats(full), cell_diff_stats(no_per)
        ft, nt = paired_tests(full), paired_tests(no_per)
        display = BACKBONE_META[m]["display"]
        lines.append(f"| {display} | {fmt_mean_std(fs['mean'], fs['std'])} (n={fs['n']}) | {_fmt_test(ft)} "
                      f"| {fmt_mean_std(ns['mean'], ns['std'])} (n={ns['n']}) | {_fmt_test(nt)} |")

    return lines


def build_markdown(results_full: dict, results_no_per: dict, dataset_order: list, row_order: list, *,
                    region_5, xcm_head_5, region_4, xcm_finetune_4) -> str:
    no_per_order = [d for d in dataset_order if d != "PER"]
    n_backbones = len(row_order)
    n_full = n_backbones * len(dataset_order)
    n_no_per = n_backbones * len(no_per_order)
    lines = [
        "# RQ5 -- Excess kurtosis of the backbone-matched soundscape MAE heatmaps",
        "",
        f"Source figure: `rq5_panel_b_soundscape_matched_grid.png` (3 studies x {n_backbones} backbones x 7 "
        "regional datasets). Excess kurtosis (Fisher's definition, bias-corrected): 0 for a normal "
        "distribution, **positive** for a heavy-tailed distribution dominated by a few extreme "
        "values, **negative** for a flatter-than-normal one. Mean MAE (± SD across the same "
        "underlying cells) is reported alongside it (lower is better) so a high-kurtosis value can be "
        "read together with whether the underlying level is generally low or high, and the SD gives a "
        "sense of spread even where kurtosis itself is unstable at these sample sizes.",
        "",
        "**Caveat on sample size**: kurtosis is a 4th-moment statistic and is unstable at the sample "
        f"sizes here (n={n_backbones} per dataset, n=6/7 per backbone, n={n_no_per}/{n_full} overall). "
        "Treat every number below as directional/exploratory, not a confirmatory test -- especially the "
        f"per-dataset (n={n_backbones}) and per-backbone (n=6/7) values, where a single outlier can swing "
        "the estimate by several units.",
        "",
    ]
    lines += build_section(results_full, dataset_order, row_order, "Analysis 1: all 7 regional datasets")
    lines += [""]
    lines += build_section(results_no_per, no_per_order, row_order, "Analysis 2: PER excluded (6 datasets)")
    lines += [""]
    lines += build_paired_diff_section(results_full, results_no_per, dataset_order, no_per_order)
    lines += [""]
    lines += build_cell_diff_section(region_5, xcm_head_5, region_4, xcm_finetune_4,
                                      dataset_order, no_per_order, row_order)

    lines += [
        "",
        "## Reading this",
        "",
        f"- A high positive per-*dataset* value means that dataset's {n_backbones} backbone values are "
        "dominated by one extreme backbone (the rest clustered) -- e.g. NES/XCM head is dragged almost "
        "entirely by one outlier backbone.",
        "- A high positive per-*backbone* value means that backbone's 7 (or 6, PER excluded) dataset "
        "values are dominated by one extreme dataset -- e.g. a backbone whose only bad result is NES.",
        "- The overall value folds both effects together across all cells at once.",
        "- Comparing Analysis 1 vs. Analysis 2 shows how much of the overall heavy-tailedness is "
        "specifically a PER effect: if the overall kurtosis drops a lot with PER excluded, PER was a "
        "major contributor to it; if it barely moves, the heavy tail comes from elsewhere (e.g. NES).",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq5"))
    args = parser.parse_args()

    xg_models = mrt.discover_models(Path("archive/XCM-Generalization"))
    pooled_long = load_study("pooled", models=xg_models)
    xg_long = load_study("xcm_gen", models=xg_models)
    # models= excludes perch_v2's raw folder (still on disk, but not a
    # genuine fine-tuning result -- see backbone_meta.py's comment).
    ft_long = load_study("xcm_gen_ft", models=list(FINETUNE_NAME_TO_CANONICAL.keys()))

    dataset_order, _ = species_ordered_datasets()
    ft_row_order = [m for m in BACKBONE_META if m in FINETUNE_NAME_TO_CANONICAL.values()]
    no_per_order = [d for d in dataset_order if d != "PER"]

    matrices = {
        "region_specific": build_heatmap_matrix(pooled_long, SCAPE_SOURCE, "range_mae", dataset_order, ft_row_order),
        "xcm_head": build_heatmap_matrix(xg_long, SCAPE_SOURCE, "range_mae", dataset_order, ft_row_order),
        "xcm_finetune_head": build_heatmap_matrix(ft_long, SCAPE_SOURCE, "range_mae", dataset_order, ft_row_order,
                                                   model_map=FINETUNE_NAME_TO_CANONICAL),
    }
    # 5-backbone (SPATIAL_BACKBONES, incl. Perch v2) matrices for Delta_frozen --
    # separate from `matrices` above, which is restricted to the 4 fine-tune-
    # capable backbones throughout so all 3 studies there stay cell-aligned.
    region_5 = build_heatmap_matrix(pooled_long, SCAPE_SOURCE, "range_mae", dataset_order, SPATIAL_BACKBONES)
    xcm_head_5 = build_heatmap_matrix(xg_long, SCAPE_SOURCE, "range_mae", dataset_order, SPATIAL_BACKBONES)

    results_full = analyze(matrices, dataset_order, ft_row_order)
    results_no_per = analyze(matrices, no_per_order, ft_row_order)
    paired_diffs = {
        "full": {key: paired_diff_stats(results_full, dataset_order, key) for key in GAP_KEYS},
        "per_excluded": {key: paired_diff_stats(results_no_per, no_per_order, key) for key in GAP_KEYS},
    }

    def cell_diff_entry(other_matrix, baseline_matrix, row_order, dataset_subset):
        diff = cell_diffs(other_matrix, baseline_matrix, dataset_subset, row_order)
        return {**cell_diff_stats(diff), **paired_tests(diff)}

    cell_diffs_json = {
        "frozen": {
            "full": cell_diff_entry(xcm_head_5, region_5, SPATIAL_BACKBONES, dataset_order),
            "per_excluded": cell_diff_entry(xcm_head_5, region_5, SPATIAL_BACKBONES, no_per_order),
        },
        "finetuned": {
            "full": cell_diff_entry(matrices["xcm_finetune_head"], matrices["region_specific"], ft_row_order,
                                     dataset_order),
            "per_excluded": cell_diff_entry(matrices["xcm_finetune_head"], matrices["region_specific"], ft_row_order,
                                             no_per_order),
            "per_backbone": {
                m: {
                    "full": cell_diff_entry(matrices["xcm_finetune_head"], matrices["region_specific"], [m],
                                             dataset_order),
                    "per_excluded": cell_diff_entry(matrices["xcm_finetune_head"], matrices["region_specific"], [m],
                                                     no_per_order),
                }
                for m in ft_row_order
            },
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq5_kurtosis.json"
    out_json.write_text(json.dumps(
        {"full": results_full, "per_excluded": results_no_per, "paired_diff_vs_region_specific": paired_diffs,
         "cell_diff_vs_region_specific": cell_diffs_json},
        indent=2))
    print(f"Wrote {out_json}")

    md = build_markdown(results_full, results_no_per, dataset_order, ft_row_order,
                         region_5=region_5, xcm_head_5=xcm_head_5,
                         region_4=matrices["region_specific"], xcm_finetune_4=matrices["xcm_finetune_head"])
    out_md = args.out_dir / "rq5_kurtosis.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
