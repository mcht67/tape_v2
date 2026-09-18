#!/usr/bin/env python3
"""RQ5 (additional) -- "What drives soundscape data difficulty?" recomputed
with per-dataset macro-MAE(1-6) in place of plain per-dataset MAE.

Background: `rq4_fig3_difficulty_table.tex` / `tab:difficulty_vs_polyphony`
and the narrative paragraph it backs report per-dataset MAE as `range_mae`,
mean across the 13 Pooled-Embeddings backbones, POOLED across every clip in
each region -- including polyphony=0, the single largest class by a wide
margin (see `compute_soundscape_per_level_accuracy.py`'s docstring). Section
4.1.2 and the RQ2 soundscape analysis established that this pooled metric
can be substantially distorted by that class imbalance, and switched to
macro-MAE(1-6): the unweighted mean of per-level MAE over true levels 1-6
(dropping the dominant zero class and the too-thin 7/8 tail), computed per
region then averaged unweighted across regions. This script recomputes every
correlation and descriptive statistic in the "difficulty is associated with
polyphonic complexity" paragraph using that same macro-MAE(1-6) metric, but
aggregated per DATASET (backbone-averaged across all 13 backbones) rather
than per BACKBONE (region-averaged, as in `rq1_soundscape_ranking.md`) --
i.e. for each of the 7 regional datasets, the mean of that region's own
macro-MAE(1-6) across the 13 backbones (each backbone's macro-MAE(1-6) for
that region computed by `per_region_stats()`, reused unmodified from
`compute_soundscape_per_level_accuracy.py`).

Difficulty metrics (ratio_polyp, mean_polyp, max_polyp) and candidate
factors (num_species, n_train_segments, range_width) are the same
`plots/data/polybirdmix_soundscape_stats.json` getters used by
`compute_rq5_polyphony_robustness.py`; this script mirrors that script's
structure (per-dataset input table, Table 1 difficulty-metric correlations,
Table 2 candidate-factor null checks, leave-one-out detail, PER-exclusion
detail) so the plain-MAE and macro-MAE versions can be compared row for row.
The plain-MAE baseline quoted throughout is `compute_rq5_polyphony_robustness
.py`'s own recomputed baseline (r=0.86/0.89/0.80 Pearson, rho=0.96/0.96/0.79
Spearman for ratio_polyp/mean_polyp/max_polyp respectively -- see that
script's "Baseline discrepancy" section for why this, not the narrative's
quoted r=0.93/0.95/0.82, is the right comparison point), not the narrative
thesis text (which is not present in this repo).

Level-coverage confound: a per-*dataset* macro-MAE(1-6) is not the same
comparison as `rq1_soundscape_ranking.md`'s per-*backbone* macro-MAE(1-6).
There, level coverage is fixed per region and so cancels out across
backbones being compared within one region. Here, coverage differs by
*dataset* (HSN has zero clips at levels 4-6; NES/UHH have none at level 6
and only n=4 at level 5), so two datasets' "macro-MAE(1-6)" can silently
average over different level sets -- a dataset missing the hardest levels
looks artificially easier under a naive reading, and (empirically) the
reverse happens here: the low-polyphony datasets with the thinnest/most
incomplete coverage end up with inflated macro-MAE, which is what collapses
the correlation with He et al.'s polyphony metrics. This script addresses
the confound directly: alongside the naive per-dataset macro-MAE(1-6) (Table
1, confound intact, kept for comparability with the request that motivated
this script), it also computes macro-MAE restricted to the level set with
data in *every* one of the 7 datasets (Table 3; a stricter n>=30-per-level
variant is checked as a robustness aside) and reruns every correlation on
that coverage-matched metric.

Writes rq5_macro_mae_difficulty_robustness.json and
rq5_macro_mae_difficulty_robustness.md to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq5_macro_mae_difficulty_robustness.py [--out-dir plots/figures/rq5]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backbone_meta import BACKBONE_META
from compute_soundscape_per_level_accuracy import ARCHIVE, REGIONS, load_region_df, per_region_stats
from plot_data import REGIONAL_DATASETS, filter_long, load_study
from plot_rq5 import SCAPE_SOURCE, STATS_JSON

DIFFICULTY_METRICS = {
    "ratio_polyp": "Ratio polyphonic",
    "mean_polyp": "Mean polyphony",
    "max_polyp": "Mean max-polyp",
}
CANDIDATE_FACTORS = {
    "num_species": "#Species (unique)",
    "n_train_segments": "#Training segments",
    "range_width": "Polyphony range width (mean per segment)",
}

# compute_rq5_polyphony_robustness.py's own recomputed 13-backbone plain-MAE
# baseline (not the narrative's quoted r=0.93/0.95/0.82 -- see that script's
# module docstring). Pearson r (full n=7, PER-excluded n=6) and Spearman rho
# (full n=7, PER-excluded n=6).
PLAIN_MAE_BASELINE = {
    "ratio_polyp": {"pearson_full": 0.86, "pearson_per_excl": 0.83, "spearman_full": 0.96, "spearman_per_excl": 0.94},
    "mean_polyp": {"pearson_full": 0.89, "pearson_per_excl": 0.79, "spearman_full": 0.96, "spearman_per_excl": 0.94},
    "max_polyp": {"pearson_full": 0.80, "pearson_per_excl": 0.85, "spearman_full": 0.79, "spearman_per_excl": 0.71},
}


def difficulty_value(stats: dict, dataset: str, key: str) -> float:
    return stats[dataset]["polyphony_metric_summary"][key]["mean"]


def factor_value(stats: dict, dataset: str, key: str) -> float:
    if key == "num_species":
        return stats[dataset]["n_unique_species"]
    if key == "n_train_segments":
        return stats[dataset]["n_train_segments"]
    if key == "range_width":
        return stats[dataset]["distributions"]["range_width"]["mean"]
    raise KeyError(key)


def corr_entry(x: list, y: list, method) -> dict:
    r = method(x, y)
    return {"stat": round(float(r.statistic), 4), "p_value": round(float(r.pvalue), 4), "n": len(x)}


def analyze_property(values_by_dataset: dict, mae_by_dataset: dict, dataset_order: list) -> dict:
    x_full = [values_by_dataset[d] for d in dataset_order]
    y_full = [mae_by_dataset[d] for d in dataset_order]

    loo = {}
    for excluded in dataset_order:
        subset = [d for d in dataset_order if d != excluded]
        x = [values_by_dataset[d] for d in subset]
        y = [mae_by_dataset[d] for d in subset]
        loo[excluded] = corr_entry(x, y, scipy.stats.pearsonr)

    no_per_order = [d for d in dataset_order if d != "PER"]
    x_no_per = [values_by_dataset[d] for d in no_per_order]
    y_no_per = [mae_by_dataset[d] for d in no_per_order]

    return {
        "full_pearson": corr_entry(x_full, y_full, scipy.stats.pearsonr),
        "leave_one_out_pearson": loo,
        "per_excluded_pearson": corr_entry(x_no_per, y_no_per, scipy.stats.pearsonr),
        "full_spearman": corr_entry(x_full, y_full, scipy.stats.spearmanr),
        "per_excluded_spearman": corr_entry(x_no_per, y_no_per, scipy.stats.spearmanr),
    }


def fmt_stat(entry: dict, symbol: str) -> str:
    return f"{symbol}={entry['stat']:.2f}, p={entry['p_value']:.3f} (n={entry['n']})"


def loo_range(loo: dict) -> str:
    stats = [v["stat"] for v in loo.values()]
    lo, hi = min(stats), max(stats)
    lo_ds = [d for d, v in loo.items() if v["stat"] == lo][0]
    hi_ds = [d for d, v in loo.items() if v["stat"] == hi][0]
    return f"r in [{lo:.2f}, {hi:.2f}] (lowest when {lo_ds} excluded, highest when {hi_ds} excluded)"


def build_correlation_table(results: dict, keys: dict, dataset_order: list) -> list:
    labels = list(keys.values())
    lines = [
        "| Row | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    row = ["Full n=7 (Pearson)"] + [fmt_stat(results[k]["full_pearson"], "r") for k in keys]
    lines.append("| " + " | ".join(row) + " |")

    row = ["Leave-one-out range, all 7 exclusions (Pearson)"] + [loo_range(results[k]["leave_one_out_pearson"]) for k in keys]
    lines.append("| " + " | ".join(row) + " |")

    row = ["PER excluded, n=6 (Pearson)"] + [fmt_stat(results[k]["per_excluded_pearson"], "r") for k in keys]
    lines.append("| " + " | ".join(row) + " |")

    row = ["Full n=7 (Spearman)"] + [fmt_stat(results[k]["full_spearman"], "rho") for k in keys]
    lines.append("| " + " | ".join(row) + " |")

    row = ["PER excluded, n=6 (Spearman)"] + [fmt_stat(results[k]["per_excluded_spearman"], "rho") for k in keys]
    lines.append("| " + " | ".join(row) + " |")

    return lines


def build_loo_detail_table(results: dict, keys: dict, dataset_order: list) -> list:
    labels = list(keys.values())
    lines = [
        "| Excluded dataset | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    for d in dataset_order:
        row = [d] + [fmt_stat(results[k]["leave_one_out_pearson"][d], "r") for k in keys]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _pooled_model_dirs() -> dict:
    model_dirs = {m: ARCHIVE / "Pooled-Embeddings" / m for m in BACKBONE_META}
    return {m: d for m, d in model_dirs.items() if (d / "scape_eval_results").exists()}


def compute_region_backbone_stats(dataset_order: list) -> dict:
    """Load every (region, backbone) pair once and return {region: {backbone:
    per_region_stats() dict}}, so every aggregate below (macro-MAE(1-6),
    coverage-restricted macro-MAE, level coverage) is derived from a single
    read of the underlying pkl files instead of reloading them per metric."""
    model_dirs = _pooled_model_dirs()
    out = {d: {} for d in dataset_order}
    for region in dataset_order:
        for name, model_dir in model_dirs.items():
            df = load_region_df(model_dir, region)
            if df is None:
                continue
            out[region][name] = per_region_stats(df)
    return out


def compute_level_coverage(region_backbone_stats: dict, dataset_order: list, min_n: int = 1) -> dict:
    """n per true polyphony level 0-6 is backbone-invariant within a region
    (ground truth only), so it's read off any one backbone's stats per
    region. min_n sets the support threshold for a level to count as
    "covered" (default: any nonzero support)."""
    coverage = {}
    for region in dataset_order:
        s = next(iter(region_backbone_stats[region].values()))
        n_by_level = {l: s[f"n_{l}"] for l in range(7)}
        coverage[region] = {
            "n_by_level": n_by_level,
            "n_levels_with_data": sum(1 for l in range(1, 7) if n_by_level[l] >= min_n),
        }
    return coverage


def common_levels(coverage: dict, dataset_order: list, min_n: int = 1) -> list:
    """Levels 1-6 with n >= min_n in every one of the 7 datasets -- the
    level set a per-dataset macro-MAE comparison can use without the
    unequal-coverage confound (a dataset missing a level, or averaging over
    a level with only a handful of clips, biases its macro-MAE in a way
    that has nothing to do with genuine polyphonic difficulty)."""
    return [l for l in range(1, 7) if all(coverage[d]["n_by_level"][l] >= min_n for d in dataset_order)]


def compute_macro_mae_by_dataset(region_backbone_stats: dict, dataset_order: list, levels: list) -> tuple[dict, dict]:
    """Per-dataset macro-MAE over `levels`, backbone-averaged across all 13
    Pooled-Embeddings backbones. Returns (mean_by_dataset, per_backbone_by_dataset)
    where per_backbone_by_dataset[dataset] is the list of per-backbone macro-MAE
    values used for the mean (and for the per-dataset SD reported alongside it)."""
    per_backbone_by_dataset = {d: [] for d in dataset_order}
    for region in dataset_order:
        for stats in region_backbone_stats[region].values():
            vals = [stats[f"mae_{l}"] for l in levels if stats[f"n_{l}"] > 0]
            per_backbone_by_dataset[region].append(float(pd.Series(vals).mean()) if vals else float("nan"))

    mean_by_dataset = {d: float(pd.Series(v).mean()) for d, v in per_backbone_by_dataset.items()}
    return mean_by_dataset, per_backbone_by_dataset


def build_markdown(results: dict, results_common: dict, dataset_order: list, macro_mae_by_dataset: dict,
                    macro_mae_sd_by_dataset: dict, plain_mae_by_dataset: dict, level_coverage: dict,
                    common_level_set: list, common_mae_by_dataset: dict, common_mae_sd_by_dataset: dict,
                    strict_common_level_set: list, strict_results: dict) -> str:
    lines = [
        "# RQ5 -- \"What drives soundscape data difficulty?\" recomputed with macro-MAE, "
        "confound-checked",
        "",
        "Per-dataset macro-MAE(1-6): region-specific (Pooled-Embeddings) head, unweighted mean of per-level "
        "MAE over true levels 1-6 (dropping the dominant zero-polyphony class and the too-thin levels 7-8), "
        "computed per backbone via `per_region_stats()` (reused unmodified from "
        "`compute_soundscape_per_level_accuracy.py`), then averaged across all 13 backbones -- the same "
        "backbone set as `rq5_polyphony_robustness.md`'s plain-MAE baseline, and the same per-region "
        "macro-MAE(1-6) computation as `rq1_soundscape_ranking.md`, but aggregated per dataset "
        "(backbone-averaged) here instead of per backbone (region-averaged) there.",
        "",
        "## Per-dataset macro-MAE(1-6) vs. plain MAE",
        "",
        "| Dataset | Macro-MAE(1-6), 13-backbone mean | Macro-MAE(1-6) SD across 13 backbones | Plain MAE, "
        "13-backbone mean (rq4_fig3 / rq5_polyphony_robustness baseline) |",
        "|---|---|---|---|",
    ]
    for d in dataset_order:
        lines.append(
            f"| {d} | {macro_mae_by_dataset[d]:.4f} | {macro_mae_sd_by_dataset[d]:.4f} | {plain_mae_by_dataset[d]:.4f} |"
        )

    macro_vals = [macro_mae_by_dataset[d] for d in dataset_order]
    plain_vals = [plain_mae_by_dataset[d] for d in dataset_order]
    macro_sd_vals = [macro_mae_sd_by_dataset[d] for d in dataset_order]
    lines += [
        "",
        f"Per-dataset macro-MAE(1-6) range (min-max across the 7 datasets): "
        f"{min(macro_vals):.4f}-{max(macro_vals):.4f}. Per-dataset plain-MAE range: "
        f"{min(plain_vals):.4f}-{max(plain_vals):.4f}.",
        "",
        f"Per-dataset SD of macro-MAE(1-6) across the 13 backbones ranges "
        f"{min(macro_sd_vals):.4f}-{max(macro_sd_vals):.4f} across the 7 datasets.",
        "",
        "**On the quoted \"0.27-1.42\" SD range**: this script could not locate a repo-computable plain-MAE "
        "quantity that reproduces 0.27-1.42 exactly. Candidates checked and their actual values: (a) "
        "per-dataset mean plain MAE range across the 7 datasets is 0.5690-1.7018, not 0.27-1.42; (b) "
        "per-dataset SD of plain MAE across the 13 backbones ranges 0.1483-0.4270 (HSN lowest, PER "
        "highest), not 0.27-1.42; (c) the raw min/max of every individual backbone x dataset plain-MAE "
        "value (91 cells) is 0.3708-2.3913, not 0.27-1.42 either. None of the three interpretations checked "
        "reproduces the quoted figures, so this claim is flagged as unreproducible from this repo's data "
        "under plain MAE, independent of the macro-MAE switch -- treat the macro-MAE SD figures above (and "
        "the correlation recomputation below, which does not depend on resolving this discrepancy) as the "
        "actionable output of this check.",
        "",
        "## Per-dataset level coverage (confound for cross-dataset macro-MAE comparison)",
        "",
        "Ground truth (hence n per true level) is backbone-invariant within a region, so this table is read "
        "from a single backbone. Unlike the per-*backbone* macro-MAE(1-6) comparison in "
        "`rq1_soundscape_ranking.md` (where level coverage is fixed per region and so cancels out across "
        "backbones), a per-*dataset* macro-MAE(1-6) comparison is directly confounded by how many of levels "
        "1-6 each dataset actually has clips at -- a dataset missing the hardest levels is averaging over an "
        "easier subset of levels than a dataset with full coverage, even though both numbers are labeled "
        "\"macro-MAE(1-6)\".",
        "",
        "| Dataset | n(level 1) | n(level 2) | n(level 3) | n(level 4) | n(level 5) | n(level 6) | Levels "
        "1-6 with data | Ratio polyphonic |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    stats_for_coverage = json.loads(STATS_JSON.read_text())["subsets"]
    for d in dataset_order:
        cov = level_coverage[d]
        n = cov["n_by_level"]
        lines.append(
            f"| {d} | {n[1]} | {n[2]} | {n[3]} | {n[4]} | {n[5]} | {n[6]} | {cov['n_levels_with_data']}/6 | "
            f"{difficulty_value(stats_for_coverage, d, 'ratio_polyp'):.2f} |"
        )
    lines += [
        "",
        "HSN has clips at only 3 of the 6 levels (1-3, none at 4-6); NES and UHH have clips at 5 of 6 (none "
        "at 6, and NES/UHH's level-5 count is a thin n=4); POW, SSW, SNE, PER have data at all 6 levels, "
        "though SSW's level-6 n=10 and SNE's level-6 n=6 are themselves thin. The three lowest-polyphony "
        "datasets (HSN, NES, UHH -- ratio_polyp 0.05-0.15) are exactly the ones with incomplete level "
        "coverage, and UHH/NES also carry a level-5 macro-average driven by only 4 clips -- both push their "
        "macro-MAE(1-6) up in a way that is not really about polyphonic complexity but about macro-averaging "
        "over a level set that differs by dataset and, for the thin levels, is itself noisy. This is very "
        "likely a major contributor to the correlation collapse in Table 1/1b below, on top of (or instead "
        "of) the low-vs-high-polyphony ordering genuinely reversing under macro-MAE.",
        "",
        "## Table 1 -- Difficulty metric correlations with macro-MAE(1-6), unequal level coverage",
        "",
        "**This table inherits the level-coverage confound above -- read Table 3 below for the "
        "coverage-restricted version instead of treating this one as the answer.**",
        "",
    ]
    lines += build_correlation_table(results, DIFFICULTY_METRICS, dataset_order)
    lines += [
        "",
        "### Leave-one-out detail (all 7 exclusions, Pearson r)",
        "",
    ]
    lines += build_loo_detail_table(results, DIFFICULTY_METRICS, dataset_order)

    lines += [
        "",
        "### PER exclusion specifically",
        "",
    ]
    for key, label in DIFFICULTY_METRICS.items():
        full = results[key]["full_pearson"]
        per_excl = results[key]["per_excluded_pearson"]
        delta = per_excl["stat"] - full["stat"]
        direction = "weakens" if delta < -0.02 else ("strengthens" if delta > 0.02 else "stays roughly stable")
        lines.append(
            f"- **{label}**: r={full['stat']:.2f} (n=7) -> r={per_excl['stat']:.2f} (n=6, PER excluded), "
            f"p={full['p_value']:.3f} -> p={per_excl['p_value']:.3f} -- correlation {direction} when PER "
            "is excluded."
        )

    lines += [
        "",
        "## Table 1b -- Macro-MAE(1-6) vs. plain-MAE correlations, side by side",
        "",
        "Plain-MAE column reproduces `rq5_polyphony_robustness.md`'s own recomputed 13-backbone baseline "
        "(r=0.86/0.89/0.80 Pearson, rho=0.96/0.96/0.79 Spearman for ratio_polyp/mean_polyp/max_polyp -- "
        "**not** the narrative's quoted r=0.93/0.95/0.82, which this repo's data does not reproduce under "
        "any backbone-set variant checked there).",
        "",
        "| Metric | Pearson r, full n=7 (macro-MAE) | Pearson r, full n=7 (plain MAE) | Pearson r, "
        "PER-excl n=6 (macro-MAE) | Pearson r, PER-excl n=6 (plain MAE) | Spearman rho, full n=7 "
        "(macro-MAE) | Spearman rho, full n=7 (plain MAE) | Spearman rho, PER-excl n=6 (macro-MAE) | "
        "Spearman rho, PER-excl n=6 (plain MAE) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, label in DIFFICULTY_METRICS.items():
        b = PLAIN_MAE_BASELINE[key]
        r = results[key]
        lines.append(
            f"| {label} | {r['full_pearson']['stat']:.2f} | {b['pearson_full']:.2f} | "
            f"{r['per_excluded_pearson']['stat']:.2f} | {b['pearson_per_excl']:.2f} | "
            f"{r['full_spearman']['stat']:.2f} | {b['spearman_full']:.2f} | "
            f"{r['per_excluded_spearman']['stat']:.2f} | {b['spearman_per_excl']:.2f} |"
        )

    lines += [
        "",
        "## Table 2 -- Candidate-factor null-result checks",
        "",
        "Factors previously claimed to have no significant relationship with MAE, recomputed against "
        "macro-MAE(1-6) the same way.",
        "",
    ]
    lines += build_correlation_table(results, CANDIDATE_FACTORS, dataset_order)
    lines += [
        "",
        "### Leave-one-out detail (all 7 exclusions, Pearson r)",
        "",
    ]
    lines += build_loo_detail_table(results, CANDIDATE_FACTORS, dataset_order)

    common_str = "-".join(str(l) for l in common_level_set)
    lines += [
        "",
        f"## Table 3 -- Coverage-restricted macro-MAE(levels {common_str}): addressing the level-coverage confound",
        "",
        f"Levels {common_str} are the only ones of 1-6 with at least one clip in **every** one of the 7 "
        "datasets (see the level-coverage table above); levels outside this set are dropped from the "
        "per-backbone macro-average entirely, for every dataset, not just the ones missing them, so every "
        "dataset's number is now an average over the *same* level set instead of whatever subset it happens "
        "to have data for. This directly removes the confound flagged above, at the cost of dropping the "
        "higher, most-polyphonic levels from the comparison (which is exactly the range the level-coverage "
        "problem is concentrated in).",
        "",
        "| Dataset | Macro-MAE(" + common_str + "), 13-backbone mean | SD across 13 backbones |",
        "|---|---|---|",
    ]
    for d in dataset_order:
        lines.append(f"| {d} | {common_mae_by_dataset[d]:.4f} | {common_mae_sd_by_dataset[d]:.4f} |")
    lines += [
        "",
        f"### Table 3a -- Difficulty metric correlations with macro-MAE({common_str})",
        "",
    ]
    lines += build_correlation_table(results_common, DIFFICULTY_METRICS, dataset_order)
    lines += [
        "",
        f"#### Leave-one-out detail (all 7 exclusions, Pearson r, macro-MAE({common_str}))",
        "",
    ]
    lines += build_loo_detail_table(results_common, DIFFICULTY_METRICS, dataset_order)
    lines += [
        "",
        f"### Table 3b -- All three metrics side by side (plain MAE / macro-MAE(1-6) confounded / "
        f"macro-MAE({common_str}) coverage-restricted)",
        "",
        "| Metric | Pearson r, full n=7 (plain MAE) | Pearson r, full n=7 (macro-MAE 1-6) | Pearson r, "
        f"full n=7 (macro-MAE {common_str}) | Spearman rho, full n=7 (plain MAE) | Spearman rho, full n=7 "
        f"(macro-MAE 1-6) | Spearman rho, full n=7 (macro-MAE {common_str}) |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, label in DIFFICULTY_METRICS.items():
        b = PLAIN_MAE_BASELINE[key]
        r16 = results[key]
        rc = results_common[key]
        lines.append(
            f"| {label} | {b['pearson_full']:.2f} | {r16['full_pearson']['stat']:.2f} | "
            f"{rc['full_pearson']['stat']:.2f} | {b['spearman_full']:.2f} | "
            f"{r16['full_spearman']['stat']:.2f} | {rc['full_spearman']['stat']:.2f} |"
        )

    strict_str = "-".join(str(l) for l in strict_common_level_set) if strict_common_level_set else "none"
    lines += [
        "",
        f"### Robustness aside: a stricter support threshold (n>=30 per level per dataset) narrows the "
        f"common level set to {strict_str}",
        "",
        "Levels 1-3 above still include HSN's n=18 at level 3 -- thin for a single-region average. Requiring "
        "n>=30 in every dataset instead of merely n>=1 leaves only levels " + strict_str + ". Headline "
        "full-n=7 correlations under that stricter set:",
        "",
    ]
    if strict_common_level_set:
        for key, label in DIFFICULTY_METRICS.items():
            r = strict_results[key]
            lines.append(
                f"- **{label}**: {fmt_stat(r['full_pearson'], 'r')}, {fmt_stat(r['full_spearman'], 'rho')}"
            )
    else:
        lines.append("No level satisfies n>=30 in every one of the 7 datasets, so this check cannot be run.")

    lines += [
        "",
        "## Metric-robustness verdict",
        "",
    ]
    for key, label in DIFFICULTY_METRICS.items():
        b = PLAIN_MAE_BASELINE[key]
        r16 = results[key]
        rc = results_common[key]
        rho_delta_16 = abs(r16["full_spearman"]["stat"] - b["spearman_full"])
        r_delta_16 = abs(r16["full_pearson"]["stat"] - b["pearson_full"])
        verdict_16 = "metric-robust" if max(rho_delta_16, r_delta_16) <= 0.05 else "diverges meaningfully"
        rho_delta_c = abs(rc["full_spearman"]["stat"] - b["spearman_full"])
        r_delta_c = abs(rc["full_pearson"]["stat"] - b["pearson_full"])
        verdict_c = "metric-robust" if max(rho_delta_c, r_delta_c) <= 0.05 else "diverges meaningfully"
        lines.append(
            f"- **{label}**: plain MAE r={b['pearson_full']:.2f}/rho={b['spearman_full']:.2f} -> macro-MAE(1-6, "
            f"confounded) r={r16['full_pearson']['stat']:.2f}/rho={r16['full_spearman']['stat']:.2f} "
            f"(**{verdict_16}**) -> macro-MAE({common_str}, coverage-restricted) "
            f"r={rc['full_pearson']['stat']:.2f}/rho={rc['full_spearman']['stat']:.2f} (**{verdict_c}**)."
        )

    lines += [
        "",
        "## Caveat",
        "",
        "n=7 (n=6 with PER excluded) is small for any correlation-based claim regardless of which MAE "
        "definition is used -- treat every result here as suggestive rather than confirmatory, consistent "
        "with the small-n caution already applied throughout RQ1/RQ4/RQ5 verification. Table 1/1b (macro-MAE "
        "1-6) carries the level-coverage confound documented above and should not be read as the answer on "
        "its own; Table 3/3a/3b (coverage-restricted) addresses that confound but at the cost of dropping "
        "levels 4-6 -- exactly the range where polyphonic difficulty is most extreme -- from the comparison, "
        "so it is a different, narrower question (\"does difficulty at levels 1-" + common_str.split('-')[-1] +
        " track polyphony\") rather than a strictly more correct answer to the original one. If the "
        "coverage-restricted correlations still diverge from the plain-MAE ones, that divergence is real and "
        "should be flagged rather than assumed away -- the qualitative \"difficulty is associated with "
        "polyphonic complexity\" conclusion should be read as holding only to the extent the metrics agree.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq5"))
    args = parser.parse_args()

    stats = json.loads(STATS_JSON.read_text())["subsets"]
    dataset_order = REGIONAL_DATASETS

    region_backbone_stats = compute_region_backbone_stats(dataset_order)
    level_coverage = compute_level_coverage(region_backbone_stats, dataset_order, min_n=1)

    macro_mae_by_dataset, per_backbone_by_dataset = compute_macro_mae_by_dataset(
        region_backbone_stats, dataset_order, levels=list(range(1, 7)))
    macro_mae_sd_by_dataset = {d: float(pd.Series(v).std()) for d, v in per_backbone_by_dataset.items()}

    common_level_set = common_levels(level_coverage, dataset_order, min_n=1)
    common_mae_by_dataset, common_per_backbone_by_dataset = compute_macro_mae_by_dataset(
        region_backbone_stats, dataset_order, levels=common_level_set)
    common_mae_sd_by_dataset = {d: float(pd.Series(v).std()) for d, v in common_per_backbone_by_dataset.items()}

    strict_coverage = compute_level_coverage(region_backbone_stats, dataset_order, min_n=30)
    strict_common_level_set = common_levels(strict_coverage, dataset_order, min_n=30)
    if strict_common_level_set:
        strict_mae_by_dataset, _ = compute_macro_mae_by_dataset(
            region_backbone_stats, dataset_order, levels=strict_common_level_set)

    pooled_long = load_study("pooled")
    plain_sub = filter_long(pooled_long, source=SCAPE_SOURCE, metric="range_mae", head="reg", dataset=REGIONAL_DATASETS)
    plain_mae_by_dataset = plain_sub.groupby("dataset")["value"].mean().reindex(REGIONAL_DATASETS).to_dict()

    results = {}
    results_common = {}
    strict_results = {}
    for key in DIFFICULTY_METRICS:
        values_by_dataset = {d: difficulty_value(stats, d, key) for d in dataset_order}
        results[key] = analyze_property(values_by_dataset, macro_mae_by_dataset, dataset_order)
        results_common[key] = analyze_property(values_by_dataset, common_mae_by_dataset, dataset_order)
        if strict_common_level_set:
            strict_results[key] = analyze_property(values_by_dataset, strict_mae_by_dataset, dataset_order)
    for key in CANDIDATE_FACTORS:
        values_by_dataset = {d: factor_value(stats, d, key) for d in dataset_order}
        results[key] = analyze_property(values_by_dataset, macro_mae_by_dataset, dataset_order)
        results_common[key] = analyze_property(values_by_dataset, common_mae_by_dataset, dataset_order)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq5_macro_mae_difficulty_robustness.json"
    out_json.write_text(json.dumps(
        {
            "macro_mae_by_dataset": {d: round(float(v), 4) for d, v in macro_mae_by_dataset.items()},
            "macro_mae_sd_by_dataset": {d: round(float(v), 4) for d, v in macro_mae_sd_by_dataset.items()},
            "plain_mae_by_dataset": {d: round(float(v), 4) for d, v in plain_mae_by_dataset.items()},
            "plain_mae_baseline_correlations": PLAIN_MAE_BASELINE,
            "level_coverage": level_coverage,
            "common_level_set": common_level_set,
            "common_mae_by_dataset": {d: round(float(v), 4) for d, v in common_mae_by_dataset.items()},
            "common_mae_sd_by_dataset": {d: round(float(v), 4) for d, v in common_mae_sd_by_dataset.items()},
            "strict_common_level_set": strict_common_level_set,
            "results": results,
            "results_common": results_common,
            "strict_results": strict_results,
        },
        indent=2))
    print(f"Wrote {out_json}")

    md = build_markdown(results, results_common, dataset_order, macro_mae_by_dataset, macro_mae_sd_by_dataset,
                         plain_mae_by_dataset, level_coverage, common_level_set, common_mae_by_dataset,
                         common_mae_sd_by_dataset, strict_common_level_set, strict_results)
    out_md = args.out_dir / "rq5_macro_mae_difficulty_robustness.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
