#!/usr/bin/env python3
"""
RQ5 (additional) -- Leave-one-out and Spearman robustness check for the
soundscape-difficulty correlations between per-dataset MAE and He et al.'s
polyphony difficulty metrics (ratio_polyp, mean_polyp, max_polyp, each the
"mean" summary stat from plots/data/polybirdmix_soundscape_stats.json's
polyphony_metric_summary -- see plot_rq5_dataset_properties.PROPERTIES for
the same getters used elsewhere in RQ5).

MAE is per-dataset mean range_mae, region-specific (Pooled-Embeddings) head,
mean across all 13 backbones (load_study("pooled") with no models=
restriction, same source pandas frame as plot_rq4.build_fig3's
rq4_fig3_difficulty_table.tex) -- across the 7 regional datasets (HSN, NES,
SSW, UHH, SNE, POW, PER).

This script does not reproduce the specific r=0.93/0.95/0.82 figures quoted
in the thesis narrative -- recomputing directly from this repo's own data
and pipeline (13-backbone pooled region-specific MAE, as instructed) gives
r=0.86/0.89/0.80 instead (see rq4_fig3_difficulty_table.tex, which already
reports the same ratio_polyp/mean_polyp values from the same 13-backbone
source). Every other backbone-set variant tried elsewhere in RQ5 (11
backbones unmatched, 4 backbones matched, top-5-by-MAE) also falls short of
matching all three of 0.93/0.95/0.82 simultaneously. The leave-one-out and
Spearman analysis below is therefore run on, and should be read against,
this script's own recomputed baseline (0.86/0.89/0.80), not the quoted
0.93/0.95/0.82 -- see the "Baseline discrepancy" section of the written
report for the full discussion.

Three candidate factors already claimed elsewhere as "no significant
relationship" with MAE are checked the same way: num_species (n_unique_
species; rq4_fig3_difficulty_table.tex already reports r=0.65, p=0.112),
n_train_segments (#training segments), and range_width (mean per-segment
polyphony range width = mean max_polyphony - min_polyphony per segment,
plots/data/polybirdmix_soundscape_stats.json's distributions.range_width).

Writes rq5_polyphony_robustness.json and rq5_polyphony_robustness.md to
--out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq5_polyphony_robustness.py [--out-dir plots/figures/rq5]
"""

import argparse
import json
import sys
from pathlib import Path

import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

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

REPORTED = {"ratio_polyp": (0.93, 0.003), "mean_polyp": (0.95, 0.001), "max_polyp": (0.82, 0.025)}


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
    """Full n=7 Pearson/Spearman, leave-one-out Pearson for each of the 7
    exclusions, and PER-excluded Pearson/Spearman (n=6), for one property."""
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


def build_markdown(results: dict, dataset_order: list, mae_by_dataset: dict) -> str:
    lines = [
        "# RQ5 -- Leave-one-out robustness and Spearman cross-check for soundscape "
        "difficulty (polyphony) correlations",
        "",
        "Per-dataset MAE: region-specific (Pooled-Embeddings) head, mean across all 13 backbones, "
        "soundscape test (`range_mae`), across the 7 regional datasets. Same source data as "
        "`rq4_fig3_difficulty_table.tex`.",
        "",
        "## Baseline discrepancy",
        "",
        "Recomputing directly from this repo's data with the instructed 13-backbone pooled MAE gives:",
        "",
    ]
    for key, label in DIFFICULTY_METRICS.items():
        recomputed = results[key]["full_pearson"]
        reported_r, reported_p = REPORTED[key]
        lines.append(
            f"- **{label}**: recomputed r={recomputed['stat']:.2f}, p={recomputed['p_value']:.3f} "
            f"vs. reported r={reported_r}, p={reported_p}"
        )
    lines += [
        "",
        "This does **not** exactly match the reported r=0.93/0.95/0.82 figures. Every backbone-set "
        "variant available elsewhere in this repo's RQ5 pipeline was checked (11 backbones unmatched, "
        "4 backbones backbone-matched, top-5-by-MAE) and none reproduces all three reported values "
        "simultaneously either -- the closest is the 11-backbone unmatched set, which reproduces "
        "max_polyp exactly (r=0.82, p=0.025) but not ratio_polyp/mean_polyp (r=0.87/0.90 there, vs. "
        "0.93/0.95 reported). The analysis below is therefore run on, and should be read against, this "
        "script's own 13-backbone recomputed baseline, not the quoted 0.93/0.95/0.82 -- the qualitative "
        "conclusions (robust vs. PER-driven) are what matters here, not exact digit-matching to the "
        "narrative text.",
        "",
        "## Per-dataset input values",
        "",
        "| Dataset | MAE (13-backbone mean) | Ratio polyphonic | Mean polyphony | Mean max-polyp | "
        "#Species | #Train segments | Range width |",
        "|---|---|---|---|---|---|---|---|",
    ]
    stats = json.loads(STATS_JSON.read_text())["subsets"]
    for d in dataset_order:
        lines.append(
            f"| {d} | {mae_by_dataset[d]:.4f} | {difficulty_value(stats, d, 'ratio_polyp'):.2f} | "
            f"{difficulty_value(stats, d, 'mean_polyp'):.2f} | {difficulty_value(stats, d, 'max_polyp'):.2f} | "
            f"{factor_value(stats, d, 'num_species')} | {factor_value(stats, d, 'n_train_segments')} | "
            f"{factor_value(stats, d, 'range_width'):.2f} |"
        )

    lines += [
        "",
        "## Table 1 -- Difficulty metric correlations with MAE",
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
        "## Table 2 -- Candidate-factor null-result checks",
        "",
        "Factors previously claimed to have no significant relationship with MAE, recomputed the same way.",
        "",
    ]
    lines += build_correlation_table(results, CANDIDATE_FACTORS, dataset_order)
    lines += [
        "",
        "### Leave-one-out detail (all 7 exclusions, Pearson r)",
        "",
    ]
    lines += build_loo_detail_table(results, CANDIDATE_FACTORS, dataset_order)

    lines += [
        "",
        "## Caveat",
        "",
        "n=7 (n=6 with PER excluded) is small for any correlation-based claim -- treat every result here "
        "as suggestive rather than confirmatory, regardless of outcome, consistent with the small-n "
        "caution already applied throughout RQ1/RQ4/RQ5 verification. Leave-one-out and Spearman checks "
        "reduce (but do not eliminate) the risk that a single extreme dataset is driving an otherwise "
        "fragile correlation; they do not make n=7 a large sample.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq5"))
    args = parser.parse_args()

    stats = json.loads(STATS_JSON.read_text())["subsets"]

    pooled_long = load_study("pooled")
    sub = filter_long(pooled_long, source=SCAPE_SOURCE, metric="range_mae", head="reg", dataset=REGIONAL_DATASETS)
    mae_series = sub.groupby("dataset")["value"].mean().reindex(REGIONAL_DATASETS)
    mae_by_dataset = mae_series.to_dict()

    dataset_order = REGIONAL_DATASETS

    results = {}
    for key in DIFFICULTY_METRICS:
        values_by_dataset = {d: difficulty_value(stats, d, key) for d in dataset_order}
        results[key] = analyze_property(values_by_dataset, mae_by_dataset, dataset_order)
    for key in CANDIDATE_FACTORS:
        values_by_dataset = {d: factor_value(stats, d, key) for d in dataset_order}
        results[key] = analyze_property(values_by_dataset, mae_by_dataset, dataset_order)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq5_polyphony_robustness.json"
    out_json.write_text(json.dumps(
        {
            "mae_by_dataset": {d: round(float(v), 4) for d, v in mae_by_dataset.items()},
            "reported_baseline": {k: {"r": v[0], "p": v[1]} for k, v in REPORTED.items()},
            "results": results,
        },
        indent=2))
    print(f"Wrote {out_json}")

    md = build_markdown(results, dataset_order, mae_by_dataset)
    out_md = args.out_dir / "rq5_polyphony_robustness.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
