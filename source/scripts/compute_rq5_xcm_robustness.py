#!/usr/bin/env python3
"""
RQ5 (additional) -- Leave-one-out and Spearman robustness check for the
xcm_head (frozen) and xcm_finetune_head soundscape-difficulty correlations
in rq5_matched_correlations.json (num_species, ratio_polyp, mean_polyp),
mirroring compute_rq5_polyphony_robustness.py's methodology for the
region-specific/He-et-al. correlations.

Per-dataset MAE: XCM head and XCM fine-tune head, soundscape test
(`range_mae`), each restricted to the same 4 backbones as XCM fine-tune head
(matched_backbones = FINETUNE_NAME_TO_CANONICAL.values(), the same
restriction as compute_rq5_matched_correlations.py) -- across the 7 regional
datasets (HSN, NES, SSW, UHH, SNE, POW, PER). This reproduces rq5_matched_
correlations.json's xcm_head/xcm_finetune_head soundscape num_species/
ratio_polyp/mean_polyp values exactly (unlike the polyphony-robustness
script's region-specific baseline, which uses all 11/13 backbones and does
not match the matched-correlations file).

Checks whether the manuscript's reported "significance crossover" (frozen:
num_species significant, mean_polyp not; fine-tuned: num_species loses
significance, mean_polyp gains it) is stable under leave-one-out and PER
exclusion, or an artifact of one extreme dataset (PER), given the adjacent
region-specific/He-et-al. correlations already showed PER-driven fragility
of similar magnitude (see rq5_polyphony_robustness.md).

Writes rq5_xcm_robustness.json and rq5_xcm_robustness.md to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq5_xcm_robustness.py [--out-dir plots/figures/rq5]
"""

import argparse
import json
import sys
from pathlib import Path

import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backbone_meta import FINETUNE_NAME_TO_CANONICAL
from compute_rq5_correlations import per_dataset_mae
from plot_data import REGIONAL_DATASETS, load_study
from plot_rq5 import SCAPE_SOURCE, STATS_JSON
from plot_rq5_dataset_properties import PROPERTIES

METRICS = {
    "num_species": "#Species (unique)",
    "ratio_polyp": "Ratio polyphonic",
    "mean_polyp": "Mean polyphony",
}
CONDITIONS = {
    "xcm_head": "XCM head (frozen)",
    "xcm_finetune_head": "XCM fine-tune head",
}

BASELINE = {
    ("xcm_head", "num_species"): (0.7722, 0.0419),
    ("xcm_finetune_head", "num_species"): (0.7276, 0.0638),
    ("xcm_head", "ratio_polyp"): (0.2854, 0.535),
    ("xcm_finetune_head", "ratio_polyp"): (0.7543, 0.0501),
    ("xcm_head", "mean_polyp"): (0.3328, 0.4657),
    ("xcm_finetune_head", "mean_polyp"): (0.7811, 0.0381),
}


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


def sig(p: float) -> str:
    return "significant" if p < 0.05 else "not significant"


def build_condition_table(results: dict, condition: str, dataset_order: list) -> list:
    labels = list(METRICS.values())
    lines = [
        "| Row | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    row = ["Full n=7 (Pearson)"] + [fmt_stat(results[condition][k]["full_pearson"], "r") for k in METRICS]
    lines.append("| " + " | ".join(row) + " |")

    row = ["Leave-one-out range, all 7 exclusions (Pearson)"] + [
        loo_range(results[condition][k]["leave_one_out_pearson"]) for k in METRICS
    ]
    lines.append("| " + " | ".join(row) + " |")

    row = ["PER excluded, n=6 (Pearson)"] + [fmt_stat(results[condition][k]["per_excluded_pearson"], "r") for k in METRICS]
    lines.append("| " + " | ".join(row) + " |")

    row = ["Full n=7 (Spearman)"] + [fmt_stat(results[condition][k]["full_spearman"], "rho") for k in METRICS]
    lines.append("| " + " | ".join(row) + " |")

    row = ["PER excluded, n=6 (Spearman)"] + [fmt_stat(results[condition][k]["per_excluded_spearman"], "rho") for k in METRICS]
    lines.append("| " + " | ".join(row) + " |")

    return lines


def build_loo_detail_table(results: dict, condition: str, dataset_order: list) -> list:
    labels = list(METRICS.values())
    lines = [
        "| Excluded dataset | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    for d in dataset_order:
        row = [d] + [fmt_stat(results[condition][k]["leave_one_out_pearson"][d], "r") for k in METRICS]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_markdown(results: dict, dataset_order: list, mae_by_dataset: dict) -> str:
    lines = [
        "# RQ5 -- Leave-one-out robustness and Spearman cross-check for XCM head "
        "soundscape difficulty-driver correlations",
        "",
        "Per-dataset MAE: XCM head (frozen) and XCM fine-tune head, soundscape test (`range_mae`), "
        "both restricted to the same 4 backbones as XCM fine-tune head (matched-backbone restriction, "
        "same as `rq5_matched_correlations.json`), across the 7 regional datasets. This reproduces "
        "`rq5_matched_correlations.json`'s xcm_head/xcm_finetune_head soundscape num_species/ratio_polyp/"
        "mean_polyp values exactly.",
        "",
        "## Baseline confirmation",
        "",
    ]
    for condition, clabel in CONDITIONS.items():
        for key, label in METRICS.items():
            recomputed = results[condition][key]["full_pearson"]
            reported_r, reported_p = BASELINE[(condition, key)]
            match = "matches" if (abs(recomputed["stat"] - reported_r) < 1e-3
                                   and abs(recomputed["p_value"] - reported_p) < 1e-3) else "DOES NOT MATCH"
            lines.append(
                f"- **{clabel} / {label}**: recomputed r={recomputed['stat']:.4f}, p={recomputed['p_value']:.4f} "
                f"vs. reported r={reported_r}, p={reported_p} -- {match}"
            )
    lines += [
        "",
        "## Per-dataset input values",
        "",
        "| Dataset | MAE (XCM head, frozen) | MAE (XCM fine-tune head) | #Species | Ratio polyphonic | "
        "Mean polyphony |",
        "|---|---|---|---|---|---|",
    ]
    stats = json.loads(STATS_JSON.read_text())["subsets"]
    num_species_getter = PROPERTIES["num_species"][1]
    ratio_polyp_getter = PROPERTIES["ratio_polyp"][1]
    mean_polyp_getter = PROPERTIES["mean_polyp"][1]
    for d in dataset_order:
        lines.append(
            f"| {d} | {mae_by_dataset['xcm_head'][d]:.4f} | {mae_by_dataset['xcm_finetune_head'][d]:.4f} | "
            f"{num_species_getter(stats[d])} | {ratio_polyp_getter(stats[d]):.2f} | "
            f"{mean_polyp_getter(stats[d]):.2f} |"
        )

    for condition, clabel in CONDITIONS.items():
        lines += [
            "",
            f"## Table -- {clabel}, soundscape difficulty-driver correlations",
            "",
        ]
        lines += build_condition_table(results, condition, dataset_order)
        lines += [
            "",
            "### Leave-one-out detail (all 7 exclusions, Pearson r)",
            "",
        ]
        lines += build_loo_detail_table(results, condition, dataset_order)

        lines += [
            "",
            "### PER exclusion specifically",
            "",
        ]
        for key, label in METRICS.items():
            full = results[condition][key]["full_pearson"]
            per_excl = results[condition][key]["per_excluded_pearson"]
            delta = per_excl["stat"] - full["stat"]
            direction = "weakens" if delta < -0.02 else ("strengthens" if delta > 0.02 else "stays roughly stable")
            flip = "CHANGES" if sig(full["p_value"]) != sig(per_excl["p_value"]) else "does not change"
            lines.append(
                f"- **{label}**: r={full['stat']:.2f} (n=7, {sig(full['p_value'])}) -> "
                f"r={per_excl['stat']:.2f} (n=6, PER excluded, {sig(per_excl['p_value'])}), "
                f"p={full['p_value']:.3f} -> p={per_excl['p_value']:.3f} -- correlation {direction} when PER "
                f"is excluded; significance status {flip}."
            )

    lines += [
        "",
        "## Crossover-survival check",
        "",
        "The manuscript sentence claims a significance crossover going from frozen XCM head to fine-tuned: "
        "num_species is significant (frozen) but loses significance (fine-tuned), while mean_polyp is not "
        "significant (frozen) but gains significance (fine-tuned). Checking both legs of this claim under "
        "PER exclusion:",
        "",
    ]
    ns_full_frozen = results["xcm_head"]["num_species"]["full_pearson"]
    ns_per_frozen = results["xcm_head"]["num_species"]["per_excluded_pearson"]
    ns_full_ft = results["xcm_finetune_head"]["num_species"]["full_pearson"]
    ns_per_ft = results["xcm_finetune_head"]["num_species"]["per_excluded_pearson"]
    mp_full_frozen = results["xcm_head"]["mean_polyp"]["full_pearson"]
    mp_per_frozen = results["xcm_head"]["mean_polyp"]["per_excluded_pearson"]
    mp_full_ft = results["xcm_finetune_head"]["mean_polyp"]["full_pearson"]
    mp_per_ft = results["xcm_finetune_head"]["mean_polyp"]["per_excluded_pearson"]

    leg1_full = sig(ns_full_frozen["p_value"]) != sig(ns_full_ft["p_value"])
    leg1_per = sig(ns_per_frozen["p_value"]) != sig(ns_per_ft["p_value"])
    leg2_full = sig(mp_full_frozen["p_value"]) != sig(mp_full_ft["p_value"])
    leg2_per = sig(mp_per_frozen["p_value"]) != sig(mp_per_ft["p_value"])

    lines.append(
        f"- **Leg 1 (num_species loses significance, frozen -> fine-tuned)**: full n=7: "
        f"p={ns_full_frozen['p_value']:.3f} ({sig(ns_full_frozen['p_value'])}) -> "
        f"p={ns_full_ft['p_value']:.3f} ({sig(ns_full_ft['p_value'])}) -- crossover "
        f"{'present' if leg1_full else 'absent'}. PER excluded, n=6: "
        f"p={ns_per_frozen['p_value']:.3f} ({sig(ns_per_frozen['p_value'])}) -> "
        f"p={ns_per_ft['p_value']:.3f} ({sig(ns_per_ft['p_value'])}) -- crossover "
        f"{'present' if leg1_per else 'absent'}."
    )
    lines.append(
        f"- **Leg 2 (mean_polyp gains significance, frozen -> fine-tuned)**: full n=7: "
        f"p={mp_full_frozen['p_value']:.3f} ({sig(mp_full_frozen['p_value'])}) -> "
        f"p={mp_full_ft['p_value']:.3f} ({sig(mp_full_ft['p_value'])}) -- crossover "
        f"{'present' if leg2_full else 'absent'}. PER excluded, n=6: "
        f"p={mp_per_frozen['p_value']:.3f} ({sig(mp_per_frozen['p_value'])}) -> "
        f"p={mp_per_ft['p_value']:.3f} ({sig(mp_per_ft['p_value'])}) -- crossover "
        f"{'present' if leg2_per else 'absent'}."
    )
    both_survive = leg1_full and leg1_per and leg2_full and leg2_per
    lines += [
        "",
        f"**Verdict**: the significance crossover {'SURVIVES' if both_survive else 'DOES NOT FULLY SURVIVE'} "
        "PER exclusion." + (
            "" if both_survive else
            " At least one leg's significance status flips once PER is removed -- see the per-leg detail above."
        ),
        "",
        "## Caveat",
        "",
        "n=7 (n=6 with PER excluded) is small for any correlation-based claim -- treat every result here "
        "as suggestive rather than confirmatory, regardless of outcome, consistent with the small-n "
        "caution already applied throughout RQ1/RQ4/RQ5 verification, and with the same caveat already "
        "applied to the region-specific/He-et-al. correlations in `rq5_polyphony_robustness.md`. "
        "Leave-one-out and Spearman checks reduce (but do not eliminate) the risk that a single extreme "
        "dataset is driving an otherwise fragile correlation; they do not make n=7 a large sample. If the "
        "significance crossover does not survive PER exclusion, the manuscript sentence should be revised "
        "to either drop the fine-tuning-shift framing entirely or explicitly caveat it as PER-dependent, "
        "consistent with how the He et al. difficulty-metric paragraph was already corrected for the same "
        "underlying fragility.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq5"))
    args = parser.parse_args()

    stats = json.loads(STATS_JSON.read_text())["subsets"]

    matched_backbones = list(FINETUNE_NAME_TO_CANONICAL.values())
    xg_long = load_study("xcm_gen", models=matched_backbones)
    ft_long = load_study("xcm_gen_ft", models=list(FINETUNE_NAME_TO_CANONICAL.keys()))

    mae_by_dataset = {
        "xcm_head": per_dataset_mae(xg_long, SCAPE_SOURCE, "range_mae").reindex(REGIONAL_DATASETS).to_dict(),
        "xcm_finetune_head": per_dataset_mae(ft_long, SCAPE_SOURCE, "range_mae").reindex(REGIONAL_DATASETS).to_dict(),
    }

    dataset_order = REGIONAL_DATASETS

    results = {}
    for condition in CONDITIONS:
        results[condition] = {}
        for key in METRICS:
            getter = PROPERTIES[key][1]
            values_by_dataset = {d: getter(stats[d]) for d in dataset_order}
            results[condition][key] = analyze_property(values_by_dataset, mae_by_dataset[condition], dataset_order)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq5_xcm_robustness.json"
    out_json.write_text(json.dumps(
        {
            "mae_by_dataset": {
                c: {d: round(float(v), 4) for d, v in m.items()} for c, m in mae_by_dataset.items()
            },
            "baseline": {f"{c}.{k}": {"r": v[0], "p": v[1]} for (c, k), v in BASELINE.items()},
            "results": results,
        },
        indent=2))
    print(f"Wrote {out_json}")

    md = build_markdown(results, dataset_order, mae_by_dataset)
    out_md = args.out_dir / "rq5_xcm_robustness.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
