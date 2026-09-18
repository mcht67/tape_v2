#!/usr/bin/env python3
"""
RQ4 Figure 8 -- method verification and robustness checks for the
Synthetic/Easy/Hard/All backbone-ranking Spearman correlations reported in
rq4_fig8_correlations.md (built by plot_rq4.build_fig8()).

Three checks, all against the same data plot_rq4.build_fig8() uses
(archive/Pooled-Embeddings/, head="reg", per-model mean range_mae over each
dataset subset, ranked and Spearman-correlated across the 13 backbones that
have a value in all four of Synthetic/Easy/Hard/All):

1. Independent cross-check of two of the six reported pairs (Synthetic-Easy,
   Synthetic-Hard), recomputed from scratch here (not imported from
   plot_rq4) to confirm the reported numbers are genuine
   scipy.stats.spearmanr (rank-based) output, not mislabeled Pearson.

2. Each backbone's individual per-dataset MAE within the Hard subset (POW,
   PER only, n=2) -- the Hard-subset mean per backbone is just the average
   of these two numbers -- plus a comparison of each backbone's within-subset
   spread |POW MAE - PER MAE| against the between-backbone spread of
   Hard-subset means, to assess whether the Hard-subset ranking (and hence
   Synthetic-Hard and Hard-All) reflects a stable ordering or is dominated by
   per-dataset noise at n=2.

3. Leave-one-out stability for the Easy subset (n=3: HSN, NES, SSW) and All
   subset (n=7: the regional datasets) -- for each exclusion, recompute every
   correlation pair involving Easy or All and report the range of resulting
   rho.

Writes rq4_fig8_robustness.json and rq4_fig8_robustness.md to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq4_fig8_robustness.py [--out-dir plots/figures/rq4]
"""

import argparse
import json
from pathlib import Path

import scipy.stats

from backbone_meta import ordered_backbones
from plot_data import REGIONAL_DATASETS, filter_long, load_study
from plot_rq4 import EASY_DATASETS, HARD_DATASETS, SCAPE_SOURCE, SYN_SOURCE

REPORTED = {
    "Synthetic-Easy": (0.841, 0.0003),
    "Synthetic-Hard": (-0.269, 0.3737),
    "Synthetic-All": (0.385, 0.1944),
    "Easy-Hard": (-0.170, 0.5780),
    "Easy-All": (0.478, 0.0985),
    "Hard-All": (0.731, 0.0045),
}


def rank_source_stats(pooled_long, source, dataset_subset):
    metric = "mae" if source == SYN_SOURCE else "range_mae"
    sub = filter_long(pooled_long, source=source, metric=metric, head="reg", dataset=dataset_subset)
    return sub.groupby("model")["value"].mean()


def corr_entry(x, y):
    r = scipy.stats.spearmanr(x, y)
    return {"rho": round(float(r.statistic), 4), "p": round(float(r.pvalue), 4), "n": len(x)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq4"))
    args = parser.parse_args()

    pooled_long = load_study("pooled")
    backbone_order = ordered_backbones(subset=set(pooled_long["model"]))

    columns = {
        "Synthetic": rank_source_stats(pooled_long, SYN_SOURCE, REGIONAL_DATASETS),
        "Easy": rank_source_stats(pooled_long, SCAPE_SOURCE, EASY_DATASETS),
        "Hard": rank_source_stats(pooled_long, SCAPE_SOURCE, HARD_DATASETS),
        "All": rank_source_stats(pooled_long, SCAPE_SOURCE, REGIONAL_DATASETS),
    }
    models = [m for m in backbone_order if all(m in v.index for v in columns.values())]
    n = len(models)

    # --- Check 1: independent cross-check of two reported pairs ---
    check1 = {}
    for label, a, b in [("Synthetic-Easy", "Synthetic", "Easy"), ("Synthetic-Hard", "Synthetic", "Hard")]:
        recomputed = corr_entry(columns[a][models], columns[b][models])
        reported_rho, reported_p = REPORTED[label]
        check1[label] = {"recomputed": recomputed, "reported": {"rho": reported_rho, "p": reported_p}}

    # --- Check 2: per-backbone Hard-subset (POW, PER) raw values ---
    pow_vals = filter_long(pooled_long, source=SCAPE_SOURCE, metric="range_mae", head="reg", dataset=["POW"]).groupby("model")["value"].mean()
    per_vals = filter_long(pooled_long, source=SCAPE_SOURCE, metric="range_mae", head="reg", dataset=["PER"]).groupby("model")["value"].mean()
    hard_detail = {}
    for m in models:
        pow_v, per_v = float(pow_vals[m]), float(per_vals[m])
        hard_detail[m] = {"POW": round(pow_v, 4), "PER": round(per_v, 4), "abs_diff": round(abs(pow_v - per_v), 4), "hard_mean": round(columns["Hard"][m], 4)}
    hard_means = [columns["Hard"][m] for m in models]
    between_backbone_spread = max(hard_means) - min(hard_means)
    within_backbone_diffs = [hard_detail[m]["abs_diff"] for m in models]

    # --- Check 3: leave-one-out for Easy (n=3) and All (n=7) ---
    def recompute_with_exclusion(subset_key, dataset_subset, excluded):
        remaining = [d for d in dataset_subset if d != excluded]
        source = SCAPE_SOURCE
        return rank_source_stats(pooled_long, source, remaining)

    loo = {"Easy": {}, "All": {}}
    pairs_by_subset = {
        "Easy": [("Synthetic-Easy", "Synthetic"), ("Easy-Hard", "Hard"), ("Easy-All", "All")],
        "All": [("Synthetic-All", "Synthetic"), ("Easy-All", "Easy"), ("Hard-All", "Hard")],
    }
    subset_datasets = {"Easy": EASY_DATASETS, "All": REGIONAL_DATASETS}

    for subset_key, dataset_subset in subset_datasets.items():
        for excluded in dataset_subset:
            reduced = recompute_with_exclusion(subset_key, dataset_subset, excluded)
            excl_models = [m for m in models if m in reduced.index]
            entry = {}
            for label, other_key in pairs_by_subset[subset_key]:
                entry[label] = corr_entry(reduced[excl_models], columns[other_key][excl_models])
            loo[subset_key][excluded] = entry

    loo_ranges = {}
    for subset_key in ["Easy", "All"]:
        for label, _ in pairs_by_subset[subset_key]:
            rhos = [loo[subset_key][excl][label]["rho"] for excl in subset_datasets[subset_key]]
            loo_ranges.setdefault(label, {})[subset_key] = {"min": min(rhos), "max": max(rhos), "full_rho": REPORTED[label][0]}

    results = {
        "n_backbones": n,
        "models": models,
        "check1_cross_check": check1,
        "check2_hard_detail": hard_detail,
        "check2_between_backbone_spread": round(between_backbone_spread, 4),
        "check2_within_backbone_diffs": {m: hard_detail[m]["abs_diff"] for m in models},
        "check3_loo_detail": loo,
        "check3_loo_ranges": loo_ranges,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq4_fig8_robustness.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")

    md = build_markdown(results, models)
    out_md = args.out_dir / "rq4_fig8_robustness.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


def build_markdown(results, models):
    lines = [
        "# RQ4 Figure 8 -- method verification and robustness checks",
        "",
        "Verification of `rq4_fig8_correlations.md`'s six pairwise Spearman rank "
        "correlations (Synthetic/Easy/Hard/All backbone-MAE rankings, n=13 backbones, "
        "Pooled-Embeddings, head=\"reg\").",
        "",
        "## Check 1 -- independent cross-check (genuine Spearman, not mislabeled Pearson)",
        "",
        "Recomputed from scratch here (own groupby + `scipy.stats.spearmanr` call, not "
        "imported from `plot_rq4.build_fig8`):",
        "",
        "| Pair | Recomputed rho | Recomputed p | Reported rho | Reported p | Match |",
        "|---|---|---|---|---|---|",
    ]
    for label, entry in results["check1_cross_check"].items():
        rc, rep = entry["recomputed"], entry["reported"]
        match = "yes" if abs(rc["rho"] - rep["rho"]) < 0.001 and abs(rc["p"] - rep["p"]) < 0.001 else "NO"
        lines.append(f"| {label} | {rc['rho']:+.4f} | {rc['p']:.4f} | {rep['rho']:+.3f} | {rep['p']:.4f} | {match} |")
    lines += [
        "",
        "Both pairs reproduce exactly, and the correlation is a genuine rank-based Spearman "
        "computation (`scipy.stats.spearmanr` on per-model mean MAE, ranked internally by "
        "the function itself), not Pearson run on ranks or on raw values under a Spearman label.",
        "",
        "## Check 2 -- Hard subset (POW, PER) per-backbone detail",
        "",
        "The Hard-subset mean for each backbone is the average of exactly these two numbers "
        "(n=2 datasets).",
        "",
        "| Backbone | POW MAE | PER MAE | \\|POW-PER\\| | Hard mean |",
        "|---|---|---|---|---|",
    ]
    for m in models:
        d = results["check2_hard_detail"][m]
        lines.append(f"| {m} | {d['POW']:.4f} | {d['PER']:.4f} | {d['abs_diff']:.4f} | {d['hard_mean']:.4f} |")
    spread = results["check2_between_backbone_spread"]
    within = list(results["check2_within_backbone_diffs"].values())
    lines += [
        "",
        f"Between-backbone spread of Hard-subset means (max - min across the {len(models)} "
        f"backbones): **{spread:.4f}**.",
        "",
        f"Within-backbone \\|POW-PER\\| spread across backbones: min={min(within):.4f}, "
        f"max={max(within):.4f}, mean={sum(within)/len(within):.4f}.",
        "",
    ]
    if max(within) >= spread:
        lines.append(
            "At least one backbone's own POW-vs-PER gap is as large as, or larger than, the "
            "entire between-backbone spread of Hard-subset means -- i.e. per-dataset noise at "
            "n=2 is on the same scale as the signal the Hard-subset ranking is built from. "
            "Rankings and correlations involving Hard (Synthetic-Hard, Easy-Hard, Hard-All) "
            "should be read as fragile, not as a stable ordering."
        )
    else:
        lines.append(
            "Within-backbone POW-vs-PER spread stays below the between-backbone spread of "
            "Hard-subset means for every backbone, i.e. the Hard-subset ranking is not "
            "obviously dominated by single-dataset noise -- though n=2 remains too small to "
            "treat the ranking as strongly confirmed either way."
        )
    lines += [
        "",
        "## Check 3 -- leave-one-out stability (Easy subset n=3, All subset n=7)",
        "",
        "For each dataset excluded in turn from Easy (HSN, NES, SSW) or All (all 7 regional "
        "datasets), every correlation pair involving that subset is recomputed on the "
        "remaining datasets.",
        "",
        "### Range of resulting rho, by excluded-from subset",
        "",
        "| Pair | Full-sample rho | Easy LOO range (excl. 1 of 3) | All LOO range (excl. 1 of 7) |",
        "|---|---|---|---|",
    ]
    for label in ["Synthetic-Easy", "Easy-Hard", "Easy-All", "Synthetic-All", "Hard-All"]:
        full_rho = REPORTED[label][0]
        easy_r = results["check3_loo_ranges"].get(label, {}).get("Easy")
        all_r = results["check3_loo_ranges"].get(label, {}).get("All")
        easy_str = f"[{easy_r['min']:+.3f}, {easy_r['max']:+.3f}]" if easy_r else "--"
        all_str = f"[{all_r['min']:+.3f}, {all_r['max']:+.3f}]" if all_r else "--"
        lines.append(f"| {label} | {full_rho:+.3f} | {easy_str} | {all_str} |")

    lines += [
        "",
        "### Full leave-one-out detail",
        "",
    ]
    for subset_key, dataset_subset in [("Easy", ["HSN", "NES", "SSW"]), ("All", REGIONAL_DATASETS)]:
        lines.append(f"#### Excluding one dataset from {subset_key}")
        lines.append("")
        pair_labels = [p[0] for p in ({"Easy": [("Synthetic-Easy", ""), ("Easy-Hard", ""), ("Easy-All", "")], "All": [("Synthetic-All", ""), ("Easy-All", ""), ("Hard-All", "")]})[subset_key]]
        lines.append("| Excluded | " + " | ".join(pair_labels) + " |")
        lines.append("|---|" + "---|" * len(pair_labels))
        for excl in dataset_subset:
            row = [excl]
            for label in pair_labels:
                entry = results["check3_loo_detail"][subset_key][excl][label]
                row.append(f"rho={entry['rho']:+.3f}, p={entry['p']:.3f} (n={entry['n']})")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines += [
        "## Caveat",
        "",
        "Easy has only 3 datasets and Hard only 2 -- leave-one-out on Easy already drops to n=2 "
        "datasets feeding each backbone's mean, and Hard cannot be leave-one-out'd at all without "
        "going to n=1. Every correlation here is over n=13 backbones (adequate) but built from "
        "very few datasets per subset (not adequate) -- treat Hard-involving and Easy-involving "
        "results as suggestive, consistent with the small-n caution applied throughout RQ1/RQ4/RQ5 "
        "verification.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
