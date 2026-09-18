#!/usr/bin/env python3
"""
RQ5 (additional) -- Does classification difficulty (BirdSet's own LT-scenario
AUROC, Rauch et al. 2025, Table 11) track the same regional-dataset
properties that predict our own region-specific soundscape MAE, or are the
two difficulty axes driven by different underlying properties?

BirdSet AUROC data (hardcoded below, BIRDSET_AUROC): mean across the 4
models in Table 11 that report an AUROC row for every column we need
(EfficientNet, ConvNext, AST, EAT -- Perch is excluded since Table 11 gives
no Val/POW row for it, W2V2 is excluded since the task's source list
specifies these 4). POW is BirdSet's Val column (not a Test column --
BirdSet uses POW for hyperparameter/checkpoint validation, not evaluation --
but it is one of our own 7 REGIONAL_DATASETS, so it is kept for full
7-dataset coverage; PER/NES/UHH/HSN/SSW/SNE are Test columns). Recomputed
and verified by hand against the source PDF (arXiv:2403.10380v6, Table 11,
page 31) before hardcoding -- e.g. PER: (0.71+0.72+0.72+0.64)/4 = 0.6975.
NBP is a Test column in Table 11 but is not one of our 7 REGIONAL_DATASETS,
so it is not used here.

"Difficulty" sign convention: BirdSet AUROC is higher-is-easier, so it is
inverted (1 - AUROC) to a "BirdSet difficulty" score before correlating,
making every variable in this script higher-is-harder (ratio_polyp,
mean_polyp, mean #species/segment, and region-specific MAE are already
higher-is-harder). A positive Spearman rho then always means "the two
difficulty measures agree on which datasets are hard", regardless of which
pair is being compared.

Our own metrics, all from the same sources as compute_rq5_correlations.py /
plot_rq5_dataset_properties.py (PROPERTIES, the single source of truth for
dataset-property definitions, and plot_rq5.py's region-specific soundscape
MAE, backbone-averaged across the same Panel-A backbone set used
throughout RQ5):
  - ratio_polyp: PROPERTIES["ratio_polyp"] (mean fraction of polyphonic
    segments).
  - mean_polyp: PROPERTIES["mean_polyp"] (mean polyphony level).
  - n_species: PROPERTIES["n_species"] (mean #species/segment -- the
    already-computed per-segment species-richness measure this task calls
    "num_species_per_segment").
  - num_species: PROPERTIES["num_species"] (#species, total unique species
    in the dataset's test set -- a dataset-level count, not per-segment;
    added alongside n_species since the two answer different questions,
    "how many species exist in this region" vs. "how many co-occur per
    segment").
  - region_specific_mae: per-dataset soundscape range_mae, mean across the
    Panel-A backbones (same as compute_rq5_correlations.py's
    "region_specific"/"soundscape" cell), i.e. our own model-based
    difficulty measure, to check whether BirdSet's classification
    difficulty and our own models' difficulty are even correlated with
    each other before comparing either to the polyphony-density measures.

Spearman (not Pearson) is used throughout since we only have BirdSet's
already-rounded, cross-4-model-averaged AUROC (not a distribution we can
assume linearity against) and n=7 is too small to justify an assumption of
linearity in either direction.

**Caveat**: n=7 is underpowered, the same caveat as the general-audio-
pretraining and per-dataset-kurtosis analyses elsewhere in this repo --
report every correlation below as suggestive, not confirmatory.

Writes rq5_birdset_auroc_correlation.json and
rq5_birdset_auroc_correlation.md to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq5_birdset_auroc_correlation.py [--out-dir plots/figures/rq5]
"""

import argparse
import json
import sys
from pathlib import Path

import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from compute_rq5_correlations import per_dataset_mae
from plot_data import REGIONAL_DATASETS, load_study
from plot_rq5 import SCAPE_SOURCE, STATS_JSON
from plot_rq5_dataset_properties import PROPERTIES

# BirdSet Table 11 (LT scenario), AUROC mean across EfficientNet, ConvNext,
# AST, EAT -- see module docstring for the per-dataset arithmetic check.
BIRDSET_AUROC = {
    "POW": 0.8125,
    "PER": 0.6975,
    "NES": 0.88,
    "UHH": 0.77,
    "HSN": 0.865,
    "SSW": 0.9175,
    "SNE": 0.825,
}

OUR_METRICS = {
    "ratio_polyp": "Ratio polyphonic",
    "mean_polyp": "Mean polyphony",
    "n_species": "Mean #species/segment",
    "num_species": "#Species (total, dataset-level)",
    "region_specific_mae": "Region-specific soundscape MAE (backbone-avg.)",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq5"))
    args = parser.parse_args()

    assert set(BIRDSET_AUROC) == set(REGIONAL_DATASETS), "BIRDSET_AUROC must cover exactly the 7 regional datasets"

    stats = json.loads(STATS_JSON.read_text())["subsets"]
    xg_models = mrt.discover_models(Path("archive/XCM-Generalization"))
    pooled_long = load_study("pooled", models=xg_models)
    mae = per_dataset_mae(pooled_long, SCAPE_SOURCE, "range_mae").reindex(REGIONAL_DATASETS)

    our_values = {
        "ratio_polyp": {d: PROPERTIES["ratio_polyp"][1](stats[d]) for d in REGIONAL_DATASETS},
        "mean_polyp": {d: PROPERTIES["mean_polyp"][1](stats[d]) for d in REGIONAL_DATASETS},
        "n_species": {d: PROPERTIES["n_species"][1](stats[d]) for d in REGIONAL_DATASETS},
        "num_species": {d: PROPERTIES["num_species"][1](stats[d]) for d in REGIONAL_DATASETS},
        "region_specific_mae": {d: float(mae[d]) for d in REGIONAL_DATASETS},
    }

    birdset_difficulty = {d: 1.0 - BIRDSET_AUROC[d] for d in REGIONAL_DATASETS}
    birdset_rank = sorted(REGIONAL_DATASETS, key=lambda d: birdset_difficulty[d])

    results = {}
    for metric_key in OUR_METRICS:
        vals = [our_values[metric_key][d] for d in REGIONAL_DATASETS]
        diffs = [birdset_difficulty[d] for d in REGIONAL_DATASETS]
        rho = scipy.stats.spearmanr(vals, diffs)
        results[metric_key] = {
            "spearman_rho": round(float(rho.statistic), 4),
            "p_value": round(float(rho.pvalue), 4),
            "n": len(REGIONAL_DATASETS),
        }

    ratio_polyp_rank = sorted(REGIONAL_DATASETS, key=lambda d: our_values["ratio_polyp"][d])
    mean_polyp_rank = sorted(REGIONAL_DATASETS, key=lambda d: our_values["mean_polyp"][d])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq5_birdset_auroc_correlation.json"
    out_json.write_text(json.dumps({
        "birdset_auroc": BIRDSET_AUROC,
        "birdset_difficulty_1_minus_auroc": birdset_difficulty,
        "our_values": our_values,
        "results": results,
        "birdset_difficulty_rank_hardest_first": list(reversed(birdset_rank)),
        "ratio_polyp_rank_hardest_first": list(reversed(ratio_polyp_rank)),
        "mean_polyp_rank_hardest_first": list(reversed(mean_polyp_rank)),
    }, indent=2))
    print(f"Wrote {out_json}")

    md = build_markdown(results, our_values, birdset_difficulty, birdset_rank, ratio_polyp_rank, mean_polyp_rank)
    out_md = args.out_dir / "rq5_birdset_auroc_correlation.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


def build_markdown(results, our_values, birdset_difficulty, birdset_rank, ratio_polyp_rank, mean_polyp_rank) -> str:
    lines = [
        "# RQ5 -- BirdSet classification difficulty vs. our own difficulty measures",
        "",
        "Spearman correlation between BirdSet's own LT-scenario classification difficulty "
        "(1 - mean AUROC across EfficientNet, ConvNext, AST, EAT; Rauch et al. 2025, Table 11) and "
        "each of our own per-dataset difficulty measures, across the 7 regional datasets (n=7). "
        "Both axes are oriented higher-is-harder (BirdSet AUROC is inverted to 1 - AUROC for this "
        "reason), so a positive rho means the two difficulty notions agree on which datasets are hard.",
        "",
        "**Caveat**: n=7 is underpowered, the same caveat as the general-audio-pretraining and "
        "per-dataset-kurtosis tests elsewhere in this repo -- read every number below as suggestive, "
        "not confirmatory.",
        "",
        "## Correlations",
        "",
        "| Our metric | Spearman rho vs. BirdSet difficulty | p-value | n |",
        "|---|---|---|---|",
    ]
    for metric_key, label in OUR_METRICS.items():
        r = results[metric_key]
        lines.append(f"| {label} | {r['spearman_rho']:+.3f} | {r['p_value']:.4f} | {r['n']} |")

    lines += [
        "",
        "## Per-dataset values",
        "",
        "| Dataset | BirdSet AUROC | BirdSet difficulty (1-AUROC) | Ratio polyphonic | Mean polyphony | "
        "Mean #species/segment | #Species (total) | Region-specific soundscape MAE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in sorted(REGIONAL_DATASETS, key=lambda d: birdset_difficulty[d]):
        lines.append(
            f"| {d} | {1 - birdset_difficulty[d]:.3f} | {birdset_difficulty[d]:+.3f} | "
            f"{our_values['ratio_polyp'][d]:.3f} | {our_values['mean_polyp'][d]:.3f} | "
            f"{our_values['n_species'][d]:.3f} | {our_values['num_species'][d]:.0f} | "
            f"{our_values['region_specific_mae'][d]:.3f} |"
        )

    lines += [
        "",
        "## Rank orderings (easiest to hardest)",
        "",
        "Side-by-side rank orderings make divergence/convergence visible directly, not just via the "
        "correlation coefficient above.",
        "",
        "| Rank | BirdSet difficulty | Ratio-polyphonic difficulty | Mean-polyphony difficulty |",
        "|---|---|---|---|",
    ]
    for i, (bd, rp, mp) in enumerate(zip(birdset_rank, ratio_polyp_rank, mean_polyp_rank), start=1):
        lines.append(f"| {i} (easiest) | {bd} | {rp} | {mp} |" if i == 1 else f"| {i} | {bd} | {rp} | {mp} |")

    lines += [
        "",
        "## Reading this",
        "",
        "If the correlations above are weak/non-significant and the rank orderings visibly reshuffle "
        "datasets between columns, that supports the idea that BirdSet's classification difficulty "
        "(species identity, confusability, and label noise) and our polyphony-density difficulty "
        "(overlapping simultaneous vocalizations) are two largely independent axes -- a dataset can be "
        "\"easy\" on one and \"hard\" on the other. If the rankings largely agree and the correlations "
        "are strong, that instead suggests both are picking up a shared underlying acoustic-complexity "
        "factor.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
