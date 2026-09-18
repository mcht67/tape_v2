#!/usr/bin/env python3
"""
RQ1 -- Difficulty-robust backbone ranking, frozen backbone comparison.

Establishes a backbone ranking on soundscape data that is robust to
cross-dataset difficulty variance, and directly tests whether the apparent
"tight leading group" (Bird-MAE, Perch v2, both BirdNET backbones) is
statistically distinguishable from the rest of the ranking or from each
other -- same paired-difference methodology already used for RQ2
(`compute_rq2_paired_cell_analysis.py`) and RQ5 (`compute_rq5_kurtosis.py`).

Source data: `archive/Pooled-Embeddings/` (13 backbones, frozen backbone +
pooled-embedding head, regression formulation), source `soundscape_test`,
metric "range_mae", the 7 REGIONAL_DATASETS (soundscape has no XCM
counterpart); source `synthetic_mixture_test`, metric "mae", the 8
ALL_DATASETS -- same conventions as `compute_rq1_paired_cell_analysis.py` /
`compute_rq1_sign_test.py`.

Four checks:

1. Difficulty-robust ranking via paired dataset-level averaging: mean MAE
   per dataset is already the raw cell value (one value per backbone x
   dataset), so ranking backbones by their mean-across-datasets MAE is
   already paired/robust to shared difficulty variance -- every backbone
   sees the same 7 soundscape datasets. Confirms this matches the top 4 of
   `rq1_top5_soundscape_table.tex` (Table~\\ref{tab:rq1_top5_soundscape_test_reg})
   and extends it to all 13 backbones.

2. Pairwise paired-difference significance within the "tight leading group"
   {Bird-MAE, Perch v2, BirdNET v2.3, BirdNET v2.4} (6 pairs): Delta =
   (backbone A MAE) - (backbone B MAE), matched by dataset (n=7 soundscape
   datasets), paired t-test and Wilcoxon signed-rank test against zero.

3. Pairwise paired-difference significance between the tight leading group
   and the next-ranked backbones (whichever sit at ranks 5-7 per check 1):
   same paired-difference test as check 2, for each of the 4 x 3 = 12 pairs.

4. Spearman rank correlation between each backbone's synthetic-data rank and
   soundscape-data rank (all 13 backbones), using the paired dataset-averaged
   mean MAE from check 1's construction (synthetic side).

Sample-size caveat, same as RQ2/RQ5: n=7 soundscape datasets is small for
the pairwise comparisons in checks 2 and 3 -- treat those as exploratory/
directional. Checks 1 and 4 are more robust: they don't depend on a single
pairwise significance threshold to be informative.

Writes rq1_ranking_robustness.json and rq1_ranking_robustness.md to
--out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq1_ranking_robustness.py [--out-dir plots/figures/rq1]
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import BACKBONE_META
from plot_data import ALL_DATASETS, REGIONAL_DATASETS, filter_long, load_study

SOURCE = "synthetic_mixture_test"
SCAPE_SOURCE = "soundscape_test"
SYNTHETIC_METRIC = "mae"
SOUNDSCAPE_METRIC = "range_mae"

# The 4 backbones flagged in the narrative as a "tight leading group" on
# soundscape data -- the top 4 of `rq1_top5_soundscape_table.tex`.
TIGHT_GROUP = ["Bird-MAE-Huge", "perch_v2_cpu", "Birdnet_V2.3", "Birdnet_V2.4"]

N_NEXT = 3  # ranks 5, 6, 7 -- "next-ranked" backbones just outside the group


def fmt_mean_std(mean: float | None, std: float | None, decimals: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def diff_stats(diff: np.ndarray) -> dict:
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    return {
        "mean": float(diff.mean()) if n else None,
        "std": float(diff.std(ddof=1)) if n > 1 else None,
        "n": n,
    }


def paired_tests(diff: np.ndarray) -> dict:
    """Paired t-test and Wilcoxon signed-rank test of diff against zero --
    same convention as compute_rq2_paired_cell_analysis.py /
    compute_rq5_kurtosis.py. None for either test below the sample size it
    needs, or if Wilcoxon raises (e.g. all-zero differences)."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
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


def build_matrix(long_df: pd.DataFrame, source: str, metric: str, dataset_order: list,
                  row_order: list) -> pd.DataFrame:
    """(backbone x dataset) matrix of raw MAE values, regression head --
    same construction as plot_rq5.py's build_heatmap_matrix."""
    sub = filter_long(long_df, source=source, metric=metric, head="reg", dataset=dataset_order)
    matrix = sub.pivot_table(index="model", columns="dataset", values="value")
    return matrix.reindex(index=row_order, columns=dataset_order)


def pairwise_diff(matrix: pd.DataFrame, a: str, b: str) -> dict:
    diff = (matrix.loc[a] - matrix.loc[b]).to_numpy(dtype=float)
    diff = diff[~np.isnan(diff)]
    return {"a": a, "b": b, **diff_stats(diff), **paired_tests(diff)}


# ---------------------------------------------------------------------------
# Check 1 -- difficulty-robust ranking, all 13 backbones, soundscape data.
# ---------------------------------------------------------------------------

def check1_ranking(long_df: pd.DataFrame, backbone_order: list) -> dict:
    scape_matrix = build_matrix(long_df, SCAPE_SOURCE, SOUNDSCAPE_METRIC, REGIONAL_DATASETS, backbone_order)
    mean_mae = scape_matrix.mean(axis=1)
    sd_mae = scape_matrix.std(axis=1, ddof=1)
    rank = mean_mae.rank(method="min").astype(int)
    order = mean_mae.sort_values().index.tolist()
    matches_table = order[:4] == TIGHT_GROUP or set(order[:4]) == set(TIGHT_GROUP)
    return {
        "order": order,
        "mean_mae": mean_mae.to_dict(),
        "sd_mae": sd_mae.to_dict(),
        "rank": rank.to_dict(),
        "n_datasets": len(REGIONAL_DATASETS),
        "matches_top5_soundscape_table": bool(matches_table),
    }


# ---------------------------------------------------------------------------
# Check 2 -- pairwise significance within the tight leading group (6 pairs).
# ---------------------------------------------------------------------------

def check2_within_group(long_df: pd.DataFrame) -> dict:
    matrix = build_matrix(long_df, SCAPE_SOURCE, SOUNDSCAPE_METRIC, REGIONAL_DATASETS, TIGHT_GROUP)
    return {f"{a}__minus__{b}": pairwise_diff(matrix, a, b)
            for a, b in itertools.combinations(TIGHT_GROUP, 2)}


# ---------------------------------------------------------------------------
# Check 3 -- pairwise significance between the tight group and the
# next-ranked backbones (ranks 5..5+N_NEXT-1 from check 1).
# ---------------------------------------------------------------------------

def check3_vs_next(long_df: pd.DataFrame, ranking_order: list) -> dict:
    next_group = [m for m in ranking_order if m not in TIGHT_GROUP][:N_NEXT]
    matrix = build_matrix(long_df, SCAPE_SOURCE, SOUNDSCAPE_METRIC, REGIONAL_DATASETS,
                           TIGHT_GROUP + next_group)
    results = {f"{a}__minus__{b}": pairwise_diff(matrix, a, b)
               for a in TIGHT_GROUP for b in next_group}
    return {"next_group": next_group, "pairs": results}


# ---------------------------------------------------------------------------
# Check 4 -- Spearman rank correlation, synthetic vs. soundscape ranking,
# all 13 backbones.
# ---------------------------------------------------------------------------

def check4_rank_correlation(long_df: pd.DataFrame, backbone_order: list, soundscape_mean: pd.Series) -> dict:
    synth_matrix = build_matrix(long_df, SOURCE, SYNTHETIC_METRIC, ALL_DATASETS, backbone_order)
    synth_mean = synth_matrix.mean(axis=1)

    rho, p = scipy.stats.spearmanr(synth_mean.to_numpy(), soundscape_mean.reindex(backbone_order).to_numpy())
    ranks = {
        "synthetic": synth_mean.rank(method="min").astype(int).to_dict(),
        "soundscape": soundscape_mean.reindex(backbone_order).rank(method="min").astype(int).to_dict(),
    }
    return {
        "spearman_rho": float(rho), "spearman_p": float(p), "n": len(backbone_order),
        "mean_mae": {"synthetic": synth_mean.to_dict(), "soundscape": soundscape_mean.to_dict()},
        "rank": ranks,
        "n_datasets": {"synthetic": len(ALL_DATASETS), "soundscape": len(REGIONAL_DATASETS)},
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _fmt_p(x):
    return f"p={x:.3f}" if x is not None else "--"


def _pair_row(s: dict) -> str:
    a_label, b_label = BACKBONE_META[s["a"]]["display"], BACKBONE_META[s["b"]]["display"]
    return (f"| {a_label} - {b_label} | {fmt_mean_std(s['mean'], s['std'])} | {s['n']} "
            f"| {_fmt_p(s['t_p'])} | {_fmt_p(s['wilcoxon_p'])} |")


def build_markdown(results: dict, backbone_order: list) -> str:
    r1, r2, r3, r4 = results["check1"], results["check2"], results["check3"], results["check4"]
    tight_labels = ", ".join(BACKBONE_META[m]["display"] for m in TIGHT_GROUP)
    next_labels = ", ".join(BACKBONE_META[m]["display"] for m in r3["next_group"])

    lines = [
        "# RQ1 -- Difficulty-robust backbone ranking (frozen backbone comparison)",
        "",
        "Establishes a backbone ranking on soundscape data that is robust to cross-dataset difficulty "
        f"variance, and tests whether the \"tight leading group\" ({tight_labels}) is statistically "
        "distinguishable from the rest of the ranking or from each other. Source data: "
        "`archive/Pooled-Embeddings/` (13 backbones, frozen backbone + pooled-MLP head, regression "
        "formulation), `soundscape_test` / range_mae over the 7 REGIONAL_DATASETS, and "
        "`synthetic_mixture_test` / mae over the 8 ALL_DATASETS for check 4.",
        "",
        "**Caveat on sample size**: n=7 soundscape datasets is small for the pairwise comparisons in "
        "checks 2 and 3 -- treat those significance tests as exploratory/directional, consistent with the "
        "RQ2/RQ5 small-n caveat (`rq2_paired_cell_analysis.md`, `rq5_kurtosis.md`). The ranking itself "
        "(check 1) and the rank correlation (check 4) are more robust, since they don't depend on a "
        "single pairwise significance threshold to be informative.",
        "",
    ]

    # --- Check 1 ---
    lines += [
        "## 1. Difficulty-robust ranking, all 13 backbones (soundscape, dataset-paired)",
        "",
        f"Mean MAE across the {r1['n_datasets']} soundscape datasets, paired by dataset (every backbone "
        "is evaluated on the same 7 datasets, so this ranking is already robust to shared difficulty "
        "variance). "
        f"**Matches the top 4 of Table~\\ref{{tab:rq1_top5_soundscape_test_reg}}: "
        f"{'yes' if r1['matches_top5_soundscape_table'] else 'no'}.**",
        "",
        "| Rank | Backbone | Mean MAE (+/- SD) |",
        "|---|---|---|",
    ]
    for m in r1["order"]:
        marker = " (tight group)" if m in TIGHT_GROUP else ""
        lines.append(f"| {r1['rank'][m]} | {BACKBONE_META[m]['display']}{marker} "
                      f"| {fmt_mean_std(r1['mean_mae'][m], r1['sd_mae'][m])} |")
    lines.append("")

    # --- Check 2 ---
    lines += [
        "## 2. Pairwise significance within the tight leading group",
        "",
        f"Delta = (backbone A MAE) - (backbone B MAE), matched by dataset (n={r1['n_datasets']} "
        f"soundscape datasets), for each of the 6 pairs among {{{tight_labels}}}.",
        "",
        "| Pair (A - B) | Mean Delta MAE (+/- SD) | n | Paired t-test | Wilcoxon |",
        "|---|---|---|---|---|",
    ]
    for s in r2.values():
        lines.append(_pair_row(s))
    lines.append("")

    # --- Check 3 ---
    lines += [
        "## 3. Pairwise significance vs. the next-ranked backbones",
        "",
        f"Same paired-difference test as check 2, between each tight-group member and each of the "
        f"next-ranked backbones (ranks 5-{4 + N_NEXT}): {next_labels}. Tests whether there is a real, "
        "statistically distinguishable gap separating the \"leading group\" from the rest, or whether "
        "the group boundary is arbitrary.",
        "",
        "| Pair (A - B) | Mean Delta MAE (+/- SD) | n | Paired t-test | Wilcoxon |",
        "|---|---|---|---|---|",
    ]
    for s in r3["pairs"].values():
        lines.append(_pair_row(s))
    lines.append("")

    # --- Check 4 ---
    lines += [
        "## 4. Spearman rank correlation, synthetic vs. soundscape ranking",
        "",
        f"Spearman's rho between each of the {r4['n']} backbones' synthetic-data rank "
        f"({r4['n_datasets']['synthetic']} datasets) and soundscape-data rank "
        f"({r4['n_datasets']['soundscape']} datasets), both from paired dataset-averaged mean MAE "
        "(lower MAE = better rank).",
        "",
        f"**Spearman rho = {r4['spearman_rho']:.3f}, p = {r4['spearman_p']:.4f}, n = {r4['n']}**",
        "",
        "| Backbone | Synthetic rank | Synthetic mean MAE | Soundscape rank | Soundscape mean MAE |",
        "|---|---|---|---|---|",
    ]
    for m in backbone_order:
        lines.append(
            f"| {BACKBONE_META[m]['display']} | {r4['rank']['synthetic'][m]} "
            f"| {r4['mean_mae']['synthetic'][m]:.3f} | {r4['rank']['soundscape'][m]} "
            f"| {r4['mean_mae']['soundscape'][m]:.3f} |"
        )
    lines.append("")

    lines += [
        "## Reading this",
        "",
        "- Check 1 confirms the reported soundscape ranking and extends it to all 13 backbones; as a "
        "dataset-paired mean over all 7 datasets shared by every backbone, it is not sensitive to a "
        "single dataset's difficulty the way a per-dataset comparison would be.",
        "- Check 2 tests the \"tight leading group\" claim directly: non-significant pairwise deltas among "
        "these 4 backbones would support treating them as statistically indistinguishable at this sample "
        "size, not that they are truly equal -- n=7 has limited power to reject small true differences.",
        "- Check 3 tests whether the group boundary is real: significant deltas against the next-ranked "
        "backbones (and non-significant deltas within the group, per check 2) would support a genuine "
        "gap; if deltas vs. the next group are similarly non-significant, the \"leading group\" cutoff is "
        "not statistically motivated at n=7.",
        "- Check 4 is well-powered (n=13 backbones) and answers a different question from checks 1-3 -- "
        "it is a rank-based, not mean-difference, test, so it is the most robust single number here for "
        "how much the *overall* ranking shifts between domains.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq1"))
    args = parser.parse_args()

    models = mrt.discover_models(Path("archive/Pooled-Embeddings"))
    long_df = load_study("pooled", models=models)
    backbone_order = [m for m in BACKBONE_META if m in models]

    check1 = check1_ranking(long_df, backbone_order)
    check2 = check2_within_group(long_df)
    check3 = check3_vs_next(long_df, check1["order"])
    check4 = check4_rank_correlation(long_df, backbone_order, pd.Series(check1["mean_mae"]))

    results = {"check1": check1, "check2": check2, "check3": check3, "check4": check4}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq1_ranking_robustness.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {out_json}")

    md = build_markdown(results, backbone_order)
    out_md = args.out_dir / "rq1_ranking_robustness.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
