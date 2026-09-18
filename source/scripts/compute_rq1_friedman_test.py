#!/usr/bin/env python3
"""
RQ1 -- Friedman test, MLP-pooled regression backbones across soundscape
datasets.

Tests whether the 13 frozen-backbone, pooled-MLP-head, regression-formulation
models (`archive/Pooled-Embeddings/`) differ significantly in rank across the
7 soundscape datasets (REGIONAL_DATASETS: UHH, HSN, PER, NES, POW, SSW, SNE),
source `soundscape_test`, metric `range_mae` -- same source data and matrix
construction as `compute_rq1_ranking_robustness.py`'s check 1.

The Friedman test (scipy.stats.friedmanchisquare) is the standard
nonparametric alternative to a repeated-measures ANOVA for this design: k
related samples (here, 13 backbones) measured on the same n blocks (7
datasets), testing whether the backbones' rank distributions differ. Follows
Demsar (2006)'s convention for comparing multiple classifiers over multiple
datasets: backbones are ranked 1 (best/lowest MAE) to 13 (worst) per dataset
(ties averaged), average ranks are reported alongside the omnibus test, and
-- since the omnibus test only says "not all backbones are equal", not which
ones -- a post-hoc Nemenyi test with critical difference (CD) is run to flag
which pairwise average-rank gaps are large enough to be significant at
alpha=0.05.

Writes rq1_friedman_test.json and rq1_friedman_test.md to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq1_friedman_test.py [--out-dir plots/figures/rq1]
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
from plot_data import REGIONAL_DATASETS, filter_long, load_study

SCAPE_SOURCE = "soundscape_test"
SOUNDSCAPE_METRIC = "range_mae"
ALPHA = 0.05


def build_matrix(long_df: pd.DataFrame, backbone_order: list) -> pd.DataFrame:
    """(backbone x dataset) matrix of raw range_mae values, regression head,
    7 REGIONAL_DATASETS -- same construction as
    compute_rq1_ranking_robustness.py's check1_ranking()."""
    sub = filter_long(long_df, source=SCAPE_SOURCE, metric=SOUNDSCAPE_METRIC, head="reg", dataset=REGIONAL_DATASETS)
    matrix = sub.pivot_table(index="model", columns="dataset", values="value")
    return matrix.reindex(index=backbone_order, columns=REGIONAL_DATASETS)


def friedman_test(matrix: pd.DataFrame) -> dict:
    """scipy.stats.friedmanchisquare across backbones -- each backbone's
    7-dataset range_mae vector is one related sample, the 7 datasets are the
    matched blocks. Requires complete data (no NaNs); any backbone missing a
    dataset is dropped first (none are, for this matrix, but the check is
    kept for robustness)."""
    complete = matrix.dropna(axis=0, how="any")
    dropped = [m for m in matrix.index if m not in complete.index]
    samples = [complete.loc[m].to_numpy() for m in complete.index]
    stat, p = scipy.stats.friedmanchisquare(*samples)
    return {
        "statistic": float(stat), "p_value": float(p),
        "df": len(complete.index) - 1,
        "k_backbones": len(complete.index), "n_datasets": complete.shape[1],
        "backbones_used": list(complete.index), "backbones_dropped_incomplete": dropped,
    }


def average_ranks(matrix: pd.DataFrame) -> pd.Series:
    """Per-dataset ranks (1=best/lowest range_mae, ties averaged), then mean
    rank per backbone across datasets -- Demsar (2006) convention."""
    ranks = matrix.rank(axis=0, method="average", ascending=True)
    return ranks.mean(axis=1).sort_values()


def nemenyi_cd(k: int, n: int, alpha: float = ALPHA) -> float:
    """Critical difference for the post-hoc Nemenyi test on average ranks
    (Demsar 2006): CD = q_alpha * sqrt(k(k+1)/(6n)), with q_alpha the
    studentized-range critical value for k means (infinite df) divided by
    sqrt(2) -- reproduces Demsar's Table 5 exactly (e.g. k=10 -> 3.164)."""
    q_alpha = scipy.stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q_alpha * np.sqrt(k * (k + 1) / (6 * n)))


def nemenyi_pairs(ranks: pd.Series, cd: float) -> dict:
    """All pairwise average-rank gaps vs. the critical difference -- a pair
    is flagged significant (at alpha=0.05) iff |rank_a - rank_b| > cd."""
    pairs = {}
    for a, b in itertools.combinations(ranks.index, 2):
        diff = abs(float(ranks[a] - ranks[b]))
        pairs[f"{a}__vs__{b}"] = {"a": a, "b": b, "rank_diff": diff, "significant": bool(diff > cd)}
    return pairs


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def build_markdown(ft: dict, ranks: pd.Series, cd: float, sig_pairs: dict) -> str:
    n_sig = sum(1 for p in sig_pairs.values() if p["significant"])
    n_total = len(sig_pairs)

    lines = [
        "# RQ1 -- Friedman test, MLP-pooled regression backbones across soundscape datasets",
        "",
        "Tests whether the 13 frozen-backbone, pooled-MLP-head, regression-formulation models "
        "(`archive/Pooled-Embeddings/`) differ significantly in ranking across the 7 soundscape "
        f"datasets ({', '.join(REGIONAL_DATASETS)}), source `soundscape_test`, metric `range_mae`. Same "
        "source data and (backbone x dataset) matrix as `compute_rq1_ranking_robustness.py`'s check 1.",
        "",
        "The Friedman test is the nonparametric alternative to a repeated-measures ANOVA for this design: "
        f"{ft['k_backbones']} related samples (backbones), each measured on the same {ft['n_datasets']} "
        "blocks (datasets). Backbones are ranked 1 (best/lowest range_mae) to "
        f"{ft['k_backbones']} (worst) per dataset (ties averaged), following Demsar (2006)'s convention "
        "for comparing multiple models over multiple datasets.",
        "",
        "## Omnibus test",
        "",
        f"**Friedman chi-squared = {ft['statistic']:.3f}, df = {ft['df']}, p = {ft['p_value']:.6f}, "
        f"k = {ft['k_backbones']} backbones, n = {ft['n_datasets']} datasets.**",
        "",
    ]
    if ft["backbones_dropped_incomplete"]:
        dropped_labels = ", ".join(BACKBONE_META.get(m, {}).get("display", m) for m in ft["backbones_dropped_incomplete"])
        lines += [f"Dropped for incomplete data (missing at least one of the 7 datasets): {dropped_labels}.", ""]

    verdict = "rejects" if ft["p_value"] < ALPHA else "does not reject"
    lines += [
        f"At alpha = {ALPHA}, the test **{verdict}** the null hypothesis that all backbones have the same "
        "rank distribution across the 7 soundscape datasets -- i.e. the backbones' soundscape performance "
        "is" + (" not" if ft["p_value"] < ALPHA else "") + " interchangeable at this sample size.",
        "",
        "## Average ranks",
        "",
        "Mean rank per backbone across the 7 datasets (1 = best/lowest range_mae on that dataset, ties "
        "averaged); lower is better.",
        "",
        "| Rank | Backbone | Average rank |",
        "|---|---|---|",
    ]
    for i, (m, r) in enumerate(ranks.items(), start=1):
        lines.append(f"| {i} | {BACKBONE_META.get(m, {}).get('display', m)} | {r:.3f} |")
    lines.append("")

    lines += [
        "## Post-hoc Nemenyi test",
        "",
        f"Since the omnibus test only says \"not all backbones are equal\", not which ones, a post-hoc "
        f"Nemenyi test on the average ranks above flags pairs whose average-rank gap exceeds the critical "
        f"difference (CD) at alpha = {ALPHA}.",
        "",
        f"**CD = {cd:.3f}** (k = {ft['k_backbones']}, n = {ft['n_datasets']}). "
        f"{n_sig} of {n_total} pairwise comparisons exceed CD.",
        "",
    ]
    if n_sig:
        lines += ["| Pair | Rank diff | Significant (> CD) |", "|---|---|---|"]
        for p in sorted(sig_pairs.values(), key=lambda x: -x["rank_diff"]):
            if not p["significant"]:
                continue
            a_label = BACKBONE_META.get(p["a"], {}).get("display", p["a"])
            b_label = BACKBONE_META.get(p["b"], {}).get("display", p["b"])
            lines.append(f"| {a_label} vs. {b_label} | {p['rank_diff']:.3f} | yes |")
        lines.append("")
    else:
        lines += [
            "No pairwise comparison exceeds the critical difference -- with k=13 backbones and only n=7 "
            "datasets, the Nemenyi test has limited power, so a significant omnibus result does not "
            "guarantee any individually significant pair.",
            "",
        ]

    lines += [
        "## Reading this",
        "",
        "- The omnibus Friedman test is well-suited to this design (k=13 backbones, n=7 matched soundscape "
        "datasets) and does not assume normality, unlike a repeated-measures ANOVA on the raw range_mae "
        "values.",
        "- The post-hoc Nemenyi CD is conservative (it corrects for all pairwise comparisons at once); "
        "n=7 datasets gives it limited power, so a non-significant pair here does not mean the backbones "
        "perform equally -- only that the gap is not distinguishable from chance at this sample size, "
        "consistent with the small-n caveat already noted for the pairwise tests in "
        "`rq1_ranking_robustness.md`.",
        "- Average ranks (not raw range_mae) are the right summary to read alongside this test -- they are "
        "what the Friedman/Nemenyi tests actually operate on, and are more robust to a single dataset's "
        "MAE being on a very different scale than the others.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq1"))
    args = parser.parse_args()

    models = mrt.discover_models(Path("archive/Pooled-Embeddings"))
    long_df = load_study("pooled", models=models)
    backbone_order = [m for m in BACKBONE_META if m in models]

    matrix = build_matrix(long_df, backbone_order)
    ft = friedman_test(matrix)
    ranks = average_ranks(matrix.loc[ft["backbones_used"]])
    cd = nemenyi_cd(ft["k_backbones"], ft["n_datasets"])
    sig_pairs = nemenyi_pairs(ranks, cd)

    results = {
        "friedman": ft,
        "average_ranks": ranks.to_dict(),
        "nemenyi_cd": cd,
        "nemenyi_pairs": sig_pairs,
        "alpha": ALPHA,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq1_friedman_test.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {out_json}")

    md = build_markdown(ft, ranks, cd, sig_pairs)
    out_md = args.out_dir / "rq1_friedman_test.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
