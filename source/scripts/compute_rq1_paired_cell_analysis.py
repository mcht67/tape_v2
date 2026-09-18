#!/usr/bin/env python3
"""
RQ1 (additional) -- Paired cell-level analysis of backbone rankings, on the
Pooled-Embeddings experiments (13 backbones, frozen backbone + pooled-
embedding head) -- same source data as `rq1_top5_table.tex` /
`rq1_top5_soundscape_table.tex` (Table~\\ref{tab:rq1_top5_reg} and
Table~\\ref{tab:rq1_top5_soundscape_test_reg}), but using all 13 backbones
rather than just the top 5.

Checks whether three RQ1 narrative claims are statistically robust, or --
as with the paired-cell/kurtosis work already done for RQ2 and RQ5 -- within
the range of noise given small sample size and high per-cell variance:
regression outperforming classification, the specific backbone ranking on
synthetic data (Bird-MAE leading, followed by Perch v2 / AudioProtoPNet /
EfficientNet-B1 / NatureLM-audio), and the "tight leading group" claim on
soundscape data (Bird-MAE, Perch v2, and both BirdNET backbones being close
together and hard to separate).

Four checks, computed on the reg/class MAE-like metric per (backbone,
dataset) cell (source `synthetic_mixture_test`, metric "mae", ALL_DATASETS
= 8 datasets; source `soundscape_test`, metric "range_mae", REGIONAL_DATASETS
= 7 datasets -- soundscape has no XCM counterpart, same convention as
`compute_rq1_sign_test.py`):

1. Regression vs. classification, paired by cell: Delta = (classification
   MAE) - (regression MAE), matched by (backbone, dataset) cell (n = 13
   backbones x 8/7 datasets = 104/91), separately for synthetic and
   soundscape. Reported as mean +/- SD, paired t-test, and Wilcoxon
   signed-rank test against zero -- a paired-difference cross-check of the
   win-count/sign-test framing already used in `compute_rq1_sign_test.py`
   and reported in the Results text.

2. "Tight leading group" claim -- pairwise significance among the top
   soundscape performers {Bird-MAE, Perch v2, BirdNET v2.3, BirdNET v2.4}
   (the 4 backbones in `rq1_top5_soundscape_table.tex`'s top 5 that also
   appear in the synthetic top 5 or lead the soundscape table): for each of
   the 6 pairs, the paired cell-level MAE difference across the 7 soundscape
   datasets (regression head), with paired t-test and Wilcoxon test against
   zero. Directly tests whether the apparent closeness in point estimates
   holds up once tested, or is within noise at n=7.

3. Backbone ranking stability, synthetic vs. soundscape: Spearman rank
   correlation between all 13 backbones' rank by mean MAE across the 8
   synthetic datasets and their rank by mean MAE across the 7 soundscape
   datasets (regression head) -- how much the *overall* ranking (not just
   the top group) shifts between domains.

4. Per-backbone domain-shift Delta, for the 7 backbones spanning both top-5
   tables (Bird-MAE, Perch v2, BirdNET v2.3, BirdNET v2.4, AudioProtoPNet,
   EfficientNet-B1, NatureLM-audio): Delta = (soundscape MAE) - (synthetic
   MAE), paired by (backbone, dataset) cell over the 7 REGIONAL_DATASETS
   common to both domains (synthetic's 8th dataset, XCM, has no soundscape
   counterpart, so it is excluded from this pairing -- same reasoning as
   `SOURCE_TABLE_DATASETS`'s soundscape restriction in `plot_rq1.py`).
   Reported overall (49 cells) and per backbone (n=7 each), same structure
   as `compute_rq5_kurtosis.py`'s per-backbone Delta_finetuned table.

Sample-size caveat, same as `compute_rq2_paired_cell_analysis.py` /
`compute_rq5_kurtosis.py`: checks 1 and 3 pool across all 13 backbones and
are reasonably powered; checks 2 and 4 operate at n=7 per pairwise/per-
backbone comparison and are exploratory, not confirmatory -- a single
dataset can swing the result.

Writes rq1_paired_cell_analysis.json and rq1_paired_cell_analysis.md to
--out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq1_paired_cell_analysis.py [--out-dir plots/figures/rq1]
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
DOMAIN_SOURCE = {
    "synthetic": (SOURCE, SYNTHETIC_METRIC, ALL_DATASETS),
    "soundscape": (SCAPE_SOURCE, SOUNDSCAPE_METRIC, REGIONAL_DATASETS),
}
DOMAIN_LABEL = {"synthetic": "Synthetic", "soundscape": "Soundscape"}

# The 4 backbones flagged as a "tight leading group" on soundscape data --
# the top 4 of `rq1_top5_soundscape_table.tex` (Perch v2, BirdNET v2.3,
# BirdNET v2.4, Bird-MAE).
TIGHT_GROUP = ["Bird-MAE-Huge", "perch_v2_cpu", "Birdnet_V2.3", "Birdnet_V2.4"]

# TIGHT_GROUP plus the synthetic top-5's other 3 backbones (AudioProtoPNet,
# EfficientNet-B1, NatureLM-audio) -- the union of both top-5 tables' leading
# backbones, for the per-backbone domain-shift check.
DOMAIN_SHIFT_BACKBONES = TIGHT_GROUP + [
    "AudioProtoPNet-20-BirdSet-XCL", "EfficientNet-B1-BirdSet-XCL", "NatureLMBEATs",
]


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


def _fmt_test(tests: dict) -> str:
    if tests["t_p"] is None:
        return "--"
    return f"t-test p={tests['t_p']:.3f}; Wilcoxon p={tests['wilcoxon_p']:.3f}" if tests["wilcoxon_p"] is not None \
        else f"t-test p={tests['t_p']:.3f}; Wilcoxon n/a"


def build_matrix(long_df: pd.DataFrame, source: str, metric: str, head: str, dataset_order: list,
                  row_order: list) -> pd.DataFrame:
    """(backbone x dataset) matrix of raw metric values, same construction
    as plot_rq5.py's build_heatmap_matrix -- one row per backbone, one
    column per dataset, reindexed so cells line up 1:1 across matrices."""
    sub = filter_long(long_df, source=source, metric=metric, head=head, dataset=dataset_order)
    matrix = sub.pivot_table(index="model", columns="dataset", values="value")
    return matrix.reindex(index=row_order, columns=dataset_order)


# ---------------------------------------------------------------------------
# Check 1 -- regression vs. classification, paired by (backbone, dataset)
# cell, separately per domain.
# ---------------------------------------------------------------------------

def check1_reg_vs_class(long_df: pd.DataFrame, backbone_order: list) -> dict:
    results = {}
    for domain, (source, metric, datasets) in DOMAIN_SOURCE.items():
        reg = build_matrix(long_df, source, metric, "reg", datasets, backbone_order)
        cls = build_matrix(long_df, source, metric, "class", datasets, backbone_order)
        diff = (cls - reg).values.flatten().astype(float)
        diff = diff[~np.isnan(diff)]
        results[domain] = {**diff_stats(diff), **paired_tests(diff)}
    return results


# ---------------------------------------------------------------------------
# Check 2 -- pairwise significance among the soundscape "tight leading
# group", regression head, paired by dataset (n=7).
# ---------------------------------------------------------------------------

def check2_tight_group(long_df: pd.DataFrame) -> dict:
    source, metric, datasets = DOMAIN_SOURCE["soundscape"]
    matrix = build_matrix(long_df, source, metric, "reg", datasets, TIGHT_GROUP)
    results = {}
    for a, b in itertools.combinations(TIGHT_GROUP, 2):
        diff = (matrix.loc[a] - matrix.loc[b]).to_numpy(dtype=float)
        diff = diff[~np.isnan(diff)]
        results[f"{a}__minus__{b}"] = {"a": a, "b": b, **diff_stats(diff), **paired_tests(diff)}
    return results


# ---------------------------------------------------------------------------
# Check 3 -- Spearman rank correlation of backbone ranking, synthetic vs.
# soundscape (all 13 backbones, regression head).
# ---------------------------------------------------------------------------

def check3_rank_stability(long_df: pd.DataFrame, backbone_order: list) -> dict:
    means = {}
    for domain, (source, metric, datasets) in DOMAIN_SOURCE.items():
        sub = filter_long(long_df, source=source, metric=metric, head="reg", dataset=datasets,
                           model=backbone_order)
        means[domain] = sub.groupby("model")["value"].mean().reindex(backbone_order)

    rho, p = scipy.stats.spearmanr(means["synthetic"].to_numpy(), means["soundscape"].to_numpy())
    ranks = {
        "synthetic": means["synthetic"].rank(method="min").astype(int).to_dict(),
        "soundscape": means["soundscape"].rank(method="min").astype(int).to_dict(),
    }
    return {
        "spearman_rho": float(rho), "spearman_p": float(p), "n": len(backbone_order),
        "mean_mae": {d: means[d].to_dict() for d in means},
        "rank": ranks,
    }


# ---------------------------------------------------------------------------
# Check 4 -- per-backbone domain-shift Delta (soundscape - synthetic),
# paired by dataset over the 7 REGIONAL_DATASETS common to both domains.
# ---------------------------------------------------------------------------

def check4_domain_shift(long_df: pd.DataFrame) -> dict:
    synthetic = build_matrix(long_df, SOURCE, SYNTHETIC_METRIC, "reg", REGIONAL_DATASETS, DOMAIN_SHIFT_BACKBONES)
    soundscape = build_matrix(long_df, SCAPE_SOURCE, SOUNDSCAPE_METRIC, "reg", REGIONAL_DATASETS,
                               DOMAIN_SHIFT_BACKBONES)
    diff_matrix = soundscape - synthetic

    overall = diff_matrix.values.flatten().astype(float)
    overall = overall[~np.isnan(overall)]
    per_backbone = {}
    for m in DOMAIN_SHIFT_BACKBONES:
        diff = diff_matrix.loc[m].to_numpy(dtype=float)
        diff = diff[~np.isnan(diff)]
        per_backbone[m] = {**diff_stats(diff), **paired_tests(diff)}
    return {"overall": {**diff_stats(overall), **paired_tests(overall)}, "per_backbone": per_backbone}


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def build_markdown(results: dict, backbone_order: list) -> str:
    n_datasets = {d: len(DOMAIN_SOURCE[d][2]) for d in DOMAIN_SOURCE}
    lines = [
        "# RQ1 -- Paired cell-level analysis of backbone rankings",
        "",
        "Checks whether RQ1's narrative claims -- regression outperforming classification, the specific "
        "backbone ranking on synthetic data (Bird-MAE leading, followed by Perch v2/AudioProtoPNet/"
        "EfficientNet-B1/NatureLM-audio), and the \"tight leading group\" claim on soundscape data (Bird-MAE, "
        "Perch v2, and both BirdNET backbones being close together and hard to separate) -- are "
        "statistically robust, or -- as with the paired-cell/kurtosis work already done for RQ2 and RQ5 -- "
        "within the range of noise given small sample size and high per-cell variance. Source data: "
        "`archive/Pooled-Embeddings/` (13 backbones, frozen backbone + pooled-embedding head), same as "
        "`rq1_top5_table.tex` / `rq1_top5_soundscape_table.tex`, but using all 13 backbones rather than "
        "just the top 5.",
        "",
        "**Caveat on sample size**: checks 1 and 3 pool across all 13 backbones and are reasonably powered. "
        "Checks 2 and 4 operate at n=7 per pairwise/per-backbone comparison (the 7 soundscape/regional "
        "datasets) -- treat those as exploratory, not confirmatory, per the same small-n caveat as "
        "`rq2_paired_cell_analysis.md` and `rq5_kurtosis.md`; a single dataset can swing the result.",
        "",
    ]

    # --- Check 1 ---
    lines += [
        "## 1. Regression vs. classification, paired by cell",
        "",
        "Delta = (classification MAE) - (regression MAE), matched by (backbone, dataset) cell "
        "(n = 13 backbones x datasets per domain). Negative = regression wins. Cross-checks the win-count/"
        "sign-test framing already used in `compute_rq1_sign_test.py` and reported in the Results text.",
        "",
        "| Domain | n | Mean Delta MAE (+/- SD) | Paired t-test | Wilcoxon |",
        "|---|---|---|---|---|",
    ]
    for domain in ("synthetic", "soundscape"):
        s = results["check1"][domain]
        t_p = f"p={s['t_p']:.3f}" if s['t_p'] is not None else "--"
        w_p = f"p={s['wilcoxon_p']:.3f}" if s['wilcoxon_p'] is not None else "--"
        lines.append(f"| {DOMAIN_LABEL[domain]} (13 x {n_datasets[domain]} = {13 * n_datasets[domain]}) "
                      f"| {s['n']} | {fmt_mean_std(s['mean'], s['std'])} | {t_p} | {w_p} |")
    lines.append("")

    # --- Check 2 ---
    lines += [
        "## 2. \"Tight leading group\" -- pairwise significance on soundscape data",
        "",
        "Paired cell-level MAE difference (regression head), matched by dataset, across the 4 backbones "
        f"flagged as a tight leading group ({', '.join(BACKBONE_META[m]['display'] for m in TIGHT_GROUP)}), "
        "n=7 soundscape datasets per pair.",
        "",
        "| Pair (A - B) | Mean Delta MAE (+/- SD) | n | Paired t-test | Wilcoxon |",
        "|---|---|---|---|---|",
    ]
    for key, s in results["check2"].items():
        pair_label = f"{BACKBONE_META[s['a']]['display']} - {BACKBONE_META[s['b']]['display']}"
        t_p = f"p={s['t_p']:.3f}" if s['t_p'] is not None else "--"
        w_p = f"p={s['wilcoxon_p']:.3f}" if s['wilcoxon_p'] is not None else "--"
        lines.append(f"| {pair_label} | {fmt_mean_std(s['mean'], s['std'])} | {s['n']} | {t_p} | {w_p} |")
    lines.append("")

    # --- Check 3 ---
    r3 = results["check3"]
    lines += [
        "## 3. Backbone ranking stability, synthetic vs. soundscape",
        "",
        f"Spearman rank correlation between all {r3['n']} backbones' rank by mean MAE across the "
        f"{n_datasets['synthetic']} synthetic datasets and their rank by mean MAE across the "
        f"{n_datasets['soundscape']} soundscape datasets (regression head, lower MAE = better rank).",
        "",
        f"**Spearman rho = {r3['spearman_rho']:.3f}, p = {r3['spearman_p']:.4f}, n = {r3['n']}**",
        "",
        "| Backbone | Synthetic rank | Synthetic mean MAE | Soundscape rank | Soundscape mean MAE |",
        "|---|---|---|---|---|",
    ]
    for m in backbone_order:
        lines.append(
            f"| {BACKBONE_META[m]['display']} | {r3['rank']['synthetic'][m]} "
            f"| {r3['mean_mae']['synthetic'][m]:.3f} | {r3['rank']['soundscape'][m]} "
            f"| {r3['mean_mae']['soundscape'][m]:.3f} |"
        )
    lines.append("")

    # --- Check 4 ---
    r4 = results["check4"]
    lines += [
        "## 4. Per-backbone domain-shift Delta (soundscape - synthetic)",
        "",
        "Delta = (soundscape MAE) - (synthetic MAE), paired by (backbone, dataset) cell over the 7 "
        "REGIONAL_DATASETS common to both domains (synthetic's 8th dataset, XCM, has no soundscape "
        "counterpart and is excluded from this pairing). Positive = worse (higher MAE) on soundscape. "
        "n=7 per backbone -- exploratory, same caveat as check 2.",
        "",
        f"**Overall ({len(DOMAIN_SHIFT_BACKBONES)} backbones x 7 datasets = "
        f"{len(DOMAIN_SHIFT_BACKBONES) * 7} cells): "
        f"{fmt_mean_std(r4['overall']['mean'], r4['overall']['std'])} (n={r4['overall']['n']}), "
        f"{_fmt_test(r4['overall'])}**",
        "",
        "| Backbone | Mean Delta MAE (+/- SD) | n | Paired t-test | Wilcoxon |",
        "|---|---|---|---|---|",
    ]
    for m in DOMAIN_SHIFT_BACKBONES:
        s = r4["per_backbone"][m]
        t_p = f"p={s['t_p']:.3f}" if s['t_p'] is not None else "--"
        w_p = f"p={s['wilcoxon_p']:.3f}" if s['wilcoxon_p'] is not None else "--"
        lines.append(f"| {BACKBONE_META[m]['display']} | {fmt_mean_std(s['mean'], s['std'])} | {s['n']} "
                      f"| {t_p} | {w_p} |")
    lines.append("")

    lines += [
        "## Reading this",
        "",
        "- Check 1 is the broadest and best-powered test here (91-104 cells per domain) -- a significant "
        "result there is the most trustworthy signal in this file, and the most direct cross-check of the "
        "sign-test result already reported in the Results text.",
        "- Check 3 is also well-powered (n=13 backbones) but answers a different question from checks 1/2/4 "
        "-- it is a rank-based, not mean-difference, test, so a low rho despite close point estimates "
        "(or vice versa) is a real, independently checkable fact about how much the ranking itself shifts.",
        "- Checks 2 and 4 operate at n=7 per comparison; treat their p-values as directional only -- a "
        "non-significant pairwise test among the \"tight group\" backbones does not by itself confirm they "
        "are indistinguishable, only that n=7 cannot reject equality here.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq1"))
    args = parser.parse_args()

    models = mrt.discover_models(Path("archive/Pooled-Embeddings"))
    long_df = load_study("pooled", models=models)
    backbone_order = [m for m in BACKBONE_META if m in models]

    results = {
        "check1": check1_reg_vs_class(long_df, backbone_order),
        "check2": check2_tight_group(long_df),
        "check3": check3_rank_stability(long_df, backbone_order),
        "check4": check4_domain_shift(long_df),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq1_paired_cell_analysis.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {out_json}")

    md = build_markdown(results, backbone_order)
    out_md = args.out_dir / "rq1_paired_cell_analysis.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
