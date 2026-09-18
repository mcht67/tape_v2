#!/usr/bin/env python3
"""
RQ1 (additional) -- Leading-group vs. lower-ranked backbones (ranks 8-13),
extending `compute_rq1_ranking_robustness.py` check 3 / `compute_rq1_sign_test_pairs.py`
check 2 (which only cover ranks 5-7: Wav2Vec2, VGGish, AST) down through the
rest of the 13-backbone soundscape ranking (NatureLM-audio, EfficientNet-B1,
Perch v1, AudioProtoPNet, YAMNet, BEANS baseline).

Purpose: determine whether there is any statistically distinguishable
boundary within the full 13-backbone soundscape ranking, or whether the
entire ranking is statistically indistinguishable at n=7 datasets. Same
source data and methodology as the two files above: `archive/Pooled-
Embeddings/`, `soundscape_test` / range_mae, regression head, the 7
REGIONAL_DATASETS.

Two checks, mirroring the two existing files exactly, for the tight leading
group {Bird-MAE, Perch v2, BirdNET v2.3, BirdNET v2.4} x ranks 8-13 (4 x 6 =
24 pairs):

1. Paired cell-level mean-difference test (mirrors `rq1_ranking_robustness.md`
   check 3 / `rq1_paired_cell_analysis.md` check 2): Delta = (leading-group
   backbone MAE) - (lower-ranked backbone MAE), matched by dataset (n=7),
   paired t-test and Wilcoxon signed-rank test against zero.

2. Exact sign test (mirrors `rq1_sign_test_pairs.md` check 2): win/loss/tie
   count per pair across the 7 soundscape datasets, exact binomial sign test.

Then, to answer the actual research question, this script also recomputes
(not hardcodes) the existing 12 ranks-5-7 pairs via the same functions
imported from the two files above, and combines them with the 24 new pairs
into:

3. A multiple-comparisons summary over all 36 leading-group-vs-lower-ranked
   comparisons (12 already reported + 24 new): how many clear the
   uncorrected p<0.05 threshold, and how many survive a Bonferroni
   correction (alpha = 0.05/36 ~ 0.00139), for the paired t-test, Wilcoxon,
   and sign test each.

4. A combined summary table (ranks 5-13) rolling the 36 pairs up by the
   lower-ranked backbone's rank, identifying the lowest rank (if any) at
   which a leading-group member is statistically distinguishable from it
   (by any of the three tests, both at the uncorrected and Bonferroni
   thresholds) -- including an explicit flag for the BEANS baseline pair
   (rank 13, by far the largest point-estimate gap of any comparison in
   either batch).

Sample-size caveat, same as the two files this extends: n=7 soundscape
datasets is the limiting factor throughout -- every pairwise test here is
exploratory/directional, not confirmatory.

Writes rq1_lower_ranks_robustness.json and rq1_lower_ranks_robustness.md to
--out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq1_lower_ranks_robustness.py [--out-dir plots/figures/rq1]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import BACKBONE_META
from compute_rq1_ranking_robustness import (
    N_NEXT, REGIONAL_DATASETS, SCAPE_SOURCE, SOUNDSCAPE_METRIC, TIGHT_GROUP,
    build_matrix, check1_ranking, fmt_mean_std, pairwise_diff,
)
from compute_rq1_sign_test_pairs import N_DATASETS, pairwise_sign_test
from plot_data import load_study

N_LOWER = 6  # ranks 8-13, the remaining backbones after the tight group (1-4) and ranks 5-7.
N_TOTAL_COMPARISONS = 36  # 4 tight-group members x (3 ranks5-7 + 6 ranks8-13)
BONFERRONI_ALPHA = 0.05 / N_TOTAL_COMPARISONS


def _fmt_p(x):
    return f"p={x:.3f}" if x is not None else "--"


def _fmt_p_raw(x):
    return f"{x:.3f}" if x is not None else "--"


def _pair_row_diff(s: dict) -> str:
    a_label, b_label = BACKBONE_META[s["a"]]["display"], BACKBONE_META[s["b"]]["display"]
    return (f"| {a_label} - {b_label} | {fmt_mean_std(s['mean'], s['std'])} | {s['n']} "
            f"| {_fmt_p(s['t_p'])} | {_fmt_p(s['wilcoxon_p'])} |")


def _fmt_sign_p(x):
    return f"{x:.4f}" if x is not None else "n/a, n<=3"


def _pair_row_sign(s: dict) -> str:
    a_label, b_label = BACKBONE_META[s["a"]]["display"], BACKBONE_META[s["b"]]["display"]
    return (f"| {a_label} - {b_label} | {s['n_datasets']} | {s['wins']} | {s['losses']} "
            f"| {s['ties']} | {_fmt_sign_p(s['p_two_sided'])} |")


def compute_group_pairs(matrix: pd.DataFrame, group: list) -> tuple[dict, dict]:
    """Paired-difference and sign-test results for TIGHT_GROUP x group."""
    diff_results = {f"{a}__minus__{b}": pairwise_diff(matrix, a, b)
                     for a in TIGHT_GROUP for b in group}
    sign_results = {f"{a}__minus__{b}": pairwise_sign_test(matrix, a, b)
                     for a in TIGHT_GROUP for b in group}
    return diff_results, sign_results


def multiple_comparisons_summary(diff_results: dict, sign_results: dict) -> dict:
    t_ps = [s["t_p"] for s in diff_results.values() if s["t_p"] is not None]
    w_ps = [s["wilcoxon_p"] for s in diff_results.values() if s["wilcoxon_p"] is not None]
    sign_ps = [s["p_two_sided"] for s in sign_results.values() if s["p_two_sided"] is not None]

    def _counts(ps):
        n = len(ps)
        n_uncorrected = sum(1 for p in ps if p < 0.05)
        n_bonferroni = sum(1 for p in ps if p < BONFERRONI_ALPHA)
        return {"n": n, "n_uncorrected_sig": n_uncorrected, "n_bonferroni_sig": n_bonferroni}

    return {
        "n_total_comparisons": N_TOTAL_COMPARISONS,
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "paired_t_test": _counts(t_ps),
        "wilcoxon": _counts(w_ps),
        "sign_test": _counts(sign_ps),
    }


def combined_rank_summary(all_diff: dict, all_sign: dict, rank_lookup: dict) -> list:
    """Roll the 36 pairs up by the lower-ranked backbone (B side): for each
    rank 5-13, whether any of the 4 tight-group-vs-that-backbone pairs is
    significant, uncorrected and Bonferroni."""
    by_backbone = {}
    for key, s in all_diff.items():
        b = s["b"]
        by_backbone.setdefault(b, {"diff": [], "sign": []})["diff"].append(s)
    for key, s in all_sign.items():
        b = s["b"]
        by_backbone.setdefault(b, {"diff": [], "sign": []})["sign"].append(s)

    rows = []
    for b, d in sorted(by_backbone.items(), key=lambda kv: rank_lookup[kv[0]]):
        t_ps = [s["t_p"] for s in d["diff"] if s["t_p"] is not None]
        w_ps = [s["wilcoxon_p"] for s in d["diff"] if s["wilcoxon_p"] is not None]
        sign_ps = [s["p_two_sided"] for s in d["sign"] if s["p_two_sided"] is not None]
        min_t = min(t_ps) if t_ps else None
        min_w = min(w_ps) if w_ps else None
        min_sign = min(sign_ps) if sign_ps else None
        any_ps = [p for p in (min_t, min_w, min_sign) if p is not None]
        best_p = min(any_ps) if any_ps else None
        rows.append({
            "backbone": b,
            "rank": rank_lookup[b],
            "min_t_p": min_t,
            "min_wilcoxon_p": min_w,
            "min_sign_p": min_sign,
            "best_p_any_method": best_p,
            "uncorrected_sig": bool(best_p is not None and best_p < 0.05),
            "bonferroni_sig": bool(best_p is not None and best_p < BONFERRONI_ALPHA),
        })
    return rows


def build_markdown(check1: dict, next_group: list, lower_group: list,
                    ranks5_7_diff: dict, ranks5_7_sign: dict,
                    ranks8_13_diff: dict, ranks8_13_sign: dict,
                    mc_summary: dict, rank_rows: list) -> str:
    tight_labels = ", ".join(BACKBONE_META[m]["display"] for m in TIGHT_GROUP)
    next_labels = ", ".join(BACKBONE_META[m]["display"] for m in next_group)
    lower_labels = ", ".join(BACKBONE_META[m]["display"] for m in lower_group)

    lines = [
        "# RQ1 -- Leading-group vs. lower-ranked backbones (ranks 8-13)",
        "",
        f"Extends `rq1_ranking_robustness.md` check 3 and `rq1_sign_test_pairs.md` check 2 (which cover only "
        f"ranks 5-7: {next_labels}) down through the rest of the 13-backbone soundscape ranking: "
        f"ranks 8-13 ({lower_labels}). Tests whether there is any statistically distinguishable boundary "
        "within the full ranking, or whether it is statistically indistinguishable at this sample size "
        "throughout. Source data: `archive/Pooled-Embeddings/` (13 backbones, frozen backbone + pooled-MLP "
        "head, regression formulation), `soundscape_test` / range_mae over the 7 REGIONAL_DATASETS.",
        "",
        "**Caveat on sample size**: n=7 soundscape datasets remains the limiting factor throughout -- treat "
        "every pairwise test below as exploratory/directional, consistent with the small-n caution already "
        "established for RQ1/RQ2/RQ5 (`rq1_ranking_robustness.md`, `rq2_paired_cell_analysis.md`, "
        "`rq5_kurtosis.md`).",
        "",
        "## 0. Full 13-backbone ranking (for reference)",
        "",
        "| Rank | Backbone | Mean MAE (+/- SD) |",
        "|---|---|---|",
    ]
    for m in check1["order"]:
        marker = " (tight group)" if m in TIGHT_GROUP else ""
        lines.append(f"| {check1['rank'][m]} | {BACKBONE_META[m]['display']}{marker} "
                      f"| {fmt_mean_std(check1['mean_mae'][m], check1['sd_mae'][m])} |")
    lines.append("")

    # --- Check 1: paired-difference, ranks 8-13 ---
    lines += [
        "## 1. Paired cell-level mean-difference test, leading group vs. ranks 8-13",
        "",
        f"Delta = (leading-group backbone MAE) - (lower-ranked backbone MAE), matched by dataset (n=7 "
        f"soundscape datasets), for each of the 4 x 6 = 24 pairs between {{{tight_labels}}} and ranks 8-13 "
        f"({lower_labels}).",
        "",
        "| Pair (A - B) | Mean Delta MAE (+/- SD) | n | Paired t-test | Wilcoxon |",
        "|---|---|---|---|---|",
    ]
    for s in ranks8_13_diff.values():
        lines.append(_pair_row_diff(s))
    lines.append("")

    # --- Check 2: sign test, ranks 8-13 ---
    lines += [
        "## 2. Exact sign test, leading group vs. ranks 8-13",
        "",
        f"Same 24 pairs as check 1: sign of (A's MAE - B's MAE) per dataset, wins for A negative, ties "
        "dropped, exact two-sided binomial sign test against p=0.5.",
        "",
        "| Pair (A - B) | n datasets | Wins | Losses | Ties | Sign-test $p$ |",
        "|---|---|---|---|---|---|",
    ]
    for s in ranks8_13_sign.values():
        lines.append(_pair_row_sign(s))
    lines.append("")

    # --- Check 3: multiple comparisons ---
    mc = mc_summary
    lines += [
        "## 3. Multiple-comparisons summary, all 36 leading-group-vs-lower-ranked comparisons",
        "",
        f"Combines this batch's 24 comparisons (ranks 8-13) with the 12 already reported against ranks 5-7 "
        f"in `rq1_ranking_robustness.md` check 3 / `rq1_sign_test_pairs.md` check 2 (recomputed here, not "
        f"hardcoded, for consistency) -- {N_TOTAL_COMPARISONS} total. Bonferroni alpha = "
        f"0.05 / {N_TOTAL_COMPARISONS} = {BONFERRONI_ALPHA:.5f}.",
        "",
        "| Method | n comparisons | Uncorrected p<0.05 | Survive Bonferroni (p<{:.5f}) |".format(BONFERRONI_ALPHA),
        "|---|---|---|---|",
        f"| Paired t-test | {mc['paired_t_test']['n']} | {mc['paired_t_test']['n_uncorrected_sig']} "
        f"| {mc['paired_t_test']['n_bonferroni_sig']} |",
        f"| Wilcoxon signed-rank | {mc['wilcoxon']['n']} | {mc['wilcoxon']['n_uncorrected_sig']} "
        f"| {mc['wilcoxon']['n_bonferroni_sig']} |",
        f"| Sign test | {mc['sign_test']['n']} | {mc['sign_test']['n_uncorrected_sig']} "
        f"| {mc['sign_test']['n_bonferroni_sig']} |",
        "",
    ]

    # --- Check 4: combined per-rank summary ---
    beans_row = next((r for r in rank_rows if r["backbone"] == "BeansBaseline"), None)
    any_bonferroni = any(r["bonferroni_sig"] for r in rank_rows)
    any_uncorrected = any(r["uncorrected_sig"] for r in rank_rows)

    lines += [
        "## 4. Combined summary, ranks 5-13 vs. the leading group",
        "",
        "Rolls the 36 pairs up by the lower-ranked backbone's rank: the best (smallest) p-value across its "
        "4 pairings with the leading group, for each of the three tests, and whether that clears the "
        "uncorrected or Bonferroni threshold.",
        "",
        "| Rank | Backbone | Best paired t-test p | Best Wilcoxon p | Best sign-test p | Uncorrected p<0.05 | Bonferroni-significant |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rank_rows:
        lines.append(
            f"| {r['rank']} | {BACKBONE_META[r['backbone']]['display']} | {_fmt_p(r['min_t_p'])} "
            f"| {_fmt_p(r['min_wilcoxon_p'])} | {_fmt_sign_p(r['min_sign_p'])} "
            f"| {'yes' if r['uncorrected_sig'] else 'no'} | {'yes' if r['bonferroni_sig'] else 'no'} |"
        )
    lines.append("")

    if any_bonferroni:
        first_bonferroni = next(r for r in rank_rows if r["bonferroni_sig"])
        conclusion = (f"**The lowest-ranked backbone that is Bonferroni-confirmed statistically "
                       f"distinguishable from the leading group is rank {first_bonferroni['rank']} "
                       f"({BACKBONE_META[first_bonferroni['backbone']]['display']}).**")
    elif any_uncorrected:
        first_uncorrected = next(r for r in rank_rows if r["uncorrected_sig"])
        conclusion = (f"**No comparison survives Bonferroni correction across all {N_TOTAL_COMPARISONS} "
                       f"tests. The lowest-ranked backbone with a nominally significant (uncorrected "
                       f"p<0.05) difference from the leading group is rank {first_uncorrected['rank']} "
                       f"({BACKBONE_META[first_uncorrected['backbone']]['display']}), but this does not "
                       "survive multiple-comparisons correction and should not be treated as a confirmed "
                       "boundary.**")
    else:
        conclusion = (f"**No rank in the full 1-13 soundscape ordering shows a confirmed significant "
                       f"difference from the leading group, by any method, at either the uncorrected or "
                       f"Bonferroni threshold. The entire 13-backbone ranking is statistically "
                       f"indistinguishable at n=7 soundscape datasets.**")

    beans_line = ""
    if beans_row is not None:
        beans_verdict = ("is" if beans_row["bonferroni_sig"] else
                          "is nominally but not Bonferroni-significantly" if beans_row["uncorrected_sig"] else
                          "is not")
        beans_line = (
            f"\n\n**BEANS baseline (rank 13, point-estimate MAE 1.411 vs. the leading group's ~0.72-0.75 -- "
            f"the largest gap of any comparison in either batch)**: best paired t-test p="
            f"{_fmt_p_raw(beans_row['min_t_p'])}, "
            f"best Wilcoxon p={_fmt_p_raw(beans_row['min_wilcoxon_p'])}, "
            f"best sign-test p={_fmt_sign_p(beans_row['min_sign_p'])}. This pair {beans_verdict} "
            "statistically significant. "
            + ("Even the largest point-estimate gap in this analysis clears the bar, which is the "
               "strongest single result in this file." if beans_row["bonferroni_sig"] else
               "Even the largest point-estimate gap in this analysis does not survive Bonferroni "
               "correction at n=7 -- a particularly strong illustration of how underpowered n=7 "
               "soundscape datasets is for ranking claims of this kind." if not beans_row["uncorrected_sig"]
               else "It clears the uncorrected threshold but not Bonferroni correction, underscoring how "
               "underpowered n=7 is even for the single largest gap in the ranking.")
        )

    lines.append(conclusion + beans_line)
    lines.append("")

    lines += [
        "## Reading this",
        "",
        "- Checks 1 and 2 are the direct extension of `rq1_ranking_robustness.md` check 3 / "
        "`rq1_sign_test_pairs.md` check 2 to ranks 8-13 -- same methodology, same source data, same n=7 "
        "caveat.",
        "- Check 3's Bonferroni threshold (alpha ~ 0.0014) is conservative by design, since it corrects "
        "across all 36 leading-group-vs-lower-ranked comparisons run across both batches; a comparison "
        "that clears the uncorrected p<0.05 threshold but not this one should be read as suggestive, not "
        "confirmatory.",
        "- Check 4 answers the batch's actual question directly: whether *any* rank in the 13-backbone "
        "soundscape ordering is a confirmed statistical boundary against the leading group. A null result "
        "even for the BEANS baseline -- by far the largest point-estimate gap available -- would be the "
        "strongest evidence in the RQ1 analyses that n=7 soundscape datasets cannot support fine-grained "
        "ranking claims, not just claims about the top of the ranking.",
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
    order = check1["order"]
    rank_lookup = check1["rank"]

    non_tight = [m for m in order if m not in TIGHT_GROUP]
    next_group = non_tight[:N_NEXT]                       # ranks 5-7
    lower_group = non_tight[N_NEXT:N_NEXT + N_LOWER]       # ranks 8-13

    full_matrix = build_matrix(long_df, SCAPE_SOURCE, SOUNDSCAPE_METRIC, REGIONAL_DATASETS,
                                TIGHT_GROUP + next_group + lower_group)

    ranks5_7_diff, ranks5_7_sign = compute_group_pairs(full_matrix, next_group)
    ranks8_13_diff, ranks8_13_sign = compute_group_pairs(full_matrix, lower_group)

    all_diff = {**ranks5_7_diff, **ranks8_13_diff}
    all_sign = {**ranks5_7_sign, **ranks8_13_sign}
    assert len(all_diff) == N_TOTAL_COMPARISONS == len(all_sign)

    mc_summary = multiple_comparisons_summary(all_diff, all_sign)
    rank_rows = combined_rank_summary(all_diff, all_sign, rank_lookup)

    results = {
        "check1_ranking": check1,
        "next_group_ranks5_7": next_group,
        "lower_group_ranks8_13": lower_group,
        "ranks8_13_diff": ranks8_13_diff,
        "ranks8_13_sign": ranks8_13_sign,
        "ranks5_7_diff_recomputed": ranks5_7_diff,
        "ranks5_7_sign_recomputed": ranks5_7_sign,
        "multiple_comparisons_summary": mc_summary,
        "combined_rank_summary": rank_rows,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq1_lower_ranks_robustness.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {out_json}")

    md = build_markdown(check1, next_group, lower_group, ranks5_7_diff, ranks5_7_sign,
                         ranks8_13_diff, ranks8_13_sign, mc_summary, rank_rows)
    out_md = args.out_dir / "rq1_lower_ranks_robustness.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
