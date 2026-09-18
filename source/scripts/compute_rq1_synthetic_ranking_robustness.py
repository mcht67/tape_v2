#!/usr/bin/env python3
"""
RQ1 -- Sign test and paired-difference robustness check, synthetic-data
backbone ranking.

Extends `compute_rq1_ranking_robustness.py` / `compute_rq1_sign_test_pairs.py`
(currently soundscape-only: `rq1_ranking_robustness.md`,
`rq1_sign_test_pairs.md`) to synthetic data, to check whether two Results
claims hold up statistically: "Bird-MAE performs best... followed, though
with a clear gap, by Perch v2..." and "backbones including bird-specific
pretraining... clearly outperform those merely trained on general audio
data". Source data: `archive/Pooled-Embeddings/` (13 backbones, frozen
backbone + pooled-embedding head, regression formulation), source
`synthetic_mixture_test`, metric "mae", the 8 ALL_DATASETS.

Six checks:

1. Difficulty-robust ranking, all 13 backbones (mirrors
   `compute_rq1_ranking_robustness.py`'s check 1, but for synthetic data):
   mean MAE +/- SD across the 8 datasets, paired by dataset (every backbone
   sees the same 8 datasets, so this is already difficulty-robust). Confirms
   the top 5 matches `rq1_top5_table.tex`
   (Table~\\ref{tab:rq1_top5_synthetic_mixture_test_reg}).

2-4. Pairwise paired-difference significance for Bird-MAE vs. each of the
   other 4 top-5 backbones (Perch v2 -- the specific "clear gap" claim --
   plus AudioProtoPNet, EfficientNet-B1, NatureLM-audio), reported together
   in one table since they are the same underlying comparison at different
   competitors: Delta = (Bird-MAE MAE) - (competitor MAE), matched by
   dataset (n=8), paired t-test and Wilcoxon signed-rank test against zero,
   *and* (same pairs, magnitude-independent complement, same method as
   `compute_rq1_sign_test_pairs.py`) an exact two-sided binomial sign test
   on the win/loss count across the 8 datasets.

5. Bird-specific vs. general-audio-pretrained, group-level check (the
   "clearly outperform" claim): using the Table3-groups coding already
   established in `compute_rq4_pretraining_domain_correlation.py`
   (TABLE3_BIRD_ONLY, n=6, vs. TABLE3_GENERAL_AUDIO, n=6; BEANS baseline
   fits neither and is excluded, same as there), paired cell-level
   differences (bird-specific MAE - general-audio MAE) for every cross-group
   (bird, general-audio) backbone pair, matched by dataset -- 6 x 6 x 8 = 288
   cells, pooled into one mean +/- SD, paired t-test/Wilcoxon, and sign test.
   Note this pools cells that are not fully independent (the same 8
   per-dataset MAE values recur across many cross-group pairs for a given
   backbone), same non-independence already accepted for the (backbone,
   dataset)-cell pooling in `compute_rq2_paired_cell_analysis.py` and
   `compute_rq5_kurtosis.py` -- reported as an aggregate exploratory summary,
   not an independent-samples test.

6. AST as the flagged exception: the manuscript carves AST out of the
   bird-specific-pretraining pattern despite Table3-groups nominally coding
   it "bird_only" (its `domain` field is "Other -> Bird", pretrained on
   general audio/speech before BirdSet-XCL fine-tuning). Same pairwise
   check as checks 2-4 (mean Delta +/- SD, paired t-test/Wilcoxon, sign
   test), AST vs. each of the other 5 TABLE3_BIRD_ONLY backbones and AST
   vs. each of the 6 TABLE3_GENERAL_AUDIO backbones (11 pairs), plus the
   same two pooled group-level summaries as check 5 (AST vs. the pooled
   bird-specific group, n=40; AST vs. the pooled general-audio group, n=48)
   -- to see whether AST's gap to its nominal "bird-specific" group is
   larger/more significant than its gap to the general-audio group, which
   is what "positions like general audio" would actually require.

Sample-size caveat: n=8 synthetic datasets is small but slightly better
powered than soundscape's n=7 (`rq1_ranking_robustness.md`,
`rq1_sign_test_pairs.md`) -- still treat every pairwise test here as
exploratory, not confirmatory, per the RQ1/RQ2/RQ5 small-n caution already
established. Separately from n, synthetic per-dataset MAE also has much
lower variance than soundscape's (see `build_markdown()`'s comparison to
`rq1_ranking_robustness.json`'s soundscape SDs) -- small n and high variance
are two distinct sources of statistical uncertainty, and this file's own
results (checked against the soundscape files) are used to see whether they
separate in practice.

Writes rq1_synthetic_ranking_robustness.json and
rq1_synthetic_ranking_robustness.md to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq1_synthetic_ranking_robustness.py [--out-dir plots/figures/rq1]
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
from compute_rq1_ranking_robustness import (
    ALL_DATASETS, SOURCE, SYNTHETIC_METRIC, build_matrix, diff_stats,
    fmt_mean_std, paired_tests,
)
from compute_rq4_pretraining_domain_correlation import (
    TABLE3_BIRD_ONLY, TABLE3_GENERAL_AUDIO,
)
from plot_data import load_study

N_DATASETS = len(ALL_DATASETS)

# Table~\ref{tab:rq1_top5_synthetic_mixture_test_reg} (rq1_top5_table.tex),
# in its own reported order.
TOP5_SYNTHETIC = ["Bird-MAE-Huge", "perch_v2_cpu", "EfficientNet-B1-BirdSet-XCL",
                   "NatureLMBEATs", "AudioProtoPNet-20-BirdSet-XCL"]
BIRD_MAE = "Bird-MAE-Huge"
# Order: Perch v2 first -- the specific "clear gap" claim (check 2) -- then
# the rest of the top 5 (check 3), same order rq1_top5_table.tex reports them.
COMPETITORS = [m for m in TOP5_SYNTHETIC if m != BIRD_MAE]

AST = "AST-Birdset-XCL"
BIRD_SPECIFIC_OTHERS = [m for m in TABLE3_BIRD_ONLY if m != AST]


def sign_test_from_diff(diff: np.ndarray) -> dict:
    """Exact two-sided binomial sign test -- same convention as
    compute_rq1_sign_test_pairs.py's sign_test(): wins = A better than B
    (diff < 0, lower MAE), losses = diff > 0, ties dropped before the test."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    wins = int((diff < 0).sum())
    losses = int((diff > 0).sum())
    ties = int(len(diff) - wins - losses)
    n_decisive = wins + losses
    p = round(float(scipy.stats.binomtest(wins, n_decisive, 0.5, alternative="two-sided").pvalue), 6) \
        if n_decisive > 0 else None
    return {"wins": wins, "losses": losses, "ties": ties, "n_decisive": n_decisive, "p_sign": p}


def pairwise_full(matrix: pd.DataFrame, a: str, b: str) -> dict:
    """Delta = A - B, matched by dataset -- mean/SD, paired t-test/Wilcoxon,
    and sign test in one place, since every pairwise check in this file
    reports all three together."""
    diff = (matrix.loc[a] - matrix.loc[b]).to_numpy(dtype=float)
    diff = diff[~np.isnan(diff)]
    return {"a": a, "b": b, **diff_stats(diff), **paired_tests(diff), **sign_test_from_diff(diff)}


def pooled_cross_group(matrix: pd.DataFrame, group_a: list, group_b: list) -> dict:
    """Pooled (group_a - group_b) cell-level differences across every
    cross-group (member_a, member_b) pair, matched by dataset -- see check
    5/6's docstring note on non-independence."""
    diffs = []
    for a in group_a:
        for b in group_b:
            diff = (matrix.loc[a] - matrix.loc[b]).to_numpy(dtype=float)
            diffs.append(diff[~np.isnan(diff)])
    pooled = np.concatenate(diffs)
    return {**diff_stats(pooled), **paired_tests(pooled), **sign_test_from_diff(pooled)}


# ---------------------------------------------------------------------------
# Check 1 -- difficulty-robust ranking, all 13 backbones, synthetic data.
# ---------------------------------------------------------------------------

def check1_ranking(long_df: pd.DataFrame, backbone_order: list) -> dict:
    matrix = build_matrix(long_df, SOURCE, SYNTHETIC_METRIC, ALL_DATASETS, backbone_order)
    mean_mae = matrix.mean(axis=1)
    sd_mae = matrix.std(axis=1, ddof=1)
    rank = mean_mae.rank(method="min").astype(int)
    order = mean_mae.sort_values().index.tolist()
    return {
        "order": order,
        "mean_mae": mean_mae.to_dict(),
        "sd_mae": sd_mae.to_dict(),
        "rank": rank.to_dict(),
        "n_datasets": N_DATASETS,
        "matches_top5_table": order[:5] == TOP5_SYNTHETIC,
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _fmt_p(x):
    return f"p={x:.3f}" if x is not None else "--"


def _sign_cell(s: dict) -> str:
    p = f"{s['p_sign']:.4f}" if s["p_sign"] is not None else "n/a"
    return f"{s['wins']}-{s['losses']}-{s['ties']} (p={p})"


def _pair_row(label: str, s: dict) -> str:
    return (f"| {label} | {fmt_mean_std(s['mean'], s['std'])} | {s['n']} | {_fmt_p(s['t_p'])} "
            f"| {_fmt_p(s['wilcoxon_p'])} | {_sign_cell(s)} |")


def build_markdown(results: dict, backbone_order: list, soundscape_ref: dict | None) -> str:
    r1 = results["check1"]
    lines = [
        "# RQ1 -- Sign test and paired-difference robustness check, synthetic-data backbone ranking",
        "",
        "Extends `rq1_ranking_robustness.md` / `rq1_sign_test_pairs.md` (soundscape-only) to synthetic "
        "data, checking whether \"Bird-MAE performs best... followed, though with a clear gap, by "
        "Perch v2...\" and \"backbones including bird-specific pretraining... clearly outperform those "
        "merely trained on general audio data\" hold up statistically. Source: "
        "`archive/Pooled-Embeddings/` (13 backbones, frozen backbone + pooled-MLP head, regression "
        "formulation), `synthetic_mixture_test` / mae, the 8 ALL_DATASETS.",
        "",
        f"**Caveat on sample size**: n={N_DATASETS} synthetic datasets is small for the pairwise checks "
        "below, though slightly better-powered than soundscape's n=7 -- treat every significance test "
        "here as exploratory, not confirmatory, consistent with the RQ1/RQ2/RQ5 small-n caution already "
        "established (`rq1_ranking_robustness.md`, `rq2_paired_cell_analysis.md`, `rq5_kurtosis.md`). "
        "Separately from n, per-dataset *variance* also matters -- see the note at the end comparing "
        "synthetic SDs here to soundscape's.",
        "",
    ]

    # --- Check 1 ---
    lines += [
        "## 1. Difficulty-robust ranking, all 13 backbones (synthetic, dataset-paired)",
        "",
        f"Mean MAE across the {r1['n_datasets']} synthetic datasets, paired by dataset. "
        f"**Matches the top 5 of Table~\\ref{{tab:rq1_top5_synthetic_mixture_test_reg}}: "
        f"{'yes' if r1['matches_top5_table'] else 'no'}.**",
        "",
        "| Rank | Backbone | Mean MAE (+/- SD) |",
        "|---|---|---|",
    ]
    for m in r1["order"]:
        marker = " (top 5)" if m in TOP5_SYNTHETIC else ""
        lines.append(f"| {r1['rank'][m]} | {BACKBONE_META[m]['display']}{marker} "
                      f"| {fmt_mean_std(r1['mean_mae'][m], r1['sd_mae'][m])} |")
    lines.append("")

    # --- Checks 2-4 (combined table) ---
    lines += [
        "## 2-4. Bird-MAE vs. Perch v2 and the full top-5, paired difference + sign test",
        "",
        "Delta = (Bird-MAE MAE) - (competitor MAE), matched by dataset (n=8). Check 2 is the first row "
        "(Perch v2, the specific \"clear gap\" claim); checks 3-4 are all 4 rows, adding the sign test "
        "columns (wins-losses-ties for Bird-MAE, exact two-sided binomial p) as a magnitude-independent "
        "complement, same method as `rq1_sign_test_pairs.md`.",
        "",
        "| Bird-MAE vs. | Mean Delta MAE (+/- SD) | n | Paired t-test | Wilcoxon | Sign test (W-L-T, p) |",
        "|---|---|---|---|---|---|",
    ]
    for s in results["checks_2_4"]:
        lines.append(_pair_row(BACKBONE_META[s["b"]]["display"], s))
    lines.append("")

    # --- Check 5 ---
    c5 = results["check5"]
    bird_labels = ", ".join(BACKBONE_META[m]["display"] for m in TABLE3_BIRD_ONLY)
    audio_labels = ", ".join(BACKBONE_META[m]["display"] for m in TABLE3_GENERAL_AUDIO)
    lines += [
        "## 5. Bird-specific vs. general-audio-pretrained, group-level check",
        "",
        f"Table3-groups coding from `rq4_pretraining_domain_correlation.py`: bird-specific = "
        f"{{{bird_labels}}} (n=6); general-audio-pretrained = {{{audio_labels}}} (n=6); BEANS baseline "
        "fits neither and is excluded, same as there. Delta = bird-specific MAE - general-audio MAE, "
        f"pooled over all 6 x 6 = 36 cross-group backbone pairs x {N_DATASETS} datasets = "
        f"{36 * N_DATASETS} cells. These cells are not fully independent (the same per-dataset MAE "
        "values recur across multiple cross-group pairs for a given backbone) -- reported as an "
        "aggregate exploratory summary, not an independent-samples test, same caveat already accepted "
        "for cell-pooling in `rq2_paired_cell_analysis.md` / `rq5_kurtosis.md`.",
        "",
        "| Comparison | Mean Delta MAE (+/- SD) | n | Paired t-test | Wilcoxon | Sign test (W-L-T, p) |",
        "|---|---|---|---|---|---|",
        f"| Bird-specific - general-audio (pooled) | {fmt_mean_std(c5['mean'], c5['std'])} | {c5['n']} "
        f"| {_fmt_p(c5['t_p'])} | {_fmt_p(c5['wilcoxon_p'])} | {_sign_cell(c5)} |",
        "",
        "**See `rq1_synthetic_group_test.md`** for a corrected version of this check using two genuinely "
        "two-sample tests (independent 6-vs-6 backbone means, and 48-vs-48 pooled per-dataset cells with "
        "no cross-pair repetition) instead of this 288-cell pooled-pairwise construction, plus an "
        "AST-excluded robustness check on each.",
        "",
    ]

    # --- Check 6 ---
    c6 = results["check6"]
    lines += [
        "## 6. AST as the flagged exception",
        "",
        "AST is nominally \"bird_only\" under the Table3-groups coding (its `domain` field is actually "
        "\"Other -> Bird\": pretrained on general audio/speech, then fine-tuned on BirdSet-XCL like the "
        "other bird-specific backbones) -- the manuscript flags it as behaving more like the "
        "general-audio group despite that nominal grouping. Same pairwise treatment as checks 2-4, AST "
        "vs. each of the other 5 bird-specific backbones and AST vs. each of the 6 general-audio "
        "backbones (Delta = AST MAE - other's MAE).",
        "",
        "| AST vs. | Group | Mean Delta MAE (+/- SD) | n | Paired t-test | Wilcoxon | Sign test (W-L-T, p) |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in c6["pairs"]:
        group = "bird-specific" if s["b"] in BIRD_SPECIFIC_OTHERS else "general-audio"
        lines.append(f"| {BACKBONE_META[s['b']]['display']} | {group} "
                      f"| {fmt_mean_std(s['mean'], s['std'])} | {s['n']} | {_fmt_p(s['t_p'])} "
                      f"| {_fmt_p(s['wilcoxon_p'])} | {_sign_cell(s)} |")
    lines += [
        "",
        "Pooled group-level summary (same construction as check 5, AST as the sole \"group\" on one "
        "side):",
        "",
        "| Comparison | Mean Delta MAE (+/- SD) | n | Paired t-test | Wilcoxon | Sign test (W-L-T, p) |",
        "|---|---|---|---|---|---|",
        f"| AST - bird-specific group (pooled) | {fmt_mean_std(c6['vs_bird']['mean'], c6['vs_bird']['std'])} "
        f"| {c6['vs_bird']['n']} | {_fmt_p(c6['vs_bird']['t_p'])} | {_fmt_p(c6['vs_bird']['wilcoxon_p'])} "
        f"| {_sign_cell(c6['vs_bird'])} |",
        f"| AST - general-audio group (pooled) | {fmt_mean_std(c6['vs_audio']['mean'], c6['vs_audio']['std'])} "
        f"| {c6['vs_audio']['n']} | {_fmt_p(c6['vs_audio']['t_p'])} | {_fmt_p(c6['vs_audio']['wilcoxon_p'])} "
        f"| {_sign_cell(c6['vs_audio'])} |",
        "",
        "If AST truly \"positions like general audio\", its gap to the bird-specific group (row 1) should "
        "be larger/more significant than its gap to the general-audio group (row 2, expected closer to "
        "zero/non-significant).",
    ]

    # --- Variance comparison note ---
    lines += ["", "## Synthetic vs. soundscape variance -- small n vs. high variance"]
    if soundscape_ref is not None:
        lines += [
            "",
            "Same backbones' per-dataset SD, synthetic (this file, n=8 datasets) vs. soundscape "
            "(`rq1_ranking_robustness.json`, n=7 datasets) -- isolates whether synthetic's better-behaved "
            "test results (if any, above) come from n=8 vs. n=7 alone, or from lower per-dataset variance "
            "at a given n, which are two separable sources of statistical uncertainty.",
            "",
            "| Backbone | Synthetic SD | Soundscape SD | Ratio (soundscape / synthetic) |",
            "|---|---|---|---|",
        ]
        for m in TOP5_SYNTHETIC:
            syn_sd = r1["sd_mae"][m]
            scape_sd = soundscape_ref.get(m)
            ratio = f"{scape_sd / syn_sd:.1f}x" if scape_sd is not None and syn_sd else "--"
            scape_str = f"{scape_sd:.3f}" if scape_sd is not None else "--"
            lines.append(f"| {BACKBONE_META[m]['display']} | {syn_sd:.3f} | {scape_str} | {ratio} |")
        lines += [
            "",
            "If this ratio is well above 1 across the top-5 backbones, synthetic's pairwise tests above "
            "being more confirmable than the soundscape ones (`rq1_ranking_robustness.md`'s checks 2-3) "
            "is attributable at least partly to lower per-dataset variance, not only to n=8 vs. n=7 -- a "
            "useful distinction, since a higher-n soundscape study (more regional datasets) would not by "
            "itself close this gap if the variance difference is the dominant factor.",
        ]
    else:
        lines += [
            "",
            "(`rq1_ranking_robustness.json` not found -- run `compute_rq1_ranking_robustness.py` first to "
            "populate this comparison.)",
        ]

    lines += [
        "",
        "## Reading this",
        "",
        "- Check 1 confirms the reported synthetic ranking and extends it to all 13 backbones.",
        "- Checks 2-4: a significant paired t-test/Wilcoxon *and* a lopsided sign test (e.g. 8-0 or 7-1) "
        "together would support the \"clear gap\" claim as more than a point-estimate artifact; a "
        "significant mean-difference test with a closer sign-test split (e.g. 5-3) would suggest the gap "
        "is consistent in direction but not dominant on every dataset.",
        "- Check 5 is the best-powered test in this file (n=288, pooled) but is not an independent-samples "
        "test (see its own caveat above) -- read its significance as suggestive of a genuine group-level "
        "gap, not as a formally valid p-value.",
        "- Check 6: if AST's delta to the general-audio group is small/non-significant while its delta to "
        "the bird-specific group is large/significant, that is direct quantitative support for the "
        "manuscript's exception claim, not just a point-estimate observation.",
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

    top5_matrix = build_matrix(long_df, SOURCE, SYNTHETIC_METRIC, ALL_DATASETS, [BIRD_MAE] + COMPETITORS)
    checks_2_4 = [pairwise_full(top5_matrix, BIRD_MAE, b) for b in COMPETITORS]

    group_matrix = build_matrix(long_df, SOURCE, SYNTHETIC_METRIC, ALL_DATASETS,
                                 TABLE3_BIRD_ONLY + TABLE3_GENERAL_AUDIO)
    check5 = pooled_cross_group(group_matrix, TABLE3_BIRD_ONLY, TABLE3_GENERAL_AUDIO)

    ast_matrix = build_matrix(long_df, SOURCE, SYNTHETIC_METRIC, ALL_DATASETS,
                               [AST] + BIRD_SPECIFIC_OTHERS + TABLE3_GENERAL_AUDIO)
    check6_pairs = [pairwise_full(ast_matrix, AST, b) for b in BIRD_SPECIFIC_OTHERS + TABLE3_GENERAL_AUDIO]
    check6_vs_bird = pooled_cross_group(ast_matrix, [AST], BIRD_SPECIFIC_OTHERS)
    check6_vs_audio = pooled_cross_group(ast_matrix, [AST], TABLE3_GENERAL_AUDIO)
    check6 = {"pairs": check6_pairs, "vs_bird": check6_vs_bird, "vs_audio": check6_vs_audio}

    results = {"check1": check1, "checks_2_4": checks_2_4, "check5": check5, "check6": check6}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq1_synthetic_ranking_robustness.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {out_json}")

    soundscape_ref = None
    soundscape_json_path = args.out_dir / "rq1_ranking_robustness.json"
    if soundscape_json_path.is_file():
        soundscape_ref = json.loads(soundscape_json_path.read_text())["check1"]["sd_mae"]

    md = build_markdown(results, backbone_order, soundscape_ref)
    out_md = args.out_dir / "rq1_synthetic_ranking_robustness.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
