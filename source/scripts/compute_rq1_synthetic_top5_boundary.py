#!/usr/bin/env python3
"""
RQ1 -- Sign test and paired-difference check, synthetic-data top-5 vs.
next-ranked backbones.

Extends `compute_rq1_synthetic_ranking_robustness.py`'s checks 2-4 (Bird-MAE
vs. each of the other top-5 members -- already confirmed) to ask the
remaining question about the synthetic-data top 5: do ranks 2-5 (Perch v2,
EfficientNet-B1, NatureLM-audio, AudioProtoPNet) collectively hold up against
rank 6 onward? This is the synthetic-data equivalent of
`compute_rq1_ranking_robustness.py`'s check 3 ("tight leading group vs.
next-ranked") for soundscape data.

Source data: same as `compute_rq1_synthetic_ranking_robustness.py` --
`archive/Pooled-Embeddings/` (13 backbones, frozen backbone + pooled-MLP
head, regression formulation), `synthetic_mixture_test` / mae, the 8
ALL_DATASETS.

Three checks:

1-2. Paired cell-level mean-difference test (paired t-test + Wilcoxon
   signed-rank) *and* exact two-sided binomial sign test, for each of the 4
   top-5-minus-Bird-MAE backbones against each of the next three ranked
   backbones (rank 6: Perch v1; rank 7: BirdNET v2.4; rank 8: BirdNET v2.3)
   -- 4 x 3 = 12 pairs, Delta = (top-5 backbone MAE) - (lower-ranked backbone
   MAE), matched by dataset (n=8). Reported together in one table, same
   convention as the parent file's checks 2-4.

3. Extension to ranks 9-13 (Wav2Vec2, VGGish, YAMNet, AST, BEANS baseline) --
   4 x 5 = 20 more pairs -- run only if any of the 12 check-1/2 pairs is
   non-significant by every method (t-test, Wilcoxon, sign test), i.e. checks
   1-2 don't already show a clean boundary at rank 6-8.

Multiple-comparisons note: reports how many of the total comparisons run (12,
or 32 if check 3 fires) clear the uncorrected p<0.05 threshold for the paired
t-test, Wilcoxon, and sign test individually, plus how many would survive a
Bonferroni correction across that same total.

Summary: the lowest-numbered rank (closest to the top 5) at which all four
top-5-minus-Bird-MAE backbones are simultaneously and significantly
distinguishable (by at least one of the three methods) from that rank and
every lower-ranked one after it -- i.e. the point past which the gap is
confirmed rather than just apparent in the mean-MAE ordering.

Sample-size caveat, same as the parent file: n=8 synthetic datasets is small
-- treat every pairwise test here as exploratory, not confirmatory, per the
RQ1/RQ2/RQ5 small-n caution already established. Synthetic per-dataset MAE
has much lower variance than soundscape's (see
`rq1_synthetic_ranking_robustness.md`'s variance-ratio table), so these tests
are plausibly better-powered than a soundscape equivalent would be -- the
summary section checks this expectation against the actual results rather
than assuming it.

Writes rq1_synthetic_top5_boundary.json and rq1_synthetic_top5_boundary.md
to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq1_synthetic_top5_boundary.py [--out-dir plots/figures/rq1]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import BACKBONE_META
from compute_rq1_ranking_robustness import (
    ALL_DATASETS, SOURCE, SYNTHETIC_METRIC, build_matrix, fmt_mean_std,
)
from compute_rq1_synthetic_ranking_robustness import (
    BIRD_MAE, COMPETITORS, N_DATASETS, TOP5_SYNTHETIC, check1_ranking,
    pairwise_full,
)
from plot_data import load_study

N_NEXT = 3   # ranks 6, 7, 8 -- the first check
N_FURTHER = 5  # ranks 9-13 -- the extension, run only if needed

ALPHA = 0.05


def is_significant(s: dict, alpha: float = ALPHA) -> bool:
    """True if p < alpha for any of the three methods (t-test, Wilcoxon,
    sign test) -- "distinguishable by at least one method"."""
    ps = [s.get("t_p"), s.get("wilcoxon_p"), s.get("p_sign")]
    return any(p is not None and p < alpha for p in ps)


def run_block(matrix, top5_backbones: list, lower_backbones: list) -> list:
    return [pairwise_full(matrix, a, b) for a in top5_backbones for b in lower_backbones]


def multiple_comparisons_summary(pairs: list, total_for_bonferroni: int) -> dict:
    n = len(pairs)
    bonf_alpha = ALPHA / total_for_bonferroni
    sig_t = sum(1 for s in pairs if s["t_p"] is not None and s["t_p"] < ALPHA)
    sig_w = sum(1 for s in pairs if s["wilcoxon_p"] is not None and s["wilcoxon_p"] < ALPHA)
    sig_both = sum(1 for s in pairs if s["t_p"] is not None and s["wilcoxon_p"] is not None
                   and s["t_p"] < ALPHA and s["wilcoxon_p"] < ALPHA)
    sig_sign = sum(1 for s in pairs if s["p_sign"] is not None and s["p_sign"] < ALPHA)
    sig_t_bonf = sum(1 for s in pairs if s["t_p"] is not None and s["t_p"] < bonf_alpha)
    sig_w_bonf = sum(1 for s in pairs if s["wilcoxon_p"] is not None and s["wilcoxon_p"] < bonf_alpha)
    sig_both_bonf = sum(1 for s in pairs if s["t_p"] is not None and s["wilcoxon_p"] is not None
                         and s["t_p"] < bonf_alpha and s["wilcoxon_p"] < bonf_alpha)
    sig_sign_bonf = sum(1 for s in pairs if s["p_sign"] is not None and s["p_sign"] < bonf_alpha)
    return {
        "n": n, "bonferroni_total": total_for_bonferroni, "bonferroni_alpha": bonf_alpha,
        "sig_t_uncorrected": sig_t, "sig_wilcoxon_uncorrected": sig_w,
        "sig_both_uncorrected": sig_both, "sig_sign_uncorrected": sig_sign,
        "sig_t_bonferroni": sig_t_bonf, "sig_wilcoxon_bonferroni": sig_w_bonf,
        "sig_both_bonferroni": sig_both_bonf, "sig_sign_bonferroni": sig_sign_bonf,
    }


def find_boundary(pairs_by_rank: dict, rank_order: list) -> dict | None:
    """rank_order is the list of lower-ranked backbones in ascending-rank
    order (rank 6, 7, 8, ...). Returns the first rank (going down the
    ordering) from which every subsequent rank (inclusive) has all 4 top-5
    comparisons significant -- i.e. once the boundary is crossed it does not
    reopen. Returns None if no such rank exists within what was tested."""
    n = len(rank_order)
    for start in range(n):
        if all(all(is_significant(s) for s in pairs_by_rank[b]) for b in rank_order[start:]):
            return {"rank_backbone": rank_order[start], "index_from_start": start}
    return None


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _fmt_p(x):
    return f"p={x:.3f}" if x is not None else "--"


def _sign_cell(s: dict) -> str:
    p = f"{s['p_sign']:.4f}" if s["p_sign"] is not None else "n/a"
    return f"{s['wins']}-{s['losses']}-{s['ties']} (p={p})"


def _pair_row(s: dict) -> str:
    label = f"{BACKBONE_META[s['a']]['display']} vs. {BACKBONE_META[s['b']]['display']}"
    sig = " **" if is_significant(s) else ""
    return (f"| {label} | {fmt_mean_std(s['mean'], s['std'])} | {s['n']} | {_fmt_p(s['t_p'])} "
            f"| {_fmt_p(s['wilcoxon_p'])} | {_sign_cell(s)}{sig} |")


def _mc_row(label: str, mc: dict) -> str:
    return (f"| {label} | {mc['n']} | {mc['sig_t_uncorrected']} | {mc['sig_wilcoxon_uncorrected']} "
            f"| {mc['sig_both_uncorrected']} | {mc['sig_sign_uncorrected']} "
            f"| {mc['bonferroni_alpha']:.5f} | {mc['sig_t_bonferroni']} | {mc['sig_wilcoxon_bonferroni']} "
            f"| {mc['sig_both_bonferroni']} | {mc['sig_sign_bonferroni']} |")


def build_markdown(results: dict) -> str:
    r1 = results["ranking"]
    checks12 = results["checks_1_2"]
    ranks_6_8 = results["ranks_6_8"]
    extended = results["extended"]
    lines = [
        "# RQ1 -- Sign test and paired-difference check, synthetic-data top-5 vs. next-ranked backbones",
        "",
        "Extends `rq1_synthetic_ranking_robustness.md`'s checks 2-4 (Bird-MAE vs. each of the other "
        "top-5 members, already confirmed) to test whether ranks 2-5 (Perch v2, EfficientNet-B1, "
        "NatureLM-audio, AudioProtoPNet) collectively hold up against rank 6 onward -- the synthetic-data "
        "equivalent of `rq1_ranking_robustness.md`'s \"leading group vs. next-ranked\" check. Source: "
        "`archive/Pooled-Embeddings/` (13 backbones, frozen backbone + pooled-MLP head, regression "
        f"formulation), `synthetic_mixture_test` / mae, the {N_DATASETS} ALL_DATASETS.",
        "",
        f"**Caveat on sample size**: n={N_DATASETS} synthetic datasets -- treat every significance test "
        "here as exploratory, not confirmatory, consistent with the RQ1/RQ2/RQ5 small-n caution already "
        "established. Synthetic per-dataset MAE has much lower variance than soundscape's "
        "(`rq1_synthetic_ranking_robustness.md`'s variance-ratio table), so these tests are plausibly "
        "better-powered than a soundscape equivalent -- checked explicitly against the actual results in "
        "the summary below rather than assumed.",
        "",
        "## Full ranking (reference, from `rq1_synthetic_ranking_robustness.md` check 1)",
        "",
        "| Rank | Backbone | Mean MAE (+/- SD) |",
        "|---|---|---|",
    ]
    for m in r1["order"]:
        marker = " (top 5)" if m in TOP5_SYNTHETIC else ""
        lines.append(f"| {r1['rank'][m]} | {BACKBONE_META[m]['display']}{marker} "
                      f"| {fmt_mean_std(r1['mean_mae'][m], r1['sd_mae'][m])} |")
    lines.append("")

    lines += [
        "## 1-2. Top-5 (minus Bird-MAE) vs. ranks 6-8, paired difference + sign test",
        "",
        f"Delta = (top-5 backbone MAE) - (lower-ranked backbone MAE), matched by dataset (n={N_DATASETS}). "
        f"Ranks 6-8: {', '.join(BACKBONE_META[m]['display'] for m in ranks_6_8)}. "
        f"{len(COMPETITORS)} x {len(ranks_6_8)} = {len(checks12)} pairs. "
        "**bold** marks pairs significant (p<0.05) by at least one of the three methods.",
        "",
        "| Comparison | Mean Delta MAE (+/- SD) | n | Paired t-test | Wilcoxon | Sign test (W-L-T, p) |",
        "|---|---|---|---|---|---|",
    ]
    for s in checks12:
        lines.append(_pair_row(s))
    lines.append("")

    if extended is not None:
        ranks_9_13 = results["ranks_9_13"]
        checks3 = extended["checks_3"]
        lines += [
            "## 3. Extension to ranks 9-13",
            "",
            "At least one of the 12 comparisons above was not significant by every method, so ranks 6-8 "
            "don't show a clean boundary on their own -- extending the same paired-difference and "
            f"sign-test treatment to ranks 9-13: {', '.join(BACKBONE_META[m]['display'] for m in ranks_9_13)}. "
            f"{len(COMPETITORS)} x {len(ranks_9_13)} = {len(checks3)} additional pairs.",
            "",
            "| Comparison | Mean Delta MAE (+/- SD) | n | Paired t-test | Wilcoxon | Sign test (W-L-T, p) |",
            "|---|---|---|---|---|---|",
        ]
        for s in checks3:
            lines.append(_pair_row(s))
        lines.append("")
        total_comparisons = len(checks12) + len(checks3)
    else:
        lines += [
            "## 3. Extension to ranks 9-13",
            "",
            "Not run: all 12 comparisons in checks 1-2 were already significant by at least one method, "
            "so ranks 6-8 show a clean boundary and there is no need to test further down the ranking.",
            "",
        ]
        total_comparisons = len(checks12)

    # --- Multiple comparisons ---
    lines += [
        "## Multiple-comparisons note",
        "",
        f"Total comparisons run: {total_comparisons}"
        + (f" (12 for ranks 6-8, {total_comparisons - 12} more for ranks 9-13)." if extended is not None
           else " (ranks 6-8 only; extension not needed)."),
        f" Bonferroni-corrected alpha across all {total_comparisons}: {ALPHA / total_comparisons:.5f}.",
        "",
        "| Block | n pairs | Sig. t-test (p<.05) | Sig. Wilcoxon (p<.05) | Sig. both | Sig. sign test "
        "(p<.05) | Bonferroni alpha | Sig. t-test (Bonf.) | Sig. Wilcoxon (Bonf.) | Sig. both (Bonf.) "
        "| Sig. sign test (Bonf.) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        _mc_row("Ranks 6-8 (checks 1-2)", results["mc_checks12"]),
    ]
    if extended is not None:
        lines.append(_mc_row("Ranks 9-13 (check 3)", extended["mc_check3"]))
        lines.append(_mc_row("All comparisons combined", results["mc_all"]))
    lines.append("")
    lines += [
        f"**Wilcoxon/sign-test floor effect**: with n={N_DATASETS} datasets, the exact two-sided "
        f"Wilcoxon signed-rank and sign tests can reach a minimum p of about 2/2^{N_DATASETS} = "
        f"{2 / 2 ** N_DATASETS:.4f} (all {N_DATASETS} datasets agreeing in sign), which is *larger* than "
        f"the Bonferroni alpha above ({ALPHA / total_comparisons:.5f}) -- so those two tests can never "
        "clear Bonferroni correction at this sample size, regardless of true effect size. The 0s in the "
        "Wilcoxon/sign \"(Bonf.)\" columns above reflect this discreteness floor, not an absence of "
        "effect; the paired t-test (continuous, not floor-limited the same way) is the only one of the "
        "three methods that can meaningfully be read after Bonferroni correction here.",
        "",
    ]

    # --- Summary ---
    boundary = results["boundary"]
    lines += ["## Summary", ""]
    if boundary is not None:
        b_meta = BACKBONE_META[boundary["rank_backbone"]]
        b_rank = r1["rank"][boundary["rank_backbone"]]
        lines.append(
            f"The lowest-ranked backbone at which all four top-5 members (Perch v2, EfficientNet-B1, "
            f"NatureLM-audio, AudioProtoPNet) are simultaneously and significantly distinguishable from it "
            f"(by at least one method), and remain so for every lower-ranked backbone tested after it, is "
            f"**rank {b_rank}, {b_meta['display']}**."
        )
    else:
        lines.append(
            "No rank within the tested range (6-13) satisfies the criterion of all four top-5 members "
            "being simultaneously and significantly distinguishable from it (and from every lower-ranked "
            "backbone after it) by at least one method -- stated explicitly per the task's instruction, "
            "rather than defaulting to the best candidate below."
        )
    lines += [
        "",
        "Per-rank status (all four top-5-minus-Bird-MAE comparisons significant by at least one method?):",
        "",
        "| Rank | Backbone | All 4 significant? |",
        "|---|---|---|",
    ]
    for b in results["all_lower_ranks_tested"]:
        all_sig = all(is_significant(s) for s in results["pairs_by_rank"][b])
        lines.append(f"| {r1['rank'][b]} | {BACKBONE_META[b]['display']} | {'yes' if all_sig else 'no'} |")

    lines += [
        "",
        "**Power expectation check**: the docstring/caveat above expected these synthetic-data tests to "
        "be better-powered than their soundscape equivalents, given synthetic's much lower per-dataset "
        "variance. Against the actual results: "
        + results["power_check_note"],
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq1"))
    args = parser.parse_args()

    models = mrt.discover_models(Path("archive/Pooled-Embeddings"))
    long_df = load_study("pooled", models=models)
    backbone_order = [m for m in BACKBONE_META if m in models]

    ranking = check1_ranking(long_df, backbone_order)
    order = ranking["order"]

    ranks_6_8 = order[5:5 + N_NEXT]
    ranks_9_13 = order[5 + N_NEXT:5 + N_NEXT + N_FURTHER]

    matrix_6_8 = build_matrix(long_df, SOURCE, SYNTHETIC_METRIC, ALL_DATASETS, COMPETITORS + ranks_6_8)
    checks_1_2 = run_block(matrix_6_8, COMPETITORS, ranks_6_8)

    pairs_by_rank = {b: [s for s in checks_1_2 if s["b"] == b] for b in ranks_6_8}

    clean_boundary = all(is_significant(s) for s in checks_1_2)

    extended = None
    total_pairs = checks_1_2
    if not clean_boundary:
        matrix_9_13 = build_matrix(long_df, SOURCE, SYNTHETIC_METRIC, ALL_DATASETS, COMPETITORS + ranks_9_13)
        checks_3 = run_block(matrix_9_13, COMPETITORS, ranks_9_13)
        for b in ranks_9_13:
            pairs_by_rank[b] = [s for s in checks_3 if s["b"] == b]
        total_pairs = checks_1_2 + checks_3
        mc_check3 = multiple_comparisons_summary(checks_3, len(checks_1_2) + len(checks_3))
        extended = {"checks_3": checks_3, "mc_check3": mc_check3}

    total_n = len(total_pairs)
    mc_checks12 = multiple_comparisons_summary(checks_1_2, total_n)
    all_lower_ranks_tested = ranks_6_8 + (ranks_9_13 if extended is not None else [])
    boundary = find_boundary(pairs_by_rank, all_lower_ranks_tested)

    if extended is not None:
        mc_all = multiple_comparisons_summary(total_pairs, total_n)

    # Power-check note: compare min/median p across these checks to the
    # analogous soundscape check (rq1_ranking_robustness.json check 3), if
    # available.
    soundscape_json = args.out_dir / "rq1_ranking_robustness.json"
    # Soundscape's check 3 (rq1_ranking_robustness.py) has no sign test, so
    # use "t-test or Wilcoxon < 0.05" -- the common subset of methods -- on
    # both sides for a like-for-like comparison, rather than mixing this
    # file's 3-method criterion against soundscape's 2-method one.
    def sig_t_or_w(s):
        return (s.get("t_p") is not None and s["t_p"] < ALPHA) or \
               (s.get("wilcoxon_p") is not None and s["wilcoxon_p"] < ALPHA)

    frac_sig_any = sum(is_significant(s) for s in total_pairs) / len(total_pairs)
    frac_sig_tw = sum(sig_t_or_w(s) for s in total_pairs) / len(total_pairs)
    if soundscape_json.is_file():
        scape = json.loads(soundscape_json.read_text())
        scape_pairs = list(scape.get("check3", {}).get("pairs", {}).values())
        if scape_pairs:
            scape_frac_sig = sum(sig_t_or_w(s) for s in scape_pairs) / len(scape_pairs)
            power_check_note = (
                f"using a like-for-like criterion (t-test *or* Wilcoxon <0.05, the two methods both files "
                f"share), {frac_sig_tw * 100:.0f}% of the {total_n} synthetic-data comparisons here are "
                f"significant, vs. {scape_frac_sig * 100:.0f}% of the {len(scape_pairs)} soundscape "
                "leading-group-vs-next-ranked comparisons in `rq1_ranking_robustness.md` check 3 -- "
                + ("confirms" if frac_sig_tw >= scape_frac_sig else "does not confirm")
                + " the expectation that synthetic's lower per-dataset variance yields better-powered "
                  "pairwise tests over this specific rank range. (Adding this file's sign test as a third "
                  f"method raises the synthetic figure to {frac_sig_any * 100:.0f}%, but that method isn't "
                  "available for the soundscape file to compare against.)"
            )
        else:
            power_check_note = "soundscape check-3 comparison not found in `rq1_ranking_robustness.json`; expectation not checked."
    else:
        power_check_note = "`rq1_ranking_robustness.json` not found; run `compute_rq1_ranking_robustness.py` first to check this expectation."

    results = {
        "ranking": ranking,
        "ranks_6_8": ranks_6_8,
        "ranks_9_13": ranks_9_13,
        "checks_1_2": checks_1_2,
        "pairs_by_rank": pairs_by_rank,
        "clean_boundary_at_ranks_6_8": clean_boundary,
        "extended": extended,
        "mc_checks12": mc_checks12,
        "mc_all": mc_all if extended is not None else mc_checks12,
        "boundary": boundary,
        "all_lower_ranks_tested": all_lower_ranks_tested,
        "power_check_note": power_check_note,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq1_synthetic_top5_boundary.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {out_json}")

    md = build_markdown(results)
    out_md = args.out_dir / "rq1_synthetic_top5_boundary.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
