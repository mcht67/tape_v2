#!/usr/bin/env python3
"""
RQ4 (additional) -- VGGish/YAMNet soundscape ranking proximity to the
"weaker bird-pretrained" backbones.

Purpose: check whether the RQ4 narrative claim "VGGish and YAMNet close
much of the gap on soundscape data, landing in the same range as the
weaker bird-pretrained models" is statistically supported, or whether --
like the RQ1 "tight leading group" claim (`rq1_ranking_robustness.md`
check 3) and the RQ4 per-backbone synthetic-vs-soundscape direction claims
(`rq4_sign_test.md` check 1) -- it is a point-estimate observation that
does not survive testing at this sample size.

Source data: `archive/Pooled-Embeddings/` (13 backbones, frozen backbone +
pooled-MLP head, regression formulation), `soundscape_test` / range_mae
over the 7 REGIONAL_DATASETS -- same source and methodology as
`compute_rq1_ranking_robustness.py` / `compute_rq1_lower_ranks_robustness.py`,
whose functions this script reuses directly rather than reimplementing.

Step 1 -- identify "the weaker bird-pretrained models": among the backbones
whose `domain` (`backbone_meta.BACKBONE_META`) is "Bird" or "Other -> Bird"
(i.e. actually bird-pretrained, either from scratch or via fine-tuning --
excludes the "Cross-taxa" BirdNET/Perch v2 pair, "General audio" VGGish/
YAMNet, "General audio + cross-taxa" NatureLM-audio, and the untrained
BEANS baseline), excluding the tight leading group's own bird-pretrained
member (Bird-MAE), the 3 lowest-ranked by the check1 soundscape ranking --
confirmed against the actual ranking, not assumed.

Step 2 -- for VGGish and YAMNet against that weaker-bird-pretrained group,
and against the tight leading group (Bird-MAE, Perch v2, BirdNET v2.3,
BirdNET v2.4) for contrast:

1. Paired cell-level mean-difference test (paired t-test, Wilcoxon),
   matched by dataset (n=7).
2. Exact sign test, same pairs: win/loss/tie count and exact binomial p.

Sample-size caveat, same as the RQ1 files this reuses: n=7 soundscape
datasets is small -- every pairwise test here is exploratory/directional.
A plausible outcome, given that ranks 5-7 (which include VGGish) were
already shown statistically indistinguishable from the leading group in
`rq1_ranking_robustness.md` and ranks 8-13 (which include YAMNet) likewise
in `rq1_lower_ranks_robustness.md`, is that essentially nothing in this
13-backbone ranking is statistically separable at n=7 -- itself a
reportable methodological finding, not a null result to discard.

Writes rq4_vggish_yamnet_proximity.json and rq4_vggish_yamnet_proximity.md
to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq4_vggish_yamnet_proximity.py [--out-dir plots/figures/rq4]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import BACKBONE_META
from compute_rq1_ranking_robustness import (
    REGIONAL_DATASETS, SCAPE_SOURCE, SOUNDSCAPE_METRIC, TIGHT_GROUP,
    build_matrix, check1_ranking, fmt_mean_std, pairwise_diff,
)
from compute_rq1_sign_test_pairs import pairwise_sign_test
from plot_data import load_study

BIRD_PRETRAINED_DOMAINS = {"Bird", "Other -> Bird"}
N_WEAK = 3  # size of the "weaker bird-pretrained" group
TARGETS = ["vggish", "yamnet"]


def _fmt_p(x):
    return f"p={x:.3f}" if x is not None else "--"


def _fmt_sign_p(x):
    return f"{x:.4f}" if x is not None else "n/a, n<=3"


def identify_weak_bird_pretrained(check1: dict, backbone_order: list) -> list:
    """The N_WEAK lowest-ranked (highest-MAE) backbones among those with
    domain "Bird" or "Other -> Bird" (bird-pretrained, from scratch or via
    fine-tuning), excluding TIGHT_GROUP's own bird-pretrained member."""
    bird_pretrained = [m for m in backbone_order
                        if BACKBONE_META[m]["domain"] in BIRD_PRETRAINED_DOMAINS
                        and m not in TIGHT_GROUP]
    ranked = sorted(bird_pretrained, key=lambda m: check1["mean_mae"][m])
    return ranked[-N_WEAK:]


def compute_pairs(matrix: pd.DataFrame, targets: list, group: list) -> tuple[dict, dict]:
    diff_results = {f"{a}__minus__{b}": pairwise_diff(matrix, a, b)
                     for a in targets for b in group}
    sign_results = {f"{a}__minus__{b}": pairwise_sign_test(matrix, a, b)
                    for a in targets for b in group}
    return diff_results, sign_results


def _diff_table(diff_results: dict) -> list:
    lines = ["| Pair (A - B) | Mean Delta MAE (+/- SD) | n | Paired t-test | Wilcoxon |",
             "|---|---|---|---|---|"]
    for s in diff_results.values():
        a_label, b_label = BACKBONE_META[s["a"]]["display"], BACKBONE_META[s["b"]]["display"]
        lines.append(f"| {a_label} - {b_label} | {fmt_mean_std(s['mean'], s['std'])} | {s['n']} "
                      f"| {_fmt_p(s['t_p'])} | {_fmt_p(s['wilcoxon_p'])} |")
    return lines


def _sign_table(sign_results: dict) -> list:
    lines = ["| Pair (A - B) | n datasets | Wins | Losses | Ties | Sign-test $p$ |",
             "|---|---|---|---|---|---|"]
    for s in sign_results.values():
        a_label, b_label = BACKBONE_META[s["a"]]["display"], BACKBONE_META[s["b"]]["display"]
        lines.append(f"| {a_label} - {b_label} | {s['n_datasets']} | {s['wins']} | {s['losses']} "
                      f"| {s['ties']} | {_fmt_sign_p(s['p_two_sided'])} |")
    return lines


def any_significant(diff_results: dict, sign_results: dict, alpha: float = 0.05) -> bool:
    for s in diff_results.values():
        if (s["t_p"] is not None and s["t_p"] < alpha) or (s["wilcoxon_p"] is not None and s["wilcoxon_p"] < alpha):
            return True
    for s in sign_results.values():
        if s["p_two_sided"] is not None and s["p_two_sided"] < alpha:
            return True
    return False


def build_markdown(check1: dict, weak_group: list, target: str,
                    weak_diff: dict, weak_sign: dict, lead_diff: dict, lead_sign: dict) -> str:
    target_label = BACKBONE_META[target]["display"]
    weak_labels = ", ".join(BACKBONE_META[m]["display"] for m in weak_group)
    lead_labels = ", ".join(BACKBONE_META[m]["display"] for m in TIGHT_GROUP)

    weak_mae = fmt_mean_std(check1["mean_mae"][target], check1["sd_mae"][target])
    weak_rank = check1["rank"][target]

    lines = [
        f"### {target_label} (rank {weak_rank}, {weak_mae}) vs. the weaker bird-pretrained group "
        f"({weak_labels})",
        "",
    ]
    lines += _diff_table(weak_diff)
    lines.append("")
    lines += _sign_table(weak_sign)
    lines.append("")
    lines += [
        f"### {target_label} vs. the leading group ({lead_labels}) -- contrast",
        "",
    ]
    lines += _diff_table(lead_diff)
    lines.append("")
    lines += _sign_table(lead_sign)
    lines.append("")

    weak_sig = any_significant(weak_diff, weak_sign)
    lead_sig = any_significant(lead_diff, lead_sign)
    if not weak_sig and not lead_sig:
        verdict = (f"**Neither comparison reaches uncorrected p<0.05 for {target_label}: it is "
                    "statistically indistinguishable from both the weaker bird-pretrained group and the "
                    "leading group at n=7.**")
    elif weak_sig and not lead_sig:
        verdict = (f"**{target_label} is nominally distinguishable from the weaker bird-pretrained group "
                    "but not from the leading group -- the only pattern consistent with a genuine "
                    "\"closes much of the gap\" effect, though still subject to the n=7 caveat below.**")
    elif not weak_sig and lead_sig:
        verdict = (f"**{target_label} is distinguishable from the leading group but not from the weaker "
                    "bird-pretrained group -- consistent with landing in the same range as the weaker "
                    "bird-pretrained models, though still subject to the n=7 caveat below.**")
    else:
        verdict = (f"**{target_label} is nominally distinguishable from both groups at n=7 -- an "
                    "inconsistent pattern that warrants reading the individual pairs above rather than "
                    "this summary alone.**")
    lines.append(verdict)
    lines.append("")
    return "\n".join(lines)


def build_full_markdown(check1: dict, weak_group: list, backbone_order: list,
                         per_target: dict) -> str:
    weak_labels = ", ".join(BACKBONE_META[m]["display"] for m in weak_group)
    lead_labels = ", ".join(BACKBONE_META[m]["display"] for m in TIGHT_GROUP)
    target_labels = ", ".join(BACKBONE_META[m]["display"] for m in TARGETS)

    lines = [
        "# RQ4 -- VGGish/YAMNet soundscape ranking proximity to the weaker bird-pretrained backbones",
        "",
        "Checks the RQ4 narrative claim that \"VGGish and YAMNet close much of the gap on soundscape "
        "data, landing in the same range as the weaker bird-pretrained models\": is this statistically "
        f"supported, or -- like the RQ1 tight-leading-group claim (`rq1_ranking_robustness.md` check 3, "
        "`rq1_lower_ranks_robustness.md`) and the RQ4 per-backbone synthetic-vs-soundscape direction "
        "claims (`rq4_sign_test.md` check 1) -- a point-estimate observation that does not survive "
        "testing at this sample size? Source data: `archive/Pooled-Embeddings/` (13 backbones, frozen "
        "backbone + pooled-MLP head, regression formulation), `soundscape_test` / range_mae over the 7 "
        "REGIONAL_DATASETS -- same source and methodology as the RQ1 ranking-robustness files, whose "
        "functions this script reuses directly.",
        "",
        "**Caveat on sample size**: n=7 soundscape datasets is small -- treat every pairwise test below "
        "as exploratory/directional, consistent with the small-n caution already established throughout "
        "RQ1/RQ2/RQ4/RQ5 (`rq1_ranking_robustness.md`, `rq1_lower_ranks_robustness.md`, "
        "`rq2_paired_cell_analysis.md`, `rq4_sign_test.md`, `rq5_kurtosis.md`). Given that ranks 5-7 "
        "(which include VGGish) were already shown statistically indistinguishable from the leading "
        "group in `rq1_ranking_robustness.md` check 3, and ranks 8-13 (which include YAMNet) likewise in "
        "`rq1_lower_ranks_robustness.md`, a plausible outcome here is that VGGish/YAMNet are *also* "
        "statistically indistinguishable from the weaker bird-pretrained group -- i.e. that essentially "
        "nothing in this ranking is statistically separable at n=7, which would itself be a notable, "
        "reportable methodological finding rather than a null result to discard.",
        "",
        "## 0. Full 13-backbone soundscape ranking (for reference)",
        "",
        "| Rank | Backbone | Domain | Mean MAE (+/- SD) |",
        "|---|---|---|---|",
    ]
    for m in check1["order"]:
        marker = (" (leading group)" if m in TIGHT_GROUP else
                  " (weaker bird-pretrained)" if m in weak_group else
                  " (VGGish/YAMNet, this analysis)" if m in TARGETS else "")
        lines.append(f"| {check1['rank'][m]} | {BACKBONE_META[m]['display']}{marker} "
                      f"| {BACKBONE_META[m]['domain']} "
                      f"| {fmt_mean_std(check1['mean_mae'][m], check1['sd_mae'][m])} |")
    lines.append("")

    lines += [
        "## 1. Identifying the \"weaker bird-pretrained\" group",
        "",
        "Bird-pretrained = domain \"Bird\" or \"Other -> Bird\" per `backbone_meta.BACKBONE_META` (trained "
        "on bird data from scratch, or fine-tuned onto it from another domain), excluding the leading "
        f"group's own bird-pretrained member (Bird-MAE, domain \"Bird\"). Within that set, the {N_WEAK} "
        f"lowest-ranked (highest-MAE) backbones by the soundscape ranking above: **{weak_labels}**. This "
        "confirms, rather than assumes, the candidates named in the task (EfficientNet-B1, Perch v1, "
        "AudioProtoPNet) -- Wav2Vec2 and AST are also bird-pretrained by this definition but rank higher "
        "(5 and 7) and are excluded from the \"weaker\" group.",
        "",
    ]

    for target in TARGETS:
        lines.append(per_target[target]["markdown"])

    lines += [
        "## Reading this",
        "",
        f"- Both {target_labels} are each tested twice: against the weaker bird-pretrained group (the claim "
        "under test) and against the leading group (for contrast, replicating/extending the existing "
        "`rq1_ranking_robustness.md` / `rq1_lower_ranks_robustness.md` results for these two backbones "
        "specifically).",
        "- A finding of \"indistinguishable from both\" for a given backbone would mean its exact position "
        "in the point-estimate ranking is not statistically load-bearing at n=7 -- consistent with the "
        "broader pattern already established across RQ1's ranking-robustness analyses that almost nothing "
        "in the full 13-backbone soundscape ordering clears a Bonferroni-corrected significance threshold.",
        "- Sign-test win/loss patterns are reported alongside the mean-difference tests because a lopsided "
        "per-dataset direction (e.g. 6-1) with a non-significant paired t-test/Wilcoxon can indicate a "
        "real but small and high-variance effect that the magnitude-based test misses -- the same "
        "rationale as `rq1_sign_test_pairs.md`.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq4"))
    args = parser.parse_args()

    models = mrt.discover_models(Path("archive/Pooled-Embeddings"))
    long_df = load_study("pooled", models=models)
    backbone_order = [m for m in BACKBONE_META if m in models]

    check1 = check1_ranking(long_df, backbone_order)
    weak_group = identify_weak_bird_pretrained(check1, backbone_order)

    full_matrix = build_matrix(long_df, SCAPE_SOURCE, SOUNDSCAPE_METRIC, REGIONAL_DATASETS, backbone_order)

    results = {
        "check1_ranking": check1,
        "weak_bird_pretrained_group": weak_group,
        "leading_group": TIGHT_GROUP,
        "targets": TARGETS,
        "per_target": {},
    }
    per_target_md = {}

    for target in TARGETS:
        weak_diff, weak_sign = compute_pairs(full_matrix, [target], weak_group)
        lead_diff, lead_sign = compute_pairs(full_matrix, [target], TIGHT_GROUP)

        results["per_target"][target] = {
            "vs_weak_bird_pretrained_diff": weak_diff,
            "vs_weak_bird_pretrained_sign": weak_sign,
            "vs_leading_group_diff": lead_diff,
            "vs_leading_group_sign": lead_sign,
        }
        per_target_md[target] = {
            "markdown": build_markdown(check1, weak_group, target, weak_diff, weak_sign, lead_diff, lead_sign)
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq4_vggish_yamnet_proximity.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {out_json}")

    md = build_full_markdown(check1, weak_group, backbone_order, per_target_md)
    out_md = args.out_dir / "rq4_vggish_yamnet_proximity.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
