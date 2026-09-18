#!/usr/bin/env python3
"""
RQ1 (additional) -- Exact sign test, regression vs. classification
formulation, on the Pooled-Embeddings experiments (13 backbones, frozen
backbone + pooled-embedding head).

For each of the 3 metrics per test type (synthetic: mae, off_by_one_accuracy,
accuracy; soundscape: range_mae, off_by_one_range_accuracy, range_accuracy --
soundscape has no exact ground truth, so its "accuracy" metrics are the
range-based ones, same convention as elsewhere in RQ1/RQ4/RQ5), each
backbone contributes one paired comparison: its own mean metric value across
datasets (all 8 for synthetic; the 7 region-matched ones for soundscape,
which has no XCM counterpart) under the "reg" head vs. the "class" head.
"Win" means regression is better on that metric's own direction (lower for
MAE, higher for accuracy/off-by-one). Each backbone's mean is rounded to
ROUND_DECIMALS (2) places -- matching the precision results are actually
reported at -- before comparing, so a difference that would print as 0.00
at that precision counts as a tie rather than a coin-flip "win" from
floating-point noise (e.g. 0.4795 vs. 0.4793 is a tie, not a win). Ties are
then dropped before the test, per standard sign-test convention -- n = wins
+ losses.

The sign test itself is the exact two-sided binomial test (wins vs. losses
against p=0.5), via scipy.stats.binomtest -- this is the standard "sign
test" convention. The one-sided p-value (testing specifically "regression
wins more often than chance") is also reported alongside it, since a
one-sided test is also a defensible choice when the comparison direction is
hypothesized in advance (as it plausibly is for MAE/off-by-one, less so for
exact-match accuracy) -- both numbers are in the output so the correct one
for a given claim can be picked deliberately, not by whichever matches a
quoted number.

Writes plots/figures/rq1/rq1_sign_test.json (wins/losses/ties/p-values per
metric), rq1_sign_test_table.tex (summary), and rq1_sign_test_per_backbone.tex
(the per-backbone reg-vs-class values behind the test, for the Supplementary).

Usage:
    complete-venv/bin/python source/scripts/compute_rq1_sign_test.py [--out-dir plots/figures/rq1]
"""

import argparse
import json
import sys
from pathlib import Path

import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import BACKBONE_META
from plot_data import ALL_DATASETS, REGIONAL_DATASETS, filter_long, load_study

# (metric, direction) -- direction: "lower" means regression wins if its
# mean is lower than classification's; "higher" the opposite.
SYNTHETIC_METRICS = [
    ("mae", "lower", "MAE"),
    ("off_by_one_accuracy", "higher", "Off-by-one accuracy"),
    ("accuracy", "higher", "Exact-match accuracy"),
]
SOUNDSCAPE_METRICS = [
    ("range_mae", "lower", "MAE"),
    ("off_by_one_range_accuracy", "higher", "Off-by-one accuracy"),
    ("range_accuracy", "higher", "Exact-match accuracy"),
]
TEST_TYPES = {
    "synthetic": ("synthetic_mixture_test", ALL_DATASETS, SYNTHETIC_METRICS),
    "soundscape": ("soundscape_test", REGIONAL_DATASETS, SOUNDSCAPE_METRICS),
}
ROUND_DECIMALS = 2


def per_backbone_means(long_df, source: str, metric: str, dataset: list, models: list):
    """One row per backbone x head: mean metric value across `dataset`,
    rounded to ROUND_DECIMALS -- see sign_test()."""
    sub = filter_long(long_df, source=source, metric=metric, head=["reg", "class"], dataset=dataset, model=models)
    return sub.groupby(["model", "head"])["value"].mean().unstack("head").round(ROUND_DECIMALS)


def sign_test(reg: "pd.Series", cls: "pd.Series", direction: str) -> dict:
    """Paired sign test: reg vs. cls, one pair per backbone (both already
    rounded to ROUND_DECIMALS by per_backbone_means()). direction
    "lower"/"higher" says which side counts as a regression win."""
    diff = reg - cls
    if direction == "lower":
        wins = int((diff < 0).sum())
        losses = int((diff > 0).sum())
    else:
        wins = int((diff > 0).sum())
        losses = int((diff < 0).sum())
    ties = int(len(diff) - wins - losses)
    n = wins + losses
    two_sided = scipy.stats.binomtest(wins, n, 0.5, alternative="two-sided").pvalue if n > 0 else float("nan")
    one_sided = scipy.stats.binomtest(wins, n, 0.5, alternative="greater").pvalue if n > 0 else float("nan")
    return {
        "wins": wins, "losses": losses, "ties": ties, "n_decisive": n,
        "p_two_sided": round(float(two_sided), 6), "p_one_sided_greater": round(float(one_sided), 6),
    }


def build_summary_table(results: dict) -> str:
    header = r"Test type & Metric & Wins & Losses & Ties & $n$ & $p$ (two-sided) & $p$ (one-sided) \\"
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Exact sign test, regression vs. classification formulation, Pooled-Embeddings "
        r"(13 backbones). Wins/losses/ties are per-backbone comparisons of that backbone's own mean "
        r"metric value (across datasets) between the two heads; ties are dropped before the test.}",
        r"\label{tab:rq1_sign_test}",
        r"\begin{tabular}{llcccccc}", r"\toprule", header, r"\midrule",
    ]
    for test_key, (_, _, metrics) in TEST_TYPES.items():
        for i, (metric_key, _, metric_label) in enumerate(metrics):
            r = results[test_key][metric_key]
            test_cell = mrt.escape_latex(test_key.capitalize()) if i == 0 else ""
            lines.append(
                f"{test_cell} & {mrt.escape_latex(metric_label)} & {r['wins']} & {r['losses']} & {r['ties']} & "
                f"{r['n_decisive']} & {r['p_two_sided']:.4f} & {r['p_one_sided_greater']:.4f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_per_backbone_table(per_backbone: dict, backbone_order: list) -> str:
    """Supplementary table: each backbone's own reg/class mean value for
    every metric x test-type combination."""
    cols = [(t, m, label) for t, (_, _, metrics) in TEST_TYPES.items() for (m, _, label) in metrics]
    header = "Backbone & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{mrt.escape_latex(t.capitalize())} {mrt.escape_latex(label)}}}" for t, _, label in cols
    ) + r" \\"
    subheader = " & " + " & ".join(r"Reg & Class" for _ in cols) + r" \\"
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Per-backbone mean metric values (regression vs. classification head), "
        r"Pooled-Embeddings -- the values the sign test in Table~\ref{tab:rq1_sign_test} is computed from.}",
        r"\label{tab:rq1_sign_test_per_backbone}",
        rf"\begin{{tabular}}{{l{'cc' * len(cols)}}}", r"\toprule", header, subheader, r"\midrule",
    ]
    for model in backbone_order:
        row = [mrt.escape_latex(BACKBONE_META[model]["display"])]
        for t, m, _ in cols:
            vals = per_backbone[t][m]
            reg = vals.loc[model, "reg"] if model in vals.index else float("nan")
            cls = vals.loc[model, "class"] if model in vals.index else float("nan")
            row.append(mrt.fmt(reg, 3))
            row.append(mrt.fmt(cls, 3))
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq1"))
    args = parser.parse_args()

    models = mrt.discover_models(Path("archive/Pooled-Embeddings"))
    long_df = load_study("pooled", models=models)

    results = {}
    per_backbone = {}
    for test_key, (source, dataset, metrics) in TEST_TYPES.items():
        results[test_key] = {}
        per_backbone[test_key] = {}
        for metric_key, direction, _ in metrics:
            means = per_backbone_means(long_df, source, metric_key, dataset, models)
            per_backbone[test_key][metric_key] = means
            results[test_key][metric_key] = sign_test(means["reg"], means["class"], direction)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq1_sign_test.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")
    print(json.dumps(results, indent=2))

    backbone_order = [m for m in BACKBONE_META if m in models]
    (args.out_dir / "rq1_sign_test_table.tex").write_text(build_summary_table(results))
    print(f"Wrote {args.out_dir / 'rq1_sign_test_table.tex'}")
    (args.out_dir / "rq1_sign_test_per_backbone.tex").write_text(build_per_backbone_table(per_backbone, backbone_order))
    print(f"Wrote {args.out_dir / 'rq1_sign_test_per_backbone.tex'}")


if __name__ == "__main__":
    main()
