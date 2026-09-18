#!/usr/bin/env python3
"""
RQ1 (additional) -- Exact sign test, magnitude-independent complement to
compute_rq1_ranking_robustness.py's checks 2 and 3 (paired t-test/Wilcoxon
on the mean per-dataset delta).

Purpose: the mean-difference tests in checks 2/3 can be inflated or
obscured by a small number of high-variance datasets (PER in particular is
the intrinsically hardest/most polyphony-dense regional dataset throughout
RQ1/RQ4/RQ5). This checks the same pairs a different way -- does backbone A
actually win on most/all of the 7 individual soundscape datasets, independent
of how large those per-dataset wins/losses are.

Same two pair sets as compute_rq1_ranking_robustness.py, same source data
(archive/Pooled-Embeddings/, soundscape_test, range_mae, 7 REGIONAL_DATASETS,
regression head):

1. Within the "tight leading group" {Bird-MAE, Perch v2, BirdNET v2.3,
   BirdNET v2.4} (6 pairs).
2. Leading-group-vs-next-ranked (4 x 3 = 12 pairs), against whichever 3
   backbones sit at ranks 5-7 (Wav2Vec2, VGGish, AST per the current
   ranking -- taken from check1's own order, not hardcoded, so this stays
   correct if the ranking ever shifts).

For each pair (A, B): sign of (A's MAE - B's MAE) per dataset -- win for A
if negative, tie if exactly zero (raw values, unrounded, unlike
compute_rq1_sign_test.py's backbone-mean comparison, since here each
"observation" is already a single raw cell, not a mean that needs rounding
to avoid floating-point-noise wins). Ties dropped before the test. Exact
two-sided binomial sign test against p=0.5 -- with n=7 datasets throughout,
comfortably above the n=4-5 floor below which a sign test has negligible
power, but still noted as small-n/exploratory per the ranking_robustness
caveat.

Writes rq1_sign_test_pairs.json and rq1_sign_test_pairs.md to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq1_sign_test_pairs.py [--out-dir plots/figures/rq1]
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
    N_NEXT, REGIONAL_DATASETS, SCAPE_SOURCE, SOUNDSCAPE_METRIC, TIGHT_GROUP,
    build_matrix, check1_ranking,
)
from plot_data import load_study

N_DATASETS = len(REGIONAL_DATASETS)


def sign_test(diff: np.ndarray) -> dict:
    """Exact sign test: wins = A better than B (diff = A - B < 0, lower
    MAE), losses = diff > 0, ties = diff == 0 (dropped before the test).
    Per the task's n<=3 rule, p-values are omitted when the dataset subset
    itself (N_DATASETS, not n_decisive) is that small -- not applicable
    here (N_DATASETS=7 throughout), but the check is kept for reuse."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    wins = int((diff < 0).sum())
    losses = int((diff > 0).sum())
    ties = int(len(diff) - wins - losses)
    n_decisive = wins + losses
    if N_DATASETS <= 3 or n_decisive == 0:
        p_two_sided = None
    else:
        p_two_sided = round(float(scipy.stats.binomtest(wins, n_decisive, 0.5, alternative="two-sided").pvalue), 6)
    return {"n_datasets": N_DATASETS, "wins": wins, "losses": losses, "ties": ties,
            "n_decisive": n_decisive, "p_two_sided": p_two_sided}


def pairwise_sign_test(matrix: pd.DataFrame, a: str, b: str) -> dict:
    diff = (matrix.loc[a] - matrix.loc[b]).to_numpy(dtype=float)
    return {"a": a, "b": b, **sign_test(diff)}


def _fmt_p(x):
    return f"{x:.4f}" if x is not None else "n/a, n≤3"


def build_table(pairs: dict, label_lookup: dict) -> list:
    lines = [
        "| Pair (A - B) | n datasets | Wins | Losses | Ties | Sign-test $p$ |",
        "|---|---|---|---|---|---|",
    ]
    for s in pairs.values():
        a_label, b_label = label_lookup[s["a"]], label_lookup[s["b"]]
        lines.append(f"| {a_label} - {b_label} | {s['n_datasets']} | {s['wins']} | {s['losses']} "
                     f"| {s['ties']} | {_fmt_p(s['p_two_sided'])} |")
    return lines


def build_markdown(within_group: dict, vs_next: dict, next_group: list) -> str:
    tight_labels = ", ".join(BACKBONE_META[m]["display"] for m in TIGHT_GROUP)
    next_labels = ", ".join(BACKBONE_META[m]["display"] for m in next_group)
    label_lookup = {m: BACKBONE_META[m]["display"] for m in BACKBONE_META}

    lines = [
        "# RQ1 -- Exact sign test, tight leading group and leading-group-vs-next-ranked pairs",
        "",
        "Magnitude-independent complement to `rq1_ranking_robustness.md` checks 2 and 3 (paired "
        "t-test/Wilcoxon on the mean per-dataset delta): for each pair, does backbone A win on most/all "
        f"of the {N_DATASETS} individual soundscape datasets, independent of how large those per-dataset "
        "wins/losses are -- complements, does not replace, the mean-difference tests. Source: "
        "`archive/Pooled-Embeddings/`, `soundscape_test` / range_mae, regression head, the 7 "
        "REGIONAL_DATASETS.",
        "",
        f"**Caveat**: n={N_DATASETS} datasets is small for a sign test (though above the n=4-5 floor "
        "below which it has negligible power) -- treat as exploratory/directional, consistent with the "
        "same caveat in `rq1_ranking_robustness.md`. Where a sign-test result here diverges from the "
        "corresponding paired t-test/Wilcoxon result in `rq1_ranking_robustness.md` (e.g. a consistent "
        "per-dataset direction alongside a non-significant mean-difference test), that combination is "
        "flagged explicitly below -- it would indicate a real, if small, directional effect that the "
        "magnitude-based test missed due to high per-dataset variance.",
        "",
        f"## 1. Within the tight leading group ({tight_labels})",
        "",
    ]
    lines += build_table(within_group, label_lookup)
    lines += [
        "",
        f"## 2. Leading group vs. next-ranked ({next_labels})",
        "",
    ]
    lines += build_table(vs_next, label_lookup)
    lines += [
        "",
        "## Reading this",
        "",
        "- A pair with a significant sign-test p-value (win/loss lopsided, e.g. 6-1 or 7-0) but a "
        "non-significant paired t-test/Wilcoxon in `rq1_ranking_robustness.md` would indicate a "
        "consistent-but-small per-dataset direction that the mean-difference test's sensitivity to "
        "high-variance datasets (PER especially) obscured.",
        "- Conversely, a pair with a roughly even win/loss split (e.g. 4-3) confirms the "
        "mean-difference test's non-significance is not masking a hidden consistent direction.",
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
    next_group = [m for m in check1["order"] if m not in TIGHT_GROUP][:N_NEXT]

    within_matrix = build_matrix(long_df, SCAPE_SOURCE, SOUNDSCAPE_METRIC, REGIONAL_DATASETS, TIGHT_GROUP)
    within_group = {f"{a}__minus__{b}": pairwise_sign_test(within_matrix, a, b)
                    for a, b in itertools.combinations(TIGHT_GROUP, 2)}

    vs_next_matrix = build_matrix(long_df, SCAPE_SOURCE, SOUNDSCAPE_METRIC, REGIONAL_DATASETS,
                                   TIGHT_GROUP + next_group)
    vs_next = {f"{a}__minus__{b}": pairwise_sign_test(vs_next_matrix, a, b)
               for a in TIGHT_GROUP for b in next_group}

    results = {"within_group": within_group, "vs_next": vs_next, "next_group": next_group}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq1_sign_test_pairs.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")

    md = build_markdown(within_group, vs_next, next_group)
    out_md = args.out_dir / "rq1_sign_test_pairs.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
