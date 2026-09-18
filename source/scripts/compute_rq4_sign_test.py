#!/usr/bin/env python3
"""
RQ4 (additional) -- Exact sign tests, magnitude-independent checks on three
of RQ4's pairwise/ranking claims (real-world generalization).

Source data throughout: archive/Pooled-Embeddings/ (13 backbones, frozen
backbone + pooled-MLP head, regression formulation), same conventions as
plot_rq4.py (EASY_DATASETS, HARD_DATASETS, REGIONAL_DATASETS).

1. Frozen-backbone synthetic-vs-soundscape direction, per backbone: is the
   sign of (soundscape MAE - synthetic MAE) consistent within a backbone
   across datasets, rather than a single aggregate-mean comparison? Per-
   dataset synthetic MAE *does* exist broken out by regional subset (the 7
   REGIONAL_DATASETS all have both a synthetic_mixture_test and a
   soundscape_test result -- XCM is synthetic-only and is excluded here so
   both sides are paired on the same 7 regions), so this runs the sign test
   on that per-region pairing for each of the 13 backbones, rather than
   relying solely on one pooled synthetic number vs. one pooled soundscape
   number.

2. BEANS baseline vs. each of the six named top-synthetic-data backbones
   (Bird-MAE, Perch v2, EfficientNet-B1, NatureLM-audio, AudioProtoPNet,
   Perch v1 -- the top 6 by synthetic-mixture MAE, per
   rq1_ranking_robustness.json's check4 ranking), restricted to the two
   Hard datasets (POW, PER). Per the task's n<=3 rule, win/loss counts only
   are reported, no p-value (n=2). This directly checks the manuscript
   claim that "four of the six best backbones on synthetic data even fall
   short of the non-learned BEANS baseline on hard datasets" at the
   per-dataset level, not just via the two datasets' mean.

3. Pairwise sign test among the same six top-synthetic-data backbones,
   restricted to the three Easy datasets (HSN, NES, SSW) (15 pairs): checks
   whether the synthetic-data ranking among these six actually transfers to
   a consistent per-dataset ordering on Easy data, operationalizing the
   manuscript's claim that "synthetic performance does partly predict
   performance on easy datasets" at the pairwise level. n=3, so per the
   task's rule, win/loss counts only, no p-value -- explicitly flagged as
   too small for a meaningful test.

Writes rq4_sign_test.json and rq4_sign_test.md to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq4_sign_test.py [--out-dir plots/figures/rq4]
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
from plot_rq4 import EASY_DATASETS, HARD_DATASETS

SOURCE = "synthetic_mixture_test"
SCAPE_SOURCE = "soundscape_test"

# Top 6 by synthetic-mixture MAE, per rq1_ranking_robustness.json's check4
# ranking (Bird-MAE 0.607, Perch v2 0.682, EfficientNet-B1 0.695,
# NatureLM-audio 0.699, AudioProtoPNet 0.720, Perch v1 0.756).
SIX_TOP_SYNTHETIC = ["Bird-MAE-Huge", "perch_v2_cpu", "EfficientNet-B1-BirdSet-XCL", "NatureLMBEATs",
                     "AudioProtoPNet-20-BirdSet-XCL", "perch_8"]
BEANS = "BeansBaseline"


def sign_test(diff: np.ndarray, n_datasets: int) -> dict:
    """Exact sign test: wins = A better than B (diff = A - B < 0, lower
    MAE), losses = diff > 0, ties = diff == 0 (dropped before the test).
    p-value omitted when n_datasets<=3, per the task's small-n rule."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    wins = int((diff < 0).sum())
    losses = int((diff > 0).sum())
    ties = int(len(diff) - wins - losses)
    n_decisive = wins + losses
    if n_datasets <= 3 or n_decisive == 0:
        p_two_sided = None
    else:
        p_two_sided = round(float(scipy.stats.binomtest(wins, n_decisive, 0.5, alternative="two-sided").pvalue), 6)
    return {"n_datasets": n_datasets, "wins": wins, "losses": losses, "ties": ties,
            "n_decisive": n_decisive, "p_two_sided": p_two_sided}


def build_matrix(long_df: pd.DataFrame, source: str, metric: str, dataset_order: list, row_order: list) -> pd.DataFrame:
    sub = filter_long(long_df, source=source, metric=metric, head="reg", dataset=dataset_order)
    matrix = sub.pivot_table(index="model", columns="dataset", values="value")
    return matrix.reindex(index=row_order, columns=dataset_order)


def _fmt_p(x):
    return f"{x:.4f}" if x is not None else "n/a, n≤3"


# ---------------------------------------------------------------------------
# Check 1 -- frozen synthetic-vs-soundscape direction, per backbone.
# ---------------------------------------------------------------------------

def check1_synthetic_vs_soundscape(long_df: pd.DataFrame, backbone_order: list) -> dict:
    synth = build_matrix(long_df, SOURCE, "mae", REGIONAL_DATASETS, backbone_order)
    scape = build_matrix(long_df, SCAPE_SOURCE, "range_mae", REGIONAL_DATASETS, backbone_order)
    results = {}
    for model in backbone_order:
        diff = (scape.loc[model] - synth.loc[model]).to_numpy(dtype=float)
        results[model] = sign_test(diff, len(REGIONAL_DATASETS))
    return results


# ---------------------------------------------------------------------------
# Check 2 -- BEANS vs. six top-synthetic backbones, Hard datasets.
# ---------------------------------------------------------------------------

def check2_beans_vs_six_hard(long_df: pd.DataFrame) -> dict:
    matrix = build_matrix(long_df, SCAPE_SOURCE, "range_mae", HARD_DATASETS, SIX_TOP_SYNTHETIC + [BEANS])
    results = {}
    for model in SIX_TOP_SYNTHETIC:
        diff = (matrix.loc[model] - matrix.loc[BEANS]).to_numpy(dtype=float)
        results[model] = sign_test(diff, len(HARD_DATASETS))
    return results


# ---------------------------------------------------------------------------
# Check 3 -- pairwise among six top-synthetic backbones, Easy datasets.
# ---------------------------------------------------------------------------

def check3_pairwise_easy(long_df: pd.DataFrame) -> dict:
    matrix = build_matrix(long_df, SCAPE_SOURCE, "range_mae", EASY_DATASETS, SIX_TOP_SYNTHETIC)
    results = {}
    for a, b in itertools.combinations(SIX_TOP_SYNTHETIC, 2):
        diff = (matrix.loc[a] - matrix.loc[b]).to_numpy(dtype=float)
        results[f"{a}__minus__{b}"] = {"a": a, "b": b, **sign_test(diff, len(EASY_DATASETS))}
    return results


def build_markdown(check1: dict, check2: dict, check3: dict, backbone_order: list) -> str:
    six_labels = ", ".join(BACKBONE_META[m]["display"] for m in SIX_TOP_SYNTHETIC)
    lines = [
        "# RQ4 -- Exact sign tests, real-world generalization",
        "",
        "Three magnitude-independent checks on RQ4's pairwise/ranking claims, complementing (not "
        "replacing) the mean-difference analyses elsewhere in RQ4. Source: `archive/Pooled-Embeddings/`, "
        "regression head throughout.",
        "",
        "## 1. Frozen-backbone synthetic-vs-soundscape direction, per backbone",
        "",
        f"Sign of (soundscape range_mae - synthetic mae), paired by the 7 REGIONAL_DATASETS (XCM excluded "
        "-- synthetic-only, no soundscape counterpart), per backbone. A consistently positive sign "
        "(all/most losses) means soundscape is uniformly harder than synthetic for that backbone across "
        "every region, not just on aggregate.",
        "",
        "| Backbone | n datasets | Soundscape-worse (losses) | Soundscape-better (wins) | Ties "
        "| Sign-test $p$ |",
        "|---|---|---|---|---|---|",
    ]
    for model in backbone_order:
        r = check1[model]
        lines.append(f"| {BACKBONE_META[model]['display']} | {r['n_datasets']} | {r['losses']} | {r['wins']} "
                     f"| {r['ties']} | {_fmt_p(r['p_two_sided'])} |")

    lines += [
        "",
        "## 2. BEANS baseline vs. six top-synthetic-data backbones, Hard datasets (POW, PER)",
        "",
        f"The six top-synthetic-data backbones: {six_labels}. n=2 (POW, PER) -- per the task's rule, "
        "win/loss counts only are reported, no p-value; too small for a meaningful sign test. Checks the "
        "manuscript claim (\"four of the six best backbones on synthetic data even fall short of the "
        "non-learned BEANS baseline on hard datasets\") at the per-dataset level rather than via the "
        "two datasets' mean.",
        "",
        "| Backbone | n datasets | Wins vs. BEANS | Losses vs. BEANS | Ties | Sign-test $p$ |",
        "|---|---|---|---|---|---|",
    ]
    for model in SIX_TOP_SYNTHETIC:
        r = check2[model]
        lines.append(f"| {BACKBONE_META[model]['display']} | {r['n_datasets']} | {r['wins']} | {r['losses']} "
                     f"| {r['ties']} | {_fmt_p(r['p_two_sided'])} |")

    lines += [
        "",
        "## 3. Pairwise ranking among six top-synthetic-data backbones, Easy datasets (HSN, NES, SSW)",
        "",
        "n=3 -- per the task's rule, win/loss counts only, no p-value; flagged explicitly as too small "
        "for a meaningful test. Checks whether the synthetic-data ranking among these six backbones "
        "transfers to a consistent per-dataset ordering on Easy soundscape data (the pairwise "
        "operationalization of \"synthetic performance does partly predict performance on easy "
        "datasets\").",
        "",
        "| Pair (A - B) | n datasets | Wins | Losses | Ties | Sign-test $p$ |",
        "|---|---|---|---|---|---|",
    ]
    for s in check3.values():
        a_label, b_label = BACKBONE_META[s["a"]]["display"], BACKBONE_META[s["b"]]["display"]
        lines.append(f"| {a_label} - {b_label} | {s['n_datasets']} | {s['wins']} | {s['losses']} "
                     f"| {s['ties']} | {_fmt_p(s['p_two_sided'])} |")

    lines += [
        "",
        "## Reading this",
        "",
        "- Check 1 replaces a single aggregate synthetic-vs-soundscape comparison per backbone with a "
        "7-dataset paired sign test -- a backbone with 6-7 losses is uniformly worse on soundscape, not "
        "just on average.",
        "- Check 2's win/loss counts (n=2, no p-value) should be read alongside the mean-based claim in "
        "the manuscript text: a backbone that wins on one Hard dataset and loses on the other is a "
        "genuinely mixed result the aggregate mean can obscure.",
        "- Check 3's counts (n=3, no p-value) are reported for pattern visibility only, per the task's "
        "small-n caveat -- not evidence of a real effect on their own.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq4"))
    args = parser.parse_args()

    models = mrt.discover_models(Path("archive/Pooled-Embeddings"))
    long_df = load_study("pooled", models=models)
    backbone_order = [m for m in BACKBONE_META if m in models]

    check1 = check1_synthetic_vs_soundscape(long_df, backbone_order)
    check2 = check2_beans_vs_six_hard(long_df)
    check3 = check3_pairwise_easy(long_df)

    results = {"check1_synthetic_vs_soundscape": check1, "check2_beans_vs_six_hard": check2,
               "check3_pairwise_six_easy": check3, "six_top_synthetic": SIX_TOP_SYNTHETIC}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq4_sign_test.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")

    md = build_markdown(check1, check2, check3, backbone_order)
    out_md = args.out_dir / "rq4_sign_test.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
