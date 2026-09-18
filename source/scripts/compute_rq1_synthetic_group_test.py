#!/usr/bin/env python3
"""
RQ1 -- Corrected group-level test, bird-specific vs. general-audio
pretraining (synthetic data).

`compute_rq1_synthetic_ranking_robustness.py`'s check 5 pools all 6 x 6 = 36
cross-group backbone pairs x 8 datasets = 288 cells into one test. That
inflates n: the same per-dataset MAE value for a given backbone recurs in
6 of those 36 pairs (once per opposing-group backbone it's paired against),
so the 288 "cells" are not 288 independent observations -- the reported
significance (mean Delta -0.062 +/- 0.218, sign test 171-117, p=0.0017) is
not a formally valid p-value. This script replaces that pooled-pairwise
construction with two properly two-sample (not pairwise-difference) tests
that avoid the repeated-cross-pairing problem, at two different points on
the independence/power trade-off:

Option A -- one independent observation per backbone: each of the 12
  backbones' mean MAE across the 8 synthetic datasets (reusing check 1's
  per-backbone means from `compute_rq1_synthetic_ranking_robustness.py`
  directly) gives a genuinely independent n=6 vs. n=6 sample (each backbone
  contributes exactly one number). Two-sample Mann-Whitney U and Welch's
  t-test between the two groups' 6 backbone means. Most statistically
  clean, but very low power at n=6 per group.

Option B -- pooled per-dataset cells, no pairwise repetition: each group's
  6 backbones x 8 datasets = 48 raw per-dataset MAE values, each cell used
  exactly once (unlike check 5's construction, no cell is repeated across
  multiple cross-group pairs). Two-sample Mann-Whitney U and Welch's t-test
  between the two groups' 48 values. This still has a mild non-independence
  caveat -- a given backbone's own 8 datasets are not independent of each
  other (same backbone, shared per-backbone effects) -- but that is a much
  less severe violation than check 5's repeated cross-pairing, where the
  same underlying values are reused across many pairs entirely. More power
  than Option A, at that reduced (not eliminated) independence cost.

Both options are repeated with AST excluded from the bird-specific group
(n=5 for that group) as a robustness check, since the manuscript flags AST
as behaving more like general-audio despite its nominal bird-specific
classification (`compute_rq1_synthetic_ranking_robustness.py` check 6) --
this checks whether the bird-specific vs. general-audio group difference is
partly carried by AST's atypical position or holds independently of it.

Same Table3-groups coding, data source, and BEANS-baseline exclusion as
`compute_rq1_synthetic_ranking_robustness.py` check 5 (and
`compute_rq4_pretraining_domain_correlation.py`'s table3_groups coding):
bird-specific = {Bird-MAE, AudioProtoPNet, EfficientNet-B1, Perch v1, AST,
Wav2Vec2} (n=6); general-audio-pretrained = {VGGish, YAMNet, NatureLM-audio,
BirdNET v2.3, BirdNET v2.4, Perch v2} (n=6); BEANS baseline excluded.

Requires `rq1_synthetic_ranking_robustness.json` (for the original 288-cell
result, quoted for direct comparison) -- run
`compute_rq1_synthetic_ranking_robustness.py` first.

Writes rq1_synthetic_group_test.json and rq1_synthetic_group_test.md to
--out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq1_synthetic_group_test.py [--out-dir plots/figures/rq1]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import BACKBONE_META
from compute_rq1_ranking_robustness import ALL_DATASETS, SOURCE, SYNTHETIC_METRIC, build_matrix
from compute_rq4_pretraining_domain_correlation import (
    TABLE3_BIRD_ONLY, TABLE3_GENERAL_AUDIO,
)
from plot_data import load_study

AST = "AST-Birdset-XCL"


def fmt_mean_std(mean, std, decimals=3):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def two_sample_test(group_a: np.ndarray, group_b: np.ndarray) -> dict:
    """Mann-Whitney U and Welch's t-test (unequal variances, same convention
    as compute_rq4_pretraining_domain_correlation.py) between two
    independent-ish samples -- no pairwise differencing, no repeated
    cross-pairing."""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    mw = scipy.stats.mannwhitneyu(a, b, alternative="two-sided")
    tt = scipy.stats.ttest_ind(a, b, equal_var=False)
    return {
        "n_a": len(a), "n_b": len(b),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "std_a": float(a.std(ddof=1)) if len(a) > 1 else None,
        "std_b": float(b.std(ddof=1)) if len(b) > 1 else None,
        "mannwhitney_u": float(mw.statistic), "mannwhitney_p": float(mw.pvalue),
        "ttest_t": float(tt.statistic), "ttest_p": float(tt.pvalue),
    }


# ---------------------------------------------------------------------------
# Option A -- one independent observation per backbone (n=6 vs. n=6 means).
# ---------------------------------------------------------------------------

def option_a(matrix, bird_group: list, audio_group: list) -> dict:
    bird_means = matrix.loc[bird_group].mean(axis=1).to_numpy(dtype=float)
    audio_means = matrix.loc[audio_group].mean(axis=1).to_numpy(dtype=float)
    return two_sample_test(bird_means, audio_means)


# ---------------------------------------------------------------------------
# Option B -- pooled per-dataset cells, each cell used exactly once
# (n=48 vs. n=48, no cross-pair repetition).
# ---------------------------------------------------------------------------

def option_b(matrix, bird_group: list, audio_group: list) -> dict:
    bird_cells = matrix.loc[bird_group].to_numpy(dtype=float).ravel()
    audio_cells = matrix.loc[audio_group].to_numpy(dtype=float).ravel()
    bird_cells = bird_cells[~np.isnan(bird_cells)]
    audio_cells = audio_cells[~np.isnan(audio_cells)]
    return two_sample_test(bird_cells, audio_cells)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _fmt_p(x):
    return f"p={x:.4f}" if x is not None else "--"


def _row(label: str, n_a, n_b, mean_a, mean_b, median_a=None, median_b=None,
         stat_label="", stat=None, mw_p=None, t_p=None) -> str:
    n = f"{n_a} vs. {n_b}"
    means = f"{mean_a:.3f} vs. {mean_b:.3f}"
    medians = f"{median_a:.3f} vs. {median_b:.3f}" if median_a is not None else "--"
    stat_str = f"{stat:.1f}" if stat is not None else "--"
    return (f"| {label} | {n} | {means} | {medians} | {stat_str} | {_fmt_p(mw_p)} | {_fmt_p(t_p)} |")


def build_markdown(original: dict, a_full: dict, b_full: dict, a_noast: dict, b_noast: dict,
                    bird_group: list, audio_group: list) -> str:
    bird_labels = ", ".join(BACKBONE_META[m]["display"] for m in bird_group)
    audio_labels = ", ".join(BACKBONE_META[m]["display"] for m in audio_group)
    bird_noast_labels = ", ".join(BACKBONE_META[m]["display"] for m in bird_group if m != AST)

    lines = [
        "# RQ1 -- Corrected group-level test, bird-specific vs. general-audio pretraining (synthetic data)",
        "",
        "`rq1_synthetic_ranking_robustness.md` check 5 pools all 6 x 6 = 36 cross-group backbone pairs "
        "x 8 datasets = 288 cells into one sign test. Because the same per-dataset MAE value for a given "
        "backbone recurs across all 6 pairs it appears in, those 288 cells are not 288 independent "
        "observations, and the resulting p-value is not formally valid. This file replaces that pooled-"
        "pairwise construction with two genuinely two-sample (not pairwise-difference) tests, at two "
        "points on the independence/power trade-off, plus an AST-excluded robustness check on each. "
        "Source: `archive/Pooled-Embeddings/` (13 backbones, frozen backbone + pooled-MLP head, "
        "regression formulation), `synthetic_mixture_test` / mae, the 8 ALL_DATASETS -- same as "
        "`rq1_synthetic_ranking_robustness.md`.",
        "",
        f"Table3-groups coding (same as `rq4_pretraining_domain_correlation.py` / "
        f"`rq1_synthetic_ranking_robustness.md` check 5): bird-specific = {{{bird_labels}}} (n=6); "
        f"general-audio-pretrained = {{{audio_labels}}} (n=6); BEANS baseline excluded (fits neither "
        "group).",
        "",
        "**Option A** (one independent observation per backbone): each of the 12 backbones' mean MAE "
        "across the 8 synthetic datasets (reusing `rq1_synthetic_ranking_robustness.md` check 1's "
        "per-backbone means) gives a genuinely independent n=6 vs. n=6 sample -- the most statistically "
        "clean of the two, but very low power at n=6 per group.",
        "",
        "**Option B** (pooled per-dataset cells, no pairwise repetition): each group's 6 backbones x 8 "
        "datasets = 48 raw per-dataset MAE values, each cell used exactly once (unlike check 5, no cell "
        "is repeated across multiple cross-group pairs). More power than Option A, at a reduced -- not "
        "eliminated -- independence cost: a given backbone's own 8 datasets are still not independent of "
        "each other (shared per-backbone effects), which is a *milder* violation than check 5's repeated "
        "cross-pairing, but a violation nonetheless. Read Option B's significance accordingly.",
        "",
        "## Results",
        "",
        "| Comparison | n (bird vs. audio) | Mean MAE (bird vs. audio) | Median MAE (bird vs. audio) "
        "| Mann-Whitney U | Mann-Whitney p | t-test p |",
        "|---|---|---|---|---|---|---|",
        f"| Original 288-cell pooled (sign test, not a two-sample test) | 288 cells (36 pairs x 8) "
        f"| Delta {original['mean']:.3f} ± {original['std']:.3f} | -- | -- | -- "
        f"| paired t: {_fmt_p(original['t_p'])} |",
        _row("Option A: 6 vs. 6 backbone means", a_full["n_a"], a_full["n_b"], a_full["mean_a"],
             a_full["mean_b"], a_full["median_a"], a_full["median_b"], stat=a_full["mannwhitney_u"],
             mw_p=a_full["mannwhitney_p"], t_p=a_full["ttest_p"]),
        _row("Option B: 48 vs. 48 per-dataset cells", b_full["n_a"], b_full["n_b"], b_full["mean_a"],
             b_full["mean_b"], b_full["median_a"], b_full["median_b"], stat=b_full["mannwhitney_u"],
             mw_p=b_full["mannwhitney_p"], t_p=b_full["ttest_p"]),
        _row("Option A, AST excluded (5 vs. 6 backbone means)", a_noast["n_a"], a_noast["n_b"],
             a_noast["mean_a"], a_noast["mean_b"], a_noast["median_a"], a_noast["median_b"],
             stat=a_noast["mannwhitney_u"], mw_p=a_noast["mannwhitney_p"], t_p=a_noast["ttest_p"]),
        _row("Option B, AST excluded (40 vs. 48 per-dataset cells)", b_noast["n_a"], b_noast["n_b"],
             b_noast["mean_a"], b_noast["mean_b"], b_noast["median_a"], b_noast["median_b"],
             stat=b_noast["mannwhitney_u"], mw_p=b_noast["mannwhitney_p"], t_p=b_noast["ttest_p"]),
        "",
        f"Original row quoted from `rq1_synthetic_ranking_robustness.json` check 5 for direct comparison "
        f"(n=288, sign test {original['wins']}-{original['losses']}-{original['ties']}, "
        f"p={original['p_sign']:.4f}) -- its columns don't line up one-to-one with Options A/B (it's a "
        "paired-difference/sign test on 288 non-independent cells, not a two-sample test on independent "
        "groups), so treat it as a magnitude/direction reference point, not an apples-to-apples row.",
        "",
        f"AST-excluded groups drop AST from the bird-specific side entirely: bird-specific (AST excluded) "
        f"= {{{bird_noast_labels}}} (n=5); general-audio unchanged (n=6).",
        "",
        "## Reading this",
        "",
        "- Option A is the most statistically clean (fully independent n=6 per group) but has very "
        "limited power given the small group size -- a non-significant Option A result does not mean "
        "there is no group-level effect, only that n=6 per group can't detect one reliably.",
        "- Option B trades some independence purity for more statistical power via a less severe "
        "violation than the original 288-cell construction -- report both A and B together so the "
        "trade-off is visible, rather than citing either alone.",
        "- If Option A and Option B agree in direction and (approximate) significance, that's reasonably "
        "strong evidence the original check 5 result wasn't purely an artifact of its inflated n=288. If "
        "they disagree -- e.g. B significant but A not -- that's consistent with the original result "
        "being real but underpowered at the true n=6 group size, which the 288-cell pooling was masking.",
        "- If the AST-excluded rows differ substantially from the full-group rows (e.g. full-group "
        "significant, AST-excluded not), that indicates the bird-specific vs. general-audio group "
        "difference is at least partly carried by AST's atypical (more general-audio-like) position "
        "rather than holding uniformly across the other 5 bird-specific backbones.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq1"))
    args = parser.parse_args()

    original_json = args.out_dir / "rq1_synthetic_ranking_robustness.json"
    if not original_json.is_file():
        raise SystemExit(f"{original_json} not found -- run compute_rq1_synthetic_ranking_robustness.py first.")
    original = json.loads(original_json.read_text())["check5"]

    models = mrt.discover_models(Path("archive/Pooled-Embeddings"))
    long_df = load_study("pooled", models=models)

    bird_group = TABLE3_BIRD_ONLY
    audio_group = TABLE3_GENERAL_AUDIO
    bird_group_noast = [m for m in bird_group if m != AST]

    matrix = build_matrix(long_df, SOURCE, SYNTHETIC_METRIC, ALL_DATASETS, bird_group + audio_group)

    a_full = option_a(matrix, bird_group, audio_group)
    b_full = option_b(matrix, bird_group, audio_group)
    a_noast = option_a(matrix, bird_group_noast, audio_group)
    b_noast = option_b(matrix, bird_group_noast, audio_group)

    results = {
        "original_288_cell": original,
        "option_a_full": a_full,
        "option_b_full": b_full,
        "option_a_ast_excluded": a_noast,
        "option_b_ast_excluded": b_noast,
        "bird_group": bird_group,
        "audio_group": audio_group,
        "bird_group_ast_excluded": bird_group_noast,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq1_synthetic_group_test.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {out_json}")

    md = build_markdown(original, a_full, b_full, a_noast, b_noast, bird_group, audio_group)
    out_md = args.out_dir / "rq1_synthetic_group_test.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
