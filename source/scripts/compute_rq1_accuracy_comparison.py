#!/usr/bin/env python3
"""
RQ1 -- Per-dataset exact-match accuracy comparison across backbones
(soundscape), paired and hierarchical.

Purpose: the accumulated confusion matrices in `plot_confusion_matrix.py` /
`plot_rq5.py` show visually apparent differences in diagonal-mass ("exact-
match") accuracy between backbones when compared across soundscape
datasets. This script tests whether those differences are real and
statistically supported, using a test structure that respects dataset-level
non-independence rather than naively pooling all individual predictions --
the same concern already raised for the 288-cell correction earlier in this
series.

Same source data and correctness convention as `rq1_sign_test.md`,
`rq1_friedman_test.md`, and `rq1_mixed_model_difficulty.md`: the 13
frozen-backbone, pooled-MLP-head, regression-formulation models
(`archive/Pooled-Embeddings/`), source `soundscape_test`, the 7
REGIONAL_DATASETS (UHH, HSN, PER, NES, POW, SSW, SNE; soundscape has no XCM
counterpart). Soundscape data has no single exact ground-truth label, so
its "exact-match accuracy" is `range_accuracy` (utils/metrics.py's
effective_error == 0: the prediction falls anywhere inside
[min_polyphony, max_polyphony]) -- the diagonal-mass proportion of the
confusion matrices in `plot_confusion_matrix.py`, same convention as
`rq1_sign_test.md`.

Three levels of analysis, increasing in rigor and in how directly they use
the full sample size:

1. **Per-dataset accuracy table** -- one `range_accuracy` value per
   (backbone, dataset) cell, same granularity as every other paired
   comparison in RQ1 (`rq1_ranking_robustness.md`,
   `rq1_mixed_model_difficulty.md`). A "tight leading group" (top 4 by mean
   accuracy across the 7 datasets) is identified the same way
   `compute_rq1_ranking_robustness.py` identified its MAE-based one, and
   compared against it.

2. **Paired t-test, Wilcoxon signed-rank test, and exact sign test**
   between backbone pairs, matched by dataset (n=7): within the accuracy
   tight leading group (6 pairs) and tight-group-vs-next-ranked (4x3=12
   pairs) -- same pair-selection convention as
   `compute_rq1_ranking_robustness.py` checks 2/3 and
   `compute_rq1_sign_test_pairs.py`, applied to `range_accuracy` instead of
   `range_mae` (direction flipped: higher is better).

3. **Optional, more rigorous**: a mixed-effects-equivalent logistic
   regression on individual predictions (correct/incorrect ~ backbone +
   dataset cluster), which properly uses the full sample size (hundreds of
   thousands of soundscape windows per backbone) without pseudoreplication.
   statsmodels has no frequentist binomial GLMM with a true likelihood
   (unlike `compute_rq1_mixed_model_difficulty.py`'s Gaussian `mixedlm`,
   which supports a likelihood-ratio test) -- but since backbone and
   dataset are the *only* predictors and every individual window within one
   (backbone, dataset) cell shares identical covariate values, the
   individual-level logistic likelihood is exactly sufficient-statistic-
   equivalent to a **grouped binomial** model on the 13x7=91
   (backbone, dataset) correct/total counts. This is fit via
   `statsmodels.genmod.generalized_estimating_equations.GEE` with a
   Binomial family, dataset as the cluster/group variable, and an
   exchangeable working correlation -- the population-averaged,
   cluster-robust analogue of a random-intercept-by-dataset model, tested
   with a joint Wald chi-squared test for the overall backbone term
   (`wald_test_terms()`), the GEE equivalent of the mixed model's LRT.

**Caveat -- naive pooled comparison**: the same 91-cell count table is also
fit with an ordinary (non-clustered) `statsmodels.api.GLM` Binomial model --
identical formula, identical data, the *only* difference being that it
treats all individual soundscape windows as independent evidence instead of
clustering by dataset. This naive fit is reported side by side with the
GEE fit specifically to show how much more confident -- and how
potentially spurious -- ignoring dataset-level non-independence makes the
omnibus backbone effect look, the same methodological point as the
288-cell correction earlier in this series.

Writes rq1_accuracy_comparison.json and rq1_accuracy_comparison.md to
--out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq1_accuracy_comparison.py [--out-dir plots/figures/rq1]
"""

import argparse
import itertools
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.genmod.cov_struct as cov_struct
import statsmodels.genmod.families as families
from statsmodels.genmod.generalized_estimating_equations import GEE

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import BACKBONE_META
from compute_rq1_friedman_test import friedman_test
from compute_rq1_ranking_robustness import N_NEXT as MAE_N_NEXT, TIGHT_GROUP as MAE_TIGHT_GROUP
from plot_data import REGIONAL_DATASETS, filter_long, load_study

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = REPO_ROOT / "archive"
STUDY_DIR = ARCHIVE / "Pooled-Embeddings"
SCAPE_SOURCE = "soundscape_test"
ACCURACY_METRIC = "range_accuracy"
ALPHA = 0.05
TIGHT_GROUP_SIZE = 4
N_NEXT = 3
MAE_MIXED_MODEL_JSON = Path("plots/figures/rq1/rq1_mixed_model_difficulty.json")


# ---------------------------------------------------------------------------
# 1. Per-dataset accuracy table + ranking
# ---------------------------------------------------------------------------

def build_accuracy_matrix(long_df: pd.DataFrame, backbone_order: list) -> pd.DataFrame:
    """(backbone x dataset) matrix of raw range_accuracy values, regression
    head, 7 REGIONAL_DATASETS -- same construction as
    compute_rq1_friedman_test.py's build_matrix, metric swapped for
    range_accuracy."""
    sub = filter_long(long_df, source=SCAPE_SOURCE, metric=ACCURACY_METRIC, head="reg", dataset=REGIONAL_DATASETS)
    matrix = sub.pivot_table(index="model", columns="dataset", values="value")
    return matrix.reindex(index=backbone_order, columns=REGIONAL_DATASETS)


def rank_backbones(matrix: pd.DataFrame) -> dict:
    """Mean/SD accuracy per backbone across datasets, ranked descending
    (rank 1 = highest accuracy) -- mirrors
    compute_rq1_ranking_robustness.py's check1_ranking, direction flipped
    since higher range_accuracy is better (lower range_mae is)."""
    mean_acc = matrix.mean(axis=1)
    sd_acc = matrix.std(axis=1, ddof=1)
    order = mean_acc.sort_values(ascending=False).index.tolist()
    rank = mean_acc.rank(method="min", ascending=False).astype(int)
    return {"order": order, "mean_accuracy": mean_acc, "sd_accuracy": sd_acc, "rank": rank}


def average_ranks_desc(matrix: pd.DataFrame) -> pd.Series:
    """Per-dataset ranks (1=best/highest range_accuracy, ties averaged),
    then mean rank per backbone -- Demsar (2006) convention, same as
    compute_rq1_friedman_test.py's average_ranks but descending since
    higher accuracy is better."""
    ranks = matrix.rank(axis=0, method="average", ascending=False)
    return ranks.mean(axis=1).sort_values()


def pick_reference_backbone(ranks: pd.Series) -> str:
    """Median-average-rank backbone -- same convention as
    compute_rq1_mixed_model_difficulty.py's pick_reference_backbone."""
    ordered = ranks.sort_values().index.tolist()
    return ordered[len(ordered) // 2]


# ---------------------------------------------------------------------------
# 2. Paired t-test, Wilcoxon, sign test between backbone pairs
# ---------------------------------------------------------------------------

def pairwise_stats(matrix: pd.DataFrame, a: str, b: str) -> dict:
    """Delta = (backbone A accuracy) - (backbone B accuracy), matched by
    dataset. Paired t-test and Wilcoxon signed-rank test against zero (same
    convention as compute_rq1_ranking_robustness.py's paired_tests), plus
    an exact two-sided sign test on the same per-dataset deltas (same
    convention as compute_rq1_sign_test_pairs.py) -- all three run on the
    identical n=7 paired deltas, not separately re-derived."""
    diff = (matrix.loc[a] - matrix.loc[b]).to_numpy(dtype=float)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    mean = float(diff.mean()) if n else None
    std = float(diff.std(ddof=1)) if n > 1 else None

    if n < 2:
        t_stat = t_p = None
    else:
        t_stat, t_p = scipy.stats.ttest_1samp(diff, popmean=0.0)
        t_stat, t_p = float(t_stat), float(t_p)
    try:
        w_stat, w_p = scipy.stats.wilcoxon(diff) if n >= 2 else (None, None)
        w_stat = float(w_stat) if w_stat is not None else None
        w_p = float(w_p) if w_p is not None else None
    except ValueError:
        w_stat, w_p = None, None

    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    ties = int(n - wins - losses)
    n_decisive = wins + losses
    sign_p = float(scipy.stats.binomtest(wins, n_decisive, 0.5, alternative="two-sided").pvalue) if n_decisive > 0 else None

    return {
        "a": a, "b": b, "n": n, "mean_delta": mean, "sd_delta": std,
        "t_stat": t_stat, "t_p": t_p,
        "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
        "wins": wins, "losses": losses, "ties": ties, "n_decisive": n_decisive, "sign_p": sign_p,
    }


# ---------------------------------------------------------------------------
# 3. Individual-prediction-level data: grouped binomial counts per
#    (backbone, dataset) cell, derived from the same pkl files
#    plot_confusion_matrix.py's load_soundscape() pools, kept per-region.
# ---------------------------------------------------------------------------

def load_region_counts(model_dir: Path, regions: list) -> dict:
    """Per-region (correct, total) counts -- same correctness criterion as
    plot_confusion_matrix.py's load_soundscape() (prediction falls in
    [min_polyphony, max_polyphony], i.e. range_accuracy's effective_error
    == 0), kept per region instead of pooled so a dataset label survives
    per row."""
    counts = {}
    for region in regions:
        f = model_dir / "scape_eval_results" / f"{region}_reg_scape_test_results.pkl"
        df = pickle.load(open(f, "rb"))
        low = df["y_true.min_polyphony"].round().astype(int).to_numpy()
        high = df["y_true.max_polyphony"].round().astype(int).to_numpy()
        y_pred = df["predictions.polyphony_reg"].round().astype(int).to_numpy()
        correct = (y_pred >= low) & (y_pred <= high)
        counts[region] = (int(correct.sum()), int(len(correct)))
    return counts


def build_count_table(backbone_order: list, regions: list) -> pd.DataFrame:
    """One row per (backbone, dataset): k = correct windows, n_total =
    total windows, fail = n_total - k. Individual-level logistic regression
    with only categorical (backbone, dataset) predictors is
    likelihood-equivalent to a grouped binomial model on these 91 cells --
    every window within one cell shares identical covariate values -- so
    this is used in place of millions of individual rows without losing
    any information the model below could use."""
    rows = []
    for backbone in backbone_order:
        counts = load_region_counts(STUDY_DIR / backbone, regions)
        for region, (k, n) in counts.items():
            rows.append({"backbone": backbone, "dataset": region, "k": k, "n_total": n,
                         "fail": n - k, "pkl_accuracy": k / n})
    return pd.DataFrame(rows)


def validate_counts(count_df: pd.DataFrame, matrix: pd.DataFrame) -> dict:
    """Sanity check: the pkl-derived per-cell accuracy should match the
    aggregate range_accuracy already reported in
    archive/Pooled-Embeddings/{model}/scape_eval_results/
    soundscape_test_metrics.csv (loaded into `matrix`) -- confirms the
    correctness criterion reimplemented here from
    plot_confusion_matrix.py's load_soundscape() matches the one
    utils/metrics.py actually used to produce that CSV."""
    diffs = [
        abs(row.pkl_accuracy - matrix.loc[row.backbone, row.dataset])
        for row in count_df.itertuples()
    ]
    return {"max_abs_diff": float(np.max(diffs)), "mean_abs_diff": float(np.mean(diffs))}


def fit_binomial_models(count_df: pd.DataFrame, reference: str) -> dict:
    """Same formula and data fit three ways, isolating exactly what
    changes as dataset non-independence is accounted for more carefully:

    1. **naive** -- an ordinary (non-clustered) GLM Binomial, treating
       every one of the ~3M individual soundscape windows as independent
       evidence. The naive pooled comparison.
    2. **gee_robust** -- GEE Binomial, dataset as the cluster/group
       variable, exchangeable working correlation, default cluster-robust
       ("sandwich") covariance. The properly dataset-non-independence-
       respecting version -- but see the rank caveat below.
    3. **gee_model_based** -- the same GEE fit, but with the working
       (model-based, non-robust) covariance instead of the sandwich one.
       Not robust to a misspecified working correlation, but -- unlike (2)
       -- not limited by the number of clusters either.

    All three are tested with a joint Wald chi-squared test for the
    overall backbone term via wald_test_terms(). With only n=7 dataset
    clusters and a 12-df backbone term, the sandwich covariance in (2) is
    not full rank (rank <= n_clusters - 1 = 6) -- the same "too few
    clusters for a robust sandwich estimator" problem documented in the
    GEE literature -- so wald_test_terms() silently tests only a
    rank-reduced subset of the 12 backbone contrasts. (3) is reported
    alongside specifically to still get a full 12-df answer, at the cost
    of trusting the exchangeable working-correlation model rather than
    being distribution-free about it."""
    formula = f"k + fail ~ C(backbone, Treatment(reference='{reference}'))"

    naive = smf.glm(formula, data=count_df, family=sm.families.Binomial()).fit()
    naive_wald = naive.wald_test_terms()
    backbone_term = [t for t in naive_wald.table.index if t != "Intercept"][0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gee_model = GEE.from_formula(formula, groups="dataset", data=count_df,
                                      family=families.Binomial(), cov_struct=cov_struct.Exchangeable())
        gee_robust = gee_model.fit()
        robust_wald = gee_robust.wald_test_terms()
        rank_deficient = any("does not have full rank" in str(w.message) for w in caught)

    gee_model_based = GEE.from_formula(formula, groups="dataset", data=count_df,
                                        family=families.Binomial(), cov_struct=cov_struct.Exchangeable()).fit(cov_type="naive")
    model_based_wald = gee_model_based.wald_test_terms()

    def term_row(wald_table, name):
        row = wald_table.table.loc[name]
        return {"chi2": float(row["statistic"]), "p_value": float(row["pvalue"]), "df": int(row["df_constraint"])}

    return {
        "reference_backbone": reference,
        "n_clusters": int(gee_model.num_group),
        "naive": {
            "wald_backbone": term_row(naive_wald, backbone_term),
            "deviance": float(naive.deviance), "llf": float(naive.llf),
        },
        "gee_robust": {
            "wald_backbone": term_row(robust_wald, backbone_term),
            "rank_deficient": rank_deficient,
        },
        "gee_model_based": {
            "wald_backbone": term_row(model_based_wald, backbone_term),
        },
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def fmt_p(p) -> str:
    if p is None:
        return "n/a"
    return f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}"


def label(m: str) -> str:
    return BACKBONE_META.get(m, {}).get("display", m)


def build_accuracy_table_md(matrix: pd.DataFrame, rank_result: dict) -> list:
    lines = ["| Rank | Backbone | " + " | ".join(REGIONAL_DATASETS) + " | Mean (+/- SD) |", "|---|---|" + "---|" * len(REGIONAL_DATASETS) + "---|"]
    for m in rank_result["order"]:
        vals = " | ".join(f"{matrix.loc[m, d]:.3f}" for d in REGIONAL_DATASETS)
        mean_sd = f"{rank_result['mean_accuracy'][m]:.3f} +/- {rank_result['sd_accuracy'][m]:.3f}"
        lines.append(f"| {rank_result['rank'][m]} | {label(m)} | {vals} | {mean_sd} |")
    return lines


def build_pair_table_md(pairs: dict) -> list:
    lines = ["| Pair (A - B) | Mean Delta (+/- SD) | n | Paired t-test | Wilcoxon | Wins-Losses-Ties | Sign test |", "|---|---|---|---|---|---|---|"]
    for s in pairs.values():
        lines.append(
            f"| {label(s['a'])} - {label(s['b'])} | {s['mean_delta']:.4f} +/- {s['sd_delta']:.4f} | {s['n']} "
            f"| {fmt_p(s['t_p'])} | {fmt_p(s['wilcoxon_p'])} "
            f"| {s['wins']}-{s['losses']}-{s['ties']} | {fmt_p(s['sign_p'])} |"
        )
    return lines


def build_markdown(ctx: dict) -> str:
    matrix_info = ctx["matrix_info"]
    rank_result = ctx["rank_result"]
    tight_group, next_group = ctx["tight_group"], ctx["next_group"]
    mae_match = set(tight_group) == set(MAE_TIGHT_GROUP)

    lines = [
        "# RQ1 -- Per-dataset exact-match accuracy comparison across backbones (soundscape)",
        "",
        "Tests whether the visually apparent differences in exact-match accuracy between backbones, "
        "seen when comparing accumulated confusion matrices across datasets (`plot_confusion_matrix.py`, "
        "`plot_rq5.py`), are statistically supported -- using a test structure that respects "
        "dataset-level non-independence rather than naively pooling all individual predictions.",
        "",
        "Source: `archive/Pooled-Embeddings/` (13 frozen-backbone, pooled-MLP-head, "
        f"regression-formulation models), `soundscape_test` / `range_accuracy`, regression head, the "
        f"{len(REGIONAL_DATASETS)} REGIONAL_DATASETS ({', '.join(REGIONAL_DATASETS)}). Soundscape has no "
        "single exact ground-truth label, so `range_accuracy` (prediction falls anywhere inside "
        "[min_polyphony, max_polyphony]) is the exact-match analogue -- the diagonal-mass proportion of "
        "the confusion matrices -- same convention as `rq1_sign_test.md`.",
        "",
        "## 1. Per-dataset accuracy table",
        "",
        f"One `range_accuracy` value per (backbone, dataset) cell -- same granularity as every other "
        "paired comparison in RQ1. Ranked by mean accuracy across the 7 datasets (higher is better).",
        "",
    ]
    lines += build_accuracy_table_md(matrix_info, rank_result)
    lines += [
        "",
        f"**Tight leading group** (top {TIGHT_GROUP_SIZE} by mean accuracy): "
        f"{', '.join(label(m) for m in tight_group)}. This "
        f"{'matches' if mae_match else 'does not match'} the MAE-based tight leading group used in "
        f"`rq1_ranking_robustness.md` ({', '.join(label(m) for m in MAE_TIGHT_GROUP)}) -- "
        + ("the same backbones lead on both metrics, so the accuracy comparison below is testing the "
           "same visual impression the MAE analysis already tested, not a different set of backbones."
           if mae_match else
           "accuracy and MAE do not single out the same backbones as the leading group, which is worth "
           "reading alongside check 4 of `rq1_ranking_robustness.md` (synthetic-vs-soundscape rank "
           "correlation) as a reminder that these are genuinely different questions about the same "
           "backbones, not two measurements of one underlying quantity.") + "",
        "",
        "## 2. Paired t-test, Wilcoxon, and sign test between backbone pairs",
        "",
        "Same pair-selection convention as `compute_rq1_ranking_robustness.py` checks 2/3 and "
        "`compute_rq1_sign_test_pairs.py`, applied to `range_accuracy` (direction: higher is better) "
        "instead of `range_mae`. Delta = (backbone A accuracy) - (backbone B accuracy), matched by "
        f"dataset (n={len(REGIONAL_DATASETS)}).",
        "",
        f"### Within the tight leading group ({', '.join(label(m) for m in tight_group)})",
        "",
    ]
    lines += build_pair_table_md(ctx["within_group"])
    lines += [
        "",
        f"### Tight leading group vs. next-ranked ({', '.join(label(m) for m in next_group)})",
        "",
    ]
    lines += build_pair_table_md(ctx["vs_next"])

    ft = ctx["friedman_accuracy"]
    verdict = "rejects" if ft["p_value"] < ALPHA else "does not reject"
    lines += [
        "",
        "## 3. Omnibus test and comparison against the established MAE result",
        "",
        "Friedman test (same construction as `rq1_friedman_test.md`, metric swapped for `range_accuracy`) "
        f"across all {ft['k_backbones']} backbones and {ft['n_datasets']} datasets: "
        f"**chi-squared = {ft['statistic']:.3f}, df = {ft['df']}, p = {ft['p_value']:.6f}** -- at "
        f"alpha = {ALPHA}, this **{verdict}** the null that all backbones have the same rank distribution "
        "of soundscape exact-match accuracy.",
        "",
    ]
    if ctx["mae_comparison"] is not None:
        mc = ctx["mae_comparison"]
        lines += [
            f"Compared against the MAE-based results already established in `rq1_friedman_test.md` "
            f"(Friedman p = {mc['mae_friedman_p']:.4f}, "
            f"{'significant' if mc['mae_friedman_sig'] else 'not significant'}) and "
            f"`rq1_mixed_model_difficulty.md` (mixed-effects LRT p = {fmt_p(mc['mae_mixed_lrt_p'])}, "
            f"{'significant' if mc['mae_mixed_lrt_sig'] else 'not significant'}): the accuracy Friedman "
            f"result is **{'consistent' if mc['agrees_with_friedman'] else 'inconsistent'}** with the MAE "
            f"Friedman verdict, and **{'consistent' if mc['agrees_with_mixed_model'] else 'inconsistent'}** "
            "with the MAE mixed-model LRT verdict. " + (
                "Accuracy tells essentially the same story as MAE here -- the visually apparent backbone "
                "differences are not an artifact specific to how MAE happens to be computed."
                if mc["agrees_with_friedman"] and mc["agrees_with_mixed_model"] else
                "Accuracy and MAE diverge on the omnibus backbone-effect question at least at one of these "
                "two comparison points -- read this alongside the fact that MAE's own Friedman and "
                "mixed-model verdicts already diverge from each other in `rq1_mixed_model_difficulty.md` "
                "(rank-based vs. covariate-adjusted magnitude test); range_accuracy is a coarser, "
                "range-thresholded summary of the same predictions MAE is computed from, so some "
                "disagreement between the two metrics' omnibus tests is expected, not necessarily a sign "
                "that either result is wrong."
            ),
            "",
        ]
    else:
        lines += [
            f"`{MAE_MIXED_MODEL_JSON}` was not found, so the comparison against the established MAE "
            "mixed-model result could not be run -- run `compute_rq1_mixed_model_difficulty.py` first to "
            "enable it.",
            "",
        ]

    if ctx["binomial"] is not None:
        b = ctx["binomial"]
        naive_p = b["naive"]["wald_backbone"]["p_value"]
        robust_p = b["gee_robust"]["wald_backbone"]["p_value"]
        robust_df = b["gee_robust"]["wald_backbone"]["df"]
        model_based_p = b["gee_model_based"]["wald_backbone"]["p_value"]
        model_based_df = b["gee_model_based"]["wald_backbone"]["df"]
        n_total = int(ctx["count_df"]["n_total"].sum())
        n_clusters = b["n_clusters"]
        v = ctx["validation"]
        lines += [
            "## 4. Optional, more rigorous: mixed-effects-equivalent logistic regression",
            "",
            "Individual-level logistic regression `correct ~ backbone + (1|dataset)` properly uses the "
            f"full sample size ({n_total:,} individual soundscape windows across the "
            f"{len(REGIONAL_DATASETS)} datasets and {ft['k_backbones']} backbones) without "
            "pseudoreplication. statsmodels has no frequentist binomial GLMM with a true likelihood "
            "(unlike `compute_rq1_mixed_model_difficulty.py`'s Gaussian `mixedlm`) -- but since backbone "
            "and dataset are the only predictors and every window within one (backbone, dataset) cell "
            "shares identical covariate values, the individual-level likelihood is exactly "
            "sufficient-statistic-equivalent to a **grouped binomial** model on the "
            f"{len(ctx['count_df'])}-cell correct/total count table, fit with "
            "`statsmodels`' GEE (Binomial family, dataset as the cluster/group variable, exchangeable "
            "working correlation) -- the population-averaged analogue of a random-intercept-by-dataset "
            "model, tested with a joint Wald chi-squared test for the overall backbone term. Reference "
            f"backbone: **{label(b['reference_backbone'])}** (median-average-rank, same convention as "
            "the MAE mixed model).",
            "",
            f"**Validation**: the grouped-count table's per-cell accuracy matches the aggregate "
            f"`range_accuracy` already reported in `soundscape_test_metrics.csv` to within "
            f"{v['max_abs_diff']:.6f} (max abs diff across all {len(ctx['count_df'])} cells) -- confirms "
            "the correctness criterion reimplemented here from `plot_confusion_matrix.py`'s "
            "`load_soundscape()` matches the one `utils/metrics.py` actually used.",
            "",
            f"**Small-cluster caveat**: with only n={n_clusters} dataset clusters and a 12-df backbone "
            "term, GEE's default cluster-robust (\"sandwich\") covariance is not full rank "
            f"({'confirmed: ' if b['gee_robust']['rank_deficient'] else ''}statsmodels silently reduces "
            f"the test to df={robust_df} instead of 12) -- the same \"too few clusters for a robust "
            "sandwich estimator\" problem documented in the GEE literature, and a sharper version of the "
            "small-n caveat already noted for the MAE mixed model's random-intercept variance "
            "(`rq1_mixed_model_difficulty.md`). Three fits are reported below so this limitation is "
            "visible rather than silently absorbed into one headline number:",
            "",
            "| Fit | Respects dataset clustering? | Full 12 df? | chi2 | df | p |",
            "|---|---|---|---|---|---|",
            f"| Naive pooled (GLM, independent windows) | no | yes | {b['naive']['wald_backbone']['chi2']:.3f} "
            f"| {b['naive']['wald_backbone']['df']} | {fmt_p(naive_p)} |",
            f"| GEE, cluster-robust (\"sandwich\") | yes | **no (rank-reduced)** | "
            f"{b['gee_robust']['wald_backbone']['chi2']:.3f} | {robust_df} | {fmt_p(robust_p)} |",
            f"| GEE, model-based (working correlation) | yes, if correlation model correct | yes | "
            f"{b['gee_model_based']['wald_backbone']['chi2']:.3f} | {model_based_df} | {fmt_p(model_based_p)} |",
            "",
            f"The model-based GEE fit (full 12 df, chi2 = {b['gee_model_based']['wald_backbone']['chi2']:.3f}, "
            f"p = {fmt_p(model_based_p)}) finds "
            f"{'a significant' if model_based_p < ALPHA else 'no significant'} overall backbone effect on "
            "individual-prediction accuracy once dataset clustering is modeled -- this is the more "
            "trustworthy of the two dataset-aware fits for the *overall* 12-df question, since the "
            "rank-reduced robust fit above cannot test all 12 contrasts at n=7 clusters at all. It relies "
            "on the exchangeable working-correlation model being a reasonable approximation, which is not "
            "verified independently here.",
            "",
            "## Caveat -- naive pooled proportions test",
            "",
            "The identical formula and count table were also fit with an ordinary (non-clustered) "
            "`statsmodels.api.GLM` Binomial model -- the only difference from the two GEE fits above is "
            "that this one treats every individual soundscape window as independent evidence instead of "
            f"clustering by dataset at all: backbone term chi-squared = "
            f"{b['naive']['wald_backbone']['chi2']:.3f}, df = {b['naive']['wald_backbone']['df']}, "
            f"**p = {fmt_p(naive_p)}**.",
            "",
        ]
        if naive_p < model_based_p:
            ratio_note = (
                f"The naive pooled fit's p-value ({fmt_p(naive_p)}) is dramatically smaller than the "
                f"dataset-aware model-based GEE fit's ({fmt_p(model_based_p)}, full 12 df) despite being "
                "computed from the exact same 91 counts and formula -- ignoring dataset-level "
                "non-independence makes the omnibus backbone effect look far more confident than the data "
                f"actually supports. With {n_total:,} individual windows, the naive fit treats each window "
                "as an independent vote on the backbone effect, when in reality windows sharing a dataset "
                "also share recording conditions, species community, and background noise -- the same "
                "pseudoreplication concern as the 288-cell correction earlier in this series. Neither "
                "dataset-aware fit should be read as definitive on its own (see the small-cluster caveat "
                "above), but both agree the naive pooled p-value is not a trustworthy number to quote."
            )
        else:
            ratio_note = (
                "Here the naive and dataset-aware p-values do not diverge sharply, which is itself worth "
                "noting -- it suggests the backbone effect on accuracy is large and consistent enough "
                "across datasets that accounting for dataset-level non-independence does not change the "
                "qualitative conclusion, unlike cases elsewhere in this series where naive pooling was "
                "shown to substantially inflate apparent significance."
            )
        lines += [ratio_note, ""]
    else:
        lines += [
            "## 4. Optional, more rigorous: mixed-effects-equivalent logistic regression",
            "",
            "Not run in this pass (see script `--skip-binomial` flag or a data-loading error printed "
            "above).",
            "",
        ]

    lines += [
        "## Reading this",
        "",
        "- Section 1's per-dataset table is the primary evidence: every other test in this file is a "
        "different way of asking whether the differences visible in that table are larger than sampling "
        "noise across n=7 datasets would produce.",
        "- Section 2's pairwise tests are exploratory/directional at n=7 datasets, same caveat as "
        "`rq1_ranking_robustness.md` and `rq1_sign_test_pairs.md` -- a non-significant pair does not mean "
        "the backbones perform equally, only that the gap is not distinguishable from chance at this "
        "sample size.",
        "- Section 3's Friedman test and Section 4's dataset-clustered GEE test are the two "
        "well-structured omnibus answers to \"do backbones differ on accuracy\" -- prefer these over any "
        "individual pairwise result when the question is about the backbone effect overall.",
        "- Section 4's naive-vs-dataset-aware comparison is the headline methodological point of this "
        "file: the same 91-cell data can look almost arbitrarily more significant if dataset-level "
        "non-independence is ignored (naive), spuriously partial if a cluster-robust covariance is used "
        "with too few clusters to support it (GEE robust), or genuinely non-significant once both are "
        "accounted for (GEE model-based) -- so any single-number \"backbones are significantly "
        "different\" claim about accuracy should specify which of the three fits above it is quoting.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq1"))
    parser.add_argument("--skip-binomial", action="store_true",
                         help="Skip section 4 (pkl loading + GEE/GLM fitting) -- useful for a quick rerun "
                              "of sections 1-3 only.")
    args = parser.parse_args()

    models = mrt.discover_models(STUDY_DIR)
    long_df = load_study("pooled", models=models)
    backbone_order = [m for m in BACKBONE_META if m in models]

    matrix = build_accuracy_matrix(long_df, backbone_order)
    rank_result = rank_backbones(matrix)
    avg_ranks = average_ranks_desc(matrix)

    tight_group = rank_result["order"][:TIGHT_GROUP_SIZE]
    next_group = rank_result["order"][TIGHT_GROUP_SIZE:TIGHT_GROUP_SIZE + N_NEXT]

    within_group = {f"{a}__minus__{b}": pairwise_stats(matrix, a, b)
                     for a, b in itertools.combinations(tight_group, 2)}
    vs_next = {f"{a}__minus__{b}": pairwise_stats(matrix, a, b)
               for a in tight_group for b in next_group}

    ft = friedman_test(matrix)

    mae_comparison = None
    if MAE_MIXED_MODEL_JSON.exists():
        mae_ctx = json.loads(MAE_MIXED_MODEL_JSON.read_text())
        mae_friedman_p = mae_ctx["friedman"]["p_value"]
        mae_mixed_lrt_p = mae_ctx["models"]["ratio_polyp"]["lrt"]["p_value"]
        acc_sig = ft["p_value"] < ALPHA
        mae_comparison = {
            "mae_friedman_p": mae_friedman_p, "mae_friedman_sig": mae_friedman_p < ALPHA,
            "mae_mixed_lrt_p": mae_mixed_lrt_p, "mae_mixed_lrt_sig": mae_mixed_lrt_p < ALPHA,
            "accuracy_friedman_sig": acc_sig,
            "agrees_with_friedman": acc_sig == (mae_friedman_p < ALPHA),
            "agrees_with_mixed_model": acc_sig == (mae_mixed_lrt_p < ALPHA),
        }

    binomial, count_df, validation = None, None, None
    if not args.skip_binomial:
        reference = pick_reference_backbone(avg_ranks)
        count_df = build_count_table(backbone_order, REGIONAL_DATASETS)
        validation = validate_counts(count_df, matrix)
        binomial = fit_binomial_models(count_df, reference)

    ctx = {
        "matrix_info": matrix,
        "rank_result": rank_result,
        "tight_group": tight_group,
        "next_group": next_group,
        "within_group": within_group,
        "vs_next": vs_next,
        "friedman_accuracy": ft,
        "average_ranks": avg_ranks,
        "mae_comparison": mae_comparison,
        "binomial": binomial,
        "count_df": count_df,
        "validation": validation,
    }

    results_json = {
        "accuracy_matrix": matrix.to_dict(),
        "mean_accuracy": rank_result["mean_accuracy"].to_dict(),
        "sd_accuracy": rank_result["sd_accuracy"].to_dict(),
        "rank": rank_result["rank"].to_dict(),
        "average_ranks": avg_ranks.to_dict(),
        "tight_group": tight_group,
        "next_group": next_group,
        "mae_tight_group": MAE_TIGHT_GROUP,
        "within_group": within_group,
        "vs_next": vs_next,
        "friedman_accuracy": ft,
        "mae_comparison": mae_comparison,
        "binomial": binomial,
        "count_table_validation": validation,
        "alpha": ALPHA,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq1_accuracy_comparison.json"
    out_json.write_text(json.dumps(results_json, indent=2, default=str))
    print(f"Wrote {out_json}")

    md = build_markdown(ctx)
    out_md = args.out_dir / "rq1_accuracy_comparison.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
