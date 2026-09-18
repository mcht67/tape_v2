#!/usr/bin/env python3
"""
RQ1 -- Linear mixed-effects model, backbone effect controlling for dataset
difficulty (soundscape data).

Extends `compute_rq1_friedman_test.py`'s rank-based omnibus result with an
explicit covariate-adjusted model. This is a distinct question from the
Friedman test -- not a re-test of "do backbones differ," but "do backbones
differ once per-dataset difficulty (polyphony) is accounted for as a
covariate rather than just a repeated-measures block."

Same source data as the Friedman test: the 13 frozen-backbone, pooled-MLP-
head, regression-formulation models (`archive/Pooled-Embeddings/`), source
`soundscape_test`, metric `range_mae`, across the 7 REGIONAL_DATASETS (UHH,
HSN, PER, NES, POW, SSW, SNE). Per-dataset difficulty (ratio_polyp,
mean_polyp) comes from `plots/data/polybirdmix_soundscape_stats.json`'s
polyphony_metric_summary, the same source used in
`compute_rq5_polyphony_robustness.py`.

Model: MAE ~ backbone + ratio_polyp + (1|dataset), fit via
statsmodels.formula.api.mixedlm (ML, not REML, so the fixed-effect
likelihood-ratio test below is valid). backbone is a 13-level categorical
fixed effect (reference = the median-average-rank backbone from the
Friedman test, i.e. a "typical" backbone rather than an arbitrary
alphabetical first); ratio_polyp is a continuous fixed-effect covariate;
dataset is a random intercept (7 groups).

The overall backbone effect is tested via a likelihood-ratio test against
the reduced model MAE ~ ratio_polyp + (1|dataset) -- the direct mixed-model
analogue of the Friedman omnibus test, but with difficulty explicitly
partialled out. The whole thing is repeated with mean_polyp in place of
ratio_polyp as a robustness check.

Diagnostics (residual skew, Shapiro-Wilk normality, Levene's test for
equal residual variance across the 7 dataset groups) are reported to flag
whether the mixed model's parametric assumptions are reasonably met, given
the heteroscedasticity across datasets already documented elsewhere in this
analysis series.

Caveat: n=7 datasets (random-effect groups) is small for a mixed model's
random-effect variance to be well estimated -- results here are exploratory
and reported alongside, not as a replacement for, the Friedman test result.

Writes rq1_mixed_model_difficulty.json and rq1_mixed_model_difficulty.md to
--out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq1_mixed_model_difficulty.py [--out-dir plots/figures/rq1]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import BACKBONE_META
from compute_rq1_friedman_test import SCAPE_SOURCE, SOUNDSCAPE_METRIC, average_ranks, build_matrix, friedman_test
from compute_rq5_polyphony_robustness import difficulty_value
from plot_data import REGIONAL_DATASETS, load_study
from plot_rq5 import STATS_JSON

ALPHA = 0.05
DIFFICULTY_COVARIATES = {"ratio_polyp": "Ratio polyphonic", "mean_polyp": "Mean polyphony"}


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_long_df(matrix: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """One row per (backbone, dataset): mae, ratio_polyp, mean_polyp."""
    long = matrix.reset_index().melt(id_vars="model", var_name="dataset", value_name="mae")
    long = long.rename(columns={"model": "backbone"})
    for key in DIFFICULTY_COVARIATES:
        long[key] = long["dataset"].map(lambda d: difficulty_value(stats, d, key))
    return long.dropna(subset=["mae"]).reset_index(drop=True)


def pick_reference_backbone(matrix: pd.DataFrame) -> str:
    """Median-average-rank backbone (Friedman/Demsar ranking) -- a "typical"
    reference level rather than an arbitrary alphabetical first."""
    ranks = average_ranks(matrix)
    ordered = ranks.sort_values().index.tolist()
    return ordered[len(ordered) // 2]


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_full_model(df: pd.DataFrame, covariate: str, reference: str):
    formula = f"mae ~ C(backbone, Treatment(reference='{reference}')) + {covariate}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, data=df, groups=df["dataset"])
        result = model.fit(reml=False)
    return result


def fit_reduced_model(df: pd.DataFrame, covariate: str):
    formula = f"mae ~ {covariate}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, data=df, groups=df["dataset"])
        result = model.fit(reml=False)
    return result


def fit_no_covariate_models(df: pd.DataFrame, reference: str):
    """backbone + (1|dataset) vs. just (1|dataset), no difficulty covariate at
    all -- isolates whether backbone's significance in the covariate-adjusted
    model above is created by the covariate absorbing variance, or is present
    regardless of the covariate (i.e. the mixed model is simply more powerful
    than the rank-based Friedman test on this data either way)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full_formula = f"mae ~ C(backbone, Treatment(reference='{reference}'))"
        full = smf.mixedlm(full_formula, data=df, groups=df["dataset"]).fit(reml=False)
        null = smf.mixedlm("mae ~ 1", data=df, groups=df["dataset"]).fit(reml=False)
    return full, null


def fixed_effects_table(result) -> dict:
    params = result.params
    bse = result.bse
    tvalues = result.tvalues
    pvalues = result.pvalues
    out = {}
    for name in params.index:
        if name == "Group Var":
            continue
        out[name] = {
            "coef": float(params[name]),
            "se": float(bse[name]),
            "z": float(tvalues[name]),
            "p_value": float(pvalues[name]),
        }
    return out


def likelihood_ratio_test(full_result, reduced_result) -> dict:
    k_full = full_result.model.exog.shape[1]
    k_reduced = reduced_result.model.exog.shape[1]
    df_diff = k_full - k_reduced
    lr_stat = 2 * (full_result.llf - reduced_result.llf)
    p_value = float(scipy.stats.chi2.sf(lr_stat, df_diff))
    return {
        "lr_statistic": float(lr_stat), "df": int(df_diff), "p_value": p_value,
        "full_llf": float(full_result.llf), "reduced_llf": float(reduced_result.llf),
    }


def random_effects(result) -> dict:
    cov_re = result.cov_re
    dataset_var = float(cov_re.iloc[0, 0]) if hasattr(cov_re, "iloc") else float(cov_re[0][0])
    return {"dataset_intercept_var": dataset_var, "residual_var": float(result.scale)}


def diagnostics(result, df: pd.DataFrame) -> dict:
    """Residual skew, Shapiro-Wilk normality, and Levene's test for equal
    residual variance across the 7 dataset groups."""
    resid = result.resid
    skew = float(scipy.stats.skew(resid))
    shapiro_stat, shapiro_p = scipy.stats.shapiro(resid)
    groups = [resid[df["dataset"] == d].to_numpy() for d in df["dataset"].unique()]
    levene_stat, levene_p = scipy.stats.levene(*groups)
    per_dataset_std = {
        d: float(resid[df["dataset"] == d].std(ddof=1)) for d in df["dataset"].unique()
    }
    return {
        "residual_skew": skew,
        "shapiro_stat": float(shapiro_stat), "shapiro_p": float(shapiro_p),
        "levene_stat": float(levene_stat), "levene_p": float(levene_p),
        "per_dataset_residual_std": per_dataset_std,
    }


def permutation_lrt(df: pd.DataFrame, covariate: str, reference: str, observed_lr: float,
                     reduced_llf: float, n_perm: int = 999, seed: int = 0) -> dict:
    """Non-parametric check on the LRT's chi-squared p-value: permute backbone
    labels within each dataset block (covariate and dataset are unchanged by
    this, so the reduced model's likelihood is unchanged too -- only the full
    model needs refitting per permutation), matching the Friedman test's own
    within-block permutation null (H0: no backbone effect). If the empirical
    permutation p-value agrees with the chi-squared p-value, the LRT's
    significance is not an artifact of the violated normality assumption."""
    rng = np.random.default_rng(seed)
    datasets = df["dataset"].unique()
    perm_lrs = []
    n_failed = 0
    for _ in range(n_perm):
        df_perm = df.copy()
        for d in datasets:
            idx = df_perm.index[df_perm["dataset"] == d]
            df_perm.loc[idx, "backbone"] = rng.permutation(df_perm.loc[idx, "backbone"].to_numpy())
        try:
            full_perm = fit_full_model(df_perm, covariate, reference)
        except Exception:
            n_failed += 1
            continue
        perm_lrs.append(2 * (full_perm.llf - reduced_llf))
    perm_lrs = np.array(perm_lrs)
    p_value = float((np.sum(perm_lrs >= observed_lr) + 1) / (len(perm_lrs) + 1))
    return {
        "n_perm": int(len(perm_lrs)), "n_failed": int(n_failed),
        "observed_lr": float(observed_lr), "p_value": p_value,
        "perm_lr_mean": float(perm_lrs.mean()), "perm_lr_95th": float(np.percentile(perm_lrs, 95)),
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def fmt_p(p: float) -> str:
    return f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}"


def term_label(name: str, covariate_key: str) -> str:
    if name == "Intercept":
        return "Intercept (reference backbone)"
    if name == covariate_key:
        return covariate_key
    if "[T." in name:
        backbone = name.split("[T.")[-1].rstrip("]")
        return BACKBONE_META.get(backbone, {}).get("display", backbone)
    return name


def build_fixed_effects_md(fe: dict, covariate_key: str) -> list:
    lines = ["| Term | Coef | SE | z | p |", "|---|---|---|---|---|"]
    for name, row in fe.items():
        label = term_label(name, covariate_key)
        sig = "*" if row["p_value"] < ALPHA else ""
        lines.append(f"| {label} | {row['coef']:.4f} | {row['se']:.4f} | {row['z']:.3f} | {fmt_p(row['p_value'])}{sig} |")
    return lines


def build_markdown(ctx: dict) -> str:
    reference = ctx["reference_backbone"]
    ref_label = BACKBONE_META.get(reference, {}).get("display", reference)
    ft = ctx["friedman"]
    lines = [
        "# RQ1 -- Linear mixed-effects model, backbone effect controlling for dataset difficulty",
        "",
        "Tests whether backbone identity predicts soundscape range_mae after controlling for "
        "per-dataset difficulty (polyphony density), extending `compute_rq1_friedman_test.py`'s "
        "rank-based omnibus result with an explicit covariate-adjusted model. This is a distinct "
        "question from the Friedman test -- not a re-test of \"do backbones differ,\" but \"do "
        "backbones differ once dataset difficulty is accounted for as a covariate rather than just "
        "a repeated-measures block.\"",
        "",
        "Same source data as the Friedman test: the 13 frozen-backbone, pooled-MLP-head, "
        f"regression-formulation models, source `{SCAPE_SOURCE}`, metric `{SOUNDSCAPE_METRIC}`, "
        f"across the 7 REGIONAL_DATASETS ({', '.join(REGIONAL_DATASETS)}). Per-dataset difficulty "
        "(ratio_polyp, mean_polyp) is the same polyphony_metric_summary source used in "
        "`compute_rq5_polyphony_robustness.py`.",
        "",
        "**Model**: `mae ~ backbone + <difficulty covariate> + (1|dataset)`, fit via "
        "`statsmodels.formula.api.mixedlm` with ML (not REML) estimation, so the fixed-effect "
        "likelihood-ratio test below is valid. backbone is a 13-level categorical fixed effect; "
        f"reference level = **{ref_label}** (the median-average-rank backbone from the Friedman "
        "test, i.e. a \"typical\" backbone rather than an arbitrary alphabetical first). dataset is "
        "a random intercept (7 groups).",
        "",
        "## Caveat -- small n for random-effect estimation",
        "",
        "n=7 datasets (random-effect groups) is small for a mixed model's random-effect variance to "
        "be well estimated. Treat the results below as exploratory, reported **alongside**, not as a "
        "replacement for, the Friedman test result "
        f"(chi-squared={ft['statistic']:.3f}, df={ft['df']}, p={ft['p_value']:.4f}, "
        f"{'rejects' if ft['p_value'] < ALPHA else 'does not reject'} the null at alpha={ALPHA}).",
        "",
    ]

    for key, covariate_label in DIFFICULTY_COVARIATES.items():
        model_ctx = ctx["models"][key]
        primary = key == "ratio_polyp"
        heading = "## Primary model" if primary else "## Robustness check"
        lines += [
            f"{heading}: `mae ~ backbone + {key} + (1|dataset)`",
            "",
            f"Difficulty covariate: **{covariate_label}** (`{key}`).",
            "",
            "### Fixed effects",
            "",
        ]
        lines += build_fixed_effects_md(model_ctx["fixed_effects"], key)
        lines += [
            "",
            f"Random intercept (dataset) variance = {model_ctx['random_effects']['dataset_intercept_var']:.6f}, "
            f"residual variance = {model_ctx['random_effects']['residual_var']:.6f}. "
            "`*` marks p < 0.05 relative to the reference backbone.",
            "",
            "### Likelihood-ratio test -- overall backbone effect",
            "",
            f"Full model (`backbone + {key}`) vs. reduced model (`{key}` only), both fit by ML: "
            f"**LR = {model_ctx['lrt']['lr_statistic']:.3f}, df = {model_ctx['lrt']['df']}, "
            f"p = {fmt_p(model_ctx['lrt']['p_value'])}**. At alpha = {ALPHA}, this "
            f"{'rejects' if model_ctx['lrt']['p_value'] < ALPHA else 'does not reject'} the null "
            f"that backbone has no effect on range_mae once {key} is controlled for.",
            "",
            "**Permutation check** (999 within-dataset permutations of backbone labels, matching "
            "the Friedman test's own within-block permutation null -- see \"Friedman vs. mixed-model "
            "comparison\" below): observed LR = "
            f"{model_ctx['permutation_lrt']['observed_lr']:.3f} vs. permutation null mean = "
            f"{model_ctx['permutation_lrt']['perm_lr_mean']:.3f} (95th pct = "
            f"{model_ctx['permutation_lrt']['perm_lr_95th']:.3f}); "
            f"empirical p = {fmt_p(model_ctx['permutation_lrt']['p_value'])} "
            f"({model_ctx['permutation_lrt']['n_perm']} valid permutations).",
            "",
        ]

    lines += [
        "## Friedman vs. mixed-model comparison",
        "",
    ]
    ratio_lrt_p = ctx["models"]["ratio_polyp"]["lrt"]["p_value"]
    mean_lrt_p = ctx["models"]["mean_polyp"]["lrt"]["p_value"]
    friedman_sig = ft["p_value"] < ALPHA
    mixed_sig = ratio_lrt_p < ALPHA
    if friedman_sig == mixed_sig:
        lines += [
            f"The mixed-model LRT (ratio_polyp: p={fmt_p(ratio_lrt_p)}, mean_polyp: p={fmt_p(mean_lrt_p)}) "
            f"agrees qualitatively with the Friedman omnibus test (p={ft['p_value']:.4f}) -- both "
            f"{'reject' if friedman_sig else 'fail to reject'} the null of no backbone effect at "
            f"alpha={ALPHA}. Controlling for difficulty as a covariate rather than only as a "
            "repeated-measures block does not change the qualitative conclusion here.",
        ]
    else:
        no_cov = ctx["no_covariate_lrt"]
        no_cov_sig = no_cov["p_value"] < ALPHA
        perm_p = ctx["models"]["ratio_polyp"]["permutation_lrt"]["p_value"]
        perm_agrees = (perm_p < ALPHA) == mixed_sig
        lines += [
            f"The mixed-model LRT (ratio_polyp: p={fmt_p(ratio_lrt_p)}, mean_polyp: p={fmt_p(mean_lrt_p)}) "
            f"**diverges** from the Friedman omnibus test (p={ft['p_value']:.4f}): Friedman "
            f"{'rejects' if friedman_sig else 'does not reject'} the null while the mixed model "
            f"{'rejects' if mixed_sig else 'does not reject'} it at alpha={ALPHA}. Two candidate "
            "explanations were checked directly, rather than just reporting the more favorable-"
            "looking result:",
            "",
            "1. **Is the covariate absorbing variance Friedman couldn't account for?** Refitting "
            "without any difficulty covariate at all (`backbone + (1|dataset)` vs. `(1|dataset)` "
            f"only) gives LR = {no_cov['lr_statistic']:.3f}, df = {no_cov['df']}, "
            f"p = {fmt_p(no_cov['p_value'])} -- backbone is "
            f"{'still' if no_cov_sig else 'no longer'} significant even **without** the difficulty "
            "covariate. " + (
                "This means the covariate is not what creates the significant backbone effect -- "
                "the mixed model detects it either way, so the divergence from Friedman is better "
                "explained by the mixed model's greater statistical power on the raw MAE values "
                "(a parametric model using magnitudes) versus Friedman's use of within-dataset ranks "
                "(which discard magnitude information) than by the covariate specifically."
                if no_cov_sig else
                "This means the difficulty covariate specifically is what makes backbone significant "
                "here -- without it, the mixed model agrees with Friedman's non-significant result, "
                "so the covariate is absorbing dataset-level variance that neither Friedman's "
                "rank-based test nor a covariate-free mixed model can attribute to backbone."
            ),
            "",
            "2. **Is the significant LRT an artifact of the violated normality assumption (see "
            "Diagnostics below)?** The permutation check above (999 within-dataset label "
            f"permutations, no distributional assumptions) gives empirical p = {fmt_p(perm_p)}, "
            f"which {'agrees' if perm_agrees else 'disagrees'} with the parametric LRT's "
            f"significance verdict. " + (
                "This means the parametric LRT's significance is not an artifact of non-normal "
                "residuals -- a distribution-free permutation null gives the same qualitative "
                "conclusion."
                if perm_agrees else
                "This means the parametric chi-squared LRT's significance does **not** survive under "
                "a distribution-free null -- the non-normal residuals flagged in Diagnostics below "
                "are plausibly inflating the parametric p-value, and the permutation result should be "
                "trusted over the chi-squared one here."
            ),
        ]
    lines.append("")

    lines += ["## Diagnostics", ""]
    for key, covariate_label in DIFFICULTY_COVARIATES.items():
        diag = ctx["models"][key]["diagnostics"]
        normal = diag["shapiro_p"] >= ALPHA
        equal_var = diag["levene_p"] >= ALPHA
        lines += [
            f"### {covariate_label} model (`{key}`)",
            "",
            f"- Residual skew = {diag['residual_skew']:.3f}",
            f"- Shapiro-Wilk normality: W = {diag['shapiro_stat']:.3f}, p = {fmt_p(diag['shapiro_p'])} "
            f"({'consistent with normal residuals' if normal else 'residuals deviate from normality'} "
            f"at alpha={ALPHA})",
            f"- Levene's test (equal residual variance across the 7 datasets): "
            f"stat = {diag['levene_stat']:.3f}, p = {fmt_p(diag['levene_p'])} "
            f"({'variance roughly equal across datasets' if equal_var else 'variance differs across datasets (heteroscedastic)'} "
            f"at alpha={ALPHA})",
            "- Per-dataset residual SD: " + ", ".join(
                f"{d}={v:.4f}" for d, v in diag["per_dataset_residual_std"].items()
            ),
            "",
        ]

    any_violation = any(
        ctx["models"][k]["diagnostics"]["shapiro_p"] < ALPHA or ctx["models"][k]["diagnostics"]["levene_p"] < ALPHA
        for k in DIFFICULTY_COVARIATES
    )
    lines += [
        "### Reading the diagnostics",
        "",
    ]
    if any_violation:
        lines += [
            "At least one of normality or equal-variance is flagged as violated above. Given the "
            "heteroscedasticity across soundscape datasets already documented elsewhere in this "
            "analysis series (datasets differ substantially in scale and difficulty), this is not "
            "surprising, and it means the model's parametric p-values (both the per-backbone "
            "z-tests and the LRT) should be read with caution -- they assume homoscedastic, "
            "normally distributed residuals that this data does not clearly satisfy. The "
            "directional/qualitative comparison against the Friedman test above is more trustworthy "
            "than the exact p-values from either test individually.",
        ]
    else:
        lines += [
            "Residuals are reasonably consistent with the mixed model's normality and "
            "equal-variance assumptions at alpha=0.05, so the parametric p-values above can be read "
            "with the usual (not extra) caution appropriate to n=7 random-effect groups.",
        ]

    lines += [
        "",
        "## Reading this",
        "",
        "- This model answers a different question than the Friedman test: it asks whether backbone "
        "identity explains variance in range_mae **after** difficulty is partialled out as a "
        "continuous covariate, rather than only treating dataset as a matched block.",
        "- n=7 datasets is small for reliably estimating a random-intercept variance; the fixed-effect "
        "estimates and LRT are the more informative outputs here, not the random-effect variance "
        "itself.",
        "- Report this analysis alongside, not in place of, `rq1_friedman_test.md` -- the two use "
        "different assumptions (rank-based/nonparametric vs. parametric with an explicit covariate) "
        "and a discrepancy between them is informative in itself, not a reason to prefer one.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq1"))
    args = parser.parse_args()

    models = mrt.discover_models(Path("archive/Pooled-Embeddings"))
    long_df_src = load_study("pooled", models=models)
    backbone_order = [m for m in BACKBONE_META if m in models]

    matrix = build_matrix(long_df_src, backbone_order)
    ft = friedman_test(matrix)
    matrix_complete = matrix.loc[ft["backbones_used"]]

    reference = pick_reference_backbone(matrix_complete)

    stats = json.loads(STATS_JSON.read_text())["subsets"]
    df = build_long_df(matrix_complete, stats)

    model_results = {}
    for key in DIFFICULTY_COVARIATES:
        full = fit_full_model(df, key, reference)
        reduced = fit_reduced_model(df, key)
        lrt = likelihood_ratio_test(full, reduced)
        model_results[key] = {
            "fixed_effects": fixed_effects_table(full),
            "random_effects": random_effects(full),
            "reduced_random_effects": random_effects(reduced),
            "lrt": lrt,
            "permutation_lrt": permutation_lrt(df, key, reference, lrt["lr_statistic"], reduced.llf),
            "diagnostics": diagnostics(full, df),
            "converged": bool(full.converged),
        }

    no_cov_full, no_cov_null = fit_no_covariate_models(df, reference)
    no_covariate_lrt = likelihood_ratio_test(no_cov_full, no_cov_null)

    ctx = {
        "reference_backbone": reference,
        "friedman": ft,
        "n_datasets": len(REGIONAL_DATASETS),
        "k_backbones": ft["k_backbones"],
        "models": model_results,
        "no_covariate_lrt": no_covariate_lrt,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq1_mixed_model_difficulty.json"
    out_json.write_text(json.dumps(ctx, indent=2, default=str))
    print(f"Wrote {out_json}")

    md = build_markdown(ctx)
    out_md = args.out_dir / "rq1_mixed_model_difficulty.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
