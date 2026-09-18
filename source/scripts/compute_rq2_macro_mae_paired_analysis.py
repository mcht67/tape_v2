#!/usr/bin/env python3
"""RQ2 -- paired significance analysis of TC+auxiliary macro-MAE(1-6) deltas,
soundscape, against both reference points the manuscript paragraph compares
against: the pooled-MLP baseline and the bare TC-head alone.

Purpose: the manuscript's soundscape RQ2 paragraph makes several claims
against ordinary (pooled, level-unbalanced) range_mae -- "every TC+auxiliary
configuration is statistically indistinguishable from the pooled-MLP
baseline (p>0.4)", "best auxiliary differs by backbone", "improvements over
the plain TC-head [don't] reach significance", "beats the pooled-MLP
baseline for four of five backbones". This script reruns the same paired
structure (same pairing convention, same test functions) with macro-MAE(1-6)
in place of range_mae, to check whether those specific claims still hold
under the level-balanced metric.

Reuses:
- compute_rq2_per_level_accuracy.py's build_region_macro_mae_table() for
  the Spatial-Embeddings (TC-head) side's (backbone, config, region)
  macro-MAE(1-6) values -- recovered via that module's run-directory
  matching (see its docstring), same 5 SPATIAL_BACKBONES x 5 CONFIG_LABELS
  x 7 REGIONAL_DATASETS.
- That same module's _region_macro_mae_1_6(), applied directly (no
  run-directory recovery needed) to archive/Pooled-Embeddings/{backbone}/
  scape_eval_results/ for the pooled-MLP baseline's own per-region
  macro-MAE(1-6) -- unlike Spatial-Embeddings, Pooled-Embeddings has
  exactly one (non-multi-task) config per backbone, so its canonical
  scape_eval_results already holds that config's raw predictions intact.
- compute_rq2_paired_cell_analysis.py's diff_stats()/paired_tests() (paired
  t-test + Wilcoxon signed-rank vs. zero) and AUX_CONFIGS (the 4 auxiliary
  configs, excluding the bare TC-head) -- unchanged, applied to macro-MAE
  deltas instead of ordinary range_mae deltas.

Two reference points, each with a pooled (n=5 backbones x 7 regions=35) and
per-backbone (n=7) breakdown, mirroring compute_rq2_paired_cell_analysis.py's
own structure:
  A. vs. pooled-MLP baseline, paired by (backbone, region).
  B. vs. bare TC-head alone, paired by (backbone, region).

Also reports, under macro-MAE: the best auxiliary config per backbone (mean
macro-MAE(1-6) across the 7 regions, among the 4 AUX_CONFIGS), and whether
that backbone's best TC+auxiliary configuration beats its own pooled-MLP
baseline mean macro-MAE(1-6) ("nominally", i.e. without a significance
claim, same wording as the manuscript paragraph).

Sample-size caveat, same as compute_rq2_paired_cell_analysis.py: n=5
backbones and 7 regions throughout -- every p-value here is exploratory,
not confirmatory.

Writes rq2_macro_mae_paired_analysis.json and .md to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq2_macro_mae_paired_analysis.py [--out-dir plots/figures/rq2]
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backbone_meta import BACKBONE_META
from compute_rq2_paired_cell_analysis import AUX_CONFIGS, diff_stats, fmt_mean_std, paired_tests
from compute_rq2_per_level_accuracy import REGIONAL_DATASETS, _region_macro_mae_1_6, build_region_macro_mae_table
from plot_rq2 import CONFIG_LABELS, SPATIAL_BACKBONES

REPO_ROOT = Path(__file__).resolve().parents[2]
POOLED_ARCHIVE = REPO_ROOT / "archive" / "Pooled-Embeddings"


def build_pooled_mlp_region_table() -> pd.DataFrame:
    """Per-(backbone, region) macro-MAE(1-6) for the pooled-MLP baseline --
    same construction as build_region_macro_mae_table(), but no run-
    directory recovery is needed (see module docstring)."""
    rows = []
    for backbone in SPATIAL_BACKBONES:
        model_dir = POOLED_ARCHIVE / backbone
        for region in REGIONAL_DATASETS:
            path = model_dir / "scape_eval_results" / f"{region}_reg_scape_test_results.pkl"
            if not path.exists():
                rows.append({"backbone": backbone, "region": region, "macro_mae_1_6": float("nan"), "n_levels_used": 0})
                continue
            df = pickle.load(open(path, "rb"))
            macro_mae, n_levels = _region_macro_mae_1_6(df)
            rows.append({"backbone": backbone, "region": region, "macro_mae_1_6": macro_mae, "n_levels_used": n_levels})
    return pd.DataFrame(rows)


def build_deltas_vs_pooled_mlp(spatial_region_table: pd.DataFrame, pooled_region_table: pd.DataFrame) -> pd.DataFrame:
    baseline = pooled_region_table.set_index(["backbone", "region"])["macro_mae_1_6"]
    rows = []
    for _, r in spatial_region_table.iterrows():
        key = (r["backbone"], r["region"])
        if key not in baseline.index:
            continue
        rows.append({"config": r["config"], "model": r["backbone"], "dataset": r["region"],
                     "delta": r["macro_mae_1_6"] - baseline.loc[key]})
    return pd.DataFrame(rows)


def build_deltas_vs_tc_alone(spatial_region_table: pd.DataFrame) -> pd.DataFrame:
    tc_alone = spatial_region_table[spatial_region_table["config"] == "+TC-head"].set_index(["backbone", "region"])["macro_mae_1_6"]
    rows = []
    for _, r in spatial_region_table[spatial_region_table["config"] != "+TC-head"].iterrows():
        key = (r["backbone"], r["region"])
        rows.append({"config": r["config"], "model": r["backbone"], "dataset": r["region"],
                     "delta": r["macro_mae_1_6"] - tc_alone.loc[key]})
    return pd.DataFrame(rows)


def pooled_stats(deltas: pd.DataFrame) -> dict:
    results = {}
    for label in AUX_CONFIGS:
        diff = deltas.loc[deltas["config"] == label, "delta"].to_numpy(dtype=float)
        results[label] = {**diff_stats(diff), **paired_tests(diff)}
    return results


def per_backbone_stats(deltas: pd.DataFrame, backbone_order: list) -> dict:
    results = {}
    for label in AUX_CONFIGS:
        results[label] = {}
        for model in backbone_order:
            diff = deltas.loc[(deltas["config"] == label) & (deltas["model"] == model), "delta"].to_numpy(dtype=float)
            results[label][model] = {**diff_stats(diff), **paired_tests(diff)}
    return results


def best_auxiliary_per_backbone(spatial_region_table: pd.DataFrame, backbone_order: list) -> dict:
    """Per backbone: the AUX_CONFIGS entry with the lowest mean macro-
    MAE(1-6) across the 7 regions -- same "best auxiliary" convention as
    compute_rq2_paired_cell_analysis.py's analysis 3, applied to macro-MAE."""
    means = spatial_region_table[spatial_region_table["config"] != "+TC-head"].groupby(
        ["backbone", "config"])["macro_mae_1_6"].mean()
    result = {}
    for model in backbone_order:
        sub = means.loc[model]
        result[model] = {"best_config": sub.idxmin(), "best_mean_macro_mae": float(sub.min()),
                          "all_means": sub.to_dict()}
    return result


def build_plain_table(spatial_region_table: pd.DataFrame, pooled_region_table: pd.DataFrame,
                       backbone_order: list) -> pd.DataFrame:
    """Plain (non-paired) macro-MAE(1-6) per (backbone, config) cell --
    mean +/- SD +/- n across the 7 REGIONAL_DATASETS, region-balanced (same
    convention as build_region_macro_mae_table()/plot_rq1.py's rewritten
    compute_soundscape_per_level_accuracy.py, not the earlier pooled-across-
    regions convention rq2_per_level_accuracy.csv/md used). Includes the
    pooled-MLP baseline (no TC-head) as its own row per backbone, for direct
    comparison against the 5 TC-head configs."""
    stats = spatial_region_table.groupby(["backbone", "config"])["macro_mae_1_6"].agg(["mean", "std", "count"])
    pooled_stats_by_backbone = pooled_region_table.groupby("backbone")["macro_mae_1_6"].agg(["mean", "std", "count"])

    rows = []
    for model in backbone_order:
        p = pooled_stats_by_backbone.loc[model]
        rows.append({"backbone": model, "config": "pooled-MLP (no TC-head)",
                     "mean": float(p["mean"]), "std": float(p["std"]), "n": int(p["count"])})
        for cfglabel in CONFIG_LABELS:
            s = stats.loc[(model, cfglabel)]
            rows.append({"backbone": model, "config": cfglabel,
                         "mean": float(s["mean"]), "std": float(s["std"]), "n": int(s["count"])})
    return pd.DataFrame(rows)


def best_overall_configuration(spatial_region_table: pd.DataFrame) -> dict:
    means = spatial_region_table.groupby("config")["macro_mae_1_6"].mean()
    return {"best_config": means.idxmin(), "means": means.to_dict()}


def nominal_wins_vs_pooled_mlp(spatial_region_table: pd.DataFrame, pooled_region_table: pd.DataFrame,
                                best_aux: dict, backbone_order: list) -> dict:
    """Per backbone: does that backbone's own best TC+auxiliary config's
    mean macro-MAE(1-6) come in below (nominally, no significance claim)
    its pooled-MLP baseline's own mean macro-MAE(1-6)."""
    pooled_means = pooled_region_table.groupby("backbone")["macro_mae_1_6"].mean()
    result = {}
    for model in backbone_order:
        best_mean = best_aux[model]["best_mean_macro_mae"]
        baseline_mean = float(pooled_means.loc[model])
        result[model] = {"best_config": best_aux[model]["best_config"], "best_mean_macro_mae": best_mean,
                          "pooled_mlp_mean_macro_mae": baseline_mean, "nominal_win": best_mean < baseline_mean}
    return result


def fmt_p(p) -> str:
    return "n/a" if p is None else (f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}")


def label(m: str) -> str:
    return BACKBONE_META.get(m, {}).get("display", m)


def build_markdown(ctx: dict) -> str:
    backbone_order = ctx["backbone_order"]
    lines = [
        "# RQ2 -- Paired significance analysis of TC+auxiliary macro-MAE(1-6) deltas (soundscape)",
        "",
        "Reruns the manuscript's soundscape RQ2 paragraph's paired structure (pooled-MLP baseline and "
        "bare-TC-head-alone comparisons, `compute_rq2_paired_cell_analysis.py`'s `diff_stats()`/`paired_"
        "tests()`) with macro-MAE(1-6) (`compute_rq2_per_level_accuracy.py`'s per-region macro-MAE, "
        "level-balanced over polyphony levels 1-6, unambiguous-label subset) in place of ordinary pooled "
        "`range_mae`, to check whether the paragraph's specific claims still hold under the level-balanced "
        "metric. n=5 SPATIAL_BACKBONES x 7 REGIONAL_DATASETS = 35 paired cells per config (pooled); n=7 "
        "per backbone.",
        "",
        "## 0. Plain macro-MAE(1-6) results (not paired/delta)",
        "",
        "Mean +/- SD across the 7 REGIONAL_DATASETS, region-balanced (per-region macro-MAE(1-6) first, then "
        "averaged across regions -- same convention as `compute_soundscape_per_level_accuracy.py`'s own "
        "rewrite, NOT the earlier pooled-across-regions convention `rq2_per_level_accuracy.csv`/`.md` used, "
        "which is region-size-biased -- see that module's docstring). Includes the pooled-MLP baseline (no "
        "TC-head) as its own row per backbone.",
        "",
        "| Backbone | Config | Mean macro-MAE(1-6) (+/- SD) | n regions |",
        "|---|---|---|---|",
    ]
    for _, r in ctx["plain_table"].iterrows():
        lines.append(f"| {label(r['backbone'])} | {r['config']} | {fmt_mean_std(r['mean'], r['std'])} | {r['n']} |")
    lines += ["",
        "## 1. vs. pooled-MLP baseline",
        "",
        "Delta = TC+auxiliary macro-MAE(1-6) - pooled-MLP macro-MAE(1-6), paired by (backbone, region). "
        "Negative = the configuration improves on the pooled-MLP baseline.",
        "",
        "### Pooled across all 5 backbones (n=35 per config)",
        "",
        "| Config | Mean Delta (+/- SD) | Paired t-test $p$ | Wilcoxon $p$ |",
        "|---|---|---|---|",
    ]
    for cfglabel in AUX_CONFIGS:
        s = ctx["pooled_vs_baseline"][cfglabel]
        lines.append(f"| {cfglabel} | {fmt_mean_std(s['mean'], s['std'])} | {fmt_p(s['t_p'])} | {fmt_p(s['wilcoxon_p'])} |")

    lines += ["", "### Per backbone (n=7)", ""]
    for cfglabel in AUX_CONFIGS:
        lines += [f"**{cfglabel}**", "", "| Backbone | Mean Delta (+/- SD) | Paired t-test $p$ | Wilcoxon $p$ |", "|---|---|---|---|"]
        for model in backbone_order:
            s = ctx["per_backbone_vs_baseline"][cfglabel][model]
            lines.append(f"| {label(model)} | {fmt_mean_std(s['mean'], s['std'])} | {fmt_p(s['t_p'])} | {fmt_p(s['wilcoxon_p'])} |")
        lines.append("")

    lines += [
        "## 2. vs. bare TC-head alone",
        "",
        "Delta = TC+auxiliary macro-MAE(1-6) - bare-TC-head macro-MAE(1-6), paired by (backbone, region). "
        "Negative = the auxiliary configuration improves on the bare TC-head.",
        "",
        "### Pooled across all 5 backbones (n=35 per config)",
        "",
        "| Config | Mean Delta (+/- SD) | Paired t-test $p$ | Wilcoxon $p$ |",
        "|---|---|---|---|",
    ]
    for cfglabel in AUX_CONFIGS:
        s = ctx["pooled_vs_tc_alone"][cfglabel]
        lines.append(f"| {cfglabel} | {fmt_mean_std(s['mean'], s['std'])} | {fmt_p(s['t_p'])} | {fmt_p(s['wilcoxon_p'])} |")

    lines += ["", "### Per backbone (n=7)", ""]
    for cfglabel in AUX_CONFIGS:
        lines += [f"**{cfglabel}**", "", "| Backbone | Mean Delta (+/- SD) | Paired t-test $p$ | Wilcoxon $p$ |", "|---|---|---|---|"]
        for model in backbone_order:
            s = ctx["per_backbone_vs_tc_alone"][cfglabel][model]
            lines.append(f"| {label(model)} | {fmt_mean_std(s['mean'], s['std'])} | {fmt_p(s['t_p'])} | {fmt_p(s['wilcoxon_p'])} |")
        lines.append("")

    lines += [
        "## 3. Best auxiliary per backbone / best overall configuration (macro-MAE)",
        "",
        "| Backbone | Best auxiliary (macro-MAE) | Mean macro-MAE(1-6) | Pooled-MLP mean macro-MAE(1-6) | Nominal win vs. pooled-MLP? |",
        "|---|---|---|---|---|",
    ]
    for model in backbone_order:
        w = ctx["nominal_wins"][model]
        lines.append(f"| {label(model)} | {w['best_config']} | {w['best_mean_macro_mae']:.3f} "
                      f"| {w['pooled_mlp_mean_macro_mae']:.3f} | {'yes' if w['nominal_win'] else 'NO'} |")
    n_wins = sum(1 for m in backbone_order if ctx["nominal_wins"][m]["nominal_win"])
    lines += [
        "",
        f"**{n_wins} of {len(backbone_order)} backbones** have their own best TC+auxiliary configuration "
        "nominally beat their pooled-MLP baseline under macro-MAE(1-6) (no significance claim -- point "
        "estimate only, same convention as the manuscript paragraph's \"nominally beats\" wording).",
        "",
        f"**Best overall configuration** (mean macro-MAE(1-6) across all 5 backbones): "
        f"**{ctx['overall_best']['best_config']}** -- means: "
        + ", ".join(f"{c}: {v:.3f}" for c, v in ctx["overall_best"]["means"].items()) + ".",
        "",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq2"))
    args = parser.parse_args()

    backbone_order = [m for m in BACKBONE_META if m in SPATIAL_BACKBONES]

    spatial_region_table = build_region_macro_mae_table()
    pooled_region_table = build_pooled_mlp_region_table()

    deltas_vs_baseline = build_deltas_vs_pooled_mlp(spatial_region_table, pooled_region_table)
    deltas_vs_tc_alone = build_deltas_vs_tc_alone(spatial_region_table)

    pooled_vs_baseline = pooled_stats(deltas_vs_baseline)
    per_backbone_vs_baseline = per_backbone_stats(deltas_vs_baseline, backbone_order)
    pooled_vs_tc_alone = pooled_stats(deltas_vs_tc_alone)
    per_backbone_vs_tc_alone = per_backbone_stats(deltas_vs_tc_alone, backbone_order)

    best_aux = best_auxiliary_per_backbone(spatial_region_table, backbone_order)
    overall_best = best_overall_configuration(spatial_region_table)
    nominal_wins = nominal_wins_vs_pooled_mlp(spatial_region_table, pooled_region_table, best_aux, backbone_order)

    plain_table = build_plain_table(spatial_region_table, pooled_region_table, backbone_order)

    ctx = {
        "backbone_order": backbone_order, "plain_table": plain_table,
        "pooled_vs_baseline": pooled_vs_baseline, "per_backbone_vs_baseline": per_backbone_vs_baseline,
        "pooled_vs_tc_alone": pooled_vs_tc_alone, "per_backbone_vs_tc_alone": per_backbone_vs_tc_alone,
        "best_aux": best_aux, "overall_best": overall_best, "nominal_wins": nominal_wins,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq2_macro_mae_paired_analysis.json"
    out_json.write_text(json.dumps(ctx, indent=2, default=str))
    print(f"Wrote {out_json}")

    md = build_markdown(ctx)
    out_md = args.out_dir / "rq2_macro_mae_paired_analysis.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
