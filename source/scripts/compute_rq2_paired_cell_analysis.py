#!/usr/bin/env python3
"""
RQ2 (additional) -- Paired cell-level analysis of TC+auxiliary configurations
vs. the pooled-MLP baseline, from archive/Spatial-Embeddings/ (5 backbones,
TemporalCNN architecture, SPATIAL_BACKBONES in plot_rq2.py) against
archive/Pooled-Embeddings/'s plain pooled-embedding MLP head -- same source
data as plot_rq2.py's build_figure_combined()/build_dual_heatmap_figure(),
just aggregated into significance tests instead of figures.

Purpose: RQ2's narrative claims (best auxiliary objective differs by domain;
best overall configuration does not transfer; more backbones beat the
pooled-MLP baseline on soundscape than synthetic) are checked here the same
way the region-specific/frozen/fine-tuned comparison was checked in
compute_rq5_kurtosis.py -- are these patterns statistically robust, or does
n=5 backbones x up to 8 datasets make them indistinguishable from noise.

Five analyses, computed separately for synthetic (8 datasets, ALL_DATASETS)
and soundscape (7 datasets, REGIONAL_DATASETS):

1. Paired cell-level differences vs. pooled-MLP baseline (compute_deltas()'s
   own (config, model, dataset) rows, unaggregated): for the bare TC-head and
   each TC+auxiliary config, Delta = config MAE - pooled-MLP MAE, matched by
   (backbone, dataset) cell (n = 5 backbones x 8/7 datasets = 40/35). Mean +/-
   SD, paired t-test and Wilcoxon signed-rank test against zero, same
   convention as compute_rq5_kurtosis.py's cell_diff_stats()/paired_tests().

2. Per-backbone breakdown of the best-performing TC+auxiliary config's Delta:
   "best" per (backbone, domain) is defined by analysis 3 below (lowest mean
   MAE among the 4 auxiliary configs, AUX_CONFIGS) -- for each backbone, that
   config's own Delta vs. pooled-MLP, mean +/- SD and tests across that
   backbone's own 7/8 datasets (same small-n caveat as
   compute_rq5_kurtosis.py's per-backbone Delta_finetuned table).

3. Rank-based check, "best auxiliary changes by domain": per backbone, the
   auxiliary config (of the 4 in AUX_CONFIGS -- excluding the bare TC-head,
   which is not itself an auxiliary) with the lowest mean MAE on synthetic
   vs. on soundscape. Reported as a same/different count across the 5
   backbones, not a significance test -- this is a categorical/rank
   question, not a mean-difference one.

4. Rank-based check, "best overall configuration doesn't transfer": the
   single (backbone, config) pair -- among all 5 backbones x 5 configs in
   CONFIG_LABELS (including the bare TC-head, since this is the overall-best
   question, not the auxiliary-only one) -- with the lowest mean raw MAE on
   synthetic, and separately on soundscape. Each winning pair's rank (by mean
   MAE, 1 = best) in the *other* domain's ranking of all 25 pairs, to
   quantify how far it falls there.

5. Win-rate significance check: per domain, how many of the 5 backbones have
   their own best-auxiliary config (per analysis 3) beat the pooled-MLP
   baseline (mean Delta < 0, paired by dataset) -- and whether that count
   differs from chance via a two-sided exact binomial test against p=0.5.
   n=5 is acknowledged as very underpowered for this.

Sample-size caveat, same as compute_rq5_kurtosis.py: n=5 backbones and up to
7/8 datasets is small throughout. Every p-value here is exploratory, not
confirmatory -- point estimates and the rank-based findings (3, 4) are
reported in full even where the mean-based tests (1, 2, 5) are underpowered,
since a rank pattern (e.g. a reversal) can be a real, checkable fact about
the data independent of whether a t-test on the same values clears p<0.05.

Writes rq2_paired_cell_analysis.json and rq2_paired_cell_analysis.md to
--out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq2_paired_cell_analysis.py [--out-dir plots/figures/rq2]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backbone_meta import BACKBONE_META
from plot_data import load_study
from plot_rq2 import (
    CONFIG_HEAD, CONFIG_LABELS, CONFIG_TICK_LABEL, SCAPE_SOURCE, SOURCE,
    SOURCE_DATASETS, SPATIAL_BACKBONES, compute_baseline, compute_deltas,
)

AUX_CONFIGS = [c for c in CONFIG_LABELS if c != "+TC-head"]
DOMAIN_SOURCE = {"synthetic": SOURCE, "soundscape": SCAPE_SOURCE}
DOMAIN_LABEL = {"synthetic": "Synthetic", "soundscape": "Soundscape"}


def fmt_mean_std(mean: float | None, std: float | None, decimals: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def paired_tests(diff: np.ndarray) -> dict:
    """Paired t-test and Wilcoxon signed-rank test of diff against zero --
    same convention as compute_rq5_kurtosis.py's paired_tests(). None for
    either test below the sample size it needs, or if Wilcoxon raises (e.g.
    all-zero differences)."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2:
        return {"t_stat": None, "t_p": None, "wilcoxon_stat": None, "wilcoxon_p": None}
    t_stat, t_p = scipy.stats.ttest_1samp(diff, popmean=0.0)
    try:
        w_stat, w_p = scipy.stats.wilcoxon(diff)
    except ValueError:
        w_stat, w_p = None, None
    return {
        "t_stat": float(t_stat), "t_p": float(t_p),
        "wilcoxon_stat": float(w_stat) if w_stat is not None else None,
        "wilcoxon_p": float(w_p) if w_p is not None else None,
    }


def diff_stats(diff: np.ndarray) -> dict:
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    return {
        "mean": float(diff.mean()) if n else None,
        "std": float(diff.std(ddof=1)) if n > 1 else None,
        "n": n,
    }


def _fmt_test(tests: dict) -> str:
    if tests["t_p"] is None:
        return "--"
    return f"t-test p={tests['t_p']:.3f}; Wilcoxon p={tests['wilcoxon_p']:.3f}" if tests["wilcoxon_p"] is not None \
        else f"t-test p={tests['t_p']:.3f}; Wilcoxon n/a"


# ---------------------------------------------------------------------------
# Analysis 1 -- paired cell-level differences vs. pooled-MLP baseline, per
# config, per domain.
# ---------------------------------------------------------------------------

def compute_domain_deltas(pooled_long: pd.DataFrame, spatial_long: pd.DataFrame, source: str) -> pd.DataFrame:
    """One row per (config, model, dataset) cell: raw MAE, the pooled-MLP
    baseline MAE for that same (model, dataset), and their delta -- i.e.
    compute_deltas() against compute_baseline(), plot_rq2.py's own pooled-MLP
    reference point."""
    baseline = compute_baseline(pooled_long, source)
    baseline_indexed = baseline.set_index(["model", "dataset"])["value"]
    return compute_deltas(spatial_long, baseline_indexed, source)


def analysis1_cell_level(domain_deltas: pd.DataFrame) -> dict:
    results = {}
    for label in CONFIG_LABELS:
        diff = domain_deltas.loc[domain_deltas["config"] == label, "delta"].to_numpy(dtype=float)
        results[label] = {**diff_stats(diff), **paired_tests(diff)}
    return results


# ---------------------------------------------------------------------------
# Analysis 3 -- best auxiliary config per (backbone, domain), by lowest mean
# raw MAE across that domain's datasets, restricted to AUX_CONFIGS. Computed
# first since analyses 2 and 5 both consume its result.
# ---------------------------------------------------------------------------

def best_aux_per_backbone(domain_deltas: pd.DataFrame, backbone_order: list) -> dict:
    aux_rows = domain_deltas[domain_deltas["config"].isin(AUX_CONFIGS)]
    mean_mae = aux_rows.groupby(["model", "config"])["value"].mean()
    best = {}
    for model in backbone_order:
        per_config = mean_mae.loc[model]
        best[model] = per_config.idxmin()
    return best


def analysis3_rank_check(best_by_domain: dict, backbone_order: list) -> dict:
    per_backbone = {
        m: {"synthetic": best_by_domain["synthetic"][m], "soundscape": best_by_domain["soundscape"][m],
            "same": best_by_domain["synthetic"][m] == best_by_domain["soundscape"][m]}
        for m in backbone_order
    }
    n_same = sum(1 for v in per_backbone.values() if v["same"])
    return {"per_backbone": per_backbone, "n_same": n_same, "n_different": len(backbone_order) - n_same,
            "n_backbones": len(backbone_order)}


# ---------------------------------------------------------------------------
# Analysis 2 -- per-backbone Delta of each backbone's own best-auxiliary
# config (analysis 3), across that backbone's own datasets.
# ---------------------------------------------------------------------------

def analysis2_per_backbone_best_aux(domain_deltas: pd.DataFrame, best_by_domain: dict, backbone_order: list) -> dict:
    results = {}
    for model in backbone_order:
        label = best_by_domain[model]
        diff = domain_deltas.loc[(domain_deltas["config"] == label) & (domain_deltas["model"] == model),
                                  "delta"].to_numpy(dtype=float)
        results[model] = {"config": label, **diff_stats(diff), **paired_tests(diff)}
    return results


# ---------------------------------------------------------------------------
# Analysis 4 -- best overall (backbone, config) pair per domain (all 5
# CONFIG_LABELS, including the bare TC-head) and its rank in the other
# domain.
# ---------------------------------------------------------------------------

def overall_ranking(domain_deltas: pd.DataFrame, backbone_order: list) -> pd.Series:
    """All 5 backbones x 5 CONFIG_LABELS -> mean raw MAE, sorted ascending
    (lower = better), indexed by (model, config)."""
    mean_mae = domain_deltas.groupby(["model", "config"])["value"].mean()
    mean_mae = mean_mae.reindex(pd.MultiIndex.from_product([backbone_order, CONFIG_LABELS],
                                                             names=["model", "config"]))
    return mean_mae.sort_values()


def analysis4_best_overall_transfer(rankings: dict) -> dict:
    results = {}
    for domain, ranking in rankings.items():
        other_domain = "soundscape" if domain == "synthetic" else "synthetic"
        other_ranking = rankings[other_domain]
        best_pair = ranking.index[0]
        other_rank = int(other_ranking.index.get_loc(best_pair)) + 1  # 1-based
        results[domain] = {
            "best_pair": {"model": best_pair[0], "config": best_pair[1]},
            "best_mae": float(ranking.iloc[0]),
            "rank_in_other_domain": other_rank,
            "n_pairs_in_other_domain": len(other_ranking),
            "mae_in_other_domain": float(other_ranking.loc[best_pair]),
        }
    return results


# ---------------------------------------------------------------------------
# Analysis 5 -- win-rate significance check: N of 5 backbones whose own
# best-auxiliary config beats the pooled-MLP baseline, binomial test vs. 0.5.
# ---------------------------------------------------------------------------

def analysis5_win_rate(per_backbone_best_aux: dict, backbone_order: list) -> dict:
    wins = [m for m in backbone_order if per_backbone_best_aux[m]["mean"] is not None
            and per_backbone_best_aux[m]["mean"] < 0]
    n = len(backbone_order)
    k = len(wins)
    p_two_sided = float(scipy.stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    return {"n_backbones": n, "n_wins": k, "winners": wins,
            "p_two_sided_binomial": p_two_sided}


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def build_markdown(all_results: dict, backbone_order: list) -> str:
    n_datasets = {d: len(SOURCE_DATASETS[DOMAIN_SOURCE[d]]) for d in DOMAIN_SOURCE}
    lines = [
        "# RQ2 -- Paired cell-level analysis of TC+auxiliary configurations vs. pooled-MLP baseline",
        "",
        "Checks whether RQ2's narrative claims (best auxiliary objective differs by domain; best overall "
        "configuration does not transfer; more backbones beat the pooled-MLP baseline on soundscape than "
        "synthetic) are statistically robust, or -- as with the region-specific/frozen/fine-tuned comparison "
        "in `rq5_kurtosis.md` -- within the range of noise given small sample size and high per-cell "
        "variance. Source data: `archive/Spatial-Embeddings/` (5 backbones x TC-head configs) vs. "
        "`archive/Pooled-Embeddings/` (same 5 backbones, plain pooled-embedding MLP, no TC-head), the same "
        "pooled-MLP baseline used by `plot_rq2.py`'s dual heatmap and backbone-bars figures.",
        "",
        f"**Caveat on sample size**: n=5 backbones and up to {max(n_datasets.values())} datasets throughout -- "
        "the same small-n regime as `rq5_kurtosis.md`. Every significance test below (t-test, Wilcoxon, "
        "binomial) is exploratory, not confirmatory, especially the per-backbone (n=7/8) and win-rate (n=5) "
        "tests, where a single backbone or dataset can swing the result. Point estimates and the rank-based "
        "findings (checks 3-4) are reported in full even where the corresponding mean-based test isn't "
        "significant -- per the earlier kurtosis/paired-difference work, a rank pattern (e.g. which "
        "configuration wins) can be a real, checkable fact even when a t-test on the underlying values isn't "
        "significant, since they answer different kinds of questions (ordinal/categorical vs. continuous "
        "mean difference).",
        "",
    ]

    # --- Analysis 1 ---
    lines += [
        "## 1. Paired cell-level differences vs. pooled-MLP baseline",
        "",
        "Delta = (TC-head or TC+auxiliary MAE) - (pooled-MLP MAE), matched by (backbone, dataset) cell "
        "(n = 5 backbones x datasets per domain). Negative = improvement over pooled-MLP.",
        "",
    ]
    for domain in ("synthetic", "soundscape"):
        r = all_results[domain]["analysis1"]
        lines += [
            f"### {DOMAIN_LABEL[domain]} (n = 5 x {n_datasets[domain]} = {5 * n_datasets[domain]} cells per config)",
            "",
            "| Config | Mean Delta MAE (+/- SD) | n | Paired t-test | Wilcoxon |",
            "|---|---|---|---|---|",
        ]
        for label in CONFIG_LABELS:
            s = r[label]
            t_p = f"p={s['t_p']:.3f}" if s['t_p'] is not None else "--"
            w_p = f"p={s['wilcoxon_p']:.3f}" if s['wilcoxon_p'] is not None else "--"
            lines.append(f"| {label} | {fmt_mean_std(s['mean'], s['std'])} | {s['n']} | {t_p} | {w_p} |")
        lines.append("")

    # --- Analysis 2 ---
    lines += [
        "## 2. Per-backbone Delta of that backbone's own best-auxiliary configuration",
        "",
        "\"Best\" per (backbone, domain) = the auxiliary configuration (of the 4 TC+auxiliary variants, "
        "excluding the bare TC-head) with the lowest mean MAE for that backbone in that domain -- see "
        "check 3 below. This table is that config's own paired Delta vs. pooled-MLP, across the backbone's "
        "own datasets (n=7 or 8). Same small-n caveat as `rq5_kurtosis.md`'s per-backbone Delta_finetuned "
        "table.",
        "",
    ]
    for domain in ("synthetic", "soundscape"):
        r = all_results[domain]["analysis2"]
        lines += [
            f"### {DOMAIN_LABEL[domain]} (n={n_datasets[domain]} datasets per backbone)",
            "",
            "| Backbone | Best auxiliary config | Mean Delta MAE (+/- SD) | n | Paired t-test | Wilcoxon |",
            "|---|---|---|---|---|---|",
        ]
        for m in backbone_order:
            s = r[m]
            t_p = f"p={s['t_p']:.3f}" if s['t_p'] is not None else "--"
            w_p = f"p={s['wilcoxon_p']:.3f}" if s['wilcoxon_p'] is not None else "--"
            lines.append(f"| {BACKBONE_META[m]['display']} | {s['config']} | {fmt_mean_std(s['mean'], s['std'])} "
                          f"| {s['n']} | {t_p} | {w_p} |")
        lines.append("")

    # --- Analysis 3 ---
    lines += [
        "## 3. Rank-based check -- does the best auxiliary configuration change by domain?",
        "",
        "Per backbone: the auxiliary configuration (of the 4 TC+auxiliary variants) with the lowest mean "
        "MAE on synthetic vs. on soundscape. A categorical/rank comparison, reported as a count/proportion, "
        "not a significance test.",
        "",
        "| Backbone | Best aux (synthetic) | Best aux (soundscape) | Same? |",
        "|---|---|---|---|",
    ]
    rc = all_results["rank_check_aux"]
    for m in backbone_order:
        pb = rc["per_backbone"][m]
        same = "yes" if pb["same"] else "no"
        lines.append(f"| {BACKBONE_META[m]['display']} | {pb['synthetic']} | {pb['soundscape']} | {same} |")
    lines += [
        "",
        f"**{rc['n_same']} of {rc['n_backbones']} backbones** have the same best auxiliary configuration in "
        f"both domains; **{rc['n_different']} of {rc['n_backbones']}** differ.",
        "",
    ]

    # --- Analysis 4 ---
    lines += [
        "## 4. Rank-based check -- does the best overall configuration transfer across domains?",
        "",
        "Single best (backbone, configuration) pair -- among all 5 backbones x 5 configs (TC-head alone + "
        "4 auxiliary variants) -- per domain, by lowest mean raw MAE. Its rank (1st = best) in the *other* "
        "domain's ranking of the same 25 pairs.",
        "",
        "| Domain | Best pair | MAE there | Rank in other domain | MAE in other domain |",
        "|---|---|---|---|---|",
    ]
    ac = all_results["rank_check_overall"]
    for domain in ("synthetic", "soundscape"):
        e = ac[domain]
        pair_label = f"{BACKBONE_META[e['best_pair']['model']]['display']} / {e['best_pair']['config']}"
        lines.append(f"| {DOMAIN_LABEL[domain]} | {pair_label} | {e['best_mae']:.3f} "
                      f"| {e['rank_in_other_domain']} of {e['n_pairs_in_other_domain']} "
                      f"| {e['mae_in_other_domain']:.3f} |")
    lines.append("")

    # --- Analysis 5 ---
    lines += [
        "## 5. Win-rate significance check",
        "",
        "For each domain: of the 5 backbones, how many have their own best-auxiliary configuration "
        "(check 2/3) beat the pooled-MLP baseline (mean Delta < 0, paired by dataset)? Two-sided exact "
        "binomial test against p=0.5. n=5 is very underpowered for this.",
        "",
        "| Domain | Wins | n | Winners | Binomial p (two-sided) |",
        "|---|---|---|---|---|",
    ]
    for domain in ("synthetic", "soundscape"):
        wr = all_results[domain]["analysis5"]
        winners = ", ".join(BACKBONE_META[m]["display"] for m in wr["winners"]) or "none"
        lines.append(f"| {DOMAIN_LABEL[domain]} | {wr['n_wins']} | {wr['n_backbones']} | {winners} "
                      f"| {wr['p_two_sided_binomial']:.3f} |")
    lines.append("")

    lines += [
        "## Reading this",
        "",
        "- Check 1 is the broadest and best-powered test here (35-40 cells per config per domain) -- a "
        "significant result there is the most trustworthy signal in this file.",
        "- Checks 2 and 5 operate at n=7/8 and n=5 respectively; treat their p-values as directional only.",
        "- Checks 3 and 4 are not significance tests by construction -- they quantify the rank-based claims "
        "directly (how many backbones agree on the best auxiliary; how far the best overall pair falls in "
        "the other domain) rather than forcing an ordinal question into a mean-difference framework.",
        "- A rank-based finding (e.g. check 4's domain-specific best pair falling well down the ranking "
        "elsewhere) can be real and worth reporting even when check 1's aggregate test for that same "
        "configuration doesn't clear significance -- they describe different things (which option wins vs. "
        "whether the average gap is distinguishable from zero).",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq2"))
    args = parser.parse_args()

    pooled_long = load_study("pooled", models=SPATIAL_BACKBONES)
    spatial_long = load_study("spatial", models=SPATIAL_BACKBONES)
    backbone_order = [m for m in BACKBONE_META if m in SPATIAL_BACKBONES]

    domain_deltas = {d: compute_domain_deltas(pooled_long, spatial_long, DOMAIN_SOURCE[d]) for d in DOMAIN_SOURCE}
    best_by_domain = {d: best_aux_per_backbone(domain_deltas[d], backbone_order) for d in DOMAIN_SOURCE}
    rankings = {d: overall_ranking(domain_deltas[d], backbone_order) for d in DOMAIN_SOURCE}

    all_results = {}
    for d in DOMAIN_SOURCE:
        analysis1 = analysis1_cell_level(domain_deltas[d])
        analysis2 = analysis2_per_backbone_best_aux(domain_deltas[d], best_by_domain[d], backbone_order)
        analysis5 = analysis5_win_rate(analysis2, backbone_order)
        all_results[d] = {"analysis1": analysis1, "analysis2": analysis2, "analysis5": analysis5}
    all_results["rank_check_aux"] = analysis3_rank_check(best_by_domain, backbone_order)
    all_results["rank_check_overall"] = analysis4_best_overall_transfer(rankings)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq2_paired_cell_analysis.json"
    out_json.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"Wrote {out_json}")

    md = build_markdown(all_results, backbone_order)
    out_md = args.out_dir / "rq2_paired_cell_analysis.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
