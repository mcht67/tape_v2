#!/usr/bin/env python3
"""RQ2 -- Per-polyphony-level accuracy/MAE breakdown for the 5 TC(+auxiliary)
configurations, soundscape test, by backbone.

Purpose: same distortion check as compute_soundscape_per_level_accuracy.py
(RQ1's plain frozen-backbone comparison), applied to RQ2's TC-head
configurations instead: the paper's headline soundscape MAE/Accuracy for
each (backbone, config) cell is pooled across every clip, including
polyphony=0 (the dominant class by far). A configuration that mostly just
gets better at presence/absence (levels 0-1) rather than at discriminating
polyphony degree can still look like the "best auxiliary" or "best overall
configuration" on the pooled metric. This script breaks accuracy/MAE out by
true polyphony level (same unambiguous-label-subset, exact-match convention
as compute_soundscape_per_level_accuracy.py) for every (backbone, config)
cell, and re-derives "best auxiliary per backbone" / "best overall
configuration" under macro-averaged, level-balanced metrics to check whether
they still agree with the currently-reported (pooled-MAE) ranking.

Data-recovery methodology (the hard part of this file)
--------------------------------------------------------
archive/Spatial-Embeddings/{backbone}/scape_eval_results/ -- the canonical,
already-merged location plot_rq2.py/compute_rq2_*.py read their MAE/Accuracy
numbers from -- only stores ONE config's raw per-clip predictions per
backbone at a time: each config's soundscape-eval DVC run writes to the same
fixed {region}_reg_scape_test_results.pkl path, so a later config's run
silently overwrites an earlier config's raw predictions there. Only the
merged soundscape_test_metrics.csv survives intact across configs, because
it is updated one column at a time (utils.evaluation.update_metrics_table)
rather than overwritten wholesale. Confirmed directly: every backbone's
currently-live pooled reg pkl carries prediction columns for *both*
event_logits and framewise_polyphony_reg, which only the
"+TC-head+both-aux" config's model produces -- so that is the one config
per backbone this script does NOT need to recover.

For the other 4 configs, this script recovers per-clip predictions the same
way plot_confusion_matrix_perchv2_polclass.py already did for one (backbone,
config) cell (Perch v2, TC+Pol-Class): archive/Spatial-Embeddings/{backbone}/
holds one training-run directory per (config, region) pair (confirmed: each
run's logs/params.yaml objectives keys are an exact, disjoint signature for
one of the 5 configs -- REG_OBJECTIVES below -- and cfg.dataset.subset
names the region), and that per-run directory's own logs/test_results.pkl
is never overwritten by a later run, so it still holds that (config, region)
cell's own raw predictions. Some backbones (EfficientNet-B1, NatureLM-audio,
Perch v2 -- confirmed by enumerating objectives signatures across every run
directory) have leftover duplicate runs per (config, region) from an earlier
hyperparameter search; these are disambiguated by matching each candidate
run's logs/test_metrics.json polyphony_reg.overall.range_mae against
soundscape_test_metrics.csv's own {region}_{head} range_mae value (same
technique plot_confusion_matrix_perchv2_polclass.py used), taking the
closest match. AudioProtoPNet and Bird-MAE have no duplicates (exactly one
run per (config, region)) and needed no disambiguation.

XCM-subset runs are dropped throughout (no soundscape ground truth for XCM,
see evaluate_on_soundscape_data.py) -- REGIONAL_DATASETS, 7 regions, same as
compute_soundscape_per_level_accuracy.py and every other RQ2 script.

Per-level breakdown and macro-averaging: identical methodology to
compute_soundscape_per_level_accuracy.py (unamb subset = min_polyphony ==
max_polyphony, exact-match round(pred) == true_level, macro_acc/mae_0_6 and
_1_6 as unweighted means over levels 0-6 / 1-6) -- that module's
per_level_accuracy() is reused directly, not reimplemented.

"Official" ranking: mean range_mae across the 7 regions (lower is better) --
the same aggregate quantity plot_rq2.py's figures and compute_rq2_paired_
cell_analysis.py's "best auxiliary"/"best overall configuration" checks are
already based on (not range_accuracy, which is RQ1's own official metric --
RQ2's manuscript claims are MAE claims).

Writes plots/figures/rq2/rq2_per_level_accuracy.csv (25 backbone x config
rows) and rq2_per_level_accuracy.md (per-backbone-best-auxiliary +
overall-best-configuration ranking tables) to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq2_per_level_accuracy.py [--out-dir plots/figures/rq2]
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backbone_meta import BACKBONE_META
# per_region_stats() replaced per_level_accuracy() when compute_soundscape_
# per_level_accuracy.py was rewritten to region-balance its own metrics
# (per-region first, then averaged across the 7 regions, rather than pooling
# every region's clips together -- see that module's docstring for why
# pooling was biased). Its output dict is a superset of the old per_level_
# accuracy()'s (same acc_*/mae_*/n_* keys plus macro_acc/mae already built
# in), so it's a drop-in replacement here under its old name.
from compute_soundscape_per_level_accuracy import MACRO_LEVEL, MAX_LEVEL, per_region_stats as per_level_accuracy
from plot_data import REGIONAL_DATASETS
from plot_rq2 import CONFIG_HEAD, CONFIG_LABELS, SPATIAL_BACKBONES

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = REPO_ROOT / "archive" / "Spatial-Embeddings"

# Each config's exact objectives-key signature, as found in a run's own
# logs/params.yaml `objectives` mapping -- disjoint across all 5 configs,
# confirmed by enumerating every run directory's signature for all 5
# backbones (see module docstring).
REG_OBJECTIVES = {
    "+TC-head": frozenset({"polyphony_reg"}),
    "+TC-head+polyphony classification": frozenset({"polyphony_reg", "polyphony_class"}),
    "+TC-head+frame-level polyphony": frozenset({"polyphony_reg", "framewise_polyphony_reg"}),
    "+TC-head+frame-level call activity": frozenset({"polyphony_reg", "event_logits"}),
    "+TC-head+both-aux": frozenset({"polyphony_reg", "framewise_polyphony_reg", "event_logits"}),
}
# soundscape_test_metrics.csv stores range_mae rounded to 4 decimals; the
# true underlying value differs from the rounded one by at most 5e-5, so any
# real (non-duplicate) match should land well under this.
MATCH_TOLERANCE = 0.001


def scan_run_dirs(backbone_dir: Path) -> dict:
    """(objectives_signature, region) -> [run_dir names], scanning every
    run's logs/params.yaml. XCM-subset runs are dropped (no soundscape
    ground truth for XCM)."""
    candidates: dict = {}
    for run_dir in sorted(backbone_dir.glob("2*")):
        params_path = run_dir / "logs" / "params.yaml"
        if not params_path.is_file():
            continue
        cfg = yaml.safe_load(params_path.read_text())
        objectives = frozenset((cfg.get("objectives") or {}).keys())
        subset = (cfg.get("dataset") or {}).get("subset")
        if subset is None or subset == "XCM":
            continue
        candidates.setdefault((objectives, subset), []).append(run_dir.name)
    return candidates


def pick_run(backbone_dir: Path, candidates: list, target_range_mae: float) -> tuple:
    """Disambiguate duplicate (config, region) runs (leftover hyperparameter-
    search reruns) by matching each candidate's logs/test_metrics.json
    polyphony_reg.overall.range_mae against target_range_mae
    (soundscape_test_metrics.csv's own value for this cell) -- same
    technique plot_confusion_matrix_perchv2_polclass.py used for Perch v2's
    TC+Pol-Class config. Returns (best_run_name, abs_diff) or (None, None)."""
    best_name, best_diff = None, float("inf")
    for name in candidates:
        metrics_path = backbone_dir / name / "logs" / "test_metrics.json"
        if not metrics_path.is_file():
            continue
        data = json.loads(metrics_path.read_text())
        range_mae = data[0]["polyphony_reg"]["overall"]["range_mae"]
        diff = abs(range_mae - target_range_mae)
        if diff < best_diff:
            best_name, best_diff = name, diff
    return best_name, (best_diff if best_name is not None else None)


def _recover_region_df(backbone: str, label: str, region: str, csv_df: pd.DataFrame) -> pd.DataFrame | None:
    """Recover one (backbone, config, region) cell's raw predictions from
    that config's own run directory (see module docstring). Returns None if
    no run could be matched."""
    backbone_dir = ARCHIVE / backbone
    candidates_by_key = scan_run_dirs(backbone_dir)
    sig = REG_OBJECTIVES[label]
    head = CONFIG_HEAD[label]

    candidates = candidates_by_key.get((sig, region), [])
    if not candidates:
        print(f"  [warn] no run found for {backbone} / {label} / {region}")
        return None
    target_col = f"{region}_{head}"
    target_range_mae = float(csv_df.loc["range_mae", target_col])
    run_name, diff = pick_run(backbone_dir, candidates, target_range_mae)
    if run_name is None:
        print(f"  [warn] no test_metrics.json among {len(candidates)} candidate(s) for "
              f"{backbone} / {label} / {region}")
        return None
    if diff > MATCH_TOLERANCE:
        print(f"  [warn] best match for {backbone} / {label} / {region} differs by {diff:.5f} "
              f"(n_candidates={len(candidates)}, run={run_name}) -- verify manually")
    pkl_path = backbone_dir / run_name / "logs" / "test_results.pkl"
    df = pickle.load(open(pkl_path, "rb"))
    return df[["y_true.min_polyphony", "y_true.max_polyphony", "predictions.polyphony_reg"]]


def load_config_predictions_by_region(backbone: str, label: str, csv_df: pd.DataFrame) -> dict:
    """{region: df} for one (backbone, config) cell, recovered from that
    config's own per-region run directories (see module docstring) --
    unlike load_config_predictions(), keeps each region separate instead of
    pooling, so a per-region statistic (e.g. plot_rq1.py-style per-region
    macro-MAE) can be computed before reducing across regions."""
    result = {}
    for region in REGIONAL_DATASETS:
        df = _recover_region_df(backbone, label, region, csv_df)
        if df is not None:
            result[region] = df
    return result


def load_config_predictions(backbone: str, label: str, csv_df: pd.DataFrame) -> pd.DataFrame:
    """Pool all 7 REGIONAL_DATASETS' raw predictions for one (backbone,
    config) cell, recovered from that config's own per-region run
    directories (see module docstring)."""
    by_region = load_config_predictions_by_region(backbone, label, csv_df)
    if not by_region:
        return pd.DataFrame(columns=["y_true.min_polyphony", "y_true.max_polyphony", "predictions.polyphony_reg"])
    return pd.concat(by_region.values(), ignore_index=True)


def _region_macro_mae_1_6(df: pd.DataFrame) -> tuple:
    """Per-(backbone, config, region) macro-MAE(1-6): unweighted mean of
    per-level MAE (round(pred) vs. true level) over whichever of levels 1-6
    have at least one unambiguous-label (min_polyphony == max_polyphony)
    clip in this region -- same construction as plot_rq1.py's own
    _region_macro_mae_1_6(), applied to an already-loaded df (this module's
    recovered per-region predictions) instead of reading a pkl path
    directly. Some regions have little/no support at the higher levels, so
    a region's average can silently be taken over fewer than 6 levels; the
    count of levels actually used is returned so that can be surfaced
    rather than hidden (see build_region_macro_mae_table()). Returns
    (nan, 0) if there are zero unambiguous clips at every level 1-6."""
    unamb = df[df["y_true.min_polyphony"] == df["y_true.max_polyphony"]].copy()
    unamb["true_level"] = unamb["y_true.min_polyphony"].round().astype(int)
    unamb["pred_level"] = unamb["predictions.polyphony_reg"].round().astype(int)

    level_maes = []
    for level in range(1, MACRO_LEVEL + 1):
        sub = unamb[unamb["true_level"] == level]
        if len(sub):
            level_maes.append(float((sub["true_level"] - sub["pred_level"]).abs().mean()))
    if not level_maes:
        return float("nan"), 0
    return float(sum(level_maes) / len(level_maes)), len(level_maes)


def build_region_macro_mae_table() -> pd.DataFrame:
    """Long (backbone, config, region, macro_mae_1_6, n_levels_used) table --
    same per-region construction as plot_rq1.py's compute_soundscape_macro_
    mae_stats() (one macro-MAE(1-6) value per (model, region), later reduced
    to mean +/- SD across the 7 regions for a plot's whiskers), applied to
    RQ2's 5 TC-head configs via this module's recovered per-region
    predictions (load_config_predictions_by_region()) instead of Pooled-
    Embeddings' un-overwritten pkl files."""
    rows = []
    for backbone in SPATIAL_BACKBONES:
        csv_path = ARCHIVE / backbone / "scape_eval_results" / "soundscape_test_metrics.csv"
        csv_df = pd.read_csv(csv_path, index_col=0)
        for label in CONFIG_LABELS:
            by_region = load_config_predictions_by_region(backbone, label, csv_df)
            for region in REGIONAL_DATASETS:
                df = by_region.get(region)
                if df is None:
                    rows.append({"backbone": backbone, "config": label, "region": region,
                                 "macro_mae_1_6": float("nan"), "n_levels_used": 0})
                    continue
                macro_mae, n_levels = _region_macro_mae_1_6(df)
                rows.append({"backbone": backbone, "config": label, "region": region,
                             "macro_mae_1_6": macro_mae, "n_levels_used": n_levels})
    long = pd.DataFrame(rows)
    thin = long[long["n_levels_used"] < MACRO_LEVEL]
    if len(thin):
        print(f"build_region_macro_mae_table: {len(thin)}/{len(long)} (backbone, config, region) cells used "
              f"fewer than {MACRO_LEVEL} levels for their macro-MAE(1-6) average (zero unambiguous clips at "
              "the missing level(s) in that region).")
    return long


def official_metrics(csv_df: pd.DataFrame, head: str) -> tuple:
    """Mean range_mae (official ranking basis, lower is better) and mean
    range_accuracy (RQ1-style secondary reference, higher is better) across
    the 7 REGIONAL_DATASETS, read directly from soundscape_test_metrics.csv
    -- the exact same aggregate plot_rq2.py's figures and compute_rq2_
    paired_cell_analysis.py's ranking checks already use."""
    cols = [f"{r}_{head}" for r in REGIONAL_DATASETS]
    return float(csv_df.loc["range_mae", cols].mean()), float(csv_df.loc["range_accuracy", cols].mean())


def build_rows() -> pd.DataFrame:
    rows = []
    for backbone in SPATIAL_BACKBONES:
        csv_path = ARCHIVE / backbone / "scape_eval_results" / "soundscape_test_metrics.csv"
        csv_df = pd.read_csv(csv_path, index_col=0)
        print(f"=== {backbone} ===")
        for label in CONFIG_LABELS:
            df = load_config_predictions(backbone, label, csv_df)
            levels = per_level_accuracy(df)
            official_mae, official_acc = official_metrics(csv_df, CONFIG_HEAD[label])

            row = {
                "backbone": backbone, "backbone_display": BACKBONE_META[backbone]["display"],
                "config": label, "official_mae": official_mae, "official_range_accuracy": official_acc,
                "acc_all_exact": levels["acc_all_exact"], "acc_nonzero": levels["acc_nonzero"],
                "n_unambiguous": levels["n_unambiguous"], "n_total": levels["n_total"],
                "n_nonzero": levels["n_nonzero"],
            }
            for level in range(MAX_LEVEL + 1):
                row[f"acc_{level}"] = levels[f"acc_{level}"]
                row[f"mae_{level}"] = levels[f"mae_{level}"]
                row[f"n_{level}"] = levels[f"n_{level}"]

            acc_vals_0_6 = [levels[f"acc_{l}"] for l in range(MACRO_LEVEL + 1)]
            acc_vals_1_6 = [levels[f"acc_{l}"] for l in range(1, MACRO_LEVEL + 1)]
            mae_vals_0_6 = [levels[f"mae_{l}"] for l in range(MACRO_LEVEL + 1)]
            mae_vals_1_6 = [levels[f"mae_{l}"] for l in range(1, MACRO_LEVEL + 1)]
            row["macro_acc_0_6"] = sum(acc_vals_0_6) / len(acc_vals_0_6)
            row["macro_acc_1_6"] = sum(acc_vals_1_6) / len(acc_vals_1_6)
            row["macro_mae_0_6"] = sum(mae_vals_0_6) / len(mae_vals_0_6)
            row["macro_mae_1_6"] = sum(mae_vals_1_6) / len(mae_vals_1_6)
            # "presence/absence-driven" signal: mean exact-match accuracy at
            # levels 0-1 vs. levels 2-6 -- a config whose official/exact rank
            # is much better than its macro_acc_1_6 rank despite a large gap
            # here is winning mostly on presence/absence, not polyphony
            # discrimination (flagged explicitly in the markdown output).
            row["acc_0_1_mean"] = (levels["acc_0"] + levels["acc_1"]) / 2
            row["acc_2_6_mean"] = sum(levels[f"acc_{l}"] for l in range(2, MACRO_LEVEL + 1)) / (MACRO_LEVEL - 1)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ranking tables (mirrors compute_soundscape_per_level_accuracy.py's
# build_table() rank columns, applied within an arbitrary group of rows)
# ---------------------------------------------------------------------------

def add_ranks(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["rank_official"] = table["official_mae"].rank(ascending=True, method="min").astype(int)
    table["rank_exact"] = table["acc_all_exact"].rank(ascending=False, method="min").astype(int)
    table["rank_nonzero"] = table["acc_nonzero"].rank(ascending=False, method="min").astype(int)
    table["rank_macro_acc_0_6"] = table["macro_acc_0_6"].rank(ascending=False, method="min").astype(int)
    table["rank_macro_acc_1_6"] = table["macro_acc_1_6"].rank(ascending=False, method="min").astype(int)
    table["rank_macro_mae_0_6"] = table["macro_mae_0_6"].rank(ascending=True, method="min").astype(int)
    table["rank_macro_mae_1_6"] = table["macro_mae_1_6"].rank(ascending=True, method="min").astype(int)
    return table


def flag_distortion(table: pd.DataFrame, rank_gap_threshold: int = 2, acc_2_6_threshold: float = 0.35) -> pd.DataFrame:
    """Flag rows where the official rank is materially better than the
    macro_acc_1_6 rank (>= rank_gap_threshold positions) while level-2+
    exact-match accuracy is low (< acc_2_6_threshold) -- the same
    "presence/absence carries the ranking" pattern RQ1's analysis found."""
    table = table.copy()
    table["rank_gap_official_vs_macro"] = table["rank_macro_acc_1_6"] - table["rank_official"]
    table["distortion_flag"] = (table["rank_gap_official_vs_macro"] >= rank_gap_threshold) & \
                                (table["acc_2_6_mean"] < acc_2_6_threshold)
    return table


def build_group_table_md(table: pd.DataFrame, group_col: str, group_label_fn, row_label_col: str) -> list:
    lines = [
        "| Backbone | Config | "
        "Rank Official (MAE) | Rank Exact | Rank Macro-Acc(1-6) | Rank Macro-MAE(1-6) | "
        "Official MAE | Exact-match Acc. | Macro-Acc(1-6) | Macro-MAE(1-6) | Acc(0-1) mean | Acc(2-6) mean | Flag |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for group_val, sub in table.groupby(group_col, sort=False):
        sub = add_ranks(sub)
        sub = flag_distortion(sub)
        sub = sub.sort_values("rank_macro_acc_1_6")
        for _, r in sub.iterrows():
            flag = "**YES**" if r["distortion_flag"] else ""
            lines.append(
                f"| {group_label_fn(group_val)} | {r[row_label_col]} | {r['rank_official']} | {r['rank_exact']} "
                f"| {r['rank_macro_acc_1_6']} | {r['rank_macro_mae_1_6']} | {r['official_mae']:.4f} "
                f"| {r['acc_all_exact']:.3f} | {r['macro_acc_1_6']:.3f} | {r['macro_mae_1_6']:.3f} "
                f"| {r['acc_0_1_mean']:.3f} | {r['acc_2_6_mean']:.3f} | {flag} |"
            )
    return lines


def build_overall_table_md(table: pd.DataFrame) -> list:
    """Mean across the 5 backbones per config (5 rows) -- "best overall
    configuration" question, includes the bare TC-head (per compute_rq2_
    paired_cell_analysis.py's analysis 4 convention, which also includes it
    for this specific question)."""
    agg_cols = ["official_mae", "official_range_accuracy", "acc_all_exact", "acc_nonzero",
                "macro_acc_0_6", "macro_acc_1_6", "macro_mae_0_6", "macro_mae_1_6",
                "acc_0_1_mean", "acc_2_6_mean"]
    mean_by_config = table.groupby("config", sort=False)[agg_cols].mean().reindex(CONFIG_LABELS)
    mean_by_config = mean_by_config.reset_index()
    mean_by_config = add_ranks(mean_by_config)
    mean_by_config = flag_distortion(mean_by_config)
    mean_by_config = mean_by_config.sort_values("rank_macro_acc_1_6")

    lines = [
        "| Config | Rank Official (MAE) | Rank Exact | Rank Macro-Acc(1-6) | Rank Macro-MAE(1-6) | "
        "Official MAE | Exact-match Acc. | Macro-Acc(1-6) | Macro-MAE(1-6) | Acc(0-1) mean | Acc(2-6) mean | Flag |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in mean_by_config.iterrows():
        flag = "**YES**" if r["distortion_flag"] else ""
        lines.append(
            f"| {r['config']} | {r['rank_official']} | {r['rank_exact']} | {r['rank_macro_acc_1_6']} "
            f"| {r['rank_macro_mae_1_6']} | {r['official_mae']:.4f} | {r['acc_all_exact']:.3f} "
            f"| {r['macro_acc_1_6']:.3f} | {r['macro_mae_1_6']:.3f} | {r['acc_0_1_mean']:.3f} "
            f"| {r['acc_2_6_mean']:.3f} | {flag} |"
        )
    return lines, mean_by_config


def build_markdown(table: pd.DataFrame) -> str:
    lines = [
        "# RQ2 -- Per-polyphony-level accuracy/MAE breakdown, TC(+auxiliary) configurations (soundscape)",
        "",
        "Extends `compute_soundscape_per_level_accuracy.py`'s per-level distortion check (RQ1's plain "
        "frozen-backbone comparison) to RQ2's 5 TC-head configurations x 5 SPATIAL_BACKBONES: does the "
        "currently-reported best-auxiliary-per-backbone / best-overall-configuration ranking (mean "
        "`range_mae` across the 7 REGIONAL_DATASETS, the same aggregate `plot_rq2.py`/`compute_rq2_paired_"
        "cell_analysis.py` already use) survive under macro-averaged, level-balanced exact-match accuracy "
        "and MAE -- or is it partly an artifact of presence/absence (level 0-1) performance the way RQ1's "
        "plain comparison was.",
        "",
        "**Data recovery**: raw per-clip soundscape predictions for 4 of these 5 configs are not present "
        "at the canonical `archive/Spatial-Embeddings/{backbone}/scape_eval_results/` location (each "
        "config's eval run overwrites the same path; only the merged metrics CSV survives every config's "
        "run). They were recovered from each config's own per-region training-run directory instead, "
        "matched by `objectives` signature (+, where duplicated by a leftover hyperparameter search, by "
        "`range_mae` against the merged CSV) -- see the module docstring in `compute_rq2_per_level_"
        "accuracy.py` for the full methodology, first established for one cell by `plot_confusion_matrix_"
        "perchv2_polclass.py`.",
        "",
        "Same per-level methodology as `compute_soundscape_per_level_accuracy.py`: exact-match accuracy "
        "(`round(pred) == true_level`) and MAE on the unambiguous-label subset (`min_polyphony == "
        "max_polyphony`) per true polyphony level 0-8, folding levels >8 into 8; macro_acc/mae_0_6 and "
        "_1_6 are unweighted means over levels 0-6 / 1-6 (levels 7-8 excluded as too small-n). \"Acc(0-1) "
        "mean\" / \"Acc(2-6) mean\" below are the presence/absence-vs-discrimination split used for the "
        "distortion flag: a row is flagged when its official-MAE rank beats its macro-Acc(1-6) rank by "
        ">= 2 positions *and* its mean level-2-6 accuracy is below 0.35 -- i.e. it ranks well mainly "
        "because it's good at levels 0-1, not because it discriminates polyphony degree.",
        "",
        "## 1. Best auxiliary per backbone",
        "",
        "Per backbone, all 5 configs ranked against each other (sorted by macro-Acc(1-6) rank; \"Rank "
        "Official\" is the currently-reported ranking basis, mean `range_mae`, lower is better).",
        "",
    ]
    lines += build_group_table_md(table, "backbone_display", lambda x: x, "config")

    overall_lines, overall_table = build_overall_table_md(table)
    lines += [
        "",
        "## 2. Best overall configuration",
        "",
        "Mean across all 5 SPATIAL_BACKBONES per config (n=5 backbones x 7 regions = 35 cells pooled per "
        "row) -- includes the bare TC-head, per the existing \"best overall configuration\" convention in "
        "`compute_rq2_paired_cell_analysis.py`'s analysis 4 (unlike \"best auxiliary\", which excludes it).",
        "",
    ]
    lines += overall_lines

    official_winner = table.loc[table.groupby("backbone_display")["official_mae"].idxmin()]
    macro_winner_rows = []
    for backbone, sub in table.groupby("backbone_display", sort=False):
        best = sub.loc[sub["macro_acc_1_6"].idxmax()]
        macro_winner_rows.append((backbone, best["config"]))
    disagreements = [
        (bb, off_row["config"], macro_cfg)
        for bb, macro_cfg in macro_winner_rows
        for _, off_row in [(None, table[(table["backbone_display"] == bb)].loc[
            table[(table["backbone_display"] == bb)]["official_mae"].idxmin()])]
        if off_row["config"] != macro_cfg
    ]

    overall_official_winner = overall_table.loc[overall_table["official_mae"].idxmin(), "config"]
    overall_macro_winner = overall_table.loc[overall_table["macro_acc_1_6"].idxmax(), "config"]

    lines += [
        "",
        "## 3. Summary: does the ranking survive?",
        "",
        f"**Best overall configuration** -- official (mean `range_mae`): **{overall_official_winner}**; "
        f"macro-Acc(1-6): **{overall_macro_winner}** -- "
        + ("same winner under both metrics." if overall_official_winner == overall_macro_winner
           else "**different winner** under the two metrics -- the official ranking's overall-best "
                "configuration is not the macro-level-balanced ranking's overall-best configuration.") + "",
        "",
        f"**Best auxiliary per backbone**: {len(disagreements)} of {table['backbone_display'].nunique()} "
        "backbones disagree between the official (mean `range_mae`) winner and the macro-Acc(1-6) winner"
        + (":" if disagreements else " -- every backbone's official winner is also its macro-Acc(1-6) winner.")
        ,
    ]
    for bb, off_cfg, macro_cfg in disagreements:
        lines.append(f"- **{bb}**: official winner = {off_cfg}; macro-Acc(1-6) winner = {macro_cfg}")

    n_flagged = int(table.pipe(lambda t: pd.concat(
        [flag_distortion(add_ranks(sub)) for _, sub in t.groupby("backbone_display", sort=False)]
    ))["distortion_flag"].sum())
    lines += [
        "",
        f"**Distortion flag**: {n_flagged} of {len(table)} (backbone, config) cells are flagged as "
        "presence/absence-driven under the per-backbone ranking (see Section 1's Flag column for which).",
        "",
        "## Reading this",
        "",
        "- Section 1 is the primary evidence for the per-backbone \"best auxiliary\" question; Section 2 "
        "for the \"best overall configuration\" question -- both ranked by the same macro-Acc(1-6) column "
        "for comparability with `compute_soundscape_per_level_accuracy.py`'s RQ1 table.",
        "- A flagged row does not mean the official ranking is wrong, only that its apparent advantage is "
        "concentrated at levels 0-1 rather than spread across the polyphony range -- exactly the caveat "
        "`compute_soundscape_per_level_accuracy.py`'s docstring raises for RQ1's plain backbone comparison.",
        "- Sample sizes are the same order as the rest of RQ2 (n=5 backbones, 7 regions pooled per cell) "
        "-- this file adds a metric-choice check, not a new significance test.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq2"))
    args = parser.parse_args()

    table = build_rows()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "rq2_per_level_accuracy.csv"
    table.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    md = build_markdown(table)
    out_md = args.out_dir / "rq2_per_level_accuracy.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
