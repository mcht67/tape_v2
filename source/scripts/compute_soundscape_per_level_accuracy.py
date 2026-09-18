#!/usr/bin/env python3
"""Per-polyphony-level accuracy breakdown, by backbone -- soundscape test
split, regression head.

Purpose: the paper's headline soundscape "Accuracy" (range_accuracy, see
utils/metrics.py) is pooled across every clip, including polyphony=0, which
is the single most common class by a wide margin. A backbone that is
essentially just detecting presence/absence of birds (rather than the
polyphony degree itself) can still post a respectable aggregate accuracy on
the strength of the 0-class alone. This script breaks accuracy out by true
polyphony level to check whether that is happening, and for which backbones.

Ground truth caveat: soundscape clips have no single exact polyphony label,
only a [min_polyphony, max_polyphony] range (see plot_confusion_matrix.py's
load_soundscape docstring) -- real recordings admit more than one valid
annotation. That range-tolerant criterion (round(pred) in [min, max]) is
what range_accuracy already reports in aggregate, but it cannot be broken
out by "true level" without either (a) trivially collapsing to ~100%/~0%
per row (if the row's own prediction is used to fix a pseudo-true label, as
plot_confusion_matrix.py does for visualization) or (b) picking one bound as
"the" true value, which no bound uniquely is. Instead this script restricts
the per-level breakdown to the subset of clips with an *unambiguous* label
(min_polyphony == max_polyphony, empirically ~85-90% of soundscape clips --
see the printed coverage line) and computes ordinary exact-match accuracy
(round(pred) == true level) within that subset. The excluded ambiguous
clips are not dropped from the aggregate-accuracy column, which is instead
read from soundscape_test_metrics.csv's mean-across-regions range_accuracy,
matching the number reported in results/result_tables_soundscape.tex. That
official range_accuracy is itself range-tolerant (correct iff the
prediction falls anywhere in [min, max], which can be a wide interval at
higher true polyphony), so it is not directly comparable to the per-level
exact-match numbers below -- acc_all_exact is the same exact-match
criterion as the per-level columns, giving an apples-to-apples aggregate
for ranking (a) instead.

Region-weighting: every metric below (per-level acc/MAE, acc_nonzero,
acc_all_exact, macro_acc_*, macro_mae_*) is computed PER REGION first, then
averaged unweighted across the 7 regions (HSN/NES/PER/POW/SNE/SSW/UHH) --
matching aggregate_range_accuracy's own convention (soundscape_test_metrics
.csv's range_accuracy is already a mean across regions) and the convention
used everywhere else in RQ1 (rq1_accuracy_comparison.md's per-dataset table
and its dataset-clustered GEE test, plot_rq1.py's compute_source_stats()).
This is NOT the same as pooling every region's clips together before
computing a metric: region size is wildly unequal (SSW alone accounts for
127,013 of the 198,952 pooled unambiguous clips -- 64% -- vs. POW's 1,293,
0.6%), so a pooled computation is effectively a SSW-weighted number wearing
a "7-region" label, and can rank backbones differently from the region-
balanced version -- confirmed empirically: pooling puts Bird-MAE 1st on
macro-MAE(1-6), region-balancing puts AST 1st instead. An earlier version of
this script pooled first; if you're diffing against old output, expect
different numbers and occasionally a different backbone ranked first.

A region can have zero unambiguous clips at some levels (e.g. HSN has none
at levels 4-6, NES and UHH none at level 6), so a per-level/macro average
"across regions" is a nanmean over whichever regions have support at that
level, not literally all 7 -- n_regions_{level} in the printed table (and
the .md's coverage note) says how many regions contributed to each level.

Two backbone sets, each its own table (they are different studies, not
comparable on one shared ranking):
  - 13 frozen-backbone, pooled-embedding-head models (archive/Pooled-Embeddings)
  - 4 fine-tuned models (archive/XCM-Generalization-fine-tune) -- perch_v2 is
    excluded, matching backbone_meta.py's FINETUNE_NAME_TO_CANONICAL (its
    "fine-tune" archive data is not a genuine fine-tuning result -- see that
    file's comment).

acc_nonzero (mean exact-match accuracy over unambiguous clips with true
level >= 1) is, within each region, still support-weighted across levels
(dominated by level 1, 63% of the pooled nonzero support -- see the printed
n table): a backbone that is merely good at telling "1" from "not 1" can
still look decent on acc_nonzero without actually discriminating levels 2-6
at all. macro_acc_* fixes the *level* weighting the same way region-
balancing fixes the *region* weighting: an unweighted mean of the per-level
exact-match accuracies (mae_* likewise, an unweighted mean of per-level
MAE), each first computed per region then averaged across regions. Both are
reported over levels 0-6 (macro_*_0_6) and 1-6 (macro_*_1_6, dropping the
dominant zero class too); levels 7-8 are excluded from every macro-average
as too small-n to be worth an equal vote (n=125 and n=45 pooled -- see the
printed n table; even level 6's n=508 is comparatively thin next to levels
0-2's tens of thousands, so a single mispredicted clump of level-6 clips
moves macro_acc_0_6/1_6 more than the same-sized clump would at level 0-2).

Writes plots/figures/polyphony_level_breakdown/{pooled,finetuned}_per_level_accuracy.csv
and prints both tables to stdout.

Usage:
    complete-venv/bin/python source/scripts/compute_soundscape_per_level_accuracy.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backbone_meta import BACKBONE_META, FINETUNE_NAME_TO_CANONICAL

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = REPO_ROOT / "archive"
REGIONS = ["HSN", "NES", "PER", "POW", "SNE", "SSW", "UHH"]
MAX_LEVEL = 8  # columns 0..8, with true levels > 8 folded into the "8+" column (negligible n)
MACRO_LEVEL = 6  # macro-average over levels 0..6 only -- 7/8 dropped as too noisy (n<130)
OUT_DIR = REPO_ROOT / "plots" / "figures" / "polyphony_level_breakdown"


def load_region_df(model_dir: Path, region: str) -> pd.DataFrame | None:
    path = model_dir / "scape_eval_results" / f"{region}_reg_scape_test_results.pkl"
    if not path.exists():
        return None
    return pickle.load(open(path, "rb"))


def per_region_stats(df: pd.DataFrame) -> dict:
    """Restrict to unambiguous clips (min_polyphony == max_polyphony), fold
    true level > MAX_LEVEL into the MAX_LEVEL column, and compute exact-match
    accuracy/MAE (n, mean) per true level 0..MAX_LEVEL, all within this one
    region's own data -- the unit build_table() averages across regions."""
    unamb = df[df["y_true.min_polyphony"] == df["y_true.max_polyphony"]].copy()
    unamb["true_level"] = unamb["y_true.min_polyphony"].round().astype(int).clip(upper=MAX_LEVEL)
    unamb["pred_level"] = unamb["predictions.polyphony_reg"].round().astype(int)
    unamb["correct"] = unamb["true_level"] == unamb["pred_level"]
    unamb["abs_err"] = (unamb["true_level"] - unamb["pred_level"]).abs()

    out = {"n_total": len(df), "n_unambiguous": len(unamb)}
    for level in range(MAX_LEVEL + 1):
        sub = unamb[unamb["true_level"] == level]
        out[f"n_{level}"] = len(sub)
        out[f"acc_{level}"] = float(sub["correct"].mean()) if len(sub) else float("nan")
        out[f"mae_{level}"] = float(sub["abs_err"].mean()) if len(sub) else float("nan")

    nonzero = unamb[unamb["true_level"] >= 1]
    out["n_nonzero"] = len(nonzero)
    out["acc_nonzero"] = float(nonzero["correct"].mean()) if len(nonzero) else float("nan")
    out["acc_all_exact"] = float(unamb["correct"].mean()) if len(unamb) else float("nan")

    acc_0_6 = [out[f"acc_{l}"] for l in range(MACRO_LEVEL + 1) if not np.isnan(out[f"acc_{l}"])]
    acc_1_6 = [out[f"acc_{l}"] for l in range(1, MACRO_LEVEL + 1) if not np.isnan(out[f"acc_{l}"])]
    mae_0_6 = [out[f"mae_{l}"] for l in range(MACRO_LEVEL + 1) if not np.isnan(out[f"mae_{l}"])]
    mae_1_6 = [out[f"mae_{l}"] for l in range(1, MACRO_LEVEL + 1) if not np.isnan(out[f"mae_{l}"])]
    out["macro_acc_0_6"] = float(np.mean(acc_0_6)) if acc_0_6 else float("nan")
    out["macro_acc_1_6"] = float(np.mean(acc_1_6)) if acc_1_6 else float("nan")
    out["macro_mae_0_6"] = float(np.mean(mae_0_6)) if mae_0_6 else float("nan")
    out["macro_mae_1_6"] = float(np.mean(mae_1_6)) if mae_1_6 else float("nan")
    return out


def aggregate_range_accuracy(model_dir: Path) -> float:
    """Mean-across-regions range_accuracy (Regression head) from
    soundscape_test_metrics.csv -- the same number result_tables_soundscape.tex
    reports, read directly rather than recomputed, so ranking (a) matches the
    paper's own aggregate exactly. Already region-balanced (mean across the
    7 per-region values), same convention as every other column here."""
    csv_path = model_dir / "scape_eval_results" / "soundscape_test_metrics.csv"
    df = pd.read_csv(csv_path, index_col=0)
    reg_cols = [c for c in df.columns if c.endswith("_reg")]
    return float(df.loc["range_accuracy", reg_cols].mean())


MEAN_COLS = (
    [f"acc_{l}" for l in range(MAX_LEVEL + 1)] + [f"mae_{l}" for l in range(MAX_LEVEL + 1)] +
    ["acc_nonzero", "acc_all_exact", "macro_acc_0_6", "macro_acc_1_6", "macro_mae_0_6", "macro_mae_1_6"]
)
SUM_COLS = [f"n_{l}" for l in range(MAX_LEVEL + 1)] + ["n_nonzero", "n_unambiguous", "n_total"]


def build_table(model_dirs: dict[str, Path], display: dict[str, str]) -> pd.DataFrame:
    rows = []
    for key, model_dir in model_dirs.items():
        region_stats = [per_region_stats(df) for r in REGIONS if (df := load_region_df(model_dir, r)) is not None]
        region_df = pd.DataFrame(region_stats)

        row = {"backbone": display[key], "aggregate_range_accuracy": aggregate_range_accuracy(model_dir)}
        for col in MEAN_COLS:
            row[col] = float(region_df[col].mean(skipna=True))
        for col in SUM_COLS:
            row[col] = int(region_df[col].sum())
        for level in range(MAX_LEVEL + 1):
            row[f"n_regions_{level}"] = int((region_df[f"n_{level}"] > 0).sum())
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("acc_nonzero", ascending=False).reset_index(drop=True)
    table.insert(0, "rank_aggregate", table["aggregate_range_accuracy"].rank(ascending=False, method="min").astype(int))
    table.insert(0, "rank_exact", table["acc_all_exact"].rank(ascending=False, method="min").astype(int))
    table.insert(0, "rank_nonzero", table["acc_nonzero"].rank(ascending=False, method="min").astype(int))
    table.insert(0, "rank_macro_acc_0_6", table["macro_acc_0_6"].rank(ascending=False, method="min").astype(int))
    table.insert(0, "rank_macro_acc_1_6", table["macro_acc_1_6"].rank(ascending=False, method="min").astype(int))
    table.insert(0, "rank_macro_mae_0_6", table["macro_mae_0_6"].rank(ascending=True, method="min").astype(int))
    table.insert(0, "rank_macro_mae_1_6", table["macro_mae_1_6"].rank(ascending=True, method="min").astype(int))
    return table


def class_distribution(model_dir: Path) -> dict:
    """Fraction of unambiguous clips at each true level, pooled across
    regions -- for context only (how skewed the raw data is), not used by
    any region-balanced metric above."""
    dfs = [df for r in REGIONS if (df := load_region_df(model_dir, r)) is not None]
    pooled = pd.concat(dfs, ignore_index=True)
    unamb = pooled[pooled["y_true.min_polyphony"] == pooled["y_true.max_polyphony"]].copy()
    unamb["true_level"] = unamb["y_true.min_polyphony"].round().astype(int).clip(upper=MAX_LEVEL)
    counts = unamb["true_level"].value_counts()
    return {level: counts.get(level, 0) / len(unamb) for level in range(MAX_LEVEL + 1)}


def print_table(table: pd.DataFrame, title: str):
    print(f"\n=== {title} ===")
    cols = ["backbone", "rank_aggregate", "rank_exact", "rank_nonzero", "rank_macro_acc_0_6", "rank_macro_acc_1_6",
            "rank_macro_mae_0_6", "rank_macro_mae_1_6", "aggregate_range_accuracy", "acc_all_exact",
            "acc_nonzero", "macro_acc_0_6", "macro_acc_1_6", "macro_mae_0_6", "macro_mae_1_6"]
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(table.sort_values("rank_macro_acc_1_6")[cols].to_string(index=False))

    acc_cols = ["backbone"] + [f"acc_{l}" for l in range(MAX_LEVEL + 1)]
    print(f"\n-- per-level exact-match accuracy, region-balanced ({title}) --")
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(table[acc_cols].to_string(index=False))

    mae_cols = ["backbone"] + [f"mae_{l}" for l in range(MAX_LEVEL + 1)]
    print(f"\n-- per-level MAE, region-balanced ({title}) --")
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(table[mae_cols].to_string(index=False))

    n_cols = ["n_unambiguous", "n_total"] + [f"n_{l}" for l in range(MAX_LEVEL + 1)]
    print(f"\n-- n per level, pooled across regions for context ({title}) --")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table[["backbone"] + n_cols].to_string(index=False))

    nr_cols = [f"n_regions_{l}" for l in range(MAX_LEVEL + 1)]
    print(f"\n-- number of regions (of 7) with >=1 unambiguous clip at each level ({title}) --")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table[["backbone"] + nr_cols].to_string(index=False))


def write_ranking_md(table: pd.DataFrame, title: str, source_desc: str, path: Path):
    """Combined-ranking table (official/exact/nonzero/macro-acc/macro-MAE
    ranks side by side), one row per backbone, sorted by macro-acc(1-6) rank
    -- the markdown counterpart of the table printed by print_table(), meant
    to be read on its own rather than alongside a terminal run."""
    t = table.sort_values("rank_macro_acc_1_6")
    n_row = table.iloc[0]
    lines = [
        f"# {title}",
        "",
        source_desc,
        "",
        "Every ranking below is region-balanced: computed per region (HSN, NES, PER, POW, SNE, SSW, UHH) first, "
        "then averaged unweighted across the 7 regions -- not pooled across every clip first. This matters because "
        "region size is wildly unequal (SSW alone is 127,013 of the 198,952 pooled unambiguous clips, 64%, vs. "
        "POW's 1,293, 0.6%), so a pooled computation would effectively be an SSW-weighted number wearing a "
        "\"7-region\" label. An earlier version of this table pooled first; region-balancing changes several "
        "ranks, including 1st place on Macro-MAE(1-6) (Bird-MAE under pooling, AST under region-balancing).",
        "",
        "Four rankings of the same backbones, each a different answer to \"how good is this backbone at "
        "soundscape polyphony estimation\":",
        "",
        "- **Official** -- `range_accuracy`, mean across the 7 regions, exactly as reported in "
        "`results/result_tables_soundscape.tex`. Range-tolerant (correct iff the prediction falls anywhere in "
        "`[min_polyphony, max_polyphony]`) and, within each region, pooled across every clip including "
        "polyphony=0.",
        "- **Exact** -- `acc_all_exact`, ordinary exact-match accuracy on every unambiguous-label clip "
        "(min_polyphony == max_polyphony) and every level 0-8, computed per region then averaged. Apples-to-apples "
        "with Nonzero/Macro below (same correctness rule), unlike Official.",
        "- **Nonzero** -- exact-match accuracy restricted to unambiguous clips with true level >= 1, per region "
        "then averaged. Still level-weighted *within* each region, so it is dominated by level 1 (63% of the "
        "pooled nonzero support).",
        "- **Macro-acc(1-6)** / **Macro-MAE(1-6)** -- unweighted mean of the per-level exact-match accuracy / MAE "
        "over true levels 1-6, computed per region then averaged, so every level counts equally regardless of its "
        "support. Levels 0 (dominant class), 7, and 8 (n=125 and n=45 pooled, too small) are excluded.",
        "",
        "## Combined ranking",
        "",
        "| Rank (macro-acc 1-6) | Backbone | Official | Exact | Nonzero | Macro-acc(0-6) | **Macro-acc(1-6)** | "
        "Macro-MAE(0-6) | **Macro-MAE(1-6)** |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in t.iterrows():
        lines.append(
            f"| {int(row['rank_macro_acc_1_6'])} | {row['backbone']} | {int(row['rank_aggregate'])} | "
            f"{int(row['rank_exact'])} | {int(row['rank_nonzero'])} | {int(row['rank_macro_acc_0_6'])} | "
            f"**{int(row['rank_macro_acc_1_6'])}** | {int(row['rank_macro_mae_0_6'])} | "
            f"**{int(row['rank_macro_mae_1_6'])}** |"
        )
    lines += [
        "",
        "## Underlying values",
        "",
        "| Backbone | Official acc. | Exact acc. | Nonzero acc. | Macro-acc(0-6) | Macro-acc(1-6) | "
        "Macro-MAE(0-6) | Macro-MAE(1-6) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in t.iterrows():
        lines.append(
            f"| {row['backbone']} | {row['aggregate_range_accuracy']:.3f} | {row['acc_all_exact']:.3f} | "
            f"{row['acc_nonzero']:.3f} | {row['macro_acc_0_6']:.3f} | {row['macro_acc_1_6']:.3f} | "
            f"{row['macro_mae_0_6']:.3f} | {row['macro_mae_1_6']:.3f} |"
        )
    lines += [
        "",
        "## Caveat -- n per true polyphony level (pooled across all 7 regions for context; the rankings above use "
        "the region-balanced values, not these pooled counts) and region coverage",
        "",
        "| Level | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ |",
        "|---|---|---|---|---|---|---|---|---|---|",
        "| n (pooled) | " + " | ".join(f"{int(n_row[f'n_{l}']):,}" for l in range(MAX_LEVEL + 1)) + " |",
        "| regions with data (of 7) | " + " | ".join(str(int(n_row[f"n_regions_{l}"])) for l in range(MAX_LEVEL + 1)) + " |",
        "",
        "Macro-acc/MAE(0-6) and (1-6) average over levels 0-6 (dropping 7 and 8, n=125 and n=45 pooled -- too thin "
        "to carry equal weight against levels with tens of thousands of clips). Even within 0-6, level 6 has data "
        "in only some of the 7 regions (see the row above) and n=508 pooled -- thin relative to levels 0-2's tens "
        "of thousands, so it can move a backbone's macro rank by more than its true reliability would justify -- "
        "read a close macro-rank call between two backbones with that in mind. SSW alone holds 64% of all pooled "
        "unambiguous clips (127,013 of 198,952) -- every ranking above is region-balanced specifically so this "
        "single region cannot dominate by raw count (see the note at the top of this file).",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pooled_dirs = {m: ARCHIVE / "Pooled-Embeddings" / m for m in BACKBONE_META}
    pooled_dirs = {m: d for m, d in pooled_dirs.items() if (d / "scape_eval_results").exists()}
    pooled_display = {m: BACKBONE_META[m]["display"] for m in pooled_dirs}
    pooled_table = build_table(pooled_dirs, pooled_display)
    pooled_table.to_csv(OUT_DIR / "pooled_per_level_accuracy.csv", index=False)
    print_table(pooled_table, "Pooled-Embeddings (13 frozen backbones, soundscape test, Regression head)")
    write_ranking_md(
        pooled_table,
        "RQ1 -- Soundscape ranking, official vs. exact-match vs. macro-averaged (13 frozen backbones)",
        "Source: `archive/Pooled-Embeddings/` (13 frozen-backbone, pooled-MLP-head, regression-formulation "
        "models), `soundscape_test`, regression head, all 7 REGIONAL_DATASETS (UHH, HSN, PER, NES, POW, SSW, "
        "SNE). Exact/Nonzero/Macro columns are restricted to the ~85-90% of soundscape clips with an unambiguous "
        "label (`min_polyphony == max_polyphony`) -- see `compute_soundscape_per_level_accuracy.py`'s module "
        "docstring for why.",
        REPO_ROOT / "plots" / "figures" / "rq1" / "rq1_soundscape_ranking.md",
    )

    ft_dirs = {folder: ARCHIVE / "XCM-Generalization-fine-tune" / folder for folder in FINETUNE_NAME_TO_CANONICAL}
    ft_dirs = {folder: d for folder, d in ft_dirs.items() if (d / "scape_eval_results").exists()}
    ft_display = {folder: BACKBONE_META[canon]["display"] for folder, canon in FINETUNE_NAME_TO_CANONICAL.items() if folder in ft_dirs}
    ft_table = build_table(ft_dirs, ft_display)
    ft_table.to_csv(OUT_DIR / "finetuned_per_level_accuracy.csv", index=False)
    print_table(ft_table, "XCM-Generalization-fine-tune (4 fine-tuned backbones, soundscape test, Regression head)")

    dist = class_distribution(next(iter(pooled_dirs.values())))
    print("\n=== Class distribution (unambiguous soundscape clips, pooled across all 7 regions, for context) ===")
    for level, frac in dist.items():
        label = f"{level}" if level < MAX_LEVEL else f"{MAX_LEVEL}+"
        print(f"  polyphony={label}: {frac:.1%}")

    print(f"\nWrote {OUT_DIR / 'pooled_per_level_accuracy.csv'}")
    print(f"Wrote {OUT_DIR / 'finetuned_per_level_accuracy.csv'}")


if __name__ == "__main__":
    main()
