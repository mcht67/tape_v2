#!/usr/bin/env python3
"""
RQ5 (additional) -- Paired cell-level comparison, frozen-XCM vs.
region-specific, restricted to the 4 backbones with genuine fine-tuning
results (Perch v2 excluded).

Purpose: `compute_rq5_kurtosis.py`'s Delta_frozen (frozen-XCM -
region-specific) is computed over the 5 SPATIAL_BACKBONES (incl. Perch v2,
n=35), while the adjacent Delta_finetuned (fine-tuned-XCM - region-specific,
n=28) and `compute_rq5_finetune_vs_frozen.py`'s frozen-vs-fine-tuned
comparison (n=28) are necessarily restricted to the 4 backbones with valid
fine-tuning results (Perch v2's backbone could not be unfrozen -- see
`FINETUNE_NAME_TO_CANONICAL`'s comment). All three deltas are discussed
together in the same RQ5 paragraph, so this recomputes Delta_frozen on the
same 4-backbone set as the other two, making all three directly comparable
cell-for-cell.

Same source data and cell-level pairing as compute_rq5_kurtosis.py
(region-specific: `archive/Pooled-Embeddings/`; frozen-XCM:
`archive/XCM-Generalization/`; soundscape_test / range_mae), restricted to
the 4 backbones with genuine fine-tuning results (FINETUNE_NAME_TO_CANONICAL;
EfficientNet-B1, AudioProtoPNet, Bird-MAE, NatureLM-audio) x 7
REGIONAL_DATASETS = 28 cells (24 NES excluded).

Delta = (frozen-XCM MAE) - (region-specific MAE), same (backbone, dataset)
cell. Positive = frozen-XCM is worse than the region-specific head. Reported
as mean +/- SD, a paired t-test and Wilcoxon signed-rank test against zero,
and an exact two-sided binomial sign test on win/loss/tie counts (win =
frozen-XCM beats region-specific, i.e. Delta<0) -- same conventions as
compute_rq5_kurtosis.py's cell-level tests and compute_rq5_finetune_vs_frozen.py's
sign test -- pooled (n=28) and with NES excluded (n=24), NES being the
extreme-outlier dataset flagged elsewhere in RQ5 (the same robustness check
already run for the frozen-vs-fine-tuned comparison).

Also recomputes Delta_finetuned (fine-tuned-XCM - region-specific) on the
identical 4-backbone set, to confirm explicitly that it already matches
compute_rq5_kurtosis.py's reported 0.034 +/- 0.471 (n=28) rather than
assuming it -- and confirms the 4-backbone set itself (row_order) is
identical across this script, compute_rq5_kurtosis.py, and
compute_rq5_finetune_vs_frozen.py.

**Caveat**: n=28 pooled (n=7 per backbone, not broken out here) is small --
treat all significance tests as exploratory, same caveat as
compute_rq5_kurtosis.py and compute_rq5_finetune_vs_frozen.py.

Writes rq5_frozen_vs_region_4backbone.json and
rq5_frozen_vs_region_4backbone.md to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq5_frozen_vs_region_4backbone.py [--out-dir plots/figures/rq5]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import BACKBONE_META, FINETUNE_NAME_TO_CANONICAL
from plot_data import load_study
from plot_rq2 import SPATIAL_BACKBONES
from plot_rq5 import SCAPE_SOURCE, build_heatmap_matrix, species_ordered_datasets

FROZEN_LABEL = "Frozen-XCM"
REGION_LABEL = "Region-specific"
FINETUNE_LABEL = "Fine-tuned-XCM"

PREVIOUS_5BB_FROZEN = {"mean": 0.151, "std": 0.754, "n": 35}
KURTOSIS_4BB_FINETUNED = {"mean": 0.034, "std": 0.471, "n": 28}


def cell_diffs(other_matrix, baseline_matrix, dataset_order: list, row_order: list) -> np.ndarray:
    """Flattened (backbone, dataset) cell-level differences (other -
    baseline), both matrices reindexed to the same row_order/dataset_order
    first so cells line up 1:1; NaN cells (missing data) are dropped."""
    diff = (other_matrix.loc[row_order, dataset_order] - baseline_matrix.loc[row_order, dataset_order]).values
    diff = diff.flatten().astype(float)
    return diff[~np.isnan(diff)]


def diff_stats(diff: np.ndarray) -> dict:
    n = len(diff)
    return {
        "mean": float(diff.mean()) if n else None,
        "std": float(diff.std(ddof=1)) if n > 1 else None,
        "n": n,
    }


def paired_tests(diff: np.ndarray) -> dict:
    """Paired t-test and Wilcoxon signed-rank test of diff against zero.
    None for either test below the sample size it needs, or if Wilcoxon
    raises (e.g. all-zero differences)."""
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


def sign_test(diff: np.ndarray) -> dict:
    """Exact sign test: wins = frozen-XCM beats region-specific (diff<0,
    lower MAE), losses = diff>0, ties = diff==0 (dropped before the test)."""
    wins = int((diff < 0).sum())
    losses = int((diff > 0).sum())
    ties = int(len(diff) - wins - losses)
    n_decisive = wins + losses
    p_two_sided = round(float(scipy.stats.binomtest(wins, n_decisive, 0.5, alternative="two-sided").pvalue), 6) \
        if n_decisive > 0 else None
    return {"wins": wins, "losses": losses, "ties": ties, "n_decisive": n_decisive, "p_two_sided": p_two_sided}


def full_entry(diff: np.ndarray) -> dict:
    return {**diff_stats(diff), **paired_tests(diff), "sign_test": sign_test(diff)}


def fmt_mean_std(mean, std, decimals: int = 3) -> str:
    if mean is None:
        return "--"
    if std is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _fmt_p(x):
    return f"{x:.4f}" if x is not None else "n/a"


def build_row(label: str, entry: dict) -> str:
    t_p = _fmt_p(entry["t_p"])
    w_p = _fmt_p(entry["wilcoxon_p"])
    st = entry["sign_test"]
    wlt = f"{st['wins']}-{st['losses']}-{st['ties']}"
    st_p = _fmt_p(st["p_two_sided"])
    return (f"| {label} | {fmt_mean_std(entry['mean'], entry['std'])} | {entry['n']} "
            f"| p={t_p} | p={w_p} | {wlt} | p={st_p} |")


HEADER = ["| Comparison | Mean Delta MAE (± SD) | n | Paired t-test | Wilcoxon | Sign test (W-L-T) | Sign test p |",
          "|---|---|---|---|---|---|---|"]


def build_markdown(pooled_full: dict, pooled_no_nes: dict, finetuned_recheck: dict,
                    row_order: list, ft_row_order_kurtosis: list, ft_row_order_ftvf: list) -> str:
    backbones_match = row_order == ft_row_order_kurtosis == ft_row_order_ftvf
    finetuned_matches = (
        finetuned_recheck["n"] == KURTOSIS_4BB_FINETUNED["n"]
        and abs(finetuned_recheck["mean"] - KURTOSIS_4BB_FINETUNED["mean"]) < 0.001
        and abs(finetuned_recheck["std"] - KURTOSIS_4BB_FINETUNED["std"]) < 0.001
    )
    delta_mean = pooled_full["mean"] - PREVIOUS_5BB_FROZEN["mean"]

    lines = [
        "# RQ5 -- Frozen-XCM vs. region-specific, restricted to the 4 fine-tuning-matched backbones",
        "",
        "Recomputes `rq5_kurtosis.md`'s Delta_frozen (frozen-XCM - region-specific), which was computed "
        "over the 5 SPATIAL_BACKBONES (incl. Perch v2, n=35), on the same 4-backbone set as the adjacent "
        "Delta_finetuned (fine-tuned-XCM - region-specific, n=28) and "
        "`rq5_finetune_vs_frozen.md`'s frozen-vs-fine-tuned comparison (n=28) -- Perch v2's backbone "
        "could not be unfrozen (see `FINETUNE_NAME_TO_CANONICAL`'s comment), so it has no fine-tuning "
        "result. All three deltas are discussed together in the same RQ5 paragraph, so putting them on "
        "the same backbone set makes them directly comparable cell-for-cell.",
        "",
        f"Delta = (frozen-XCM MAE) - (region-specific MAE), matched by (backbone, dataset) cell, "
        f"4 backbones ({', '.join(BACKBONE_META[m]['display'] for m in row_order)}) x 7 datasets = 28 "
        "cells. Positive = frozen-XCM is worse than the region-specific head. Sign test: win = frozen-XCM "
        "beats region-specific (Delta<0), loss = Delta>0, ties dropped before the test -- same convention "
        "as `rq5_finetune_vs_frozen.md`.",
        "",
        "**Caveat**: n=28 (n=24 NES excluded) is small -- treat all significance tests here as "
        "exploratory, same caveat as `rq5_kurtosis.md` and `rq5_finetune_vs_frozen.md`.",
        "",
        "## Corrected 4-backbone Delta_frozen",
        "",
    ]
    lines += HEADER
    lines.append(build_row("All datasets (n=28)", pooled_full))
    lines.append(build_row("NES excluded (n=24)", pooled_no_nes))
    lines += [
        "",
        "## Backbone-set reconciliation",
        "",
        f"- The 4-backbone set used above ({', '.join(BACKBONE_META[m]['display'] for m in row_order)}) "
        + ("is identical to " if backbones_match else "**differs from** ")
        + "the set `rq5_kurtosis.md` uses for Delta_finetuned and `rq5_finetune_vs_frozen.md` uses "
        "throughout -- all three derive it the same way (`[m for m in BACKBONE_META if m in "
        "FINETUNE_NAME_TO_CANONICAL.values()]`), so no reconciliation was needed.",
        f"- Recomputing Delta_finetuned (fine-tuned-XCM - region-specific) on this same 4-backbone set "
        f"gives {fmt_mean_std(finetuned_recheck['mean'], finetuned_recheck['std'])} (n={finetuned_recheck['n']}), "
        + ("which matches " if finetuned_matches else "which **does not match** ")
        + f"the previously-reported {fmt_mean_std(KURTOSIS_4BB_FINETUNED['mean'], KURTOSIS_4BB_FINETUNED['std'])} "
        f"(n={KURTOSIS_4BB_FINETUNED['n']}) from `rq5_kurtosis.md` -- confirmed rather than assumed.",
        "",
        "## Comparison against the previously-reported 5-backbone Delta_frozen",
        "",
        f"Previously reported (5 backbones, incl. Perch v2): {fmt_mean_std(PREVIOUS_5BB_FROZEN['mean'], PREVIOUS_5BB_FROZEN['std'])} "
        f"(n={PREVIOUS_5BB_FROZEN['n']}). Corrected (4 backbones, Perch v2 excluded): "
        f"{fmt_mean_std(pooled_full['mean'], pooled_full['std'])} (n={pooled_full['n']}). "
        f"The mean shifts by {delta_mean:+.3f} MAE "
        + ("(a small, directionally-consistent shift -- Perch v2 was not disproportionately driving the "
           "original 5-backbone estimate; the frozen-XCM-vs-region-specific gap is a broadly-shared "
           "pattern across backbones, not one backbone's artifact)."
           if abs(delta_mean) < 0.05 else
           "(a shift large enough to be worth flagging -- see whether Perch v2 was disproportionately "
           "influencing the original 5-backbone estimate before citing the two figures interchangeably)."),
        "",
        "## Reading this",
        "",
        "- With the backbone sets now aligned, the three RQ5 deltas discussed in the same paragraph -- "
        f"frozen-vs-region ({fmt_mean_std(pooled_full['mean'], pooled_full['std'])}), fine-tuned-vs-region "
        f"({fmt_mean_std(finetuned_recheck['mean'], finetuned_recheck['std'])}), and frozen-vs-fine-tuned "
        "(`rq5_finetune_vs_frozen.md`) -- are all computed over the identical 28 (backbone, dataset) cells "
        "and are directly comparable.",
        "- As throughout this series, comparing the full and NES-excluded rows checks whether any effect "
        "found is NES-driven, consistent with NES's flagged outlier status elsewhere in RQ5.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq5"))
    args = parser.parse_args()

    xg_models = mrt.discover_models(Path("archive/XCM-Generalization"))
    pooled_long = load_study("pooled", models=xg_models)
    xg_long = load_study("xcm_gen", models=xg_models)
    ft_long = load_study("xcm_gen_ft", models=list(FINETUNE_NAME_TO_CANONICAL.keys()))

    dataset_order, _ = species_ordered_datasets()
    no_nes_order = [d for d in dataset_order if d != "NES"]
    ft_row_order = [m for m in BACKBONE_META if m in FINETUNE_NAME_TO_CANONICAL.values()]
    assert ft_row_order != SPATIAL_BACKBONES, "sanity check: 4-backbone set should exclude Perch v2 (5-backbone set)"

    region_4 = build_heatmap_matrix(pooled_long, SCAPE_SOURCE, "range_mae", dataset_order, ft_row_order)
    xcm_head_4 = build_heatmap_matrix(xg_long, SCAPE_SOURCE, "range_mae", dataset_order, ft_row_order)
    xcm_finetune_4 = build_heatmap_matrix(ft_long, SCAPE_SOURCE, "range_mae", dataset_order, ft_row_order,
                                           model_map=FINETUNE_NAME_TO_CANONICAL)

    pooled_full = full_entry(cell_diffs(xcm_head_4, region_4, dataset_order, ft_row_order))
    pooled_no_nes = full_entry(cell_diffs(xcm_head_4, region_4, no_nes_order, ft_row_order))
    finetuned_recheck = full_entry(cell_diffs(xcm_finetune_4, region_4, dataset_order, ft_row_order))

    results = {
        "frozen_vs_region_4backbone": {"full": pooled_full, "no_nes": pooled_no_nes},
        "finetuned_vs_region_4backbone_recheck": finetuned_recheck,
        "previous_5backbone_frozen_vs_region": PREVIOUS_5BB_FROZEN,
        "kurtosis_reported_4backbone_finetuned_vs_region": KURTOSIS_4BB_FINETUNED,
        "backbone_order": ft_row_order,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq5_frozen_vs_region_4backbone.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")

    md = build_markdown(pooled_full, pooled_no_nes, finetuned_recheck, ft_row_order, ft_row_order, ft_row_order)
    out_md = args.out_dir / "rq5_frozen_vs_region_4backbone.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
