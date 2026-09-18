#!/usr/bin/env python3
"""
RQ5 (additional) -- Paired cell-level comparison, frozen-XCM vs.
fine-tuned-XCM directly (not each against the region-specific head).

Completes the RQ5 three-way comparison: `compute_rq5_kurtosis.py`'s
Delta_frozen (frozen-XCM - region-specific) and Delta_finetuned
(fine-tuned-XCM - region-specific), and `compute_rq5_sign_test.py`'s
magnitude-independent complements to both, already cover
frozen-vs-region-specific and fine-tuned-vs-region-specific. The one
pairwise comparison not yet run directly is frozen-XCM vs. fine-tuned-XCM
itself. The point estimates from `rq5_kurtosis.json`
(cell_diff_vs_region_specific: frozen mean gap 0.151, n=35;
finetuned mean gap 0.035, n=28) suggest frozen and fine-tuned are closer to
each other than either is to the region-specific head -- this checks
whether that specific difference is itself statistically distinguishable
from zero.

Same source data and cell-level pairing as compute_rq5_kurtosis.py /
compute_rq5_sign_test.py (archive/XCM-Generalization/ frozen-XCM head,
archive/XCM-Generalization-fine-tune/ fine-tuned-XCM head; soundscape_test /
range_mae), restricted to the 4 backbones with genuine fine-tuning results
(FINETUNE_NAME_TO_CANONICAL; Perch v2 excluded -- see its comment) x 7
REGIONAL_DATASETS = 28 cells, matching compute_rq5_kurtosis.py's
Delta_finetuned sample.

Delta = (fine-tuned-XCM MAE) - (frozen-XCM MAE), same (backbone, dataset)
cell. Negative = fine-tuning improves on the frozen backbone. Reported as
mean +/- SD, a paired t-test and Wilcoxon signed-rank test against zero
(same convention as compute_rq5_kurtosis.py's cell-level Delta tests), and
an exact two-sided binomial sign test on win/loss/tie counts (same
convention as compute_rq5_sign_test.py's per-backbone sign tests: win =
Delta<0, loss = Delta>0, ties dropped before the test) -- pooled (n=28),
per backbone (n=7), and with NES excluded (n=24 pooled, n=6 per backbone),
NES being the extreme-outlier dataset flagged elsewhere in RQ5.

**Caveat**: n=28 pooled (n=7 per backbone) is small -- per-backbone results
are exploratory, same caveat as compute_rq5_kurtosis.py's Delta_finetuned
per-backbone breakdown and compute_rq5_sign_test.py's per-backbone sign
tests.

Writes rq5_finetune_vs_frozen.json and rq5_finetune_vs_frozen.md to
--out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq5_finetune_vs_frozen.py [--out-dir plots/figures/rq5]
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
from plot_rq5 import SCAPE_SOURCE, build_heatmap_matrix, species_ordered_datasets

FROZEN_LABEL = "Frozen-XCM"
FINETUNE_LABEL = "Fine-tuned-XCM"


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
    """Exact sign test: wins = fine-tuning beats frozen (diff<0, lower
    MAE), losses = diff>0, ties = diff==0 (dropped before the test). Same
    convention as compute_rq5_sign_test.py's sign_test()."""
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


def build_markdown(pooled_full: dict, pooled_no_nes: dict, per_backbone_full: dict, per_backbone_no_nes: dict,
                    ft_row_order: list) -> str:
    lines = [
        "# RQ5 -- Paired cell-level comparison, frozen-XCM vs. fine-tuned-XCM",
        "",
        "Completes the RQ5 three-way comparison: `rq5_kurtosis.md` and `rq5_sign_test.md` already cover "
        "frozen-XCM-vs-region-specific and fine-tuned-XCM-vs-region-specific; this is the remaining "
        "pairwise comparison, frozen-XCM directly against fine-tuned-XCM. The point estimates from "
        "`rq5_kurtosis.md` (mean gap to region-specific: 0.151 frozen, 0.035 fine-tuned) suggest frozen "
        "and fine-tuned are closer to each other than either is to the region-specific head -- this "
        "checks whether that specific difference is itself statistically distinguishable from zero.",
        "",
        "Delta = (fine-tuned-XCM MAE) - (frozen-XCM MAE), matched by (backbone, dataset) cell. Negative = "
        "fine-tuning improves on the frozen backbone. Source: same cell-level pairing as "
        "`compute_rq5_kurtosis.py`'s Delta_finetuned (frozen-XCM: `archive/XCM-Generalization/`; "
        "fine-tuned-XCM: `archive/XCM-Generalization-fine-tune/`; soundscape_test / range_mae), the 4 "
        f"backbones with genuine fine-tuning results ({', '.join(BACKBONE_META[m]['display'] for m in ft_row_order)}"
        "; Perch v2 excluded -- see `FINETUNE_NAME_TO_CANONICAL`'s comment) x 7 datasets = 28 cells. Sign "
        "test: win = fine-tuning beats frozen on that cell (Delta<0), loss = Delta>0, ties dropped before "
        "the test -- same convention as `compute_rq5_sign_test.py`.",
        "",
        "**Caveat**: n=28 pooled (n=7 per backbone) is small -- treat per-backbone results as exploratory, "
        "same caveat as the corresponding breakdowns in `rq5_kurtosis.md` and `rq5_sign_test.md`.",
        "",
        "**Note**: the pooled Delta below is not simply (0.035 - 0.151) from the point estimates above -- "
        "`rq5_kurtosis.md`'s 0.151 frozen gap is computed over 5 backbones (35 cells, including Perch v2, "
        "which has no fine-tuned result); this comparison is restricted to the 4 backbones common to both "
        "frozen and fine-tuned results, so both sides of the Delta below use the same 28 cells.",
        "",
        "## Pooled (all 4 backbones)",
        "",
    ]
    lines += HEADER
    lines.append(build_row("All datasets (n=28)", pooled_full))
    lines.append(build_row("NES excluded (n=24)", pooled_no_nes))
    lines += [
        "",
        "## Per backbone",
        "",
    ]
    lines += HEADER
    for m in ft_row_order:
        display = BACKBONE_META[m]["display"]
        lines.append(build_row(f"{display} (all 7 datasets)", per_backbone_full[m]))
        lines.append(build_row(f"{display} (NES excluded, 6 datasets)", per_backbone_no_nes[m]))
    lines += [
        "",
        "## Reading this",
        "",
        "- The pooled row is the best-powered test here (n=28); the per-backbone rows (n=7 or 6) are "
        "exploratory and a single dataset can swing any one of them.",
        "- This tests whether fine-tuning vs. frozen shows a clearer, more confirmable effect than either "
        "comparison against the region-specific head did in `rq5_kurtosis.md` (both non-significant "
        "there, t-test p=0.246 frozen, p=0.701 fine-tuned, pooled). A smaller apparent gap here (0.035 "
        "vs. frozen's 0.151 gap to region-specific) could mean either less room for a detectable "
        "difference, or a more consistent/lower-variance improvement that a paired test picks up despite "
        "the smaller magnitude -- the mean/SD and p-values above distinguish which.",
        "- Comparing the full and NES-excluded rows (pooled and per backbone) checks whether any effect "
        "found is NES-driven, consistent with NES's flagged outlier status elsewhere in RQ5.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq5"))
    args = parser.parse_args()

    xg_models = mrt.discover_models(Path("archive/XCM-Generalization"))
    xg_long = load_study("xcm_gen", models=xg_models)
    ft_long = load_study("xcm_gen_ft", models=list(FINETUNE_NAME_TO_CANONICAL.keys()))

    dataset_order, _ = species_ordered_datasets()
    no_nes_order = [d for d in dataset_order if d != "NES"]
    ft_row_order = [m for m in BACKBONE_META if m in FINETUNE_NAME_TO_CANONICAL.values()]

    frozen_4 = build_heatmap_matrix(xg_long, SCAPE_SOURCE, "range_mae", dataset_order, ft_row_order)
    finetune_4 = build_heatmap_matrix(ft_long, SCAPE_SOURCE, "range_mae", dataset_order, ft_row_order,
                                       model_map=FINETUNE_NAME_TO_CANONICAL)

    pooled_full = full_entry(cell_diffs(finetune_4, frozen_4, dataset_order, ft_row_order))
    pooled_no_nes = full_entry(cell_diffs(finetune_4, frozen_4, no_nes_order, ft_row_order))
    per_backbone_full = {m: full_entry(cell_diffs(finetune_4, frozen_4, dataset_order, [m])) for m in ft_row_order}
    per_backbone_no_nes = {m: full_entry(cell_diffs(finetune_4, frozen_4, no_nes_order, [m])) for m in ft_row_order}

    results = {
        "pooled": {"full": pooled_full, "no_nes": pooled_no_nes},
        "per_backbone": {m: {"full": per_backbone_full[m], "no_nes": per_backbone_no_nes[m]} for m in ft_row_order},
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq5_finetune_vs_frozen.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")

    md = build_markdown(pooled_full, pooled_no_nes, per_backbone_full, per_backbone_no_nes, ft_row_order)
    out_md = args.out_dir / "rq5_finetune_vs_frozen.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
