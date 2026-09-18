#!/usr/bin/env python3
"""
RQ5 (additional) -- Exact sign test, magnitude-independent complement to
compute_rq5_kurtosis.py's cell-level Delta_frozen/Delta_finetuned paired
t-test/Wilcoxon analysis.

Purpose: Delta_frozen/Delta_finetuned's paired-difference tests could in
principle be driven by a subset of datasets (NES was flagged elsewhere in
RQ5 as an extreme outlier) rather than reflecting a consistent per-dataset
direction -- this checks, per backbone, whether the cross-region head
(frozen-XCM or fine-tuned-XCM) actually loses to the region-specific head
on most/all of the 7 soundscape datasets, independent of how large those
per-dataset gaps are, and whether dropping NES changes that pattern.

Same source data and cell-level pairing as compute_rq5_kurtosis.py
(archive/Pooled-Embeddings/ region-specific head, archive/XCM-Generalization/
frozen-XCM head, archive/XCM-Generalization-fine-tune/ fine-tuned-XCM head;
soundscape_test / range_mae; SPATIAL_BACKBONES (5, incl. Perch v2) for the
frozen comparison, the 4 genuinely-fine-tuned backbones
(FINETUNE_NAME_TO_CANONICAL) for the fine-tuned comparison):

1. Frozen-XCM vs. region-specific, per backbone (5 backbones): sign of
   (XCM-head MAE - region-specific MAE) across the 7 REGIONAL_DATASETS,
   with and without NES.

2. Fine-tuned-XCM vs. region-specific, per backbone (4 backbones): same,
   restricted to the 4 backbones with genuine fine-tuning results.

Both reported with all 7 datasets and with NES excluded (n=6), side by
side, so the NES sensitivity check the task asks for ("confirm whether
excluding NES changes any of the above sign-test win/loss patterns
substantially") is read directly off the same table rather than a separate
one.

Writes rq5_sign_test.json and rq5_sign_test.md to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq5_sign_test.py [--out-dir plots/figures/rq5]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import BACKBONE_META, FINETUNE_NAME_TO_CANONICAL
from plot_data import load_study
from plot_rq2 import SPATIAL_BACKBONES
from plot_rq5 import SCAPE_SOURCE, build_heatmap_matrix, species_ordered_datasets


def sign_test(diff: np.ndarray, n_datasets: int) -> dict:
    """Exact sign test: wins = cross-region head beats region-specific
    (diff < 0, lower MAE), losses = diff > 0, ties = diff == 0 (dropped
    before the test). p-value omitted when n_datasets<=3, per the task's
    small-n rule (not triggered here -- n=6/7 throughout)."""
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


def per_backbone_sign_tests(other_matrix: pd.DataFrame, region_matrix: pd.DataFrame, row_order: list,
                             dataset_order: list, no_nes_order: list) -> dict:
    results = {}
    for model in row_order:
        diff_full = (other_matrix.loc[model, dataset_order] - region_matrix.loc[model, dataset_order]).to_numpy(dtype=float)
        diff_no_nes = (other_matrix.loc[model, no_nes_order] - region_matrix.loc[model, no_nes_order]).to_numpy(dtype=float)
        results[model] = {
            "full": sign_test(diff_full, len(dataset_order)),
            "no_nes": sign_test(diff_no_nes, len(no_nes_order)),
        }
    return results


def _fmt_p(x):
    return f"{x:.4f}" if x is not None else "n/a, n≤3"


def _cell(r: dict) -> str:
    return f"{r['losses']}-{r['wins']}-{r['ties']} (p={_fmt_p(r['p_two_sided'])})"


def build_table(results: dict, row_order: list) -> list:
    lines = [
        "| Backbone | n (full) | Losses-Wins-Ties (full) | n (no NES) | Losses-Wins-Ties (no NES) |",
        "|---|---|---|---|---|",
    ]
    for model in row_order:
        r = results[model]
        lines.append(f"| {BACKBONE_META[model]['display']} | {r['full']['n_datasets']} "
                     f"| {_cell(r['full'])} | {r['no_nes']['n_datasets']} | {_cell(r['no_nes'])} |")
    return lines


def build_markdown(check_frozen: dict, check_finetuned: dict, frozen_order: list, ft_order: list,
                    dataset_order: list) -> str:
    lines = [
        "# RQ5 -- Exact sign test, cross-region heads vs. region-specific head",
        "",
        "Magnitude-independent complement to `rq5_kurtosis.md`'s cell-level Delta_frozen/Delta_finetuned "
        "paired t-test/Wilcoxon: per backbone, does the cross-region head (frozen-XCM or fine-tuned-XCM) "
        "actually lose to the region-specific head on most/all of the 7 soundscape datasets, independent "
        "of how large those per-dataset gaps are -- and does dropping NES (flagged elsewhere in RQ5 as "
        "an extreme outlier) change that pattern. \"Losses\" = cross-region head worse (higher MAE) than "
        "region-specific on that dataset; \"Wins\" = cross-region head better. Source: same cell-level "
        "pairing as `compute_rq5_kurtosis.py` (region-specific: `archive/Pooled-Embeddings/`; frozen-XCM: "
        "`archive/XCM-Generalization/`; fine-tuned-XCM: `archive/XCM-Generalization-fine-tune/`; "
        "soundscape_test / range_mae).",
        "",
        "**Caveat**: n=6/7 datasets per backbone is small for a sign test -- treat as "
        "exploratory/directional, consistent with the same caveat in `rq5_kurtosis.md`. Where a "
        "sign-test result here diverges from the corresponding paired t-test/Wilcoxon result there "
        "(Delta_frozen/Delta_finetuned), that combination is flagged explicitly below.",
        "",
        "## 1. Frozen-XCM vs. region-specific, per backbone",
        "",
        f"5 backbones (SPATIAL_BACKBONES, incl. Perch v2 -- frozen embeddings don't have Perch v2's "
        "gradient problem). Complements Delta_frozen in `rq5_kurtosis.md`.",
        "",
    ]
    lines += build_table(check_frozen, frozen_order)
    lines += [
        "",
        "## 2. Fine-tuned-XCM vs. region-specific, per backbone",
        "",
        "4 backbones with genuine fine-tuning results (Perch v2 excluded -- see "
        "`FINETUNE_NAME_TO_CANONICAL`'s comment). Complements Delta_finetuned in `rq5_kurtosis.md`.",
        "",
    ]
    lines += build_table(check_finetuned, ft_order)
    lines += [
        "",
        "## Reading this",
        "",
        "- A backbone with a lopsided full-column split (e.g. 6-1 or 7-0 losses) but a non-significant "
        "paired t-test/Wilcoxon for the corresponding Delta in `rq5_kurtosis.md` would indicate a "
        "consistent-but-small per-dataset direction that the mean-difference test's sensitivity to "
        "high-variance datasets (NES especially) obscured.",
        "- Comparing the full and no-NES columns directly answers the NES-sensitivity check: if the "
        "win/loss split is materially different with NES dropped (e.g. 6-1 becomes 3-3), NES was driving "
        "the apparent consistency; if it barely moves, the pattern holds independent of NES.",
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

    region_5 = build_heatmap_matrix(pooled_long, SCAPE_SOURCE, "range_mae", dataset_order, SPATIAL_BACKBONES)
    xcm_head_5 = build_heatmap_matrix(xg_long, SCAPE_SOURCE, "range_mae", dataset_order, SPATIAL_BACKBONES)
    region_4 = build_heatmap_matrix(pooled_long, SCAPE_SOURCE, "range_mae", dataset_order, ft_row_order)
    xcm_finetune_4 = build_heatmap_matrix(ft_long, SCAPE_SOURCE, "range_mae", dataset_order, ft_row_order,
                                           model_map=FINETUNE_NAME_TO_CANONICAL)

    check_frozen = per_backbone_sign_tests(xcm_head_5, region_5, SPATIAL_BACKBONES, dataset_order, no_nes_order)
    check_finetuned = per_backbone_sign_tests(xcm_finetune_4, region_4, ft_row_order, dataset_order, no_nes_order)

    results = {"frozen_vs_region_specific": check_frozen, "finetuned_vs_region_specific": check_finetuned}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq5_sign_test.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")

    md = build_markdown(check_frozen, check_finetuned, SPATIAL_BACKBONES, ft_row_order, dataset_order)
    out_md = args.out_dir / "rq5_sign_test.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
