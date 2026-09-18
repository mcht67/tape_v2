#!/usr/bin/env python3
"""
RQ4 (additional) -- Paired significance test of the NES-specific outlier
claim ("region-specific models perform substantially better on NES,
outperforming both frozen and fine-tuned XCM by a wide margin").

Source data: same soundscape_test / range_mae cells used throughout RQ4/
RQ5 (archive/Pooled-Embeddings/ region-specific head, archive/
XCM-Generalization/ frozen-XCM head, archive/XCM-Generalization-fine-tune/
fine-tuned-XCM head), restricted to dataset=NES only. Backbone sets match
compute_rq5_kurtosis.py/compute_rq5_sign_test.py's convention: the 5
SPATIAL_BACKBONES (incl. Perch v2, which has no fine-tuning result -- see
backbone_meta.py's comment on FINETUNE_NAME_TO_CANONICAL) for the
region-specific-vs-frozen-XCM comparison, and the 4 genuinely-fine-tuned
backbones (FINETUNE_NAME_TO_CANONICAL) for the region-specific-vs-fine-
tuned-XCM comparison.

Two paired comparisons, each per-backbone (Delta = other head's NES MAE -
region-specific NES MAE, negative = region-specific wins):
  - Delta_frozen: frozen-XCM vs. region-specific, n=5.
  - Delta_finetuned: fine-tuned-XCM vs. region-specific, n=4.

For each: paired t-test and Wilcoxon signed-rank test against zero (plus
the exact minimum achievable two-sided Wilcoxon p-value at that n, since
n=4/5 gives the exact signed-rank test almost no resolution -- with no
ties, the smallest possible two-sided p is 2^(1-n): 0.125 at n=4, 0.0625
at n=5, reached only when every backbone agrees in sign), and an exact
binomial sign test on the win/loss count (same convention as
compute_rq4_sign_test.py/compute_rq5_sign_test.py's sign_test(), p omitted
below n<=3, not triggered here).

Also reports where NES's polyphony (mean_polyp / ratio_polyp, from
plots/data/polybirdmix_soundscape_stats.json, same source plot_rq4.py's
difficulty table and compute_rq1_mixed_model_difficulty.py use) ranks
among the 7 REGIONAL_DATASETS -- context for whether NES's pattern here is
plausible on independent grounds (an unusually *easy* dataset by
polyphony, yet unusually bad for cross-region heads), not just a
significance-test result.

Writes rq4_nes_outlier_test.json and rq4_nes_outlier_test.md to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq4_nes_outlier_test.py [--out-dir plots/figures/rq4]
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
from plot_rq5 import SCAPE_SOURCE, STATS_JSON, build_heatmap_matrix

NES = "NES"
REGIONAL_DATASETS = ["UHH", "HSN", "PER", "NES", "POW", "SSW", "SNE"]


def wilcoxon_min_p(n: int) -> float:
    """Smallest two-sided p-value the exact Wilcoxon signed-rank test can
    produce at sample size n with no ties/zeros -- reached only when every
    paired difference has the same sign. 2^(1-n); verified against
    scipy.stats.wilcoxon on an all-same-sign vector (n=4 -> 0.125, n=5 ->
    0.0625)."""
    return 2.0 ** (1 - n)


def paired_tests(diff: np.ndarray) -> dict:
    """Paired t-test and Wilcoxon signed-rank test of diff against zero,
    same convention as compute_rq5_kurtosis.py's paired_tests()."""
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


def sign_test(diff: np.ndarray, n_datasets: int) -> dict:
    """Exact sign test: wins = region-specific beats the other head (diff >
    0, since diff = other - region_specific), losses = diff < 0, ties
    dropped. Same convention as compute_rq4_sign_test.py/
    compute_rq5_sign_test.py: p omitted when n_datasets<=3."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    ties = int(len(diff) - wins - losses)
    n_decisive = wins + losses
    if n_datasets <= 3 or n_decisive == 0:
        p_two_sided = None
    else:
        p_two_sided = round(float(scipy.stats.binomtest(wins, n_decisive, 0.5, alternative="two-sided").pvalue), 6)
    return {"n_datasets": n_datasets, "wins": wins, "losses": losses, "ties": ties,
            "n_decisive": n_decisive, "p_two_sided": p_two_sided}


def fmt_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _fmt_p(x):
    return f"{x:.4f}" if x is not None else "n/a, n≤3"


def polyphony_ranking() -> list:
    """NES's rank (1 = least polyphonic/easiest) among the 7
    REGIONAL_DATASETS by mean_polyp and ratio_polyp, from the same stats
    file plot_rq4.py's difficulty table reads."""
    stats = json.loads(STATS_JSON.read_text())["subsets"]
    rows = [(d, stats[d]["polyphony_metric_summary"]["mean_polyp"]["mean"],
             stats[d]["polyphony_metric_summary"]["ratio_polyp"]["mean"]) for d in REGIONAL_DATASETS]
    rows.sort(key=lambda r: r[2])
    return [{"rank": i + 1, "dataset": d, "mean_polyp": round(mp, 4), "ratio_polyp": round(rp, 4)}
            for i, (d, mp, rp) in enumerate(rows)]


def build_markdown(region_5, xcm_5, region_4, ft_4, diff_frozen, diff_ft, test_frozen, test_ft,
                    sign_frozen, sign_ft, poly_rank) -> str:
    lines = [
        "# RQ4 -- Paired significance test of the NES-specific outlier claim",
        "",
        "Tests the manuscript claim that \"region-specific models perform substantially better on NES, "
        "outperforming both frozen and fine-tuned XCM by a wide margin\" as a formal paired comparison, "
        "restricted to NES only, per backbone -- rather than relying on the aggregate mean±SD alone "
        "(region-specific 0.53±0.09, frozen XCM 2.04±0.83, fine-tuned XCM 1.23±0.52). Source: "
        "same soundscape_test/range_mae cells as `rq5_kurtosis.md`/`rq5_sign_test.md`, restricted to "
        "dataset=NES. Delta = (other head's NES MAE) - (region-specific NES MAE); negative = "
        "region-specific wins on that backbone.",
        "",
        "**Caveat**: only 5 backbones have a region-specific/frozen-XCM pair on NES, and only 4 have a "
        "genuine fine-tuned-XCM result (Perch v2 excluded -- see `FINETUNE_NAME_TO_CANONICAL`'s comment), "
        "so both paired tests here run at n=4-5. As with the Perch v2 vs. BEANS baseline case elsewhere "
        "in this series, a non-significant result at this n should not be read as \"no real difference "
        "exists\" -- only that n=4-5 cannot confirm or rule out the effect with confidence. The point "
        "estimates and win/loss pattern below should be read on their own terms, not dressed up with "
        "statistical certainty the sample size doesn't support in either direction.",
        "",
        "## 1. Raw per-backbone NES MAE",
        "",
        "| Backbone | Region-specific | Frozen XCM | Fine-tuned XCM |",
        "|---|---|---|---|",
    ]
    for model in SPATIAL_BACKBONES:
        display = BACKBONE_META[model]["display"]
        rs = region_5.loc[model, NES]
        fx = xcm_5.loc[model, NES]
        ft = ft_4.loc[model, NES] if model in ft_4.index else None
        ft_str = f"{ft:.4f}" if ft is not None and not np.isnan(ft) else "— (excluded)"
        lines.append(f"| {display} | {rs:.4f} | {fx:.4f} | {ft_str} |")

    lines += [
        "",
        "## 2. Paired differences vs. region-specific, NES only",
        "",
        "| Delta | n | Mean ± SD | Paired t-test p | Wilcoxon p |",
        "|---|---|---|---|---|",
        f"| Frozen XCM − region-specific | {len(diff_frozen)} "
        f"| {fmt_mean_std(diff_frozen.mean(), diff_frozen.std(ddof=1))} | {test_frozen['t_p']:.4f} "
        f"| {test_frozen['wilcoxon_p']:.4f} |",
        f"| Fine-tuned XCM − region-specific | {len(diff_ft)} "
        f"| {fmt_mean_std(diff_ft.mean(), diff_ft.std(ddof=1))} | {test_ft['t_p']:.4f} "
        f"| {test_ft['wilcoxon_p']:.4f} |",
        "",
        f"Wilcoxon's minimum achievable two-sided p-value at this n (reached only when every backbone "
        f"agrees in sign, i.e. exactly the win/loss split observed here): **{wilcoxon_min_p(len(diff_frozen)):.4f} "
        f"at n={len(diff_frozen)}**, **{wilcoxon_min_p(len(diff_ft)):.4f} at n={len(diff_ft)}** -- the exact "
        "signed-rank test has almost no resolution at these sample sizes; both Delta rows above already sit "
        "at that floor (all backbones agree in sign), so Wilcoxon cannot report anything more extreme "
        "regardless of how large the gaps are.",
        "",
        "## 3. Exact binomial sign test, NES only",
        "",
        "Wins = region-specific beats the cross-region head on that backbone's NES result.",
        "",
        "| Comparison | n | Wins (region-specific) | Losses | Ties | Sign-test p |",
        "|---|---|---|---|---|---|",
        f"| vs. frozen XCM | {sign_frozen['n_datasets']} | {sign_frozen['wins']} | {sign_frozen['losses']} "
        f"| {sign_frozen['ties']} | {_fmt_p(sign_frozen['p_two_sided'])} |",
        f"| vs. fine-tuned XCM | {sign_ft['n_datasets']} | {sign_ft['wins']} | {sign_ft['losses']} "
        f"| {sign_ft['ties']} | {_fmt_p(sign_ft['p_two_sided'])} |",
        "",
        "## 4. Context check: NES's polyphony rank among the 7 soundscape datasets",
        "",
        "Rank 1 = least polyphonic (nominally easiest) of the 7 REGIONAL_DATASETS, by "
        "`polybirdmix_soundscape_stats.json`'s `mean_polyp`/`ratio_polyp` (same source as `plot_rq4.py`'s "
        "difficulty table).",
        "",
        "| Rank | Dataset | mean_polyp | ratio_polyp |",
        "|---|---|---|---|",
    ]
    for r in poly_rank:
        dataset_str = f"**{r['dataset']}**" if r["dataset"] == NES else r["dataset"]
        lines.append(f"| {r['rank']} | {dataset_str} | {r['mean_polyp']} | {r['ratio_polyp']} |")

    lines += [
        "",
        "## Reading this",
        "",
        "- Unlike several other point-estimate claims checked elsewhere in this series, the direction "
        "here is perfectly consistent: every one of the 5 backbones has region-specific beating frozen "
        "XCM on NES, and every one of the 4 fine-tune-capable backbones has region-specific beating "
        "fine-tuned XCM on NES -- a 5-0 and 4-0 sweep respectively, which is the most extreme win/loss "
        "pattern n=4/5 can produce (hence both Delta rows sit exactly at Wilcoxon's p-floor above).",
        "- The paired t-test reaches conventional significance for the frozen comparison (p<0.05) but not "
        "for the fine-tuned comparison (p≈0.08); Wilcoxon and the sign test cannot distinguish the two "
        "at this n because both already sit at their respective floors -- the t-test result should not be "
        "over-read as stronger evidence than the identical win/loss sweep it's computed from.",
        "- NES ranks as the *least* polyphonically dense (nominally easiest) of the 7 regional datasets by "
        "both polyphony metrics, essentially tied with HSN -- yet it is the dataset on which cross-region "
        "(frozen and fine-tuned XCM) heads do worst relative to region-specific heads. That combination "
        "(easy by an independent difficulty proxy, but the site of the largest region-specific advantage) "
        "is consistent with the claim describing something region-specific about NES rather than NES "
        "simply being a hard dataset in general -- though with only 4-5 backbones, this remains a "
        "plausibility check, not confirmation.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq4"))
    args = parser.parse_args()

    xg_models = mrt.discover_models(Path("archive/XCM-Generalization"))
    pooled_long = load_study("pooled", models=xg_models)
    xg_long = load_study("xcm_gen", models=xg_models)
    ft_long = load_study("xcm_gen_ft", models=list(FINETUNE_NAME_TO_CANONICAL.keys()))

    ft_row_order = [m for m in BACKBONE_META if m in FINETUNE_NAME_TO_CANONICAL.values()]

    region_5 = build_heatmap_matrix(pooled_long, SCAPE_SOURCE, "range_mae", [NES], SPATIAL_BACKBONES)
    xcm_5 = build_heatmap_matrix(xg_long, SCAPE_SOURCE, "range_mae", [NES], SPATIAL_BACKBONES)
    region_4 = build_heatmap_matrix(pooled_long, SCAPE_SOURCE, "range_mae", [NES], ft_row_order)
    ft_4 = build_heatmap_matrix(ft_long, SCAPE_SOURCE, "range_mae", [NES], ft_row_order,
                                 model_map=FINETUNE_NAME_TO_CANONICAL)

    diff_frozen = (xcm_5[NES] - region_5[NES]).reindex(SPATIAL_BACKBONES).to_numpy(dtype=float)
    diff_ft = (ft_4[NES] - region_4[NES]).reindex(ft_row_order).to_numpy(dtype=float)

    test_frozen = paired_tests(diff_frozen)
    test_ft = paired_tests(diff_ft)
    sign_frozen = sign_test(diff_frozen, len(diff_frozen))
    sign_ft = sign_test(diff_ft, len(diff_ft))
    poly_rank = polyphony_ranking()

    results = {
        "raw_nes_mae": {
            "region_specific": region_5[NES].to_dict(),
            "frozen_xcm": xcm_5[NES].to_dict(),
            "fine_tuned_xcm": {k: v for k, v in ft_4[NES].to_dict().items() if not np.isnan(v)},
        },
        "delta_frozen": {"values": diff_frozen.tolist(), "mean": float(diff_frozen.mean()),
                          "std": float(diff_frozen.std(ddof=1)), "n": len(diff_frozen),
                          "wilcoxon_min_p": wilcoxon_min_p(len(diff_frozen)), **test_frozen,
                          "sign_test": sign_frozen},
        "delta_finetuned": {"values": diff_ft.tolist(), "mean": float(diff_ft.mean()),
                             "std": float(diff_ft.std(ddof=1)), "n": len(diff_ft),
                             "wilcoxon_min_p": wilcoxon_min_p(len(diff_ft)), **test_ft,
                             "sign_test": sign_ft},
        "nes_polyphony_rank": poly_rank,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq4_nes_outlier_test.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")

    md = build_markdown(region_5, xcm_5, region_4, ft_4, diff_frozen, diff_ft, test_frozen, test_ft,
                         sign_frozen, sign_ft, poly_rank)
    out_md = args.out_dir / "rq4_nes_outlier_test.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
