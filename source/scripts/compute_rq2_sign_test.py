#!/usr/bin/env python3
"""
RQ2 (additional) -- Exact sign test, magnitude-independent complement to
compute_rq2_paired_cell_analysis.py's analyses 1 and 2 (paired t-test/
Wilcoxon on the cell-level delta vs. the pooled-MLP baseline).

Purpose: analysis 1/2's aggregate paired-difference results could in
principle be driven by a subset of datasets rather than reflecting a
consistent per-dataset direction -- this checks per backbone whether the
config actually wins on most/all of that backbone's own datasets,
independent of how large those per-dataset deltas are.

Same source data as compute_rq2_paired_cell_analysis.py
(archive/Spatial-Embeddings/ vs. archive/Pooled-Embeddings/'s pooled-MLP
baseline, SPATIAL_BACKBONES, CONFIG_LABELS/CONFIG_HEAD from plot_rq2.py),
computed separately for synthetic (8 datasets, ALL_DATASETS) and soundscape
(7 datasets, REGIONAL_DATASETS):

1. Bare TC-head vs. pooled-MLP baseline, per backbone (5 backbones x 2
   domains = 10 rows): sign of delta = TC-head MAE - pooled-MLP MAE, across
   that backbone's own datasets. Complements analysis 1 of
   compute_rq2_paired_cell_analysis.py (Check 1 in the RQ2 narrative).

2. Each backbone's best-auxiliary config (per
   compute_rq2_paired_cell_analysis.py's analysis 3 -- lowest mean MAE among
   AUX_CONFIGS for that backbone/domain) vs. pooled-MLP baseline, per
   backbone (10 rows). Complements analysis 2 (Check 2 in the RQ2
   narrative).

Sample-size caveat, same as compute_rq2_paired_cell_analysis.py: n=7/8
datasets per backbone is small -- results here are exploratory/directional,
per the task's n<=3 rule no p-value would be reported below that, though
n=7/8 clears it.

Writes rq2_sign_test.json and rq2_sign_test.md to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq2_sign_test.py [--out-dir plots/figures/rq2]
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
from compute_rq2_paired_cell_analysis import best_aux_per_backbone, compute_domain_deltas
from plot_data import load_study
from plot_rq2 import CONFIG_LABELS, SCAPE_SOURCE, SOURCE, SOURCE_DATASETS, SPATIAL_BACKBONES

DOMAIN_SOURCE = {"synthetic": SOURCE, "soundscape": SCAPE_SOURCE}
DOMAIN_LABEL = {"synthetic": "Synthetic", "soundscape": "Soundscape"}
BARE_TC_HEAD = "+TC-head"


def sign_test(diff: np.ndarray, n_datasets: int) -> dict:
    """Exact sign test: wins = config beats baseline (delta < 0, lower
    MAE), losses = delta > 0, ties = delta == 0 (dropped before the test).
    p-value omitted when n_datasets<=3, per the task's small-n rule."""
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


def per_backbone_sign_tests(domain_deltas: pd.DataFrame, config_by_backbone: dict, backbone_order: list,
                             n_datasets: int) -> dict:
    """config_by_backbone: {model: config_label} -- either the constant
    BARE_TC_HEAD for every backbone (analysis 1) or each backbone's own
    best-auxiliary config (analysis 2)."""
    results = {}
    for model in backbone_order:
        label = config_by_backbone[model]
        diff = domain_deltas.loc[(domain_deltas["config"] == label) & (domain_deltas["model"] == model),
                                  "delta"].to_numpy(dtype=float)
        results[model] = {"config": label, **sign_test(diff, n_datasets)}
    return results


def _fmt_p(x):
    return f"{x:.4f}" if x is not None else "n/a, n≤3"


def build_table(results: dict, backbone_order: list, show_config: bool) -> list:
    header = "| Backbone | Config | n datasets | Wins | Losses | Ties | Sign-test $p$ |" if show_config else \
        "| Backbone | n datasets | Wins | Losses | Ties | Sign-test $p$ |"
    sep = "|---|---|---|---|---|---|---|" if show_config else "|---|---|---|---|---|---|"
    lines = [header, sep]
    for model in backbone_order:
        r = results[model]
        display = BACKBONE_META[model]["display"]
        if show_config:
            lines.append(f"| {display} | {r['config']} | {r['n_datasets']} | {r['wins']} | {r['losses']} "
                         f"| {r['ties']} | {_fmt_p(r['p_two_sided'])} |")
        else:
            lines.append(f"| {display} | {r['n_datasets']} | {r['wins']} | {r['losses']} "
                         f"| {r['ties']} | {_fmt_p(r['p_two_sided'])} |")
    return lines


def build_markdown(check1: dict, check2: dict, backbone_order: list) -> str:
    lines = [
        "# RQ2 -- Exact sign test, TC-head and best-auxiliary configs vs. pooled-MLP baseline",
        "",
        "Magnitude-independent complement to `rq2_paired_cell_analysis.md` analyses 1 and 2 (paired "
        "t-test/Wilcoxon on the cell-level delta): per backbone, does the config actually win on "
        "most/all of that backbone's own datasets, independent of how large those per-dataset deltas "
        "are -- complements, does not replace, the mean-difference tests. Source: "
        "`archive/Spatial-Embeddings/` vs. `archive/Pooled-Embeddings/`'s pooled-MLP baseline, the 5 "
        "SPATIAL_BACKBONES, synthetic (8 ALL_DATASETS) and soundscape (7 REGIONAL_DATASETS) domains "
        "computed separately.",
        "",
        "**Caveat**: n=7/8 datasets per backbone is small for a sign test -- treat as "
        "exploratory/directional, consistent with the same caveat in `rq2_paired_cell_analysis.md`. "
        "Where a sign-test result here diverges from the corresponding paired t-test/Wilcoxon result "
        "there, that combination is flagged explicitly below.",
        "",
        "## 1. Bare TC-head vs. pooled-MLP baseline",
        "",
        "Complements Check 1 (analysis 1) of `rq2_paired_cell_analysis.md`.",
        "",
    ]
    for domain in DOMAIN_SOURCE:
        lines += [f"### {DOMAIN_LABEL[domain]}", ""]
        lines += build_table(check1[domain], backbone_order, show_config=False)
        lines.append("")

    lines += [
        "## 2. Each backbone's best-auxiliary config vs. pooled-MLP baseline",
        "",
        "\"Best\" per (backbone, domain) is the auxiliary config with the lowest mean raw MAE, per "
        "`rq2_paired_cell_analysis.md`'s analysis 3 -- reused here (not recomputed), so the config "
        "named in each row matches that file's per-backbone breakdown (Check 2 / analysis 2) exactly.",
        "",
    ]
    for domain in DOMAIN_SOURCE:
        lines += [f"### {DOMAIN_LABEL[domain]}", ""]
        lines += build_table(check2[domain], backbone_order, show_config=True)
        lines.append("")

    lines += [
        "## Reading this",
        "",
        "- A backbone with a lopsided win/loss split (e.g. 7-0 or 6-1) but a non-significant paired "
        "t-test/Wilcoxon in `rq2_paired_cell_analysis.md` would indicate a consistent-but-small "
        "per-dataset direction that the mean-difference test's sensitivity to high-variance datasets "
        "obscured.",
        "- A roughly even split (e.g. 4-3, 3-4) alongside a non-significant mean-difference test "
        "confirms there is no hidden consistent direction being missed.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq2"))
    args = parser.parse_args()

    pooled_long = load_study("pooled", models=SPATIAL_BACKBONES)
    spatial_long = load_study("spatial", models=SPATIAL_BACKBONES)
    backbone_order = [m for m in BACKBONE_META if m in SPATIAL_BACKBONES]

    domain_deltas = {domain: compute_domain_deltas(pooled_long, spatial_long, source)
                      for domain, source in DOMAIN_SOURCE.items()}
    n_datasets = {domain: len(SOURCE_DATASETS[source]) for domain, source in DOMAIN_SOURCE.items()}

    check1 = {
        domain: per_backbone_sign_tests(domain_deltas[domain], {m: BARE_TC_HEAD for m in backbone_order},
                                         backbone_order, n_datasets[domain])
        for domain in DOMAIN_SOURCE
    }

    best_by_domain = {domain: best_aux_per_backbone(domain_deltas[domain], backbone_order)
                       for domain in DOMAIN_SOURCE}
    check2 = {
        domain: per_backbone_sign_tests(domain_deltas[domain], best_by_domain[domain], backbone_order,
                                         n_datasets[domain])
        for domain in DOMAIN_SOURCE
    }

    results = {"check1_bare_tc_head": check1, "check2_best_aux": check2}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq2_sign_test.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")

    md = build_markdown(check1, check2, backbone_order)
    out_md = args.out_dir / "rq2_sign_test.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
