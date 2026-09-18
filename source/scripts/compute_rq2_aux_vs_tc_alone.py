#!/usr/bin/env python3
"""
RQ2 (additional) -- Paired cell-level analysis and sign test of the four
TC+auxiliary configurations vs. the bare TC-head alone (not the pooled-MLP
baseline tested by compute_rq2_paired_cell_analysis.py / compute_rq2_sign_
test.py).

Purpose: the RQ2 narrative claims that only frame-level polyphony improves
the TC-head on average, and that polyphony classification "hurts every
backbone" -- both are claims about auxiliary configs relative to the bare
TC-head, not relative to the pooled-MLP baseline those two existing files
test. This file supplies that comparison directly, using the same
TC-head-alone reference point already used (for the figures, not
significance tests) by plot_rq2.py's build_figure()/build_figure_combined()
via compute_tc_alone_indexed().

Delta = (TC+auxiliary MAE) - (bare TC-head MAE), matched by (backbone,
dataset) cell. Negative = the auxiliary configuration improves on the bare
TC-head; positive = it hurts.

Two breakdowns, computed separately for synthetic (8 datasets, ALL_DATASETS)
and soundscape (7 datasets, REGIONAL_DATASETS), for each of the 4 auxiliary
configs (AUX_CONFIGS, reused from compute_rq2_paired_cell_analysis):

1. Pooled across all 5 backbones (n=40/35): mean +/- SD, paired t-test,
   Wilcoxon signed-rank, and exact sign test (win/loss/tie + binomial p) --
   same statistics and same helper functions as the pooled-MLP-baseline
   analyses, just against the TC-alone reference point.

2. Per backbone (n=7/8): same four statistics, so a claim like "hurts every
   backbone" can be checked backbone-by-backbone rather than only on the
   pooled aggregate -- the whole reason this file exists (see the docstring
   above).

Also reproduces, without recomputing, a plain-language reading of
compute_rq2_sign_test.py's existing Check 1 (bare TC-head vs. pooled-MLP
baseline, per backbone): that table's "wins" column counts backbones/
datasets where the bare TC-head beat the pooled-MLP baseline, so a manuscript
claim that the "plain TC-head performs worse... for all backbones" is
checked directly against those win/loss counts here (not recomputed, since
rq2_sign_test.json already has them).

Sample-size caveat, same as compute_rq2_paired_cell_analysis.py and compute_
rq2_sign_test.py: n=5 backbones and up to 7/8 datasets throughout. The
per-backbone (n=7/8) breakdown in particular is exploratory/directional, not
confirmatory -- consistent with the same caveat already established in the
RQ1/RQ2/RQ5 analyses.

Writes rq2_aux_vs_tc_alone.json and rq2_aux_vs_tc_alone.md to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq2_aux_vs_tc_alone.py [--out-dir plots/figures/rq2]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backbone_meta import BACKBONE_META
from compute_rq2_paired_cell_analysis import AUX_CONFIGS, diff_stats, fmt_mean_std, paired_tests
from compute_rq2_sign_test import sign_test, _fmt_p
from plot_data import load_study
from plot_rq2 import SCAPE_SOURCE, SOURCE, SOURCE_DATASETS, SPATIAL_BACKBONES, compute_deltas, compute_tc_alone_indexed

DOMAIN_SOURCE = {"synthetic": SOURCE, "soundscape": SCAPE_SOURCE}
DOMAIN_LABEL = {"synthetic": "Synthetic", "soundscape": "Soundscape"}


def compute_domain_deltas_vs_tc_alone(spatial_long: pd.DataFrame, source: str) -> pd.DataFrame:
    """Same (config, model, dataset, delta) shape as compute_rq2_paired_cell_
    analysis.compute_domain_deltas(), but delta = config MAE - bare TC-head
    MAE (compute_tc_alone_indexed()) instead of - pooled-MLP MAE. Includes
    the bare TC-head's own trivially-zero rows (dropped by callers below via
    AUX_CONFIGS)."""
    tc_alone_indexed = compute_tc_alone_indexed(spatial_long, source)
    return compute_deltas(spatial_long, tc_alone_indexed, source)


def pooled_stats(domain_deltas: pd.DataFrame) -> dict:
    results = {}
    for label in AUX_CONFIGS:
        diff = domain_deltas.loc[domain_deltas["config"] == label, "delta"].to_numpy(dtype=float)
        results[label] = {**diff_stats(diff), **paired_tests(diff), **sign_test(diff, n_datasets=len(diff))}
    return results


def per_backbone_stats(domain_deltas: pd.DataFrame, backbone_order: list, n_datasets: int) -> dict:
    results = {}
    for label in AUX_CONFIGS:
        results[label] = {}
        for model in backbone_order:
            diff = domain_deltas.loc[(domain_deltas["config"] == label) & (domain_deltas["model"] == model),
                                      "delta"].to_numpy(dtype=float)
            results[label][model] = {**diff_stats(diff), **paired_tests(diff), **sign_test(diff, n_datasets)}
    return results


def _fmt_test(t_p, w_p):
    if t_p is None:
        return "--", "--"
    return f"p={t_p:.3f}", (f"p={w_p:.3f}" if w_p is not None else "n/a")


def build_pooled_table(pooled: dict, n_cells: int) -> list:
    lines = [
        f"| Config | Mean Delta MAE (+/- SD) | n | Paired t-test | Wilcoxon | Sign test W-L-T | Sign-test $p$ |",
        "|---|---|---|---|---|---|---|",
    ]
    for label in AUX_CONFIGS:
        s = pooled[label]
        t_p, w_p = _fmt_test(s["t_p"], s["wilcoxon_p"])
        lines.append(f"| {label} | {fmt_mean_std(s['mean'], s['std'])} | {s['n']} | {t_p} | {w_p} "
                     f"| {s['wins']}-{s['losses']}-{s['ties']} | {_fmt_p(s['p_two_sided'])} |")
    return lines


def build_per_backbone_table(per_backbone: dict, label: str, backbone_order: list) -> list:
    lines = [
        "| Backbone | Mean Delta MAE (+/- SD) | n | Paired t-test | Wilcoxon | Sign test W-L-T | Sign-test $p$ |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in backbone_order:
        s = per_backbone[label][model]
        t_p, w_p = _fmt_test(s["t_p"], s["wilcoxon_p"])
        lines.append(f"| {BACKBONE_META[model]['display']} | {fmt_mean_std(s['mean'], s['std'])} | {s['n']} "
                     f"| {t_p} | {w_p} | {s['wins']}-{s['losses']}-{s['ties']} | {_fmt_p(s['p_two_sided'])} |")
    return lines


def build_markdown(all_results: dict, backbone_order: list, tc_head_check: dict) -> str:
    n_datasets = {d: len(SOURCE_DATASETS[DOMAIN_SOURCE[d]]) for d in DOMAIN_SOURCE}
    lines = [
        "# RQ2 -- Auxiliary configurations vs. bare TC-head (not the pooled-MLP baseline)",
        "",
        "Complements `rq2_paired_cell_analysis.md`/`rq2_sign_test.md`, which test each TC-head "
        "configuration against the pooled-MLP (pre-TC-head) baseline. This file instead tests each of "
        "the 4 TC+auxiliary configurations against the *bare TC-head alone* -- the comparison the "
        "manuscript claims \"only frame-level polyphony improves the TC-head on average\" and "
        "\"polyphony classification instead hurts every backbone\" are actually about. Delta = "
        "(TC+auxiliary MAE) - (bare TC-head MAE), matched by (backbone, dataset) cell. Negative = the "
        "auxiliary configuration improves on the bare TC-head. Source: `archive/Spatial-Embeddings/` "
        "only (no pooled-MLP baseline involved) -- same TC-head-alone reference point already used, for "
        "the figures rather than significance tests, by `plot_rq2.py`'s `compute_tc_alone_indexed()` "
        "(the ΔMAE basis of `rq2_multitask`/`rq2_multitask_combined`).",
        "",
        f"**Caveat on sample size**: n=5 backbones and up to {max(n_datasets.values())} datasets "
        "throughout, the same small-n regime as `rq2_paired_cell_analysis.md`/`rq2_sign_test.md`. The "
        "pooled tests (n=35/40 cells) are the best-powered numbers here; the per-backbone tests (n=7/8) "
        "are exploratory/directional only -- a pooled/aggregate significant result does not by itself "
        "establish a per-backbone claim (e.g. \"hurts every backbone\"), which is exactly what the "
        "per-backbone tables below check directly.",
        "",
        "## 1-2. Pooled across all 5 backbones",
        "",
        "Each auxiliary configuration vs. the bare TC-head, pooled across backbones (n = 5 x datasets "
        "per domain).",
        "",
    ]
    for domain in ("synthetic", "soundscape"):
        lines += [f"### {DOMAIN_LABEL[domain]} (n = 5 x {n_datasets[domain]} = {5 * n_datasets[domain]} cells per config)", ""]
        lines += build_pooled_table(all_results[domain]["pooled"], 5 * n_datasets[domain])
        lines.append("")

    lines += [
        "## 3. Per-backbone breakdown",
        "",
        "Same comparison, computed separately for each backbone's own datasets (n=7/8), for each "
        "auxiliary configuration -- this is the table that determines whether a pooled-significant "
        "result also holds \"for every backbone individually\" or only on average.",
        "",
    ]
    for domain in ("synthetic", "soundscape"):
        lines += [f"### {DOMAIN_LABEL[domain]} (n={n_datasets[domain]} datasets per backbone)", ""]
        for label in AUX_CONFIGS:
            lines += [f"**{label}**", ""]
            lines += build_per_backbone_table(all_results[domain]["per_backbone"], label, backbone_order)
            lines.append("")

    # --- Explicit verification of the two headline claims ---
    syn_pooled = all_results["synthetic"]["pooled"]
    syn_perbb = all_results["synthetic"]["per_backbone"]
    fp_label = "+TC-head+frame-level polyphony"
    pc_label = "+TC-head+polyphony classification"
    fp = syn_pooled[fp_label]
    pc = syn_pooled[pc_label]
    pc_positive_every = all(syn_perbb[pc_label][m]["mean"] is not None and syn_perbb[pc_label][m]["mean"] > 0
                             for m in backbone_order)
    pc_worse_count = sum(1 for m in backbone_order
                          if syn_perbb[pc_label][m]["mean"] is not None and syn_perbb[pc_label][m]["mean"] > 0)
    lines += [
        "## Verification: \"only frame-level polyphony improves the TC-head on average, polyphony "
        "classification hurts every backbone\" (synthetic)",
        "",
        f"- Pooled frame-level polyphony vs. TC-alone (n=40): mean Delta = {fp['mean']:.3f} ± {fp['std']:.3f}, "
        f"paired t-test p={fp['t_p']:.3f}, Wilcoxon p={fp['wilcoxon_p']:.3f} -- "
        f"{'a significant improvement (negative Delta)' if fp['mean'] < 0 and fp['t_p'] < 0.05 else 'not both negative and significant'}.",
        f"- Pooled polyphony classification vs. TC-alone (n=40): mean Delta = {pc['mean']:.3f} ± {pc['std']:.3f}, "
        f"paired t-test p={pc['t_p']:.3f}, Wilcoxon p={pc['wilcoxon_p']:.3f} -- "
        f"{'a significant degradation (positive Delta)' if pc['mean'] > 0 and pc['t_p'] < 0.05 else 'not both positive and significant'} "
        "on the pooled aggregate.",
        f"- Per-backbone mean Delta for polyphony classification is positive (i.e. hurts that backbone "
        f"on average) for **{pc_worse_count} of {len(backbone_order)} backbones**: "
        + (", ".join(f"{BACKBONE_META[m]['display']} ({syn_perbb[pc_label][m]['mean']:+.3f})"
                      for m in backbone_order)) + ".",
        f"- \"Hurts every backbone\" (literal, backbone-by-backbone) is "
        f"{'SUPPORTED' if pc_positive_every else 'NOT SUPPORTED'} by this data"
        + ("." if pc_positive_every else
           f" -- {len(backbone_order) - pc_worse_count} of {len(backbone_order)} backbone(s) have a "
           "negative (improving) mean Delta for polyphony classification vs. the bare TC-head, even "
           "though the pooled/aggregate effect across all 5 backbones is positive/significant. This is "
           "a \"true on average, not true for every individual backbone\" case -- the per-backbone "
           "sign-test and mean-Delta tables above give the exact split."),
        "",
    ]

    # --- Check 4: re-verification against existing rq2_sign_test.md Check 1 ---
    lines += [
        "## 4. Re-verification: \"plain TC-head performs worse... for all backbones\" vs. pooled-MLP baseline",
        "",
        "Not recomputed -- reuses `rq2_sign_test.json`'s existing Check 1 (bare TC-head vs. pooled-MLP "
        "baseline, per backbone, synthetic domain) directly. In that table, \"wins\" = the bare TC-head "
        "beat the pooled-MLP baseline (its own delta = TC-head MAE - pooled-MLP MAE was negative on that "
        "dataset); \"losses\" = the TC-head performed worse than the pooled-MLP baseline. A claim that "
        "the plain TC-head \"performs worse... for all backbones\" therefore predicts a losses-majority "
        "(or all-loss) row for every one of the 5 backbones.",
        "",
        "| Backbone | n datasets | Wins (TC-head beat pooled-MLP) | Losses (TC-head worse) | Sign-test $p$ "
        "| TC-head worse on majority? |",
        "|---|---|---|---|---|---|",
    ]
    for model in backbone_order:
        r = tc_head_check[model]
        majority_worse = r["losses"] > r["wins"]
        lines.append(f"| {BACKBONE_META[model]['display']} | {r['n_datasets']} | {r['wins']} | {r['losses']} "
                     f"| {_fmt_p(r['p_two_sided'])} | {'yes' if majority_worse else 'NO'} |")
    n_majority_worse = sum(1 for m in backbone_order if tc_head_check[m]["losses"] > tc_head_check[m]["wins"])
    exceptions = [BACKBONE_META[m]["display"] for m in backbone_order if tc_head_check[m]["losses"] <= tc_head_check[m]["wins"]]
    lines += [
        "",
        f"**{n_majority_worse} of {len(backbone_order)} backbones** have the bare TC-head losing to the "
        "pooled-MLP baseline on a majority of synthetic datasets"
        + (f"; the exception is **{', '.join(exceptions)}**." if exceptions else "."),
        "",
    ]
    if exceptions:
        exc_rows = "; ".join(
            f"{BACKBONE_META[m]['display']} ({tc_head_check[m]['wins']}W-{tc_head_check[m]['losses']}L, "
            f"p={_fmt_p(tc_head_check[m]['p_two_sided'])})"
            for m in backbone_order if tc_head_check[m]["losses"] <= tc_head_check[m]["wins"]
        )
        lines += [
            f"A claim of the plain TC-head performing worse than the pooled-MLP baseline \"for all "
            f"backbones\" is **not fully supported**: {exc_rows} does not show a losses-majority (its "
            "sign test is also not individually significant). The data supports a precise claim of the "
            f"form \"for {n_majority_worse} of {len(backbone_order)} backbones\" (naming the exception(s) "
            "above), not \"for all backbones.\"",
            "",
        ]

    lines += [
        "## Reading this",
        "",
        "- Sections 1-2 (pooled, n=35/40) are the best-powered tests in this file and the most trustworthy "
        "signal for an \"on average\" claim.",
        "- Section 3 (per-backbone, n=7/8) is exploratory/directional -- but it is the only table that can "
        "confirm or refute a literal \"every backbone\" claim, since a pooled significant result can be "
        "driven by a subset of backbones.",
        "- Section 4 does not recompute anything; it re-reads `rq2_sign_test.md`'s existing Check 1 table "
        "to check the exact wording of a separate claim (TC-head vs. pooled-MLP, not vs. auxiliary "
        "configs) against data already on disk.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq2"))
    args = parser.parse_args()

    spatial_long = load_study("spatial", models=SPATIAL_BACKBONES)
    backbone_order = [m for m in BACKBONE_META if m in SPATIAL_BACKBONES]

    domain_deltas = {d: compute_domain_deltas_vs_tc_alone(spatial_long, DOMAIN_SOURCE[d]) for d in DOMAIN_SOURCE}
    n_datasets = {d: len(SOURCE_DATASETS[DOMAIN_SOURCE[d]]) for d in DOMAIN_SOURCE}

    all_results = {}
    for d in DOMAIN_SOURCE:
        pooled = pooled_stats(domain_deltas[d])
        per_backbone = per_backbone_stats(domain_deltas[d], backbone_order, n_datasets[d])
        all_results[d] = {"pooled": pooled, "per_backbone": per_backbone}

    # Reuse existing Check 1 (bare TC-head vs. pooled-MLP) from rq2_sign_test.json, synthetic only.
    sign_test_json = args.out_dir / "rq2_sign_test.json"
    with open(sign_test_json) as f:
        existing = json.load(f)
    tc_head_check = existing["check1_bare_tc_head"]["synthetic"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq2_aux_vs_tc_alone.json"
    out_json.write_text(json.dumps({"domain_results": all_results, "tc_head_vs_pooled_mlp_check_reused": tc_head_check},
                                    indent=2, default=str))
    print(f"Wrote {out_json}")

    md = build_markdown(all_results, backbone_order, tc_head_check)
    out_md = args.out_dir / "rq2_aux_vs_tc_alone.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
