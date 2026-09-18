#!/usr/bin/env python3
"""
RQ5 (additional) -- Same Pearson correlations as compute_rq5_correlations.py
(per-dataset MAE vs. each dataset property, across the 7 regional datasets),
but with "region_specific" and "xcm_head" restricted to the same 4
backbones as "xcm_finetune_head" (which was always only 4, perch_v2
excluded -- see backbone_meta.py's comment on FINETUNE_NAME_TO_CANONICAL),
instead of averaged over all 11 Panel-A backbones -- so all three heads are
compared over an identical backbone population, the same "matched"
restriction as rq5_panel_a_soundscape_matched / rq5_panel_b_soundscape_
*_matched / build_panel_a(matched_backbones=...) elsewhere in RQ5.

Does not touch compute_rq5_correlations.py's own outputs (rq5_correlations.
json, rq5_correlations_table.tex) -- writes separate rq5_matched_
correlations.json and rq5_matched_correlations_table.tex instead. The
dataset-property-*values* table (rq5_properties_table.tex) is unaffected by
backbone choice, so it isn't duplicated here.

rq5_matched_correlations_table.tex also carries a leave-one-out robustness
check (same correlations recomputed with NES excluded, n=6) below the real
table, commented out (every line prefixed "% ") so it's available in the
source for reference without being part of the compiled document or
duplicating a 2nd real table -- NES is the one consistent outlier found
throughout RQ5 (see plot_rq5.py's confusion matrices/heatmaps), so this is
the natural single dataset to check robustness against.

Usage:
    complete-venv/bin/python source/scripts/compute_rq5_matched_correlations.py [--out plots/figures/rq5/rq5_matched_correlations.json]
"""

import argparse
import json
import sys
from pathlib import Path

import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import FINETUNE_NAME_TO_CANONICAL
from compute_rq5_correlations import HEAD_LABEL, TEST_LABEL, TEST_TYPES, per_dataset_mae
from plot_data import REGIONAL_DATASETS, load_study
from plot_rq5 import STATS_JSON
from plot_rq5_dataset_properties import PROPERTIES


def compute_correlations(long_dfs: dict, property_values: dict, dataset_subset: list) -> dict:
    """Same computation as main()'s original loop, generalized to an
    arbitrary dataset_subset -- the full 7 regional datasets for the real
    table, or a leave-one-out subset for the robustness check appended
    below it."""
    results = {}
    for head_key, long_df in long_dfs.items():
        results[head_key] = {}
        for test_key, (source, metric) in TEST_TYPES.items():
            mae = per_dataset_mae(long_df, source, metric).reindex(dataset_subset)
            results[head_key][test_key] = {}
            for prop_key, values_by_dataset in property_values.items():
                prop_values = [values_by_dataset[d] for d in dataset_subset]
                r = scipy.stats.pearsonr(prop_values, mae.values)
                results[head_key][test_key][prop_key] = {
                    "pearson_r": round(float(r.statistic), 4),
                    "p_value": round(float(r.pvalue), 4),
                    "n": len(dataset_subset),
                }
    return results


def build_matched_correlations_table(results, n_datasets: int = 7, note: str = "") -> str:
    """Same layout as compute_rq5_correlations.build_correlations_table(),
    with a caption noting the backbone restriction (and, for the
    leave-one-out check, which dataset was dropped)."""
    heads = list(HEAD_LABEL)
    tests = list(TEST_LABEL)
    header = "Property & " + " & ".join(
        f"{HEAD_LABEL[h]} ({TEST_LABEL[t]})" for h in heads for t in tests
    ) + r" \\"
    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{Pearson correlation ($r$, with $p$-value) between each dataset property and "
        rf"per-dataset mean MAE, across the {n_datasets} regional datasets ($n={n_datasets}$){note} -- "
        r"Region-specific and XCM head restricted to the same 4 backbones as XCM fine-tune head. "
        r"Bold: $p < 0.05$.}",
        r"\label{tab:rq5_matched_correlations}",
        rf"\begin{{tabular}}{{l{'c' * (len(heads) * len(tests))}}}", r"\toprule", header, r"\midrule",
    ]
    for prop_key, (label, _, _) in PROPERTIES.items():
        row = []
        for h in heads:
            for t in tests:
                stat = results[h][t][prop_key]
                cell = f"${stat['pearson_r']:.2f}$ ({stat['p_value']:.3f})"
                row.append(mrt.bold(cell) if stat["p_value"] < 0.05 else cell)
        lines.append(f"{mrt.escape_latex(label)} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("plots/figures/rq5/rq5_matched_correlations.json"))
    args = parser.parse_args()

    matched_backbones = list(FINETUNE_NAME_TO_CANONICAL.values())
    pooled_long = load_study("pooled", models=matched_backbones)
    xg_long = load_study("xcm_gen", models=matched_backbones)
    long_dfs = {
        "region_specific": pooled_long,
        "xcm_head": xg_long,
        # models= excludes perch_v2's raw folder (still on disk, but not a
        # genuine fine-tuning result -- see backbone_meta.py's comment).
        "xcm_finetune_head": load_study("xcm_gen_ft", models=list(FINETUNE_NAME_TO_CANONICAL.keys())),
    }

    stats = json.loads(STATS_JSON.read_text())["subsets"]
    property_values = {
        prop_key: {d: getter(stats[d]) for d in REGIONAL_DATASETS}
        for prop_key, (_, getter, _) in PROPERTIES.items()
    }

    results = compute_correlations(long_dfs, property_values, REGIONAL_DATASETS)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")

    correlations_tex = build_matched_correlations_table(results)

    # Leave-one-out robustness check (NES excluded) -- same computation on
    # the other 6 datasets, appended as a LaTeX comment (every line prefixed
    # "% ") so it's in the source for reference without affecting the
    # compiled document or duplicating a 2nd real table.
    no_nes = [d for d in REGIONAL_DATASETS if d != "NES"]
    loo_results = compute_correlations(long_dfs, property_values, no_nes)
    loo_tex = build_matched_correlations_table(loo_results, n_datasets=len(no_nes), note=", NES excluded")
    loo_comment = "\n".join(f"% {line}" for line in loo_tex.splitlines())
    header_comment = "% Leave-one-out robustness check (NES excluded) -- not part of the compiled document.\n"

    full_tex = correlations_tex + "\n\n\n" + header_comment + loo_comment
    (args.out.parent / "rq5_matched_correlations_table.tex").write_text(full_tex)
    print(f"Wrote {args.out.parent / 'rq5_matched_correlations_table.tex'}")


if __name__ == "__main__":
    main()
