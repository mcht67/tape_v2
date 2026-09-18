#!/usr/bin/env python3
"""
RQ5 (additional) -- Pearson correlation between the *cross-region
generalization gap* and each dataset property from PROPERTIES (see
plot_rq5_dataset_properties.py) -- the same properties and method as
compute_rq5_correlations.py, but correlating against the difference between
two heads' per-dataset MAE instead of a single head's raw MAE.

Gap = (cross-region head's per-dataset mean MAE) - (region-specific head's
per-dataset mean MAE), paired by dataset, same test type. Positive means the
cross-region head is worse than the region-specific head on that dataset --
i.e. it measures how much a head loses by not being trained on that
specific region. Two such gaps: "XCM head" (11 backbones each) and "XCM
fine-tune head" (4 vs. 11 backbones -- per_dataset_mae() pools whatever
backbones are present in each study, same as the non-diff script), each for
both the synthetic and soundscape test types.

"XCM fine-tune head" excludes perch_v2 (via an explicit models= restriction
on its load_study("xcm_gen_ft") call, to FINETUNE_NAME_TO_CANONICAL's keys):
perch_v2's SavedModel backbone cannot receive gradients at all, so its
archive data only ever trained the head on frozen embeddings -- not a
genuine fine-tuning result. See backbone_meta.py's comment for the full
explanation.

Writes three new files under --out-dir, alongside (not replacing) the
non-diff script's outputs:
  - rq5_diff_correlations.json  -- same shape as rq5_correlations.json, one
    top-level key per gap instead of per raw head.
  - rq5_diff_correlations_table.tex -- LaTeX table, Pearson r (p-value) per
    property x gap x test type, bolded where p < 0.05.
  - rq5_diff_grid.png -- 9 properties x 2 test types, one row per property
    (same property-based dataset ordering as rq5_dataset_properties.png),
    both gap lines plus a zero reference line (zero = no generalization
    penalty).

Usage:
    complete-venv/bin/python source/scripts/compute_rq5_diff_correlations.py [--out-dir plots/figures/rq5]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import FINETUNE_NAME_TO_CANONICAL
from compute_rq5_correlations import TEST_TYPES, per_dataset_mae
from plot_data import REGIONAL_DATASETS, load_study
from plot_rq5 import ACCENT_COLOR, FT_ACCENT_COLOR, STATS_JSON
from plot_rq5_dataset_properties import PROPERTIES, property_ordered_datasets
from plot_style import save_fig, set_rcparams, zero_ref_line, TWO_COL_WIDTH_IN

# (display label, line color) per gap -- the gap key names double as the
# rq5_diff_correlations.json top-level keys.
DIFF_HEADS = {
    "xcm_head_minus_region_specific": ("XCM head", ACCENT_COLOR),
    "xcm_finetune_head_minus_region_specific": ("XCM fine-tune head", FT_ACCENT_COLOR),
}
TEST_LABEL = {"synthetic": "Synth.", "soundscape": "Scape"}


def compute_diff_series(long_dfs: dict, source: str, metric: str) -> dict:
    """One {gap_key: pandas Series indexed by dataset} per DIFF_HEADS entry,
    for this (source, metric) test type."""
    region_specific = per_dataset_mae(long_dfs["region_specific"], source, metric).reindex(REGIONAL_DATASETS)
    diffs = {}
    for diff_key in DIFF_HEADS:
        other_key = diff_key.removesuffix("_minus_region_specific")
        other = per_dataset_mae(long_dfs[other_key], source, metric).reindex(REGIONAL_DATASETS)
        diffs[diff_key] = other - region_specific
    return diffs


def build_diff_correlations_table(results: dict) -> str:
    diff_keys = list(DIFF_HEADS)
    tests = list(TEST_LABEL)
    header = "Property & " + " & ".join(
        f"{DIFF_HEADS[k][0]} ({TEST_LABEL[t]})" for k in diff_keys for t in tests
    ) + r" \\"
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Pearson correlation ($r$, with $p$-value) between each dataset property and the "
        r"cross-region generalization gap (cross-region head's per-dataset mean MAE minus the "
        r"region-specific head's, same dataset; positive = cross-region head is worse), across the "
        r"7 regional datasets ($n=7$). Bold: $p < 0.05$.}",
        r"\label{tab:rq5_diff_correlations}",
        rf"\begin{{tabular}}{{l{'c' * (len(diff_keys) * len(tests))}}}", r"\toprule", header, r"\midrule",
    ]
    for prop_key, (label, _, _) in PROPERTIES.items():
        row = []
        for k in diff_keys:
            for t in tests:
                stat = results[k][t][prop_key]
                cell = f"${stat['pearson_r']:.2f}$ ({stat['p_value']:.3f})"
                row.append(mrt.bold(cell) if stat["p_value"] < 0.05 else cell)
        lines.append(f"{mrt.escape_latex(label)} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def draw_diff_panel(ax, long_dfs: dict, source: str, metric: str, dataset_order: list, title: str):
    """One panel: both gap lines (XCM head / XCM fine-tune head, each minus
    region-specific) across dataset_order, plus a zero reference line."""
    region_specific = per_dataset_mae(long_dfs["region_specific"], source, metric).reindex(dataset_order)
    xs = range(len(dataset_order))
    zero_ref_line(ax)
    for diff_key, (label, color) in DIFF_HEADS.items():
        other_key = diff_key.removesuffix("_minus_region_specific")
        other = per_dataset_mae(long_dfs[other_key], source, metric).reindex(dataset_order)
        diff = other - region_specific
        ax.plot(xs, diff.values, color=color, marker="o", markersize=6, linewidth=2,
                 label=f"{label} - Region-specific")
    ax.set_title(title)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    return xs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq5"))
    args = parser.parse_args()

    set_rcparams()
    xg_models = mrt.discover_models(Path("archive/XCM-Generalization"))
    long_dfs = {
        "region_specific": load_study("pooled", models=xg_models),
        "xcm_head": load_study("xcm_gen", models=xg_models),
        "xcm_finetune_head": load_study("xcm_gen_ft", models=list(FINETUNE_NAME_TO_CANONICAL.keys())),
    }

    stats = json.loads(STATS_JSON.read_text())["subsets"]
    property_values = {
        prop_key: {d: getter(stats[d]) for d in REGIONAL_DATASETS}
        for prop_key, (_, getter, _) in PROPERTIES.items()
    }

    results = {k: {} for k in DIFF_HEADS}
    for test_key, (source, metric) in TEST_TYPES.items():
        diffs = compute_diff_series(long_dfs, source, metric)
        for diff_key, diff_series in diffs.items():
            results[diff_key][test_key] = {}
            for prop_key, values_by_dataset in property_values.items():
                prop_values = [values_by_dataset[d] for d in REGIONAL_DATASETS]
                r = scipy.stats.pearsonr(prop_values, diff_series.values)
                results[diff_key][test_key][prop_key] = {
                    "pearson_r": round(float(r.statistic), 4),
                    "p_value": round(float(r.pvalue), 4),
                    "n": len(REGIONAL_DATASETS),
                }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq5_diff_correlations.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")

    table_tex = build_diff_correlations_table(results)
    (args.out_dir / "rq5_diff_correlations_table.tex").write_text(table_tex)
    print(f"Wrote {args.out_dir / 'rq5_diff_correlations_table.tex'}")

    prop_keys = list(PROPERTIES.keys())
    fig, axes = plt.subplots(len(prop_keys), 2, figsize=(TWO_COL_WIDTH_IN, 2.15 * len(prop_keys)),
                              constrained_layout=True)
    for row, prop_key in enumerate(prop_keys):
        prop_label = PROPERTIES[prop_key][0]
        dataset_order, tick_labels = property_ordered_datasets(stats, prop_key)

        xs = draw_diff_panel(axes[row, 0], long_dfs, *TEST_TYPES["synthetic"], dataset_order,
                              "Synthetic test" if row == 0 else "")
        draw_diff_panel(axes[row, 1], long_dfs, *TEST_TYPES["soundscape"], dataset_order,
                         "Soundscape" if row == 0 else "")
        axes[row, 0].set_ylabel(f"$\\Delta$MAE\n(ordered by {prop_label})")
        for ax in axes[row]:
            ax.set_xticks(list(xs))
            ax.set_xticklabels(tick_labels, rotation=75, ha="right")
        if row == 0:
            axes[row, 0].legend(loc="upper left", frameon=False, fontsize=7)

    # Shared y-scale across all 18 sub-panels.
    lo = min(ax.get_ylim()[0] for ax in axes.flat)
    hi = max(ax.get_ylim()[1] for ax in axes.flat)
    for ax in axes.flat:
        ax.set_ylim(lo, hi)

    # No in-figure heading -- the caption describes this figure already.
    save_fig(fig, args.out_dir / "rq5_diff_grid")
    plt.close(fig)
    print(f"Wrote {args.out_dir / 'rq5_diff_grid'}.(png|pdf)")


if __name__ == "__main__":
    main()
