#!/usr/bin/env python3
"""
RQ5 (additional) -- Pearson correlation between per-dataset MAE (region-
specific head and XCM head, each mean across the 11 backbones from Panel A;
plus the XCM fine-tune head, mean across the 5 backbones from Panel B; for
both the synthetic test and soundscape test types) and each dataset
property from plots/data/polybirdmix_soundscape_stats.json -- #species,
#segments, #train segments, mean #species/segment, mean #detections/segment,
mean min/max polyphony, ratio polyphonic, mean polyphony, mean max-polyp
(see plot_rq5_dataset_properties.PROPERTIES, the single source of truth for
which properties this script and that figure both use) -- across the 7
regional datasets.

Reuses the same data-pull as plot_rq5.py/plot_rq5_dataset_properties.py
(archive/Pooled-Embeddings/ for region-specific, archive/XCM-Generalization/
for XCM head, archive/XCM-Generalization-fine-tune/ for the XCM fine-tune
head) -- this script only adds the correlation statistics on top, it does
not recompute or alter any figure.

Besides rq5_correlations.json, also writes two LaTeX tables: rq5_properties_
table.tex (each dataset's actual property values) and rq5_correlations_
table.tex (Pearson r (p-value) per property x head x test-type, bolded where
p < 0.05) -- the numeric values behind rq5_dataset_properties.png's 11x2
sub-panel grid, in one place instead of read off the panels' x-axes.

Usage:
    complete-venv/bin/python source/scripts/compute_rq5_correlations.py [--out plots/figures/rq5/rq5_correlations.json]
"""

import argparse
import json
import sys
from pathlib import Path

import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import FINETUNE_NAME_TO_CANONICAL
from plot_data import REGIONAL_DATASETS, filter_long, load_study
from plot_rq5 import SCAPE_SOURCE, STATS_JSON, SYN_SOURCE
from plot_rq5_dataset_properties import PROPERTIES

TEST_TYPES = {
    "synthetic": (SYN_SOURCE, "mae"),
    "soundscape": (SCAPE_SOURCE, "range_mae"),
}
HEAD_LABEL = {
    "region_specific": "Region-specific",
    "xcm_head": "XCM head",
    "xcm_finetune_head": "XCM fine-tune head",
}
TEST_LABEL = {"synthetic": "Synth.", "soundscape": "Scape"}


def per_dataset_mae(long_df, source, metric):
    """Mean MAE per dataset, averaged across whatever backbones are present."""
    sub = filter_long(long_df, source=source, metric=metric, head="reg", dataset=REGIONAL_DATASETS)
    return sub.groupby("dataset")["value"].mean()


def species_ordered_datasets(stats):
    return sorted(REGIONAL_DATASETS, key=lambda d: stats[d]["n_unique_species"])


def build_properties_table(stats, dataset_order) -> str:
    """Each regional dataset's actual value for every PROPERTIES entry --
    the numbers rq5_dataset_properties.png's x-axis tick labels are drawn
    from, collected in one table instead of read off 9 different panels."""
    prop_items = list(PROPERTIES.items())
    header = "Dataset & " + " & ".join(mrt.escape_latex(label) for _, (label, _, _) in prop_items) + r" \\"
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Regional soundscape test subset properties, ordered by increasing species count.}",
        r"\label{tab:rq5_properties}",
        rf"\begin{{tabular}}{{l{'c' * len(prop_items)}}}", r"\toprule", header, r"\midrule",
    ]
    for d in dataset_order:
        row = [fmt.format(getter(stats[d])) for _, (_, getter, fmt) in prop_items]
        lines.append(f"{d} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_correlations_table(results) -> str:
    """Pearson r (p-value) between each PROPERTIES entry and per-dataset MAE,
    for every (head, test type) combination -- bolded where p < 0.05."""
    heads = list(HEAD_LABEL)
    tests = list(TEST_LABEL)
    header = "Property & " + " & ".join(
        f"{HEAD_LABEL[h]} ({TEST_LABEL[t]})" for h in heads for t in tests
    ) + r" \\"
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Pearson correlation ($r$, with $p$-value) between each dataset property and "
        r"per-dataset mean MAE, across the 7 regional datasets ($n=7$). Bold: $p < 0.05$.}",
        r"\label{tab:rq5_correlations}",
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
    parser.add_argument("--out", type=Path, default=Path("plots/figures/rq5/rq5_correlations.json"))
    args = parser.parse_args()

    xg_models = mrt.discover_models(Path("archive/XCM-Generalization"))
    long_dfs = {
        "region_specific": load_study("pooled", models=xg_models),
        "xcm_head": load_study("xcm_gen", models=xg_models),
        # 4 backbones, archive/XCM-Generalization-fine-tune/ -- models=
        # excludes perch_v2's raw folder (still on disk, but its backbone
        # cannot receive gradients at all -- see backbone_meta.py's comment
        # on FINETUNE_NAME_TO_CANONICAL -- so it is never a genuine
        # fine-tuning result).
        "xcm_finetune_head": load_study("xcm_gen_ft", models=list(FINETUNE_NAME_TO_CANONICAL.keys())),
    }

    stats = json.loads(STATS_JSON.read_text())["subsets"]
    property_values = {
        prop_key: {d: getter(stats[d]) for d in REGIONAL_DATASETS}
        for prop_key, (_, getter, _) in PROPERTIES.items()
    }

    results = {}
    for head_key, long_df in long_dfs.items():
        results[head_key] = {}
        for test_key, (source, metric) in TEST_TYPES.items():
            mae = per_dataset_mae(long_df, source, metric).reindex(REGIONAL_DATASETS)
            results[head_key][test_key] = {}
            for prop_key, values_by_dataset in property_values.items():
                prop_values = [values_by_dataset[d] for d in REGIONAL_DATASETS]
                r = scipy.stats.pearsonr(prop_values, mae.values)
                results[head_key][test_key][prop_key] = {
                    "pearson_r": round(float(r.statistic), 4),
                    "p_value": round(float(r.pvalue), 4),
                    "n": len(REGIONAL_DATASETS),
                }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")

    dataset_order = species_ordered_datasets(stats)
    properties_tex = build_properties_table(stats, dataset_order)
    (args.out.parent / "rq5_properties_table.tex").write_text(properties_tex)
    print(f"Wrote {args.out.parent / 'rq5_properties_table.tex'}")

    correlations_tex = build_correlations_table(results)
    (args.out.parent / "rq5_correlations_table.tex").write_text(correlations_tex)
    print(f"Wrote {args.out.parent / 'rq5_correlations_table.tex'}")


if __name__ == "__main__":
    main()
