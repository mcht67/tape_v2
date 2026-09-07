#!/usr/bin/env python3
"""
RQ5 (additional) -- Pearson correlation between per-dataset MAE (region-
specific head and XCM head, each mean across the 11 backbones, for both
the synthetic test and soundscape test types) and each dataset property
from plots/data/polybirdmix_soundscape_stats.json (#species, #segments,
ratio_polyp, mean_polyp), across the 7 regional datasets.

Reuses the same data-pull as plot_rq5.py/plot_rq5_dataset_properties.py
(archive/Pooled-Embeddings/ for region-specific, archive/XCM-Generalization/
for XCM head) -- this script only adds the correlation statistics on top,
it does not recompute or alter any figure.

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
from plot_data import REGIONAL_DATASETS, filter_long, load_study
from plot_rq5 import SCAPE_SOURCE, STATS_JSON, SYN_SOURCE
from plot_rq5_dataset_properties import PROPERTIES

HEAD_SOURCES = {
    "region_specific": "pooled",   # archive/Pooled-Embeddings/, dataset X's own single-dataset-trained model
    "xcm_head": "xcm_gen",         # archive/XCM-Generalization/, XCM-trained head tested on dataset X
}
TEST_TYPES = {
    "synthetic": (SYN_SOURCE, "mae"),
    "soundscape": (SCAPE_SOURCE, "range_mae"),
}


def per_dataset_mae(long_df, source, metric):
    """Mean MAE per dataset, averaged across whatever backbones are present."""
    sub = filter_long(long_df, source=source, metric=metric, head="reg", dataset=REGIONAL_DATASETS)
    return sub.groupby("dataset")["value"].mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("plots/figures/rq5/rq5_correlations.json"))
    args = parser.parse_args()

    xg_models = mrt.discover_models(Path("archive/XCM-Generalization"))
    long_dfs = {
        "region_specific": load_study("pooled", models=xg_models),
        "xcm_head": load_study("xcm_gen", models=xg_models),
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
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
