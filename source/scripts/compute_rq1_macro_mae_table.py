#!/usr/bin/env python3
"""
RQ1 (additional) -- Macro-MAE(1-6) table, synthetic vs. soundscape, on the
Pooled-Embeddings experiments (13 backbones, regression head, the 7
region-matched datasets).

Tabulates the exact numbers behind rq1_backbone_comparison_macro_mae[_single_
panel].{png,pdf}: per backbone, mean +/- SD across the 7 regions of the
per-region macro-MAE(1-6), reusing plot_rq1.py's own
_region_synthetic_macro_mae_1_6() / _region_macro_mae_1_6() so the table
cannot drift from the figures. Also lists each per-region value and the
number of levels (out of 6) that region's average was actually taken over --
soundscape regions with no unambiguous clips at a level are averaged over
fewer than 6 (see plot_rq1.compute_soundscape_macro_mae_stats()'s caveat).

Rows are sorted by soundscape mean macro-MAE (best first). Writes
rq1_macro_mae_table.md and rq1_macro_mae_table.json.

Usage:
    complete-venv/bin/python source/scripts/compute_rq1_macro_mae_table.py [--out-dir plots/figures/rq1]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_rq1
from backbone_meta import BACKBONE_META
from plot_data import REGIONAL_DATASETS, load_study


def _cell(value: float, n_levels: int):
    return {"macro_mae_1_6": None if np.isnan(value) else value, "n_levels_used": n_levels}


def _summary(cells: dict) -> dict:
    vals = [c["macro_mae_1_6"] for c in cells.values() if c["macro_mae_1_6"] is not None]
    return {
        "mean": float(np.mean(vals)) if vals else None,
        "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
        "n": len(vals),
    }


def compute(models: list[str]) -> dict:
    archive_dir = plot_rq1.ARCHIVE / "Pooled-Embeddings"
    out = {}
    for model in models:
        model_dir = archive_dir / model
        syn = {r: _cell(*plot_rq1._region_synthetic_macro_mae_1_6(model_dir, r)) for r in REGIONAL_DATASETS}
        scape = {r: _cell(*plot_rq1._region_macro_mae_1_6(model_dir, r)) for r in REGIONAL_DATASETS}
        out[model] = {
            "display": BACKBONE_META[model]["display"],
            "synthetic": {"summary": _summary(syn), "per_region": syn},
            "soundscape": {"summary": _summary(scape), "per_region": scape},
        }
    return out


def _fmt(s: dict) -> str:
    if s["mean"] is None:
        return "n/a"
    sd = "n/a" if s["sd"] is None else f"{s['sd']:.3f}"
    return f"{s['mean']:.3f} +/- {sd}"


def build_markdown(results: dict) -> str:
    order = sorted(results, key=lambda m: (results[m]["soundscape"]["summary"]["mean"] is None,
                                           results[m]["soundscape"]["summary"]["mean"]))
    lines = [
        "# RQ1 -- Macro-MAE(1-6), synthetic vs. soundscape",
        "",
        "Pooled-Embeddings, regression head, 13 backbones. Macro-MAE(1-6) = unweighted mean of per-true-level MAE "
        "(round(pred) vs. true level) over polyphony levels 1-6; computed per region, then mean +/- SD across the "
        f"{len(REGIONAL_DATASETS)} region-matched datasets ({'/'.join(REGIONAL_DATASETS)}). Lower is better. "
        "Rows sorted by soundscape mean (best first). Numbers behind `rq1_backbone_comparison_macro_mae*.{png,pdf}`.",
        "",
        "Soundscape uses the unambiguous-label subset (min_polyphony == max_polyphony) only; synthetic uses every clip.",
        "",
        "## Summary",
        "",
        "| Rank (soundscape) | Backbone | Synthetic macro-MAE | Soundscape macro-MAE |",
        "|---|---|---|---|",
    ]
    for i, m in enumerate(order, 1):
        r = results[m]
        lines.append(f"| {i} | {r['display']} | {_fmt(r['synthetic']['summary'])} | {_fmt(r['soundscape']['summary'])} |")

    for source, title in (("synthetic", "Synthetic"), ("soundscape", "Soundscape")):
        lines += ["", f"## {title}: per-region macro-MAE(1-6)", "",
                  "Values in parentheses = number of levels (of 6) the region's average was taken over.", "",
                  "| Backbone | " + " | ".join(REGIONAL_DATASETS) + " |",
                  "|---|" + "---|" * len(REGIONAL_DATASETS)]
        for m in order:
            cells = results[m][source]["per_region"]
            row = []
            for reg in REGIONAL_DATASETS:
                c = cells[reg]
                row.append("n/a" if c["macro_mae_1_6"] is None else f"{c['macro_mae_1_6']:.3f} ({c['n_levels_used']})")
            lines.append(f"| {results[m]['display']} | " + " | ".join(row) + " |")

    thin = sorted({reg for m in results for reg, c in results[m]["soundscape"]["per_region"].items()
                   if c["n_levels_used"] < 6})
    if thin:
        lines += ["", "**Caveat**: soundscape regions with fewer than 6 levels used (" + ", ".join(thin) +
                  ") are averaged over fewer levels, so per-region values and the SD across regions are not perfectly "
                  "like-for-like."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq1"))
    args = parser.parse_args()

    models = sorted(load_study("pooled")["model"].unique())
    results = compute(models)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rq1_macro_mae_table.json").write_text(json.dumps(results, indent=2))
    (args.out_dir / "rq1_macro_mae_table.md").write_text(build_markdown(results))
    print(f"Wrote {args.out_dir}/rq1_macro_mae_table.{{md,json}}")


if __name__ == "__main__":
    main()
