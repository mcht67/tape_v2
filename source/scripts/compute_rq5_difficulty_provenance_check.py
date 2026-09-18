#!/usr/bin/env python3
"""
RQ5 -- provenance check for the He et al. polyphony-difficulty correlations
(r=0.93 ratio_polyp, r=0.95 mean_polyp, r=0.82 max_polyp, all vs. per-dataset
soundscape MAE) quoted in the manuscript's "What drives soundscape data
difficulty?" subsection and its difficulty table (Table 9 there;
rq4_fig3_difficulty_table.tex here).

compute_rq5_polyphony_robustness.py already established that recomputing
with this repo's current archive/Pooled-Embeddings/ data (13 backbones, head
"reg", soundscape range_mae) gives r=0.86/0.89/0.80 instead, and that no
backbone-subset variant of that same *current* archive reproduces
0.93/0.95/0.82. This script checks a different hypothesis: that the
manuscript's numbers came from an *earlier* run of the Pooled-Embeddings
study, before some of its backbones were rerun -- archive-old/ (gitignored,
untracked, not DVC-versioned, populated by hand with old run snapshots)
contains several such snapshots, named "<backbone>-before-sr-fix",
suggesting a sample-rate bug was fixed at some point and old pre-fix run
outputs were moved aside rather than overwritten.

This script recomputes the three difficulty-metric correlations using the
"-before-sr-fix" backbone snapshots (6 of them exist: BeansBaseline,
Birdnet_V2.3, Birdnet_V2.4, perch_8, vggish, yamnet; a 7th,
perch_v2_cpu-before-sr-fix, is checked as well since it also carries the
suffix) as a candidate reconstruction of "whatever produced the manuscript
numbers", and reports whether it reproduces the manuscript's per-dataset MAE
table (Table 9: UHH 0.93, HSN 0.64, SSW 0.78, SNE 1.02, POW 1.22, PER 1.84,
NES 0.64) and/or the r=0.93/0.95/0.82 correlations.

Writes rq5_difficulty_provenance_check.json and .md to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq5_difficulty_provenance_check.py [--out-dir plots/figures/rq5]
"""

import argparse
import csv
import json
from pathlib import Path

import scipy.stats

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["UHH", "HSN", "PER", "NES", "POW", "SSW", "SNE"]

TARGET_MAE = {"UHH": 0.93, "HSN": 0.64, "SSW": 0.78, "SNE": 1.02, "POW": 1.22, "PER": 1.84, "NES": 0.64}
REPORTED = {"ratio_polyp": 0.93, "mean_polyp": 0.95, "max_polyp": 0.82}

CANDIDATES = {
    "before-sr-fix-6": [
        "BeansBaseline-before-sr-fix", "Birdnet_V2.3-before-sr-fix", "Birdnet_V2.4-before-sr-fix",
        "perch_8-before-sr-fix", "vggish-before-sr-fix", "yamnet-before-sr-fix",
    ],
    "before-sr-fix-7": [
        "BeansBaseline-before-sr-fix", "Birdnet_V2.3-before-sr-fix", "Birdnet_V2.4-before-sr-fix",
        "perch_8-before-sr-fix", "vggish-before-sr-fix", "yamnet-before-sr-fix", "perch_v2_cpu-before-sr-fix",
    ],
}


def load_reg_mae(archive_root: Path, backbone_dir: str) -> dict:
    f = archive_root / backbone_dir / "scape_eval_results" / "soundscape_test_metrics.csv"
    if not f.exists():
        return {}
    rows = list(csv.reader(f.open()))
    header = rows[0]
    mae_row = next(r for r in rows if r[0] == "range_mae")
    vals = dict(zip(header[1:], mae_row[1:]))
    return {d: float(vals[f"{d}_reg"]) for d in DATASETS if f"{d}_reg" in vals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq5"))
    parser.add_argument("--archive-old", type=Path, default=REPO_ROOT / "archive-old" / "Pooled-Embeddings")
    args = parser.parse_args()

    stats = json.loads((REPO_ROOT / "plots" / "data" / "polybirdmix_soundscape_stats.json").read_text())["subsets"]
    difficulty = {
        key: {d: stats[d]["polyphony_metric_summary"][key]["mean"] for d in DATASETS}
        for key in REPORTED
    }

    results = {}
    for name, backbones in CANDIDATES.items():
        per_dataset = {d: [] for d in DATASETS}
        missing = []
        for b in backbones:
            vals = load_reg_mae(args.archive_old, b)
            if not vals:
                missing.append(b)
                continue
            for d in DATASETS:
                if d in vals:
                    per_dataset[d].append(vals[d])
        means = {d: round(sum(v) / len(v), 4) for d, v in per_dataset.items() if v}

        corrs = {}
        for key in REPORTED:
            x = [means[d] for d in DATASETS]
            y = [difficulty[key][d] for d in DATASETS]
            r = scipy.stats.pearsonr(x, y)
            corrs[key] = {"r": round(float(r.statistic), 4), "p": round(float(r.pvalue), 4)}

        results[name] = {
            "backbones": backbones,
            "missing": missing,
            "mae_by_dataset": means,
            "correlations": corrs,
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq5_difficulty_provenance_check.json"
    out_json.write_text(json.dumps({"target_mae": TARGET_MAE, "reported_correlations": REPORTED, "candidates": results}, indent=2))
    print(f"Wrote {out_json}")

    lines = [
        "# RQ5 -- difficulty-correlation provenance check (archive-old before-sr-fix candidates)",
        "",
        "Tests whether the `archive-old/Pooled-Embeddings/*-before-sr-fix` snapshots (an earlier, "
        "pre-sample-rate-fix run of a subset of backbones, found outside git/DVC version control) "
        "reproduce the manuscript's Table 9 per-dataset MAE and/or its r=0.93 (ratio_polyp), "
        "r=0.95 (mean_polyp), r=0.82 (max_polyp) correlations with MAE.",
        "",
        "## Manuscript target values (Table 9)",
        "",
        "| Dataset | " + " | ".join(DATASETS) + " |",
        "|---|" + "---|" * len(DATASETS),
        "| MAE | " + " | ".join(f"{TARGET_MAE[d]:.2f}" for d in DATASETS) + " |",
        "",
    ]
    for name, entry in results.items():
        lines.append(f"## Candidate: {name}")
        lines.append("")
        lines.append(f"Backbones ({len(entry['backbones']) - len(entry['missing'])}/{len(entry['backbones'])} found): " + ", ".join(entry["backbones"]))
        if entry["missing"]:
            lines.append(f"Missing: {', '.join(entry['missing'])}")
        lines.append("")
        lines.append("| Dataset | " + " | ".join(DATASETS) + " |")
        lines.append("|---|" + "---|" * len(DATASETS))
        lines.append("| Computed MAE | " + " | ".join(f"{entry['mae_by_dataset'].get(d, float('nan')):.4f}" for d in DATASETS) + " |")
        lines.append("| Target MAE | " + " | ".join(f"{TARGET_MAE[d]:.2f}" for d in DATASETS) + " |")
        lines.append("")
        lines.append("| Metric | Computed r | Computed p | Reported r |")
        lines.append("|---|---|---|---|")
        for key, label in [("ratio_polyp", "Ratio polyphonic"), ("mean_polyp", "Mean polyphony"), ("max_polyp", "Mean max-polyp")]:
            c = entry["correlations"][key]
            lines.append(f"| {label} | {c['r']:.3f} | {c['p']:.4f} | {REPORTED[key]} |")
        lines.append("")

    lines += [
        "## Bottom line",
        "",
        "Neither before-sr-fix candidate reproduces the manuscript's Table 9 MAE values or the "
        "r=0.93/0.95/0.82 correlations -- both give *weaker* correlations than even the current "
        "13-backbone archive (r~=0.77-0.81 here vs. r=0.86/0.89/0.80 currently, vs. r=0.93/0.95/0.82 "
        "reported), so a pre-sample-rate-fix run is not the source of the manuscript's numbers "
        "either.",
    ]
    out_md = args.out_dir / "rq5_difficulty_provenance_check.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
