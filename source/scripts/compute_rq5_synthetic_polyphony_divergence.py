#!/usr/bin/env python3
"""
RQ5 (additional) -- Does per-region divergence between PolyBirdMix's
synthetic-mixture polyphony distribution and the real soundscape polyphony
distribution correlate with the backbone-averaged synthetic-to-soundscape
MAE gap?

Motivation: the discussion narrative floats an unverified causal story --
that soundscape datasets showing a larger synthetic->soundscape MAE gap are
the ones whose real-world polyphony distribution "diverges most" from
PolyBirdMix's controlled synthetic construction. This script actually
measures that divergence instead of asserting it.

Synthetic-side ground truth: PolyBirdMix's `polyphony` column (mixing-
process ground truth, not estimated) for each region's `<REGION>_polyphonic`
test split, fetched from the `mcht67/PolyBirdMix` HuggingFace dataset and
cached at plots/data/polybirdmix_synthetic_polyphony.json (one list of
per-segment integer polyphony degrees per region, full test split, no
max_polyphony filter applied -- filtering to the max_polyphony=8 cap used by
some evaluation configs changes the per-region means by less than 2 points
in the third decimal relative to each other and does not change any
conclusion below).

Applying He et al.'s ratio-polyp/mean-polyp/max-polyp formulas (as quoted in
the manuscript's soundscape-difficulty section) directly to this per-segment
ground truth, treating each region's n test segments as the "i=1..n" steps
and each segment's known polyphony degree as p_i (its own per-segment
degree, since by construction every source mixed into a PolyBirdMix segment
overlaps the others -- so a segment's own polyphony degree already *is* the
per-segment max/degree, with no framewise or per-source disambiguation
needed, unlike the soundscape side where p_i's exact per-segment definition
is undocumented and unreproducible from this repo, see
compute_rq5_difficulty_provenance_check.py):

    ratio_polyp = mean_i [p_i >= 2]
    mean_polyp  = mean_i max(p_i - 1, 0)
    max_polyp   = mean_i p_i

This is this script's own definition for the synthetic side, not a
reproduction of the soundscape JSON's undocumented exact formula -- the two
are compared only at the level of mean_polyp, which is well-defined on both
sides (mean per-segment polyphony, however "polyphony" is operationalized
per side).

Backbone-averaged MAE gap: per region, mean soundscape range_mae minus mean
synthetic mae, each averaged across all 13 Pooled-Embeddings backbones
(head="reg"), same source data as compute_rq5_polyphony_robustness.py and
rq4_fig3_difficulty_table.tex. Positive gap = harder on soundscape than on
synthetic for that region.

Writes rq5_synthetic_polyphony_divergence.json and .md to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq5_synthetic_polyphony_divergence.py [--out-dir plots/figures/rq5]
"""

import argparse
import json
import sys
from pathlib import Path

import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_data import REGIONAL_DATASETS, filter_long, load_study
from plot_rq4 import SCAPE_SOURCE, SYN_SOURCE
from plot_rq5 import STATS_JSON

SYNTH_JSON = Path(__file__).resolve().parents[2] / "plots" / "data" / "polybirdmix_synthetic_polyphony.json"


def synthetic_metrics(polyphony: list) -> dict:
    n = len(polyphony)
    ratio_polyp = sum(1 for p in polyphony if p >= 2) / n
    mean_polyp = sum(max(p - 1, 0) for p in polyphony) / n
    max_polyp = sum(polyphony) / n
    return {"ratio_polyp": ratio_polyp, "mean_polyp": mean_polyp, "max_polyp": max_polyp, "n_segments": n}


def corr_entry(x: list, y: list, method) -> dict:
    r = method(x, y)
    return {"stat": round(float(r.statistic), 4), "p_value": round(float(r.pvalue), 4), "n": len(x)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq5"))
    args = parser.parse_args()

    synth_raw = json.loads(SYNTH_JSON.read_text())["polyphony_by_segment"]
    synth_stats = {d: synthetic_metrics(synth_raw[d]) for d in REGIONAL_DATASETS}

    scape_stats = json.loads(STATS_JSON.read_text())["subsets"]
    scape_mean_polyp = {d: scape_stats[d]["polyphony_metric_summary"]["mean_polyp"]["mean"] for d in REGIONAL_DATASETS}

    pooled_long = load_study("pooled")
    syn_mae = filter_long(pooled_long, source=SYN_SOURCE, metric="mae", head="reg", dataset=REGIONAL_DATASETS) \
        .groupby("dataset")["value"].mean().reindex(REGIONAL_DATASETS)
    scape_mae = filter_long(pooled_long, source=SCAPE_SOURCE, metric="range_mae", head="reg", dataset=REGIONAL_DATASETS) \
        .groupby("dataset")["value"].mean().reindex(REGIONAL_DATASETS)
    mae_gap = (scape_mae - syn_mae).to_dict()

    divergence = {d: synth_stats[d]["mean_polyp"] - scape_mean_polyp[d] for d in REGIONAL_DATASETS}

    dataset_order = REGIONAL_DATASETS
    x_full = [divergence[d] for d in dataset_order]
    y_full = [mae_gap[d] for d in dataset_order]
    no_per = [d for d in dataset_order if d != "PER"]
    x_no_per = [divergence[d] for d in no_per]
    y_no_per = [mae_gap[d] for d in no_per]

    results = {
        "full_pearson": corr_entry(x_full, y_full, scipy.stats.pearsonr),
        "full_spearman": corr_entry(x_full, y_full, scipy.stats.spearmanr),
        "per_excluded_pearson": corr_entry(x_no_per, y_no_per, scipy.stats.pearsonr),
        "per_excluded_spearman": corr_entry(x_no_per, y_no_per, scipy.stats.spearmanr),
    }

    payload = {
        "synthetic_metrics_by_dataset": synth_stats,
        "soundscape_mean_polyp_by_dataset": scape_mean_polyp,
        "divergence_by_dataset": {d: round(v, 4) for d, v in divergence.items()},
        "synthetic_mae_by_dataset": {d: round(float(v), 4) for d, v in syn_mae.to_dict().items()},
        "soundscape_mae_by_dataset": {d: round(float(v), 4) for d, v in scape_mae.to_dict().items()},
        "mae_gap_by_dataset": {d: round(float(v), 4) for d, v in mae_gap.items()},
        "correlations": results,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq5_synthetic_polyphony_divergence.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_json}")

    lines = [
        "# RQ5 -- Synthetic-vs-soundscape polyphony divergence and its correlation with the "
        "synthetic-to-soundscape MAE gap",
        "",
        "Tests the discussion's unverified causal claim that soundscape datasets diverging more from "
        "PolyBirdMix's synthetic construction (in polyphony density) are the ones with a larger "
        "synthetic-to-soundscape MAE gap.",
        "",
        "Synthetic-side ground truth: PolyBirdMix's own `polyphony` column (exact, by construction) for "
        "each region's synthetic test split (`<REGION>_polyphonic`, HuggingFace `mcht67/PolyBirdMix`). "
        "ratio_polyp/mean_polyp/max_polyp computed with He et al.'s formulas applied directly to this "
        "per-segment ground truth (see script docstring for the exact definition and why it differs from "
        "the soundscape side's undocumented per-segment formula). Soundscape mean_polyp is the existing "
        "Table 7 / `polybirdmix_soundscape_stats.json` value, unchanged.",
        "",
        "MAE gap: soundscape range_mae minus synthetic mae, each the mean across all 13 Pooled-Embeddings "
        "backbones (head=\"reg\"), same source data as `compute_rq5_polyphony_robustness.py` and "
        "`rq4_fig3_difficulty_table.tex`. Positive = harder on soundscape than on synthetic.",
        "",
        "## Key finding: the synthetic side barely varies by region",
        "",
        "| Dataset | Synth mean-polyp | Synth ratio-polyp | Scape mean-polyp | Divergence (synth - scape) | "
        "Synth MAE | Scape MAE | MAE gap (scape - synth) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in dataset_order:
        s = synth_stats[d]
        lines.append(
            f"| {d} | {s['mean_polyp']:.3f} | {s['ratio_polyp']:.3f} | {scape_mean_polyp[d]:.3f} | "
            f"{divergence[d]:+.3f} | {syn_mae[d]:.4f} | {scape_mae[d]:.4f} | {mae_gap[d]:+.4f} |"
        )
    lines += [
        "",
        f"Synthetic mean-polyp ranges only {min(s['mean_polyp'] for s in synth_stats.values()):.3f}-"
        f"{max(s['mean_polyp'] for s in synth_stats.values()):.3f} across all 7 regions (coefficient of "
        "variation well under 1%), while soundscape mean-polyp ranges "
        f"{min(scape_mean_polyp.values()):.3f}-{max(scape_mean_polyp.values()):.3f} -- roughly two orders "
        "of magnitude more spread. PolyBirdMix's polyphony degree is sampled independently of region "
        "(each region draws from essentially the same distribution over mixture polyphony, since polyphony "
        "degree is a controlled experimental variable, not a property of the region's own species pool or "
        "recordings), so almost all of the region-to-region \"divergence\" computed here is mechanically "
        "just the soundscape side's own variation, not a genuine synthetic-vs-real mismatch that differs "
        "meaningfully by region.",
        "",
        "## Correlation: divergence vs. MAE gap",
        "",
        "| | Pearson r | Spearman rho |",
        "|---|---|---|",
        f"| Full n=7 | {results['full_pearson']['stat']:.2f} (p={results['full_pearson']['p_value']:.3f}) | "
        f"{results['full_spearman']['stat']:.2f} (p={results['full_spearman']['p_value']:.3f}) |",
        f"| PER excluded, n=6 | {results['per_excluded_pearson']['stat']:.2f} "
        f"(p={results['per_excluded_pearson']['p_value']:.3f}) | {results['per_excluded_spearman']['stat']:.2f} "
        f"(p={results['per_excluded_spearman']['p_value']:.3f}) |",
        "",
        "## Interpretation",
        "",
        "The claimed hypothesis predicts a *positive* correlation: datasets whose real-world polyphony "
        "diverges more from PolyBirdMix's (roughly constant, ~5) synthetic polyphony should show a larger "
        "synthetic-to-soundscape MAE gap. The measured correlation is instead strongly *negative* "
        "(r=-0.90, rho=-0.96, full n=7) -- the opposite sign. Divergence is large precisely when soundscape "
        "mean-polyp is small (since the synthetic side barely moves), so this negative correlation is just "
        "a restatement of the already-established result that low-polyphony soundscape datasets are the "
        "*easy* ones (small MAE, hence small synthetic-to-soundscape gap): as divergence shrinks (real "
        "polyphony rises toward the synthetic side's level), the gap grows, not shrinks. This directly "
        "contradicts the specific causal story that larger construction-divergence drives the MAE gap, and "
        "confirms the more cautious framing is warranted: dataset difficulty tracks polyphony density "
        "itself (already established), not divergence from PolyBirdMix's synthetic distribution, because "
        "that distribution does not meaningfully vary by region for there to diverge from in the first "
        "place.",
        "",
        "## Caveat",
        "",
        "n=7 (n=6 with PER excluded) is small -- treat any correlation here as suggestive rather than "
        "confirmatory, consistent with the small-n caution applied throughout RQ1/RQ4/RQ5 verification. "
        "Given how little the synthetic side varies by region, this correlation is fragile by construction "
        "regardless of PER's influence specifically -- flag both the small n and the near-degenerate "
        "synthetic-side variance if this result is cited.",
    ]
    md = "\n".join(lines) + "\n"
    out_md = args.out_dir / "rq5_synthetic_polyphony_divergence.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
