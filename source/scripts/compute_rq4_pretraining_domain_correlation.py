#!/usr/bin/env python3
"""
RQ4 (additional) -- Direct test of "general-audio pretraining reduces
synthetic-to-soundscape degradation", instead of citing VGGish/YAMNet as a
two-example anecdote.

Per backbone (archive/Pooled-Embeddings/, all 13, region-matched scope --
same synthetic/soundscape MAE values as RQ4 Figure 1's dumbbell plot):
  delta = soundscape MAE (range_mae, mean across the 7 regional datasets)
          - synthetic MAE (mae, mean across the same 7 datasets)
  (positive = performance gets worse going from synthetic to real soundscape
  audio; this is exactly the gap build_fig1's dumbbell plot draws.)

Each backbone is coded 1/0 from its `domain` field in
backbone_meta.BACKBONE_META -- the source of truth for that classification
is stated in backbone_meta.py's own docstring as "the user's own paper
table" (i.e. Table 3). Three codings are reported, since "general audio" is
not perfectly binary in that table, and the underlying hypothesis could
reasonably be either "general-audio pretraining specifically" or the
broader "not bird-native pretraining":
  - primary: domain contains "General audio" (covers both "General audio"
    and "General audio + cross-taxa") -> VGGish, YAMNet, NatureLM-audio = 1;
    the other 10 = 0.
  - sensitivity: only the pure "General audio" domain counts (VGGish,
    YAMNet = 1, n=2) -- NatureLM-audio's "General audio + cross-taxa" hybrid
    excluded from the "1" group entirely (treated as 0) for this check,
    since it isn't purely general-audio-pretrained.
  - broad: primary's 3 plus "Other -> Bird" (AST, Wav2Vec2, EfficientNet-B1)
    = 1, n=6 -- the broader "originally pretrained on non-bird-native
    audio, not just general-audio specifically" grouping. Added because AST
    and Wav2Vec2 (both "Other -> Bird", coded 0 in the other two schemes)
    showed larger negative deltas than either pure general-audio backbone
    in the primary/sensitivity results, suggesting the relevant factor
    might be broader than "general audio" alone.
  - table3_groups: an explicit membership list (not derived from `domain`)
    -- "has general-audio pretraining" = VGGish, YAMNet, NatureLM-audio,
    BirdNET v2.3, BirdNET v2.4, Perch v2 (n=6) vs. "bird-only" = Bird-MAE,
    AudioProtoPNet, EfficientNet-B1, Perch v1, AST, Wav2Vec2 (n=6); BEANS
    baseline excluded from both groups (domain "None" -- no pretraining at
    all, so it fits neither). This groups BirdNET/Perch v2 with the
    general-audio side (their pretraining data includes AudioSet/cross-taxa
    audio per Table 3, even though backbone_meta.py's own `domain` field --
    a different, coarser axis -- labels them "Cross-taxa" and the other 3
    codings above therefore treat them as 0) and EfficientNet-B1 with
    bird-only (unlike the broad coding, which puts every "Other -> Bird"
    domain backbone, including EfficientNet-B1, on the general-audio side).
  - table3_no_vy: robustness check on table3_groups -- same "bird-only"
    group 0 (n=6), but group 1 drops VGGish and YAMNet entirely (not moved
    to group 0, excluded from the test altogether, alongside BEANS
    baseline), leaving just NatureLM-audio, BirdNET v2.3, BirdNET v2.4,
    Perch v2 (n=4). Checks whether table3_groups' result depends on VGGish/
    YAMNet specifically (the original two-example anecdote) or holds up
    with them removed from the general-audio side entirely.

For each coding: point-biserial correlation (pearsonr against the 0/1
variable -- point-biserial r *is* Pearson's r for a binary/continuous pair),
Welch's t-test (unequal variances, since the compared backbone groups have
no reason to share a variance), and a Mann-Whitney U test (distribution-
free, since these small, unequal-sized groups are a poor fit for a
normality assumption) between the two groups' delta values. n=13 total (12
for table3_groups, 10 for table3_no_vy), with as few as 2-6 backbones in
the "1" group -- all five codings' tests are underpowered; report
accordingly.

Writes rq4_pretraining_domain_correlation.json and
rq4_pretraining_domain_correlation.md to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq4_pretraining_domain_correlation.py [--out-dir plots/figures/rq4]
"""

import argparse
import json
import sys
from pathlib import Path

import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backbone_meta import BACKBONE_META
from plot_data import REGIONAL_DATASETS, filter_long, load_study, mean_sd_n

SYN_SOURCE = "synthetic_mixture_test"
SCAPE_SOURCE = "soundscape_test"


def compute_deltas(pooled_long) -> dict:
    """Same region-matched scope as plot_rq4.build_fig1(): per backbone,
    mean synthetic MAE and mean soundscape range_mae across the 7 regional
    datasets, and their difference."""
    syn = filter_long(pooled_long, source=SYN_SOURCE, metric="mae", head="reg", dataset=REGIONAL_DATASETS)
    scape = filter_long(pooled_long, source=SCAPE_SOURCE, metric="range_mae", head="reg", dataset=REGIONAL_DATASETS)
    syn_stats = mean_sd_n(syn, ["model"])
    scape_stats = mean_sd_n(scape, ["model"])
    models = sorted(set(syn_stats.index) & set(scape_stats.index))
    return {
        m: {
            "synthetic_mae": float(syn_stats.loc[m, "mean"]),
            "soundscape_mae": float(scape_stats.loc[m, "mean"]),
            "delta": float(scape_stats.loc[m, "mean"] - syn_stats.loc[m, "mean"]),
        }
        for m in models
    }


def run_test(deltas: dict, coding: dict, label: str) -> dict:
    """coding may cover only a subset of deltas' backbones (e.g.
    table3_groups excludes BEANS baseline, which fits neither group) --
    models is taken from coding, not deltas, for that reason."""
    models = list(coding)
    x = [coding[m] for m in models]
    y = [deltas[m]["delta"] for m in models]
    group1 = [deltas[m]["delta"] for m in models if coding[m] == 1]
    group0 = [deltas[m]["delta"] for m in models if coding[m] == 0]

    pb = scipy.stats.pointbiserialr(x, y)
    mw = scipy.stats.mannwhitneyu(group1, group0, alternative="two-sided") if group1 and group0 else None
    # Welch's t-test (unequal variances) -- needs at least 2 per group.
    tt = scipy.stats.ttest_ind(group1, group0, equal_var=False) if len(group1) >= 2 and len(group0) >= 2 else None

    return {
        "label": label,
        "n_total": len(models),
        "n_general_audio": len(group1),
        "n_other": len(group0),
        "mean_delta_general_audio": float(sum(group1) / len(group1)) if group1 else None,
        "mean_delta_other": float(sum(group0) / len(group0)) if group0 else None,
        "point_biserial_r": round(float(pb.statistic), 4),
        "point_biserial_p": round(float(pb.pvalue), 4),
        "ttest_t": round(float(tt.statistic), 4) if tt is not None else None,
        "ttest_p": round(float(tt.pvalue), 4) if tt is not None else None,
        "mannwhitney_u": float(mw.statistic) if mw is not None else None,
        "mannwhitney_p": round(float(mw.pvalue), 4) if mw is not None else None,
    }


def build_markdown(deltas: dict, codings: list) -> str:
    """codings: list of (short_column_label, title, coding_dict, result_dict) --
    one entry per robustness-check coding, in report order."""
    order = sorted(deltas, key=lambda m: deltas[m]["delta"])
    lines = [
        "# RQ4 -- General-audio pretraining vs. synthetic-to-soundscape degradation",
        "",
        "Direct test of whether general-audio-pretrained backbones show smaller synthetic-to-"
        "soundscape degradation, instead of citing VGGish/YAMNet as a two-example anecdote. "
        "`delta = soundscape MAE (range_mae) - synthetic MAE`, both mean across the 7 region-matched "
        "datasets (same values as RQ4 Figure 1's dumbbell plot) -- positive means performance gets "
        "worse going from synthetic to real soundscape audio.",
        "",
        f"**Caveat**: n=13 backbones total (fewer for codings that exclude some backbones from both "
        f"groups -- see each coding's own n below), with as few as 2 of them in the smaller group "
        f"depending on the coding below. All {len(codings)} codings' tests are underpowered at this "
        f"group size -- read the numbers as suggestive, not as a confirmed effect.",
        "",
        "## Per-backbone data",
        "",
        "| Backbone | Domain (Table 3 `domain` field) | " + " | ".join(c[0] for c in codings) +
        " | Synthetic MAE | Soundscape MAE | Delta |",
        "|---|---|" + "---|" * len(codings) + "---|---|---|",
    ]
    for m in order:
        d = deltas[m]
        display = BACKBONE_META[m]["display"]
        domain = BACKBONE_META[m]["domain"]
        coding_cells = " | ".join(str(coding[m]) if m in coding else "--" for _, _, coding, _ in codings)
        lines.append(f"| {display} | {domain} | {coding_cells} | {d['synthetic_mae']:.3f} | "
                     f"{d['soundscape_mae']:.3f} | {d['delta']:+.3f} |")

    for _, title, coding, result in codings:
        # List which backbones are in the "1"/"0" groups from the coding
        # dict itself, not a hand-written name list -- a hardcoded list
        # silently goes stale if BACKBONE_META's domain assignments change
        # (this bit a first draft of this script: "Other -> Bird" is 3
        # backbones, EfficientNet-B1 included, not just AST/Wav2Vec2).
        members1 = ", ".join(BACKBONE_META[m]["display"] for m in coding if coding[m] == 1)
        members0 = ", ".join(BACKBONE_META[m]["display"] for m in coding if coding[m] == 0)
        lines += [
            "",
            f"## {title}",
            "",
            f"- Group 1 ({members1}): n={result['n_general_audio']}, mean delta = "
            f"{result['mean_delta_general_audio']:+.3f}" if result["mean_delta_general_audio"] is not None else
            f"- Group 1: n={result['n_general_audio']} (empty)",
            f"- Group 0 ({members0}): n={result['n_other']}, mean delta = {result['mean_delta_other']:+.3f}",
            f"- Point-biserial correlation: r = {result['point_biserial_r']:+.3f}, p = {result['point_biserial_p']:.4f}",
            f"- Welch's t-test: t = {result['ttest_t']:+.3f}, p = {result['ttest_p']:.4f}"
            if result["ttest_t"] is not None else "- Welch's t-test: not computed (group too small, n<2)",
            f"- Mann-Whitney U: U = {result['mannwhitney_u']:.1f}, p = {result['mannwhitney_p']:.4f}"
            if result["mannwhitney_u"] is not None else "- Mann-Whitney U: not computed (empty group)",
        ]

    lines += [
        "",
        "## Reading this",
        "",
        "A negative point-biserial r means general-audio-pretrained backbones tend to have a *smaller* "
        "(or more negative) delta -- i.e. they degrade less (or even improve) going from synthetic to "
        "soundscape audio, which is the direction the VGGish/YAMNet anecdote claims. Check the sign, "
        "the p-value, and the group means together -- with groups this small, statistical significance "
        "is unlikely regardless of the true effect, so the group means and per-backbone table above "
        "matter at least as much as the p-value for judging whether this is worth reporting.",
    ]
    return "\n".join(lines) + "\n"


# Explicit group membership from the user's Table 3 pretraining-data
# column -- a different, finer-grained axis than BACKBONE_META's own
# `domain` field (which is why BirdNET/Perch v2 land on the general-audio
# side here despite being "Cross-taxa" in `domain`). BEANS baseline (no
# pretraining at all) is deliberately absent from both lists.
TABLE3_GENERAL_AUDIO = ["vggish", "yamnet", "NatureLMBEATs", "Birdnet_V2.3", "Birdnet_V2.4", "perch_v2_cpu"]
TABLE3_BIRD_ONLY = ["Bird-MAE-Huge", "AudioProtoPNet-20-BirdSet-XCL", "EfficientNet-B1-BirdSet-XCL",
                     "perch_8", "AST-Birdset-XCL", "Wav2Vec2-Base-BirdSet-XCL"]
# table3_no_vy: table3_groups' group 1 with VGGish/YAMNet dropped (not moved
# to group 0 -- excluded from the test entirely, like BEANS baseline).
TABLE3_GENERAL_AUDIO_NO_VY = [m for m in TABLE3_GENERAL_AUDIO if m not in ("vggish", "yamnet")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq4"))
    args = parser.parse_args()

    pooled_long = load_study("pooled")
    deltas = compute_deltas(pooled_long)

    coding_primary = {m: (1 if "General audio" in BACKBONE_META[m]["domain"] else 0) for m in deltas}
    coding_sensitivity = {m: (1 if BACKBONE_META[m]["domain"] == "General audio" else 0) for m in deltas}
    coding_broad = {m: (1 if BACKBONE_META[m]["domain"] in ("General audio", "General audio + cross-taxa",
                                                              "Other -> Bird") else 0) for m in deltas}
    coding_table3 = {**{m: 1 for m in TABLE3_GENERAL_AUDIO}, **{m: 0 for m in TABLE3_BIRD_ONLY}}
    coding_table3_no_vy = {**{m: 1 for m in TABLE3_GENERAL_AUDIO_NO_VY}, **{m: 0 for m in TABLE3_BIRD_ONLY}}
    for coding in (coding_table3, coding_table3_no_vy):
        assert set(coding) <= set(deltas), "a TABLE3_* list has a name not in deltas"

    codings = [
        ("Primary", 'Primary coding: "General audio" or "General audio + cross-taxa" = 1',
         coding_primary, run_test(deltas, coding_primary, "primary")),
        ("Sensitivity", 'Sensitivity coding: only pure "General audio" = 1',
         coding_sensitivity, run_test(deltas, coding_sensitivity, "sensitivity")),
        ("Broad", 'Broad coding: primary + "Other -> Bird" = 1',
         coding_broad, run_test(deltas, coding_broad, "broad")),
        ("Table3 groups", 'Table3-groups coding: explicit "has general-audio pretraining" vs. '
         '"bird-only" membership (not derived from `domain`; BEANS baseline excluded)',
         coding_table3, run_test(deltas, coding_table3, "table3_groups")),
        ("Table3, no VGGish/YAMNet", 'Table3-no-VY coding: robustness check on table3_groups with '
         'VGGish and YAMNet dropped from group 1 entirely (excluded from the test, not moved to '
         'group 0) -- checks whether table3_groups\' result depends on the two backbones from the '
         'original anecdote specifically',
         coding_table3_no_vy, run_test(deltas, coding_table3_no_vy, "table3_no_vy")),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq4_pretraining_domain_correlation.json"
    out_json.write_text(json.dumps({
        "deltas": deltas,
        **{f"coding_{result['label']}": coding for _, _, coding, result in codings},
        **{f"result_{result['label']}": result for _, _, coding, result in codings},
    }, indent=2))
    print(f"Wrote {out_json}")

    md = build_markdown(deltas, codings)
    out_md = args.out_dir / "rq4_pretraining_domain_correlation.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
