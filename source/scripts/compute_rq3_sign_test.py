#!/usr/bin/env python3
"""
RQ3 (additional) -- Fine-tuned vs. frozen baseline, per backbone: sign-test
applicability check.

Unlike RQ1/RQ2/RQ4/RQ5, RQ3's fine-tuning archive (archive/Fine-Tuning/)
only has results on a single dataset -- XCM -- per backbone (see
plot_rq3.py's compute_points(): both `frozen` and `finetuned` are indexed
purely by `model`, with no per-dataset breakdown at all, XCM being the only
dataset both the Pooled-Embeddings and Fine-Tuning studies were run on for
these backbones). A sign test needs multiple paired observations per
comparison; with n=1 dataset there is nothing to compute a sign test over
-- not even the n<=3 "report raw counts, no p-value" case, since that still
assumes more than one data point.

This script does not compute a sign test. It documents why (per the task:
"if only a single aggregate result exists per backbone, note this and
skip"), and reports the one data point each backbone actually has --
frozen vs. fine-tuned MAE on XCM, and which direction fine-tuning moved it
-- for completeness, since that single-point direction is still a fact
worth having next to the RQ1/RQ2/RQ4/RQ5 sign tests even though it isn't a
test.

Writes rq3_sign_test.json and rq3_sign_test.md to --out-dir.

Usage:
    complete-venv/bin/python source/scripts/compute_rq3_sign_test.py [--out-dir plots/figures/rq3]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backbone_meta import BACKBONE_META, FINETUNE_NAME_TO_CANONICAL
from plot_data import load_study
from plot_rq3 import compute_points


def build_markdown(points) -> str:
    lines = [
        "# RQ3 -- Fine-tuned vs. frozen baseline: sign-test applicability",
        "",
        "**No sign test is computed here.** A sign test requires multiple paired (dataset-level) "
        "observations per comparison; `archive/Fine-Tuning/` only has results on a single dataset -- "
        "XCM -- per backbone (the only dataset shared with `archive/Pooled-Embeddings/`'s frozen "
        "baseline for these 4 backbones -- see `plot_rq3.py`'s `compute_points()`). With n=1, there is "
        "nothing to compute a sign test over, not even the n≤3 \"raw counts, no p-value\" case, "
        "since that still assumes more than one data point.",
        "",
        "The single frozen-vs-fine-tuned data point each backbone does have is reported below for "
        "completeness (this is the same data `rq3_finetuning.png` plots), with the direction fine-tuning "
        "moved MAE -- a fact, not a test.",
        "",
        "| Backbone | Frozen MAE (XCM) | Fine-tuned MAE (XCM) | Direction |",
        "|---|---|---|---|",
    ]
    for model in [m for m in BACKBONE_META if m in points.index]:
        row = points.loc[model]
        direction = "fine-tuning improved (lower MAE)" if row["finetuned"] < row["frozen"] else \
            "fine-tuning worsened (higher MAE)" if row["finetuned"] > row["frozen"] else "no change"
        lines.append(f"| {BACKBONE_META[model]['display']} | {row['frozen']:.3f} | {row['finetuned']:.3f} "
                     f"| {direction} |")
    lines += [
        "",
        "## Reading this",
        "",
        "- perch_v2 is excluded (as in `plot_rq3.py`): its SavedModel backbone cannot receive gradients "
        "at all, so its \"fine-tuned\" archive data only ever trained the head on frozen embeddings, "
        "identical to the frozen baseline -- not a genuine fine-tuning result.",
        "- If per-dataset fine-tuning results become available on further datasets in the future (not "
        "just XCM), this script should be extended to a real sign test at that point.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq3"))
    args = parser.parse_args()

    pooled_long = load_study("pooled", models=list(FINETUNE_NAME_TO_CANONICAL.values()))
    finetune_long = load_study("finetune", models=list(FINETUNE_NAME_TO_CANONICAL.keys()))
    points = compute_points(pooled_long, finetune_long)

    results = {
        "note": "n=1 dataset (XCM) per backbone -- sign test not applicable, see script docstring.",
        "points": points.to_dict(orient="index"),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "rq3_sign_test.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")

    md = build_markdown(points)
    out_md = args.out_dir / "rq3_sign_test.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
