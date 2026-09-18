#!/usr/bin/env python3
"""
RQ3 -- Fine-tuning.

Single paired-slope plot: one line per backbone, connecting its frozen MAE
(archive/Pooled-Embeddings/, pooled MLP head) to its fine-tuned MAE
(archive/Fine-Tuning/, backbone unfrozen and trained end-to-end), both
restricted to the XCM dataset -- the only dataset both studies were run on
for these 4 backbones. Point/line color = the backbone's fixed identity
color (reused from RQ1/RQ2). Since XCM is the only shared dataset, each
side is a single value per backbone (no repeats), so there is no whisker
and no significance test -- lines are drawn solid/uniform throughout.

perch_v2 is excluded (via backbone_meta.FINETUNE_NAME_TO_CANONICAL, the
single source of truth for which backbones this figure includes): its
SavedModel backbone cannot receive gradients at all (jax2tf export without
with_gradient=True -- see backbone_meta.py's comment), so its "fine-tuned"
archive data only ever trained the head on frozen embeddings, identical to
the frozen baseline -- not a genuine fine-tuning data point.

Usage:
    complete-venv/bin/python source/scripts/plot_rq3.py [--out-dir plots/figures/rq3]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backbone_meta import BACKBONE_META, FINETUNE_NAME_TO_CANONICAL
from plot_data import filter_long, load_study
from plot_style import save_fig, set_rcparams, ONE_COL_WIDTH_IN

SOURCE = "synthetic_mixture_test"
DATASET = "XCM"
FIG_HEIGHT = 2.0


def compute_points(pooled_long: pd.DataFrame, finetune_long: pd.DataFrame) -> pd.DataFrame:
    """One row per canonical backbone: frozen MAE (Pooled-Embeddings) and
    fine-tuned MAE (Fine-Tuning), both on the XCM dataset."""
    frozen = filter_long(pooled_long, source=SOURCE, metric="mae", head="reg",
                          model=list(FINETUNE_NAME_TO_CANONICAL.values()), dataset=DATASET)
    frozen = frozen.set_index("model")["value"]

    finetuned = filter_long(finetune_long, source=SOURCE, metric="mae", head="reg",
                             model=list(FINETUNE_NAME_TO_CANONICAL.keys()), dataset=DATASET)
    finetuned = finetuned.set_index("model")["value"]
    finetuned.index = finetuned.index.map(FINETUNE_NAME_TO_CANONICAL)

    return pd.DataFrame({"frozen": frozen, "finetuned": finetuned}).dropna()


def build_figure(points: pd.DataFrame, out_dir: Path):
    """No in-figure heading -- the caption describes this figure already."""
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_IN, FIG_HEIGHT))

    backbone_order = [m for m in BACKBONE_META if m in points.index]
    for model in backbone_order:
        color = BACKBONE_META[model]["identity_color"]
        row = points.loc[model]
        ax.plot([0, 1], [row["frozen"], row["finetuned"]], color=color,
                marker="o", markersize=4, linewidth=1, solid_capstyle="round")

    # Draw labels after all lines so each label's background box sits on top
    # of every line, including ones that would otherwise cross behind it.
    for model in backbone_order:
        color = BACKBONE_META[model]["identity_color"]
        row = points.loc[model]
        ax.annotate(BACKBONE_META[model]["display"], (1, row["finetuned"]),
                    textcoords="offset points", xytext=(8, 0), va="center",
                    fontsize=9, color=color,
                    bbox=dict(boxstyle="square,pad=0.25", facecolor="white",
                              edgecolor=color, linewidth=0.6),
                    zorder=5)

    ax.set_xlim(-0.25, 1.65)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Frozen", "Fine-tuned"])
    ax.set_ylabel("MAE (lower is better)")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)

    fig.tight_layout()
    save_fig(fig, out_dir / "rq3_finetuning")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq3"))
    args = parser.parse_args()

    set_rcparams()
    pooled_long = load_study("pooled", models=list(FINETUNE_NAME_TO_CANONICAL.values()))
    finetune_long = load_study("finetune", models=list(FINETUNE_NAME_TO_CANONICAL.keys()))

    points = compute_points(pooled_long, finetune_long)
    build_figure(points, args.out_dir)
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
