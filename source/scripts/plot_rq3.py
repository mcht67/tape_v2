#!/usr/bin/env python3
"""
RQ3 -- Fine-tuning.

Single paired-slope plot: one line per backbone, connecting its frozen MAE
(archive/Pooled-Embeddings/, pooled MLP head) to its fine-tuned MAE
(archive/Fine-Tuning/, backbone unfrozen and trained end-to-end), both
restricted to the XCM dataset -- the only dataset both studies were run on
for these 5 backbones. Point/line color = the backbone's fixed identity
color (reused from RQ1/RQ2). Since XCM is the only shared dataset, each
side is a single value per backbone (no repeats), so there is no whisker
and no significance test -- lines are drawn solid/uniform throughout.

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
from plot_style import save_fig

SOURCE = "synthetic_mixture_test"
DATASET = "XCM"


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
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    backbone_order = [m for m in BACKBONE_META if m in points.index]
    for model in backbone_order:
        color = BACKBONE_META[model]["identity_color"]
        row = points.loc[model]
        ax.plot([0, 1], [row["frozen"], row["finetuned"]], color=color,
                marker="o", markersize=8, linewidth=2, solid_capstyle="round")
        ax.annotate(BACKBONE_META[model]["display"], (1, row["finetuned"]),
                    textcoords="offset points", xytext=(8, 0), va="center",
                    fontsize=9, color=color)

    ax.set_xlim(-0.25, 1.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Frozen", "Fine-tuned"])
    ax.set_ylabel("MAE on XCM (lower is better)")
    ax.set_title("RQ3 -- Fine-tuning (frozen vs. fine-tuned, XCM dataset)")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)

    fig.tight_layout()
    save_fig(fig, out_dir / "rq3_finetuning")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/rq3"))
    args = parser.parse_args()

    pooled_long = load_study("pooled", models=list(FINETUNE_NAME_TO_CANONICAL.values()))
    finetune_long = load_study("finetune", models=list(FINETUNE_NAME_TO_CANONICAL.keys()))

    points = compute_points(pooled_long, finetune_long)
    build_figure(points, args.out_dir)
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
