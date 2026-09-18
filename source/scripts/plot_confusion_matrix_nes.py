#!/usr/bin/env python3
"""
NES-only confusion matrices for the 3 backbone-matched RQ5 studies --
region-specific head (Pooled-Embeddings), XCM head/frozen backbone
(XCM-Generalization), and XCM fine-tune head (XCM-Generalization-fine-tune)
-- restricted to the 4 backbones that were actually fine-tuned
(backbone_meta.FINETUNE_NAME_TO_CANONICAL) and to the NES soundscape test
set. Reuses plot_confusion_matrix.py's loaders/figure-drawing helpers
unchanged (same load_soundscape/load_soundscape_multi, prune_confusion_matrix,
draw_confusion_matrix as the rest of the confusion-matrix figures), so panel
styling (row-normalized % color scale, red diagonal outline, count+% cell
text) matches every other confusion matrix in the repo.

Two kinds of figures, both 3 panels stacked vertically (region-specific /
XCM frozen / XCM fine-tune, top to bottom -- the same order as plot_rq5.py's
build_panel_b_soundscape_matched_grid) with a shared row-normalized-%
color scale:

  - One "accumulated" figure per study, pooling all 4 backbones' NES
    soundscape predictions together (confusion_matrix_nes_accumulated).
  - One figure per backbone, using only that backbone's own NES soundscape
    predictions (confusion_matrix_nes_{backbone_slug}).

Usage:
    complete-venv/bin/python source/scripts/plot_confusion_matrix_nes.py \\
        [--out-dir plots/figures/confusion_matrix/rq5/NES]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backbone_meta import BACKBONE_META, FINETUNE_NAME_TO_CANONICAL
from plot_confusion_matrix import (
    ARCHIVE,
    SPINE_COLOR,
    _set_rcparams,
    draw_confusion_matrix,
    load_soundscape,
    load_soundscape_multi,
    prune_confusion_matrix,
)
from plot_style import TWO_COL_WIDTH_IN, save_fig

REGION = "NES"

# Predictions are clipped to [0, PRED_CLIP_MAX] before building any matrix --
# see plot_rq5.py's build_confusion_matrix_pooled_soundscape docstring: a
# handful of wildly miscalibrated predictions (observed here for NES's
# EfficientNet-B1 backbone in particular, both alone and pooled with the
# other 3) would otherwise balloon a panel to dozens of sparse columns. True
# polyphony never exceeds 11 and can't be negative, so this only saturates
# the rare extreme outliers into the boundary bins and leaves the actual
# (low-degree) distribution that matters for the figure untouched.
PRED_CLIP_MAX = 14

# Raw archive folder name per canonical backbone, for the fine-tune study
# (whose folder names don't match backbone_meta.BACKBONE_META's canonical
# keys -- see FINETUNE_NAME_TO_CANONICAL's own docstring).
CANONICAL_TO_FINETUNE_NAME = {v: k for k, v in FINETUNE_NAME_TO_CANONICAL.items()}

# The 4 backbones that were actually fine-tuned, in backbone_meta.py's
# canonical display order (matches plot_rq5.py's ft_row_order).
BACKBONES = [m for m in BACKBONE_META if m in FINETUNE_NAME_TO_CANONICAL.values()]

BACKBONE_SLUG = {
    "EfficientNet-B1-BirdSet-XCL": "efficientnet",
    "AudioProtoPNet-20-BirdSet-XCL": "audioprotopnet",
    "Bird-MAE-Huge": "birdmae",
    "NatureLMBEATs": "naturelm",
}

# (study folder under archive/, panel title), in the same top-to-bottom
# order as plot_rq5.py's build_panel_b_soundscape_matched_grid.
STUDIES = [
    ("region_specific", "Pooled-Embeddings", "Region-specific head"),
    ("xcm_frozen", "XCM-Generalization", "XCM head (frozen backbone)"),
    ("xcm_finetune", "XCM-Generalization-fine-tune", "XCM fine-tune head"),
]


def model_dir(study_key: str, study_folder: str, backbone: str) -> Path:
    name = CANONICAL_TO_FINETUNE_NAME[backbone] if study_key == "xcm_finetune" else backbone
    return ARCHIVE / study_folder / name


def build_grid_figure(panels, out_path: Path):
    """3 panels stacked vertically, sized like plot_confusion_matrix.py's
    build_figure() (height per panel proportional to its row count, so a
    study with fewer surviving polyphony classes doesn't get needlessly
    stretched), sharing one row-normalized-% color scale via a single
    colorbar spanning all 3 axes."""
    _set_rcparams()
    # Per-panel target height matches build_figure_single()'s density (one
    # panel per column-width slot), computed per panel rather than shared
    # off the tallest/widest panel -- unlike build_figure()'s stacked
    # synthetic+soundscape panels (which are always the same 2 splits of one
    # model), these 3 panels are 3 different studies with independently
    # varying pruned grid sizes. hspace/titles need extra headroom on top of
    # the raw cell area, found empirically via plot_style.check_figure_layout.
    row_counts = [len(row_labels) for _, (_, row_labels, _, _) in panels]
    col_counts = [len(col_labels) for _, (_, _, col_labels, _) in panels]
    target_heights = [1.02 * TWO_COL_WIDTH_IN * r / c for r, c in zip(row_counts, col_counts)]
    fig, axes = plt.subplots(
        3, 1,
        figsize=(TWO_COL_WIDTH_IN, sum(target_heights)),
        gridspec_kw={"height_ratios": target_heights},
        constrained_layout=True,
    )

    ims = []
    for ax, (title, (cm, row_labels, col_labels, n)) in zip(axes, panels):
        im, _ = draw_confusion_matrix(ax, cm, row_labels, col_labels, "True polyphony degree")
        ax.set_title(f"{title}  (n={n:,})", loc="left")
        ims.append(im)

    cbar = fig.colorbar(ims[0], ax=axes, shrink=0.7, aspect=35, pad=0.02, label="Row-normalized %")
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.outline.set_edgecolor(SPINE_COLOR)
    cbar.outline.set_linewidth(0.8)

    save_fig(fig, out_path)
    plt.close(fig)
    print(f"Wrote figures to {out_path}.(png|pdf)")


def panel_data(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, PRED_CLIP_MAX)
    cm, row_labels, col_labels = prune_confusion_matrix(y_true, y_pred)
    return cm, row_labels, col_labels, int(cm.sum())


def build_accumulated_figure(out_dir: Path):
    """One figure, 3 panels (region-specific / XCM frozen / XCM fine-tune),
    each pooling all 4 backbones' NES soundscape predictions together."""
    panels = []
    for study_key, study_folder, title in STUDIES:
        dirs = [model_dir(study_key, study_folder, b) for b in BACKBONES]
        y_true, y_pred = load_soundscape_multi(dirs, regions=[REGION])
        panels.append((f"{title} (pooled across 4 backbones)", panel_data(y_true, y_pred)))
    build_grid_figure(panels, out_dir / "confusion_matrix_nes_accumulated")


def build_per_backbone_figures(out_dir: Path):
    """One figure per backbone, 3 panels (region-specific / XCM frozen /
    XCM fine-tune), each using only that backbone's own NES soundscape
    predictions."""
    for backbone in BACKBONES:
        panels = []
        for study_key, study_folder, title in STUDIES:
            y_true, y_pred = load_soundscape(model_dir(study_key, study_folder, backbone), regions=[REGION])
            panels.append((title, panel_data(y_true, y_pred)))
        slug = BACKBONE_SLUG[backbone]
        build_grid_figure(panels, out_dir / f"confusion_matrix_nes_{slug}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/confusion_matrix/rq5/NES"))
    args = parser.parse_args()

    build_accumulated_figure(args.out_dir)
    build_per_backbone_figures(args.out_dir)


if __name__ == "__main__":
    main()
