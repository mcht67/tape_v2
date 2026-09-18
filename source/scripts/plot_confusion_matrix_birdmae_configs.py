#!/usr/bin/env python3
"""
Bird-MAE confusion matrices, one panel per configuration, pooled across all
evaluation datasets of one split (reusing plot_confusion_matrix.py's loaders
and drawing helpers, so conventions match the rest of the paper's confusion
matrices: soundscape ground truth is the [min, max] interval, so the "true"
axis is the rounded prediction clipped into it; row-normalized % colors).

Soundscape figure (7 regional datasets pooled), 4 panels:
  - Region-specific pooled-MLP   archive/Pooled-Embeddings/Bird-MAE-Huge
  - TC + frame-level call act.   archive/Spatial-Embeddings/Bird-MAE-Huge (per-run recovery,
                                  see compute_rq2_per_level_accuracy.py)
  - XCM-frozen                   archive/XCM-Generalization/Bird-MAE-Huge
  - XCM-fine-tuned               archive/XCM-Generalization-fine-tune/BirdSetBirdMAE

Synthetic figure (7 regional test sets pooled; TC+frame-pol omitted, its per-config
synthetic test predictions were overwritten on disk), 2 panels: region-specific
pooled-MLP and XCM-fine-tuned.

All panels share one row/column label set (union of all panels' populated
classes, predictions clipped to [0, PRED_CLIP_MAX]) so cells line up across
panels.

Usage:
    complete-venv/bin/python source/scripts/plot_confusion_matrix_birdmae_configs.py \\
        [--out-dir plots/figures/confusion_matrix]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_rq2_per_level_accuracy import load_config_predictions
from plot_confusion_matrix import (ARCHIVE, build_figure_single, SPINE_COLOR, _set_rcparams, draw_confusion_matrix,
                                    load_soundscape, load_synthetic)
from plot_data import REGIONAL_DATASETS
from plot_style import TWO_COL_WIDTH_IN, save_fig

PRED_CLIP_MAX = 14  # same clip as plot_rq5.py's pooled confusion matrices
BIRDMAE = "Bird-MAE-Huge"


def _soundscape_from_df(df: pd.DataFrame):
    low = df["y_true.min_polyphony"].round().astype(int).to_numpy()
    high = df["y_true.max_polyphony"].round().astype(int).to_numpy()
    y_pred = df["predictions.polyphony_reg"].round().astype(int).to_numpy()
    return np.clip(y_pred, low, high), y_pred


def load_tc_config_soundscape(label: str):
    csv_df = pd.read_csv(ARCHIVE / "Spatial-Embeddings" / BIRDMAE / "scape_eval_results"
                         / "soundscape_test_metrics.csv", index_col=0)
    return _soundscape_from_df(load_config_predictions(BIRDMAE, label, csv_df))


def soundscape_panels():
    return [
        ("Region-specific pooled-MLP", load_soundscape(ARCHIVE / "Pooled-Embeddings" / BIRDMAE, REGIONAL_DATASETS)),
        ("TC + frame-act", load_tc_config_soundscape("+TC-head+frame-level call activity")),
        ("XCM-frozen", load_soundscape(ARCHIVE / "XCM-Generalization" / BIRDMAE, REGIONAL_DATASETS)),
        ("XCM-fine-tuned", load_soundscape(ARCHIVE / "XCM-Generalization-fine-tune" / "BirdSetBirdMAE",
                                            REGIONAL_DATASETS)),
    ]


def synthetic_panels():
    return [
        ("Region-specific pooled-MLP", load_synthetic(ARCHIVE / "Pooled-Embeddings" / BIRDMAE, REGIONAL_DATASETS)),
        ("XCM-fine-tuned", load_synthetic(ARCHIVE / "XCM-Generalization-fine-tune" / "BirdSetBirdMAE",
                                           REGIONAL_DATASETS)),
    ]


def soundscape_region_vs_xcm_ft_panels():
    panels = soundscape_panels()
    return [panels[0], panels[3]]


def xcm_finetuned_xcm_synthetic():
    return load_synthetic(ARCHIVE / "Fine-Tuning" / "BirdSetBirdMAE", ["XCM"])


def build_grid_figure(panels, ncols: int = 2):
    _set_rcparams()
    clipped = [(yt, np.clip(yp, 0, PRED_CLIP_MAX)) for _, (yt, yp) in panels]
    labels_all = sorted({0} | {int(v) for yt, yp in clipped for v in np.concatenate([yt, yp])})
    cms = [confusion_matrix(yt, yp, labels=labels_all) for yt, yp in clipped]
    row_keep = [i for i in range(len(labels_all)) if any(cm[i, :].sum() > 0 for cm in cms)]
    col_keep = [j for j in range(len(labels_all)) if any(cm[:, j].sum() > 0 for cm in cms)]
    row_labels = [labels_all[i] for i in row_keep]
    col_labels = [labels_all[j] for j in col_keep]

    nrows = int(np.ceil(len(panels) / ncols))
    panel_w = TWO_COL_WIDTH_IN / ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(TWO_COL_WIDTH_IN, nrows * panel_w * len(row_labels) / len(col_labels) + 0.6),
                              squeeze=False)
    im = None
    for ax, (title, _), cm in zip(axes.flat, panels, cms):
        cm = cm[np.ix_(row_keep, col_keep)]
        im, n = draw_confusion_matrix(ax, cm, row_labels, col_labels, "True polyphony degree", annotate="pct")
        ax.set_title(f"{title}  (n={n:,})", loc="left")
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axes, shrink=0.7, aspect=35, pad=0.02)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels([f"{v}%" for v in [0, 20, 40, 60, 80, 100]])
    cbar.set_label("Row-normalized % (cell labels; <1% omitted)")
    cbar.outline.set_edgecolor(SPINE_COLOR)
    cbar.outline.set_linewidth(0.8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/confusion_matrix"))
    args = parser.parse_args()

    for name, panels in [("soundscape", soundscape_panels()), ("synthetic", synthetic_panels())]:
        fig = build_grid_figure(panels)
        save_fig(fig, args.out_dir / f"confusion_matrix_birdmae_configs_{name}")
        plt.close(fig)
        print(f"Wrote {args.out_dir / f'confusion_matrix_birdmae_configs_{name}'}.(png|pdf)")

    fig = build_grid_figure(soundscape_region_vs_xcm_ft_panels())
    save_fig(fig, args.out_dir / "confusion_matrix_birdmae_region_vs_xcmft_soundscape")
    plt.close(fig)
    _set_rcparams()
    fig = build_figure_single(xcm_finetuned_xcm_synthetic(), "synthetic")
    save_fig(fig, args.out_dir / "confusion_matrix_birdmae_xcmft_xcm_synthetic")
    plt.close(fig)
    print("Wrote region_vs_xcmft_soundscape and xcmft_xcm_synthetic figures")


if __name__ == "__main__":
    main()
