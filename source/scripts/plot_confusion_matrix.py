#!/usr/bin/env python3
"""
Confusion matrix -- synthetic vs. soundscape test data, for a fine-tuned
backbone's polyphony-degree regression head. Two heatmap panels, sharing a
row-normalized-percentage color scale, built by pooling per-region result
pickles from one archive/{study}/{model}/ directory:

  - Synthetic panel: eval_results/{region}_reg_test_results.pkl, ground
    truth is the exact polyphony degree (y_true.polyphony) -- the diagonal
    is an exact match.

  - Soundscape panel: scape_eval_results/{region}_reg_scape_test_results.pkl,
    ground truth is a [min_polyphony, max_polyphony] interval (real
    soundscape clips have no single exact label). A prediction is "correct"
    whenever it falls anywhere inside that interval -- the same criterion
    utils/metrics.py uses for range_accuracy (effective_error == 0) -- so
    the row plotted against each prediction is the prediction itself
    clipped into [min, max]: within-range predictions land on the diagonal
    by construction, and off-diagonal cells show which bound (and by how
    much) an incorrect prediction missed.

Default source is archive/XCM-Generalization-fine-tune/BirdSetBirdMAE
(Bird-MAE fine-tuned), pooling all 7 region-specific fine-tunes (HSN/NES/
PER/POW/SNE/SSW/UHH) it contains -- the only study with a soundscape eval
for Bird-MAE at all (the plain Fine-Tuning archive's Bird-MAE run has a
synthetic/XCM eval but never had a soundscape eval). Pooling means the two
panels are not from one single checkpoint: the synthetic panel mixes 7
region-specific fine-tunes' own-region synthetic test sets, matching how
RQ5's Panel B aggregates this same study.

--study/--model/--regions make the source configurable for any other
archive/model with the same eval_results/scape_eval_results layout;
--split restricts to just the synthetic or soundscape panel (e.g. for a
study/model that never ran a soundscape eval, such as Pooled-Embeddings'
per-dataset linear probes). --models (plural) pools multiple backbones of
one study together instead of a single --model -- e.g. every backbone of
archive/XCM-Generalization/, for a study-level rather than backbone-level
confusion matrix (see plot_rq5.py's rq5_confusion_matrix_xcm_head_soundscape).

Usage:
    complete-venv/bin/python source/scripts/plot_confusion_matrix.py \\
        [--study XCM-Generalization-fine-tune] [--model BirdSetBirdMAE] \\
        [--regions HSN NES PER POW SNE SSW UHH] [--split both|synthetic|soundscape] \\
        [--out-dir plots/figures/confusion_matrix]

    # Bird-MAE, frozen backbone (Pooled-Embeddings), synthetic only, pooled
    # across all 8 datasets (HSN/NES/PER/POW/SNE/SSW/UHH/XCM):
    complete-venv/bin/python source/scripts/plot_confusion_matrix.py \\
        --study Pooled-Embeddings --model Bird-MAE-Huge --split synthetic
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, MultipleLocator
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_style import save_fig, FIGURE_FONT_PT, FIGURE_SERIF_FONTS, ONE_COL_WIDTH_IN, TWO_COL_WIDTH_IN

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = REPO_ROOT / "archive"
FONT_SIZE = FIGURE_FONT_PT
# Both panels carry per-cell count+percentage text at up to a 12x12 grid --
# too dense to read at single-column width, so both figures below are sized
# to span both columns of the target document (see plot_style.py).
CMAP = "Blues"
GRID_COLOR = "white"
SPINE_COLOR = "0.3"
DIAGONAL_COLOR = "#d62728"
RED_BORDER_LINEWIDTH = 0.8


def _set_rcparams():
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "font.family": "serif",
        "font.serif": FIGURE_SERIF_FONTS,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "axes.edgecolor": SPINE_COLOR,
    })


def _matching_files(result_dir: Path, suffix: str, regions: list[str] | None) -> list[Path]:
    files = sorted(result_dir.glob(f"*{suffix}"))
    if regions is not None:
        files = [f for f in files if f.name[: -len(suffix)] in regions]
    if not files:
        raise FileNotFoundError(f"No '*{suffix}' files found in {result_dir}")
    return files


def load_synthetic(model_dir: Path, regions: list[str] | None):
    """Pool y_true.polyphony / predictions.polyphony_reg across every
    matching eval_results/{region}_reg_test_results.pkl."""
    files = _matching_files(model_dir / "eval_results", "_reg_test_results.pkl", regions)
    df = pd.concat([pickle.load(open(f, "rb")) for f in files], ignore_index=True)
    y_true = df["y_true.polyphony"].round().astype(int).to_numpy()
    y_pred = df["predictions.polyphony_reg"].round().astype(int).to_numpy()
    return y_true, y_pred


def load_soundscape(model_dir: Path, regions: list[str] | None):
    """Pool across every matching scape_eval_results/
    {region}_reg_scape_test_results.pkl. A prediction counts as correct iff
    it falls in [min_polyphony, max_polyphony] (utils/metrics.py's
    range_accuracy), so the label plotted as "true" is the rounded
    prediction clipped into that interval: unclipped (in-range) predictions
    always land on the diagonal, clipped ones show the violated bound."""
    files = _matching_files(model_dir / "scape_eval_results", "_reg_scape_test_results.pkl", regions)
    df = pd.concat([pickle.load(open(f, "rb")) for f in files], ignore_index=True)
    low = df["y_true.min_polyphony"].round().astype(int).to_numpy()
    high = df["y_true.max_polyphony"].round().astype(int).to_numpy()
    y_pred = df["predictions.polyphony_reg"].round().astype(int).to_numpy()
    y_true = np.clip(y_pred, low, high)
    return y_true, y_pred


def load_synthetic_multi(model_dirs: list[Path], regions: list[str] | None):
    """load_synthetic(), pooled across multiple model dirs -- e.g. every
    backbone of a study, rather than one particular backbone."""
    y_trues, y_preds = zip(*(load_synthetic(d, regions) for d in model_dirs))
    return np.concatenate(y_trues), np.concatenate(y_preds)


def load_soundscape_multi(model_dirs: list[Path], regions: list[str] | None):
    """load_soundscape(), pooled across multiple model dirs -- e.g. every
    backbone of a study, rather than one particular backbone."""
    y_trues, y_preds = zip(*(load_soundscape(d, regions) for d in model_dirs))
    return np.concatenate(y_trues), np.concatenate(y_preds)


def prune_confusion_matrix(y_true, y_pred):
    """Confusion matrix with classes that have no support on an axis
    dropped -- an all-zero row (never occurred as a true label) or column
    (never predicted) carries no information, so removing it is safe for a
    paper figure. Rows and columns are pruned independently, since a class
    can have samples on one axis but not the other, which can leave a
    non-square matrix (e.g. a class predicted at least once by some other
    true class, but that itself never occurred as a true label)."""
    all_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()) | {0})
    cm_full = confusion_matrix(y_true, y_pred, labels=all_labels)

    row_keep = [i for i in range(len(all_labels)) if cm_full[i, :].sum() > 0]
    col_keep = [j for j in range(len(all_labels)) if cm_full[:, j].sum() > 0]
    row_labels = [all_labels[i] for i in row_keep]
    col_labels = [all_labels[j] for j in col_keep]
    cm = cm_full[np.ix_(row_keep, col_keep)]
    return cm, row_labels, col_labels


def _fmt_count(count: int) -> str:
    """Abbreviate above 999 -- the soundscape panel counts windows (tens of
    thousands per cell), and the raw digit string is wide enough to overlap
    a neighboring cell's text at one-column width. Above 9999, drop the
    decimal too (e.g. "29k" not "29.1k") -- at that width even the 1-decimal
    form is too wide for a dense (~12x10) grid's cells."""
    if count >= 10_000:
        return f"{round(count / 1000)}k"
    if count >= 1_000:
        return f"{count / 1000:.1f}k"
    return str(count)


def draw_confusion_matrix(ax, cm, row_labels, col_labels, ylabel: str, annotate: str = "both"):
    """Color by row-normalized percentage (0-100) rather than raw count, so
    the two panels share one meaningful scale despite very different sample
    sizes (synthetic: thousands of clips; soundscape: hundreds of thousands
    of windows) -- a shared count scale would wash out the smaller panel.

    annotate="pct" prints only the rounded row-percentage (and skips cells
    under 1%) -- for dense multi-panel grids where "count\npct" labels
    collide; the default "both" is unchanged."""
    with np.errstate(all="ignore"):
        cm_pct = np.where(cm.sum(axis=1, keepdims=True) > 0,
                           cm / cm.sum(axis=1, keepdims=True) * 100, np.nan)

    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad("white")
    im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100, aspect="equal")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.xaxis.set_major_locator(FixedLocator(range(len(col_labels))))
    ax.yaxis.set_major_locator(FixedLocator(range(len(row_labels))))
    ax.invert_yaxis()

    # Thin white cell separators (classic heatmap "gridded cell" look).
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color=GRID_COLOR, linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.8)

    ax.set_xlabel("Predicted polyphony degree")
    ax.set_ylabel(ylabel)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            if count == 0:
                continue
            color = "white" if cm_pct[i, j] > 55 else "0.15"
            if annotate == "pct":
                if cm_pct[i, j] < 1:
                    continue
                ax.text(j, i, f"{cm_pct[i, j]:.0f}", ha="center", va="center",
                         fontsize=FONT_SIZE - 2.5, color=color)
                continue
            ax.text(j, i, f"{_fmt_count(count)}\n{cm_pct[i, j]:.0f}%", ha="center", va="center",
                     fontsize=FONT_SIZE - 2.5, color=color, linespacing=1.35)

    # Outline the "correct" diagonal -- matched by label value, not index,
    # since pruning can leave row_labels != col_labels (a non-square
    # matrix), so the diagonal cell for a given class isn't always (i, i).
    col_index = {label: j for j, label in enumerate(col_labels)}
    for i, label in enumerate(row_labels):
        j = col_index.get(label)
        if j is not None:
            ax.add_patch(patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                            edgecolor=DIAGONAL_COLOR, linewidth=RED_BORDER_LINEWIDTH, zorder=3))
    return im, int(cm.sum())


SPLIT_TITLE = {"synthetic": "Synthetic test", "soundscape": "Soundscape test"}
SPLIT_SUBTITLE = {"synthetic": "diagonal = exact match", "soundscape": "diagonal = within [min, max]"}

# Output-filename slug per (--study, --model), overriding the raw lowercased
# model folder name -- needed for the default (study, model) pair since its
# folder name "BirdSetBirdMAE" doesn't read as fine-tuned Bird-MAE-Huge, and
# is otherwise indistinguishable from the frozen Pooled-Embeddings backbone
# of the same canonical model (both are "Bird-MAE-Huge" per backbone_meta.py's
# FINETUNE_NAME_TO_CANONICAL). Any (study, model) not listed here still falls
# back to the raw lowercased model name.
MODEL_SLUG = {
    ("XCM-Generalization-fine-tune", "BirdSetBirdMAE"): "birdmae_finetune",
    ("Pooled-Embeddings", "Bird-MAE-Huge"): "birdmae_pooled",
}


def build_figure(synthetic, soundscape):
    """Panels stacked vertically, not side by side: each cell carries a
    2-line count+percentage label at a fixed font size, so at
    TWO_COL_WIDTH_IN there isn't enough width for both ~10-12-column panels
    side by side without crushing that text -- stacking gives each panel the
    full column width instead of half of it. No in-figure heading -- the
    model name / config that used to go there belongs in the caption, which
    describes each instance of this figure already."""
    _set_rcparams()
    cm_syn, row_syn, col_syn = prune_confusion_matrix(*synthetic)
    cm_sc, row_sc, col_sc = prune_confusion_matrix(*soundscape)

    fig, axes = plt.subplots(
        2, 1, figsize=(TWO_COL_WIDTH_IN, 0.53 * TWO_COL_WIDTH_IN * (len(row_syn) + len(row_sc)) / max(len(col_syn), len(col_sc))),
        gridspec_kw={"height_ratios": [len(row_syn), len(row_sc)], "hspace": 0.35},
    )
    im, n_syn = draw_confusion_matrix(axes[0], cm_syn, row_syn, col_syn, "True polyphony degree")
    axes[0].set_title(f"{SPLIT_TITLE['synthetic']}  (n={n_syn:,})\n{SPLIT_SUBTITLE['synthetic']}", loc="left")
    _, n_sc = draw_confusion_matrix(axes[1], cm_sc, row_sc, col_sc, "True polyphony degree")
    axes[1].set_title(f"{SPLIT_TITLE['soundscape']}  (n={n_sc:,})\n{SPLIT_SUBTITLE['soundscape']}", loc="left")

    cbar = fig.colorbar(im, ax=axes, shrink=0.7, aspect=35, pad=0.02, label="Row-normalized %")
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.outline.set_edgecolor(SPINE_COLOR)
    cbar.outline.set_linewidth(0.8)
    return fig


def build_figure_single(data, split: str):
    """Same styling as build_figure(), but a single panel spanning one
    column instead of two -- for a study/model that only has one of the two
    splits available (e.g. Pooled-Embeddings never runs a soundscape eval
    for its per-dataset linear probes). No in-figure title -- the split name,
    n, and the diagonal convention (SPLIT_TITLE/SPLIT_SUBTITLE) belong in the
    caption instead, freeing that vertical space for the (already dense)
    cells. Colorbar ticks carry their own "%" rather than a separate
    "Row-normalized %" axis label, for the same reason."""
    _set_rcparams()
    cm, row_labels, col_labels = prune_confusion_matrix(*data)

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_IN, 1.02 * ONE_COL_WIDTH_IN * len(row_labels) / len(col_labels)))
    im, n = draw_confusion_matrix(ax, cm, row_labels, col_labels, "True polyphony degree")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=25, pad=0.015)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels([f"{v}%" for v in [0, 20, 40, 60, 80, 100]])
    cbar.outline.set_edgecolor(SPINE_COLOR)
    cbar.outline.set_linewidth(0.8)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", default="XCM-Generalization-fine-tune",
                         help="Archive study folder under archive/ (default: %(default)s)")
    parser.add_argument("--model", default="BirdSetBirdMAE",
                         help="Model folder under archive/{study}/ (default: %(default)s)")
    parser.add_argument("--models", nargs="+", default=None,
                         help="Multiple model folders under archive/{study}/ to pool together "
                              "(overrides --model) -- e.g. every backbone of a study")
    parser.add_argument("--regions", nargs="+", default=None,
                         help="Region/dataset codes to pool (default: every one found in eval_results/)")
    parser.add_argument("--split", choices=["both", "synthetic", "soundscape"], default="both",
                         help="Which panel(s) to build (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=Path("plots/figures/confusion_matrix"))
    parser.add_argument("--out-name", default=None,
                         help="Output file stem (default: derived from --model and --split)")
    parser.add_argument("--clip-max", type=int, default=None,
                         help="Clip predictions to [0, clip-max] before building the matrix -- "
                              "some backbones (e.g. EfficientNet-B1) produce a handful of wildly "
                              "miscalibrated predictions when pooled across many regions/backbones, "
                              "which otherwise balloon the matrix to dozens of sparse columns (see "
                              "plot_rq5.py's PRED_CLIP_MAX). Off by default.")
    args = parser.parse_args()

    if args.models:
        model_dirs = [ARCHIVE / args.study / m for m in args.models]
        synthetic_loader = lambda regions: load_synthetic_multi(model_dirs, regions)
        soundscape_loader = lambda regions: load_soundscape_multi(model_dirs, regions)
    else:
        model_dir = ARCHIVE / args.study / args.model
        synthetic_loader = lambda regions: load_synthetic(model_dir, regions)
        soundscape_loader = lambda regions: load_soundscape(model_dir, regions)

    def clip(data):
        if args.clip_max is None:
            return data
        y_true, y_pred = data
        return y_true, np.clip(y_pred, 0, args.clip_max)

    if args.split == "both":
        synthetic = clip(synthetic_loader(args.regions))
        soundscape = clip(soundscape_loader(args.regions))
        fig = build_figure(synthetic, soundscape)
    else:
        loader = synthetic_loader if args.split == "synthetic" else soundscape_loader
        data = clip(loader(args.regions))
        fig = build_figure_single(data, args.split)

    slug = MODEL_SLUG.get((args.study, args.model), args.model.lower())
    out_name = args.out_name or f"confusion_matrix_{slug}_{'combined' if args.split == 'both' else args.split}"
    save_fig(fig, args.out_dir / out_name)
    plt.close(fig)

    print(f"Wrote figures to {args.out_dir / out_name}.(png|pdf)")


if __name__ == "__main__":
    main()
