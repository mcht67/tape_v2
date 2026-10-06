#!/usr/bin/env python3
"""Render confusion-matrix PNGs from the *_confusion_matrix.csv files the
evaluation scripts write.

evaluate_on_test_split.py, evaluate_on_val_split.py, and
evaluate_on_soundscape_data.py each call utils/metrics.py's
compute_polyphony_metrics()/compute_polyphony_range_metrics() with a
cm_csv_path per objective, which writes the aggregate confusion matrix
(labels x labels integer counts -- for the soundscape/range-ground-truth
case, restricted to unambiguous clips; see that module's docstring) as a
small CSV next to that run's test_metrics.json/val_metrics.json:
{objective}_confusion_matrix.csv, e.g. polyphony_reg_confusion_matrix.csv.

Those runs may happen on a remote/cluster machine (see exp_workflow_job_*.sh),
and the CSV -- a few hundred integers at most, since labels are polyphony
levels (0..max_polyphony), not species -- is cheap to write there and sync
back through the existing archive/ pipeline alongside test_metrics.json.
Rendering the actual heatmap PNG is deliberately left to this script instead,
meant to be run locally against a synced archive/ checkout: a remote run
never needs to produce, and you never need to download, a PNG per confusion
matrix, only the CSV.

Reuses plot_confusion_matrix.py's draw_confusion_matrix() (row-normalized
percentage color scale, count+percentage cell labels, diagonal outline) for
one CSV -> one single-panel PNG, so the rendered style matches the paper
figures that script builds from raw predictions directly.

Usage:
    # Render every *_confusion_matrix.csv found under archive/ (default),
    # each PNG saved alongside its source CSV:
    complete-venv/bin/python source/scripts/render_confusion_matrices.py

    # Restrict to one study/run directory, or a single CSV file:
    complete-venv/bin/python source/scripts/render_confusion_matrices.py \\
        --root archive/Pooled-Embeddings/Bird-MAE-Huge

    complete-venv/bin/python source/scripts/render_confusion_matrices.py \\
        --root eval_output_to_archive/test-study/study_subfolder/20260927_120000_abcdef/logs/polyphony_reg_confusion_matrix.csv

    # Collect every rendered PNG into one flat directory instead of
    # alongside each source CSV:
    complete-venv/bin/python source/scripts/render_confusion_matrices.py \\
        --out-dir plots/figures/confusion_matrices
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_confusion_matrix import _set_rcparams, draw_confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "archive"
DEFAULT_PATTERN = "*_confusion_matrix.csv"


def find_csvs(root: Path, pattern: str) -> list[Path]:
    """A single CSV file is rendered as-is; a directory is searched
    recursively (covers scanning the whole archive/ tree, one study, or one
    run's own logs/ dir alike)."""
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    return sorted(root.rglob(pattern))


def load_confusion_matrix(csv_path: Path):
    """print_confusion_matrix()'s CSV layout: first column 'true' is the
    index, remaining columns are the predicted-label header -- both round-
    trip through CSV as strings, so they're cast back to int here (labels
    are always polyphony levels, e.g. -1..9, never species names)."""
    df = pd.read_csv(csv_path, index_col=0)
    row_labels = [int(x) for x in df.index]
    col_labels = [int(x) for x in df.columns]
    return df.to_numpy(), row_labels, col_labels


def render_one(csv_path: Path, out_dir: Path | None) -> Path | None:
    cm, row_labels, col_labels = load_confusion_matrix(csv_path)
    if cm.sum() == 0:
        print(f"[skip] {csv_path}: empty confusion matrix (no data)")
        return None

    objective = csv_path.stem.replace("_confusion_matrix", "")
    _set_rcparams()
    # Sized to the label count, same idea as plot_confusion_matrix.py's own
    # figures, since the number of polyphony levels present (hence the grid
    # size) varies by objective/dataset.
    fig, ax = plt.subplots(figsize=(0.55 * len(col_labels) + 2.2, 0.55 * len(row_labels) + 2.2))
    draw_confusion_matrix(ax, cm, row_labels, col_labels, ylabel="True polyphony degree")
    ax.set_title(objective)
    fig.tight_layout()

    target_dir = out_dir if out_dir is not None else csv_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    # Flat --out-dir can collect CSVs of the same objective name from
    # different runs, so disambiguate with a short parent-path stem when
    # writing there instead of alongside the source CSV.
    stem = csv_path.stem if out_dir is None else f"{csv_path.parent.name}_{csv_path.stem}"
    png_path = target_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help=f"Directory to search recursively, or a single CSV file (default: {DEFAULT_ROOT}).")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN,
                        help=f"Glob pattern used when --root is a directory (default: {DEFAULT_PATTERN}).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to save PNGs. Default: alongside each source CSV.")
    args = parser.parse_args()

    csvs = find_csvs(args.root, args.pattern)
    if not csvs:
        print(f"No files matching '{args.pattern}' found under {args.root}")
        return

    print(f"Found {len(csvs)} confusion-matrix CSV(s) under {args.root}")
    written = 0
    for csv_path in csvs:
        png_path = render_one(csv_path, args.out_dir)
        if png_path is not None:
            print(f"  Wrote {png_path}")
            written += 1
    print(f"Rendered {written}/{len(csvs)} confusion matrix PNG(s).")


if __name__ == "__main__":
    main()
