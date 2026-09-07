#!/usr/bin/env python3
"""
Recompute and fill in missing columns in per-model metrics CSVs from their
raw prediction pickles.

For each model directory under --target_dir, this looks at:

    eval_results/{subset}_{head}_test_results.pkl        -> eval_results/test_metrics.csv
    val_eval_results/{subset}_{head}_val_results.pkl      -> val_eval_results/val_metrics.csv
    scape_eval_results/{subset}_{head}_scape_test_results.pkl
                                                           -> scape_eval_results/soundscape_test_metrics.csv

`head` is "reg" or "class" (optionally prefixed with "species_", which is
folded into the same "reg"/"class" column key, mirroring
utils.evaluation.build_update_metrics_table). For every pickle whose
corresponding "{subset}_{head}" column is missing from the target CSV, the
metrics are recomputed from the pickle (using the same
compute_polyphony_metrics / compute_polyphony_range_metrics functions the
original evaluation scripts use) and written into the CSV.

Usage:
    python fill_missing_metrics.py --target_dir /path/to/study [--dry_run]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> source/

from utils.metrics import compute_polyphony_metrics, compute_polyphony_range_metrics
from utils.evaluation import flatten_metrics

# (subdir, pkl suffix, csv filename, is_range)
SOURCE_KINDS = [
    ("eval_results", "_test_results.pkl", "test_metrics.csv", False),
    ("val_eval_results", "_val_results.pkl", "val_metrics.csv", False),
    ("scape_eval_results", "_scape_test_results.pkl", "soundscape_test_metrics.csv", True),
]

STEM_RE = re.compile(r"^(?P<dataset>.+?)_(?:species_)?(?P<head>reg|class)$")


def parse_stem(stem: str):
    """'{subset}_{reg|class}' or '{subset}_species_{reg|class}' -> (subset, head, column)."""
    m = STEM_RE.match(stem)
    if not m:
        return None
    dataset, head = m.group("dataset"), m.group("head")
    return dataset, head, f"{dataset}_{head}"


def load_arrays(pkl_path: Path, head: str):
    df = pd.read_pickle(pkl_path)

    y_true = {}
    if "y_true.polyphony" in df.columns:
        y_true["polyphony"] = df["y_true.polyphony"].to_numpy(dtype=float)
    if "y_true.min_polyphony" in df.columns:
        y_true["min_polyphony"] = df["y_true.min_polyphony"].to_numpy(dtype=float)
    if "y_true.max_polyphony" in df.columns:
        y_true["max_polyphony"] = df["y_true.max_polyphony"].to_numpy(dtype=float)

    pred_col = f"predictions.polyphony_{head}"
    if pred_col not in df.columns:
        raise ValueError(f"Expected column '{pred_col}' not found in {pkl_path} (found {df.columns.tolist()})")

    if head == "reg":
        predictions = df[pred_col].to_numpy(dtype=float)
    else:
        predictions = np.stack(df[pred_col].to_numpy())

    return y_true, predictions


def compute_report(y_true, predictions, head: str, is_range: bool):
    cm_type = "regression_round" if head == "reg" else "classification"

    if not is_range:
        if head == "reg":
            return compute_polyphony_metrics(
                y_true["polyphony"][:, None], predictions[:, None],
                cm_type=cm_type, per_species=False,
            )
        return compute_polyphony_metrics(
            y_true["polyphony"][:, None], predictions[:, None, :],
            cm_type=cm_type, num_classes=predictions.shape[-1], per_species=False,
        )

    if head == "reg":
        return compute_polyphony_range_metrics(
            y_true, predictions[:, None], min_key="min_polyphony", max_key="max_polyphony",
            cm_type=cm_type, per_species=False,
        )
    return compute_polyphony_range_metrics(
        y_true, predictions[:, None, :], min_key="min_polyphony", max_key="max_polyphony",
        cm_type=cm_type, num_classes=predictions.shape[-1], per_species=False,
    )


def fill_model(model_dir: Path, dry_run: bool) -> int:
    n_filled = 0
    for subdir, suffix, csv_name, is_range in SOURCE_KINDS:
        result_dir = model_dir / subdir
        if not result_dir.is_dir():
            continue

        pkl_paths = sorted(result_dir.glob(f"*{suffix}"))
        if not pkl_paths:
            continue

        csv_path = result_dir / csv_name
        if csv_path.is_file():
            table = pd.read_csv(csv_path, index_col=0)
        else:
            table = pd.DataFrame()

        changed = False
        for pkl_path in pkl_paths:
            stem = pkl_path.name[: -len(suffix)]
            parsed = parse_stem(stem)
            if parsed is None:
                print(f"  [warn] {model_dir.name}/{subdir}: skipping unrecognized pickle name '{pkl_path.name}'")
                continue
            dataset, head, column = parsed

            if column in table.columns:
                continue

            print(f"  {model_dir.name}/{subdir}: filling missing column '{column}' from {pkl_path.name}")
            y_true, predictions = load_arrays(pkl_path, head)
            report = compute_report(y_true, predictions, head, is_range)
            flat = flatten_metrics(report)
            table[column] = pd.Series(flat, name=column)
            changed = True
            n_filled += 1

        if changed and not dry_run:
            table.to_csv(csv_path, float_format="%.4f")
            print(f"  -> wrote {csv_path}")

    return n_filled


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target_dir", type=Path, required=True, help="Study root containing per-model result subfolders")
    parser.add_argument("--dry_run", action="store_true", help="Report what would be filled without writing any CSVs")
    args = parser.parse_args()

    total = 0
    for model_dir in sorted(args.target_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        total += fill_model(model_dir, args.dry_run)

    if total == 0:
        print("No missing columns found.")
    else:
        print(f"\n{'Would fill' if args.dry_run else 'Filled'} {total} missing column(s).")


if __name__ == "__main__":
    main()
