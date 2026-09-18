#!/usr/bin/env python3
"""
Confusion matrix for Perch v2's TC+Pol-Class (reg_and_class) multi-task
config, soundscape test only -- the config behind the top_models table's
bolded Soundscape row (MAE 0.68 / Accuracy 0.52 / Off-by-one 0.86).

Unlike the plain single-task and fine-tuned studies (plot_confusion_matrix.py),
archive/Spatial-Embeddings/perch_v2_cpu/ ran ~11 head configs per region as
separate individual training runs, and only ONE config's raw predictions
survive at the pooled top-level scape_eval_results/{region}_reg_scape_test_
results.pkl (each got overwritten by whichever config ran last in the sweep
-- currently the "both-aux" config for every region, confirmed by inspecting
those files' columns). The aggregate metrics for every config did survive in
scape_eval_results/soundscape_test_metrics.csv, and each individual run's own
logs/test_results.pkl still holds that run's raw soundscape predictions
(soundscape ground truth, not synthetic -- confirmed via test_metrics.json's
range_accuracy-style metrics and matching against the CSV's
{region}_reg_and_class_reg column).

RUN_DIR_BY_REGION below was found by scanning every run's logs/params.yaml
for objectives == {polyphony_class, polyphony_reg} and matching its
logs/test_metrics.json range_mae against soundscape_test_metrics.csv's
{region}_reg_and_class_reg row -- exact match for all 7 regions (BirdSet has
no soundscape recordings for XCM, so only 7 of the study's 8 regions apply).
"""

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_confusion_matrix import build_figure_single, prune_confusion_matrix, draw_confusion_matrix
from plot_style import save_fig, ONE_COL_WIDTH_IN

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = REPO_ROOT / "archive" / "Spatial-Embeddings" / "perch_v2_cpu"

RUN_DIR_BY_REGION = {
    "HSN": "20260808_150133_based-trey",
    "POW": "20260808_150126_split-cuss",
    "UHH": "20260808_150143_misty-loof",
    "SNE": "20260808_150127_soupy-fids",
    "PER": "20260808_150126_piled-mete",
    "NES": "20260808_150132_zesty-colt",
    "SSW": "20260808_150129_swish-tola",
}


def load_run_soundscape(run_dir: Path):
    df = pickle.load(open(run_dir / "logs" / "test_results.pkl", "rb"))
    low = df["y_true.min_polyphony"].round().astype(int).to_numpy()
    high = df["y_true.max_polyphony"].round().astype(int).to_numpy()
    y_pred = df["predictions.polyphony_reg"].round().astype(int).to_numpy()
    y_true = np.clip(y_pred, low, high)
    return y_true, y_pred


def main():
    y_trues, y_preds = [], []
    for region, run_name in RUN_DIR_BY_REGION.items():
        y_true, y_pred = load_run_soundscape(ARCHIVE / run_name)
        y_trues.append(y_true)
        y_preds.append(y_pred)
    data = np.concatenate(y_trues), np.concatenate(y_preds)

    fig = build_figure_single(data, "soundscape")
    out_dir = REPO_ROOT / "confusion_matrix" / "top_models"
    out_name = "confusion_matrix_perchv2_tc_polclass_soundscape"
    save_fig(fig, out_dir / out_name)
    plt.close(fig)
    print(f"Wrote figures to {out_dir / out_name}.(png|pdf)")


if __name__ == "__main__":
    main()
