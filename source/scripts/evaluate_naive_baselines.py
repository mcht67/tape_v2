#!/usr/bin/env python3
"""Audio-free baseline predictors for polyphony estimation, scored on the
synthetic test split and the soundscape test_5s split.

Purpose: give the model rows in the result tables a floor -- what can be
achieved without looking at the audio at all. Every predictor is fitted on
the synthetic *train* split only (after the same max_polyphony/SNR filtering
the models get, via filter_dataset_by_polyphony_and_snr()) and then applied
unchanged to both test sets, mirroring how the trained models are evaluated.

Predictors (total polyphony only, integer outputs):
  * constant_mode   -- always the most frequent train level.
  * constant_median -- always the (rounded) train median; the MAE-optimal
                       constant, which can differ from the mode on skewed data.
  * uniform_random  -- uniform over 0..K, K = max_polyphony (default 8, or the
                       highest train level if None), i.e. the label space of
                       the trained heads.
  * train_prior     -- sampled from the empirical train histogram p(k).
  * test_mode / test_median / test_prior -- the same, but fitted on the label
                       distribution of the split being evaluated (synthetic
                       test labels; for soundscape, the levels of its
                       unambiguous clips, min_polyphony == max_polyphony).
                       Oracle baselines: they know the test label frequencies.
The stochastic predictors are run over --n-seeds seeds; every metric is
reported as mean/std across seeds (std = 0 for the constant ones).

Metrics come from the same functions the evaluation scripts use
(utils/metrics.py), with cm_type="regression_round":
  * synthetic:  compute_polyphony_metrics()       -- exact ground truth.
  * soundscape: compute_polyphony_range_metrics() -- [min, max] ground truth.
    Headline: macro_mae_unambiguous_1_6 = macro-MAE over true levels 1-6 on
    unambiguous clips (min_polyphony == max_polyphony), the paper's
    Macro-MAE(1-6) (see compute_rq1_macro_mae_table.py); second metric:
    macro_mae_all_bins = the range metrics' macro_mae_unambiguous_all_bins.
No macro-MAE is summarised for synthetic data: its levels are balanced, so
macro-MAE ~= MAE there.

Outputs (in --out-dir):
  naive_baselines_long.csv  -- every scalar metric, one row per
                               subset/dataset/predictor/metric.
  naive_baselines_wide.csv  -- key metrics, one row per subset/dataset/predictor.
  fitted_priors.json        -- train histogram, mode, median, K, filters.

Usage:
  venv/bin/python source/scripts/evaluate_naive_baselines.py --subsets HSN --n-seeds 5
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_from_disk
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.dataset import filter_dataset_by_polyphony_and_snr, get_local_data_dir  # noqa: E402
from utils.metrics import (compute_polyphony_metrics, compute_polyphony_range_metrics,  # noqa: E402
                           macro_mae_by_level)

NO_SOUNDSCAPE_SUBSETS = {"XCM", "XCL"}

WIDE_METRICS = {
    "synthetic": ["mae", "accuracy", "macro_f1"],
    "soundscape": ["macro_mae_unambiguous_1_6", "macro_mae_unambiguous_1_6_n_levels_used",
                   "macro_mae_all_bins", "range_mae", "range_accuracy"],
}


def fit_distribution(levels, max_level=None):
    """Histogram, mode and median of a set of integer polyphony levels."""
    levels = np.asarray(levels, dtype=int)
    K = int(max_level) if max_level is not None else int(levels.max())
    counts = np.bincount(levels, minlength=K + 1)[:K + 1]
    return {
        "K": K,
        "counts": counts.tolist(),
        "p": (counts / counts.sum()).tolist(),
        "mode": int(np.argmax(counts)),
        "median": int(np.round(np.median(levels))),
        "n": int(len(levels)),
    }


# predictor -> (kind, which fitted distribution it uses). "test" is the label
# distribution of the split being evaluated (for soundscape: its unambiguous
# clips), i.e. an oracle that knows the test label frequencies but not the audio.
PREDICTORS = {
    "constant_mode": ("mode", "train"),
    "constant_median": ("median", "train"),
    "test_mode": ("mode", "test"),
    "test_median": ("median", "test"),
    "uniform_random": ("uniform", "train"),
    "train_prior": ("prior", "train"),
    "test_prior": ("prior", "test"),
}
STOCHASTIC = {name for name, (kind, _) in PREDICTORS.items() if kind in ("uniform", "prior")}


def predict(name, fits, n, rng):
    kind, source = PREDICTORS[name]
    fitted = fits[source]
    if kind in ("mode", "median"):
        return np.full(n, fitted[kind])
    if kind == "uniform":
        return rng.integers(0, fitted["K"] + 1, n)
    if kind == "prior":
        return rng.choice(fitted["K"] + 1, n, p=fitted["p"])
    raise ValueError(f"Unknown predictor: {name}")


def _scalars(d):
    """Keep only the scalar entries of a metrics 'overall' dict."""
    return {k: float(v) for k, v in d.items()
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)}


def synthetic_metrics(y_true, y_pred):
    report = compute_polyphony_metrics(y_true[:, None], y_pred[:, None],
                                       cm_type="regression_round", per_species=False, print_cm=False)
    return _scalars(report["overall"])


def soundscape_metrics(y_min, y_max, y_pred):
    report = compute_polyphony_range_metrics({"min_polyphony": y_min, "max_polyphony": y_max},
                                             y_pred[:, None], cm_type="regression_round",
                                             per_species=False, print_cm=False)
    metrics = _scalars(report["overall"])
    for width, stats in report["overall_by_interval_width"].items():
        metrics[f"{width}/range_accuracy"] = stats["range_accuracy"]
        metrics[f"{width}/support"] = stats["support"]

    unamb = y_min == y_max
    stats_1_6 = macro_mae_by_level(y_min[unamb], np.abs(y_min[unamb] - y_pred[unamb]), range(1, 7))
    metrics["macro_mae_unambiguous_1_6"] = stats_1_6["macro_mae"]
    metrics["macro_mae_unambiguous_1_6_n_levels_used"] = stats_1_6["n_levels_used"]
    metrics["macro_mae_all_bins"] = metrics["macro_mae_unambiguous_all_bins_macro_mae"]
    return metrics


def evaluate_predictors(metric_fn, n, fits, n_seeds, seed, **truth):
    """Run every predictor (stochastic ones over n_seeds) -> {predictor: {metric: [values]}}."""
    out = {}
    for name in PREDICTORS:
        runs = n_seeds if name in STOCHASTIC else 1
        per_metric = {}
        for r in range(runs):
            rng = np.random.default_rng(seed + r)
            for k, v in metric_fn(y_pred=predict(name, fits, n, rng), **truth).items():
                per_metric.setdefault(k, []).append(v)
        out[name] = per_metric
    return out


def to_rows(subset, dataset, results, n_clips):
    rows = []
    for predictor, per_metric in results.items():
        for metric, values in per_metric.items():
            values = np.asarray(values, dtype=float)
            rows.append({"subset": subset, "dataset": dataset, "predictor": predictor, "metric": metric,
                         "mean": float(np.nanmean(values)) if np.isfinite(values).any() else float("nan"),
                         "std": float(np.nanstd(values)) if np.isfinite(values).any() else float("nan"),
                         "n_seeds": len(values), "n_clips": n_clips})
    return rows


def load_soundscape(subset, default_dir):
    path = os.path.join(default_dir, get_local_data_dir(dataset_config=f"{subset}_soundscape_test", subset=subset))
    if not os.path.exists(path):
        return None
    ds = load_from_disk(path)
    if isinstance(ds, DatasetDict):
        ds = ds["test_5s"]
    return ds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subsets", nargs="+", default=None,
                        help="Subsets to evaluate (default: every data/<SUBSET>/<SUBSET>_<suffix> found).")
    parser.add_argument("--train-config-suffix", default="polyphonic",
                        help="Synthetic dataset config is <SUBSET>_<suffix> (default: polyphonic).")
    parser.add_argument("--max-polyphony", type=int, default=8,
                        help="Filter the synthetic splits to polyphony <= this and use 0..this as the label space.")
    parser.add_argument("--min-snr", type=float, default=None)
    parser.add_argument("--snr-range", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("results/naive_baselines"))
    args = parser.parse_args()

    default_dir = os.environ.get("DEFAULT_DIR", "")
    subsets = args.subsets or sorted(
        p.name for p in Path(default_dir, "data").iterdir()
        if (p / f"{p.name}_{args.train_config_suffix}").exists())
    filter_cfg = OmegaConf.create({"dataset": {"max_polyphony": args.max_polyphony, "min_snr": args.min_snr,
                                               "snr_range": args.snr_range}})

    rows, priors = [], {}
    for subset in subsets:
        train_config = f"{subset}_{args.train_config_suffix}"
        dataset_dir = os.path.join(default_dir, get_local_data_dir(dataset_config=train_config, subset=subset))
        print(f"\n=== {subset}: loading {dataset_dir}")
        dataset = load_from_disk(dataset_dir)
        for split in list(dataset.keys()):
            if "polyphony" not in dataset[split].column_names and "polyphony_degree" in dataset[split].column_names:
                dataset[split] = dataset[split].rename_column("polyphony_degree", "polyphony")
            label_cols = [c for c in ("polyphony", "snr_dB") if c in dataset[split].column_names]
            dataset[split] = dataset[split].select_columns(label_cols)
        dataset = filter_dataset_by_polyphony_and_snr(dataset, filter_cfg)

        train_fit = fit_distribution(dataset["train"]["polyphony"], args.max_polyphony)
        y_test = np.asarray(dataset["test"]["polyphony"], dtype=int)
        synthetic_test_fit = fit_distribution(y_test)
        priors[subset] = {"train": train_fit, "synthetic_test": synthetic_test_fit, "train_config": train_config,
                          "max_polyphony": args.max_polyphony, "min_snr": args.min_snr, "snr_range": args.snr_range}
        print(f"train: K={train_fit['K']}, mode={train_fit['mode']}, median={train_fit['median']}, "
              f"counts={train_fit['counts']}")

        results = evaluate_predictors(synthetic_metrics, len(y_test),
                                      {"train": train_fit, "test": synthetic_test_fit},
                                      args.n_seeds, args.seed, y_true=y_test)
        rows += to_rows(subset, "synthetic", results, len(y_test))

        if subset in NO_SOUNDSCAPE_SUBSETS:
            continue
        scape = load_soundscape(subset, default_dir)
        if scape is None:
            print(f"No soundscape data found for {subset}, skipping soundscape evaluation.")
            continue
        scape = scape.select_columns(["min_polyphony", "max_polyphony"])
        y_min = np.asarray(scape["min_polyphony"], dtype=int)
        y_max = np.asarray(scape["max_polyphony"], dtype=int)
        unamb = y_min == y_max
        scape_fit = fit_distribution(y_min[unamb])
        priors[subset]["soundscape_unambiguous"] = scape_fit
        print(f"Soundscape: {len(y_min)} clips, {int(unamb.sum())} unambiguous; mode={scape_fit['mode']}, "
              f"median={scape_fit['median']}, counts={scape_fit['counts']}")
        results = evaluate_predictors(soundscape_metrics, len(y_min), {"train": train_fit, "test": scape_fit},
                                      args.n_seeds, args.seed, y_min=y_min, y_max=y_max)
        rows += to_rows(subset, "soundscape", results, len(y_min))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long_df = pd.DataFrame(rows)
    long_df.to_csv(args.out_dir / "naive_baselines_long.csv", index=False)

    wide_parts = []
    for dataset_name, metrics in WIDE_METRICS.items():
        part = long_df[(long_df["dataset"] == dataset_name) & long_df["metric"].isin(metrics)]
        if part.empty:
            continue
        part = part.pivot_table(index=["subset", "dataset", "predictor"], columns="metric",
                                values=["mean", "std"], dropna=False)
        part.columns = [f"{metric}_{stat}" for stat, metric in part.columns]
        wide_parts.append(part[[f"{m}_{s}" for m in metrics for s in ("mean", "std") if f"{m}_{s}" in part.columns]])
    wide_df = pd.concat(wide_parts).reset_index() if wide_parts else pd.DataFrame()
    wide_df.to_csv(args.out_dir / "naive_baselines_wide.csv", index=False)

    with open(args.out_dir / "fitted_priors.json", "w") as f:
        json.dump(priors, f, indent=2)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        for dataset_name, part in wide_df.groupby("dataset", sort=False):
            print(f"\n[{dataset_name}]\n", part.dropna(axis=1, how="all").round(3).to_string(index=False))
    print(f"\nSaved results to {args.out_dir}")


if __name__ == "__main__":
    main()
