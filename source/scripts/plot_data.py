"""Shared data-loading/joining layer for the RQ1-RQ5 plotting scripts.

Thin wrappers around make_result_tables.py's tidy long-dataframe loader
(model, source, dataset, head, metric, value) -- reused rather than
reimplemented, since it already handles CSV discovery and column parsing
for every archive/{study}/{model}/{eval_results,val_eval_results,
scape_eval_results} folder.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

import make_result_tables as mrt

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = REPO_ROOT / "archive"

STUDY_DIRS = {
    "pooled": ARCHIVE / "Pooled-Embeddings",
    "spatial": ARCHIVE / "Spatial-Embeddings",
    "finetune": ARCHIVE / "Fine-Tuning",
    "xcm_gen": ARCHIVE / "XCM-Generalization",
    "xcm_gen_ft": ARCHIVE / "XCM-Generalization-fine-tune",
}

REGIONAL_DATASETS = ["UHH", "HSN", "PER", "NES", "POW", "SSW", "SNE"]  # non-XCM
ALL_DATASETS = REGIONAL_DATASETS + ["XCM"]


def load_study(key: str, models: list[str] | None = None) -> pd.DataFrame:
    """discover_models (unless overridden) + load_long_df for one study root."""
    target_dir = STUDY_DIRS[key]
    if models is None:
        models = mrt.discover_models(target_dir)
    return mrt.load_long_df(target_dir, models)


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def filter_long(df: pd.DataFrame, *, source, metric, head=None, dataset=None, model=None) -> pd.DataFrame:
    """One-stop boolean filter over a tidy long dataframe. Scalars, lists, or
    None (= any) are accepted for head/dataset/model; source/metric are
    required and may also be a scalar or list."""
    mask = df["source"].isin(_as_list(source)) & df["metric"].isin(_as_list(metric))
    if head is not None:
        mask &= df["head"].isin(_as_list(head))
    if dataset is not None:
        mask &= df["dataset"].isin(_as_list(dataset))
    if model is not None:
        mask &= df["model"].isin(_as_list(model))
    return df[mask]


def mean_sd_n(df: pd.DataFrame, group_cols: list[str], value_col: str = "value") -> pd.DataFrame:
    """groupby(group_cols)[value_col].agg(['mean','std','count']) -- count
    surfaces missing-data gaps (some backbones are missing individual
    dataset columns) so callers can see when n is less than expected rather
    than silently averaging over fewer values."""
    return df.groupby(group_cols)[value_col].agg(["mean", "std", "count"])


def _ci95(values: pd.Series) -> float:
    values = values.dropna()
    n = len(values)
    if n < 2:
        return np.nan
    sem = scipy.stats.sem(values)
    return sem * scipy.stats.t.ppf(0.975, df=n - 1)


def mean_ci95_n(df: pd.DataFrame, group_cols: list[str], value_col: str = "value") -> pd.DataFrame:
    """groupby(group_cols)[value_col].agg(mean, 95% CI half-width, count) --
    used wherever the spread is across repeated measurements of the SAME
    model/config (e.g. across datasets for one backbone), per the shared
    convention that CI/SEM (not SD) is appropriate there."""
    grouped = df.groupby(group_cols)[value_col]
    return pd.DataFrame({
        "mean": grouped.mean(),
        "ci95": grouped.apply(_ci95),
        "count": grouped.count(),
    })


def paired_join(config_df: pd.DataFrame, baseline: pd.Series, on=("model", "dataset")) -> pd.DataFrame:
    """Inner-join config values to a baseline series on `on`, returning a
    dataframe with the original config values, a 'baseline' column, and a
    'delta' column. The inner join is itself the pairing check: rows in
    config_df that have no matching baseline are silently dropped, so
    callers should compare len(result) to len(config_df) and warn if they
    differ (partial-pairing fallback)."""
    config_indexed = config_df.set_index(list(on))
    merged = config_indexed.join(baseline.rename("baseline"), how="inner")
    merged["delta"] = merged["value"] - merged["baseline"]
    if len(merged) != len(config_indexed):
        missing = len(config_indexed) - len(merged)
        print(f"  [warn] paired_join: {missing} row(s) dropped (no matching baseline) -- pairing is not 1:1")
    return merged.reset_index()
