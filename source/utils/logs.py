# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.


"""
This module handles the logging and summary writing for the project.
"""

import os
from pathlib import Path, PosixPath
from typing import Any, Dict, Optional, Union
import datetime
import json
import shutil
import filelock
from collections import Counter


from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib import gridspec
import librosa
from dvclive import Live
#import h5py

import tensorflow as tf

if __name__ == "__main__":
    import config
else:
    from utils import config

def save_to_report(entry: dict, report_path: str = "mix_report.json"):
    lock_path = report_path + ".lock"
    with filelock.FileLock(lock_path):
        report = []
        if os.path.exists(report_path):
            try:
                with open(report_path, "r") as f:
                    report = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {report_path} was corrupted, starting fresh.")
                report = []

        report.append(entry)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)



def plot_polyphony_distribution(ds, save_path=None):
    """
    Plot the distribution of min_polyphony, max_polyphony, and their range in the dataset.
    Args:
        ds: The dataset to plot.
    """
    min_poly = ds["min_polyphony"]
    max_poly = ds["max_polyphony"]
    poly_range = [mx - mn for mn, mx in zip(min_poly, max_poly)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, values, title, col in zip(
        axes,
        [min_poly, max_poly, poly_range],
        ["Min polyphony", "Max polyphony", "Range (max - min)"],
        ["min_polyphony", "max_polyphony", "range"],
    ):
        counts = Counter(values)
        xs = sorted(counts.keys())
        ys = [counts[x] for x in xs]
        ax.bar(xs, ys)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("number of clips")
        ax.set_yscale("log")  # long tail -> log scale keeps small bars visible

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    # plt.show()
    

def plot_confusion_matrix(y_pred, y_true):
    # Convert to flat NumPy arrays
    # y_true =  np.concatenate(y_true, axis=0) #np.array(y_true)
    # y_pred = np.concatenate(y_pred, axis=0) #np.array(y_pred)
    y_pred

    # Get predictions and ground truth as int
    y_true_rounded = y_true.astype(int)
    y_pred_rounded = np.round(y_pred).astype(int).flatten()

    # Get all unique polyphony degrees in true total counts
    unique_total_classes = np.unique(np.concatenate([y_true_rounded, y_pred_rounded]))

    # Compute confusion matrix for total polyphony degrees
    cm_total = confusion_matrix(y_true_rounded, y_pred_rounded, labels=unique_total_classes)

    # Calculate percentages per true class (row-wise)
    with np.errstate(all='ignore'):
        cm_total_percent = cm_total / cm_total.sum(axis=1, keepdims=True) * 100

    # Create annotation strings (count + percentage)
    annot_total = np.empty_like(cm_total).astype(str)
    for r in range(cm_total.shape[0]):
        for c in range(cm_total.shape[1]):
            count = cm_total[r, c]
            pct = cm_total_percent[r, c]
            annot_total[r, c] = f"{count}\n({pct:.1f}%)"

    # Plot heatmap for total polyphony degree
    figure = plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm_total, annot=annot_total, fmt='', cmap='Blues', cbar=True,
                    xticklabels=unique_total_classes,
                    yticklabels=unique_total_classes,
                    annot_kws={"fontsize": 10})

    # Highlight diagonal cells with a red rectangle
    for i in range(len(unique_total_classes)):
        ax.add_patch(patches.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=3))

    plt.title("Confusion Matrix for Polyphony Degree")
    plt.xlabel("Predicted Polyphony Degree")
    plt.ylabel("True Polyphony Degree")
    plt.tight_layout()
    return figure

def prepare_classification_for_cm(y_true, y_pred):
    """
    Prepare multi-class classification logits for confusion matrix.
    
    Args:
        y_true: integer class labels, shape (N,)
        y_pred: raw logits, shape (N, num_classes)
    
    Returns:
        yt: integer labels as numpy array
        yp: predicted class indices as numpy array
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Convert logits to predicted class index
    yp = np.argmax(y_pred, axis=-1)
    yt = y_true.astype(int)
    
    return yt, yp

# def prepare_polyphony_for_cm(y_true, y_pred):
#     """
#     y_true: (N,) or (N, 1)
#     y_pred: (N, 1)
#     """
#     y_true = np.asarray(y_true).squeeze().astype(int)
#     y_pred = np.asarray(y_pred).squeeze()

#     y_pred =np.round(y_pred).astype(int)

#     return y_true, y_pred

def prepare_polyphony_for_cm(y_true, y_pred):
    """
    y_true: (N,) or (N, 1)
    y_pred: (N,) or (N, 1)
    """
    y_true = np.atleast_1d(np.asarray(y_true).squeeze()).astype(int)
    y_pred = np.atleast_1d(np.round(np.asarray(y_pred).squeeze())).astype(int)
    return y_true, y_pred

def prepare_event_logits_for_cm(y_true, y_pred_logits, threshold=0.5):
    """
    y_true: (N, T)
    y_pred_logits: (N, T)
    """
    y_true = np.asarray(y_true).astype(int)

    # sigmoid
    y_pred_probs = 1 / (1 + np.exp(-y_pred_logits))
    y_pred = (y_pred_probs >= threshold).astype(int)

    # flatten time
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    return y_true_flat, y_pred_flat

def plot_confusion_matrix_sklearn(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    with np.errstate(all="ignore"):
        cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    return fig

##############################
# Custom Metrics
#############################

class RegressionAccuracy(tf.keras.metrics.Metric):
    """Accuracy after rounding predictions to nearest integer (for regression polyphony)."""
    def __init__(self, name="rounded_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total   = self.add_weight(name="total",   initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_rounded = tf.round(tf.reshape(y_pred, [-1]))
        y_true_flat = tf.cast(tf.reshape(y_true, [-1]), y_pred_rounded.dtype)
        matches = tf.cast(tf.equal(y_pred_rounded, y_true_flat), tf.float32)
        self.correct.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(tf.cast(tf.size(matches), tf.float32))
        
    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

class RegressionPrecision(tf.keras.metrics.Metric):
    def __init__(self, name="precision", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_rounded = tf.cast(tf.round(tf.maximum(y_pred, 0)), tf.float32)
        y_true_flat    = tf.cast(y_true, tf.float32)
        pred_present   = y_pred_rounded > 0
        true_present   = y_true_flat > 0
        self.tp.assign_add(tf.reduce_sum(tf.cast(pred_present & true_present, tf.float32)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(pred_present & ~true_present, tf.float32)))

    def result(self):
        return tf.math.divide_no_nan(self.tp, self.tp + self.fp)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)

class RegressionRecall(tf.keras.metrics.Metric):
    def __init__(self, name="recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_rounded = tf.cast(tf.round(tf.maximum(y_pred, 0)), tf.float32)
        y_true_flat    = tf.cast(y_true, tf.float32)
        pred_present   = y_pred_rounded > 0
        true_present   = y_true_flat > 0
        self.tp.assign_add(tf.reduce_sum(tf.cast(pred_present & true_present, tf.float32)))
        self.fn.assign_add(tf.reduce_sum(tf.cast(~pred_present & true_present, tf.float32)))

    def result(self):
        return tf.math.divide_no_nan(self.tp, self.tp + self.fn)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fn.assign(0.0)
class RegressionF1(tf.keras.metrics.Metric):
    def __init__(self, name="f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_rounded = tf.cast(tf.round(tf.maximum(y_pred, 0)), tf.float32)
        y_true_flat    = tf.cast(y_true, tf.float32)
        pred_present   = y_pred_rounded > 0
        true_present   = y_true_flat > 0
        self.tp.assign_add(tf.reduce_sum(tf.cast(pred_present & true_present, tf.float32)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(pred_present & ~true_present, tf.float32)))
        self.fn.assign_add(tf.reduce_sum(tf.cast(~pred_present & true_present, tf.float32)))

    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        recall    = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        return tf.math.divide_no_nan(2 * precision * recall, precision + recall)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

class ClassificationAccuracy(tf.keras.metrics.Metric):
    """Exact count match after argmax (per species slot)."""
    def __init__(self, name="accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total   = self.add_weight(name="total",   initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_pred: [B, num_species, num_classes]
        # y_true: [B, num_species]
        y_pred_class = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)  # [B, num_species]
        y_true_flat  = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred_flat  = tf.reshape(y_pred_class, [-1])
        matches = tf.cast(tf.equal(y_pred_flat, y_true_flat), tf.float32)
        self.correct.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(tf.cast(tf.size(matches), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

class ClassificationPrecision(tf.keras.metrics.Metric):
    """Of species predicted present (argmax > 0), how many are truly present."""
    def __init__(self, name="precision", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_class  = tf.argmax(y_pred, axis=-1)          # [B, num_species]
        pred_present  = y_pred_class > 0
        true_present  = tf.cast(y_true, tf.int64) > 0
        self.tp.assign_add(tf.reduce_sum(tf.cast(pred_present & true_present, tf.float32)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(pred_present & ~true_present, tf.float32)))

    def result(self):
        return tf.math.divide_no_nan(self.tp, self.tp + self.fp)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)


class ClassificationRecall(tf.keras.metrics.Metric):
    """Of species truly present, how many are predicted present (argmax > 0)."""
    def __init__(self, name="recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_class  = tf.argmax(y_pred, axis=-1)
        pred_present  = y_pred_class > 0
        true_present  = tf.cast(y_true, tf.int64) > 0
        self.tp.assign_add(tf.reduce_sum(tf.cast(pred_present & true_present, tf.float32)))
        self.fn.assign_add(tf.reduce_sum(tf.cast(~pred_present & true_present, tf.float32)))

    def result(self):
        return tf.math.divide_no_nan(self.tp, self.tp + self.fn)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fn.assign(0.0)


class ClassificationF1(tf.keras.metrics.Metric):
    def __init__(self, name="f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_class  = tf.argmax(y_pred, axis=-1)
        pred_present  = y_pred_class > 0
        true_present  = tf.cast(y_true, tf.int64) > 0
        self.tp.assign_add(tf.reduce_sum(tf.cast(pred_present & true_present, tf.float32)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(pred_present & ~true_present, tf.float32)))
        self.fn.assign_add(tf.reduce_sum(tf.cast(~pred_present & true_present, tf.float32)))

    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        recall    = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        return tf.math.divide_no_nan(2 * precision * recall, precision + recall)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

class _MacroCountStats(tf.keras.metrics.Metric):
    """
    Shared TP/FP/FN accumulation per count-class for macro precision/recall/F1
    on exact-count correctness. Subclasses provide _get_pred_classes().
    num_classes = max_polyphony + 1 (counts 0..max_polyphony).
    """
    def __init__(self, num_classes, name="macro_count_stats", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros")

    def _get_pred_classes(self, y_pred):
        raise NotImplementedError

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_class = self._get_pred_classes(y_pred)                     # int, any shape
        y_true_class = tf.cast(tf.round(y_true), tf.int32)                # exact count label
        y_pred_class = tf.reshape(tf.cast(y_pred_class, tf.int32), [-1])
        y_true_class = tf.reshape(y_true_class, [-1])

        # Clip to valid range so stray predictions don't break one_hot indexing
        y_pred_class = tf.clip_by_value(y_pred_class, 0, self.num_classes - 1)
        y_true_class = tf.clip_by_value(y_true_class, 0, self.num_classes - 1)

        pred_oh = tf.one_hot(y_pred_class, self.num_classes)   # [N, C]
        true_oh = tf.one_hot(y_true_class, self.num_classes)   # [N, C]

        tp = tf.reduce_sum(pred_oh * true_oh, axis=0)
        fp = tf.reduce_sum(pred_oh * (1 - true_oh), axis=0)
        fn = tf.reduce_sum((1 - pred_oh) * true_oh, axis=0)

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def reset_state(self):
        self.tp.assign(tf.zeros((self.num_classes,)))
        self.fp.assign(tf.zeros((self.num_classes,)))
        self.fn.assign(tf.zeros((self.num_classes,)))


class _MacroCountStatsRegression(_MacroCountStats):
    def _get_pred_classes(self, y_pred):
        return tf.round(tf.maximum(y_pred, 0))


class _MacroCountStatsClassification(_MacroCountStats):
    def _get_pred_classes(self, y_pred):
        return tf.argmax(y_pred, axis=-1)


class RegressionCountPrecision(_MacroCountStatsRegression):
    def __init__(self, num_classes, name="precision", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def result(self):
        per_class = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        return tf.reduce_mean(per_class)  # macro avg


class RegressionCountRecall(_MacroCountStatsRegression):
    def __init__(self, num_classes, name="recall", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def result(self):
        per_class = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        return tf.reduce_mean(per_class)


class RegressionCountF1(_MacroCountStatsRegression):
    def __init__(self, num_classes, name="f1", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        recall    = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        f1_per_class = tf.math.divide_no_nan(2 * precision * recall, precision + recall)
        return tf.reduce_mean(f1_per_class)


class ClassificationCountPrecision(_MacroCountStatsClassification):
    def __init__(self, num_classes, name="precision", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def result(self):
        per_class = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        return tf.reduce_mean(per_class)


class ClassificationCountRecall(_MacroCountStatsClassification):
    def __init__(self, num_classes, name="recall", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def result(self):
        per_class = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        return tf.reduce_mean(per_class)


class ClassificationCountF1(_MacroCountStatsClassification):
    def __init__(self, num_classes, name="f1", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        recall    = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        f1_per_class = tf.math.divide_no_nan(2 * precision * recall, precision + recall)
        return tf.reduce_mean(f1_per_class)

#########################
# Custom Summary Writer
#########################

class CustomSummaryWriter(SummaryWriter):
    """
    A custom subclass of the TensorBoard SummaryWriter that allows for logging hyperparameters,
    displaying scalar metrics in the HParams tab, and automatically synchronizing logs with a remote directory.

    Args:
        log_dir (Union[str, PosixPath]): Directory where the TensorBoard logs will be stored.
        params (Optional[config.Params[str, Any]]): config.Params object of DVC hyperparameters to display. Defaults to None.
        metrics (Optional[Dict[str, None]]): Dictionary of initial metrics to display in the HParams tab. Defaults to {}.
        sync_interval (Optional[int]): Number of steps between automatic syncs to the remote directory.
                                       Defaults to the value of the 'SYNC_INTERVAL' environment variable.
                                       If set to 0, no automatic syncs will be performed.
        remote_dir (Optional[Union[str, PosixPath]]): Remote directory with format 'host:dir' to which logs are synced.
                                Defaults to None, in which case the remote directory is constructed from environment variables.
    """

    def __init__(
        self,
        log_dir: Union[str, PosixPath],
        params: Optional[config.Params[str, Any]] = None,
        metrics: Optional[Dict[str, None]] = {},
        sync_interval: Optional[int] = None,
        remote_dir: Optional[Union[str, PosixPath]] = None
    ):
        super().__init__(log_dir=log_dir)

        self.sync_interval = (
            sync_interval
            if sync_interval is not None
            else int(config.get_env_variable("SYNC_INTERVAL"))
        )
        self.remote_dir = (
            remote_dir or self._construct_remote_dir()
            if self.sync_interval != 0
            else None
        )
        self.datetime = self._extract_datetime_from_log_dir(log_dir)

        self.params = params
        self.metrics = metrics

        if params:
            self._log_hyperparameters()

        self.current_step = 0

    def _construct_remote_dir(self) -> str:
        """Constructs the remote directory path based on environment variables."""
        tensorboard_host_dir = config.get_env_variable("TENSORBOARD_HOST_DIR")
        tensorboard_host = config.get_env_variable("TENSORBOARD_HOST")
        tensorboard_host_savepath = Path(
            f'{tensorboard_host_dir}/{config.get_env_variable("PROJECT_NAME")}/logs/tensorboard'
        )
        os.system(f"ssh {tensorboard_host} 'mkdir -p {tensorboard_host_savepath}'")
        return f"{tensorboard_host}:{tensorboard_host_savepath}"

    def _extract_datetime_from_log_dir(self, log_dir: Union[str, PosixPath]) -> str:
        """Extracts datetime information from the log directory path."""
        return str(log_dir).split("/")[-1].split("_")[0]

    # def _log_hyperparameters(
    #     self,
    #     params: config.Params[str, Any],
    #     metrics: Dict[str, None],
    #     log_dir: str,
    # ) -> None:
    #     """Logs hyperparameters and initial metrics to TensorBoard."""
    #     clean_params = params.tensorboard_compatible_copy()
    #     clean_params["datetime"] = self.datetime
    #     # params = params.flattened_copy()
    #     #cparams["datetime"] = self.datetime
    #     self._add_hparams(hparam_dict=clean_params, metric_dict=metrics, run_name=log_dir)

    def _log_hyperparameters(self) -> None:
        """Logs hyperparameters and initial metrics to TensorBoard."""
        clean_params = self.params.tensorboard_compatible_copy()
        clean_params["datetime"] = self.datetime
        best_metrics = {f"best/{k}": v for k, v in self.metrics.items()}
        self._add_hparams(hparam_dict=clean_params, metric_dict=best_metrics, run_name=self.log_dir)

    def step(self) -> None:
        """
        Increments the current step and triggers log synchronization if the sync interval is reached.
        """
        self.current_step += 1
        if self.sync_interval != 0:
            if self.current_step % self.sync_interval == 0:
                self.flush()
                self._sync_logs()

    def _sync_logs(self) -> None:
        """Synchronizes the logs with the remote directory."""
        # path = f'mkdir -p {self.remote_dir} && rsync'
        os.system(f"rsync -rv --inplace --progress {self.log_dir} {self.remote_dir}")

    def _add_hparams(
        self,
        hparam_dict: Dict[str, Any],
        metric_dict: Dict[str, Optional[float]],
        hparam_domain_discrete: Optional[Dict[str, list]] = None,
        run_name: Optional[str] = None,
    ) -> None:
        """
        Adds hyperparameters and metrics to the same TensorBoard log file and enables scalar metrics in the HParams tab.

        Args:
            hparam_dict (Dict[str, float]): Dictionary of hyperparameters.
            metric_dict (Dict[str, Optional[float]]): Dictionary of metrics.
            hparam_domain_discrete (Optional[Dict[str, list]]): Discrete domains for hyperparameters.
            run_name (Optional[str]): Name of the run in TensorBoard.

        Raises:
            TypeError: If `hparam_dict` or `metric_dict` are not dictionaries.
        """
        if not isinstance(hparam_dict, dict) or not isinstance(metric_dict, dict):
            raise TypeError("hparam_dict and metric_dict should be dictionary.")

        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            if v is not None:
                self.add_scalar(k, v)

# def get_confusion_matrix_specs(cfg):
#     """Get confusion matrix specs from config."""
#     if hasattr(cfg.metrics, 'confusion_matrix_specs'):
#         return cfg.metrics.confusion_matrix_specs
#     return []

def build_confusion_matrix_specs(objectives):
    """
    Build confusion matrix specs from objectives config.
    
    Args:
        objectives: OmegaConf dict of objectives
        
    Returns:
        List of confusion matrix spec dicts
    """
    specs = []
    
    for obj_name, obj_config in objectives.items():
        if "confusion_matrix" in obj_config:
            spec = {
                "name": obj_name, 
                **obj_config["confusion_matrix"]
            }
            if spec['type']=='classification':
                try:
                    spec['num_classes'] = obj_config["num_classes"]
                except:
                    raise Exception("Number of classes for classification task is undefined.")
            specs.append(spec)
    
    return specs

# class CustomSummaryWriterCallback(tf.keras.callbacks.Callback):
#     """
#     Custom callback that integrates with your CustomSummaryWriter
#     Focuses on custom metrics and syncing, while standard TensorBoard handles built-in features
#     """
#     def __init__(self, writer, include_standard_tensorboard=True, val_dataset=None,
#                 log_confusion_matrix=True, confusion_matrix_frequency=5,
#                 confusion_matrix_specs=None, input_shape=None, cfg=None,
#                 loss_objects={}, previous_history=None, use_dvclive = True,
#                 # dvclive_tracked_val_metrices=["val_loss"]
#                 ):
#         super().__init__()
#         self.writer = writer
#         self.val_dataset = val_dataset
#         #self.val_results_save_path = val_results_save_path
#         self.log_confusion_matrix = log_confusion_matrix
#         self.confusion_matrix_frequency = confusion_matrix_frequency
#         self.confusion_matrix_specs = confusion_matrix_specs or []
#         self.metrics = {}
#         self.input_shape = input_shape
#         self.cfg = cfg
#         self.loss_objects = loss_objects
#         self.previous_history = previous_history

#         # self.live = Live(dir=self.writer.log_dir / "dvclive", dvcyaml=False) if use_dvclive else None
#         self.live = Live(dvcyaml=False) if use_dvclive else None
#         # self.metrics = self.writer.metrics #dvclive_tracked_val_metrices

#         self._best_val = {}
#         self._best_step = {}

#          # Optionally create standard TensorBoard callback
#         self.standard_tb_callback = None
#         if include_standard_tensorboard:
#             # Create a standard TensorBoard callback that logs to the same directory
#             self.standard_tb_callback = tf.keras.callbacks.TensorBoard(
#                 log_dir=str(writer.log_dir),
#                 histogram_freq=1,  # Log histograms every epoch
#                 write_graph=True,  # Log the model graph
#                 write_images=False,
#                 update_freq='epoch',
#                 profile_batch=0,  # Disable profiling by default
#                 embeddings_freq=0
#             )

#         # Create separate file writers for train and validation
#         self.train_writer = tf.summary.create_file_writer(
#             str(Path(writer.log_dir) / 'train')
#         )
#         self.val_writer = tf.summary.create_file_writer(
#             str(Path(writer.log_dir) / 'validation')
#         )

#         # Replay previous history
#         if previous_history:
#             self._replay_history_to_tensorboard(previous_history)
#             self.train_writer.flush()
#             self.val_writer.flush()
class CustomSummaryWriterCallback(tf.keras.callbacks.Callback):

    HIGHER_IS_BETTER = {"accuracy", "f1", "auc"}

    def __init__(self, writer, include_standard_tensorboard=False, val_dataset=None,
                log_confusion_matrix=True, confusion_matrix_frequency=5,
                confusion_matrix_specs=None, input_shape=None, cfg=None,
                loss_objects={}, previous_history=None, use_dvclive=True):
        super().__init__()
        self.writer = writer
        self.val_dataset = val_dataset
        self.log_confusion_matrix = log_confusion_matrix
        self.confusion_matrix_frequency = confusion_matrix_frequency
        self.confusion_matrix_specs = confusion_matrix_specs or []
        self.input_shape = input_shape
        self.cfg = cfg
        self.loss_objects = loss_objects
        self.previous_history = previous_history
        self.live = Live(dvcyaml=False) if use_dvclive else None
        self._best_val = {}
        self._best_step = {}
        self._last_logs = {}

        # Optionally create standard TensorBoard callback
        self.standard_tb_callback = None
        if include_standard_tensorboard:
            # Create a standard TensorBoard callback that logs to the same directory
            self.standard_tb_callback = tf.keras.callbacks.TensorBoard(
                log_dir=str(writer.log_dir),
                histogram_freq=1,  # Log histograms every epoch
                write_graph=True,  # Log the model graph
                write_images=False,
                update_freq='epoch',
                profile_batch=0,  # Disable profiling by default
                embeddings_freq=0
            )

        # Train/val subdirectory writers
        self.train_writer = tf.summary.create_file_writer(str(Path(writer.log_dir) / 'train'))
        self.val_writer = tf.summary.create_file_writer(str(Path(writer.log_dir) / 'validation'))

        if previous_history:
            self._replay_history_to_tensorboard(previous_history)
            self._init_best_from_history(previous_history)  # ← initialize _best_val from history
            self.train_writer.flush()
            self.val_writer.flush()

    def _init_best_from_history(self, previous_history):
        """Seed _best_val/_best_step from replayed history so resumed training continues correctly."""
        num_epochs = len(next(iter(previous_history.values())))
        for epoch in range(num_epochs):
            logs = {key: values[epoch] for key, values in previous_history.items()}
            self._update_best(epoch, logs)

    def _update_best(self, epoch, logs):
        """Core best-tracking logic, shared by on_epoch_end and _init_best_from_history."""
        for metric_key in self.writer.metrics:
            metric = logs.get(metric_key)
            if metric is None:
                continue

            higher_is_better = any(m in metric_key for m in self.HIGHER_IS_BETTER)
            is_best = (
                self._best_val.get(metric_key) is None or
                (higher_is_better and metric > self._best_val[metric_key]) or
                (not higher_is_better and metric < self._best_val[metric_key])
            )
            if is_best:
                self._best_val[metric_key] = metric
                self._best_step[metric_key] = epoch
                if self.live:
                    self.live.summary[f"{metric_key}_best"] = float(metric)
                    self.live.summary[f"{metric_key}_best_step"] = epoch

            # Staircase curve in TensorBoard — every epoch, only if we have a best value
            if self._best_val.get(metric_key) is not None:
                self.writer.add_scalar(f'best/{metric_key}_epoch', self._best_step[metric_key], epoch) if self._best_step.get(metric_key) is not None else None
                self.writer.add_scalar(f'best/{metric_key}', self._best_val[metric_key], epoch)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        # 1. Log unweighted per-objective losses
        self._log_losses(epoch, logs)

        # 2. Confusion matrix
        if (self.val_dataset is not None
            and self.confusion_matrix_specs
            and (epoch + 1) % self.confusion_matrix_frequency == 0
        ):
            for spec in self.confusion_matrix_specs:
                self._log_confusion_matrix(epoch, spec)
                self._log_f1_breakdown(epoch, spec)   

        # 3. DVCLive — log all metrics every epoch
        if self.live:
            for key, value in logs.items():
                self.live.log_metric(key, value)
            self.live.next_step()

        # Log all Keras metrics continuously to val_writer
        with self.val_writer.as_default():
            for key, value in logs.items():
                if key.startswith('val_'):
                    tf.summary.scalar(key.removeprefix('val_'), value, step=epoch)

        with self.train_writer.as_default():
            for key, value in logs.items():
                if not key.startswith('val_'):
                    tf.summary.scalar(key, value, step=epoch)

        # 4. Best tracking → TensorBoard staircase + DVCLive summary
        self._update_best(epoch, logs)

        if self.live:
            self.live.make_summary()

        # 5. Sync
        self.writer.step()
        self.train_writer.flush()
        self.val_writer.flush()

        self._last_logs = logs

    def on_train_end(self, logs=None):
        if self.log_confusion_matrix and self.val_dataset is not None:
            for spec in self.confusion_matrix_specs:
                self._log_confusion_matrix(epoch=-1, spec=spec)

        if self.live:
            self.live.end()

        # Write best values to HParams
        self.writer.metrics = {
            k: float(self._best_val[k]) if k in self._best_val else None
            for k in self.writer.metrics
        }
        self.writer._log_hyperparameters()
        self.writer.close()
        print("Training completed!")

    def set_model(self, model):
        """Called when the callback is attached to a model"""
        super().set_model(model)
        if self.standard_tb_callback:
            self.standard_tb_callback.set_model(model)

        # Log the model graph once
        self._log_model_graph(model)

    def _replay_history_to_tensorboard(self, previous_history):
        """Replay previous training history into TensorBoard so graphs are continuous."""
        if not previous_history:
            return

        num_objectives = len(self.loss_objects)
        num_epochs = len(next(iter(previous_history.values())))

        print(f"Replaying {num_epochs} epochs of history to TensorBoard...")

        for epoch in range(num_epochs):
            # Reconstruct logs dict for this epoch
            logs = {key: values[epoch] for key, values in previous_history.items()}

            # --- Replay loss weights and per-objective losses ---
            for obj_name, loss_obj in self.loss_objects.items():
                weight_key = f'loss_weight/{obj_name}'

                # Use stored weight if available, else fall back to current
                current_weight = (
                    previous_history[weight_key][epoch]
                    if weight_key in previous_history
                    else float(loss_obj.weight.numpy())
                )

                # Log weight to main writer
                self.writer.add_scalar(
                    f'loss_weights/{obj_name}',
                    current_weight,
                    epoch
                )

                # Replay train loss
                weighted_loss = self._get_loss_from_logs(logs, obj_name, '', num_objectives)
                if weighted_loss is not None:
                    base_loss = self._calculate_base_loss(weighted_loss, current_weight)
                    with self.train_writer.as_default():
                        tf.summary.scalar(f'{obj_name}_loss', base_loss, step=epoch)

                # Replay validation loss
                val_weighted_loss = self._get_loss_from_logs(logs, obj_name, 'val_', num_objectives)
                if val_weighted_loss is not None:
                    val_base_loss = self._calculate_base_loss(val_weighted_loss, current_weight)
                    with self.val_writer.as_default():
                        tf.summary.scalar(f'{obj_name}_loss', val_base_loss, step=epoch)

            # --- Replay all other scalars (total loss, metrics, lr, etc.) ---
            with self.train_writer.as_default():
                for key, values in previous_history.items():
                    if not key.startswith('val_') and not key.startswith('loss_weight/'):
                        tf.summary.scalar(key, values[epoch], step=epoch)

            with self.val_writer.as_default():
                for key, values in previous_history.items():
                    if key.startswith('val_') and not key.startswith('loss_weight/'):
                        tf.summary.scalar(key.removeprefix('val_'), values[epoch], step=epoch)

        self.train_writer.flush()
        self.val_writer.flush()
        print(f"✓ Replayed {num_epochs} epochs of history to TensorBoard")

    def _log_model_graph(self, model):
        """Log the model computational graph to TensorBoard"""
        try:
            print("Logging model graph to TensorBoard...")
            
            # Start tracing - only graph, no profiler for simplicity
            tf.summary.trace_on(graph=True, profiler=False)
            
            # # Create dummy input and run forward pass to build the graph
            # if model.input_shape:
            #     dummy_input = tf.zeros((1,) + tuple(self.input_shape))
            # elif self.input_shape:
            #     dummy_input = tf.zeros((1,) + tuple(self.input_shape))
            # else:
            #     raise Exception('Input shape was not found.')
            
            dummy_input = tf.zeros((1,) + tuple(self.input_shape))
            
            model(dummy_input)

            log_dir_str = str(self.writer.log_dir)
            
            # Use tf.summary.create_file_writer instead
            with tf.summary.create_file_writer(log_dir_str).as_default():
                tf.summary.trace_export(
                    name="model_trace",
                    step=0,
                    profiler_outdir=log_dir_str,
                )
            tf.summary.trace_off()
                
            # # Execute forward pass to capture the graph
            # _ = model(dummy_input, training=False)
            
            # # Export the traced graph
            # with self.writer.file_writer.as_default():
            #     tf.summary.trace_export(
            #         name="model_graph",
            #         step=0,
            #         profiler_outdir=None  # No profiler output needed
            #     )
            
            # Clean up tracing
            tf.summary.trace_off()
            print("Model graph successfully logged to TensorBoard.")
            
        except Exception as e:
            print(f"Failed to log model graph: {e}")
            # Ensure tracing is turned off even if there's an error
            try:
                tf.summary.trace_off()
            except:
                pass

    def on_train_begin(self, logs=None):
        print("Training started with CustomSummaryWriter logging")

        # if self.previous_history:
        #     self._replay_history_to_tensorboard(self.previous_history)

        if self.standard_tb_callback:
            self.standard_tb_callback.on_train_begin(logs)

    def on_epoch_begin(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        if self.standard_tb_callback:
            self.standard_tb_callback.on_epoch_begin(epoch, logs)

    # def on_epoch_end(self, epoch, logs=None):
    #     if logs is None:
    #         return

    #     # # --- Run inference ONCE, cache for all consumers ---
    #     # val_results = self._run_inference()
    #     # y_pred, y_true = 

    #     # --- Consumers read from cache ---
    #     # 1. Loss logging (already comes from `logs`, no inference needed)
    #     self._log_losses(epoch, logs)

    #     # 2. Confusion matrix
    #     if (self.val_dataset is not None
    #         and self.confusion_matrix_specs
    #         and (epoch + 1) % self.confusion_matrix_frequency == 0
    #     ):
    #         for spec in self.confusion_matrix_specs:
    #             self._log_confusion_matrix(epoch, spec)

    #     # # 3. Save raw results
    #     # if self.val_dataset is not None:
    #     #     for target, (y_pred, y_true) in self._epoch_cache.items():
    #     #         self._save_val_results(y_pred, y_true, target, self.val_results_save_path, epoch)

    #     # Call standard TensorBoard callback
    #     if self.standard_tb_callback:
    #         self.standard_tb_callback.on_epoch_end(epoch, logs)

    #     # DVCLive logging
    #     if self.live:
    #         for key, value in logs.items():
    #             self.live.log_metric(key, value)
    #         self.live.next_step()
        
    #     # Manual best tracking
    #     # Metrics where higher is better
    #     HIGHER_IS_BETTER = {"accuracy", "f1", "auc"}

    #     for metric_key in self.writer.metrics:
    #         metric = logs.get(metric_key)
    #         if metric is not None:
    #             higher_is_better = any(m in metric_key for m in HIGHER_IS_BETTER)
    #             is_best = (
    #                 self._best_val.get(metric_key) is None or
    #                 (higher_is_better and metric > self._best_val[metric_key]) or
    #                 (not higher_is_better and metric < self._best_val[metric_key])
    #             )
    #             if is_best:
    #                 # Update best value and step
    #                 self._best_val[metric_key] = metric
    #                 self._best_step[metric_key] = epoch

    #                 # Log best metric to DVCLive summary
    #                 self.live.summary[f"{metric_key}_best"] = float(metric)
    #                 self.live.summary[f"{metric_key}_best_step"] = epoch

    #         # Log best to TensorBoard every epoch
    #         self.writer.add_scalar(
    #             f'best/{metric_key}',
    #             metric,
    #             epoch)

    #     self.live.make_summary()  # once after the loop

    #     # Step the writer (handles syncing)
    #     self.writer.step()
        
    #     # Flush all writers
    #     self.train_writer.flush()
    #     self.val_writer.flush()

    #     self._last_logs = logs

    def _log_losses(self, epoch, logs):
        num_objectives = len(self.loss_objects)

        for obj_name, loss_obj in self.loss_objects.items():

            # Get current weight
            current_weight = float(loss_obj.weight.numpy())
            
            # Log weights to main directory (using main writer)
            self.writer.add_scalar(
                f'loss_weights/{obj_name}',
                current_weight,
                epoch
            )
            
            # Process train losses
            weighted_loss = self._get_loss_from_logs(logs, obj_name, '', num_objectives)
            if weighted_loss is not None:
                base_loss = self._calculate_base_loss(weighted_loss, current_weight)
                
                # Log to train directory
                with self.train_writer.as_default():
                    tf.summary.scalar(f'{obj_name}_loss', base_loss, step=epoch)
            
            # Process validation losses
            val_weighted_loss = self._get_loss_from_logs(logs, obj_name, 'val_', num_objectives)
            if val_weighted_loss is not None:
                val_base_loss = self._calculate_base_loss(val_weighted_loss, current_weight)
                
                # Log to validation directory
                with self.val_writer.as_default():
                    tf.summary.scalar(f'{obj_name}_loss', val_base_loss, step=epoch)

    def _run_inference(self):
            self._epoch_cache = {}
            if self.val_dataset is not None:
                for spec in self.confusion_matrix_specs:
                    target = spec['name']
                    if target not in self._epoch_cache:
                        return self._get_predictions_and_true_labels(
                            self.val_dataset, target
                        )
    
    def _get_loss_from_logs(self, logs, obj_name, prefix, num_objectives):
        """Extract loss value from logs."""
        if num_objectives == 1:
            # Single objective: loss is logged as 'loss' or 'val_loss'
            key = f'{prefix}loss'
            return logs.get(key)
        else:
            # Multi-objective: loss is logged with output name
            possible_keys = [
                f'{prefix}{obj_name}',
                f'{prefix}{obj_name}_loss',
            ]
            for key in possible_keys:
                if key in logs:
                    return logs[key]
        return None
    
    def _calculate_base_loss(self, weighted_loss, weight):
        """Calculate base loss (as if weight=1.0)."""
        if weight > 1e-8:
            return weighted_loss / weight
        else:
            return weighted_loss
 
    def _log_confusion_matrix(self, epoch, spec):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            import io
            from PIL import Image

            target = spec["name"]

            y_pred, y_true = self._get_predictions_and_true_labels(
                self.val_dataset, target
            )
            cm_type = spec["type"]
            threshold = spec.get("threshold", 0.5)

            if cm_type == "regression_round":
                yt, yp = prepare_polyphony_for_cm(y_true, y_pred)
                labels = np.unique(yt)
                title = "Polyphony Degree"
                self._log_single_confusion_matrix(epoch, target, yt, yp, labels, title, spec)

            elif cm_type == "binary":
                yt, yp = prepare_event_logits_for_cm(y_true, y_pred, threshold=threshold)
                labels = [0, 1]
                title = "Event Detection"
                self._log_single_confusion_matrix(epoch, target, yt, yp, labels, title, spec)

            elif cm_type == "classification":
                yt, yp = prepare_classification_for_cm(y_true, y_pred)
                labels = list(range(spec['num_classes']))
                title = "Polyphony Degree Class"
                self._log_single_confusion_matrix(epoch, target, yt, yp, labels, title, spec)

            elif cm_type == "species_regression_round":
                # y_true, y_pred shape: (num_samples, num_species)
                # 1. Aggregated CM across all species (flatten)
                yt_flat = y_true.flatten()
                yp_flat = y_pred.flatten()
                yt_agg, yp_agg = prepare_polyphony_for_cm(yt_flat, yp_flat)
                labels_agg = np.unique(yt_agg)
                self._log_single_confusion_matrix(
                    epoch, target, yt_agg, yp_agg, labels_agg,
                    "Species Polyphony Regression (Aggregated)", spec,
                    tb_tag=f"Confusion_Matrix/{target}/aggregated"
                )
                # 2. Per-species CMs as a grid figure
                species_mapping = spec.get("species_mapping", None)
                self._log_species_confusion_matrix_grid(
                    epoch, target, y_true, y_pred,
                    cm_type="regression_round",
                    num_classes=None,
                    species_mapping=species_mapping,
                    spec=spec
                )

            elif cm_type == "species_classification":
                # y_true shape: (num_samples, num_species) — integer class per species
                # y_pred shape: (num_samples, num_species, num_classes) — logits per species
                num_classes = spec.get("num_classes", None)
                # 1. Aggregated CM across all species
                num_species = y_true.shape[1]
                yt_parts, yp_parts = [], []
                for s in range(num_species):
                    yt_s, yp_s = prepare_classification_for_cm(y_true[:, s], y_pred[:, s, :])
                    yt_parts.append(yt_s)
                    yp_parts.append(yp_s)
                yt_agg = np.concatenate(yt_parts)
                yp_agg = np.concatenate(yp_parts)
                labels_agg = list(range(num_classes)) if num_classes else sorted(np.unique(yt_agg).tolist())
                self._log_single_confusion_matrix(
                    epoch, target, yt_agg, yp_agg, labels_agg,
                    "Species Polyphony Classification (Aggregated)", spec,
                    tb_tag=f"Confusion_Matrix/{target}/aggregated"
                )
                # 2. Per-species grid
                species_mapping = spec.get("species_mapping", None)
                self._log_species_confusion_matrix_grid(
                    epoch, target, y_true, y_pred,
                    cm_type="classification",
                    num_classes=num_classes,
                    species_mapping=species_mapping,
                    spec=spec
                )
            else:
                raise ValueError(f"Unknown confusion matrix type: {cm_type}")

        except Exception as e:
            print(f"Failed to log confusion matrix for '{spec['name']}': {e}")


    def _build_metadata_text(self, epoch):
        """Reusable metadata block for confusion matrix figures."""
        from omegaconf import DictConfig
        lines = [
            f"Model: {self.cfg.model._target_ if hasattr(self.cfg.model, '_target_') else self.cfg.model.get('name', 'N/A')}",
            f"Dataset config: {self.cfg.dataset.config if hasattr(self.cfg.dataset, 'config') else 'N/A'}",
            f"Input Feature: {self.cfg.train.get('input_feature_name', 'N/A')}",
            f"Epoch: {epoch + 1}",
        ]
        if hasattr(self.cfg.log, 'hyperparameters') and self.cfg.log.hyperparameters:
            lines.append("\nHyperparameters:")
            for hp_key in self.cfg.log.hyperparameters:
                value = self.cfg
                for key_part in hp_key.split('.'):
                    value = getattr(value, key_part, 'N/A')
                if isinstance(value, (dict, DictConfig)):
                    lines.append(f"  {hp_key}: {', '.join(value.keys())}")
                else:
                    lines.append(f"  {hp_key}: {value}")
        return "\n".join(lines)


    def _log_single_confusion_matrix(self, epoch, target, yt, yp, labels, title, spec, tb_tag=None):
        """Log a single confusion matrix figure to TensorBoard."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import io
        from PIL import Image

        if tb_tag is None:
            tb_tag = f"Confusion_Matrix/{target}"

        cm_fig = plot_confusion_matrix_sklearn(yt, yp, labels=labels, title=title)

        buf = io.BytesIO()
        cm_fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        cm_image = Image.open(buf)
        plt.close(cm_fig)

        metadata_text = self._build_metadata_text(epoch)
        num_lines = len(metadata_text.splitlines())
        metadata_height_ratio = max(0.15, num_lines * 0.02)

        cm_width = cm_image.width / 100
        cm_height = cm_image.height / 100
        metadata_height = cm_height * metadata_height_ratio

        combined_fig = plt.figure(figsize=(cm_width, cm_height + metadata_height))
        gs = GridSpec(2, 1, figure=combined_fig,
                    height_ratios=[cm_height, metadata_height],
                    hspace=0.15)

        ax_cm = combined_fig.add_subplot(gs[0])
        ax_cm.imshow(cm_image)
        ax_cm.axis('off')

        ax_meta = combined_fig.add_subplot(gs[1])
        ax_meta.axis('off')
        ax_meta.text(
            0.5, 0.5, metadata_text,
            fontsize=8, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            transform=ax_meta.transAxes, family='monospace'
        )
        buf.close()

        self.writer.add_figure(tb_tag, combined_fig, epoch)
        print(f"Confusion matrix logged for '{target}' at epoch {epoch + 1} -> {tb_tag}")


    def _log_species_confusion_matrix_grid(
        self, epoch, target, y_true, y_pred,
        cm_type, num_classes, species_mapping, spec
    ):
        """
        Plot a grid of per-species confusion matrices as a single TensorBoard figure.

        species_mapping: optional dict {species_idx: (birdset_id, label_str)}
                        from the species_polyphony_mapping.json saved at training time.
        """
        import matplotlib.pyplot as plt

        num_species = y_true.shape[1]
        ncols = min(4, num_species)
        nrows = int(np.ceil(num_species / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 4, nrows * 3.5),
            squeeze=False
        )

        for s in range(num_species):
            row, col = divmod(s, ncols)
            ax = axes[row][col]

            # Resolve species label
            if species_mapping and str(s) in species_mapping:
                _, label_str = species_mapping[str(s)]
                species_label = label_str if label_str else str(s)
            elif species_mapping and s in species_mapping:
                _, label_str = species_mapping[s]
                species_label = label_str if label_str else str(s)
            else:
                species_label = str(s)

            try:
                if cm_type == "regression_round":
                    yt_s, yp_s = prepare_polyphony_for_cm(y_true[:, s], y_pred[:, s])
                    labels_s = sorted(np.unique(np.concatenate([yt_s, yp_s])).tolist())
                elif cm_type == "classification":
                    yt_s, yp_s = prepare_classification_for_cm(y_true[:, s], y_pred[:, s, :])
                    labels_s = list(range(num_classes)) if num_classes else sorted(np.unique(yt_s).tolist())
                else:
                    raise ValueError(f"Unknown cm_type for species grid: {cm_type}")

                # Use sklearn directly to draw into the existing axis
                from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
                cm = confusion_matrix(yt_s, yp_s, labels=labels_s)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_s)
                disp.plot(ax=ax, colorbar=False, xticks_rotation='horizontal')
                ax.set_title(species_label, fontsize=8, pad=3)
                ax.tick_params(axis='both', labelsize=6)
                ax.set_xlabel("Predicted", fontsize=6)
                ax.set_ylabel("True", fontsize=6)
            except Exception as e:
                ax.axis('off')
                ax.text(0.5, 0.5, f"{species_label}\n(error)", ha='center', va='center',
                        fontsize=7, transform=ax.transAxes)

        # Hide unused subplot cells
        for s in range(num_species, nrows * ncols):
            row, col = divmod(s, ncols)
            axes[row][col].axis('off')

        fig.suptitle(
            f"Per-Species Confusion Matrices — Epoch {epoch + 1}",
            fontsize=10, y=1.01
        )
        plt.tight_layout()

        tb_tag = f"Confusion_Matrix/{target}/per_species"
        self.writer.add_figure(tb_tag, fig, epoch)
        plt.close(fig)
        print(f"Per-species confusion matrix grid logged for '{target}' at epoch {epoch + 1} -> {tb_tag}")

    def _log_f1_breakdown(self, epoch, spec):
        """
        Log per-species and per-polyphony-degree F1 breakdowns as bar charts
        to TensorBoard, plus per-species/per-degree scalars for trend tracking.
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import f1_score

            target = spec["name"]
            cm_type = spec["type"]

            if cm_type not in ("species_regression_round", "species_classification"):
                return  # F1 breakdown only applies to species-level objectives

            y_pred, y_true = self._get_predictions_and_true_labels(
                self.val_dataset, target
            )
            species_mapping = spec.get("species_mapping", None)
            num_species = y_true.shape[1]

            # --- Prepare per-species (yt, yp) pairs, in integer-class space ---
            per_species_yt = []
            per_species_yp = []
            if cm_type == "species_regression_round":
                for s in range(num_species):
                    yt_s, yp_s = prepare_polyphony_for_cm(y_true[:, s], y_pred[:, s])
                    per_species_yt.append(yt_s)
                    per_species_yp.append(yp_s)
                all_labels = sorted(np.unique(np.concatenate(per_species_yt + per_species_yp)).tolist())
            else:  # species_classification
                num_classes = spec.get("num_classes", None)
                for s in range(num_species):
                    yt_s, yp_s = prepare_classification_for_cm(y_true[:, s], y_pred[:, s, :])
                    per_species_yt.append(yt_s)
                    per_species_yp.append(yp_s)
                all_labels = list(range(num_classes)) if num_classes else sorted(
                    np.unique(np.concatenate(per_species_yt)).tolist()
                )

            # --- Per-species F1 (macro across that species' own classes present) ---
            species_f1 = {}
            species_labels_text = {}
            for s in range(num_species):
                yt_s, yp_s = per_species_yt[s], per_species_yp[s]
                labels_present = sorted(set(yt_s.tolist()) | set(yp_s.tolist()))
                if len(labels_present) == 0:
                    f1 = 0.0
                else:
                    f1 = f1_score(yt_s, yp_s, labels=labels_present, average='macro', zero_division=0)
                species_f1[s] = f1

                if species_mapping and str(s) in species_mapping:
                    _, label_str = species_mapping[str(s)]
                elif species_mapping and s in species_mapping:
                    _, label_str = species_mapping[s]
                else:
                    label_str = None
                species_labels_text[s] = label_str if label_str else str(s)

            # --- Per-polyphony-degree F1 (pooled across all species) ---
            yt_pooled = np.concatenate(per_species_yt)
            yp_pooled = np.concatenate(per_species_yp)
            degree_f1 = {}
            for degree in all_labels:
                f1 = f1_score(
                    (yt_pooled == degree).astype(int),
                    (yp_pooled == degree).astype(int),
                    average='binary', zero_division=0
                )
                degree_f1[degree] = f1

            # --- Scalars for trend tracking across epochs ---
            for s, f1 in species_f1.items():
                self.writer.add_scalar(f"F1_per_species/{target}/{species_labels_text[s]}", f1, epoch)
            for degree, f1 in degree_f1.items():
                self.writer.add_scalar(f"F1_per_degree/{target}/degree_{degree}", f1, epoch)

            macro_species_f1 = float(np.mean(list(species_f1.values()))) if species_f1 else 0.0
            macro_degree_f1 = float(np.mean(list(degree_f1.values()))) if degree_f1 else 0.0
            self.writer.add_scalar(f"F1_macro/{target}/per_species", macro_species_f1, epoch)
            self.writer.add_scalar(f"F1_macro/{target}/per_degree", macro_degree_f1, epoch)

            # --- Bar chart: per-species F1, sorted worst-first ---
            sorted_species = sorted(species_f1.items(), key=lambda kv: kv[1])
            species_names = [species_labels_text[s] for s, _ in sorted_species]
            species_scores = [f1 for _, f1 in sorted_species]

            fig_h = max(3, 0.25 * num_species)
            fig1, ax1 = plt.subplots(figsize=(8, fig_h))
            bars = ax1.barh(species_names, species_scores, color='steelblue')
            ax1.set_xlabel("F1 score")
            ax1.set_xlim(0, 1)
            ax1.set_title(f"Per-Species F1 — {target} (epoch {epoch + 1}, macro avg: {macro_species_f1:.3f})")
            ax1.invert_yaxis()  # worst at top
            for bar, score in zip(bars, species_scores):
                ax1.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{score:.2f}", va='center', fontsize=7)
            plt.tight_layout()
            self.writer.add_figure(f"F1_breakdown/{target}/per_species", fig1, epoch)
            plt.close(fig1)

            # --- Bar chart: per-polyphony-degree F1 ---
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            degree_names = [str(d) for d in all_labels]
            degree_scores = [degree_f1[d] for d in all_labels]
            bars2 = ax2.bar(degree_names, degree_scores, color='darkorange')
            ax2.set_ylabel("F1 score")
            ax2.set_xlabel("Polyphony degree")
            ax2.set_ylim(0, 1)
            ax2.set_title(f"Per-Degree F1 — {target} (epoch {epoch + 1}, macro avg: {macro_degree_f1:.3f})")
            for bar, score in zip(bars2, degree_scores):
                ax2.text(bar.get_x() + bar.get_width() / 2, score + 0.01,
                        f"{score:.2f}", ha='center', fontsize=8)
            plt.tight_layout()
            self.writer.add_figure(f"F1_breakdown/{target}/per_degree", fig2, epoch)
            plt.close(fig2)

            print(f"F1 breakdown logged for '{target}' at epoch {epoch + 1} "
                f"(macro per-species: {macro_species_f1:.3f}, macro per-degree: {macro_degree_f1:.3f})")

        except Exception as e:
            print(f"Failed to log F1 breakdown for '{spec['name']}': {e}")

    # def _log_confusion_matrix(self, epoch, spec):
    #     try:
    #         import matplotlib.pyplot as plt
    #         from matplotlib.gridspec import GridSpec
    #         import io
    #         from PIL import Image
            
    #         target = spec["name"]
    
    #         # if cache and target in cache:
    #         #     y_pred, y_true = cache[target]
    #         # else:
    #         y_pred, y_true = self._get_predictions_and_true_labels(
    #             self.val_dataset, target
    #         )
    #         cm_type = spec["type"]
    #         threshold = spec.get("threshold", 0.5)
    #         # TODO: Separate semantic and logical categories ("regression" does not always mean "Polyphony Degree")
    #         if cm_type == "regression_round":
    #             yt, yp = prepare_polyphony_for_cm(y_true, y_pred)
    #             labels = np.unique(yt)
    #             title = "Polyphony Degree"
    #         elif cm_type == "binary":
    #             yt, yp = prepare_event_logits_for_cm(
    #                 y_true, y_pred, threshold=threshold
    #             )
    #             labels = [0, 1]
    #             title = "Event Detection"
    #         elif cm_type == "classification":
    #             yt, yp = prepare_classification_for_cm(y_true, y_pred)
    #             labels = list(range(spec['num_classes']))
    #             title = "Polyphony Degree Class"
    #         else:
    #             raise ValueError(f"Unknown confusion matrix type: {cm_type}")
            
    #         # Create the confusion matrix figure (original)
    #         cm_fig = plot_confusion_matrix_sklearn(
    #             yt, yp, labels=labels, title=title
    #         )
            
    #         # Convert confusion matrix to image
    #         buf = io.BytesIO()
    #         cm_fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    #         buf.seek(0)
    #         cm_image = Image.open(buf)
    #         plt.close(cm_fig)
            
    #         # Add metadata
    #         metadata_lines = [
    #             f"Model: {self.cfg.model._target_ if hasattr(self.cfg.model, '_target_') else self.cfg.model.get('name', 'N/A')}",
    #             f"Dataset config: {self.cfg.dataset.config if hasattr(self.cfg.dataset, 'config') else 'N/A'}",
    #             f"Input Feature: {self.cfg.train.get('input_feature_name', 'N/A')}",
    #             f"Epoch: {epoch + 1}",
    #         ]
            
    #         # Add hyperparameters if they exist
    #         if hasattr(self.cfg.log, 'hyperparameters') and self.cfg.log.hyperparameters:
    #             metadata_lines.append("\nHyperparameters:")
    #             for hp_key in self.cfg.log.hyperparameters:
    #                 # Navigate nested config keys (e.g., 'train.learning_rate')
    #                 value = self.cfg
    #                 for key_part in hp_key.split('.'):
    #                     value = getattr(value, key_part, 'N/A')
                    
    #                 # Check if value is a dict or DictConfig - if so, extract keys only
    #                 from omegaconf import DictConfig
    #                 if isinstance(value, (dict, DictConfig)):
    #                     dict_keys = ", ".join(value.keys())
    #                     metadata_lines.append(f"  {hp_key}: {dict_keys}")
    #                 else:
    #                     metadata_lines.append(f"  {hp_key}: {value}")
            
    #         metadata_text = "\n".join(metadata_lines)
            
    #         # Calculate metadata height needed
    #         num_lines = len(metadata_lines)
    #         metadata_height_ratio = max(0.15, num_lines * 0.02)
            
    #         # Create a new combined figure
    #         cm_width = cm_image.width / 100  # Convert pixels to inches (100 dpi)
    #         cm_height = cm_image.height / 100
    #         metadata_height = cm_height * metadata_height_ratio
            
    #         combined_fig = plt.figure(figsize=(cm_width, cm_height + metadata_height))
            
    #         # Create grid: confusion matrix on top, metadata below
    #         gs = GridSpec(2, 1, figure=combined_fig, 
    #                     height_ratios=[cm_height, metadata_height],
    #                     hspace=0.15)
            
    #         # Display confusion matrix image in top subplot
    #         ax_cm = combined_fig.add_subplot(gs[0])
    #         ax_cm.imshow(cm_image)
    #         ax_cm.axis('off')
            
    #         # Create metadata subplot below
    #         ax_meta = combined_fig.add_subplot(gs[1])
    #         ax_meta.axis('off')
    #         ax_meta.text(
    #             0.5, 0.5,
    #             metadata_text,
    #             fontsize=8,
    #             verticalalignment='center',
    #             horizontalalignment='center',
    #             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    #             transform=ax_meta.transAxes,
    #             family='monospace'
    #         )
            
    #         buf.close()
            
    #         self.writer.add_figure(
    #             f"Confusion_Matrix/{target}",
    #             combined_fig,
    #             epoch,
    #         )
    #         print(f"Confusion matrix logged for '{target}' at epoch {epoch + 1}")
    #     except Exception as e:
    #         print(f"Failed to log confusion matrix for '{spec['name']}': {e}")

    @tf.function
    def predict_batch(self, batch_x):
        return self.model(batch_x, training=False)

    def _get_predictions_and_true_labels(self, dataset, target_name):
        y_pred_all = []
        y_true_all = []

        for batch_x, batch_y in iter(dataset):
            #preds = self.model(batch_x, training=False)
            preds = self.predict_batch(batch_x)

            # Model outputs
            if isinstance(preds, dict):
                batch_pred = preds[target_name]
            else:
                batch_pred = preds #.squeeze(axis=-1)??

            # Labels
            if isinstance(batch_y, dict):
                batch_true = batch_y[target_name]
            else:
                batch_true = batch_y
            
            y_pred_all.append(batch_pred)
            y_true_all.append(batch_true)

        return np.concatenate([t.numpy() for t in y_pred_all]), np.concatenate([t.numpy() for t in y_true_all])
    
    # def _save_val_results(self, y_pred, y_true, target_name, save_path):
    #     """Save raw validation results to HDF5 for later statistical analysis."""
    #     timestamp = datetime.now().isoformat()
    #     epoch = getattr(self, 'current_epoch', 0)  # if you track this on the callback

    #     with h5py.File(save_path, 'a') as f:  # 'a' = append, so epochs accumulate
    #         group_key = f"epoch_{epoch:04d}/{target_name}"
    #         grp = f.require_group(group_key)

    #         # Overwrite if re-running same epoch
    #         for key in ('y_pred', 'y_true'):
    #             if key in grp:
    #                 del grp[key]

    #         grp.create_dataset('y_pred', data=y_pred, compression='gzip')
    #         grp.create_dataset('y_true', data=y_true, compression='gzip')

    #         # Store metadata alongside the arrays
    #         grp.attrs['timestamp'] = timestamp
    #         grp.attrs['target_name'] = target_name
    #         grp.attrs['n_samples'] = len(y_true)
    #         grp.attrs['model_name'] = self.model.name

    # def _get_predictions_and_true_labels(self, dataset):
    #     """Get predictions and true labels from validation dataset"""
    #     y_pred_list = []
    #     y_true_list = []
        
    #     for batch_x, batch_y in dataset:
    #         predictions = self.model(batch_x, training=False)
    #         y_pred_list.append(predictions.numpy())
    #         y_true_list.append(batch_y.numpy())
        
    #     return y_pred_list, y_true_list

    # def on_train_end(self, logs=None):
    #     # Log final confusion matrix
    #     if self.log_confusion_matrix and self.val_dataset is not None:
    #         for spec in self.confusion_matrix_specs:
    #             self._log_confusion_matrix(epoch=-1, spec=spec)

    #     if self.standard_tb_callback:
    #         self.standard_tb_callback.on_train_end(logs)

    #     if self.live:
    #         self.live.end()

    #     # Update writer metrics with best values and log to HParams
    #     self.writer.metrics = {
    #         k: float(self._best_val[k]) if k in self._best_val else None
    #         for k in self.writer.metrics
    #     }
    #     self.writer._log_hyperparameters()

    #     self.writer.close()
    #     print("Training completed!")

    # def on_train_end(self, logs=None):
    #     """Final logging and cleanup"""
    #     # Log final confusion matrix
    #     if self.log_confusion_matrix and self.val_dataset is not None:
    #         for spec in self.confusion_matrix_specs:
    #             self._log_confusion_matrix(epoch=-1, spec=spec) # Special epoch for final

    #     if self.standard_tb_callback:
    #         self.standard_tb_callback.on_train_end(logs)

    #     if self.live:
    #         self.live.end()
        
    #     # # Use the last recorded metrics from self.final_metrics
    #     # logs = logs or {}

    #     # self.writer._log_hyperparameters(self.params, self.metrics)
        
    #     # # Add any other metrics you want here, e.g. accuracy
        
    #     # # Assuming `self.params` holds your hparams dictionary
    #     # hparam_dict = self.params.tensorboard_compatible_copy()
        
    #     # # Now write the hparams summary with final metrics
    #     # self._add_hparams(hparam_dict, metrics)

    #     logs = logs or self._last_logs or {}
    #     print(logs)

    #     print(f"Final logs keys: {list(logs.keys())}")
    
    #     num_objectives = len(self.loss_objects)
        
    #     # Extract final metrics from logs
    #     for obj_name, loss_obj in self.loss_objects.items():
    #         current_weight = float(loss_obj.weight.numpy())
            
    #         # Get train loss
    #         train_weighted = self._get_loss_from_logs(logs, obj_name, '', num_objectives)
    #         if train_weighted is not None:
    #             train_base = self._calculate_base_loss(train_weighted, current_weight)
    #             self.metrics[f"{obj_name}_loss"] = float(train_base)
    #             print(f"Added {obj_name}_loss = {train_base}")
            
    #         # Get validation loss
    #         val_weighted = self._get_loss_from_logs(logs, obj_name, 'val_', num_objectives)
    #         if val_weighted is not None:
    #             val_base = self._calculate_base_loss(val_weighted, current_weight)
    #             self.metrics[f"{obj_name}_val_loss"] = float(val_base)
    #             print(f"Added {obj_name}_val_loss = {val_base}")

    #         for log_prefix in ['', 'val_']:
    #             acc_key = f"{log_prefix}{obj_name}_accuracy"
    #             if acc_key in logs:
    #                 self.metrics[acc_key] = float(logs[acc_key])
        
    #     print(f"Final metrics for hParams: {self.metrics}")

    #     # # Update latest_metrics with latest logs keys you want
    #     # for key in self.writer.metrics.keys():
    #     #     print(key)
    #     #     if key in logs:
    #     #         print(key)
    #     #         self.metrics[key] = logs[key]
        
    #     # Log hyperparameters + final metrics
    #     self.writer._log_hyperparameters(self.writer.params, self.metrics, log_dir=self.writer.log_dir)
    #     self.writer.close()
        
    #     print("Training completed!")
    #     self.writer.close()


def return_tensorboard_dir(subfolder=None, suffix='') -> PosixPath:
    """
    Returns the path to the TensorBoard logs directory for the current experiment.
    The path is constructed using the default directory, current datetime, and DVC experiment name.

    Returns:
        PosixPath: The path to the TensorBoard logs directory.
    """
    default_dir = config.get_env_variable("DEFAULT_DIR")
    dvc_exp_name = config.get_env_variable("DVC_EXP_NAME")
    current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    
    if subfolder:
        tensorboard_path = Path(
            f"{default_dir}/logs/tensorboard/{subfolder}/{current_datetime}_{dvc_exp_name}{suffix}"
        )
    else:
        tensorboard_path = Path(
            f"{default_dir}/logs/tensorboard/{current_datetime}_{dvc_exp_name}{suffix}"
        )

    tensorboard_path.mkdir(parents=True, exist_ok=True)

    return tensorboard_path

def return_checkpoint_path(subfolder=None, suffix='') -> PosixPath:
    """
    Returns the path to the checkpoint directory for the current experiment.
    The path is constructed using the default directory, current datetime, and DVC experiment name.

    Returns:
        PosixPath: The path to the TensorBoard logs directory.
    """
    default_dir = config.get_env_variable("DEFAULT_DIR")
    dvc_exp_name = config.get_env_variable("DVC_EXP_NAME")
    current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    
    if subfolder:
        checkpoint_path = Path(
            f"{default_dir}/checkpoints/{subfolder}/{current_datetime}_{dvc_exp_name}{suffix}.weights.h5"
        )
    else:
        checkpoint_path = Path(
            f"{default_dir}/checkpoints/{current_datetime}_{dvc_exp_name}{suffix}.weights.h5"
        )

    return checkpoint_path


def copy_tensorboard_logs() -> str:
    """
    Copies the TensorBoard logs specific to the current experiment from the host directory
    to the temporary experiment directory.

    Returns:
        str: The name of the copied directory.
    """
    default_dir = config.get_env_variable("DEFAULT_DIR")
    dvc_exp_name = config.get_env_variable("DVC_EXP_NAME")

    tensorboard_logs_source = Path(f"{default_dir}/logs/tensorboard")
    tensorboard_logs_destination = Path(f"exp_logs/tensorboard")
    tensorboard_logs_destination.mkdir(parents=True, exist_ok=True)
    for f in tensorboard_logs_source.iterdir():
        if f.is_dir() and f.name.endswith(dvc_exp_name):
            dir_name = f.name
            shutil.copytree(
                f, tensorboard_logs_destination / f.name, dirs_exist_ok=True
            )
            print(
                f"TensorBoard log '{f.name}' copied to '{tensorboard_logs_destination / f.name}'"
            )
            return f.name
    print("No TensorBoard logs found. Skipping copying.")
    return f"no_tensorboard_logs_{dvc_exp_name}"


def copy_slurm_logs(dir_name) -> None:
    """
    Copies the SLURM logs specific to the current experiment from the host directory
    to the temporary experiment directory.

    If the SLURM_JOB_ID is not found, the copying process is skipped.

    Args:
        dir_name (str): The name of the directory to copy the SLURM logs to.

    Raises:
        ValueError: If the directory name does not end with the DVC experiment name.
    """
    default_dir = config.get_env_variable("DEFAULT_DIR")
    current_slurm_job_id = config.get_env_variable("SLURM_JOB_ID")
    dvc_exp_name = config.get_env_variable("DVC_EXP_NAME")

    if dir_name is None:
        raise ValueError("Directory name is None.")
    elif not dir_name.endswith(dvc_exp_name):
        raise ValueError(f"Directory '{dir_name}' does not end with '{dvc_exp_name}'")

    if current_slurm_job_id:
        slurm_logs_source = Path(f"{default_dir}/logs/slurm")
        slurm_logs_destination = Path(f"exp_logs/slurm/{dir_name}")
        slurm_logs_destination.mkdir(parents=True, exist_ok=True)
        if current_slurm_job_id is not None:
            for f in slurm_logs_source.iterdir():
                if f.is_file() and f.name.endswith(current_slurm_job_id + ".out"):
                    shutil.copy(f, slurm_logs_destination)
        print(f"SLURM log 'slurm-{current_slurm_job_id}.out' copied to {slurm_logs_destination / f.name}.")
    else:
        print("No SLURM_JOB_ID found. Skipping SLURM logs copying.")
        
# def plot_spectrogram_with_metrics(
#     audio_array, sampling_rate,
#     split_name=None, example_idx=None, 
#     gt_polyphony=None, pred_polyphony=None,
#     gt_event_logits=None, pred_event_logits=None,
#     events=None, filename=None
# ):
#     """
#     Plot spectrogram with bounding boxes and metrics for a single example.
    
#     Args:
#         audio_array: Audio array for spectrogram
#         sampling_rate: Sampling rate of audio
#         split_name: Name of the dataset split
#         example_idx: Index of the example
#         gt_polyphony: Ground truth polyphony degree
#         pred_polyphony: Predicted polyphony degree
#         gt_event_logits: Ground truth event logits (array)
#         pred_event_logits: Predicted event logits (array)
#         events: Event bounding boxes [(start_time, end_time, freq_low, freq_high), ...]
#         filename: Optional filename to display in title
#     """
#     # Enforce mono audio
#     if audio_array.ndim != 1:
#         audio_array = np.mean(audio_array, axis=0)
    
#     # Compute Mel spectrogram
#     mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sampling_rate, fmax=8000)
#     S_dB = librosa.power_to_db(mel_spec, ref=np.max)
    
#     # --- Create figure with GridSpec ---
#     fig = plt.figure(figsize=(14, 8))
#     gs = gridspec.GridSpec(3, 2, width_ratios=[20, 1], height_ratios=[1, 3, 1.2], 
#                           figure=fig, hspace=0.25, wspace=0.1)
    
#     # --- Spectrogram components ---
#     ax_wave = fig.add_subplot(gs[0, 0])  # waveform
#     ax_spec = fig.add_subplot(gs[1, 0], sharex=ax_wave)  # spectrogram
#     cax_spec = fig.add_subplot(gs[1, 1])  # colorbar for spectrogram
    
#     # --- Plot waveform ---
#     times = np.arange(audio_array.size) / sampling_rate
#     ax_wave.plot(times, audio_array, color="gray")
#     ax_wave.set_ylabel("Amplitude")
    
#     # Build title
#     title = ""
#     if split_name is not None and example_idx is not None:
#         title += f"{split_name}[{example_idx}]"
#     if filename:
#         title += f" - {filename}"
#     ax_wave.set_title(title, fontsize=12, fontweight='bold')
#     ax_wave.grid(True, linestyle="--", alpha=0.3)
#     plt.setp(ax_wave.get_xticklabels(), visible=False)
    
#     # --- Plot spectrogram ---
#     img = librosa.display.specshow(
#         S_dB,
#         x_axis="time",
#         y_axis="mel",
#         sr=sampling_rate,
#         fmax=8000,
#         ax=ax_spec,
#     )
    
#     # Plot event bounding boxes if provided
#     if events is not None:
#         plot_event_bounding_boxes(
#             ax=ax_spec,
#             events=events,
#             edgecolor="cyan",
#             linewidth=2,
#         )
    
#     fig.colorbar(img, cax=cax_spec, format="%+2.0f dB", label="dB")
#     ax_spec.set_xlabel("Time [s]")
#     ax_spec.set_ylabel("Mel frequency [Hz]")
    
#     # --- Add metrics display ---
#     ax_metrics = fig.add_subplot(gs[2, :])
#     ax_metrics.axis('off')
    
#     metrics_text = []
    
#     # Display polyphony degree
#     if gt_polyphony is not None and pred_polyphony is not None:
#         metrics_text.append(f"Polyphony Degree - GT: {gt_polyphony:.3f}  |  Pred: {pred_polyphony:.3f}\n")
    
#     # Display event logits element-by-element comparison
#     if gt_event_logits is not None and pred_event_logits is not None:
#         gt_array = np.array(gt_event_logits).flatten()
#         pred_array = np.array(pred_event_logits).flatten()
        
#         metrics_text.append("Event Logits Comparison:")
#         metrics_text.append("Index | Ground Truth | Prediction | Difference")
#         metrics_text.append("-" * 50)
        
#         # Display each element
#         max_display = min(len(gt_array), len(pred_array), 10)  # Limit to 10 elements for readability
#         for i in range(max_display):
#             diff = pred_array[i] - gt_array[i]
#             metrics_text.append(f"  {i:3d}  |    {gt_array[i]:7.3f}   |   {pred_array[i]:7.3f}  |   {diff:+7.3f}")
        
#         if len(gt_array) > max_display:
#             metrics_text.append(f"  ... ({len(gt_array) - max_display} more elements)")
        
#         # Add summary statistics
#         metrics_text.append("")
#         metrics_text.append(f"Mean - GT: {np.mean(gt_array):.3f}  |  Pred: {np.mean(pred_array):.3f}")
#         metrics_text.append(f"Std  - GT: {np.std(gt_array):.3f}  |  Pred: {np.std(pred_array):.3f}")
#         metrics_text.append(f"MAE: {np.mean(np.abs(gt_array - pred_array)):.3f}")
    
#     if metrics_text:
#         textstr = '\n'.join(metrics_text)
#         props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
#         ax_metrics.text(0.5, 0.5, textstr, transform=ax_metrics.transAxes, 
#                       fontsize=9, verticalalignment='center', 
#                       horizontalalignment='center', bbox=props, family='monospace')
    
#     plt.tight_layout()
#     return fig

# def plot_spectrogram_with_metrics(
#     audio_array, sampling_rate,
#     split_name=None, example_idx=None, 
#     gt_polyphony=None, pred_polyphony=None,
#     gt_event_logits=None, pred_event_logits=None,
#     events=None, filename=None
# ):
#     """
#     Plot spectrogram with bounding boxes and metrics for a single example.
    
#     Args:
#         audio_array: Audio array for spectrogram
#         sampling_rate: Sampling rate of audio
#         split_name: Name of the dataset split
#         example_idx: Index of the example
#         gt_polyphony: Ground truth polyphony degree
#         pred_polyphony: Predicted polyphony degree
#         gt_event_logits: Ground truth event logits (array)
#         pred_event_logits: Predicted event logits (array)
#         events: Event bounding boxes [(start_time, end_time, freq_low, freq_high), ...]
#         filename: Optional filename to display in title
#     """
#     # Enforce mono audio
#     if audio_array.ndim != 1:
#         audio_array = np.mean(audio_array, axis=0)
    
#     # Compute Mel spectrogram
#     mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sampling_rate, fmax=8000)
#     S_dB = librosa.power_to_db(mel_spec, ref=np.max)
    
#     # --- Create figure with GridSpec ---
#     fig = plt.figure(figsize=(14, 9))
#     gs = gridspec.GridSpec(4, 2, width_ratios=[20, 1], height_ratios=[1, 0.8, 3, 1.2], 
#                           figure=fig, hspace=0.15, wspace=0.1)
    
#     # --- Spectrogram components ---
#     ax_wave = fig.add_subplot(gs[0, 0])  # waveform
#     ax_logits = fig.add_subplot(gs[1, 0], sharex=ax_wave)  # event logits bar
#     ax_spec = fig.add_subplot(gs[2, 0], sharex=ax_wave)  # spectrogram
#     cax_spec = fig.add_subplot(gs[2, 1])  # colorbar for spectrogram
    
#     # --- Plot waveform ---
#     times = np.arange(audio_array.size) / sampling_rate
#     ax_wave.plot(times, audio_array, color="gray")
#     ax_wave.set_ylabel("Amplitude", fontsize=9)
    
#     # Build title
#     title = ""
#     if split_name is not None and example_idx is not None:
#         title += f"{split_name}[{example_idx}]"
#     if filename:
#         title += f" - {filename}"
#     ax_wave.set_title(title, fontsize=12, fontweight='bold')
#     ax_wave.grid(True, linestyle="--", alpha=0.3)
#     plt.setp(ax_wave.get_xticklabels(), visible=False)
    
#     # --- Plot predicted event logits as bars ---
#     if pred_event_logits is not None:
#         pred_array = np.array(pred_event_logits).flatten()
#         duration = len(audio_array) / sampling_rate
#         n_steps = len(pred_array)
        
#         # Create time bins for each logit step
#         time_step = duration / n_steps
#         time_bins = np.linspace(0, duration, n_steps + 1)
#         time_centers = (time_bins[:-1] + time_bins[1:]) / 2
        
#         # Determine which logits represent predicted events (e.g., > 0.5 threshold)
#         threshold = 0.5
#         predicted_events = pred_array > threshold
        
#         # Create bar colors: highlight predicted events
#         colors = ['#ff6b6b' if pred else '#4ecdc4' for pred in predicted_events]
        
#         # Plot bars
#         ax_logits.bar(time_centers, pred_array, width=time_step * 0.95, 
#                      color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
        
#         # Add threshold line
#         ax_logits.axhline(y=threshold, color='red', linestyle='--', 
#                          linewidth=1.5, alpha=0.7, label=f'Threshold ({threshold})')
        
#         ax_logits.set_ylabel("Event Logits", fontsize=9)
#         ax_logits.set_ylim([0, max(1.0, np.max(pred_array) * 1.1)])
#         ax_logits.grid(True, linestyle="--", alpha=0.3, axis='y')
#         ax_logits.legend(loc='upper right', fontsize=8)
#         plt.setp(ax_logits.get_xticklabels(), visible=False)
        
#         # Add legend for colors
#         from matplotlib.patches import Patch
#         legend_elements = [
#             Patch(facecolor='#ff6b6b', edgecolor='black', label='Predicted Event'),
#             Patch(facecolor='#4ecdc4', edgecolor='black', label='No Event')
#         ]
#         ax_logits.legend(handles=legend_elements, loc='upper left', fontsize=8, ncol=2)
#     else:
#         ax_logits.axis('off')
    
#     # --- Plot spectrogram ---
#     img = librosa.display.specshow(
#         S_dB,
#         x_axis="time",
#         y_axis="mel",
#         sr=sampling_rate,
#         fmax=8000,
#         ax=ax_spec,
#     )
    
#     # Plot event bounding boxes if provided
#     if events is not None:
#         plot_event_bounding_boxes(
#             ax=ax_spec,
#             events=events,
#             edgecolor="cyan",
#             linewidth=2,
#         )
    
#     fig.colorbar(img, cax=cax_spec, format="%+2.0f dB", label="dB")
#     ax_spec.set_xlabel("Time [s]")
#     ax_spec.set_ylabel("Mel frequency [Hz]")
    
#     # --- Add metrics display ---
#     ax_metrics = fig.add_subplot(gs[3, :])
#     ax_metrics.axis('off')
    
#     metrics_text = []
    
#     # Display polyphony degree
#     if gt_polyphony is not None and pred_polyphony is not None:
#         metrics_text.append(f"Polyphony Degree - GT: {gt_polyphony:.3f}  |  Pred: {pred_polyphony:.3f}")
    
#     # Add summary statistics for event logits
#     if pred_event_logits is not None:
#         pred_array = np.array(pred_event_logits).flatten()
#         threshold = 0.5
#         n_predicted = np.sum(pred_array > threshold)
        
#         metrics_text.append(f"\nEvent Logits Summary:")
#         metrics_text.append(f"  Total Steps: {len(pred_array)}  |  Predicted Events: {n_predicted}  |  Mean: {np.mean(pred_array):.3f}  |  Std: {np.std(pred_array):.3f}")
    
#     if metrics_text:
#         textstr = '\n'.join(metrics_text)
#         props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
#         ax_metrics.text(0.5, 0.5, textstr, transform=ax_metrics.transAxes, 
#                       fontsize=9, verticalalignment='center', 
#                       horizontalalignment='center', bbox=props, family='monospace')
    
#     plt.tight_layout()
#     return fig

def plot_spectrogram_with_metrics(
    audio_array, sampling_rate,
    split_name=None, example_idx=None, 
    gt_polyphony=None, pred_polyphony=None,
    gt_event_logits=None, pred_event_logits=None,
    events=None, filename=None
):
    """
    Plot spectrogram with bounding boxes and metrics for a single example.
    
    Args:
        audio_array: Audio array for spectrogram
        sampling_rate: Sampling rate of audio
        split_name: Name of the dataset split
        example_idx: Index of the example
        gt_polyphony: Ground truth polyphony degree
        pred_polyphony: Predicted polyphony degree
        gt_event_logits: Ground truth event logits (array)
        pred_event_logits: Predicted event logits (array)
        events: Event bounding boxes [(start_time, end_time, freq_low, freq_high), ...]
        filename: Optional filename to display in title
    """
    # Enforce mono audio
    if audio_array.ndim != 1:
        audio_array = np.mean(audio_array, axis=0)
    
    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sampling_rate, fmax=8000)
    S_dB = librosa.power_to_db(mel_spec, ref=np.max)
    
    # --- Create figure with GridSpec ---
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(4, 2, width_ratios=[20, 1], height_ratios=[1, 0.8, 3, 1.2], 
                          figure=fig, hspace=0.15, wspace=0.1)
    
    # --- Spectrogram components ---
    ax_wave = fig.add_subplot(gs[0, 0])  # waveform
    ax_logits = fig.add_subplot(gs[1, 0], sharex=ax_wave)  # event logits bar
    ax_spec = fig.add_subplot(gs[2, 0], sharex=ax_wave)  # spectrogram
    cax_spec = fig.add_subplot(gs[2, 1])  # colorbar for spectrogram
    
    # --- Plot waveform ---
    times = np.arange(audio_array.size) / sampling_rate
    ax_wave.plot(times, audio_array, color="gray")
    ax_wave.set_ylabel("Amplitude", fontsize=9)
    
    # Build title
    title = ""
    if split_name is not None and example_idx is not None:
        title += f"{split_name}[{example_idx}]"
    if filename:
        title += f" - {filename}"
    ax_wave.set_title(title, fontsize=12, fontweight='bold')
    ax_wave.grid(True, linestyle="--", alpha=0.3)
    plt.setp(ax_wave.get_xticklabels(), visible=False)
    
    # --- Plot predicted event logits as bars ---
    if pred_event_logits is not None:
        pred_array = np.array(pred_event_logits).flatten()
        duration = len(audio_array) / sampling_rate
        n_steps = len(pred_array)
        
        # Create time bins for each logit step
        time_step = duration / n_steps
        time_bins = np.linspace(0, duration, n_steps + 1)
        time_centers = (time_bins[:-1] + time_bins[1:]) / 2
        
        # Determine which logits represent predicted events (e.g., > 0.5 threshold)
        threshold = 0.5
        predicted_events = pred_array > threshold
        
        # Prepare ground truth for comparison if available
        if gt_event_logits is not None:
            gt_array = np.array(gt_event_logits).flatten()
            # Ensure gt_array matches pred_array length
            if len(gt_array) != len(pred_array):
                # Interpolate or truncate to match
                if len(gt_array) < len(pred_array):
                    gt_array = np.interp(
                        np.linspace(0, len(gt_array)-1, len(pred_array)),
                        np.arange(len(gt_array)),
                        gt_array
                    )
                else:
                    gt_array = gt_array[:len(pred_array)]
            
            gt_events = gt_array > threshold
            
            # Create color map based on TP, TN, FP, FN
            colors = []
            for pred, gt in zip(predicted_events, gt_events):
                if pred and gt:
                    colors.append('#2ecc71')  # True Positive - green
                elif not pred and not gt:
                    colors.append('#3498db')  # True Negative - blue
                elif pred and not gt:
                    colors.append('#e74c3c')  # False Positive - red
                else:  # not pred and gt
                    colors.append('#f39c12')  # False Negative - orange
        else:
            # No ground truth available, use simple coloring
            colors = ['#ff6b6b' if pred else '#4ecdc4' for pred in predicted_events]
        
        # Plot bars
        ax_logits.bar(time_centers, pred_array, width=time_step * 0.95, 
                     color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # Add threshold line
        ax_logits.axhline(y=threshold, color='red', linestyle='--', 
                         linewidth=1.5, alpha=0.7, label=f'Threshold ({threshold})')
        
        # Add zero line for reference
        ax_logits.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        
        ax_logits.set_ylabel("Event Logits", fontsize=9)
        
        # Set y-axis limits to include negative values
        y_min = min(0, np.min(pred_array)) * 1.1
        y_max = max(1.0, np.max(pred_array)) * 1.1
        ax_logits.set_ylim([y_min, y_max])
        
        ax_logits.grid(True, linestyle="--", alpha=0.3, axis='y')
        plt.setp(ax_logits.get_xticklabels(), visible=False)
        
        # Add legend for colors
        from matplotlib.patches import Patch
        if gt_event_logits is not None:
            legend_elements = [
                Patch(facecolor='#2ecc71', edgecolor='black', label='True Positive'),
                Patch(facecolor='#3498db', edgecolor='black', label='True Negative'),
                Patch(facecolor='#e74c3c', edgecolor='black', label='False Positive'),
                Patch(facecolor='#f39c12', edgecolor='black', label='False Negative')
            ]
        else:
            legend_elements = [
                Patch(facecolor='#ff6b6b', edgecolor='black', label='Predicted Event'),
                Patch(facecolor='#4ecdc4', edgecolor='black', label='No Event')
            ]
        ax_logits.legend(handles=legend_elements, loc='upper left', fontsize=8, ncol=2)
    else:
        ax_logits.axis('off')
    
    # --- Plot spectrogram ---
    img = librosa.display.specshow(
        S_dB,
        x_axis="time",
        y_axis="mel",
        sr=sampling_rate,
        fmax=8000,
        ax=ax_spec,
    )
    
    # Plot event bounding boxes if provided
    if events is not None:
        plot_event_bounding_boxes(
            ax=ax_spec,
            events=events,
            edgecolor="cyan",
            linewidth=2,
        )
    
    fig.colorbar(img, cax=cax_spec, format="%+2.0f dB", label="dB")
    ax_spec.set_xlabel("Time [s]")
    ax_spec.set_ylabel("Mel frequency [Hz]")
    
    # --- Add metrics display ---
    ax_metrics = fig.add_subplot(gs[3, :])
    ax_metrics.axis('off')
    
    metrics_text = []
    
    # Display polyphony degree with larger font
    if gt_polyphony is not None and pred_polyphony is not None:
        metrics_text.append(f"Polyphony Degree - GT: {gt_polyphony:.3f}  |  Pred: {pred_polyphony:.3f}")
    
    if metrics_text:
        textstr = '\n'.join(metrics_text)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax_metrics.text(0.5, 0.5, textstr, transform=ax_metrics.transAxes, 
                      fontsize=14, verticalalignment='center', 
                      horizontalalignment='center', bbox=props, 
                      family='monospace', fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_event_bounding_boxes(
    ax,
    events,
    edgecolor="red",
    linewidth=2,
    linestyle="-",
    alpha=0.9,
    label=None,
):
    """
    Plot time–frequency bounding boxes on a spectrogram axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis containing the spectrogram (e.g., ax_spec).
    events : iterable of tuples
        Each event is (start_time, end_time, freq_low, freq_high),
        where time is in seconds and frequency in Hz.
    edgecolor : str
        Color of the bounding box edges.
    linewidth : float
        Line width of the bounding box edges.
    linestyle : str
        Line style of the bounding boxes.
    alpha : float
        Transparency of the bounding boxes.
    label : str or None
        Optional label for legend (only applied to first box).
    """

    for i, (t_start, t_end, f_low, f_high) in enumerate(events):
        width = t_end - t_start
        height = f_high - f_low

        rect = Rectangle(
            (t_start, f_low),
            width,
            height,
            fill=False,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            label=label if i == 0 else None,
        )

        ax.add_patch(rect)

def get_dvc_exp_name():
    try:
        return config.get_env_variable("DVC_EXP_NAME")
    except OSError:
        return "test-experiment"

def get_log_paths(cfg) -> dict[str, Path]:
    exp_name = get_dvc_exp_name()
    run_dir = f"{cfg.datetime}_{exp_name}"

    return {
        "train_log_dir": Path(cfg.path.train_output) / run_dir / "logs",
        "eval_log_dir": Path(cfg.path.eval_output) / run_dir / "logs",
        "soundscape_eval_log_dir": Path(cfg.path.soundscape_eval_output) / run_dir / "logs",
        "checkpoint_dir": Path(cfg.path.train_output) / run_dir / "checkpoints",
    }

class ModelAndHistorySaver(tf.keras.callbacks.Callback):
        def __init__(self, checkpoint_dir, loss_objects, previous_history=None, save_full_model_every_n_epochs=5, keep_last_n=5):
            super().__init__()
            self.checkpoint_path = checkpoint_dir
            self.combined_history = {k: list(v) for k, v in previous_history.items()} \
                                    if previous_history else {}
            self.save_model_every_n_epochs = save_full_model_every_n_epochs
            self.keep_last_n = keep_last_n
            self.best_val_loss = float('inf')
            self.loss_objects = loss_objects

            self.epoch_weights_dir = Path(checkpoint_dir) #+ '/epoch_weights/'
            self.best_weights_dir = Path(checkpoint_dir) #+ '/best_weights/'
            self.resumable_dir = Path(checkpoint_dir) #+ '/resumable_checkpoints/'

            os.makedirs(self.epoch_weights_dir, exist_ok=True)
            os.makedirs(self.best_weights_dir, exist_ok=True)
            os.makedirs(self.resumable_dir, exist_ok=True)

        def on_epoch_end(self, epoch, logs=None):
            
            # Update regular metrics
            for key, value in logs.items():
                self.combined_history.setdefault(key, []).append(float(value))
            
            # Update loss weights
            for obj_name, loss_obj in self.loss_objects.items():
                weight_key = f'loss_weight/{obj_name}'
                current_weight = float(loss_obj.weight.numpy())
                self.combined_history.setdefault(weight_key, []).append(current_weight)

            # # Update history
            for key, value in logs.items():
                self.combined_history.setdefault(key, []).append(float(value))

            # Save history
            history_path = self.resumable_dir / 'train_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.combined_history, f, indent=2)
            print(f"✓ Saved history at epoch {epoch + 1}")

            # Save current checkpoint
            val_loss = logs.get('val_loss')
            # self.model.save_weights(self.epoch_weights_dir / f'epoch_{epoch+1:03d}.weights.h5')
            # print(f"✓ Saved weights {val_loss:.4f} at epoch {epoch + 1}")

            # Save best checkpoint and model
            if val_loss and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.model.save_weights(self.best_weights_dir / f'best.weights.h5')
                self.model.save(self.best_weights_dir / f'best.keras')
                print(f"✓ New best val_loss {val_loss:.4f} at epoch {epoch + 1}. Saved new best weights.")

            # # Cleanup old checkpoints if configured
            # if self.keep_last_n:
            #     self._cleanup_old_checkpoints(epoch)

            # # Save model
            # if (epoch + 1) % self.save_model_every_n_epochs == 0:
            #     self.model.save(self.resumable_dir / f'epoch_{epoch+1:03d}.keras')
            #     self._cleanup_old_models(epoch)
            #     print(f"✓ Saved model at epoch {epoch + 1}")
               
                
        def _cleanup_old_checkpoints(self, current_epoch):
            for old_epoch in range(current_epoch - self.keep_last_n):
                path = self.epoch_weights_dir / f'epoch_{old_epoch+1:03d}.weights.h5'
                if os.path.exists(path):
                    os.remove(path)

        def _cleanup_old_models(self, current_epoch):
            for old_epoch in range(current_epoch - self.keep_last_n):
                path = self.resumable_dir / f'epoch_{old_epoch+1:03d}.keras'
                if os.path.exists(path):
                    os.remove(path)

    

def main():
    """Main function to copy SLURM and TensorBoard logs."""
    dir_name = copy_tensorboard_logs()
    copy_slurm_logs(dir_name=dir_name)

if __name__ == "__main__":
    main()
