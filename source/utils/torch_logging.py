# torch_logging.py
#
# Manual-loop equivalent of the logging half of logs.py
# (CustomSummaryWriterCallback + ModelAndHistorySaver). Since a plain PyTorch
# training loop has no Keras callback lifecycle, TorchSummaryWriterLogger's
# `log_epoch(...)` is meant to be called once per epoch, at the point where
# on_epoch_end() would have fired.
#
# Reuses, unmodified, from logs.py:
#   - CustomSummaryWriter        (already a torch.utils.tensorboard.SummaryWriter
#                                  subclass -- no TF dependency in its own logic)
#   - plot_confusion_matrix_sklearn
#   - build_confusion_matrix_specs
#   - get_log_paths
#
# TensorBoard tag scheme is kept identical to CustomSummaryWriterCallback:
#   loss_weights/{obj}                          (main writer)
#   {obj}_loss                                  (train/ and validation/ subwriters)
#   {metric_key}                                (train/ and validation/ subwriters,
#                                                 val_ prefix stripped for validation/)
#   best/{metric_key}, best/{metric_key}_epoch   (main writer, staircase)
#   Confusion_Matrix/{obj}[/aggregated|/per_species]
#   F1_per_species/{obj}/{species}, F1_per_degree/{obj}/degree_{d}
#   F1_macro/{obj}/per_species, F1_macro/{obj}/per_degree
#   F1_breakdown/{obj}/per_species, F1_breakdown/{obj}/per_degree

import io
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter

import torch

from utils.logs import CustomSummaryWriter, plot_confusion_matrix_sklearn
from utils.metrics import (
    prepare_polyphony_for_cm,
    prepare_classification_for_cm,
)


# spec["type"] -> which prepare_* to use. Mirrors _log_confusion_matrix in logs.py,
# restricted to the four objectives MultiTaskHead supports.
_CM_KIND = {
    "regression_round": "regression_round",
    "classification": "classification",
    "species_regression_round": "species_regression_round",
    "species_classification": "species_classification",
}


class TorchSummaryWriterLogger:
    HIGHER_IS_BETTER = {"accuracy", "f1", "auc"}

    def __init__(
        self,
        log_dir,
        params,
        metric_keys,
        loss_objects,
        cfg,
        confusion_matrix_specs=None,
        confusion_matrix_frequency=1,
        log_confusion_matrix=True,
        previous_history=None,
    ):
        self.log_dir = Path(log_dir)
        self.writer = CustomSummaryWriter(
            log_dir=str(self.log_dir),
            params=params,
            metrics={k: None for k in metric_keys},
            sync_interval=0,
        )
        self.train_writer = TorchSummaryWriter(log_dir=str(self.log_dir / "train"))
        self.val_writer = TorchSummaryWriter(log_dir=str(self.log_dir / "validation"))

        self.loss_objects = loss_objects
        self.cfg = cfg
        self.confusion_matrix_specs = confusion_matrix_specs or []
        self.confusion_matrix_frequency = confusion_matrix_frequency
        self.log_confusion_matrix = log_confusion_matrix

        self._best_val = {}
        self._best_step = {}

        if previous_history:
            self._replay_history(previous_history)

    # ---- loss bookkeeping (mirrors _get_loss_from_logs / _calculate_base_loss) ----

    def _get_loss_from_logs(self, logs, obj_name, prefix, num_objectives):
        if num_objectives == 1:
            return logs.get(f"{prefix}loss")
        for key in (f"{prefix}{obj_name}", f"{prefix}{obj_name}_loss"):
            if key in logs:
                return logs[key]
        return None

    @staticmethod
    def _calculate_base_loss(weighted_loss, weight):
        return weighted_loss / weight if weight > 1e-8 else weighted_loss

    def _replay_history(self, previous_history):
        """Replay a resumed run's history so TensorBoard curves stay continuous."""
        num_objectives = len(self.loss_objects)
        num_epochs = len(next(iter(previous_history.values())))
        print(f"Replaying {num_epochs} epochs of history to TensorBoard...")

        for epoch in range(num_epochs):
            logs = {key: values[epoch] for key, values in previous_history.items()}
            for obj_name, loss_obj in self.loss_objects.items():
                weight_key = f"loss_weight/{obj_name}"
                current_weight = (
                    previous_history[weight_key][epoch]
                    if weight_key in previous_history
                    else float(loss_obj.weight.item())
                )
                self.writer.add_scalar(f"loss_weights/{obj_name}", current_weight, epoch)

                w = self._get_loss_from_logs(logs, obj_name, "", num_objectives)
                if w is not None:
                    self.train_writer.add_scalar(f"{obj_name}_loss", self._calculate_base_loss(w, current_weight), epoch)

                vw = self._get_loss_from_logs(logs, obj_name, "val_", num_objectives)
                if vw is not None:
                    self.val_writer.add_scalar(f"{obj_name}_loss", self._calculate_base_loss(vw, current_weight), epoch)

            self._update_best(epoch, logs)

        self.train_writer.flush()
        self.val_writer.flush()

    def _log_losses(self, epoch, logs):
        num_objectives = len(self.loss_objects)
        for obj_name, loss_obj in self.loss_objects.items():
            current_weight = float(loss_obj.weight.item())
            self.writer.add_scalar(f"loss_weights/{obj_name}", current_weight, epoch)

            w = self._get_loss_from_logs(logs, obj_name, "", num_objectives)
            if w is not None:
                self.train_writer.add_scalar(f"{obj_name}_loss", self._calculate_base_loss(w, current_weight), epoch)

            vw = self._get_loss_from_logs(logs, obj_name, "val_", num_objectives)
            if vw is not None:
                self.val_writer.add_scalar(f"{obj_name}_loss", self._calculate_base_loss(vw, current_weight), epoch)

    def _update_best(self, epoch, logs):
        for metric_key in self.writer.metrics:
            metric = logs.get(metric_key)
            if metric is None:
                continue
            higher_is_better = any(m in metric_key for m in self.HIGHER_IS_BETTER)
            is_best = (
                self._best_val.get(metric_key) is None
                or (higher_is_better and metric > self._best_val[metric_key])
                or (not higher_is_better and metric < self._best_val[metric_key])
            )
            if is_best:
                self._best_val[metric_key] = metric
                self._best_step[metric_key] = epoch

            if self._best_val.get(metric_key) is not None:
                if self._best_step.get(metric_key) is not None:
                    self.writer.add_scalar(f"best/{metric_key}_epoch", self._best_step[metric_key], epoch)
                self.writer.add_scalar(f"best/{metric_key}", self._best_val[metric_key], epoch)

    # ---- confusion matrices / F1 breakdown ----
    # val_predictions: dict {obj_name: (y_pred_np, y_true_np)}, computed once per
    # epoch by the training loop and passed in here (avoids the double inference
    # pass that the original TF callback does for CM + F1 separately).

    def _log_confusion_matrix(self, epoch, spec, val_predictions):
        try:
            target = spec["name"]
            y_pred, y_true = val_predictions[target]
            cm_type = spec["type"]

            if cm_type == "regression_round":
                yt, yp = prepare_polyphony_for_cm(y_true, y_pred)
                labels = np.unique(yt)
                self._log_single_confusion_matrix(epoch, target, yt, yp, labels, "Polyphony Degree", spec)

            elif cm_type == "classification":
                yt, yp = prepare_classification_for_cm(y_true, y_pred)
                labels = list(range(spec["num_classes"]))
                self._log_single_confusion_matrix(epoch, target, yt, yp, labels, "Polyphony Degree Class", spec)

            elif cm_type == "species_regression_round":
                yt_agg, yp_agg = prepare_polyphony_for_cm(y_true.flatten(), y_pred.flatten())
                labels_agg = np.unique(yt_agg)
                self._log_single_confusion_matrix(
                    epoch, target, yt_agg, yp_agg, labels_agg,
                    "Species Polyphony Regression (Aggregated)", spec,
                    tb_tag=f"Confusion_Matrix/{target}/aggregated",
                )
                self._log_species_confusion_matrix_grid(
                    epoch, target, y_true, y_pred, cm_type="regression_round",
                    num_classes=None, species_mapping=spec.get("species_mapping"),
                )

            elif cm_type == "species_classification":
                num_classes = spec.get("num_classes")
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
                    tb_tag=f"Confusion_Matrix/{target}/aggregated",
                )
                self._log_species_confusion_matrix_grid(
                    epoch, target, y_true, y_pred, cm_type="classification",
                    num_classes=num_classes, species_mapping=spec.get("species_mapping"),
                )
            else:
                raise ValueError(f"Unknown confusion matrix type: {cm_type}")

        except Exception as e:
            print(f"Failed to log confusion matrix for '{spec['name']}': {e}")

    def _build_metadata_text(self, epoch):
        lines = [
            f"Model: {getattr(self.cfg.model, '_target_', self.cfg.model.get('name', 'N/A'))}",
            f"Dataset config: {getattr(self.cfg.dataset, 'config', 'N/A')}",
            f"Input Feature: {self.cfg.train.get('input_feature_name', 'N/A')}",
            f"Epoch: {epoch + 1}",
        ]
        return "\n".join(lines)

    def _log_single_confusion_matrix(self, epoch, target, yt, yp, labels, title, spec, tb_tag=None):
        from matplotlib.gridspec import GridSpec

        tb_tag = tb_tag or f"Confusion_Matrix/{target}"
        cm_fig = plot_confusion_matrix_sklearn(yt, yp, labels=labels, title=title)

        buf = io.BytesIO()
        cm_fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
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
        gs = GridSpec(2, 1, figure=combined_fig, height_ratios=[cm_height, metadata_height], hspace=0.15)
        ax_cm = combined_fig.add_subplot(gs[0])
        ax_cm.imshow(cm_image)
        ax_cm.axis("off")
        ax_meta = combined_fig.add_subplot(gs[1])
        ax_meta.axis("off")
        ax_meta.text(
            0.5, 0.5, metadata_text, fontsize=8, verticalalignment="center",
            horizontalalignment="center", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            transform=ax_meta.transAxes, family="monospace",
        )
        buf.close()

        self.writer.add_figure(tb_tag, combined_fig, epoch)
        plt.close(combined_fig)
        print(f"Confusion matrix logged for '{target}' at epoch {epoch + 1} -> {tb_tag}")

    def _log_species_confusion_matrix_grid(self, epoch, target, y_true, y_pred, cm_type, num_classes, species_mapping):
        num_species = y_true.shape[1]
        ncols = min(4, num_species)
        nrows = int(np.ceil(num_species / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5), squeeze=False)

        for s in range(num_species):
            row, col = divmod(s, ncols)
            ax = axes[row][col]
            if species_mapping and str(s) in species_mapping:
                species_label = species_mapping[str(s)][1] or str(s)
            elif species_mapping and s in species_mapping:
                species_label = species_mapping[s][1] or str(s)
            else:
                species_label = str(s)

            try:
                if cm_type == "regression_round":
                    yt_s, yp_s = prepare_polyphony_for_cm(y_true[:, s], y_pred[:, s])
                    labels_s = sorted(np.unique(np.concatenate([yt_s, yp_s])).tolist())
                else:
                    yt_s, yp_s = prepare_classification_for_cm(y_true[:, s], y_pred[:, s, :])
                    labels_s = list(range(num_classes)) if num_classes else sorted(np.unique(yt_s).tolist())

                cm = confusion_matrix(yt_s, yp_s, labels=labels_s)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_s)
                disp.plot(ax=ax, colorbar=False, xticks_rotation="horizontal")
                ax.set_title(species_label, fontsize=8, pad=3)
                ax.tick_params(axis="both", labelsize=6)
                ax.set_xlabel("Predicted", fontsize=6)
                ax.set_ylabel("True", fontsize=6)
            except Exception:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{species_label}\n(error)", ha="center", va="center", fontsize=7, transform=ax.transAxes)

        for s in range(num_species, nrows * ncols):
            row, col = divmod(s, ncols)
            axes[row][col].axis("off")

        fig.suptitle(f"Per-Species Confusion Matrices — Epoch {epoch + 1}", fontsize=10, y=1.01)
        plt.tight_layout()
        tb_tag = f"Confusion_Matrix/{target}/per_species"
        self.writer.add_figure(tb_tag, fig, epoch)
        plt.close(fig)
        print(f"Per-species confusion matrix grid logged for '{target}' at epoch {epoch + 1} -> {tb_tag}")

    def _log_f1_breakdown(self, epoch, spec, val_predictions):
        try:
            target = spec["name"]
            cm_type = spec["type"]
            if cm_type not in ("species_regression_round", "species_classification"):
                return

            y_pred, y_true = val_predictions[target]
            species_mapping = spec.get("species_mapping")
            num_species = y_true.shape[1]

            per_species_yt, per_species_yp = [], []
            if cm_type == "species_regression_round":
                for s in range(num_species):
                    yt_s, yp_s = prepare_polyphony_for_cm(y_true[:, s], y_pred[:, s])
                    per_species_yt.append(yt_s)
                    per_species_yp.append(yp_s)
                all_labels = sorted(np.unique(np.concatenate(per_species_yt + per_species_yp)).tolist())
            else:
                num_classes = spec.get("num_classes")
                for s in range(num_species):
                    yt_s, yp_s = prepare_classification_for_cm(y_true[:, s], y_pred[:, s, :])
                    per_species_yt.append(yt_s)
                    per_species_yp.append(yp_s)
                all_labels = (
                    list(range(num_classes)) if num_classes
                    else sorted(np.unique(np.concatenate(per_species_yt)).tolist())
                )

            species_f1, species_labels_text = {}, {}
            for s in range(num_species):
                yt_s, yp_s = per_species_yt[s], per_species_yp[s]
                labels_present = sorted(set(yt_s.tolist()) | set(yp_s.tolist()))
                f1 = f1_score(yt_s, yp_s, labels=labels_present, average="macro", zero_division=0) if labels_present else 0.0
                species_f1[s] = f1
                if species_mapping and str(s) in species_mapping:
                    label_str = species_mapping[str(s)][1]
                elif species_mapping and s in species_mapping:
                    label_str = species_mapping[s][1]
                else:
                    label_str = None
                species_labels_text[s] = label_str or str(s)

            yt_pooled = np.concatenate(per_species_yt)
            yp_pooled = np.concatenate(per_species_yp)
            degree_f1 = {
                degree: f1_score((yt_pooled == degree).astype(int), (yp_pooled == degree).astype(int), average="binary", zero_division=0)
                for degree in all_labels
            }

            for s, f1 in species_f1.items():
                self.writer.add_scalar(f"F1_per_species/{target}/{species_labels_text[s]}", f1, epoch)
            for degree, f1 in degree_f1.items():
                self.writer.add_scalar(f"F1_per_degree/{target}/degree_{degree}", f1, epoch)

            macro_species_f1 = float(np.mean(list(species_f1.values()))) if species_f1 else 0.0
            macro_degree_f1 = float(np.mean(list(degree_f1.values()))) if degree_f1 else 0.0
            self.writer.add_scalar(f"F1_macro/{target}/per_species", macro_species_f1, epoch)
            self.writer.add_scalar(f"F1_macro/{target}/per_degree", macro_degree_f1, epoch)

            sorted_species = sorted(species_f1.items(), key=lambda kv: kv[1])
            species_names = [species_labels_text[s] for s, _ in sorted_species]
            species_scores = [f1 for _, f1 in sorted_species]
            fig_h = max(3, 0.25 * num_species)
            fig1, ax1 = plt.subplots(figsize=(8, fig_h))
            bars = ax1.barh(species_names, species_scores, color="steelblue")
            ax1.set_xlabel("F1 score")
            ax1.set_xlim(0, 1)
            ax1.set_title(f"Per-Species F1 — {target} (epoch {epoch + 1}, macro avg: {macro_species_f1:.3f})")
            ax1.invert_yaxis()
            for bar, score in zip(bars, species_scores):
                ax1.text(score + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.2f}", va="center", fontsize=7)
            plt.tight_layout()
            self.writer.add_figure(f"F1_breakdown/{target}/per_species", fig1, epoch)
            plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            degree_names = [str(d) for d in all_labels]
            degree_scores = [degree_f1[d] for d in all_labels]
            bars2 = ax2.bar(degree_names, degree_scores, color="darkorange")
            ax2.set_ylabel("F1 score")
            ax2.set_xlabel("Polyphony degree")
            ax2.set_ylim(0, 1)
            ax2.set_title(f"Per-Degree F1 — {target} (epoch {epoch + 1}, macro avg: {macro_degree_f1:.3f})")
            for bar, score in zip(bars2, degree_scores):
                ax2.text(bar.get_x() + bar.get_width() / 2, score + 0.01, f"{score:.2f}", ha="center", fontsize=8)
            plt.tight_layout()
            self.writer.add_figure(f"F1_breakdown/{target}/per_degree", fig2, epoch)
            plt.close(fig2)

            print(f"F1 breakdown logged for '{target}' at epoch {epoch + 1} "
                  f"(macro per-species: {macro_species_f1:.3f}, macro per-degree: {macro_degree_f1:.3f})")

        except Exception as e:
            print(f"Failed to log F1 breakdown for '{spec['name']}': {e}")

    # ---- public API ----

    def log_epoch(self, epoch, logs, val_predictions=None):
        """
        logs: flat dict, same keys the Keras `logs` dict would have had, e.g.
            {'loss': ..., 'val_loss': ...,
             'polyphony_reg_loss': ..., 'val_polyphony_reg_loss': ...,
             'val_polyphony_reg_accuracy': ..., ...}
        val_predictions: dict {obj_name: (y_pred_np, y_true_np)} over the full
            validation set this epoch (only needed if confusion matrices are on).
        """
        self._log_losses(epoch, logs)

        if (
            val_predictions is not None
            and self.confusion_matrix_specs
            and (epoch + 1) % self.confusion_matrix_frequency == 0
        ):
            for spec in self.confusion_matrix_specs:
                self._log_confusion_matrix(epoch, spec, val_predictions)
                self._log_f1_breakdown(epoch, spec, val_predictions)

        for key, value in logs.items():
            if key.startswith("val_"):
                self.val_writer.add_scalar(key.removeprefix("val_"), value, epoch)
            else:
                self.train_writer.add_scalar(key, value, epoch)

        self._update_best(epoch, logs)

        self.writer.step()
        self.train_writer.flush()
        self.val_writer.flush()

    def close(self):
        if self.log_confusion_matrix:
            pass  # final CM already logged as part of the last regular epoch
        self.writer.metrics = {
            k: float(self._best_val[k]) if k in self._best_val else None
            for k in self.writer.metrics
        }
        self.writer._log_hyperparameters()
        self.writer.close()
        self.train_writer.close()
        self.val_writer.close()
        print("Training completed!")


class ModelAndHistorySaverTorch:
    """Torch equivalent of logs.ModelAndHistorySaver."""

    def __init__(self, checkpoint_dir, loss_objects, previous_history=None, keep_last_n=1):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.combined_history = {k: list(v) for k, v in previous_history.items()} if previous_history else {}
        self.loss_objects = loss_objects
        self.best_val_loss = float("inf")
        self.keep_last_n = keep_last_n

    def on_epoch_end(self, epoch, logs, model, optimizer):
        for key, value in logs.items():
            self.combined_history.setdefault(key, []).append(float(value))
        for obj_name, loss_obj in self.loss_objects.items():
            self.combined_history.setdefault(f"loss_weight/{obj_name}", []).append(float(loss_obj.weight.item()))

        history_path = self.checkpoint_dir / "train_history.json"
        with open(history_path, "w") as f:
            json.dump(self.combined_history, f, indent=2)
        print(f"✓ Saved history at epoch {epoch + 1}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_weights": {k: float(v.weight.item()) for k, v in self.loss_objects.items()},
        }
        torch.save(checkpoint, self.checkpoint_dir / f"epoch_{epoch + 1:03d}.pt")

        val_loss = logs.get("val_loss")
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")
            print(f"✓ New best val_loss {val_loss:.4f} at epoch {epoch + 1}. Saved new best checkpoint.")

        if self.keep_last_n:
            self._cleanup_old_checkpoints(epoch)

    def _cleanup_old_checkpoints(self, current_epoch):
        for old_epoch in range(current_epoch - self.keep_last_n):
            path = self.checkpoint_dir / f"epoch_{old_epoch + 1:03d}.pt"
            if path.exists():
                path.unlink()
