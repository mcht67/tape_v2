# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.


"""
This module handles the logging and summary writing for the project.
"""

import os
from pathlib import Path, PosixPath
from typing import Any, Dict, Optional, Union
from omegaconf import DictConfig
import datetime

from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.patches as patches
from matplotlib import gridspec
import librosa

import tensorflow as tf

if __name__ == "__main__":
    import config
else:
    from utils import config

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

def prepare_polyphony_for_cm(y_true, y_pred):
    """
    y_true: (N,) or (N, 1)
    y_pred: (N, 1)
    """
    y_true = np.asarray(y_true).squeeze().astype(int)
    y_pred = np.asarray(y_pred).squeeze()

    y_pred = np.round(y_pred).astype(int)

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

    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    return fig


class CustomSummaryWriter(SummaryWriter):
    """
    A custom subclass of the TensorBoard SummaryWriter that allows for logging hyperparameters,
    displaying scalar metrics in the HParams tab, and automatically synchronizing logs with a remote directory.

    Args:
        log_dir (Union[str, PosixPath]): Directory where the TensorBoard logs will be stored.
        params (Optional[config.Params[str, Any]]): config.Params object of DVC hyperparameters to display. Defaults to None.
        metrics (Optional[Dict[str, None]]): Dictionary of initial metrics to display in the HParams tab. Defaults to {}.
        sync_interval (Optional[int]): Number of steps between automatic syncs to the remote directory.
                                       Defaults to the value of the 'TUSTU_SYNC_INTERVAL' environment variable.
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
        remote_dir: Optional[Union[str, PosixPath]] = None,
    ):
        super().__init__(log_dir=log_dir)

        self.sync_interval = (
            sync_interval
            if sync_interval is not None
            else int(config.get_env_variable("TUSTU_SYNC_INTERVAL"))
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
            self._log_hyperparameters(params, metrics, log_dir)

        self.current_step = 0

    def _construct_remote_dir(self) -> str:
        """Constructs the remote directory path based on environment variables."""
        tensorboard_host_dir = config.get_env_variable("TUSTU_TENSORBOARD_HOST_DIR")
        tensorboard_host = config.get_env_variable("TUSTU_TENSORBOARD_HOST")
        tensorboard_host_savepath = Path(
            f'{tensorboard_host_dir}/{config.get_env_variable("TUSTU_PROJECT_NAME")}/logs/tensorboard'
        )
        os.system(f"ssh {tensorboard_host} 'mkdir -p {tensorboard_host_savepath}'")
        return f"{tensorboard_host}:{tensorboard_host_savepath}"

    def _extract_datetime_from_log_dir(self, log_dir: Union[str, PosixPath]) -> str:
        """Extracts datetime information from the log directory path."""
        return str(log_dir).split("/")[-1].split("_")[0]

    def _log_hyperparameters(
        self,
        params: config.Params[str, Any],
        metrics: Dict[str, None],
        log_dir: str,
    ) -> None:
        """Logs hyperparameters and initial metrics to TensorBoard."""
        clean_params = params.tensorboard_compatible_copy()
        clean_params["datetime"] = self.datetime
        # params = params.flattened_copy()
        #cparams["datetime"] = self.datetime
        self._add_hparams(hparam_dict=clean_params, metric_dict=metrics, run_name=log_dir)

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
                spec['num_classes'] = obj_config["num_classes"]
            specs.append(spec)
    
    return specs

class CustomSummaryWriterCallback(tf.keras.callbacks.Callback):
    """
    Custom callback that integrates with your CustomSummaryWriter
    Focuses on custom metrics and syncing, while standard TensorBoard handles built-in features
    """
    def __init__(self, writer, include_standard_tensorboard=True, val_dataset=None, 
                 log_confusion_matrix=True, confusion_matrix_frequency=5, confusion_matrix_specs=None, input_shape=None, cfg=None, loss_objects={}):
        super().__init__()
        self.writer = writer
        self.val_dataset = val_dataset
        self.log_confusion_matrix = log_confusion_matrix
        self.confusion_matrix_frequency = confusion_matrix_frequency
        self.confusion_matrix_specs = confusion_matrix_specs or []
        self.metrics = {}
        self.input_shape = input_shape
        self.cfg = cfg
        self.loss_objects = loss_objects

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

        # Create separate file writers for train and validation
        self.train_writer = tf.summary.create_file_writer(
            str(Path(writer.log_dir) / 'train')
        )
        self.val_writer = tf.summary.create_file_writer(
            str(Path(writer.log_dir) / 'validation')
        )

    def set_model(self, model):
        """Called when the callback is attached to a model"""
        super().set_model(model)
        if self.standard_tb_callback:
            self.standard_tb_callback.set_model(model)

        # Log the model graph once
        self._log_model_graph(model)

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
        if self.standard_tb_callback:
            self.standard_tb_callback.on_train_begin(logs)

    def on_epoch_begin(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        if self.standard_tb_callback:
            self.standard_tb_callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Log base losses and weights following TensorBoard conventions."""
        if logs is None:
            return
        
        # Debug: Print available log keys on first epoch
        if epoch == 0:
            print(f"Available log keys: {list(logs.keys())}")
            
        # Handle single vs multi-objective scenarios
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

            # Log confusion matrix every N epochs
            if (
                self.val_dataset is not None
                and self.confusion_matrix_specs
                and (epoch + 1) % self.confusion_matrix_frequency == 0
            ):
                for spec in self.confusion_matrix_specs:
                    self._log_confusion_matrix(epoch, spec)

        # if (self.log_confusion_matrix and self.val_dataset is not None 
        #     and (epoch + 1) % self.confusion_matrix_frequency == 0):
        #     self._log_confusion_matrix(epoch)

        # Call standard TensorBoard callback
        if self.standard_tb_callback:
            self.standard_tb_callback.on_epoch_end(epoch, logs)

        # Step the writer (handles syncing)
        self.writer.step()
        
        # Flush all writers
        self.train_writer.flush()
        self.val_writer.flush()
    
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
    
    # def on_train_end(self, logs=None):
    #     """Close file writers when training ends."""
    #     self.train_writer.close()
    #     self.val_writer.close()

    # def on_epoch_end(self, epoch, logs=None):
    #     # """Log epoch-level metrics and confusion matrix"""
    #     """Log base losses (weight=1.0), weighted losses, and weights for train and val."""
    #     if logs is None:
    #         return
        
    #     # Handle single vs multi-objective scenarios
    #     num_objectives = len(self.loss_objects)
            
    #     for obj_name, loss_obj in self.loss_objects.items():
    #         # Get current weight
    #         current_weight = float(loss_obj.weight.numpy())
            
    #         # Log the weight itself
    #         self.writer.add_scalar(
    #             f'loss_weights/{obj_name}',
    #             current_weight,
    #             epoch
    #         )
            
    #         # Process both train and validation losses
    #         for prefix in ['', 'val_']:

    #             if num_objectives == 1:
    #                 # Single objective: loss is logged as 'loss' or 'val_loss'
    #                 possible_keys = [
    #                     f'{prefix}loss',
    #                 ]
    #             else:
    #                 # Multi-objective: loss is logged with output name
    #                 possible_keys = [
    #                     f'{prefix}{obj_name}',
    #                     f'{prefix}{obj_name}_loss',
    #                 ]
                
    #             for key in possible_keys:
    #                 if key in logs:
    #                     weighted_loss = logs[key]
    #                     break
                
    #             if weighted_loss is not None:
    #                 tb_prefix = 'val_' if prefix else 'train_'
                    
    #                 # Log the weighted loss (what actually affects training)
    #                 self.writer.add_scalar(
    #                     f'{tb_prefix}losses_weighted/{obj_name}',
    #                     weighted_loss,
    #                     epoch
    #                 )
                    
    #                 # Calculate base loss (as if weight=1.0)
    #                 # base_loss = weighted_loss / weight
    #                 if current_weight > 1e-8:
    #                     base_loss = weighted_loss / current_weight
    #                 else:
    #                     # If weight is 0, we can't recover the base loss
    #                     # Log weighted loss (which is also ~0)
    #                     base_loss = weighted_loss
                    
    #                 # Log base loss (unweighted, i.e., weight=1.0)
    #                 self.writer.add_scalar(
    #                     f'{tb_prefix}losses_base/{obj_name}',
    #                     base_loss,
    #                     epoch
    #                 )
    #     # train_loss = logs.get('loss', 0)
    #     # val_loss = logs.get('val_loss', 0)
        
    #     # print(f"Train loss: {train_loss:>8f}")
    #     # if val_loss > 0:
    #     #     print(f"Val Error: \n Avg loss: {val_loss:>8f} \n")
        
    #     # # Log basic epoch metrics to your CustomSummaryWriter
    #     # self.writer.add_scalar("Epoch_Loss/train", train_loss, epoch)
    #     # if val_loss > 0:
    #     #     self.writer.add_scalar("Epoch_Loss/val", val_loss, epoch)
        
    #     # Log confusion matrix every N epochs
    #     if (
    #         self.val_dataset is not None
    #         and self.confusion_matrix_specs
    #         and (epoch + 1) % self.confusion_matrix_frequency == 0
    #     ):
    #         for spec in self.confusion_matrix_specs:
    #             self._log_confusion_matrix(epoch, spec)

    #     # if (self.log_confusion_matrix and self.val_dataset is not None 
    #     #     and (epoch + 1) % self.confusion_matrix_frequency == 0):
    #     #     self._log_confusion_matrix(epoch)

    #     # Call standard TensorBoard callback
    #     if self.standard_tb_callback:
    #         self.standard_tb_callback.on_epoch_end(epoch, logs)
        
    #     # Step the writer (handles syncing)
    #     self.writer.step()

    # def _log_confusion_matrix(self, epoch, spec):
    #     try:
    #         target = spec["name"]
    #         cm_type = spec["type"]
    #         threshold = spec.get("threshold", 0.5)

    #         y_pred, y_true = self._get_predictions_and_true_labels(
    #             self.val_dataset, target
    #         )

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

    #         else:
    #             raise ValueError(f"Unknown confusion matrix type: {cm_type}")

    #         fig = plot_confusion_matrix_sklearn(
    #             yt, yp, labels=labels, title=title
    #         )

    #         self.writer.add_figure(
    #             f"Confusion_Matrix/{target}",
    #             fig,
    #             epoch,
    #         )

    #         print(f"Confusion matrix logged for '{target}' at epoch {epoch + 1}")

    #     except Exception as e:
    #         print(f"Failed to log confusion matrix for '{spec['name']}': {e}")

    def _log_confusion_matrix(self, epoch, spec):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            import io
            from PIL import Image
            
            target = spec["name"]
            cm_type = spec["type"]
            threshold = spec.get("threshold", 0.5)
            y_pred, y_true = self._get_predictions_and_true_labels(
                self.val_dataset, target
            )
            # TODO: Separate semantic and logical categories ("regression" does not always mean "Polyphony Degree")
            if cm_type == "regression_round":
                yt, yp = prepare_polyphony_for_cm(y_true, y_pred)
                labels = np.unique(yt)
                title = "Polyphony Degree"
            elif cm_type == "binary":
                yt, yp = prepare_event_logits_for_cm(
                    y_true, y_pred, threshold=threshold
                )
                labels = [0, 1]
                title = "Event Detection"
            elif cm_type == "classification":
                yt, yp = prepare_classification_for_cm(y_true, y_pred)
                labels = list(range(spec['num_classes']))
                title = "Polyphony Degree Class"
            else:
                raise ValueError(f"Unknown confusion matrix type: {cm_type}")
            
            # Create the confusion matrix figure (original)
            cm_fig = plot_confusion_matrix_sklearn(
                yt, yp, labels=labels, title=title
            )
            
            # Convert confusion matrix to image
            buf = io.BytesIO()
            cm_fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            cm_image = Image.open(buf)
            plt.close(cm_fig)
            
            # Add metadata
            metadata_lines = [
                f"Model: {self.cfg.model._target_ if hasattr(self.cfg.model, '_target_') else self.cfg.model.get('name', 'N/A')}",
                f"Dataset subset: {self.cfg.dataset.subset if hasattr(self.cfg.dataset, 'subset') else 'N/A'}",
                f"Input Feature: {self.cfg.train.get('input_feature_name', 'N/A')}",
                f"Epoch: {epoch + 1}",
            ]
            
            # Add hyperparameters if they exist
            if hasattr(self.cfg.log, 'hyperparameters') and self.cfg.log.hyperparameters:
                metadata_lines.append("\nHyperparameters:")
                for hp_key in self.cfg.log.hyperparameters:
                    # Navigate nested config keys (e.g., 'train.learning_rate')
                    value = self.cfg
                    for key_part in hp_key.split('.'):
                        value = getattr(value, key_part, 'N/A')
                    
                    # Check if value is a dict or DictConfig - if so, extract keys only
                    from omegaconf import DictConfig
                    if isinstance(value, (dict, DictConfig)):
                        dict_keys = ", ".join(value.keys())
                        metadata_lines.append(f"  {hp_key}: {dict_keys}")
                    else:
                        metadata_lines.append(f"  {hp_key}: {value}")
            
            metadata_text = "\n".join(metadata_lines)
            
            # Calculate metadata height needed
            num_lines = len(metadata_lines)
            metadata_height_ratio = max(0.15, num_lines * 0.02)
            
            # Create a new combined figure
            cm_width = cm_image.width / 100  # Convert pixels to inches (100 dpi)
            cm_height = cm_image.height / 100
            metadata_height = cm_height * metadata_height_ratio
            
            combined_fig = plt.figure(figsize=(cm_width, cm_height + metadata_height))
            
            # Create grid: confusion matrix on top, metadata below
            gs = GridSpec(2, 1, figure=combined_fig, 
                        height_ratios=[cm_height, metadata_height],
                        hspace=0.15)
            
            # Display confusion matrix image in top subplot
            ax_cm = combined_fig.add_subplot(gs[0])
            ax_cm.imshow(cm_image)
            ax_cm.axis('off')
            
            # Create metadata subplot below
            ax_meta = combined_fig.add_subplot(gs[1])
            ax_meta.axis('off')
            ax_meta.text(
                0.5, 0.5,
                metadata_text,
                fontsize=8,
                verticalalignment='center',
                horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                transform=ax_meta.transAxes,
                family='monospace'
            )
            
            buf.close()
            
            self.writer.add_figure(
                f"Confusion_Matrix/{target}",
                combined_fig,
                epoch,
            )
            print(f"Confusion matrix logged for '{target}' at epoch {epoch + 1}")
        except Exception as e:
            print(f"Failed to log confusion matrix for '{spec['name']}': {e}")

    # def _log_confusion_matrix(self, epoch, spec):
    #     try:
    #         target = spec["name"]
    #         cm_type = spec["type"]
    #         threshold = spec.get("threshold", 0.5)
    #         y_pred, y_true = self._get_predictions_and_true_labels(
    #             self.val_dataset, target
    #         )
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
    #         else:
    #             raise ValueError(f"Unknown confusion matrix type: {cm_type}")
            
    #         fig = plot_confusion_matrix_sklearn(
    #             yt, yp, labels=labels, title=title
    #         )
            
    #         # Add metadata to the figure
    #         metadata_lines = [
    #             f"Model: {self.cfg.model._target_ if hasattr(self.cfg.model, '_target_') else self.cfg.model.get('name', 'N/A')}",
    #             f"Dataset: {self.cfg.dataset.subset if hasattr(self.cfg.dataset, 'subset') else 'N/A'}",
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
                    
    #                 # Check if value is a dict - if so, extract keys only
    #                 if isinstance(value, (dict, DictConfig)):
    #                     dict_keys = ", ".join(value.keys())
    #                     metadata_lines.append(f"  {hp_key}: {dict_keys}")
    #                 else:
    #                     metadata_lines.append(f"  {hp_key}: {value}")

    #         metadata_text = "\n".join(metadata_lines)
            
    #         # Count the number of lines to estimate required space
    #         num_lines = len(metadata_lines)
    #         # Estimate height needed: increase multiplier for more space
    #         text_height = max(0.2, num_lines * 0.02)  # Increased from 0.015 to 0.02
            
    #         # Get current figure size and ONLY expand vertically (preserve width)
    #         current_size = fig.get_size_inches()
    #         original_width = current_size[0]
    #         new_height = current_size[1] + text_height * current_size[1] * 2.5
    #         fig.set_size_inches(original_width, new_height)

    #         # Adjust layout to make room at the bottom (proportion based on new height)
    #         bottom_margin = (text_height * 1.3) / (text_height * 1.3 + 1)
    #         # Use subplots_adjust instead of tight_layout to preserve original sizing
    #         fig.subplots_adjust(bottom=bottom_margin)

    #         # Add text below the confusion matrix
    #         fig.text(
    #             0.5, bottom_margin * 0.45,
    #             metadata_text,
    #             fontsize=8,
    #             verticalalignment='center',
    #             horizontalalignment='center',
    #             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    #             transform=fig.transFigure,
    #             family='monospace'
    #         )
            
    #         self.writer.add_figure(
    #             f"Confusion_Matrix/{target}",
    #             fig,
    #             epoch,
    #         )
    #         print(f"Confusion matrix logged for '{target}' at epoch {epoch + 1}")
    #     except Exception as e:
    #         print(f"Failed to log confusion matrix for '{spec['name']}': {e}")

    # def _log_confusion_matrix(self, epoch, spec):
    #     try:
    #         target = spec["name"]
    #         cm_type = spec["type"]
    #         threshold = spec.get("threshold", 0.5)

    #         y_pred, y_true = self._get_predictions_and_true_labels(
    #             self.val_dataset, target
    #         )

    #         if cm_type == "regression_round":
    #             y_pred_cls = y_pred #np.rint(y_pred).astype(int)
    #             y_true_cls = y_true.astype(int)

    #         elif cm_type == "binary":
    #             y_pred_cls = (y_pred >= threshold).astype(int)
    #             y_true_cls = y_true.astype(int)

    #         else:
    #             raise ValueError(f"Unknown confusion matrix type: {cm_type}")

    #         figure = plot_confusion_matrix(y_pred_cls, y_true_cls)

    #         self.writer.add_figure(
    #             f"Confusion_Matrix/{target}",
    #             figure,
    #             epoch,
    #         )

    #         print(f"Confusion matrix logged for '{target}' at epoch {epoch + 1}")

    #     except Exception as e:
    #         print(f"Failed to log confusion matrix for '{spec['name']}': {e}")

    # def _log_confusion_matrix(self, epoch):
    #     """Generate and log confusion matrix"""
    #     try:
            
    #         # Get predictions and true labels
    #         y_pred, y_true = self._get_predictions_and_true_labels(self.val_dataset)
            
    #         # Generate confusion matrix plot
    #         figure = plot_confusion_matrix(y_pred, y_true)
            
    #         # Convert to image and log
    #         self.writer.add_figure("Confusion_Matrix", figure, epoch)
            
    #         print(f"Confusion matrix logged at epoch {epoch + 1}")
            
    #     except Exception as e:
    #         print(f"Failed to log confusion matrix: {e}")

    def _get_predictions_and_true_labels(self, dataset, target_name):
        y_pred_all = []
        y_true_all = []

        for batch_x, batch_y in dataset:
            preds = self.model(batch_x, training=False)

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

            
            y_pred_all.append(batch_pred.numpy())
            y_true_all.append(batch_true.numpy())

        return np.concatenate(y_pred_all), np.concatenate(y_true_all)

    # def _get_predictions_and_true_labels(self, dataset):
    #     """Get predictions and true labels from validation dataset"""
    #     y_pred_list = []
    #     y_true_list = []
        
    #     for batch_x, batch_y in dataset:
    #         predictions = self.model(batch_x, training=False)
    #         y_pred_list.append(predictions.numpy())
    #         y_true_list.append(batch_y.numpy())
        
    #     return y_pred_list, y_true_list

    def on_train_end(self, logs=None):
        """Final logging and cleanup"""
        # Log final confusion matrix
        if self.log_confusion_matrix and self.val_dataset is not None:
            for spec in self.confusion_matrix_specs:
                self._log_confusion_matrix(epoch=-1, spec=spec) # Special epoch for final

        if self.standard_tb_callback:
            self.standard_tb_callback.on_train_end(logs)
        
        # # Use the last recorded metrics from self.final_metrics
        # logs = logs or {}

        # self.writer._log_hyperparameters(self.params, self.metrics)
        
        # # Add any other metrics you want here, e.g. accuracy
        
        # # Assuming `self.params` holds your hparams dictionary
        # hparam_dict = self.params.tensorboard_compatible_copy()
        
        # # Now write the hparams summary with final metrics
        # self._add_hparams(hparam_dict, metrics)

        logs = logs or {}
        print(logs)

        print(f"Final logs keys: {list(logs.keys())}")
    
        num_objectives = len(self.loss_objects)
        
        # Extract final metrics from logs
        for obj_name, loss_obj in self.loss_objects.items():
            current_weight = float(loss_obj.weight.numpy())
            
            # Get train loss
            train_weighted = self._get_loss_from_logs(logs, obj_name, '', num_objectives)
            if train_weighted is not None:
                train_base = self._calculate_base_loss(train_weighted, current_weight)
                self.metrics[f"{obj_name}_loss"] = float(train_base)
                print(f"Added {obj_name}_loss = {train_base}")
            
            # Get validation loss
            val_weighted = self._get_loss_from_logs(logs, obj_name, 'val_', num_objectives)
            if val_weighted is not None:
                val_base = self._calculate_base_loss(val_weighted, current_weight)
                self.metrics[f"val_{obj_name}_loss"] = float(val_base)
                print(f"Added val_{obj_name}_loss = {val_base}")
        
        print(f"Final metrics for hParams: {self.metrics}")

        # # Update latest_metrics with latest logs keys you want
        # for key in self.writer.metrics.keys():
        #     print(key)
        #     if key in logs:
        #         print(key)
        #         self.metrics[key] = logs[key]
        
        # Log hyperparameters + final metrics
        self.writer._log_hyperparameters(self.writer.params, self.metrics, log_dir=self.writer.log_dir)
        self.writer.close()
        
        print("Training completed!")
        self.writer.close()


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

from matplotlib.patches import Rectangle

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
    return config.get_env_variable("DVC_EXP_NAME")

def main():
    """Main function to copy SLURM and TensorBoard logs."""
    dir_name = copy_tensorboard_logs()
    copy_slurm_logs(dir_name=dir_name)


if __name__ == "__main__":
    main()
