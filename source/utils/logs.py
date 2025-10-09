# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.


"""
This module handles the logging and summary writing for the project.
"""

import datetime
import os
import shutil
from pathlib import Path, PosixPath
from typing import Any, Dict, Optional, Union

from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.patches as patches

import tensorflow as tf

if __name__ == "__main__":
    import config
else:
    from utils import config

def plot_confusion_matrix(y_pred, y_true):
    # Convert to flat NumPy arrays
    y_true =  np.concatenate(y_true, axis=0) #np.array(y_true)
    y_pred = np.concatenate(y_pred, axis=0) #np.array(y_pred)

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

class CustomSummaryWriterCallback(tf.keras.callbacks.Callback):
    """
    Custom callback that integrates with your CustomSummaryWriter
    Focuses on custom metrics and syncing, while standard TensorBoard handles built-in features
    """
    def __init__(self, writer, include_standard_tensorboard=True, test_dataset=None, 
                 log_confusion_matrix=True, confusion_matrix_frequency=5, input_shape=None):
        super().__init__()
        self.writer = writer
        self.test_dataset = test_dataset
        self.log_confusion_matrix = log_confusion_matrix
        self.confusion_matrix_frequency = confusion_matrix_frequency
        self.metrics = {}
        self.input_shape = input_shape


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
        # """Log epoch-level metrics and confusion matrix"""
        # train_loss = logs.get('loss', 0)
        # val_loss = logs.get('val_loss', 0)
        
        # print(f"Train loss: {train_loss:>8f}")
        # if val_loss > 0:
        #     print(f"Val Error: \n Avg loss: {val_loss:>8f} \n")
        
        # # Log basic epoch metrics to your CustomSummaryWriter
        # self.writer.add_scalar("Epoch_Loss/train", train_loss, epoch)
        # if val_loss > 0:
        #     self.writer.add_scalar("Epoch_Loss/val", val_loss, epoch)
        
        # Log confusion matrix every N epochs
        if (self.log_confusion_matrix and self.test_dataset is not None 
            and (epoch + 1) % self.confusion_matrix_frequency == 0):
            self._log_confusion_matrix(epoch)

        # Call standard TensorBoard callback
        if self.standard_tb_callback:
            self.standard_tb_callback.on_epoch_end(epoch, logs)
        
        # Step the writer (handles syncing)
        self.writer.step()
    

    def _log_confusion_matrix(self, epoch):
        """Generate and log confusion matrix"""
        try:
            
            # Get predictions and true labels
            y_pred, y_true = self._get_predictions_and_true_labels(self.test_dataset)
            
            # Generate confusion matrix plot
            figure = plot_confusion_matrix(y_pred, y_true)
            
            # Convert to image and log
            self.writer.add_figure("Confusion_Matrix", figure, epoch)
            
            print(f"Confusion matrix logged at epoch {epoch + 1}")
            
        except Exception as e:
            print(f"Failed to log confusion matrix: {e}")

    def _get_predictions_and_true_labels(self, dataset):
        """Get predictions and true labels from validation dataset"""
        y_pred_list = []
        y_true_list = []
        
        for batch_x, batch_y in dataset:
            predictions = self.model(batch_x, training=False)
            y_pred_list.append(predictions.numpy())
            y_true_list.append(batch_y.numpy())
        
        return y_pred_list, y_true_list

    def on_train_end(self, logs=None):
        """Final logging and cleanup"""
        # Log final confusion matrix
        if self.log_confusion_matrix and self.test_dataset is not None:
            self._log_confusion_matrix(epoch=-1)  # Special epoch for final

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
        # Update latest_metrics with latest logs keys you want
        for key in self.writer.metrics.keys():
            print(key)
            if key in logs:
                print(key)
                self.metrics[key] = logs[key]
        
        # Log hyperparameters + final metrics
        self.writer._log_hyperparameters(self.writer.params, self.metrics, log_dir=self.writer.log_dir)
        self.writer.close()
        
        print("Training completed!")
        self.writer.close()


def return_tensorboard_path(subfolder=None, suffix='') -> PosixPath:
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


def main():
    """Main function to copy SLURM and TensorBoard logs."""
    dir_name = copy_tensorboard_logs()
    copy_slurm_logs(dir_name=dir_name)


if __name__ == "__main__":
    main()
