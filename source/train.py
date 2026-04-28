# import matplotlib
# matplotlib.use("Agg")

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from datasets import concatenate_datasets
from omegaconf import OmegaConf
import os
import model
from hydra.utils import instantiate
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import huggingface_hub

from utils.logs import plot_spectrogram_with_metrics, return_tensorboard_dir, CustomSummaryWriter, CustomSummaryWriterCallback, build_confusion_matrix_specs, get_dvc_exp_name
from utils.general import reshape_tensor_data
from utils.config import set_random_seeds, Params
from utils.dataset import add_labels, load_dataset_with_retry
from losses import create_losses_from_objectives, setup_loss_scheduler

tf.keras.backend.clear_session()

def make_tf_dataset(hf_dataset, features, labels, batch_size, shuffle=False):
    X = np.stack(hf_dataset[features]).astype(np.float32)
    y = np.array(hf_dataset[labels]).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(batch_size)
    return ds

def get_tf_datasets(dataset, features, labels, batch_size):
    train_dataset = dataset['train'].to_tf_dataset(
        columns=features,
        label_cols=labels, 
        batch_size=batch_size,
        shuffle=True,
        prefetch=False
    )

    test_dataset = dataset['test'].to_tf_dataset(
        columns=features,
        label_cols=labels,
        batch_size=batch_size,
        shuffle=False,
        prefetch=False
    )

    val_dataset = dataset['validation'].to_tf_dataset(
        columns=features,
        label_cols=labels,
        batch_size=batch_size,
        shuffle=False,
        prefetch=False
    )

    return train_dataset, test_dataset, val_dataset

def apply_batched_reshape(dataset, features, input_dim, pooling_strategy, suffix, batch_size=100):
    """Apply reshape_tensor_data in batches to avoid PyArrow offset overflow"""
    
    # Process in batches
    processed_datasets = []
    total_samples = len(dataset)
    
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        print(f"Processing reshape batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")
        
        # Select batch
        batch_dataset = dataset.select(range(i, end_idx))
        
        # Process batch
        batch_processed = batch_dataset.map(
            lambda x: reshape_tensor_data(x, features, input_dim, pooling_strategy, suffix),
            desc=f"Reshaping batch {i//batch_size + 1}"
        )
        
        processed_datasets.append(batch_processed)
    
    # Concatenate all processed batches
    print(f"Concatenating {len(processed_datasets)} reshape batches...")
    return concatenate_datasets(processed_datasets)

def get_predictions_and_true_labels(model, dataset):
    y_pred = []
    y_true = []

    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        y_pred.append(preds)
        y_true.append(y_batch.numpy())

    return y_pred, y_true

# def sanitize_objectives_cfg(cfg, keys_to_keep):
#     """Strip non-serializable objects (loss fns, omegaconf) from objectives_cfg."""
#     import omegaconf
    
#     serializable = {}
#     KEEP_KEYS = keys_to_keep#{"num_classes", "type", "threshold", "label", "weight"}
    
#     for obj_name, obj_cfg in cfg.items():
#         # Convert omegaconf DictConfig to plain dict
#         if isinstance(obj_cfg, omegaconf.DictConfig):
#             obj_cfg = omegaconf.OmegaConf.to_container(obj_cfg, resolve=True)
        
#         # Keep only JSON-serializable metadata, drop loss objects
#         clean = {}
#         # for k, v in obj_cfg.items():
#         #     if k in KEEP_KEYS and isinstance(v, (str, int, float, bool, type(None))):
#         #         clean[k] = v
#         #     elif k == "confusion_matrix" and isinstance(v, dict):
#         #         clean[k] = {ck: cv for ck, cv in v.items() 
#         #                     if isinstance(cv, (str, int, float, bool, type(None)))}
#         for k, v in obj_cfg.items():
#             if isinstance(v, (str, int, float, bool, type(None))):
#                 clean[k] = v
#             elif isinstance(v, dict):
#                 clean[k] = {ck: cv for ck, cv in v.items() 
#                             if isinstance(cv, (str, int, float, bool, type(None)))}
#             elif isinstance(v, list):
#                 clean[k] = [cv for cv in v
#                             if isinstance(cv, (str, int, float, bool, type(None)))]

#         serializable[obj_name] = clean
    
#     return serializable

# Configuration
cfg = OmegaConf.load("params.yaml")

# Load the hyperparameters from the "params.yaml" file for usage with Tensorboard SummaryWriter
params = Params()

random_seed = cfg.general.random_seed
set_random_seeds(random_seed)

# dataset_path =  cfg.path.dataset
huggingface_path = cfg.dataset.huggingface_path
dataset_config = cfg.dataset.config

study_name = cfg.log.study_name
load_model_path = cfg.train.load_model_path if 'load_model_path' in cfg.train else None
load_checkpoint_path = cfg.train.load_checkpoint_path if 'load_checkpoint_path' in cfg.train else  None
load_history_path = cfg.train.load_history_path if 'load_history_path' in cfg.train else None

input_feature_name = cfg.train.input_feature_name
total_epochs = cfg.train.epochs
initial_epoch = cfg.train.initial_epoch if 'initial_epoch' in cfg.train and cfg.train.initial_epoch else 0
learning_rate = cfg.train.learning_rate
batch_size = cfg.train.batch_size
num_batches_train = cfg.train.num_batches_train if 'num_batches_train' in cfg.train else None
num_batches_val = cfg.train.num_batches_val if 'num_batches_val' in cfg.train else None

# dvc_exp_name = get_dvc_exp_name()
# current_datetime = datetime.now().strftime("%Y%m%d-%H%M")

#path_suffix = cfg.log.path_suffix if 'path_suffix' in cfg.log else None
checkpoint_dir = 'checkpoints' #f'checkpoints/{study_name}/{current_datetime}_{dvc_exp_name}_{path_suffix}' if path_suffix else f'checkpoints/{study_name}/{current_datetime}_{dvc_exp_name}/'
log_dir = 'logs'

model_cfg = cfg.model
objectives_cfg = cfg.objectives
#objectives_dict =  OmegaConf.to_container(objectives_cfg, resolve=True)

# print("Objectives cfg:")
# print(objectives_cfg)

labels = [objectives_cfg[x]['label'] for x in objectives_cfg]

# Add objectives to the config
model_cfg.objectives_cfg = objectives_cfg

tensorboard_subfolder = cfg.log.tensorboard_subfolder
tensorboard_suffix = cfg.log.tensorboard_suffix

print(f"Training with {input_feature_name} as input feature and {labels} as labels on dataset {huggingface_path} with config {dataset_config}.")

# Get tensorboard path based on path, dataset subset, features and datetime
# Set DEFAULT_DIR if not set (usually when running without dvc)
os.environ.setdefault('DEFAULT_DIR', os.getcwd())
os.environ.setdefault('DVC_EXP_NAME', 'test-experiment')

# tensorboard_path = return_tensorboard_dir(subfolder=study_name) #get_tensorboard_path(cfg) #TODO: refactor to work in a similar manner with dvc and without
# os.makedirs(tensorboard_path, exist_ok=True)

#################################
# Handle GPU
#################################
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {gpus}")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
if not gpus:
    print("WARNING: No GPU found, training on CPU")

#################################
# Load dataset
#################################



 # Load environment variables from .env file
load_dotenv('local.env')
token=os.getenv('HUGGINGFACE_TOKEN')

# Huggingface login
# huggingface_hub.login(token=os.getenv('HUGGINGFACE_TOKEN'))
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

from huggingface_hub import HfApi
api = HfApi()
info = api.dataset_info("mcht67/Polyphonic-Bird-Set", token=huggingface_token)
for config in info.card_data.get("configs", []):
    if config.get("config_name") == dataset_config:
        print(config.get("data_files"))
        print(config.get("data_dir"))

# Load Dataset
print(f"[INFO] HF_DATASETS_OFFLINE={os.environ.get('HF_DATASETS_OFFLINE', 'NOT SET')} (ommits updating datasets to avoid hitting rate limit on Huggingface Hub)")
dataset = load_dataset_with_retry(huggingface_path, dataset_config, token=huggingface_token, download_mode='force_redownload') #TODO: change to 'reuse_cache_if_exists' after testing to avoid hitting rate limits on Huggingface Hub

# Get input dim
dataset_train = dataset['train']
dataset_train_first = dataset_train[0]
embeddings = dataset['train'][0][input_feature_name]
input_dim = tf.squeeze(np.array(dataset['train'][0][input_feature_name])).shape

# Compute additional labels
time_dim = input_dim[0] if len(input_dim) > 1 else None
freq_dim = input_dim[1] if len(input_dim) > 2 else None
dataset, added_labels = add_labels(dataset, labels, time_dim=time_dim, freq_dim=freq_dim)

print("Added labels: ", added_labels)

existing_labels = added_labels + ['polyphony_degree']
missing_labels = set(labels) ^ set(existing_labels)
if missing_labels:
    raise Exception("Not all requested labels could be computed.")


# Get tensorflow datasets
train_dataset, test_dataset, val_dataset = get_tf_datasets(dataset, input_feature_name, labels, batch_size)
if num_batches_train: train_dataset = train_dataset.take(num_batches_train) # take fewer batches to reduce train dataset size
if num_batches_val: val_dataset = val_dataset.take(num_batches_val)
train_size = num_batches_train * batch_size if num_batches_train else len(dataset['train'])
val_size = num_batches_val * batch_size if num_batches_val else len(dataset['validation'])

train_dataset = train_dataset #.cache().prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset #.cache().prefetch(tf.data.AUTOTUNE)

metrics = {}
for key in objectives_cfg.keys():
    metrics[f"{key}_loss"] = None
    metrics[f"val_{key}_loss"] = None

print("Metrics:", metrics)

params['dataset']['train_size'] = str(train_size) #str(len(dataset['train']))
params['dataset']['val_size'] = str(val_size) #str(len(dataset['validation']))
params['dataset']['test_size'] = str(len(dataset['test']))
params['train']['objectives'] = list(cfg.objectives.keys())
print(params)

confusion_matrix_specs = build_confusion_matrix_specs(objectives_cfg)

# checkpoint_path = f"logs/models/{study_name}/{current_datetime}_{dvc_exp_name}_{input_feature_name}.weights.h5" #return_checkpoint_path(subfolder=f'{study_name}_{input_feature_name}')
# # TEMPORARY checkpoint solution should be handled by resuming experiment later TODO: 
# checkpoint_dir = os.path.dirname(checkpoint_path)
# # checkpoint_names = os.listdir(checkpoint_dir)
# # checkpoint_path = os.path.join(checkpoint_dir, checkpoint_names[0])
# print("Checkpoint path:", checkpoint_path)

num_batches = len(train_dataset) 

# model_path = f'models/{input_feature_name}_large.keras'
# current_datetime = datetime.now().strftime("%Y%m%d-%H%M")
# save_model_path = f'checkpoints/{study_name}/{current_datetime}_{dvc_exp_name}/{dvc_exp_name}_{input_feature_name}.keras'
# save_model_dir = Path(save_model_path).parent
# model_save_path = f'{study_name}/{current_datetime}_{dvc_exp_name}/{dvc_exp_name}_{input_feature_name}'
# os.makedirs(save_model_dir, exist_ok=True)
# history_path = save_model_path.replace('.keras', '_history.json')

losses = create_losses_from_objectives(objectives_cfg) 

##############
# Model
###############
tf.keras.backend.clear_session()

# Get model and history
previous_history = None
if load_model_path and os.path.isfile(load_model_path): 
    print("Loading model from", load_model_path)
    model = tf.keras.models.load_model(load_model_path)

    load_history_path = load_model_path.replace('.keras', '_history.json')

    # Load previous history
    if os.path.exists(load_history_path):
        with open(load_history_path, 'r') as f:
            previous_history = json.load(f)
        initial_epoch = len(previous_history['loss'])
        print(f"✓ Loaded history from {load_history_path}")
        print(f"Resuming from epoch {initial_epoch}")
    else:
        previous_history = None
        print(f"No history found, starting from epoch {initial_epoch}")

elif load_checkpoint_path and os.path.isfile(load_checkpoint_path):

    print(f"Loading weights from {load_checkpoint_path}")
    model = instantiate(cfg.model)
    model.compile(optimizer=Adam(learning_rate), loss=losses)
    
    # Initialize variables with forward pass
    sample_batch = next(iter(train_dataset))
    _ = model(sample_batch[0], training=False)

    model.load_weights(load_checkpoint_path)
    print(f"Resuming from epoch {initial_epoch}")
else:
    print("Creating new model")
    model = instantiate(model_cfg)
    print("Model objectives config:")
    print(model.objectives_cfg)
    # Initialize new model with forward pass
    sample_batch = next(iter(train_dataset))
    _ = model(sample_batch[0], training=False)
    print(f"New model has {len(model.trainable_variables)} trainable variables")

    model.compile(optimizer=Adam(learning_rate), loss=losses)
    
    previous_history = None
    initial_epoch = 0

model.summary()

##############
# Callbacks
###############

# class HistorySaver(tf.keras.callbacks.Callback):
#     def __init__(self, filepath, initial_history=None):
#         super().__init__()
#         self.filepath = filepath
#         self.combined_history = initial_history if initial_history else {}
    
#     def on_epoch_end(self, epoch, logs=None):
#         # Append current epoch's metrics
#         for key, value in logs.items():
#             if key not in self.combined_history:
#                 self.combined_history[key] = []
#             self.combined_history[key].append(float(value))
        
#         # Save after each epoch
#         with open(self.filepath, 'wb') as f:
#             pickle.dump(self.combined_history, f)

# class DebugCallback(tf.keras.callbacks.Callback):
#     def on_epoch_begin(self, epoch, logs=None):
#         print(f"\n=== EPOCH {epoch} BEGIN ===")
#         print(f"Model compiled: {self.model.compiled}")
#         print(f"Optimizer: {type(self.model.optimizer)}")
#         print(f"Number of variables: {len(self.model.trainable_variables)}")
#         print(f"First variable shape: {self.model.trainable_variables[0].shape}")
    
#     def on_epoch_end(self, epoch, logs=None):
#         print(f"\n=== EPOCH {epoch} END ===")
#         print(f"Model compiled: {self.model.compiled}")

# class ModelAndHistorySaver(tf.keras.callbacks.Callback):
#     def __init__(self, model_path, history_path, initial_history=None, save_every_n_epochs=1):
#         super().__init__()
#         self.model_path = model_path
#         self.history_path = history_path
#         self.combined_history = initial_history if initial_history else {}
#         self.save_every_n_epochs = save_every_n_epochs
    
#     def on_epoch_end(self, epoch, logs=None):
#         # Append current epoch's metrics to history
#         for key, value in logs.items():
#             if key not in self.combined_history:
#                 self.combined_history[key] = []
#             self.combined_history[key].append(float(value))
        
#         # Save history every epoch
#         with open(self.history_path, 'wb') as f:
#             pickle.dump(self.combined_history, f)
        
#         # Save model at specified intervals
#         if (epoch + 1) % self.save_every_n_epochs == 0:
#             self.model.save(self.model_path)
            # print(f"✓ Saved model and history at epoch {epoch + 1}")

import json
class ModelAndHistorySaver(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, loss_objects, previous_history=None, save_full_model_every_n_epochs=5, keep_last_n=None):
        super().__init__()
        self.checkpoint_path = checkpoint_dir
        self.combined_history = {k: list(v) for k, v in previous_history.items()} \
                                 if previous_history else {}
        self.save_model_every_n_epochs = save_full_model_every_n_epochs
        self.keep_last_n = keep_last_n
        self.best_val_loss = float('inf')
        self.loss_objects = loss_objects

        self.epoch_weights_dir = checkpoint_dir + '/epoch_weights/'
        self.best_weights_dir = checkpoint_dir + '/best_weights/'
        self.resumable_dir = checkpoint_dir + '/resumable_checkpoints/'

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
        history_path = self.resumable_dir + 'train_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.combined_history, f, indent=2)
        print(f"✓ Saved history at epoch {epoch + 1}")

        # Save current checkpoint
        val_loss = logs.get('val_loss')
        self.model.save_weights(self.epoch_weights_dir + f'epoch_{epoch+1:03d}.weights.h5')
        print(f"✓ Saved weights {val_loss:.4f} at epoch {epoch + 1}")

        # Save best checkpoint
        if val_loss and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.model.save_weights(self.best_weights_dir + f'best.weights.h5')
            print(f"✓ New best val_loss {val_loss:.4f} at epoch {epoch + 1}. Saved new best weights.")

        # Save rolling last-N checkpoints
        if self.keep_last_n:
            self._cleanup_old_checkpoints(epoch)

        # Save model
        if (epoch + 1) % self.save_model_every_n_epochs == 0:
            self.model.save(self.resumable_dir + f'epoch_{epoch+1:03d}.keras')
            print(f"✓ Saved model at epoch {epoch + 1}")
            
    def _cleanup_old_checkpoints(self, current_epoch):
        for old_epoch in range(current_epoch - self.keep_last_n):
            path = self.epoch_weights_dir + f'epoch_{old_epoch+1:03d}.weights.h5'
            if os.path.exists(path):
                os.remove(path)

model_and_history_saver = ModelAndHistorySaver(checkpoint_dir=checkpoint_dir, loss_objects=losses, previous_history=previous_history)


writer = CustomSummaryWriter(log_dir=log_dir, params=params, metrics=metrics, sync_interval=0)
tensorboard_callback = CustomSummaryWriterCallback(writer=writer, include_standard_tensorboard=False, val_dataset=val_dataset,
            log_confusion_matrix=True, confusion_matrix_frequency=1, 
            confusion_matrix_specs=confusion_matrix_specs, input_shape=input_dim, cfg=cfg, loss_objects=losses,
            previous_history=previous_history, tracked_val_metrices=["val_loss", "polyphony_degree_val_loss", "polyphony_degree_class_val_loss"])
# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                 save_weights_only=True,
#                                                 verbose=1,
#                                                 save_freq=num_batches
#                                                 )
#history_saver= HistorySaver(history_path, initial_history=old_history)

callbacks = [tensorboard_callback, model_and_history_saver] #[model_and_history_saver, tensorboard_callback] # TODO: test model_and_history_saver and remove 
if loss_weight_callback := setup_loss_scheduler(objectives_cfg, losses):
    callbacks.append(loss_weight_callback)

# Train model
history = model.fit(train_dataset, 
                    validation_data=val_dataset, 
                    epochs=total_epochs,
                    initial_epoch=initial_epoch, 
                    callbacks=callbacks) #LossWeightScheduler(switch_epochs=[0,10,20,30,40], event_loss_weights=[1.0, 1.0, 1.0, 0.5, 0.1], count_loss_weights=[0.1, 0.5, 1.0, 1.0, 2.0])


# DIAGNOSTIC: Check model state before saving
# print("\n=== BEFORE SAVING ===")
# print(f"Model has {len(model.trainable_variables)} trainable variables")
# print("First few variable values:")
# for i, var in enumerate(model.trainable_variables[:2]):
#     print(f"  {var.path}: mean={tf.reduce_mean(var).numpy():.4f}, std={tf.math.reduce_std(var).numpy():.4f}")

# # Save the model
# model.save(save_model_path)
# print(f"\n✓ Saved model to {save_model_path}")

# # DIAGNOSTIC: Load it back and check
# print("\n=== VERIFYING SAVE ===")
# test_model = tf.keras.models.load_model(save_model_path, compile=False)
# print(f"Loaded model has {len(test_model.trainable_variables)} trainable variables")

# # Initialize the loaded model
# test_sample = next(iter(train_dataset))
# _ = test_model(test_sample[0], training=False)
# print(f"After forward pass: {len(test_model.trainable_variables)} trainable variables")

# if len(test_model.trainable_variables) > 0:
#     print("First few variable values after loading:")
#     for i, var in enumerate(test_model.trainable_variables[:2]):
#         print(f"  {var.path}: mean={tf.reduce_mean(var).numpy():.4f}, std={tf.math.reduce_std(var).numpy():.4f}")
# else:
#     print("WARNING: No variables loaded!")


# Store config in file/logs
print(OmegaConf.to_yaml(cfg))
OmegaConf.save(cfg, os.path.join(log_dir, "params.yaml"))

# Add some examples to tensorboard
import matplotlib.pyplot as plt

# Get some examples
for idx, (split_name, example_idx) in enumerate([
    ('train', 2), 
    #('train', 23), 
    ('validation', 0), 
    #('validation', 23), 
    ('test', 0),
    #('test', 23)
]):
    example = dataset[split_name][example_idx]
    embedding = example[input_feature_name]
    
    # Make prediction
    single_input = np.expand_dims(embedding, axis=0)
    single_input = tf.constant(single_input, dtype=tf.float32)
    predictions = model.predict(single_input)

    # Extract data
    objectives_list = list(objectives_cfg.keys())
    if 'polyphony_degree' in objectives_list:
        gt_polyphony = example['polyphony_degree']
        pred_polyphony = predictions['polyphony_degree'][0][0]
    else:
        gt_polyphony = pred_polyphony = None
        
    if 'event_logits' in objectives_list:
        gt_event_logits = example['event_logits']
        pred_event_logits = predictions['event_logits'][0]
    else:
        gt_event_logits = pred_event_logits = None

    if 'framewise_polyphony' in objectives_list:
        gt_framewise_polyphony = example['framewise_polyphony']
        pred_framewise_polyphony = predictions['framewise_polyphony'][0]
    else:
        gt_framewise_polyphony = pred_framewise_polyphony = None

    # Get audio and events
    audio_array = example['audio']['array']
    sampling_rate = example['audio']['sampling_rate']

    # Get all events
    all_events = []
    for events in example['sources_time_freq_bounds']:
        for event in events:
            all_events.append(event)
    
    # Create combined figure
    fig = plot_spectrogram_with_metrics(
        audio_array=audio_array,
        sampling_rate=sampling_rate,
        split_name=split_name,
        example_idx=example_idx,
        gt_polyphony=gt_polyphony,
        pred_polyphony=pred_polyphony,
        gt_event_logits=gt_event_logits,
        pred_event_logits=pred_event_logits,
        events=None,#all_events,
        filename=example.get('filename', None)
    )
    
    writer.add_figure('test_examples_{idx}', fig, global_step=idx)
    plt.close(fig)

# # Flush to ensure all figures are written
# writer.flush()

# # Create new model
# # Define model
# new_model = instantiate(cfg.model)
# #new_model.build(input_dim)
# new_model.compile(optimizer=Adam(learning_rate), loss=losses)
# # new_model.compile(
# #     optimizer=Adam(learning_rate),
# #     loss={
# #         "perch2_event_logits": weighted_event_loss,
# #         "framewise_polyphony": weighted_frame_loss,
# #         "polyphony_degree": weighted_count_loss, 
# #     },
# # )
# new_model.summary()

# print("New model objectives config:")
# print(new_model.objectives_cfg)

# # Test model without weights
# example = dataset['train'][example_idx]
# embedding = example[input_feature_name]

# # Make prediction
# print("Untrained Model:")
# single_input = np.expand_dims(embedding, axis=0)
# single_input = tf.constant(single_input, dtype=tf.float32)
# predictions = new_model.predict(single_input)
# print(predictions['polyphony_degree'][0][0])
# print("Ground truth: polyphony degree", example['polyphony_degree'])#, ", event logits", example['perch2_event_logits'])

# # Load best weights and test model again
# # try:
# print("Trained Model with best checkpoints:")
# # Get files in best weights folder
# from os import listdir
# from os.path import isfile, join
# best_weights_paths = [f for f in listdir(checkpoint_path) if isfile(join(checkpoint_path, f))]

# new_model.load_weights(best_weights_paths[0]) #save_model_path.replace('.keras', '_best.weights.h5'))
# predictions = new_model.predict(single_input)
# print(predictions['polyphony_degree'][0][0])
# print("Ground truth: polyphony degree", example['polyphony_degree'])#, ", event logits", example['perch2_event_logits'])
# best_model_path = save_model_path.replace('.keras', '_best.keras')
# new_model.save(best_model_path)

# Get best epoch from history
# with open(save_model_path.replace('.keras', '_history.json')) as f:
#     saved_history = json.load(f)
# best_epoch = np.argmin(saved_history['val_loss']) + 1
# print(f"✓ Saved model with best weights from epoch {best_epoch} [val_loss: {saved_history['val_loss'][best_epoch - 1]}] to {best_model_path}.")
# except:
#     pass

# # TODO: needs register_keras_serializable() for losses
# try:
#     print("Trained saved full model")
#     full_model = tf.keras.models.load_model(save_model_path)
#     predictions = full_model.predict(single_input)
#     print("Full model objectives config:")
#     print(full_model.objectives_cfg)

#     print("Full model predicitions")
#     print(predictions['polyphony_degree'][0][0])
#     print("Ground truth: polyphony degree", example['polyphony_degree'])#, ", event logits", example['perch2_event_logits'])

#     print("Model history:")
#     history_path = save_model_path.replace('.keras', '_history.json')

#     with open(history_path, 'r') as f:
#         history = json.load(f)

#     for metric, values in history.items():
#         for epoch, value in enumerate(values, start=1):
#             print(f"Epoch {epoch:03d} | {metric}: {value:.4f}")
    
    
# except:
#     pass

dataset.cleanup_cache_files()

