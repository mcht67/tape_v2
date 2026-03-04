# import matplotlib
# matplotlib.use("Agg")

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from datasets import load_from_disk, concatenate_datasets
from omegaconf import OmegaConf
import os
import model
from hydra.utils import instantiate
import pickle
from datetime import datetime

from utils.logs import plot_spectrogram_with_metrics, return_tensorboard_dir, CustomSummaryWriter, CustomSummaryWriterCallback, build_confusion_matrix_specs, get_dvc_exp_name
from utils.general import reshape_tensor_data
from utils.config import set_random_seeds, Params
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

def sanitize_objectives_cfg(cfg, keys_to_keep):
    """Strip non-serializable objects (loss fns, omegaconf) from objectives_cfg."""
    import omegaconf
    
    serializable = {}
    KEEP_KEYS = keys_to_keep#{"num_classes", "type", "threshold", "label", "weight"}
    
    for obj_name, obj_cfg in cfg.items():
        # Convert omegaconf DictConfig to plain dict
        if isinstance(obj_cfg, omegaconf.DictConfig):
            obj_cfg = omegaconf.OmegaConf.to_container(obj_cfg, resolve=True)
        
        # Keep only JSON-serializable metadata, drop loss objects
        clean = {}
        # for k, v in obj_cfg.items():
        #     if k in KEEP_KEYS and isinstance(v, (str, int, float, bool, type(None))):
        #         clean[k] = v
        #     elif k == "confusion_matrix" and isinstance(v, dict):
        #         clean[k] = {ck: cv for ck, cv in v.items() 
        #                     if isinstance(cv, (str, int, float, bool, type(None)))}
        for k, v in obj_cfg.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                clean[k] = v
            elif isinstance(v, dict):
                clean[k] = {ck: cv for ck, cv in v.items() 
                            if isinstance(cv, (str, int, float, bool, type(None)))}
            elif isinstance(v, list):
                clean[k] = [cv for cv in v
                            if isinstance(cv, (str, int, float, bool, type(None)))]

        serializable[obj_name] = clean
    
    return serializable

# Configuration
cfg = OmegaConf.load("params.yaml")

# Load the hyperparameters from the "params.yaml" file for usage with Tensorboard SummaryWriter
params = Params()

random_seed = cfg.general.random_seed
set_random_seeds(random_seed)

dataset_path =  cfg.path.dataset

experiment_name = cfg.log.experiment_name
load_model_path = None
# load_checkpoint_path = None
load_history_path = None

input_feature_name = cfg.train.input_feature_name
total_epochs = cfg.train.epochs
initial_epoch = 0
learning_rate = cfg.train.learning_rate
batch_size = cfg.train.batch_size
train_size_batches = 1 #None
val_size_batches = 1 #None

if 'initial_epoch' in cfg.train:
    initial_epoch = cfg.train.initial_epoch    

if 'train_size_batches' in cfg.train: 
    train_size_batches = 1 #cfg.train.train_size_batches 

if 'val_size_batches' in cfg.train: 
    val_size_batches = 1 #cfg.train.val_size_batches

if 'load_model_path' in cfg.train:
    load_model_path = cfg.train.load_model_path

# if 'load_checkpoint_path' in cfg.train:
#     load_checkpoint_path = cfg.train.load_checkpoint_path

if 'load_history_path' in cfg.train:
    load_history_path = cfg.train.load_history_path

model_cfg = cfg.model
objectives_cfg = cfg.objectives
#objectives_dict =  OmegaConf.to_container(objectives_cfg, resolve=True)

for x in objectives_cfg:
    print(objectives_cfg[x]['label'])
labels = [objectives_cfg[x]['label'] for x in objectives_cfg]

# Add objectives to the config
model_cfg.objectives_cfg = objectives_cfg

tensorboard_subfolder = cfg.log.tensorboard_subfolder
tensorboard_suffix = cfg.log.tensorboard_suffix

# Get tensorboard path based on path, dataset subset, features and datetime
# Set DEFAULT_DIR if not set (usually when running without dvc)
os.environ.setdefault('DEFAULT_DIR', os.getcwd())
os.environ.setdefault('DVC_EXP_NAME', 'test-experiment')

tensorboard_path = return_tensorboard_dir(subfolder=experiment_name) #get_tensorboard_path(cfg) #TODO: refactor to work in a similar manner with dvc and without
os.makedirs(tensorboard_path, exist_ok=True)

# Load dataset
dataset = load_from_disk(dataset_path)

# Get input dim
embeddings = dataset['train'][0][input_feature_name]
input_dim = tf.squeeze(np.array(dataset['train'][0][input_feature_name])).shape

# Get tensorflow datasets
train_dataset, test_dataset, val_dataset = get_tf_datasets(dataset, input_feature_name, labels, batch_size)
if train_size_batches: train_dataset = train_dataset.take(train_size_batches) # take fewer batches to reduce train dataset size
if val_size_batches: val_dataset = val_dataset.take(val_size_batches)

metrics = {}
for key in objectives_cfg.keys():
    metrics[f"{key}_loss"] = None
    metrics[f"val_{key}_loss"] = None

print("Metrics:", metrics)

params['dataset']['train_size'] = str(len(dataset['train']))
params['dataset']['val_size'] = str(len(dataset['validation']))
params['dataset']['test_size'] = str(len(dataset['test']))
params['train']['objectives'] = list(cfg.objectives.keys())
print(params)

confusion_matrix_specs = build_confusion_matrix_specs(objectives_cfg)

dvc_exp_name = get_dvc_exp_name()
current_datetime = datetime.now().strftime("%Y%m%d-%H%M")

# checkpoint_path = f"logs/models/{experiment_name}/{current_datetime}_{dvc_exp_name}_{input_feature_name}.weights.h5" #return_checkpoint_path(subfolder=f'{experiment_name}_{input_feature_name}')
# # TEMPORARY checkpoint solution should be handled by resuming experiment later TODO: 
# checkpoint_dir = os.path.dirname(checkpoint_path)
# # checkpoint_names = os.listdir(checkpoint_dir)
# # checkpoint_path = os.path.join(checkpoint_dir, checkpoint_names[0])
# print("Checkpoint path:", checkpoint_path)

num_batches = len(train_dataset) 

# model_path = f'models/{input_feature_name}_large.keras'

save_model_path = f'logs/models/{experiment_name}/{dvc_exp_name}_{input_feature_name}_large.keras'
# history_path = save_model_path.replace('.keras', '_history.pkl')

losses = create_losses_from_objectives(objectives_cfg) 

##############
# Model
###############
tf.keras.backend.clear_session()

# Get model and history
if load_model_path and os.path.isfile(load_model_path): 
    print("Loading model from", load_model_path)
    model = tf.keras.models.load_model(load_model_path)

    # Load previous history
    if load_history_path and os.path.isfile(load_history_path):
        with open(load_history_path, 'rb') as f:
            old_history = pickle.load(f)
        initial_epoch = len(old_history['loss'])
        print(f"Resuming from epoch {initial_epoch}")
    else:
        old_history = None
        print(f"No history found, starting from epoch {initial_epoch}")

# elif load_checkpoint_path and os.path.isfile(load_checkpoint_path):

#     print(f"Loading weights from {load_checkpoint_path}")
#     model = instantiate(cfg.model)
#     #this_model.build(input_dim)
#     model.compile(optimizer=Adam(learning_rate), loss=losses)
    
#     # Initialize variables with forward pass
#     sample_batch = next(iter(train_dataset))
#     _ = model(sample_batch[0], training=False)

#     model.load_weights(load_checkpoint_path)
    
    # # Load previous history
    # if load_history_path and os.path.isfile(load_history_path):
    #     with open(load_history_path, 'rb') as f:
    #         old_history = pickle.load(f)
    #     initial_epoch = len(old_history['loss'])
    #     print(f"Resuming from epoch {initial_epoch}")
    # else:
    #     old_history = None
    #     print(f"No history found, starting from epoch {initial_epoch}")
else:
    print("Creating new model")
    model = instantiate(model_cfg)
    
    # Initialize new model with forward pass
    sample_batch = next(iter(train_dataset))
    _ = model(sample_batch[0], training=False)
    print(f"New model has {len(model.trainable_variables)} trainable variables")

    model.compile(optimizer=Adam(learning_rate), loss=losses)
    
    old_history = None
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
    def __init__(self, model_path, initial_history=None, save_every_n_epochs=5, keep_last_n=None):
        super().__init__()
        self.model_path = model_path
        self.history_path = model_path.replace('.keras', '_history.pkl')
        # self.checkpoint_path = model_path.replace('.keras', '_history.pkl')
        self.combined_history = {k: list(v) for k, v in initial_history.items()} \
                                 if initial_history else {}
        self.save_every_n_epochs = save_every_n_epochs
        self.keep_last_n = keep_last_n
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        # Always update history in-memory
        for key, value in logs.items():
            self.combined_history.setdefault(key, []).append(float(value))

        # Save model and history together (always in sync)
        if (epoch + 1) % self.save_every_n_epochs == 0:
            self.model.save(self.model_path)
            with open(self.history_path, 'w') as f:
                json.dump(self.combined_history, f, indent=2)
            print(f"✓ Saved model and history at epoch {epoch + 1}")

        # Save best checkpoint
        val_loss = logs.get('val_loss')
        if val_loss and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.model.save_weights(self.model_path.replace('.keras', '_chkpt_best.weights.h5'))
            print(f"✓ New best val_loss {val_loss:.4f} at epoch {epoch + 1}")

        # Save rolling last-N checkpoints
        if self.keep_last_n:
            self.model.save_weights(self.model_path.replace('.keras', f'_ckpt-epoch{epoch+1:03d}.weights.h5'))
            self._cleanup_old_checkpoints(epoch)

    def _cleanup_old_checkpoints(self, current_epoch):
        for old_epoch in range(current_epoch - self.keep_last_n):
            path = self.model_path.replace('.keras', f'_ckpt-epoch{old_epoch+1:03d}.weights.h5')
            if os.path.exists(path):
                os.remove(path)

model_and_history_saver = ModelAndHistorySaver(
    model_path=save_model_path,
    # history_path=history_path,
    initial_history=old_history,
    save_every_n_epochs=1,
    keep_last_n=10
)


writer = CustomSummaryWriter(log_dir=tensorboard_path, params=params, metrics=metrics, sync_interval=0)
tensorboard_callback = CustomSummaryWriterCallback(writer=writer, include_standard_tensorboard=False, val_dataset=val_dataset,
            log_confusion_matrix=True, confusion_matrix_frequency=1, confusion_matrix_specs=confusion_matrix_specs, input_shape=input_dim, cfg=cfg, loss_objects=losses)
# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                 save_weights_only=True,
#                                                 verbose=1,
#                                                 save_freq=num_batches
#                                                 )
#history_saver= HistorySaver(history_path, initial_history=old_history)

callbacks = [model_and_history_saver, tensorboard_callback] # TODO: test model_and_history_saver and remove 
if loss_weight_callback := setup_loss_scheduler(objectives_cfg, losses):
    callbacks.append(loss_weight_callback)

# Train model
history = model.fit(train_dataset, 
                    validation_data=val_dataset, 
                    epochs=total_epochs,
                    initial_epoch=initial_epoch, 
                    callbacks=callbacks) #LossWeightScheduler(switch_epochs=[0,10,20,30,40], event_loss_weights=[1.0, 1.0, 1.0, 0.5, 0.1], count_loss_weights=[0.1, 0.5, 1.0, 1.0, 2.0])


# DIAGNOSTIC: Check model state before saving
print("\n=== BEFORE SAVING ===")
print(f"Model has {len(model.trainable_variables)} trainable variables")
print("First few variable values:")
for i, var in enumerate(model.trainable_variables[:2]):
    print(f"  {var.path}: mean={tf.reduce_mean(var).numpy():.4f}, std={tf.math.reduce_std(var).numpy():.4f}")

# Save the model
model.save(save_model_path)
print(f"\n✓ Saved model to {save_model_path}")

# DIAGNOSTIC: Load it back and check
print("\n=== VERIFYING SAVE ===")
test_model = tf.keras.models.load_model(save_model_path, compile=False)
print(f"Loaded model has {len(test_model.trainable_variables)} trainable variables")

# Initialize the loaded model
test_sample = next(iter(train_dataset))
_ = test_model(test_sample[0], training=False)
print(f"After forward pass: {len(test_model.trainable_variables)} trainable variables")

if len(test_model.trainable_variables) > 0:
    print("First few variable values after loading:")
    for i, var in enumerate(test_model.trainable_variables[:2]):
        print(f"  {var.path}: mean={tf.reduce_mean(var).numpy():.4f}, std={tf.math.reduce_std(var).numpy():.4f}")
else:
    print("WARNING: No variables loaded!")

# TODO: needs register_keras_serializable() for losses
#model.save(save_model_path)

# TODO: Store config in file/logs
print(OmegaConf.to_yaml(cfg))
OmegaConf.save(cfg, os.path.join(tensorboard_path, "params.yaml"))

# Add some examples to tensorboard
import matplotlib.pyplot as plt

# Get some examples
for idx, (split_name, example_idx) in enumerate([
    ('train', 2), 
    ('train', 23), 
    ('validation', 2), 
    ('validation', 23), 
    ('test', 2),
    ('test', 23)
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
    for events in example['raw_files_time_freq_bounds']:
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

# Flush to ensure all figures are written
writer.flush()

# Create new model
# Define model
new_model = instantiate(cfg.model)
new_model.build(input_dim)
new_model.compile(optimizer=Adam(learning_rate), loss=losses)
# new_model.compile(
#     optimizer=Adam(learning_rate),
#     loss={
#         "perch2_event_logits": weighted_event_loss,
#         "framewise_polyphony": weighted_frame_loss,
#         "polyphony_degree": weighted_count_loss, 
#     },
# )
new_model.summary()

# Test model without weights
example = dataset['train'][example_idx]
embedding = example[input_feature_name]

# Make prediction
print("Untrained Model:")
single_input = np.expand_dims(embedding, axis=0)
single_input = tf.constant(single_input, dtype=tf.float32)
predictions = new_model.predict(single_input)
print(predictions['polyphony_degree'][0][0])
print("Ground truth: polyphony degree", example['polyphony_degree'])#, ", event logits", example['perch2_event_logits'])

# Load best weights and test model again
try:
    print("Trained Model with best checkpoints:")
    new_model.load_weights(save_model_path.replace('.keras', '_chkpt_best.weights.h5'))
    predictions = new_model.predict(single_input)
    print(predictions['polyphony_degree'][0][0])
    print("Ground truth: polyphony degree", example['polyphony_degree'])#, ", event logits", example['perch2_event_logits'])
except:
    pass

# TODO: needs register_keras_serializable() for losses
try:
    print("Trained saved full model")
    full_model = tf.keras.models.load_model(save_model_path)
    predictions = full_model.predict(single_input)
    print("Model history:")
    history_path = save_model_path.replace('.keras', '_history.pkl')

    with open(history_path, 'r') as f:
        history = json.load(f)

    for metric, values in history.items():
        for epoch, value in enumerate(values, start=1):
            print(f"Epoch {epoch:03d} | {metric}: {value:.4f}")
    
    print(predictions['polyphony_degree'][0][0])
    print("Ground truth: polyphony degree", example['polyphony_degree'])#, ", event logits", example['perch2_event_logits'])
except:
    pass

dataset.cleanup_cache_files()

