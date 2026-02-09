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

from utils.logs import plot_spectrogram_with_metrics, return_checkpoint_path, return_tensorboard_dir, CustomSummaryWriter, CustomSummaryWriterCallback, build_confusion_matrix_specs
from utils.general import reshape_tensor_data
from utils.config import set_random_seeds, Params
from losses import create_losses_from_objectives, setup_loss_scheduler

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

# Setup tensorboard
# If defined in cfg use tensorboard path (define it in params.yaml for debugging purposes)
# else use return_tensorboard_path (default with dvc run)
# def get_tensorboard_path(cfg):
#     if 'tensorboard_path' in cfg.train.keys():
#         default_dir = os.getcwd()
#         #dvc_exp_name = 'debug'
#         current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M")
#         os.environ['DEFAULT_DIR'] = default_dir
#         tensorboard_subfolder = f'{cfg.dataset.subset}/{cfg.train.input_feature_name}'
#         tensorboard_path_suffix = f'_{input_feature_name}'

#         tensorboard_path = Path(
#             f"{default_dir}/{cfg.train.tensorboard_path}/{tensorboard_subfolder}/{tensorboard_path_suffix}{current_datetime}"
#         )     
#     else:
#         tensorboard_subfolder = f'{cfg.dataset.subset}'
#         tensorboard_path_suffix = f'_{input_feature_name}'
#         tensorboard_path = return_tensorboard_dir(subfolder=tensorboard_subfolder, suffix=tensorboard_path_suffix) # './logs/' + features + version #return_tensorboard_path()
#     os.makedirs(tensorboard_path, exist_ok=True)
#     return tensorboard_path

# class LossWeightScheduler(tf.keras.callbacks.Callback):
#     def __init__(
#         self,
#         switch_epochs,
#         event_loss_weights,
#         frame_loss_weights,
#         count_loss_weights,
#     ):
#         super().__init__()
#         self.switch_epochs = switch_epochs
#         self.event_loss_weights = event_loss_weights
#         self.frame_loss_weights = frame_loss_weights
#         self.count_loss_weights = count_loss_weights

#     def on_epoch_begin(self, epoch, logs=None):
#         for e, ew, fw, cw in zip(
#             self.switch_epochs,
#             self.event_loss_weights,
#             self.frame_loss_weights,
#             self.count_loss_weights,
#         ):
#             if epoch == e:
#                 event_loss_weight.assign(ew)
#                 frame_loss_weight.assign(fw)
#                 count_loss_weight.assign(cw)

#                 print(
#                     f"\n[LossWeightScheduler] epoch {epoch} | "
#                     f"event={ew}, frame={fw}, count={cw}"
#                 )

# Configuration
cfg = OmegaConf.load("params.yaml")

# Load the hyperparameters from the "params.yaml" file for usage with Tensorboard SummaryWriter
params = Params()

random_seed = cfg.general.random_seed
set_random_seeds(random_seed)

dataset_path =  cfg.path.dataset

experiment_name = cfg.log.experiment_name

input_feature_name = cfg.train.input_feature_name
epochs = cfg.train.epochs
learning_rate = cfg.train.learning_rate
batch_size = cfg.train.batch_size

if 'train_size_batches' in cfg.train: 
    train_size_batches = cfg.train.train_size_batches
else:
    train_size_batches = None

if 'val_size_batches' in cfg.train: 
    val_size_batches = cfg.train.val_size_batches
else:
    val_size_batches = None

model_cfg = cfg.model
objectives_cfg = cfg.objectives
objectives_list = list(objectives_cfg.keys()) 

# Add objectives to the config
model_cfg.objectives = objectives_list

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
train_dataset, test_dataset, val_dataset = get_tf_datasets(dataset, input_feature_name, objectives_list, batch_size)
if train_size_batches: train_dataset = train_dataset.take(train_size_batches) # take fewer batches to reduce train dataset size
if val_size_batches: val_dataset = val_dataset.take(val_size_batches)


# Create a SummaryWriter object to write the tensorboard logs
#metrics = {'loss': None, 'val_loss': None, 'mae': None, 'val_mae': None} # TODO: Investigate: What is this doing with the variable losses?
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

# # Define loss weights as variables
# event_loss_weight = tf.Variable(1.0, trainable=False, dtype=tf.float32)
# count_loss_weight = tf.Variable(0.3, trainable=False, dtype=tf.float32)
# frame_loss_weight = tf.Variable(1.0, trainable=False, dtype=tf.float32)

# bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# mse = tf.keras.losses.MeanSquaredError()
# frame_mse = tf.keras.losses.MeanSquaredError()

# def weighted_event_loss(y_true, y_pred):
#     return event_loss_weight * bce(y_true, y_pred)

# def weighted_count_loss(y_true, y_pred):
#     return count_loss_weight * mse(y_true, y_pred) 

# def weighted_frame_loss(y_true, y_pred):
#     return frame_loss_weight * frame_mse(y_true, y_pred)

# # Merge objectives into model config before instantiation
# model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
# model_cfg['objectives'] = list(cfg.objectives.keys())
# print(model_cfg)
# print(model_cfg['objectives'])



# Instantiate model with merged config
model = instantiate(model_cfg)
model.build(input_dim)
print(f'Input shape of model: {input_dim}')

# Define confusion matrix specs
# confusion_matrix_specs = [
#         {"name": "polyphony_degree", "type": "regression_round", "threshold": 0.5}, 
#         {"name": "perch2_event_logits", "type": "binary", "threshold": 0.5,},
#     ]
# confusion_matrix_specs = get_confusion_matrix_specs(cfg)
# Build confusion matrix specs
confusion_matrix_specs = build_confusion_matrix_specs(objectives_cfg)

checkpoint_path = return_checkpoint_path(subfolder=f'{experiment_name}_{input_feature_name}')
print("Checkpoint path:", checkpoint_path)
checkpoint_dir = os.path.dirname(checkpoint_path)

num_batches = len(train_dataset)

losses = create_losses_from_objectives(objectives_cfg)  

writer = CustomSummaryWriter(log_dir=tensorboard_path, params=params, metrics=metrics, sync_interval=0)
tensorboard_callback = CustomSummaryWriterCallback(writer=writer, include_standard_tensorboard=True, val_dataset=val_dataset, 
            log_confusion_matrix=True, confusion_matrix_frequency=5, confusion_matrix_specs=confusion_matrix_specs, input_shape=input_dim, cfg=cfg, loss_objects=losses)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1,
                                                save_freq=5*num_batches
                                                )
callbacks = [tensorboard_callback, checkpoint_callback]
if loss_weight_callback := setup_loss_scheduler(objectives_cfg, losses):
    callbacks.append(loss_weight_callback)


# loss_weight_callback = LossWeightScheduler(switch_epochs=[0, 10, 20, 30, 40],
#                                        event_loss_weights=[10.0, 5.0, 1.0, 0.5, 0.1],
#                                        frame_loss_weights=[5.0, 10.0, 5.0, 1.0, 0.5],
#                                        count_loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0])
# Create losses from objectives

# Train model
# losses = create_losses_from_model(model, cfg)
model.compile(optimizer=Adam(learning_rate), loss=losses) # TODO: use optimizer=instantiate(cfg.train.optimizer/optimizer_cfg)
# model.compile(
#     optimizer=Adam(learning_rate),
#     loss={
#         "perch2_event_logits": weighted_event_loss,
#         "framewise_polyphony": weighted_frame_loss,
#         "polyphony_degree": weighted_count_loss, 
#     },
# )
model.summary()

history = model.fit(train_dataset, 
                    validation_data=val_dataset, 
                    epochs=epochs, 
                    callbacks=callbacks) #LossWeightScheduler(switch_epochs=[0,10,20,30,40], event_loss_weights=[1.0, 1.0, 1.0, 0.5, 0.1], count_loss_weights=[0.1, 0.5, 1.0, 1.0, 2.0])

# TODO: needs register_keras_serializable() for losses
model_path = f'models/{input_feature_name}.keras'
model.save(model_path)

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

# Load weights and test model again
try:
    print("Trained Model from loaded checkpoints:")
    new_model.load_weights(checkpoint_path)
    predictions = new_model.predict(single_input)
    print(predictions['polyphony_degree'][0][0])
    print("Ground truth: polyphony degree", example['polyphony_degree'])#, ", event logits", example['perch2_event_logits'])
except:
    pass

# TODO: needs register_keras_serializable() for losses
try:
    print("Trained saved full model")
    full_model = tf.keras.models.load_model(model_path)
    predictions = full_model.predict(single_input)
    print(predictions['polyphony_degree'][0][0])
    print("Ground truth: polyphony degree", example['polyphony_degree'])#, ", event logits", example['perch2_event_logits'])
except:
    pass

dataset.cleanup_cache_files()

