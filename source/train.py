import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from datasets import concatenate_datasets
from omegaconf import OmegaConf
import os
from hydra.utils import instantiate
from dotenv import load_dotenv
import json
from datetime import datetime

from utils.logs import plot_spectrogram_with_metrics, return_tensorboard_dir, CustomSummaryWriter, CustomSummaryWriterCallback, build_confusion_matrix_specs, ModelAndHistorySaver, get_dvc_exp_name
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

def main():

    # Configuration
    cfg = OmegaConf.load("params.yaml")

    # Load the hyperparameters from the "params.yaml" file for usage with Tensorboard SummaryWriter
    params = Params()

    random_seed = cfg.general.random_seed
    set_random_seeds(random_seed)

    # dataset_path =  cfg.path.dataset
    huggingface_path = cfg.dataset.huggingface_path
    dataset_config = cfg.dataset.config
    val_dataset_path = cfg.path.val_dataset
    log_dir = cfg.path.log_dir
    checkpoint_dir = cfg.path.checkpoint_dir

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(val_dataset_path, exist_ok=True)

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

    model_cfg = cfg.model
    objectives_cfg = cfg.objectives

    model_cfg.objectives_cfg = objectives_cfg

    labels = [objectives_cfg[x]['label'] for x in objectives_cfg]

    print(f"Training with {input_feature_name} as input feature and {labels} as labels on dataset {huggingface_path} with config {dataset_config}.")

    #################################
    # Load dataset
    #################################

    # Load environment variables from .env file
    load_dotenv('local.env')
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
    dataset = load_dataset_with_retry(huggingface_path, dataset_config, token=huggingface_token)

    if dataset is None:
        raise RuntimeError("Dataset failed to load after all retry attempts. Check network/cache or force redownload in dataset preparation.")

    #################################
    # Add labels
    #################################

    # Get input dim
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
    
    ###########################################
    # Transform dataset to tensorflow datasets 
    ###########################################

    # Get tensorflow datasets
    train_dataset, test_dataset, val_dataset = get_tf_datasets(dataset, input_feature_name, labels, batch_size)
    if num_batches_train: train_dataset = train_dataset.take(num_batches_train) # take fewer batches to reduce train dataset size
    if num_batches_val: val_dataset = val_dataset.take(num_batches_val)
    train_size = num_batches_train * batch_size if num_batches_train else len(dataset['train'])
    val_size = num_batches_val * batch_size if num_batches_val else len(dataset['validation'])

    train_dataset = train_dataset #.cache().prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset #.cache().prefetch(tf.data.AUTOTUNE)

    # Save validiation dataset for later evaluation
    val_dataset.save(val_dataset_path)

    #################################
    # Model
    #################################
    tf.keras.backend.clear_session()

    # Create loss objects based on objectives config
    losses = create_losses_from_objectives(objectives_cfg) 

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
    # Logging setup
    #################################

    # dvc_exp_name = get_dvc_exp_name()
    # current_datetime = datetime.now().strftime("%Y%m%d-%H%M")

    # # Get tensorboard path based on path, dataset subset, features and datetime
    # # Set DEFAULT_DIR if not set (usually when running without dvc)
    # os.environ.setdefault('DEFAULT_DIR', os.getcwd())
    # os.environ.setdefault('DVC_EXP_NAME', 'test-experiment')

    # # checkpoint_dir = f'train_output/{study_name}/{current_datetime}_{dvc_exp_name}/checkpoints' #f'checkpoints/{study_name}/{current_datetime}_{dvc_exp_name}_{path_suffix}' if path_suffix else
    # # log_dir = f'train_output/{study_name}/{current_datetime}_{dvc_exp_name}/logs'

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

    #################################
    # Callbacks
    #################################

    early_stopping = EarlyStopping(
        monitor='val_loss',  
        patience=5,            
        restore_best_weights=False
    )             

    model_and_history_saver = ModelAndHistorySaver(checkpoint_dir=checkpoint_dir, loss_objects=losses, previous_history=previous_history)

    writer = CustomSummaryWriter(log_dir=log_dir, params=params, metrics=metrics, sync_interval=0)
    tensorboard_callback = CustomSummaryWriterCallback(writer=writer, include_standard_tensorboard=False, val_dataset=val_dataset,
                log_confusion_matrix=True, confusion_matrix_frequency=1, 
                confusion_matrix_specs=confusion_matrix_specs, input_shape=input_dim, cfg=cfg, loss_objects=losses,
                previous_history=previous_history, tracked_val_metrices=["val_loss", "polyphony_degree_val_loss", "polyphony_degree_class_val_loss"])

    callbacks = [tensorboard_callback, model_and_history_saver, early_stopping] #[model_and_history_saver, tensorboard_callback] # TODO: test model_and_history_saver and remove 
    if loss_weight_callback := setup_loss_scheduler(objectives_cfg, losses):
        callbacks.append(loss_weight_callback)

    #################################
    # Train model
    #################################

    history = model.fit(train_dataset, 
                        validation_data=val_dataset, 
                        epochs=total_epochs,
                        initial_epoch=initial_epoch, 
                        callbacks=callbacks) #LossWeightScheduler(switch_epochs=[0,10,20,30,40], event_loss_weights=[1.0, 1.0, 1.0, 0.5, 0.1], count_loss_weights=[0.1, 0.5, 1.0, 1.0, 2.0])

if __name__=="__main__":
     main()

