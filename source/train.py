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

from utils.logs import RegressionAccuracy, RegressionCountPrecision, RegressionCountRecall, RegressionCountF1, ClassificationAccuracy, ClassificationCountPrecision, ClassificationCountRecall, ClassificationCountF1, CustomSummaryWriter, CustomSummaryWriterCallback, build_confusion_matrix_specs, ModelAndHistorySaver, get_log_paths
from utils.general import reshape_tensor_data
from utils.config import set_random_seeds, Params
from utils.dataset import add_labels, load_dataset_with_retry, get_birdset_id2label
from losses import create_losses_from_objectives, setup_loss_scheduler, compute_species_count_class_weights

tf.keras.backend.clear_session()

# def make_tf_dataset(hf_dataset, features, labels, batch_size, shuffle=False):
#     X = np.stack(hf_dataset[features]).astype(np.float32)
#     y = np.array(hf_dataset[labels]).astype(np.float32)

#     ds = tf.data.Dataset.from_tensor_slices((X, y))
#     if shuffle:
#         ds = ds.shuffle(buffer_size=len(X))
#     ds = ds.batch(batch_size)
#     return ds

def get_tf_datasets(dataset, features, labels, batch_size):
    # train_dataset = dataset['train'].to_tf_dataset(
    #     columns=features,
    #     label_cols=labels, 
    #     batch_size=batch_size,
    #     shuffle=True,
    #     prefetch=False
    # )

    # test_dataset = dataset['test'].to_tf_dataset(
    #     columns=features,
    #     label_cols=labels,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     prefetch=False
    # )

    # val_dataset = dataset['validation'].to_tf_dataset(
    #     columns=features,
    #     label_cols=labels,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     prefetch=False
    # )
    train_dataset = get_tf_dataset_from_split(dataset, 'train', features, labels, batch_size, shuffle=True)
    val_dataset = get_tf_dataset_from_split(dataset, 'validation', features, labels, batch_size, shuffle=False)
    test_dataset = get_tf_dataset_from_split(dataset, 'test', features, labels, batch_size, shuffle=False)

    return train_dataset, test_dataset, val_dataset

# def get_tf_dataset_from_split(dataset, split_name, features, labels, batch_size, shuffle=False):
#     if split_name not in dataset:
#         raise ValueError(f"Split {split_name} not found in dataset. Available splits: {dataset.keys()}")
    
#     # Keep only the columns actually needed
#     cols_to_keep = set(features if isinstance(features, list) else [features]) | set(labels)
#     cols_to_remove = [c for c in dataset[split_name].column_names if c not in cols_to_keep]
#     split = dataset[split_name].remove_columns(cols_to_remove)
    
#     return split.to_tf_dataset(
#         columns=features,
#         label_cols=labels,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         prefetch=False
#     )

def get_tf_dataset_from_split(dataset, split_name, features, labels, batch_size, shuffle=False):
    if split_name not in dataset:
        raise ValueError(f"Split {split_name} not found in dataset. Available splits: {dataset.keys()}")

    # Keep only the columns actually needed
    feature_list = features if isinstance(features, list) else [features]
    cols_to_keep = set(feature_list) | set(labels)
    cols_to_remove = [c for c in dataset[split_name].column_names if c not in cols_to_keep]
    split = dataset[split_name].remove_columns(cols_to_remove)

    # Remove duplicated labels if any
    labels = list(set(labels))

    # Check for None values and report which columns are affected
    none_cols = []
    for col in cols_to_keep:
        if any(v is None for v in split[col]):
            none_cols.append(col)
    
    print("None columns: ", none_cols)

    if none_cols:
        print(f"[WARNING] Columns with None values found: {none_cols}. Filtering out affected rows.")
        split = split.filter(lambda row: all(row[col] is not None for col in none_cols))
        print(f"[INFO] Remaining rows after filtering: {len(split)}")

    return split.to_tf_dataset(
        columns=features,
        label_cols=labels,
        batch_size=batch_size,
        shuffle=shuffle,
        prefetch=False
    )

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

# TODO: remove after handling in model output processing
def reshape_to_tfe(example, input_feature_name):
    """
    Convert audio spatial embeddings to (time, freq, embedding).

    Returns:
        example with transformed embedding stored back into the same field.
    """

    x = np.asarray(example[input_feature_name])
    if x.ndim == 0:
        return example

    # Models with explicit (embedding, freq, time)
    if input_feature_name in {
        "EfficientNet-B1-BirdSet-XCL_audio_spatial_embeddings",
        "AudioProtoPNet-20-BirdSet-XCL_audio_spatial_embeddings",
        "EfficientNet-B1-BirdSet-XCL_no_noise_audio_spatial_embeddings",
        "AudioProtoPNet-20-BirdSet-XCL_no_noise_audio_spatial_embeddings",
    }:
        # (C, F, T) -> (T, F, C)
        x = np.transpose(x, (2, 1, 0))

    # Perch models already have (T, F, C)
    elif input_feature_name in {
        "perch_v2_cpu_audio_spatial_embeddings",
        "perch_v2_cpu_no_noise_audio_spatial_embeddings",
    }:
        pass

    # Sequence models: (T, C) -> (T, 1, C)
    elif input_feature_name in {
        "yamnet_audio_spatial_embeddings",
        "vggish_audio_spatial_embeddings",
        "Wav2Vec2-Base-BirdSet-XCL_audio_spatial_embeddings",
        "beans_baseline_audio_spatial_embeddings",
        "yamnet_no_noise_audio_spatial_embeddings",
        "vggish_no_noise_audio_spatial_embeddings",
        "Wav2Vec2-Base-BirdSet-XCL_no_noise_audio_spatial_embeddings",
        "beans_baseline_no_noise_audio_spatial_embeddings",
    }:
        x = x[:, None, :]

    # AST patch tokens
    elif input_feature_name in {
        "AST-Birdset-XCL_audio_spatial_embeddings",
        "AST-Birdset-XCL_no_noise_audio_spatial_embeddings",
    }:

        x = x[:, None, :]
    # elif input_feature_name == "AST-Birdset-XCL_audio_spatial_embeddings":
    #     #
    #     # Expected shape approximately:
    #     #   (1214, 768)
    #     # where first token is CLS.
    #     #
    #     # Remove CLS and reconstruct patch grid if possible.
    #     #
    #     if x.ndim != 2:
    #         raise ValueError(
    #             f"Unexpected AST shape {x.shape}"
    #         )

    #     tokens = x[1:]  # remove CLS

    #     n_tokens, emb_dim = tokens.shape

    #     # Standard BirdSet AST typically yields:
    #     # 1214 total tokens = 1 CLS + 1213 patches
    #     #
    #     # 1213 = 17 * 71
    #     #
    #     if n_tokens == 1213:
    #         x = tokens.reshape(71, 17, emb_dim)
    #     else:
    #         raise ValueError(
    #             f"Cannot infer AST patch grid from shape {x.shape}"
            # )

    else:
        raise ValueError(
            f"Unsupported feature name: {input_feature_name}"
        )

    example[input_feature_name] = x.astype(np.float32)

    return example

# def build_metrics(objectives_cfg):
#     compile_metrics = {}
#     log_metrics = {}
#     multiple_objectives = len(objectives_cfg) > 1

#     def metric_key(objective, metric_name):
#         if multiple_objectives:
#             return f"val_{objective}_{metric_name}"
#         return f"val_{metric_name}"

#     for objective, obj_cfg in objectives_cfg.items():
#         if objective == "polyphony_reg":
#             compile_metrics[objective] = RegressionAccuracy(name='accuracy')
#             log_metrics[metric_key(objective, 'accuracy')] = None
#             log_metrics[metric_key(objective, 'loss')] = None

#         elif objective == "polyphony_class":
#             compile_metrics[objective] = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
#             log_metrics[metric_key(objective, 'accuracy')] = None
#             log_metrics[metric_key(objective, 'loss')] = None

#         elif objective == "binary":
#             compile_metrics[objective] = tf.keras.metrics.BinaryAccuracy(name='accuracy')
#             log_metrics[metric_key(objective, 'accuracy')] = None
#             log_metrics[metric_key(objective, 'loss')] = None

#         elif objective == 'event_logits':
#             compile_metrics[objective] = tf.keras.metrics.BinaryAccuracy(name='accuracy')
#             log_metrics[metric_key(objective, 'accuracy')] = None
#             log_metrics[metric_key(objective, 'loss')] = None

#         elif objective == 'framewise_polyphony_reg':
#             compile_metrics[objective] = RegressionAccuracy(name='accuracy')
#             log_metrics[metric_key(objective, 'accuracy')] = None
#             log_metrics[metric_key(objective, 'loss')] = None

#         elif objective == 'species_polyphony':
#             compile_metrics[objective] = [
#                 RegressionAccuracy(name='accuracy'),
#                 RegressionPrecision(name='precision'),
#                 RegressionRecall(name='recall'),
#                 RegressionF1(name='f1'),
#             ]
#             for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
#                 log_metrics[metric_key(objective, metric_name)] = None
#             log_metrics[metric_key(objective, 'loss')] = None

#     # Add top-level val_loss
#     log_metrics['val_loss'] = None
#     epoch_keys = [f'{key}_epoch' for key in log_metrics.keys()]
#     for epoch_key in epoch_keys:
#         log_metrics[epoch_key] = None

#     return compile_metrics, log_metrics

def build_compile_metrics(objectives_cfg):
    compile_metrics = {}
    multiple_objectives = len(objectives_cfg) > 1

    def metric_key(objective, metric_name):
        if multiple_objectives:
            return f"val_{objective}_{metric_name}"
        return f"val_{metric_name}"

    for objective, obj_cfg in objectives_cfg.items():
        if objective == "polyphony_reg":
            compile_metrics[objective] = RegressionAccuracy(name='accuracy')

        elif objective == "polyphony_class":
            compile_metrics[objective] = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

        elif objective == "binary":
            compile_metrics[objective] = tf.keras.metrics.BinaryAccuracy(name='accuracy')

        elif objective == 'event_logits':
            compile_metrics[objective] = tf.keras.metrics.BinaryAccuracy(name='accuracy')

        elif objective == 'framewise_polyphony_reg':
            compile_metrics[objective] = RegressionAccuracy(name='accuracy')

        elif objective == 'framewise_polyphony_class':
            compile_metrics[objective] = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

        elif objective == 'species_polyphony_reg':
            num_classes = obj_cfg['num_classes']
            compile_metrics[objective] = [
                RegressionAccuracy(name='accuracy'),
                RegressionCountPrecision(num_classes=num_classes, name='precision'),
                RegressionCountRecall(num_classes=num_classes, name='recall'),
                RegressionCountF1(num_classes=num_classes, name='f1'),
            ]
        elif objective == 'species_polyphony_class':
            num_classes = obj_cfg['num_classes']
            compile_metrics[objective] = [
                ClassificationAccuracy(name='accuracy'),
                ClassificationCountPrecision(num_classes=num_classes, name='precision'),
                ClassificationCountRecall(num_classes=num_classes, name='recall'),
                ClassificationCountF1(num_classes=num_classes, name='f1'),
            ]

        # elif objective == 'species_polyphony':
        #     compile_metrics[objective] = [
        #         RegressionAccuracy(name='accuracy'),
        #         RegressionPrecision(name='precision'),
        #         RegressionRecall(name='recall'),
        #         RegressionF1(name='f1'),
        #     ]

    return compile_metrics

def build_log_metrics(objectives_to_log):
    if objectives_to_log is None:
        return {}
    log_metrics = {}
    multiple_objectives = len(objectives_to_log) > 1

    def metric_key(objective, metric_name):
        if multiple_objectives:
            return f"val_{objective}_{metric_name}"
        return f"val_{metric_name}"

    for objective in objectives_to_log:
        # if objective == "polyphony_reg":
        #     log_metrics[metric_key(objective, 'accuracy')] = None
        #     log_metrics[metric_key(objective, 'loss')] = None

        # elif objective == "polyphony_class":
        #     log_metrics[metric_key(objective, 'accuracy')] = None
        #     log_metrics[metric_key(objective, 'loss')] = None

        # elif objective == "binary":
        #     log_metrics[metric_key(objective, 'accuracy')] = None
        #     log_metrics[metric_key(objective, 'loss')] = None

        # elif objective == 'event_logits':
        #     log_metrics[metric_key(objective, 'accuracy')] = None
        #     log_metrics[metric_key(objective, 'loss')] = None

        # elif objective == 'framewise_polyphony_reg':
        #     log_metrics[metric_key(objective, 'accuracy')] = None
        #     log_metrics[metric_key(objective, 'loss')] = None

        # elif objective == 'framewise_polyphony_reg':
        #     log_metrics[metric_key(objective, 'accuracy')] = None
        #     log_metrics[metric_key(objective, 'loss')] = None
         
        if objective in {"polyphony_reg", "polyphony_class", "binary", "event_logits", "framewise_polyphony_reg", "framewise_polyphony_class"}:
            log_metrics[metric_key(objective, 'accuracy')] = None
            log_metrics[metric_key(objective, 'loss')] = None    

        elif objective in ['species_polyphony_reg', 'species_polyphony_class']:
            for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
                log_metrics[metric_key(objective, metric_name)] = None
            log_metrics[metric_key(objective, 'loss')] = None

    # Add top-level metrics
    log_metrics['val_accuracy'] = None
    log_metrics['val_loss'] = None

    # Add epoch for all metrics
    epoch_keys = [f'{key}_epoch' for key in log_metrics.keys()]
    for epoch_key in epoch_keys:
        log_metrics[epoch_key] = None

    return log_metrics

def main():

    # Configuration
    cfg = OmegaConf.load("params.yaml")

    # Load the hyperparameters from the "params.yaml" file for usage with Tensorboard SummaryWriter
    params = Params()

    random_seed = cfg.general.random_seed
    set_random_seeds(random_seed)

    huggingface_path = cfg.dataset.huggingface_path
    dataset_config = cfg.dataset.train_config
    keep_last_n_checkpoints = cfg.log.keep_last_n_checkpoints if 'keep_last_n_checkpoints' in cfg.log else 5

    log_paths = get_log_paths(cfg)
    log_dir = log_paths['train_log_dir']
    checkpoint_dir = log_paths['checkpoint_dir']

    print("Log dir:", log_dir)
    print("Checkpoint dir:", checkpoint_dir)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    study_name = cfg.log.study_name
    load_model_path = cfg.train.load_model_path if 'load_model_path' in cfg.train else None
    load_checkpoint_path = cfg.train.load_checkpoint_path if 'load_checkpoint_path' in cfg.train else  None
    load_history_path = cfg.train.load_history_path if 'load_history_path' in cfg.train else None

    input_feature_name = cfg.train.input_feature_name
    total_epochs = cfg.train.epochs
    initial_epoch = cfg.train.initial_epoch if 'initial_epoch' in cfg.train and cfg.train.initial_epoch else 0
    learning_rate = cfg.train.learning_rate
    early_stopping_patience = cfg.train.early_stopping_patience if 'early_stopping_patience' in cfg.train else 10
    early_stopping_delay_epochs = cfg.train.early_stopping_delay_epochs if 'early_stopping_delay_epochs' in cfg.train else 0
    batch_size = cfg.train.batch_size
    num_batches_train = cfg.train.num_batches_train if 'num_batches_train' in cfg.train else None
    num_batches_val = cfg.train.num_batches_val if 'num_batches_val' in cfg.train else None

    objectives_to_log = cfg.log.objectives_to_log if 'objectives_to_log' in cfg.log else None
    model_cfg = cfg.model
    objectives_cfg = cfg.objectives
    
    #################################
    # Load dataset
    #################################

    # Load environment variables from .env file
    load_dotenv('local.env')
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

    # Load Dataset
    print(f"[INFO] HF_DATASETS_OFFLINE={os.environ.get('HF_DATASETS_OFFLINE', 'NOT SET')} (ommits updating datasets to avoid hitting rate limit on Huggingface Hub)")
   
    dataset = load_dataset_with_retry(huggingface_path, dataset_config, token=huggingface_token)
    # TODO: remove after testing - keep only a subset of the dataset to speed up testing
    for split in dataset.keys():
        dataset[split] = dataset[split].select(range(100))
    #  # TODO: reset after testing
    # from datasets import load_from_disk
    # dataset = load_from_disk("test_data/HSN")
    # from datasets import load_dataset, DatasetDict, Dataset
    # print(f"Loading dataset {huggingface_path} with config {dataset_config} from Huggingface Hub...")
    # dataset = load_dataset(huggingface_path, dataset_config, token=huggingface_token, streaming=True)
    # print("Dataset loaded. Converting to in-memory format for processing...")
    # dataset = DatasetDict({
    #     split: Dataset.from_list(list(ds.take(2)))
    #     for split, ds in dataset.items()
    # })

    # dataset.save_to_disk("test_data/HSN")

    # TODO: Remove after handling in model output processing
    # Reshape input features if needed based on model requirements
    if input_feature_name in {
        "EfficientNet-B1-BirdSet-XCL_audio_spatial_embeddings",
        "AudioProtoPNet-20-BirdSet-XCL_audio_spatial_embeddings",
        "yamnet_audio_spatial_embeddings",
        "vggish_audio_spatial_embeddings",
        "Wav2Vec2-Base-BirdSet-XCL_audio_spatial_embeddings",
        "beans_baseline_audio_spatial_embeddings",
        "AST-Birdset-XCL_audio_spatial_embeddings",

        "EfficientNet-B1-BirdSet-XCL_no_noise_audio_spatial_embeddings",
        "AudioProtoPNet-20-BirdSet-XCL_no_noise_audio_spatial_embeddings",
        "yamnet_no_noise_audio_spatial_embeddings",
        "vggish_no_noise_audio_spatial_embeddings",
        "Wav2Vec2-Base-BirdSet-XCL_no_noise_audio_spatial_embeddings",
        "beans_baseline_no_noise_audio_spatial_embeddings",
        "AST-Birdset-XCL_no_noise_audio_spatial_embeddings"
    }:
        print(f"Applying reshape to input feature '{input_feature_name}' for all splits...")
        for split in dataset.keys():
            dataset[split] = dataset[split].map(lambda x: reshape_to_tfe(x, input_feature_name), keep_in_memory=False)

    if dataset is None:
        raise RuntimeError("Dataset failed to load after all retry attempts. Check network/cache or force redownload in dataset preparation.")
    
    #####################################
    # Update model and objectives config
    #####################################
    if 'species_polyphony_reg' in objectives_cfg or 'species_polyphony_class' in objectives_cfg:
        # Get birdset ids
        birdset_id2label = get_birdset_id2label(dataset)
        # Save mapping
        mapping = {i: (bird_id, birdset_id2label[bird_id]) for i, bird_id in enumerate(birdset_id2label)}
        path = os.path.join(log_dir, "species_polyphony_mapping.json")
        with open(path, "w") as f:
            json.dump(mapping, f, indent=2)

    # Set number of classes for polyphony degree classification based on dataset config
    num_classes = cfg.dataset.max_polyphony + 1
    num_species = len(birdset_id2label)
    if 'polyphony_class' in objectives_cfg:
        objectives_cfg.polyphony_class.num_classes = num_classes
        print(f"Using {num_classes} classes for polyphony degree classification based on config.")
    if 'framewise_polyphony_class' in objectives_cfg:
        objectives_cfg.framewise_polyphony_class.num_classes = num_classes
        print(f"Using {num_classes} classes for framewise polyphony classification based on config.")
    if 'species_polyphony_reg' in objectives_cfg:
        objectives_cfg.species_polyphony_reg.num_classes = num_classes
        objectives_cfg.species_polyphony_reg.num_species = num_species
        print(f"Using {num_species} species for species polyphony regression based on dataset.")
    if 'species_polyphony_class' in objectives_cfg:
        objectives_cfg.species_polyphony_class.num_classes = num_classes
        objectives_cfg.species_polyphony_class.num_species = num_species
        print(f"Using {num_species} species and {num_classes} classes for species polyphony classification based on config and dataset.")

    # Set objectives config in model config for easy access when building model and losses
    model_cfg.objectives_cfg = objectives_cfg

    labels = list(objectives_cfg.keys()) #[objectives_cfg[x]['label'] for x in objectives_cfg]

    print(f"Training with {input_feature_name} as input feature and {labels} as labels on dataset {huggingface_path} with config {dataset_config}.")

    #################################
    # Add labels
    #################################
    print("Adding labels to dataset based on objectives config...")

    # Get input dim
    input_dim = tf.squeeze(np.array(dataset['train'][0][input_feature_name])).shape

    print(f"Input feature '{input_feature_name}' has shape {input_dim} for the first example. Assuming this is the input shape for the model.")

    # Add labels if more than polyphony degree is requested
    if labels == ['polyphony_reg'] or labels == ['polyphony_class']:
        labels = ['polyphony_degree']
    else:        
        # Compute additional labels
        time_dim = input_dim[0] if len(input_dim) > 1 else None
        freq_dim = input_dim[1] if len(input_dim) > 2 else None
        dataset, added_labels = add_labels(dataset, labels, birdset_id2label=birdset_id2label, time_dim=time_dim, freq_dim=freq_dim)

        print("Added labels: ", added_labels)

        existing_labels = added_labels # + ['polyphony_reg']
        missing_labels = set(labels) - set(existing_labels)
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

     #################################
    # Logging setup
    #################################

    # # Build per-output accuracy metrics depending on objective type
    # compile_metrics = {}

    # for objective, obj_cfg in objectives_cfg.items():

    #     # Handle polyphony accuracy
    #     # metric_name = 'accuracy' if "val_accuracy" in log_metrics else f"{objective}_accuracy"
    #     if objective == "polyphony_reg":
    #         compile_metrics[objective] = RoundedAccuracy(name='accuracy')
    #     elif objective == "polyphony_class":
    #         compile_metrics[objective] = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy') 

    #     elif objective == "binary":
    #         compile_metrics[objective] = tf.keras.metrics.BinaryAccuracy(name='accuracy')

    #     # Handle event logits accuracy
    #     elif objective == 'event_logits':
    #         compile_metrics[objective] = tf.keras.metrics.BinaryAccuracy(name='accuracy')

    #     # Handle frame-wise polyphony accuracy
    #     elif objective == 'framewise_polyphony_reg':
    #         compile_metrics[objective] = RoundedAccuracy(name='accuracy')

    #     # Handle species polyphony accuracy
    #     elif objective == 'species_polyphony':
    #         compile_metrics[objective] = [
    #             RoundedAccuracy(name='accuracy'),
    #             RoundedPrecision(name='precision'),
    #             RoundedRecall(name='recall'),
    #             RoundedF1(name='f1'),
    #         ]

    # Build metrics from objectives config        
    compile_metrics = build_compile_metrics(objectives_cfg)
    log_metrics = build_log_metrics(objectives_to_log)
    print("Metrics to log in hParam tab of tensorboard:", log_metrics)

    params['dataset']['train_size'] = str(train_size) #str(len(dataset['train']))
    params['dataset']['val_size'] = str(val_size) #str(len(dataset['validation']))
    params['dataset']['test_size'] = str(len(dataset['test']))
    params['train']['input_feature_name'] = input_feature_name
    params['train']['objectives'] = list(cfg.objectives.keys())
    print(params)

    confusion_matrix_specs = build_confusion_matrix_specs(objectives_cfg)

    #################################
    # Model
    #################################

    tf.keras.backend.clear_session()

    # Create loss objects based on objectives config
    class_weights_by_objective = {}
    num_classes = cfg.dataset.max_polyphony + 1

    if "species_polyphony_class" in objectives_cfg:
        class_weights_by_objective["species_polyphony_class"] = compute_species_count_class_weights(
            dataset["train"], label_column="species_polyphony_class", num_classes=num_classes
        )

    if "species_polyphony_reg" in objectives_cfg:
        class_weights_by_objective["species_polyphony_reg"] = compute_species_count_class_weights(
            dataset["train"], label_column="species_polyphony_reg", num_classes=num_classes
        )

    # losses = create_losses_from_objectives(objectives_cfg, class_weights_by_objective)
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
        model = instantiate(model_cfg)
        model.compile(optimizer=Adam(learning_rate), loss=losses, metrics=compile_metrics)
        
        # Initialize variables with forward pass
        sample_batch = next(iter(train_dataset))
        _ = model(sample_batch[0], training=False)

        model.load_weights(load_checkpoint_path)
        print(f"Resuming from epoch {initial_epoch}")
    else:
        print("Creating new model")
        model = instantiate(model_cfg)
        print("Model config:")
        print(model_cfg)
        # Initialize new model with forward pass
        sample_batch = next(iter(train_dataset))
        print(sample_batch[0].shape)  # full shape including all dims
        print(sample_batch[0].dtype)  
        print(input_dim)  
        input_dim = sample_batch[0].shape
        # _ = model(tf.zeros((1, 20, 8, 1280)), training=False) 
        _ = model(tf.zeros((input_dim)), training=False)
        _ = model(sample_batch[0], training=False)
        print(f"New model has {len(model.trainable_variables)} trainable variables")
        print(f"Compiling model with losses: {losses}")
        print(f"Compile metrics: {compile_metrics}")

        model.compile(optimizer=Adam(learning_rate), loss=losses, metrics=compile_metrics)
        
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
    # Callbacks
    #################################

    early_stopping = EarlyStopping(
        monitor='val_loss',  
        patience=early_stopping_patience,            
        restore_best_weights=False,
        start_from_epoch=early_stopping_delay_epochs
    )             

    model_and_history_saver = ModelAndHistorySaver(checkpoint_dir=checkpoint_dir, loss_objects=losses, previous_history=previous_history, keep_last_n=keep_last_n_checkpoints)

    writer = CustomSummaryWriter(log_dir=log_dir, params=params, metrics=log_metrics, sync_interval=0)
    tensorboard_callback = CustomSummaryWriterCallback(writer=writer, include_standard_tensorboard=False, val_dataset=val_dataset,
                log_confusion_matrix=True, confusion_matrix_frequency=1, 
                confusion_matrix_specs=confusion_matrix_specs, input_shape=input_dim, cfg=cfg, loss_objects=losses,
                previous_history=previous_history) #, dvclive_tracked_val_metrices=list(metrics.keys())) 

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

