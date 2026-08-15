import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
from dotenv import load_dotenv
import shutil

# Set temporary directory for HuggingFace datasets cache to avoid conflicts in parallel runs
# has to be set before datasets is imported, otherwise it will not take effect
# load_dotenv('global.env')
# huggingface_cache_dir = os.environ.get("HF_DATASETS_CACHE", "/beegfs/scratch/cohrt/.cache/huggingface/datasets")
# slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
# tmp_dir = f"{huggingface_cache_dir}/.tmp/job_{slurm_job_id}"
# print("TMPDIR:", tmp_dir)
# os.environ["TMPDIR"] =  tmp_dir #f"{huggingface_cache_dir}/.tmp/job_{slurm_job_id}"
# os.makedirs(os.environ["TMPDIR"], exist_ok=True)

from datasets import concatenate_datasets, load_from_disk, Audio
from omegaconf import OmegaConf
from hydra.utils import instantiate, get_class
import json

# BioacousticsModel/HopliteBackbone must live in the same module as the
# SimpleMLP/TemporalCNN heads referenced by cfg.model._target_ (e.g. "model.py"),
# since they're used together for the raw-audio / full-model fine-tuning path.
# Adjust this import path if you place them in a separate module.
from model import BioacousticsModel, HopliteBackbone, build_new_model
from perch_hoplite.zoo import model_configs as hoplite_model_configs

from utils.logs import RegressionAccuracy, RegressionCountPrecision, RegressionCountRecall, RegressionCountF1, ClassificationAccuracy, ClassificationCountPrecision, ClassificationCountRecall, ClassificationCountF1, CustomSummaryWriter, CustomSummaryWriterCallback, build_confusion_matrix_specs, ModelAndHistorySaver, get_log_paths, get_archive_paths
from utils.general import reshape_tensor_data, get_num_workers
from utils.config import set_random_seeds, Params
from utils.dataset import add_labels, get_birdset_id2label, get_local_data_dir, filter_dataset_by_polyphony_and_snr
from utils.losses import create_losses_from_objectives, setup_loss_scheduler, compute_species_count_class_weights

# Disable caching to avoid huggingface caching, forces huggingface to store data in tmp_dir
from datasets import disable_caching
# disable_caching()

tf.keras.backend.clear_session()

def get_tf_datasets(dataset, features, labels, batch_size):
    train_dataset = get_tf_dataset_from_split(dataset, 'train', features, labels, batch_size, shuffle=True)
    val_dataset = get_tf_dataset_from_split(dataset, 'validation', features, labels, batch_size, shuffle=False)
    test_dataset = get_tf_dataset_from_split(dataset, 'test', features, labels, batch_size, shuffle=False)

    return train_dataset, test_dataset, val_dataset

def get_tf_dataset_from_split(dataset, split_name, features, labels, batch_size, shuffle=False, num_workers=get_num_workers(gb_per_worker=5, cpu_percentage=0.8)):
    if split_name not in dataset:
        raise ValueError(f"Split {split_name} not found in dataset. Available splits: {dataset.keys()}")

    feature_list = features if isinstance(features, list) else [features]
    cols_to_keep = set(feature_list) | set(labels)
    cols_to_remove = [c for c in dataset[split_name].column_names if c not in cols_to_keep]
    split = dataset[split_name].remove_columns(cols_to_remove)

    labels = list(set(labels))

    # Cheap, metadata-only null check (no data materialization)
    none_cols = [col for col in cols_to_keep if split.data.column(col).null_count > 0]
    print("None columns: ", none_cols)

    if none_cols:
        print(f"[WARNING] Columns with None values found: {none_cols}. Filtering out affected rows.")
        split = split.filter(
            lambda batch: [
                all(v is not None for v in vals)
                for vals in zip(*(batch[c] for c in none_cols))
            ],
            batched=True,
            batch_size=1000,
            input_columns=none_cols,
            num_proc=num_workers,
        )
        print(f"[INFO] Remaining rows after filtering: {len(split)}")

    return split.to_tf_dataset(
        columns=features,
        label_cols=labels,
        batch_size=batch_size,
        shuffle=shuffle,
        prefetch=False,
    )

# def get_tf_dataset_from_split(dataset, split_name, features, labels, batch_size, shuffle=False):
#     if split_name not in dataset:
#         raise ValueError(f"Split {split_name} not found in dataset. Available splits: {dataset.keys()}")

#     # Keep only the columns actually needed
#     feature_list = features if isinstance(features, list) else [features]
#     cols_to_keep = set(feature_list) | set(labels)
#     cols_to_remove = [c for c in dataset[split_name].column_names if c not in cols_to_keep]
#     split = dataset[split_name].remove_columns(cols_to_remove)

#     # Remove duplicated labels if any
#     labels = list(set(labels))

#     # Check for None values and report which columns are affected
#     none_cols = []
#     for col in cols_to_keep:
#         if any(v is None for v in split[col]):
#             none_cols.append(col)
    
#     print("None columns: ", none_cols)

#     if none_cols:
#         print(f"[WARNING] Columns with None values found: {none_cols}. Filtering out affected rows.")
#         split = split.filter(lambda row: all(row[col] is not None for col in none_cols))
#         print(f"[INFO] Remaining rows after filtering: {len(split)}")

#     return split.to_tf_dataset(
#         columns=features,
#         label_cols=labels,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         prefetch=False
#     )

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
    train_config = cfg.dataset.train_config
    subset = cfg.dataset.subset
    keep_last_n_checkpoints = cfg.log.keep_last_n_checkpoints if 'keep_last_n_checkpoints' in cfg.log else 5

    log_paths = get_log_paths(cfg)
    log_dir = log_paths['train_log_dir']
    checkpoint_dir = log_paths['checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("Log dir:", log_dir)
    print("Checkpoint dir:", checkpoint_dir)

    archive_paths = get_archive_paths(cfg)
    archive_log_dir = archive_paths['train_log_dir']
    checkpoint_archive_dir = archive_paths['checkpoint_dir']
    os.makedirs(archive_log_dir, exist_ok=True)
    os.makedirs(checkpoint_archive_dir, exist_ok=True)

    default_dir = os.environ.get('DEFAULT_DIR', '')

    study_name = cfg.log.study_name
    load_model_path = cfg.train.load_model_path if 'load_model_path' in cfg.train else None
    load_checkpoint_path = cfg.train.load_checkpoint_path if 'load_checkpoint_path' in cfg.train else  None
    load_history_path = cfg.train.load_history_path if 'load_history_path' in cfg.train else None

    input_feature = cfg.train.input_feature
    input_feature_name = cfg.train.input_feature_name
    input_feature_name = input_feature_name if input_feature_name else input_feature
    total_epochs = cfg.train.epochs
    initial_epoch = cfg.train.initial_epoch if 'initial_epoch' in cfg.train and cfg.train.initial_epoch else 0
    learning_rate = cfg.train.learning_rate
    early_stopping_patience = cfg.train.early_stopping_patience if 'early_stopping_patience' in cfg.train and cfg.train.early_stopping_patience is not None else 0
    early_stopping_delay_epochs = cfg.train.early_stopping_delay_epochs if 'early_stopping_delay_epochs' in cfg.train and cfg.train.early_stopping_delay_epochs is not None else 0
    batch_size = cfg.train.batch_size if 'batch_size' in cfg.train and cfg.train.batch_size is not None else 32
    num_batches_train = cfg.train.num_batches_train if 'num_batches_train' in cfg.train else None
    num_batches_val = cfg.train.num_batches_val if 'num_batches_val' in cfg.train else None

    objectives_to_log = cfg.log.objectives_to_log if 'objectives_to_log' in cfg.log else None
    model_cfg = cfg.model
    model_cfg.pop("name", None)
    objectives_cfg = cfg.objectives

    # --- Precomputed-embeddings vs. full raw-audio fine-tuning ---
    # Detected from cfg.train.backend, same field evaluate_on_test_split.py
    # already uses to distinguish 'torch' from TF -- 'perch' is a third
    # value meaning "TF + BioacousticsModel wrapping a perch_hoplite
    # backbone", vs. the TF default meaning "cfg.model as the whole model,
    # trained on precomputed embeddings" (unchanged, existing behavior).
    backend = cfg.train.get("backend", "tensorflow")
    precomputed_embeddings = (backend != "perch")
    freeze_encoder = cfg.train.freeze_encoder if 'freeze_encoder' in cfg.train else True
    freeze_epochs = cfg.train.freeze_epochs if 'freeze_epochs' in cfg.train else None
    finetune_learning_rate = cfg.train.finetune_learning_rate if 'finetune_learning_rate' in cfg.train else None

    backbone_cfg = cfg.backbone if 'backbone' in cfg else None
    head_cfg = cfg.head if 'head' in cfg else None
    if not precomputed_embeddings and (backbone_cfg is None or head_cfg is None):
        raise ValueError(
            "cfg.train.backend='perch' requires both cfg.backbone (with a "
            "'name' matching a perch_hoplite preset, e.g. 'perch_v2') and "
            "cfg.head (with a '_target_' pointing to SimpleMLP or TemporalCNN, "
            "same as cfg.model does for the precomputed-embeddings path)."
        )
    
    # #################################
    # # Load dataset
    # #################################

    print("default_dir:", default_dir)
    print("train_config:", train_config)
    print("subset:", subset)
    local_data_dir = get_local_data_dir(dataset_config=train_config, subset=subset)
    print("local_data_dir:", local_data_dir)
    dataset_dir = os.path.join(default_dir, local_data_dir)
    print("dataset_dir:", dataset_dir)
    dataset = load_from_disk(dataset_dir)

    num_workers = get_num_workers(gb_per_worker=5, cpu_percentage=0.8)
    dataset = filter_dataset_by_polyphony_and_snr(dataset, cfg, num_workers=num_workers)

    # # Filter dataset by polyphony degree if specified in the config
    # if 'max_polyphony' in cfg.dataset and cfg.dataset.max_polyphony is not None:
    #     max_polyphony = cfg.dataset.max_polyphony
    #     print(f"Filtering dataset to include only examples with polyphony degree <= {max_polyphony}...")
    #     for split in dataset.keys():
    #         if 'polyphony' not in dataset[split].column_names and 'polyphony_degree' in dataset[split].column_names:
    #             dataset[split] = dataset[split].rename_column('polyphony_degree', 'polyphony')
    #         dataset[split] = dataset[split].filter(lambda polyphony: [p <= max_polyphony for p in polyphony], input_columns=['polyphony'], batched=True, num_proc=num_workers, batch_size=100)
    #         print(f"After filtering, {split} split has {len(dataset[split])} examples.")

    # # Filter dataset by SNR if specified in the config
    # if 'snr_range' in cfg.dataset and cfg.dataset.snr_range is not None:
    #     snr_range = cfg.dataset.snr_range
    #     print(f"Filtering dataset to include only examples with SNR in range {snr_range}...")
    #     for split in dataset.keys():
    #         dataset[split] = dataset[split].filter(lambda snr: [snr_range[0] <= s <= snr_range[1] for s in snr], input_columns=['snr_dB'], batched=True, num_proc=num_workers, batch_size=100)
    #         print(f"After filtering, {split} split has {len(dataset[split])} examples.")
    # if 'min_snr' in cfg.dataset and cfg.dataset.min_snr is not None:
    #     min_snr = cfg.dataset.min_snr
    #     print(f"Filtering dataset to include only examples with SNR >= {min_snr}...")
    #     for split in dataset.keys():
    #         dataset[split] = dataset[split].filter(lambda snr: [s >= min_snr for s in snr], input_columns=['snr_dB'], batched=True, num_proc=num_workers, batch_size=100)
    #         print(f"After filtering, {split} split has {len(dataset[split])} examples.")

    # dataset.save_to_disk("test_data/HSN")

    print("Input shape:", np.array(dataset['train'][0][input_feature_name]).shape)

    # # TODO: Remove after handling in model output processing
    # # Reshape input features if needed based on model requirements
    # if input_feature_name in {
    #     "EfficientNet-B1-BirdSet-XCL_audio_spatial_embeddings",
    #     "AudioProtoPNet-20-BirdSet-XCL_audio_spatial_embeddings",
    #     "yamnet_audio_spatial_embeddings",
    #     "vggish_audio_spatial_embeddings",
    #     "Wav2Vec2-Base-BirdSet-XCL_audio_spatial_embeddings",
    #     "beans_baseline_audio_spatial_embeddings",
    #     "AST-Birdset-XCL_audio_spatial_embeddings",

    #     "EfficientNet-B1-BirdSet-XCL_no_noise_audio_spatial_embeddings",
    #     "AudioProtoPNet-20-BirdSet-XCL_no_noise_audio_spatial_embeddings",
    #     "yamnet_no_noise_audio_spatial_embeddings",
    #     "vggish_no_noise_audio_spatial_embeddings",
    #     "Wav2Vec2-Base-BirdSet-XCL_no_noise_audio_spatial_embeddings",
    #     "beans_baseline_no_noise_audio_spatial_embeddings",
    #     "AST-Birdset-XCL_no_noise_audio_spatial_embeddings"
    # }:
    #     print(f"Applying reshape to input feature '{input_feature_name}' for all splits...")
    #     for split in dataset.keys():
    #         dataset[split] = dataset[split].map(lambda x: reshape_to_tfe(x, input_feature_name), keep_in_memory=False)

    print("Input shape after reshape:", np.array(dataset['train'][0][input_feature_name]).shape)

    if dataset is None:
        raise RuntimeError("Dataset failed to load after all retry attempts. Check network/cache or force redownload in dataset preparation.")
    
    #####################################
    # Update model and objectives config
    #####################################

    # Get birdset ids
    # TODO: remove and use ClassLabels from dataset instead of hardcoding
    birdset_id2label = get_birdset_id2label(subset, dataset=dataset)
    num_species = len(birdset_id2label)

    if 'species_polyphony_reg' in objectives_cfg or 'species_polyphony_class' in objectives_cfg:
        
        # Save mapping
        mapping = {i: (bird_id, birdset_id2label[bird_id]) for i, bird_id in enumerate(birdset_id2label)}
        path = os.path.join(log_dir, "species_polyphony_mapping.json")
        with open(path, "w") as f:
            json.dump(mapping, f, indent=2)
        

    # Set number of classes for polyphony degree classification based on dataset config
    num_classes = cfg.dataset.max_polyphony + 1
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

    print(f"Training with {input_feature_name} as input feature and {labels} as labels on dataset {huggingface_path} with config {train_config}.")

    print(f"Dataset feature schema: {dataset['train'].features[input_feature_name]}")

    #################################
    # Add labels
    #################################
    print("Adding labels to dataset based on objectives config...")

    from datasets import config

    print("HF_DATASETS_CACHE (config):", config.HF_DATASETS_CACHE)
    print("HF_CACHE_HOME:", config.HF_CACHE_HOME)
    print("HF_HOME env:", __import__("os").environ.get("HF_HOME"))
    print("HF_DATASETS_CACHE env:", __import__("os").environ.get("HF_DATASETS_CACHE"))

    # Most reliable check: ask the actual dataset object where it's writing
    print("Cache files for this split:", dataset["train"].cache_files[0])

    # Get input dim. NOTE: for raw-audio mode (precomputed_embeddings=False)
    # this is just the raw waveform's shape, not the model's actual input
    # dim -- it's only used as a fallback below and gets overwritten with
    # the real head input dim after model creation.
    # input_dim = tf.squeeze(np.array(dataset['train'][0][input_feature_name])).shape
    input_dim = np.array(dataset['train'][0][input_feature_name]).shape

    print(f"Input feature '{input_feature_name}' has shape {input_dim} for the first example. Assuming this is the input shape for the model.")

    # Add labels if more than polyphony degree is requested
    if labels == ['polyphony_reg'] or labels == ['polyphony_class']:
        labels = ['polyphony']
    else:
        framewise_objectives = {"framewise_polyphony_reg", "framewise_polyphony_class", "event_logits"}
        needs_frame_dims = bool(framewise_objectives & set(labels))

        if precomputed_embeddings:
            # Existing behavior: read straight off the precomputed array's shape.
            time_dim = input_dim[0] if len(input_dim) > 1 else None
            freq_dim = input_dim[1] if len(input_dim) > 2 else None
        elif needs_frame_dims:
            # Raw-audio mode + a framewise objective: time_dim/freq_dim must
            # come from a real backbone forward pass, since they depend on
            # the backbone's internal spatial grid, not on anything in the
            # dataset. Cheap standalone probe -- NOT the training model --
            # so this loads the backbone a second time (the real model is
            # built later, after add_labels); acceptable one-time cost,
            # cached downloads make repeat runs fast. See HopliteBackbone
            # in model.py for what spatial_embeddings actually contains.
            print(f"Probing backbone '{backbone_cfg.name}' for time_dim/freq_dim "
                  f"needed by framewise objective(s) {framewise_objectives & set(labels)}...")
            probe_backbone = HopliteBackbone(backbone_cfg.name)

            sample_feat = dataset['train'][0][input_feature_name]
            if isinstance(sample_feat, dict) and 'array' in sample_feat:
                sample_array = np.asarray(sample_feat['array'], dtype=np.float32)
                native_sr = sample_feat.get('sampling_rate', probe_backbone.sample_rate)
            else:
                sample_array = np.asarray(sample_feat, dtype=np.float32)
                native_sr = probe_backbone.sample_rate  # assumed already correct

            if native_sr != probe_backbone.sample_rate:
                import librosa
                sample_array = librosa.resample(
                    sample_array, orig_sr=native_sr, target_sr=probe_backbone.sample_rate
                )

            probe_waveform = tf.constant(sample_array[np.newaxis, :], dtype=tf.float32)
            probe_out = probe_backbone.embed(probe_waveform)
            spatial = probe_out["spatial_embeddings"]
            if spatial is None:
                raise ValueError(
                    f"Backbone '{backbone_cfg.name}' has no spatial structure "
                    f"(HopliteBackbone.has_structure() is False for it), so "
                    f"framewise objectives {framewise_objectives & set(labels)} "
                    f"can't be computed. Use a backbone that preserves structure "
                    f"(e.g. 'perch_v2'), or drop these objectives."
                )
            time_dim = int(spatial.shape[1])
            freq_dim = int(spatial.shape[2]) if len(spatial.shape) == 4 else None
            print(f"Probed time_dim={time_dim}, freq_dim={freq_dim} from backbone spatial_embeddings.")
            del probe_backbone  # done with it -- real model built later, separately
        else:
            # Raw-audio mode, no framewise objective -- these dims are unused.
            time_dim, freq_dim = None, None

        dataset, added_labels = add_labels(dataset, labels, birdset_id2label=birdset_id2label, time_dim=time_dim, freq_dim=freq_dim)

        print("Added labels: ", added_labels)

        existing_labels = added_labels # + ['polyphony_reg']
        missing_labels = set(labels) - set(existing_labels)
        if missing_labels:
            raise Exception("Not all requested labels could be computed.")
    
    ###########################################
    # Raw-audio mode: cast the audio column to the backbone's expected
    # sample rate. Looked up from perch_hoplite's preset metadata directly
    # (not by instantiating HopliteBackbone) to avoid loading the full
    # model twice -- BioacousticsModel below loads it for real, once.
    ###########################################
    if not precomputed_embeddings:
        backbone_sample_rate = hoplite_model_configs.get_preset_model_config(
            backbone_cfg.name
        ).model_config.sample_rate
        print(f"Casting '{input_feature_name}' to raw audio at {backbone_sample_rate}Hz "
              f"for backbone '{backbone_cfg.name}'.")
        for split in dataset:
            if 'sources_audio' in dataset[split].column_names:
                dataset[split] = dataset[split].remove_columns(['sources_audio'])
            dataset[split] = dataset[split].cast_column(
                input_feature_name, Audio(sampling_rate=backbone_sample_rate)
            )
        # to_tf_dataset() needs a plain numeric column, not the {'array',
        # 'sampling_rate'} dict an Audio feature decodes to -- extract the
        # array explicitly. NOTE: assumes fixed-length clips (same
        # assumption the precomputed-embedding path already makes); ragged
        # variable-length audio would need padding logic added here.
        def _extract_waveform(example, feature_name=input_feature_name):
            example[feature_name] = np.asarray(example[feature_name]["array"], dtype=np.float32)
            return example
        for split in dataset:
            dataset[split] = dataset[split].map(_extract_waveform)

    ###########################################
    # Transform dataset to tensorflow datasets 
    ###########################################

    # Get tensorflow datasets
    train_dataset, test_dataset, val_dataset = get_tf_datasets(dataset, input_feature_name, labels, batch_size)
    if num_batches_train: train_dataset = train_dataset.take(num_batches_train) # take fewer batches to reduce train dataset size
    if num_batches_val: val_dataset = val_dataset.take(num_batches_val)
    train_size = num_batches_train * batch_size if num_batches_train else len(dataset['train'])
    val_size = num_batches_val * batch_size if num_batches_val else len(dataset['validation'])

    batch = next(iter(train_dataset))
    print(f'batch[0].shape: {batch[0].shape}') 

     #################################
    # Logging setup
    #################################

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
        model = build_new_model(precomputed_embeddings, model_cfg, backbone_cfg,
                                 head_cfg, objectives_cfg, freeze_encoder)
        # run_eagerly required for raw-audio mode: BioacousticsModel's
        # backbone bypass (_raw_embed, in model.py) uses numpy ops that only
        # work on concrete tensors, not the symbolic tensors model.fit()
        # traces call() with by default. Precomputed-embedding mode doesn't
        # need this -- SimpleMLP/TemporalCNN are pure TF ops, keep them graph-compiled.
        model.compile(optimizer=Adam(learning_rate), loss=losses, metrics=compile_metrics,
                      run_eagerly=not precomputed_embeddings)
        
        # Initialize variables with forward pass
        sample_batch = next(iter(train_dataset))
        _ = model(sample_batch[0], training=False)
        # _ = model(tf.zeros((input_dim)), training=False)

        model.load_weights(load_checkpoint_path)
        print(f"Resuming from epoch {initial_epoch}")
    else:
        print("Creating new model")
        model = build_new_model(precomputed_embeddings, model_cfg, backbone_cfg,
                                 head_cfg, objectives_cfg, freeze_encoder)
        print("Model config:")
        print(model_cfg if precomputed_embeddings else {"backbone": backbone_cfg, "head": head_cfg})
        # Initialize new model with forward pass
        sample_batch = next(iter(train_dataset))
        print(sample_batch[0].shape)  # full shape including all dims ( without batch dimension)
        print(sample_batch[0].dtype)  
        print(input_dim)  
        # input_dim = sample_batch[0].shape
        # _ = model(tf.zeros((1, 20, 8, 1280)), training=False) 
        # _ = model(tf.zeros((input_dim)), training=False)
        _ = model(sample_batch[0], training=False)
        if not precomputed_embeddings:
            # input_dim above was just the raw waveform shape -- overwrite
            # with the real head input dim now that the backbone has run once.
            wants_spatial = getattr(get_class(head_cfg._target_), "takes_spatial_embeddings", False)
            input_dim = model.backbone.get_head_input_size(pooled=not wants_spatial)
            print(f"Real head input dim from backbone '{backbone_cfg.name}': {input_dim}")
        print(f"New model has {len(model.trainable_variables)} trainable variables")
        print(f"Compiling model with losses: {losses}")
        print(f"Compile metrics: {compile_metrics}")

        # See note on run_eagerly above the other compile() call.
        model.compile(optimizer=Adam(learning_rate), loss=losses, metrics=compile_metrics,
                      run_eagerly=not precomputed_embeddings)

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

    print(f"Starting training for {total_epochs} epochs from initial epoch {initial_epoch} with learning rate {learning_rate} and batch size {batch_size} on {input_feature_name} with input shape {input_dim}.")

    staged_finetuning = (not precomputed_embeddings) and freeze_encoder and freeze_epochs is not None
    if staged_finetuning and freeze_epochs > initial_epoch:
        print(f"Phase 1: training with frozen backbone, epochs {initial_epoch} -> {freeze_epochs}")
        history = model.fit(train_dataset,
                            validation_data=val_dataset,
                            epochs=freeze_epochs,
                            initial_epoch=initial_epoch,
                            callbacks=callbacks)

        print(f"Unfreezing backbone for fine-tuning"
              + (f" (encoder lr={finetune_learning_rate})" if finetune_learning_rate else ""))
        model.set_backbone_trainable(True)
        # Recompile is required here: Keras fixes the trainable-variable list
        # at compile() time, unlike torch's optimizer param groups, which
        # re-read requires_grad on the fly -- see torch_train.py's freeze/
        # unfreeze loop for the equivalent behavior without a recompile.
        model.compile(optimizer=Adam(finetune_learning_rate or learning_rate),
                      loss=losses, metrics=compile_metrics, run_eagerly=True)

        print(f"Phase 2: training with unfrozen backbone, epochs {freeze_epochs} -> {total_epochs}")
        history = model.fit(train_dataset,
                            validation_data=val_dataset,
                            epochs=total_epochs,
                            initial_epoch=freeze_epochs,
                            callbacks=callbacks)
    else:
        history = model.fit(train_dataset, 
                            validation_data=val_dataset, 
                            epochs=total_epochs,
                            initial_epoch=initial_epoch, 
                            callbacks=callbacks) #LossWeightScheduler(switch_epochs=[0,10,20,30,40], event_loss_weights=[1.0, 1.0, 1.0, 0.5, 0.1], count_loss_weights=[0.1, 0.5, 1.0, 1.0, 2.0])


    # Copy log files and subfolders to archive directory for later analysis
    if os.path.isdir(log_dir):
        print(f"Copying log files and subfolders from {log_dir} to archive directory {archive_log_dir}...")
        shutil.copytree(log_dir, archive_log_dir, dirs_exist_ok=True)
    else:
        print(f"Archive log directory {archive_log_dir} or log directory {log_dir} does not exist. Skipping copy.")

    if os.path.isdir(checkpoint_dir):
        print(f"Copying checkpoint files and subfolders from {checkpoint_dir} to archive directory {checkpoint_archive_dir}...")
        shutil.copytree(checkpoint_dir, checkpoint_archive_dir, dirs_exist_ok=True)
    else:
        print(f"Checkpoint archive directory {checkpoint_archive_dir} or checkpoint directory {checkpoint_dir} does not exist. Skipping copy.")


if __name__=="__main__":
     main()

