from omegaconf import OmegaConf
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from hydra.utils import instantiate
from dotenv import load_dotenv
from datasets import load_from_disk, Audio
import shutil
from pathlib import Path

from utils.dataset import get_birdset_id2label, get_local_data_dir, filter_dataset_by_polyphony_and_snr
from utils.logs import SummaryWriter, save_to_report, get_log_paths, get_archive_paths, get_dvc_exp_name
from utils.metrics import compute_polyphony_metrics
from utils.evaluation import arrays_to_records, collect_predictions, update_metrics_table
from utils.general import get_num_workers

from utils.torch_evaluation import load_torch_model_for_eval, collect_predictions_torch

# Disable caching to avoid huggingface caching issues when running multiple experiments in parallel
from datasets import disable_caching
disable_caching()

def main():

    #################################
    # Configuration
    #################################
    cfg = OmegaConf.load("params.yaml")
    print(cfg)

    study_name = cfg.log.study_name

    huggingface_path = cfg.dataset.huggingface_path
    train_config = cfg.dataset.train_config
    soundscape_dataset_config = cfg.dataset.soundscape_config
    subset = cfg.dataset.subset

    log_paths = get_log_paths(cfg)
    log_dir = Path(cfg.path.val_eval_output) / "logs", #log_paths['eval_log_dir']
    os.makedirs(log_dir, exist_ok=True)

    archive_paths = get_archive_paths(cfg)
    exp_name = get_dvc_exp_name()
    run_dir = f"{cfg.datetime}_{exp_name}"
    archive_log_dir = Path(cfg.path.val_eval_output) / run_dir / "logs" #archive_paths['eval_log_dir']
    os.makedirs(archive_log_dir, exist_ok=True)

    default_dir = os.environ.get('DEFAULT_DIR', '')
    checkpoint_dir = log_paths['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        checkpoint_dir = os.path.join(default_dir, checkpoint_dir)
    
    model_cfg = cfg.model
    model_cfg.pop("name", None)
    objectives_cfg = cfg.objectives

    input_feature = cfg.train.input_feature
    input_feature_name = cfg.train.get("input_feature_name", input_feature)
    input_feature_name = input_feature_name if input_feature_name is not None else input_feature
    # embedding_type = cfg.embeddings.type
    # embedding_dim_type = cfg.embeddings.dimension_type

    #################################
    # Setup
    #################################

    # Get summary writer for TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # Store config in file/logs
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.save(cfg, os.path.join(log_dir, "params.yaml"))

    # #################################
    # # Load dataset
    # #################################

    # val_dataset = tf.data.Dataset.load(val_dataset_path)

    # if not val_dataset:
    #     raise ValueError(f"Validation dataset not found at {val_dataset_path}. Please make sure to run the training script first to save the validation dataset for later evaluation.")

    #################################
    # Load dataset
    #################################

    # Load environment variables from .env file
    load_dotenv('local.env')
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

    # from huggingface_hub import HfApi
    # api = HfApi()
    # info = api.dataset_info("mcht67/Polyphonic-BirdSet-train", token=huggingface_token)
    # for config in info.card_data.get("configs", []):
    #     if config.get("config_name") == train_dataset_config:
    #         print(config.get("data_files"))
    #         print(config.get("data_dir"))

    # # Load Dataset
    # print(f"[INFO] HF_DATASETS_OFFLINE={os.environ.get('HF_DATASETS_OFFLINE', 'NOT SET')} (ommits updating datasets to avoid hitting rate limit on Huggingface Hub)")
    # train_dataset = load_dataset(huggingface_path, train_dataset_config, token=huggingfce_token, streaming=True)

    
    # test_dataset = load_dataset(huggingface_path, train_config, split='test', token=huggingface_token)
    # from datasets import Dataset
    # train_dataset = load_from_disk('data/HSN')
    # test_dataset = load_dataset(huggingface_path, train_config, split='test', token=huggingface_token, streaming=True)
    # print("Dataset loaded. Converting to in-memory format for processing...")
    # test_dataset = Dataset.from_list(list(test_dataset.take(2)))

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

    val_dataset = dataset['validation']

    if val_dataset is None:
        raise RuntimeError("Dataset failed to load after all retry attempts. Check network/cache or force redownload in dataset preparation.")

    # # Filter dataset by polyphony degree if specified in the confi
    # if 'polyphony' not in test_dataset.column_names and 'polyphony_degree' in test_dataset.column_names:
    #     test_dataset = test_dataset.rename_column('polyphony_degree', 'polyphony')
    # if 'max_polyphony' in cfg.dataset and cfg.dataset.max_polyphony is not None:
    #     max_polyphony = cfg.dataset.max_polyphony
    #     print(f"Filtering test dataset to include only examples with polyphony degree <= {max_polyphony}...")
    #     test_dataset = test_dataset.filter(lambda x: x['polyphony'] <= max_polyphony)
    #     print(f"After filtering, test split has {len(test_dataset)} examples.")

    # # Filter dataset by SNR if specified in the config
    # if 'snr_range' in cfg.dataset and cfg.dataset.snr_range is not None:
    #     snr_range = cfg.dataset.snr_range
    #     print(f"Filtering test dataset to include only examples with SNR in range {snr_range}...")
    #     test_dataset = test_dataset.filter(lambda x: snr_range[0] <= x['snr_dB'] <= snr_range[1])
    #     print(f"After filtering, test split has {len(test_dataset)} examples.")
    # if 'min_snr' in cfg.dataset and cfg.dataset.min_snr is not None:
    #     min_snr = cfg.dataset.min_snr
    #     print(f"Filtering test dataset to include only examples with SNR >= {min_snr}...")
    #     test_dataset = test_dataset.filter(lambda x: x['snr_dB'] >= min_snr)
    #     print(f"After filtering, test split has {len(test_dataset)} examples.")
    
    #################################
    # Update objectives config based on dataset
    #################################

    ebird_class_labels = None
    birdset_id2label = None

    if 'species_polyphony_reg' in objectives_cfg or 'species_polyphony_class' in objectives_cfg:
        
        #TODO: get from ClassLabels in dataset
        # ebird_class_labels = test_dataset.features['ebird_code_multilabel'].feature.names
        # Get birdset ids
        # scape_ds = load_dataset(huggingface_path, soundscape_dataset_config, split='test_5s', token=huggingface_token, download_mode='force_redownload')
        birdset_id2label = get_birdset_id2label(subset, dataset=dataset)
        ebird_class_labels = [k for k in birdset_id2label.values()]
        # ebird_code_class_labels = scape_ds.features['ebird_code_multilabel'].feature.names
        print(f"Found {len(ebird_class_labels)} species in the dataset: {ebird_class_labels}")
        num_species = len(ebird_class_labels)


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


    #################################
    # Load model
    #################################

    backend = cfg.train.get("backend", "tensorflow")

    print(cfg.model) 

    if backend == "torch":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint not found at {checkpoint_path}. Please make sure to run torch_train.py first to save the best checkpoint for later evaluation.")

        model = load_torch_model_for_eval(cfg.model, cfg.head, objectives_cfg, checkpoint_path, device)
        print(f"Loaded torch checkpoint from {checkpoint_path}")

        y_true, predictions, variable_values = collect_predictions_torch(
            model, val_dataset, input_feature, objectives_cfg, device,
            batch_size=cfg.train.get("eval_batch_size", 32), variables=['snr_dB'],
            birdset_id2label=birdset_id2label,
        )

    else:
        # Load best model from checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, "best.weights.h5")

        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint not found at {checkpoint_path}. Please make sure to run the training script first to save the best model checkpoint for later evaluation.")

        # Define model
        # Set objectives config in model config for easy access when building model and losses
        model_cfg.objectives_cfg = objectives_cfg
        model = instantiate(model_cfg)

        # Build model by calling it on a sample input
        first_example = val_dataset[0]
        print("input features", val_dataset.features)
        input_dim = int(tf.squeeze(np.array(first_example[input_feature_name])).shape[0])
        sample_input = tf.zeros((1, input_dim), dtype=tf.float32)
        _ = model(sample_input, training=False)

        print(f"Loading weights from {checkpoint_path}")
        model.load_weights(checkpoint_path)
        print("Model loaded successfully.")

        ###########################################
        # Metrics computation on test split
        ###########################################

        print(tf.config.list_physical_devices('GPU'))
        y_true, predictions, variable_values = collect_predictions(model, val_dataset, input_feature_name, variables=['snr_dB'], birdset_id2label=birdset_id2label)

    print("y_true:", y_true)
    print("predictions:", predictions)
    print("variable_values:", variable_values)

    # Define ouput dir for metrics and results
    out_dir = os.path.join(default_dir, "archive", cfg.log.study_name, cfg.path.study_subfolder, "eval_results")
    os.makedirs(out_dir, exist_ok=True)

    # Store results in a pickle file for later analysis
    records = arrays_to_records(y_true, predictions, variable_values)
    df = pd.json_normalize(records)
    df.to_pickle(os.path.join(log_dir, f"test_results.pkl"))

    if "polyphony_reg" in predictions:
        df.to_pickle(os.path.join(out_dir, f"{subset}_reg_test_results.pkl"))
    if "polyphony_class" in predictions:
        df.to_pickle(os.path.join(out_dir, f"{subset}_class_test_results.pkl"))
    if "species_polyphony_reg" in predictions:
        df.to_pickle(os.path.join(out_dir, f"{subset}_species_reg_test_results.pkl"))
    if "species_polyphony_class" in predictions:
        df.to_pickle(os.path.join(out_dir, f"{subset}_species_class_test_results.pkl"))

    # Calculate metrics for each objective and save to report
    report = {}

    num_classes = cfg.dataset.max_polyphony + 1
    species_mapping = None #birdset_id2label #{i: (i, name) for i, name in enumerate(ebird_class_labels)}

    if "polyphony_reg" in predictions:
        report["polyphony_reg"] = compute_polyphony_metrics(
            y_true["polyphony"][:, None], predictions["polyphony_reg"][:, None],
            cm_type="regression_round", per_species=False)

    if "polyphony_class" in predictions:
        report["polyphony_class"] = compute_polyphony_metrics(
            y_true["polyphony"][:, None], predictions["polyphony_class"][:, None, :],
            cm_type="classification", num_classes=num_classes, per_species=False)
        
    if "species_polyphony_reg" in predictions:
        report["species_polyphony_reg"] = compute_polyphony_metrics(
            y_true["species_polyphony"], predictions["species_polyphony_reg"],
            cm_type="species_regression_round", species_mapping=species_mapping, per_species=True)

    if "species_polyphony_class" in predictions:
        report["species_polyphony_class"] = compute_polyphony_metrics(
            y_true["species_polyphony"], predictions["species_polyphony_class"],
            cm_type="species_classification", species_mapping=species_mapping,
            num_classes=num_classes, per_species=True)
        
    save_to_report(report, os.path.join(log_dir, "val_metrics.json"))

    # Save to metrics overview table
    for objective in report:
        if objective == "polyphony_reg" or objective == "species_polyphony_reg":
            obj_key = "reg"
        elif objective == "polyphony_class" or objective == "species_polyphony_class":
            obj_key = "class"
        metrics_dict = report[objective]
        table_name = f"val_metrics"

        if study_name == "Pooled-Embeddings-SNR-Range-Effects":
            column_name = f"{subset}_{obj_key}_{cfg.dataset.snr_range[0]}-{cfg.dataset.snr_range[1]}"
        elif study_name == "Pooled-Embeddings-Min-SNR-Effects":
            column_name = f"{subset}_{obj_key}_{cfg.dataset.min_snr}"
        elif study_name == "Pooled-Embeddings-Max-Polyphony-Effects":
            column_name = f"{subset}_{obj_key}_{cfg.dataset.max_polyphony}"
        else:
            column_name = f"{subset}_{obj_key}"

        csv_path = os.path.join(out_dir, f"{table_name}.csv")
        df, csv_path = update_metrics_table(column_name, metrics_dict, csv_path)
        print(f"Saved {table_name} to {csv_path}")

    # for example in dataset['test']:

    #     embedding = example[input_feature_name]
        
    #     # Make prediction
    #     single_input = np.expand_dims(embedding, axis=0)
    #     single_input = tf.constant(single_input, dtype=tf.float32)
    #     predictions = model.predict(single_input)

    #     # Extract data
    #     objectives_list = list(objectives_cfg.keys())
    #     gt_polyphony = example['polyphony']

    # # Handle regression predictions
    # if objectives_cfg.get('polyphony_reg', None) is not None:
    #     pred_polyphony_reg = predictions['polyphony_reg'][0][0]

    # # Handle classification predictions
    # # This expects the polyphony degree to match the index of the class, i.e. class 0 = polyphony 0, class 1 = polyphony 1, etc.
    # if objectives_cfg.get('polyphony_class', None) is not None:
    #     pred_polyphony_class = np.argmax(predictions['polyphony_class'][0])

    # Copy log files and subfolders to archive directory for later analysis
    if os.path.isdir(log_dir):
        print(f"Copying log files and subfolders from {log_dir} to archive directory {archive_log_dir}...")
        shutil.copytree(log_dir, archive_log_dir, dirs_exist_ok=True)
    else:
        print(f"Archive log directory {archive_log_dir} or log directory {log_dir} does not exist. Skipping copy.")
    

if __name__ == "__main__":
    main()


