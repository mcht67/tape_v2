from omegaconf import OmegaConf
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from hydra.utils import instantiate
from dotenv import load_dotenv
from datasets import load_dataset, Audio, load_from_disk

from utils.dataset import get_birdset_id2label, get_local_data_dir
from utils.logs import SummaryWriter, save_to_report, get_log_paths
from utils.metrics import compute_polyphony_metrics
from utils.evaluation import arrays_to_records, collect_predictions, update_metrics_table

import integrations.birdset as birdset
import integrations.perch as perch

def main():

    #################################
    # Configuration
    #################################
    cfg = OmegaConf.load("params.yaml")

    study_name = cfg.log.study_name

    huggingface_path = cfg.dataset.huggingface_path
    train_config = cfg.dataset.train_config
    soundscape_dataset_config = cfg.dataset.soundscape_config
    subset = cfg.dataset.subset

    log_paths = get_log_paths(cfg)
    log_dir = log_paths['eval_log_dir']
    os.makedirs(log_dir, exist_ok=True)
    default_dir = os.environ.get('DEFAULT_DIR', '')
    checkpoint_dir = log_paths['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        checkpoint_dir = os.path.join(default_dir, checkpoint_dir)
    
    model_cfg = cfg.model
    objectives_cfg = cfg.objectives

    input_feature_name = cfg.train.input_feature_name
    input_feature = cfg.train.input_feature
    embedding_type = cfg.embeddings.type
    embedding_dim_type = cfg.embeddings.dimension_type

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
    test_dataset = dataset['test']

    if test_dataset is None:
        raise RuntimeError("Dataset failed to load after all retry attempts. Check network/cache or force redownload in dataset preparation.")

    # Filter dataset by polyphony degree if specified in the config
    if 'max_polyphony' in cfg.dataset and cfg.dataset.max_polyphony is not None:
        max_polyphony = cfg.dataset.max_polyphony
        print(f"Filtering test dataset to include only examples with polyphony degree <= {max_polyphony}...")
        test_dataset = test_dataset.filter(lambda x: x['polyphony_degree'] <= max_polyphony)
        print(f"After filtering, test split has {len(test_dataset)} examples.")

    # Filter dataset by SNR if specified in the config
    if 'snr_range' in cfg.dataset and cfg.dataset.snr_range is not None:
        snr_range = cfg.dataset.snr_range
        print(f"Filtering test dataset to include only examples with SNR in range {snr_range}...")
        test_dataset = test_dataset.filter(lambda x: snr_range[0] <= x['snr_dB'] <= snr_range[1])
        print(f"After filtering, test split has {len(test_dataset)} examples.")
    if 'min_snr' in cfg.dataset and cfg.dataset.min_snr is not None:
        min_snr = cfg.dataset.min_snr
        print(f"Filtering test dataset to include only examples with SNR >= {min_snr}...")
        test_dataset = test_dataset.filter(lambda x: x['snr_dB'] >= min_snr)
        print(f"After filtering, test split has {len(test_dataset)} examples.")
    
    #################################
    # Update objectives config based on dataset
    #################################

    ebird_class_labels = None

    # if 'species_polyphony_reg' in objectives_cfg or 'species_polyphony_class' in objectives_cfg:
        
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

    # Set objectives config in model config for easy access when building model and losses
    model_cfg.objectives_cfg = objectives_cfg


    #################################
    # Load model
    #################################

    # Load best model from checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "best.weights.h5")
    # TODO: remove after testing
    # checkpoint_path = 'archive/Pooled-Embeddings/perch_v2_cpu/20260703_172045_level-arcs/checkpoints/best.weights.h5'
    # checkpoint_path = 'archive/Pooled-Embeddings/perch_v2_cpu/20260703_172515_brood-weld/checkpoints/best.weights.h5'
    # checkpoint_path = 'archive/Pooled-Embeddings/perch_v2_cpu/20260609_012305_bosom-byes/checkpoints/best.weights.h5'

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found at {checkpoint_path}. Please make sure to run the training script first to save the best model checkpoint for later evaluation.")
    
    # Define model
    model = instantiate(cfg.model)

    # Build model by calling it on a sample input
    first_example = test_dataset[0]
    print("input features", test_dataset.features)
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
    y_true, predictions, variable_values = collect_predictions(model, test_dataset, input_feature_name, variables=['snr_dB'], birdset_id2label=birdset_id2label)
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
        
    save_to_report(report, os.path.join(log_dir, "test_metrics.json"))

    # Save to metrics overview table
    for objective in report:
        if objective == "polyphony_reg" or objective == "species_polyphony_reg":
            obj_key = "reg"
        elif objective == "polyphony_class" or objective == "species_polyphony_class":
            obj_key = "class"
        metrics_dict = report[objective]
        table_name = f"test_metrics"

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
    #     gt_polyphony = example['polyphony_degree']

    # # Handle regression predictions
    # if objectives_cfg.get('polyphony_reg', None) is not None:
    #     pred_polyphony_reg = predictions['polyphony_reg'][0][0]

    # # Handle classification predictions
    # # This expects the polyphony degree to match the index of the class, i.e. class 0 = polyphony 0, class 1 = polyphony 1, etc.
    # if objectives_cfg.get('polyphony_class', None) is not None:
    #     pred_polyphony_class = np.argmax(predictions['polyphony_class'][0])
    

if __name__ == "__main__":
    main()


