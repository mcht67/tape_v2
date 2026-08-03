from omegaconf import OmegaConf
import os
import numpy as np
import tensorflow as tf
from hydra.utils import instantiate
from dotenv import load_dotenv
from datasets import Audio, load_from_disk
import torch
import pandas as pd
from functools import partial
import shutil

from utils.logs import SummaryWriter, get_log_paths, get_archive_paths, save_to_report, plot_polyphony_distribution
from utils.dataset import add_min_max_polyphony, get_local_data_dir, get_birdset_id2label
from utils.evaluation import collect_predictions, arrays_to_records, build_update_metrics_table
from utils.metrics import compute_polyphony_range_metrics
import integrations.birdset as birdset
import integrations.perch as perch

from utils.torch_evaluation import load_torch_model_for_eval, collect_predictions_torch

# Disable caching to avoid huggingface caching issues when running multiple experiments in parallel
from datasets import disable_caching
disable_caching()

def main():

    #################################
    # Configuration
    #################################
    cfg = OmegaConf.load("params.yaml")

    study_name = cfg.log.study_name
    subset = cfg.dataset.subset

    huggingface_path = cfg.dataset.huggingface_path
    train_dataset_config = cfg.dataset.train_config
    soundscape_dataset_config = cfg.dataset.soundscape_config

    log_paths = get_log_paths(cfg)
    log_dir = log_paths['soundscape_eval_log_dir']
    os.makedirs(log_dir, exist_ok=True)

    archive_paths = get_archive_paths(cfg)
    archive_log_dir = archive_paths['soundscape_eval_log_dir']
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
    embedding_type = cfg.embeddings.type
    embedding_dim_type = cfg.embeddings.dimension_type

    if subset == "XCM" or subset == "XCL":
        print("Note: The XCM and XCL datasets do not have soundscape data. Skipping evaluation on soundscape data.")
        # Create empty test directory for consistent dvc tracking
        os.makedirs(log_dir, exist_ok=True)
        return
    
    #################################
    # Setup
    #################################

    # Get summary writer for TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # Store config in file/logs
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.save(cfg, os.path.join(log_dir, "params.yaml"))

    #################################
    # Load soundscape dataset
    #################################

    # Load environment variables from .env file
    load_dotenv('local.env')
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

    print("default_dir:", default_dir)
    scape_test_config = subset + '_soundscape_test'
    print("scape_test_config:", scape_test_config)
    print("subset:", subset)
    local_data_dir = get_local_data_dir(dataset_config=scape_test_config, subset=subset)
    print("local_data_dir:", local_data_dir)
    dataset_dir = os.path.join(default_dir, local_data_dir)
    print("dataset_dir:", dataset_dir)

    soundscape_test_dataset = load_from_disk(dataset_dir)
 
    soundscape_test5s_split = soundscape_test_dataset['test_5s']
    print(soundscape_test5s_split)

    #################################
    # Update objectives config based on dataset
    #################################

    ebird_class_labels = None
    num_species = None

    # if 'species_polyphony_reg' in objectives_cfg or 'species_polyphony_class' in objectives_cfg:
        
        #TODO: get from ClassLabels in dataset
        # ebird_class_labels = test_dataset.features['ebird_code_multilabel'].feature.names
        # Get birdset ids
        # scape_ds = load_dataset(huggingface_path, soundscape_dataset_config, split='test_5s', token=huggingface_token, download_mode='force_redownload')
    birdset_id2label = get_birdset_id2label(subset, dataset=soundscape_test5s_split)
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

    if backend == "torch":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint not found at {checkpoint_path}. Please make sure to run torch_train.py first to save the best checkpoint for later evaluation.")

        # model = load_torch_model_for_eval(cfg.model, objectives_cfg, checkpoint_path, device)
        model = load_torch_model_for_eval(
            cfg.model, cfg.head, objectives_cfg, checkpoint_path, device
        )
        print(f"Loaded torch checkpoint from {checkpoint_path}")

    else:
        # Load best model from checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, "best.weights.h5")

        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint not found at {checkpoint_path}. Please make sure to run the training script first to save the best model checkpoint for later evaluation.")

        # Define model
        model_cfg.objectives_cfg = objectives_cfg
        model = instantiate(cfg.model)

        # Build model by calling it on a sample input
        first_example = soundscape_test5s_split[0]
        print("input features", soundscape_test5s_split.features)
        # input_dim = int(tf.squeeze(np.array(first_example[input_feature_name])).shape[0])
        # sample_input = tf.zeros((1, input_dim), dtype=tf.float32)
        # _ = model(sample_input, training=False)

        first_example_arr = np.array(first_example[input_feature_name])
        print("raw feature shape:", first_example_arr.shape)

        # Preserve full shape except batch dim, add batch dim of 1
        sample_shape = (1,) + first_example_arr.shape
        sample_input = tf.zeros(sample_shape, dtype=tf.float32)
        print("sample_input shape:", sample_input.shape)

        _ = model(sample_input, training=False)

        print(f"Loading weights from {checkpoint_path}")
        model.load_weights(checkpoint_path)
        print("Model loaded successfully.")

    #########################################
    # Validation on soundscape data
    #########################################

    # Polyphony degree can not be computed for soundscapes directly, but we can compute a minimum and maximum polyphony degree
   
    # Minimum polyphony degree: get maximum number of overlapping events at any time
    # Minimum polyphony degree: get number of species active in soundscape
    # Maximum polyphony degree: get total number of events that occur in the soundscape

    # Check if embeddings have been precomputed
    if not input_feature in soundscape_test5s_split.features:
        print(f"Feature '{input_feature}' not found in soundscape dataset. Can not run evaluation.")
        return
    
    # Cast input_feature to Audio
    soundscape_test5s_split = soundscape_test5s_split.cast_column(input_feature, Audio())

    # Add min/max polyphony labels to soundscape dataset
    # soundscape_test5s_split = soundscape_test5s_split.map(partial(add_min_max_polyphony, num_species=num_species))

    plot_save_dir = os.path.join(cfg.path.soundscape_eval_output, f"{subset}_soundscape_polyphony_distribution.png")
    os.makedirs(os.path.dirname(plot_save_dir), exist_ok=True)
    plot_polyphony_distribution(soundscape_test5s_split, save_path=plot_save_dir)

    embeddings_precomputed = input_feature_name in soundscape_test5s_split.features

    if backend == "torch" or embeddings_precomputed:

        if backend == "torch":
            # The fine-tuned torch model embeds + predicts end-to-end from raw
            # audio in one batched pass -- there's no separate "precomputed vs.
            # on-the-fly embedding" distinction to make.
            print("Running the torch model end-to-end on raw audio (batched) for soundscape evaluation...")
            truth_columns = ["min_polyphony", "max_polyphony"]
            if num_species:
                truth_columns += ["min_species_polyphony", "max_species_polyphony"]
            y_true, predictions, variable_values = collect_predictions_torch(
                model, soundscape_test5s_split, input_feature, objectives_cfg, device,
                batch_size=cfg.train.get("eval_batch_size", 32), variables=[],
                truth_columns=truth_columns, birdset_id2label=birdset_id2label,
            )
        else:
            print(f"Embeddings have been precomputed and stored in feature '{input_feature_name}'. Using precomputed embeddings for evaluation.")
            y_true, predictions, variable_values = collect_predictions(model, soundscape_test5s_split, input_feature_name, birdset_id2label=birdset_id2label)

        # Store predictions
        print("y_true:", y_true)
        print("predictions:", predictions)
        print("variable_values:", variable_values)

        # Define ouput dir for metrics and results
        out_dir = os.path.join(default_dir, "archive", cfg.log.study_name, cfg.path.study_subfolder, "scape_eval_results")
        os.makedirs(out_dir, exist_ok=True)

        # Store results in a pickle file for later analysis
        records = arrays_to_records(y_true, predictions, variable_values)
        df = pd.json_normalize(records)
        df.to_pickle(os.path.join(log_dir, f"test_results.pkl"))

        if "polyphony_reg" in predictions:
            df.to_pickle(os.path.join(out_dir, f"{subset}_reg_scape_test_results.pkl"))
        if "polyphony_class" in predictions:
            df.to_pickle(os.path.join(out_dir, f"{subset}_class_scape_test_results.pkl"))
        if "species_polyphony_reg" in predictions:
            df.to_pickle(os.path.join(out_dir, f"{subset}_species_reg_scape_test_results.pkl"))
        if "species_polyphony_class" in predictions:
            df.to_pickle(os.path.join(out_dir, f"{subset}_species_class_scape_test_results.pkl"))

        report = {}
        num_classes = cfg.dataset.max_polyphony + 1

        if "polyphony_reg" in predictions:
            report["polyphony_reg"] = compute_polyphony_range_metrics(
                y_true, predictions["polyphony_reg"][:, None], min_key="min_polyphony", max_key="max_polyphony",
                cm_type="regression_round", per_species=False)

        if "polyphony_class" in predictions:
            report["polyphony_class"] = compute_polyphony_range_metrics(
                y_true, predictions["polyphony_class"][:, None], min_key="min_polyphony", max_key="max_polyphony",
                cm_type="classification", num_classes=num_classes, per_species=False)
            
        if "species_polyphony_reg" in predictions:
            report["species_polyphony_reg"] = compute_polyphony_range_metrics(
                y_true, predictions["species_polyphony_reg"], min_key="min_species_polyphony", max_key="max_species_polyphony",
                cm_type="species_regression_round", per_species=False)

        if "species_polyphony_class" in predictions:
            report["species_polyphony_class"] = compute_polyphony_range_metrics(
                y_true, predictions["species_polyphony_class"], min_key="min_species_polyphony", max_key="max_species_polyphony",
                cm_type="species_classification", num_classes=num_classes, per_species=False)
        
        save_to_report(report, os.path.join(log_dir, "test_metrics.json"))

        # # Save to metrics overview table
        # for objective in report:
        #     if objective == "polyphony_reg" or objective == "species_polyphony_reg":
        #         obj_key = "reg"
        #     elif objective == "polyphony_class" or objective == "species_polyphony_class":
        #         obj_key = "class"
        #     metrics_dict = report[objective]
        #     table_name = f"soundscape_test_metrics"
        #     csv_path = os.path.join(out_dir, f"{table_name}.csv")

        #     if study_name == "Pooled-Embeddings-SNR-Range-Effects":
        #         column_name = f"{subset}_{obj_key}_{cfg.dataset.snr_range[0]}-{cfg.dataset.snr_range[1]}"
        #     elif study_name == "Pooled-Embeddings-Min-SNR-Effects":
        #         column_name = f"{subset}_{obj_key}_{cfg.dataset.min_snr}"
        #     elif study_name == "Pooled-Embeddings-Max-Polyphony-Effects":
        #         column_name = f"{subset}_{obj_key}_{cfg.dataset.max_polyphony}"
        #     else:
        #         column_name = f"{subset}_{obj_key}"

        #     df, csv_path = update_metrics_table(column_name, metrics_dict, csv_path)
        #     print(f"Saved {table_name} to {csv_path}")

        build_update_metrics_table(report, cfg, out_dir, table_name="soundscape_test_metrics")
     
    else:
        print(f"Embeddings have not been precomputed. Computing embeddings on-the-fly using embedding type '{embedding_type}' and dimension type '{embedding_dim_type}'.")
        # Prepare embedding model if necessary
        if not embeddings_precomputed:

            embedding_type = cfg.embeddings.type
            
            if embedding_type=='birdset':

                # Request GPU
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Using device: {device}")

                # Define embedding model
                embedding_model_cfg = cfg.embeddings.model_cfg
                embedding_model = instantiate(embedding_model_cfg)

                if hasattr(embedding_model, 'to'):
                    embedding_model = embedding_model.to(device) 
            elif embedding_type=='perch_v1' or embedding_type=='perch_v2':

                model_key = cfg.embeddings.name

                # Auto detect gpu
                gpus = tf.config.list_physical_devices('GPU')
                device = '/GPU:0' if gpus else '/CPU:0'
                print(f"Using device: {device}")

                # Prevent TF from grabbing all GPU memory at once
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                
                # Load model
                if embedding_type == 'perch_v1':
                    embedding_model, sampling_rate = perch.load_perch1_model(model_key)
                elif embedding_type == 'perch_v2':
                    embedding_model, sampling_rate = perch.load_perch2_model(model_key)
                elif embedding_type == 'birdset':
                    embedding_model, sampling_rate = perch.load_birdset_model(model_key)
                else:
                    print(f"Model family unknown. Can not load model {model_key}. Skipping.")

        polyphony_range_logits_reg = []
        distances_to_min_polyphony_reg = []
        polyphony_range_logits_class = []
        distances_to_min_polyphony_class = []

        # Get predictions and metrics on soundscape data   
        # for idx, example in enumerate(islice(soundscape_dataset['test_5s'], 100)):
        for example in soundscape_test5s_split['test_5s']:

            if embeddings_precomputed:
                embedding = example[input_feature_name]
            else:
                audio = example[input_feature]

                # Compute embedding for example
                if embedding_type=='birdset':
                    outputs = birdset.compute_embedding(audio, embedding_model, device)
                    pooled_embedding = outputs.pooled_embeddings.cpu().numpy()
                    spatial_embedding = outputs.spatial_embeddings.cpu().numpy()
                elif embedding_type=='perch_hoplite' or embedding_type=='perch_v2':
                    pooled_embedding, spatial_embedding = perch.compute_embedding(audio, embedding_model, model_key, embedding_type, sampling_rate, device=device)

                if embedding_dim_type == 'pooled':
                    embedding = pooled_embedding
                elif embedding_dim_type == 'spatial':
                    embedding = spatial_embedding

            # Make prediction
            single_input = np.expand_dims(embedding, axis=0)
            single_input = tf.constant(single_input, dtype=tf.float32)
            predictions = model.predict(single_input)
            # print(predictions)

            # Get polyphony ground truth
            gt_min_polyphony = example['min_polyphony']
            gt_max_polyphony = example['max_polyphony']

            # Handle regression predictions
            if objectives_cfg.get('polyphony_reg', None) is not None:
                pred_polyphony_reg = predictions['polyphony_reg'][0][0]
                print(f"Predicted polyphony (regression): {pred_polyphony_reg}, Ground truth min polyphony: {gt_min_polyphony}, Ground truth max polyphony: {gt_max_polyphony}")

                # Get polyphony logit
                polyphony_range_logit = int(gt_min_polyphony <= pred_polyphony_reg <= gt_max_polyphony)
                polyphony_range_logits_reg.append(polyphony_range_logit)

                # Get distance to min polyphony
                distance_to_min_polyphony_reg = abs(pred_polyphony_reg - gt_min_polyphony)
                distances_to_min_polyphony_reg.append(distance_to_min_polyphony_reg)

            # Handle classification predictions
            # This expects the polyphony degree to match the index of the class, i.e. class 0 = polyphony 0, class 1 = polyphony 1, etc.
            if objectives_cfg.get('polyphony_class', None) is not None:
                pred_polyphony_class = np.argmax(predictions['polyphony_class'][0])
                print(f"Predicted polyphony class: {pred_polyphony_class}, Ground truth min polyphony: {gt_min_polyphony}, Ground truth max polyphony: {gt_max_polyphony}")

                # Get polyphony logit
                polyphony_range_logit = int(gt_min_polyphony <= pred_polyphony_class <= gt_max_polyphony)
                polyphony_range_logits_class.append(polyphony_range_logit)         

                # Get distance to min polyphony
                distance_to_min_polyphony_class = abs(pred_polyphony_class - gt_min_polyphony)
                distances_to_min_polyphony_class.append(distance_to_min_polyphony_class)

        if objectives_cfg.get('polyphony_reg', None) is not None:
            # Get metrics for regression
            polyphony_range_accuracy_reg = np.mean(polyphony_range_logits_reg)
            mean_distance_to_min_polyphony_reg = np.mean(distances_to_min_polyphony_reg)
            print(f"Polyphony range accuracy (regression): {polyphony_range_accuracy_reg:.4f}")
            print(f"Mean distance to min polyphony (regression): {mean_distance_to_min_polyphony_reg:.4f}")

            # Write metrics to TensorBoard
            writer.add_scalar('soundscape/polyphony_range_accuracy', polyphony_range_accuracy_reg, global_step=0)
            writer.add_scalar('soundscape/mean_distance_to_min_polyphony', mean_distance_to_min_polyphony_reg, global_step=0)

        if objectives_cfg.get('polyphony_class', None) is not None:
            # Get metrics for classification
            polyphony_range_accuracy_class = np.mean(polyphony_range_logits_class)
            mean_distance_to_min_polyphony_class = np.mean(distances_to_min_polyphony_class)
            print(f"Polyphony range accuracy (classification): {polyphony_range_accuracy_class:.4f}")
            print(f"Mean distance to min polyphony (classification): {mean_distance_to_min_polyphony_class:.4f}")

            # Write metrics to TensorBoard
            writer.add_scalar('soundscape/polyphony_range_accuracy_class', polyphony_range_accuracy_class, global_step=0)
            writer.add_scalar('soundscape/mean_distance_to_min_polyphony_class', mean_distance_to_min_polyphony_class, global_step=0)

    # Copy log files and subfolders to archive directory for later analysis
    if os.path.exists(archive_log_dir) and os.path.isdir(archive_log_dir) and os.path.exists(log_dir) and os.path.isdir(log_dir):
        print(f"Copying log files and subfolders from {log_dir} to archive directory {archive_log_dir}...")
        shutil.copytree(log_dir, archive_log_dir, dirs_exist_ok=True)
    else:
        print(f"Archive log directory {archive_log_dir} or log directory {log_dir} does not exist. Skipping copy.")

if __name__ == "__main__":
    main()


