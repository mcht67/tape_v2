from omegaconf import OmegaConf
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from hydra.utils import instantiate
from dotenv import load_dotenv
from datasets import load_dataset, Audio, load_from_disk
from itertools import islice
import torch

from utils.logs import get_dvc_exp_name, plot_spectrogram_with_metrics, SummaryWriter
from utils.dataset import load_dataset_with_retry, add_polyphony_range
import integrations.birdset as birdset
import integrations.perch as perch

def main():

    #################################
    # Configuration
    #################################
    cfg = OmegaConf.load("params.yaml")

    study_name = cfg.log.study_name

    huggingface_path = cfg.dataset.huggingface_path
    train_dataset_config = cfg.dataset.train_config
    soundscape_dataset_config = cfg.dataset.soundscape_config

    log_dir = cfg.path.eval_log_dir
    checkpoint_dir = cfg.path.checkpoint_dir
    os.makedirs(log_dir, exist_ok=True)

    model_cfg = cfg.model
    objectives_cfg = cfg.objectives
    model_cfg.objectives_cfg = objectives_cfg
    labels = [objectives_cfg[x]['label'] for x in objectives_cfg]
    input_feature_name = cfg.train.input_feature_name
    input_feature = cfg.train.input_feature
    embedding_type = cfg.embeddings.type
    embedding_dim_type = cfg.embeddings.dimension_type

    num_examples = cfg.evaluation.num_examples

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

    # train_dataset = load_dataset(huggingface_path, train_dataset_config, token=huggingface_token)
    train_dataset = load_from_disk('data/HSN')

    if train_dataset is None:
        raise RuntimeError("Dataset failed to load after all retry attempts. Check network/cache or force redownload in dataset preparation.")

    # train_split, val_split, test_split

    #################################
    # Add labels
    #################################
    # TODO: do per example to avoid downloading entire dataset / is already been done in train.py

    # Get input dim
    # input_dim = tf.squeeze(np.array(train_dataset['train'][0][input_feature_name])).shape

    # Compute additional labels
    # time_dim = input_dim[0] if len(input_dim) > 1 else None
    # freq_dim = input_dim[1] if len(input_dim) > 2 else None
    # train_dataset, added_labels = add_labels(train_dataset, labels, time_dim=time_dim, freq_dim=freq_dim)

    # print("Added labels: ", added_labels)

    # existing_labels = added_labels + ['polyphony_degree']
    # missing_labels = set(labels) ^ set(existing_labels)
    # if missing_labels:
    #     raise Exception("Not all requested labels could be computed.")

    #################################
    # Load model
    #################################

    # Load best model from checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "best.weights.h5")
    # TODO: remove after testing
    checkpoint_path = 'archive/Pooled-Embeddings-Comparison/20260505_075038_/checkpoints/best_weights/best.weights.h5'

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found at {checkpoint_path}. Please make sure to run the training script first to save the best model checkpoint for later evaluation.")
    
    # Define model
    model = instantiate(cfg.model)

    # Build model by calling it on a sample input
    first_example = next(iter(train_dataset['train']))
    print("input features", train_dataset['train'].features)
    input_dim = int(tf.squeeze(np.array(first_example[input_feature_name])).shape[0])
    sample_input = tf.zeros((1, input_dim), dtype=tf.float32)
    _ = model(sample_input, training=False)

    print(f"Loading weights from {checkpoint_path}")
    model.load_weights(checkpoint_path)

    ###########################################
    # Example visualization in TensorBoard
    ###########################################

    for split_name in train_dataset.keys():

        for example_idx in range(num_examples):

            example = train_dataset[split_name][example_idx]
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
                events=all_events,
                filename=example.get('filename', None)
            )
            
            writer.add_figure(f'{split_name}', fig, global_step=example_idx) # 'example_{idx}'
            plt.close(fig)

    # Flush to ensure all figures are written
    writer.flush()

    ###########################################
    # Metrics computation 
    ###########################################


    for split_name in train_dataset.keys():

        for example_idx in range(num_examples):

            example = train_dataset[split_name][example_idx]
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



    

if __name__ == "__main__":
    main()


