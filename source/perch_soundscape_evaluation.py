from omegaconf import OmegaConf
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from hydra.utils import instantiate
from dotenv import load_dotenv
from datasets import load_dataset, Audio
from itertools import islice
import torch

from utils.logs import get_dvc_exp_name, plot_spectrogram_with_metrics, SummaryWriter
from utils.dataset import load_dataset_with_retry, add_polyphony_range

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

num_examples = cfg.evaluation.num_examples

#################################
# Setup
#################################

# Get summary writer for TensorBoard
writer = SummaryWriter(log_dir=log_dir)

# Store config in file/logs
print(OmegaConf.to_yaml(cfg))
OmegaConf.save(cfg, os.path.join(log_dir, "params.yaml"))

#################################
# Load dataset
#################################

# Load environment variables from .env file
load_dotenv('local.env')
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

from huggingface_hub import HfApi
api = HfApi()
info = api.dataset_info("mcht67/Polyphonic-BirdSet-train", token=huggingface_token)
for config in info.card_data.get("configs", []):
    if config.get("config_name") == train_dataset_config:
        print(config.get("data_files"))
        print(config.get("data_dir"))

# Load Dataset
print(f"[INFO] HF_DATASETS_OFFLINE={os.environ.get('HF_DATASETS_OFFLINE', 'NOT SET')} (ommits updating datasets to avoid hitting rate limit on Huggingface Hub)")
train_dataset = load_dataset(huggingface_path, train_dataset_config, token=huggingface_token, streaming=True)

if train_dataset is None:
    raise RuntimeError("Dataset failed to load after all retry attempts. Check network/cache or force redownload in dataset preparation.")

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

#########################################
    # Validation on soundscape data
    #########################################

    # soundscape test_5s has start/end_time and low/high_freq 

    # Load soundscape data
    soundscape_dataset = load_dataset_with_retry(huggingface_path, soundscape_dataset_config, token=huggingface_token)

    # TODO: Add labels for multi task setup
    print(soundscape_dataset['test_5s'])
    # Polyphony degree can not be computed for soundscapes directly, but we can compute a minimum and maximum polyphony degree

    # Add polyphony range labels to soundscape dataset
    # soundscape_dataset = soundscape_dataset.map(add_polyphony_range)
   
    # Minimum polyphony degree: get maximum number of overlapping events at any time
    # Minimum polyphony degree: get number of species active in soundscape
    # Maximum polyphony degree: get total number of events that occur in the soundscape

    # Check if embeddings have been precomputed
    embeddings_precomputed = input_feature_name in soundscape_dataset['test_5s'].features

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
                model, sampling_rate = load_perch1_model(model_key)
            elif embedding_type == 'perch_v2':
                model, sampling_rate = load_perch2_model(model_key)
            elif embedding_type == 'birdset':
                load_birdset_model = load_birdset_model(model_key)
            else:
                print(f"Model family unknown. Can not load model {model_key}. Skipping.")
                



    polyphony_range_logits = []
    distances_to_min_polyphony = []

    # Get predictions and metrics on soundscape data   
    for idx, example in enumerate(islice(soundscape_dataset['test_5s'])):

        if embeddings_precomputed:
            embedding = example[input_feature_name]
        else:
            # Compute embedding for example
            embedding = compute_embedding(example[input_feature_name], model)
        
        # Make prediction
        single_input = np.expand_dims(embedding, axis=0)
        single_input = tf.constant(single_input, dtype=tf.float32)
        predictions = model.predict(single_input)
        pred_polyphony = predictions['polyphony_degree'][0][0]

        # Get polyphony logit
        gt_min_polyphony = example['min_polyphony']
        gt_max_polyphony = example['max_polyphony']
        polyphony_range_logit = int(gt_min_polyphony <= pred_polyphony <= gt_max_polyphony)
        polyphony_range_logits.append(polyphony_range_logit)

        # Get distance to min polyphony
        distance_to_min_polyphony = abs(pred_polyphony - gt_min_polyphony)
        distances_to_min_polyphony.append(distance_to_min_polyphony)

    # Get metrics on soundscape data
    polyphony_range_accuracy = np.mean(polyphony_range_logit)
    mean_distance_to_min_polyphony = np.mean(distances_to_min_polyphony)

    # Write metrics to TensorBoard
    writer.add_scalar('soundscape/polyphony_range_accuracy', polyphony_range_accuracy, global_step=0)
    writer.add_scalar('soundscape/mean_distance_to_min_polyphony', mean_distance_to_min_polyphony, global_step=0)