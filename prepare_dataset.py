import subprocess
import argparse
import json
import os
from dotenv import load_dotenv
from omegaconf import OmegaConf

    # Collect all input features and embeddings used
    # Check if embeddings are included in dataset 
    # Compute missing embeddings

    # Collect all labels used
    # Check if labels are included in dataset
    # Compute missing label

if __name__ == "__main__":

    # Define arguments
    parser = argparse.ArgumentParser(
        description="Updatess dataset when provided with lists of input_features, embeddings and labels by computing missing ones."
    )

    parser.add_argument("--dataset_config", type=str)
    parser.add_argument("--input_features", type=json.loads)
    parser.add_argument("--embeddings", type=json.loads)
    parser.add_argument("--labels", type=json.loads)
    #parser.add_argument("--recompute_embeddings", type=bool, default=False)
    parser.add_argument('--recompute_embeddings', action='store_true')
    #parser.add_argument("--recompute_labels", type=bool, default=False)
    parser.add_argument('--recompute_labels', action='store_true')
    args = parser.parse_args()

    ########################
    # Python versions
    ########################

    # Get docker python paths
    base_python = os.getenv('DOCKER_BASE_PYTHON')
    perch_python = os.getenv('DOCKER_PERCH_PYTHON')
    train_python = os.getenv('DOCKER_TRAIN_PYTHON')

    # Get local python paths from default config if docker paths not defined
    cfg = OmegaConf.load("params.yaml")
    if not base_python:
        base_python = cfg.python.base
    if not perch_python:
        perch_python = cfg.python.perch
    if not train_python:
        train_python = cfg.python.train

    ########################
    # Setup
    ########################
    dataset_config = args.dataset_config
    input_features = args.input_features
    embeddings = args.embeddings
    labels = args.labels
    recompute_embeddings = args.recompute_embeddings 
    recompute_labels = args.recompute_labels

    #########################
    # Embed audio with perch
    ##########################
    # Check if any embeddings are missing

    # 1. Option: Check cfg.embeddings
    if input_features and embeddings:
        cmd =   [
                    perch_python, 
                    "source/perch_embed_audio_stream.py",
                    "--dataset_config", dataset_config,
                    "--input_features", json.dumps(input_features), 
                    "--embeddings", json.dumps(embeddings),
                ]
        if recompute_embeddings: cmd.append("--force_recompute")
        subprocess.run(cmd)

    # 2. Option: Check input features [remove cfg.embeddings]
    # get embedding model and input feature from input_feature_name string

    # --> 3. Option: Define embeddings and train.input_features as hyperparameters <--
    # handle feature_key in train.py:
    # if embeddings:
    # feature key = embeddings + input_feature   
    # else:
    # feature_key = input_feature
    # embeddings config has to be defined as embeddings.yaml


    ###############################################################

    # Collect all input features and embeddings used
    # Check if embeddings are included in dataset 
    # Compute missing embeddings




    #########################
    # Add labels
    ##########################


    # get labels from objectives
