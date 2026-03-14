import subprocess
import argparse
import json
import os
# from dotenv import load_dotenv
from omegaconf import OmegaConf
# import huggingface_hub

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

    parser.add_argument("--huggingface_path", type=str)
    parser.add_argument("--dataset_config", type=str)
    parser.add_argument("--input_features", type=json.loads)
    parser.add_argument("--embeddings", type=json.loads)
    parser.add_argument("--labels", type=json.loads)
    parser.add_argument('--objectives', type=json.loads)
    parser.add_argument('--recompute_embeddings', action='store_true')
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
    huggingface_path = args.huggingface_path
    dataset_config = args.dataset_config
    input_features = args.input_features
    embeddings = args.embeddings
    labels = args.labels
    objectives = args.objectives
    recompute_embeddings = args.recompute_embeddings 
    recompute_labels = args.recompute_labels

    ##########################
    # Embed audio with perch
    ##########################

    if input_features and embeddings:
        cmd =   [
                    perch_python, 
                    "source/perch_embed_audio_stream.py",
                    "--huggingface_path", huggingface_path,
                    "--dataset_config", dataset_config,
                    "--input_features", json.dumps(input_features), 
                    "--embeddings", json.dumps(embeddings),
                ]
        if recompute_embeddings: cmd.append("--force_recompute")
        subprocess.run(cmd)

    ###########################
    # Embed audio with birdset
    ###########################

    if input_features and embeddings:
        cmd =   [
                    train_python,
                    "source/birdset_embed_audio_stream.py",
                    "--huggingface_path", huggingface_path,
                    "--dataset_config", dataset_config,
                    "--input_features", json.dumps(input_features), 
                    "--embeddings", json.dumps(embeddings),
                ]
        if recompute_embeddings: cmd.append("--force_recompute")
        subprocess.run(cmd)

    # DO IN TRAIN 
    # #########################
    # # Add labels
    # ##########################

    # if objectives:
    #     cmd =   [
    #                 base_python, 
    #                 "source/add_labels_stream.py",
    #                 "--huggingface_path", huggingface_path,
    #                 "--dataset_config", dataset_config,
    #                 "--objectives", json.dumps(objectives), 
    #             ]
    #     if recompute_labels: cmd.append("--force_recompute")
    #     subprocess.run(cmd)

