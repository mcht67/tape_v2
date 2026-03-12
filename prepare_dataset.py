import subprocess
import argparse
import json
import sys

    # Collect all input features and embeddings used
    # Check if embeddings are included in dataset 
    # Compute missing embeddings

    # Collect all labels used
    # Check if labels are included in dataset
    # Compute missing labels


if __name__ == "__main__":

    # Define arguments
    parser = argparse.ArgumentParser(
        description="Prepares dataset when provided with lists of input_features, embeddings and labels by computing missing ones."
    )

    parser.add_argument("--input_features", type=json.loads)
    parser.add_argument("--embeddings", type=json.loads)
    parser.add_argument("--labels", type=json.loads)
    parser.add_argument("--recompute_embeddings", type=bool)
    parser.add_argument("--recompute_labels", type=bool)
    args = parser.parse_args()

    ########################
    # Setup
    ########################
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
        subprocess.run(["stage-venvs/perch-venv/bin/python", 
                        "source/perch_embed_audio_stream.py", 
                        "--input_features", json.dumps(input_features), 
                        "--embeddings", json.dumps(embeddings),
                        ])

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
