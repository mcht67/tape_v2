#!/bin/bash

# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.

# Description: This script runs an experiment with DVC within a temporary directory copy and pushes the results to the DVC and Git remote.
set -e

# Set environment variables defined in global.env
set -o allexport
source global.env
set +o allexport

# Define DEFAULT_DIR in the host environment
export DEFAULT_DIR="$PWD"
TMP_DIR=tmp

echo "Experiment name: "
echo $EXP_NAME

# Define python paths
if [ -n "$SINGULARITY_CONTAINER" ] || [ -n "$APPTAINER_CONTAINER" ] || [ -f /.dockerenv ]; then
    echo "define python paths to use inside singularity container"
    # In Docker - use Docker venvs from global.env
    BASE_PYTHON="$DOCKER_BASE_PYTHON"
    PERCH_PYTHON="$DOCKER_PERCH_PYTHON"
    TRAIN_PYTHON="$DOCKER_TRAIN_PYTHON"
    COMPLETE_PYTHON="$DOCKER_COMPLETE_PYTHON"
else
    echo "define python paths to use locally"
    # Local - use local venvs from global.env (with DEFAULT_DIR prefix)
    BASE_PYTHON="$DEFAULT_DIR$LOCAL_BASE_PYTHON"
    PERCH_PYTHON="$DEFAULT_DIR$LOCAL_PERCH_PYTHON"
    TRAIN_PYTHON="$DEFAULT_DIR$LOCAL_TRAIN_PYTHON"
    COMPLETE_PYTHON="$DEFAULT_DIR$LOCAL_COMPLETE_PYTHON"
fi

echo "print env"
printenv

echo "export python paths"
export BASE_PYTHON
export PERCH_PYTHON
export TRAIN_PYTHON
export COMPLETE_PYTHON

echo "print env"
printenv

# # Set default python
# source "$BASE_VENV/bin/activate"

if [ -f local.env ]; then
        source local.env;
fi

# Set Hugging Face token as environment variable if available (used for download of dataset and upload of embeddings)
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    export HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN"
    echo "[INFO] Hugging Face token set successfully"
fi

# Set Huggingface cache ENVs
if [ -n "$HF_HOME" ]; then
    export HF_HOME="$HF_HOME"
    echo "[INFO] HF_HOME set successfully"
fi
if [ -n "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HF_HUB_CACHE"
    echo "[INFO] HF_CACHE_HUB set successfully"
fi
if [ -n "$HF_DATASETS_CACHE" ]; then
    export HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
    echo "[INFO] HF_DATASETS_CACHE set successfully"
fi

# Setup a global git configuration if beeing inside a docker container
# Docker containers create a /.dockerenv file in the root directory
if [ -n "$SINGULARITY_CONTAINER" ] || [ -n "$APPTAINER_CONTAINER" ] || [ -f /.dockerenv ]; then
    # if [ -f local.env ]; then
    #     source local.env;
    # fi
    if [ -z "$GIT_USERNAME" ] || [ -z "$GIT_EMAIL" ] || [ -z "$HUGGINGFACE_TOKEN" ] || [ -z "$DOCKERHUB_USERNAME" ]; then
        echo "[ERROR] Please create a local.env with the vars:";
        echo "GIT_USERNAME=MY NAME";
        echo "GIT_EMAIL=myemail@domain.com";
        echo HUGGINGFACE_TOKEN="your_hf_token";
        echo DOCKERHUB_USERNAME="your_dockerhub_username";
        exit 1;
    fi
    echo "set git user config"
    git config --global user.name "$GIT_USERNAME"
    git config --global user.email "$GIT_EMAIL"
    git config --global safe.directory "$PWD"

    # Set dockerhub username as environment variable if available (used in slurm_jobs.sh)
    if [ -n "$DOCKERHUB_USERNAME" ]; then
        export DOCKERHUB_USERNAME="$DOCKERHUB_USERNAME"
        echo "[INFO] Dockerhub Username set successfully"
    fi  
fi

# Create a new sub-directory in the temporary directory for the experiment
echo "Creating temporary sub-directory..." &&
# Generate a unique ID with the current timestamp, process ID, and hostname for the sub-directory
UNIQUE_ID=$(date +%s)-$$-$HOSTNAME &&
EXP_TMP_DIR="$TMP_DIR/$UNIQUE_ID" &&
mkdir -p $EXP_TMP_DIR &&

# Copy the necessary files to the temporary directory
echo "Copying files..." &&
{
# Add all git-tracked files
git ls-files;
if [ -f ".dvc/config.local" ]; then
    echo ".dvc/config.local";
fi;
if [ -f "secrets/dvc-token.json" ]; then
    echo "secrets/dvc-token.json"
else
  echo "WARNING: secrets/dvc-token.json not found" >&2
  ls -la secrets/ >&2   # show what's actually there
fi

echo ".git";
} | while read file; do
    # --chown flag is needed for docker to avoid permission issues
    rsync -aR --chown $(id -u):$(id -g) "$file" $EXP_TMP_DIR;
done &&

# Change the working directory to the temporary sub-directory
cd $EXP_TMP_DIR &&

# Set the DVC cache directory to the shared cache located in the host directory
echo "Setting DVC cache directory..." &&
dvc cache dir $DEFAULT_DIR/.dvc/cache &&

# # Pull the data from the DVC remote repository
# if [ -f "dataset.dvc" ]; then
#     echo "Pulling data with DVC..." 
#     dvc pull dataset;
# fi &&

echo "python path:"
echo $COMPLETE_PYTHON

# Run the experiment with passed parameters. Runs with the default parameters if none are passed.
echo "Running experiment..." &&
dvc exp run \
  --set-param python.complete="$COMPLETE_PYTHON" \
  $EXP_PARAMS
#   --set-param python.base="$BASE_PYTHON" \
#   --set-param python.perch="$PERCH_PYTHON" \
#   --set-param python.train="$TRAIN_PYTHON" \

dvc status

# Push the results to the DVC remote repository
echo "Pushing experiment..." &&
dvc exp push origin && \
echo "✅ Push successful!" || echo "❌ Push failed!"

# Moving everythin to archive
DATETIME=$(date +%Y%m%d_%H%M%S)
if [ -n "$EXP_NAME" ]; then
    ARCHIVE_DIR=${DEFAULT_DIR}/archive/$EXP_NAME/${DATETIME}_$DVC_EXP_NAME
else
    ARCHIVE_DIR=${DEFAULT_DIR}/archive/unnamed_exp/${DATETIME}_$DVC_EXP_NAME
fi

mkdir -p ${ARCHIVE_DIR}/{logs,checkpoints,metrics}

rsync -rv logs/        ${ARCHIVE_DIR}/logs/
rsync -rv checkpoints/ ${ARCHIVE_DIR}/checkpoints/
#rsync -rv metrics/     ${ARCHIVE_DIR}/metrics/

# Clean up the temporary sub-directory
echo "Cleaning up..." &&
cd .. &&
rm -rf $UNIQUE_ID 