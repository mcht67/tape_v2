#!/bin/bash

# Copyright 2026 cohrt
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.

# Job name and logs
#SBATCH -J tustu
#SBATCH -D /beegfs/scratch/cohrt/tape_v2/ # Working Directory
#SBATCH --output=./logs/slurm/slurm-%j.out

# Resources needed
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --time=01:00:00
#SBATCH --partition=standard

# Get email notifications for job status
#SBATCH --mail-type=ALL
#SBATCH --mail-user=malte.crt@gmail.com

# source ./slurm_common.sh "$@"

# Debugging options
set -euo pipefail
set -x

echo "Running slurm job"

whoami

# Load necessary modules
module load singularity/4.3.7

# Set environment variables defined in global.env
set -o allexport
source global.env
set +o allexport

# Define DEFAULT_DIR in the host environment
export DEFAULT_DIR="$(realpath $PWD)"

# # Define python paths
# if [ -n "${SINGULARITY_CONTAINER:-}" ] || [ -n "${APPTAINER_CONTAINER:-}" ] || [ -f /.dockerenv ]; then
#     echo "define python paths to use inside singularity container"
#     # In Docker - use Docker venvs from global.env
#     # BASE_PYTHON="$DOCKER_BASE_PYTHON"
#     # PERCH_PYTHON="$DOCKER_PERCH_PYTHON"
#     # TRAIN_PYTHON="$DOCKER_TRAIN_PYTHON"
#     COMPLETE_PYTHON="$DOCKER_COMPLETE_PYTHON"
# else
#     echo "define python paths to use locally"
#     # Local - use local venvs from global.env (with DEFAULT_DIR prefix)
#     # BASE_PYTHON="$DEFAULT_DIR$LOCAL_BASE_PYTHON"
#     # PERCH_PYTHON="$DEFAULT_DIR$LOCAL_PERCH_PYTHON"
#     # TRAIN_PYTHON="$DEFAULT_DIR$LOCAL_TRAIN_PYTHON"
#     COMPLETE_PYTHON="$DEFAULT_DIR$LOCAL_COMPLETE_PYTHON"
# fi

echo "export python paths"
# export BASE_PYTHON
# export PERCH_PYTHON
# export TRAIN_PYTHON
COMPLETE_PYTHON="$DOCKER_COMPLETE_PYTHON"
export COMPLETE_PYTHON

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

singularity exec \
    --bind $HF_HOME:$HF_HOME \
    --bind $DEFAULT_DIR \
    --pwd $DEFAULT_DIR \
    $PROJECT_NAME-image-latest$container_extension \
    $COMPLETE_PYTHON ./prepare_dataset.py "$@"