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

echo "Running dataset preparation slurm job"

whoami

#################################
# Handle options
#################################

# Default variable values
rebuild_container=false
sif_container=false

# Function to display script usage
usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo " -h, --help                Display this help message"
  echo " -b, --rebuild-container   Force the rebuild of the singularity container (default: false)"
  echo " -s, --sif-container       Build the singularity container as SIF (Singularity Image Format) file (default: false)"
}

# Function to handle options and arguments
handle_options() {
  while [ $# -gt 0 ]; do
    case $1 in
      -h | --help)
        usage
        exit 0
        ;;
      -b | --rebuild-container)
        rebuild_container=true
        ;;
      -s | --sif-container)
        sif_container=true
        ;;
      *)
        echo "Invalid option: $1" >&2
        usage
        exit 1
        ;;
    esac
    shift
  done
}

# Main script execution
handle_options "$@"

# Perform the desired actions based on the provided flags and arguments
if [ "$rebuild_container" = true ]; then
  echo "Forcing the rebuild of the singularity container..."
fi

if [ "$sif_container" = true ]; then
  echo "Singularity container format set to SIF (Singularity Image Format) file..."
  echo "When executed, the container will be converted to a temporary sandboxed image. This may take a while..."
fi

#################################
# Load modules
#################################

# Load necessary modules
module load singularity/4.3.7

#################################
# Import environment variables
#################################

# Set environment variables defined in global.env
set -o allexport
source global.env
set +o allexport

# Print info
[ -n "$HF_HOME" ] && echo "[INFO] HF_HOME=$HF_HOME" || echo "[WARNING] HF_HOME not set"
[ -n "$HF_HUB_CACHE" ] && echo "[INFO] HF_HUB_CACHE=$HF_HUB_CACHE" || echo "[WARNING] HF_HUB_CACHE not set"
[ -n "$HF_DATASETS_CACHE" ] && echo "[INFO] HF_DATASETS_CACHE=$HF_DATASETS_CACHE" || echo "[WARNING] HF_DATASETS_CACHE not set"

# Import local environment variables and set those needed
if [ -f local.env ]; then
        source local.env;
        export HUGGINGFACE_TOKEN;
        export DOCKERHUB_USERNAME;
fi

# Check if necessary variables are set in local.env
if [ -z "$HUGGINGFACE_TOKEN" ] || [ -z "$DOCKERHUB_USERNAME" ]; then
    echo "[ERROR] Please create a local.env with the vars:";
    echo "GIT_USERNAME=MY NAME";
    echo "GIT_EMAIL=myemail@domain.com";
    echo HUGGINGFACE_TOKEN="your_hf_token";
    echo DOCKERHUB_USERNAME="your_dockerhub_username";
    exit 1;
fi

# Print info about necessary variables
[ -n "$HUGGINGFACE_TOKEN" ] && echo "[INFO] Huggingface token set" || echo "[WARNING] Huggingface token not set"
[ -n "$DOCKERHUB_USERNAME" ] && echo "[INFO] Dockerhub Username set" || echo "[WARNING] Dockerhub Username not set"

#################################
# Set environment variables
#################################

# # Set Hugging Face token as environment variable if available (used for download of dataset and upload of embeddings)
# if [ -n "$HUGGINGFACE_TOKEN" ]; then
#     export HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN"
#     echo "[INFO] Hugging Face token set successfully"
# else
#     echo "[WARNING] Hugging Face token was not set."
# fi

# # Set Huggingface cache ENVs
# if [ -n "$HF_HOME" ]; then
#     export HF_HOME="$HF_HOME"
#     echo "[INFO] HF_HOME set successfully"
# else
#     echo "[WARNING] HF_HOME variable was not set."
# fi
# if [ -n "$HF_HUB_CACHE" ]; then
#     export HF_HUB_CACHE="$HF_HUB_CACHE"
#     echo "[INFO] HF_CACHE_HUB set successfully"
# else
#     echo "[WARNING] HF_HUB_CACHE variable was not set."
# fi
# if [ -n "$HF_DATASETS_CACHE" ]; then
#     export HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
#     echo "[INFO] HF_DATASETS_CACHE set successfully"
# else
#     echo "[WARNING] HF_DATASETS_CACHE variable was not set."
# fi


#################################
# Python paths
#################################

echo "export python paths"
# export BASE_PYTHON
# export PERCH_PYTHON
# export TRAIN_PYTHON
COMPLETE_PYTHON="$DOCKER_COMPLETE_PYTHON"
export COMPLETE_PYTHON

#################################
# Build singularity container
#################################

if [ "$sif_container" = true ]; then
  container_extension=".sif"
  container_build_flags=""
else
  container_extension="/"
  container_build_flags="--sandbox"
fi

# Remove existing container if --rebuild-container flag is set
if { [ -d $PROJECT_NAME-image-latest$container_extension ] || [ -f $PROJECT_NAME-image-latest$container_extension ]; } && [ "$rebuild_container" = true ]; then
  echo "Removing the existing container as --rebuild-container flag is set..."
  rm -rf $PROJECT_NAME-image-latest$container_extension
fi

# Build the singularity container from the docker image if it does not exist
if ! { [ -d $PROJECT_NAME-image-latest$container_extension ] || [ -f $PROJECT_NAME-image-latest$container_extension ]; } ; then
  echo "Building the singularity container from docker image..."
  # Pull the latest docker image from Docker Hub and convert it to a singularity image. This will automatically take the a cached image if it exists.
  singularity build $container_build_flags $PROJECT_NAME-image-latest$container_extension docker://$DOCKERHUB_USERNAME/$PROJECT_NAME-image:latest
fi

#################################
# Run dataset preparation
#################################

# Define DEFAULT_DIR in the host environment
export DEFAULT_DIR="$(realpath $PWD)"

singularity exec \
    --bind $HF_HOME:$HF_HOME \
    --bind $DEFAULT_DIR \
    --pwd $DEFAULT_DIR \
    $PROJECT_NAME-image-latest${container_extension:-} \
    $COMPLETE_PYTHON ./prepare_dataset.py "$@"