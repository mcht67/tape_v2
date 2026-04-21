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

##SBATCH --mem=100GB
##SBATCH --time=00:30:00
##SBATCH --partition=standard

#SBATCH --gres=gpu:a100
#SBATCH --mem=10GB
#SBATCH --time=00:10:00
#SBATCH --partition=gpu

# Get email notifications for job status
#SBATCH --mail-type=ALL
#SBATCH --mail-user=malte.crt@gmail.com

# Debugging options
set -euo pipefail
set -x

echo "Running exp_workflow_job.sh"

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
        export DOCKERHUB_USERNAME;
fi

# Check if necessary variables are set in local.env
if [ -z "$DOCKERHUB_USERNAME" ]; then
    echo "[ERROR] Please create a local.env with the vars:";
    echo HUGGINGFACE_TOKEN="your_hf_token";
    echo DOCKERHUB_USERNAME="your_dockerhub_username";
    exit 1;
fi

# Print info about necessary variables
[ -n "$DOCKERHUB_USERNAME" ] && echo "[INFO] Dockerhub Username set" || echo "[WARNING] Dockerhub Username not set"

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
# Run experiment workflow
#################################

# Define DEFAULT_DIR in the host environment
export DEFAULT_DIR="$(realpath $PWD)"

echo "Starting execution from singularity container..."

# Run the singularity container
# singularity exec --nv --bind $DEFAULT_DIR $PROJECT_NAME-image-latest$container_extension ./exp_workflow.sh # GPU
# HF_DATASETS_OFFLINE=1 avoids locks on the Huggignface cache dir by forbidding fetches and updates

# # CPU
# singularity exec \
#     --env HF_DATASETS_OFFLINE=1 \
#     --bind $HF_HOME:$HF_HOME \
#     --bind $DEFAULT_DIR \
#     --pwd $DEFAULT_DIR \
#     $PROJECT_NAME-image-latest$container_extension \
#     ./exp_workflow.sh

# GPU
singularity exec \
    --nv \
    --env HF_DATASETS_OFFLINE=1 \
    --bind $HF_HOME:$HF_HOME \
    --bind $DEFAULT_DIR \
    --pwd $DEFAULT_DIR \
    $PROJECT_NAME-image-latest$container_extension \
    ./exp_workflow.sh
