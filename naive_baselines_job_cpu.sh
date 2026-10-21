#!/bin/bash

# Copyright 2026 cohrt
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.

# Runs source/scripts/evaluate_naive_baselines.py (audio-free constant /
# random / train- and test-prior baselines) on the synthetic test split and
# the soundscape test_5s split of every dataset under data/, inside the
# singularity container. Results go to results/naive_baselines/.
#
# Usage:
#   sbatch naive_baselines_job_cpu.sh                                  # all datasets, max polyphony 8
#   sbatch naive_baselines_job_cpu.sh -- --subsets HSN POW
#   sbatch naive_baselines_job_cpu.sh -- --max-polyphony 12 --out-dir results/naive_baselines_poly12

# Job name and logs
#SBATCH -J tape-naive-baselines
#SBATCH -D /beegfs/scratch/cohrt/tape_v2/ # Working Directory
#SBATCH --output=./logs/slurm/slurm-%j.out

# Resources needed (only label columns are read, no audio is decoded)
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --partition=standard

# Get email notifications for job status
#SBATCH --mail-type=ALL
#SBATCH --mail-user=malte.crt@gmail.com

# Debugging options
set -euo pipefail
# set -x

echo "Running naive baselines evaluation slurm job"

whoami

#################################
# Handle options
#################################

# Default variable values
rebuild_container=false
sif_container=false

usage() {
  echo "Usage: $0 [OPTIONS] -- [PYTHON ARGS...]"
  echo "Options:"
  echo "  -h, --help                Display this help message"
  echo "  -b, --rebuild-container   Force rebuild of the singularity container"
  echo "  -s, --sif-container       Build the singularity container as SIF"
}

handle_options() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      -b|--rebuild-container)
        rebuild_container=true
        shift
        ;;
      -s|--sif-container)
        sif_container=true
        shift
        ;;
      --)
        shift
        break
        ;;
      *)
        break
        ;;
    esac
  done

  PY_ARGS=("$@")
}

handle_options "$@"

#################################
# Load modules
#################################

module load singularity/4.4.0

#################################
# Import environment variables
#################################

# Set environment variables defined in global.env
set -o allexport
source global.env
set +o allexport

[ -n "$HF_HOME" ] && echo "[INFO] HF_HOME=$HF_HOME" || echo "[WARNING] HF_HOME not set"

# Import local environment variables and set those needed
if [ -f local.env ]; then
        source local.env;
        export DOCKERHUB_USERNAME;
fi

if [ -z "$DOCKERHUB_USERNAME" ]; then
    echo "[ERROR] Please create a local.env with the vars:";
    echo DOCKERHUB_USERNAME="your_dockerhub_username";
    exit 1;
fi

COMPLETE_PYTHON="$DOCKER_COMPLETE_PYTHON"
export COMPLETE_PYTHON
echo "COMPLETE_PYTHON: $COMPLETE_PYTHON"

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
  singularity build $container_build_flags $PROJECT_NAME-image-latest$container_extension docker://$DOCKERHUB_USERNAME/$PROJECT_NAME-image:latest
fi

#################################
# Run naive baselines evaluation
#################################

# Define DEFAULT_DIR in the host environment
export DEFAULT_DIR="$(realpath $PWD)"

# Checks
: "${PROJECT_NAME:?ERROR: PROJECT_NAME is not set}"
: "${COMPLETE_PYTHON:?ERROR: COMPLETE_PYTHON is not set}"
: "${DEFAULT_DIR:?ERROR: DEFAULT_DIR is not set}"

CONTAINER="${PROJECT_NAME}-image-latest${container_extension:-}"

if [ ! -e "$CONTAINER" ]; then
    echo "ERROR: Container not found: $CONTAINER"
    exit 1
fi

[ -d data ] || { echo "ERROR: data/ not found in $DEFAULT_DIR"; exit 1; }
echo "Datasets found:"
ls -d data/*/*_polyphonic data/*/*_soundscape_test 2>/dev/null || true

singularity exec \
    --env HF_DATASETS_OFFLINE=1 \
    --env MPLCONFIGDIR=/tmp/mpl-$SLURM_JOB_ID \
    --bind $HF_HOME:$HF_HOME \
    --bind $DEFAULT_DIR \
    --pwd $DEFAULT_DIR \
    $CONTAINER \
    $COMPLETE_PYTHON source/scripts/evaluate_naive_baselines.py ${PY_ARGS[@]+"${PY_ARGS[@]}"}
