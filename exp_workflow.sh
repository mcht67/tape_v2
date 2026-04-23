#!/bin/bash

# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.

# Description: This script runs an experiment with DVC within a temporary directory copy and pushes the results to the DVC and Git remote.
set -e
set -x

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
if [ -n "$SINGULARITY_CONTAINER" ] || [ -n "$APPTAINER_CONTAINER" ] || [ -f /.dockerenv ]; then
    if  [ -z "$HUGGINGFACE_TOKEN" ] || [ -z "$DOCKERHUB_USERNAME" ]; then
        echo "[ERROR] Please create a local.env with the vars:";
        echo HUGGINGFACE_TOKEN="your_hf_token";
        echo DOCKERHUB_USERNAME="your_dockerhub_username";
        exit 1;
    fi
fi

# Print info about necessary variables
[ -n "$HUGGINGFACE_TOKEN" ] && echo "[INFO] Huggingface token set" || echo "[WARNING] Huggingface token not set"
[ -n "$DOCKERHUB_USERNAME" ] && echo "[INFO] Dockerhub Username set" || echo "[WARNING] Dockerhub Username not set"

#################################
# Study name
#################################

echo "Study name: "
echo $STUDY_NAME

#################################
# Python paths
#################################

# Define python paths
if [ -n "$SINGULARITY_CONTAINER" ] || [ -n "$APPTAINER_CONTAINER" ] || [ -f /.dockerenv ]; then
    echo "Define python paths to use inside singularity container..."
    # In Docker - use Docker venvs from global.env
    # BASE_PYTHON="$DOCKER_BASE_PYTHON"
    # PERCH_PYTHON="$DOCKER_PERCH_PYTHON"
    # TRAIN_PYTHON="$DOCKER_TRAIN_PYTHON"
    COMPLETE_PYTHON="$DOCKER_COMPLETE_PYTHON"
else
    echo "Define python paths to use locally..."
    # Local - use local venvs from global.env (with DEFAULT_DIR prefix)
    # BASE_PYTHON="$DEFAULT_DIR$LOCAL_BASE_PYTHON"
    # PERCH_PYTHON="$DEFAULT_DIR$LOCAL_PERCH_PYTHON"
    # TRAIN_PYTHON="$DEFAULT_DIR$LOCAL_TRAIN_PYTHON"
    COMPLETE_PYTHON="$DEFAULT_DIR$LOCAL_COMPLETE_PYTHON"
fi

# export BASE_PYTHON
# export PERCH_PYTHON
# export TRAIN_PYTHON
export COMPLETE_PYTHON

echo "Python paths:"
echo $COMPLETE_PYTHON

#################################
# Create temporary directory
#################################

# Define DEFAULT_DIR in the host environment
export DEFAULT_DIR="$PWD"
TMP_DIR=tmp

# Create a new sub-directory in the temporary directory for the experiment
echo "Creating temporary sub-directory..." &&
# Generate a unique ID with the current timestamp, process ID, and hostname for the sub-directory
# UNIQUE_ID=$(date +%s)-$$-$HOSTNAME &&
# EXP_TMP_DIR="$(realpath "$TMP_DIR/$UNIQUE_ID")" &&
# mkdir -p $EXP_TMP_DIR &&

UNIQUE_ID=$(date +%s)-$$-$HOSTNAME &&
EXP_TMP_DIR="$TMP_DIR/$UNIQUE_ID" &&
mkdir -p $EXP_TMP_DIR &&

# # Copy the necessary files to the temporary directory
# echo "Copying files..." &&
# {
# # Add all git-tracked files
# git ls-files;
# if [ -f ".dvc/config.local" ]; then
#     echo ".dvc/config.local";
# fi;

# if [ -f "secrets/dvc-token.json" ]; then
#     echo "secrets/dvc-token.json"
# else
#   echo "WARNING: secrets/dvc-token.json not found" >&2
#   ls -la secrets/ >&2   # show what's actually there
# fi

# echo ".git";
# } | while read file; do
#     # --chown flag is needed for docker to avoid permission issues
#     rsync -aR --chown $(id -u):$(id -g) "$file" $EXP_TMP_DIR;
# done &&

echo "Copying files..." &&
{
  git ls-files
  [ -f ".dvc/config.local" ] && echo ".dvc/config.local"
  if [ -f "secrets/dvc-token.json" ]; then
    echo "secrets/dvc-token.json"
  else
    echo "WARNING: secrets/dvc-token.json not found" >&2
    ls -la secrets/ >&2
  fi
  echo ".git"
} | while IFS= read -r file; do
  rsync -aR "$file" "$EXP_TMP_DIR/"
done &&
# Fix ownership after copying
chown -R "$(id -u):$(id -g)" "$EXP_TMP_DIR"

# Change the working directory to the temporary sub-directory
cd $EXP_TMP_DIR &&

#################################
# DVC cache
#################################

# Set the DVC cache directory to the shared cache located in the host directory
echo "Setting DVC cache directory..." &&
dvc cache dir $DEFAULT_DIR/.dvc/cache &&

# # Pull the data from the DVC remote repository
# if [ -f "dataset.dvc" ]; then
#     echo "Pulling data with DVC..." 
#     dvc pull dataset;
# fi &&

#################################
# Run dvc experiment
#################################

# Run the experiment with passed parameters. Runs with the default parameters if none are passed.
echo "Running experiment..." &&
dvc exp run \
  --set-param python.complete="$COMPLETE_PYTHON" \
  $EXP_PARAMS
#   --set-param python.base="$BASE_PYTHON" \
#   --set-param python.perch="$PERCH_PYTHON" \
#   --set-param python.train="$TRAIN_PYTHON" \

#################################
# Pushing results
#################################

dvc status

# Push the results to the DVC remote repository
echo "Pushing experiment..." &&
dvc exp push origin && \
echo "✅ Push successful!" || echo "❌ Push failed!"

# Moving everythin to archive
DATETIME=$(date +%Y%m%d_%H%M%S)
if [ -n "$STUDY_NAME" ]; then
    ARCHIVE_DIR=${DEFAULT_DIR}/archive/$STUDY_NAME/${DATETIME}_$DVC_EXP_NAME
else
    ARCHIVE_DIR=${DEFAULT_DIR}/archive/unnamed_exp/${DATETIME}_$DVC_EXP_NAME
fi

echo "Archiving results to $ARCHIVE_DIR..."

mkdir -p ${ARCHIVE_DIR}/{logs,checkpoints,metrics}

rsync -rv logs/        ${ARCHIVE_DIR}/logs/
rsync -rv checkpoints/ ${ARCHIVE_DIR}/checkpoints/
#rsync -rv metrics/     ${ARCHIVE_DIR}/metrics/

# Clean up the temporary sub-directory
echo "Cleaning up..." &&
cd .. &&
rm -rf $UNIQUE_ID 