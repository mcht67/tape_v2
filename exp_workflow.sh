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
# set -o allexport
source global.env
# set +o allexport

export PROJECT_NAME
[ -n "$PROJECT_NAME" ] && echo "[INFO] PROJECT_NAME=$PROJECT_NAME" || echo "[WARNING] PROJECT_NAME not set"

if [ -n "$SINGULARITY_CONTAINER" ] || [ -n "$APPTAINER_CONTAINER" ] || [ -f /.dockerenv ]; then
    export HF_HOME
    export HF_HUB_CACHE
    export HF_DATASETS_CACHE
    export DOCKER_COMPLETE_PYTHON
    [ -n "$HF_HOME" ] && echo "[INFO] HF_HOME=$HF_HOME" || echo "[WARNING] HF_HOME not set"
    [ -n "$HF_HUB_CACHE" ] && echo "[INFO] HF_HUB_CACHE=$HF_HUB_CACHE" || echo "[WARNING] HF_HUB_CACHE not set"
    [ -n "$HF_DATASETS_CACHE" ] && echo "[INFO] HF_DATASETS_CACHE=$HF_DATASETS_CACHE" || echo "[WARNING] HF_DATASETS_CACHE not set"
    [ -n "$DOCKER_COMPLETE_PYTHON" ] && echo "[INFO] DOCKER_COMPLETE_PYTHON=$DOCKER_COMPLETE_PYTHON" || echo "[WARNING] DOCKER_COMPLETE_PYTHON not set"
else
    export LOCAL_COMPLETE_PYTHON
    [ -n "$LOCAL_COMPLETE_PYTHON" ] && echo "[INFO] LOCAL_COMPLETE_PYTHON=$LOCAL_COMPLETE_PYTHON" || echo "[WARNING] LOCAL_COMPLETE_PYTHON not set"
fi


# Import local environment variables and set those needed
if [ -f local.env ]; then
        source local.env;
        export HUGGINGFACE_TOKEN;
        export DOCKERHUB_USERNAME;
fi
[ -n "$HUGGINGFACE_TOKEN" ] && echo "[INFO] Huggingface token set" || echo "[WARNING] Huggingface token not set"
[ -n "$DOCKERHUB_USERNAME" ] && echo "[INFO] Dockerhub Username set" || echo "[WARNING] Dockerhub Username not set"

# Check if necessary variables are set in local.env
if [ -n "$SINGULARITY_CONTAINER" ] || [ -n "$APPTAINER_CONTAINER" ] || [ -f /.dockerenv ]; then
    if  [ -z "$HUGGINGFACE_TOKEN" ] || [ -z "$DOCKERHUB_USERNAME" ]; then
        echo "[ERROR] Please create a local.env with the vars:";
        echo HUGGINGFACE_TOKEN="your_hf_token";
        echo DOCKERHUB_USERNAME="your_dockerhub_username";
        exit 1;
    fi
fi

#################################
# Study name
#################################

echo "Study name: "
echo $STUDY_NAME

##########################################
# Set or create DVC remote for the study
##########################################

BASE_REMOTE="base-remote"
CONFIG_LOCAL=".dvc/config.local"

# Get study name from hydra/optuna config
STUDY_NAME=$(python -c "import yaml; print(yaml.safe_load(open('conf/config.yaml'))['hydra']['sweeper']['study_name'])")

# Check if remote already exists
if grep -q "\[remote \"$STUDY_NAME\"\]" "$CONFIG_LOCAL" 2>/dev/null; then
    echo "Remote '$STUDY_NAME' already exists, skipping creation."
    # Update default
    sed -i "s/defaultremote = .*/defaultremote = $STUDY_NAME/" "$CONFIG_LOCAL"
    exit 0
fi

# Get base URL from local config
BASE_URL=$(grep -A1 "\[remote \"$BASE_REMOTE\"\]" "$CONFIG_LOCAL" | grep "url" | awk -F'= ' '{print $2}' | tr -d ' ')

# Read all settings from base remote section
ACKNOWLEDGE=$(grep -A10 "\[remote \"$BASE_REMOTE\"\]" "$CONFIG_LOCAL" | grep "gdrive_acknowledge_abuse" | awk -F'= ' '{print $2}' | tr -d ' ')
CLIENT_ID=$(grep -A10 "\[remote \"$BASE_REMOTE\"\]" "$CONFIG_LOCAL" | grep "gdrive_client_id" | awk -F'= ' '{print $2}' | tr -d ' ')
CLIENT_SECRET=$(grep -A10 "\[remote \"$BASE_REMOTE\"\]" "$CONFIG_LOCAL" | grep "gdrive_client_secret" | awk -F'= ' '{print $2}' | tr -d ' ')
CREDENTIALS_FILE=$(grep -A10 "\[remote \"$BASE_REMOTE\"\]" "$CONFIG_LOCAL" | grep "gdrive_user_credentials_file" | awk -F'= ' '{print $2}' | tr -d ' ')

# Append new remote section
cat >> "$CONFIG_LOCAL" << EOF

[remote "$STUDY_NAME"]
    url = $BASE_URL/$STUDY_NAME
    gdrive_acknowledge_abuse = $ACKNOWLEDGE
    gdrive_client_id = $CLIENT_ID
    gdrive_client_secret = $CLIENT_SECRET
    gdrive_user_credentials_file = $CREDENTIALS_FILE
EOF

# Update default remote in [core] section
sed -i "s/defaultremote = .*/defaultremote = $STUDY_NAME/" "$CONFIG_LOCAL"

echo "Created remote '$STUDY_NAME' -> $BASE_URL/$STUDY_NAME"

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
echo "Experiment parameters: $EXP_PARAMS" &&
dvc exp run \
  --set-param python.complete="$COMPLETE_PYTHON" \
  $EXP_PARAMS

#################################
# Archive
#################################

# Moving everythin to archive
# DATETIME=$(date +%Y%m%d_%H%M%S)
# if [ -n "$STUDY_NAME" ]; then
#     ARCHIVE_DIR=${DEFAULT_DIR}/archive/$STUDY_NAME/${DATETIME}_$DVC_EXP_NAME
# else
#     ARCHIVE_DIR=${DEFAULT_DIR}/archive/unnamed_exp/${DATETIME}_$DVC_EXP_NAME
# fi

ARCHIVE_DIR=${DEFAULT_DIR}/archive/

echo "Archiving results to $ARCHIVE_DIR..."

# Check if train_output exists and rsync if it does
if [ -d "train_output" ]; then
  rsync -rv train_output/ ${ARCHIVE_DIR}
else
  echo "train_output directory not found, skipping..."
fi

# Check if dvclive/ exists and rsync if it does
if [ -d "dvclive" ]; then
  rsync -rv dvclive/ ${ARCHIVE_DIR}
else
  echo "dvclive directory not found, skipping..."
fi

# Check if eval_output exists and rsync if it does
if [ -d "eval_output" ]; then
  rsync -rv eval_output/ ${ARCHIVE_DIR}
else
  echo "eval_output directory not found, skipping..."
fi

#################################
# Pushing results
#################################

# Push the results to the DVC remote repository
echo "Pushing experiment..."
if ! dvc exp push origin; then
    echo "❌ Push failed!"
    exit 1
fi
echo "✅ Push successful!"

# Clean up the temporary sub-directory
echo "Cleaning up..." &&
cd .. &&
rm -rf $UNIQUE_ID 