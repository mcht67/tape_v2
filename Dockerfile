# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.

#FROM ubuntu:24.04
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
# RUN apt-get update && apt-get install -y --no-install-recommends \
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.12 and tools (from Ubuntu itself)
    python3.12 \
    python3.12-venv \
    python3-pip \
    # Build tools 
    build-essential \
    # SSL and compression libraries
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    # Python build dependencies
    libncurses5-dev \
    libgdbm-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    # Audio processing
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    # Utilities
    wget \
    curl \
    git \
    openssh-client \
    rsync \
 && rm -rf /var/lib/apt/lists/*


# Copy the global.env file
COPY global.env /tmp/global.env

# Set the working directory
WORKDIR /home/app

################
# Envs
################

# COPY requirements.txt .
# COPY perch_requirements.txt .
# COPY train_requirements.txt .
COPY complete_requirements.txt .

# Create venvs in /opt (fast, inside container)
# ENV BASE_VENV=/opt/venv
# ENV PERCH_VENV=/opt/perch-venv
# ENV TRAIN_VENV=/opt/train-venv
ARG DOCKER_COMPLETE_VENV
ENV DOCKER_COMPLETE_VENV=${DOCKER_COMPLETE_VENV}

# # Create base venv
# RUN python3.12 -m venv $BASE_VENV && \
#     $BASE_VENV/bin/pip install --no-cache-dir --upgrade pip && \
#     $BASE_VENV/bin/pip install --force-reinstall setuptools==75.0.0 && \
#     $BASE_VENV/bin/pip install --no-cache-dir -r requirements.txt

# # The very latest setuptools (82.x) has been removing pkg_resources from its default exports in some configurations. 
# # Downgrading to a slightly older but still recent version should restore it.

# ## Alternatively (if requirements are broken) install these packages
# # && $BASE_VENV/bin/pip install dvc dvc-gdrive omegaconf datasets==3.6.0 soundfile dotenv

# # Perch venv for embeddings
# RUN python3.12 -m venv $PERCH_VENV &&\
#     $PERCH_VENV/bin/pip install --no-cache-dir --upgrade pip && \
#     $PERCH_VENV/bin/pip install --no-cache-dir -r perch_requirements.txt

# ## Alternatively (if requirements are broken) install these packages
# # && $PERCH_VENV/bin/pip install --no-cache-dir \
# #     git+https://github.com/google-research/perch-hoplite.git \
# #     tensorflow \
# #     tensorflow_hub \
# #     omegaconf \
# #     datasets==3.6.0

# # Note: gcsfs 2026.1.0 conflicts with datasets 3.6.0's fsspec requirement
# # but both work in practice. Keep datasets at 3.6.0 due to cast_column(Audio()) issue.

# # Train venv
# RUN python3.12 -m venv $TRAIN_VENV &&\
#     $TRAIN_VENV/bin/pip install --no-cache-dir --upgrade pip && \
#     #$TRAIN_VENV/bin/pip install --no-cache-dir -r train_requirements.txt

#     ## Alternatively (if requirements are broken) install these packages
#     $TRAIN_VENV/bin/pip install --no-cache-dir \
#         datasets==3.6.0 \
#         librosa \
#         soundfile \
#         omegaconf \
#         numpy \
#         tensorflow \
#         matplotlib \
#         hydra-core \
#         torch \
#         seaborn \
#         scikit-learn \
#         psutil \
#         ruamel.yaml \
#         python-dotenv

# Create complete venv
RUN python3.12 -m venv $DOCKER_COMPLETE_VENV &&\
    #$DOCKER_COMPLETE_VENV/bin/pip install -r complete_requirements.txt
    ## Alternatively (if requirements are broken) install these packages
    $DOCKER_COMPLETE_VENV/bin/pip install \
        datasets==3.6.0 \
        dvc==3.67.1 \
        dvclive==3.49.0 \
        dvc_gdrive==3.0.1 \
        librosa==0.11.0 \
        soundfile==0.13.1 \
        omegaconf==2.3.0 \
        numpy==2.4.4 \
        tensorflow[and-cuda]==2.21.0 \
        tensorflow_hub==0.16.1 \
        tensorboard==2.20.0 \
        matplotlib==3.10.8 \
        hydra-core==1.3.2 \
        torch==2.11.0 \
        torchvision==0.26.0 \
        torchaudio==2.11.0 \
        transformers==4.44.2 \
        seaborn==0.13.2 \
        scikit-learn==1.8.0 \
        psutil==7.2.2 \
        ruamel.yaml==0.19.1 \
        python-dotenv==1.2.2 \
        git+https://github.com/google-research/perch-hoplite.git

# TODO: fix properly; for now: "#fsspec==2026.2.0" and "gcsfs==2025.3.0" to avoid conflict.

# Add source to python path
ENV PYTHONPATH="/home/app/source"

# # Add base venv to PATH by default
# ENV PATH="$BASE_VENV/bin:$PATH"
ENV PATH="$DOCKER_COMPLETE_VENV/bin:$PATH"

# # Set Python paths for scripts to use
# ENV DOCKER_BASE_PYTHON=$BASE_VENV/bin/python
# ENV DOCKER_PERCH_PYTHON=$PERCH_VENV/bin/python
# ENV DOCKER_TRAIN_PYTHON=$TRAIN_VENV/bin/python

ENV DOCKER_COMPLETE_PYTHON=$DOCKER_COMPLETE_VENV/bin/python