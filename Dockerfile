# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.

# # Use an official Debian runtime with fixed version as a parent image
# FROM debian:13-slim

FROM ubuntu:24.04

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

# Base venv (in PATH for DVC and general commands)
RUN python3.12 -m venv /envs/base-venv
ENV PATH="/envs/base-venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir dvc dvc-gdrive omegaconf

# Perch venv (call explicitly in DVC stages)
RUN python3.12 -m venv /envs/perch-venv \
    && /envs/perch-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /envs/perch-venv/bin/pip install --no-cache-dir \
        git+https://github.com/google-research/perch-hoplite.git \
        tensorflow \
        omegaconf \
        datasets==3.6.0

# BIRDSET ENV
# # Create virtual environment
# RUN python3 -m venv /opt/birdset-env
# ENV PATH="/opt/birdset-env/bin:$PATH"

# # Install your requirements first
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# # Verify ruamel.yaml is installed
# RUN python -c "import ruamel.yaml; print('ruamel.yaml installed successfully')"

# # Then install BirdSet
# RUN git clone --no-recurse-submodules https://github.com/DBD-research-group/BirdSet.git /tmp/birdset && \
#     pip install -e /tmp/birdset && \
#     rm -rf /tmp/birdset

# # Final verification
# RUN python -c "import ruamel.yaml; import datasets; print('All packages working')"

# RUN rm requirements.txt

# Copy secrets
COPY secrets/dvc-token.json /home/app/secrets/dvc-token.json
RUN chmod 600 /home/app/secrets/dvc-token.json

COPY .dvc/config.local /home/app/.dvc/config.local

ENV PYTHONPATH="/home/app/source"



