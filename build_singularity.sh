#!/bin/bash

# Debugging options
set -euo pipefail
set -x

echo "Running singularity container build script..."

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
# Import environment variables
#################################

# Set environment variables defined in global.env
set -o allexport
source global.env
set +o allexport

# Import local environment variables and set those needed
if [ -f local.env ]; then
        source local.env;
        export DOCKERHUB_USERNAME;
fi

if [ -z "$DOCKERHUB_USERNAME" ]; then
    echo "[ERROR] Please create a local.env with the vars:";
    echo "GIT_USERNAME=MY NAME";
    echo "GIT_EMAIL=myemail@domain.com";
    echo HUGGINGFACE_TOKEN="your_hf_token";
    echo DOCKERHUB_USERNAME="your_dockerhub_username";
    exit 1;
fi
#################################
# Build singularity container
#################################

# Define DEFAULT_DIR in the host environment
export DEFAULT_DIR="$(realpath $PWD)"

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

echo "Finished setup"