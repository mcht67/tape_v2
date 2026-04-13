# Debugging options
set -euo pipefail
set -x

echo "Running setup..."

whoami

# Set environment variables defined in global.env
set -o allexport
source global.env
set +o allexport

# Import local environment variables
if [ -f local.env ]; then
        source local.env;
fi

# Check if necessary variables are set in local.env
if [ -z "$GIT_USERNAME" ] || [ -z "$GIT_EMAIL" ] || [ -z "$HUGGINGFACE_TOKEN" ] || [ -z "$DOCKERHUB_USERNAME" ]; then
    echo "[ERROR] Please create a local.env with the vars:";
    echo "GIT_USERNAME=MY NAME";
    echo "GIT_EMAIL=myemail@domain.com";
    echo HUGGINGFACE_TOKEN="your_hf_token";
    echo DOCKERHUB_USERNAME="your_dockerhub_username";
    exit 1;
fi

# Set git user config
if [ "$GIT_USERNAME" ] | [ "$GIT_EMAIL" ]
    git config --global user.name "$GIT_USERNAME"
    git config --global user.email "$GIT_EMAIL"
    git config --global safe.directory "$PWD"
    echo "[INFO] Git User Config set succesfully"
fi

# Set dockerhub username as environment variable if available (used in slurm_jobs.sh)
    if [ -n "$DOCKERHUB_USERNAME" ]; then
        export DOCKERHUB_USERNAME="$DOCKERHUB_USERNAME"
        echo "[INFO] Dockerhub Username set successfully"
    fi 

# Set Hugging Face token as environment variable if available (used for download of dataset and upload of embeddings)
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    export HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN"
    echo "[INFO] Hugging Face token set successfully"
else
    echo "[WARNING] Hugging Face token was not set."
fi

# Set Huggingface cache ENVs
if [ -n "$HF_HOME" ]; then
    export HF_HOME="$HF_HOME"
    echo "[INFO] HF_HOME set successfully"
else
    echo "[WARNING] HF_HOME variable was not set."
fi
if [ -n "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HF_HUB_CACHE"
    echo "[INFO] HF_CACHE_HUB set successfully"
else
    echo "[WARNING] HF_HUB_CACHE variable was not set."
fi
if [ -n "$HF_DATASETS_CACHE" ]; then
    export HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
    echo "[INFO] HF_DATASETS_CACHE set successfully"
else
    echo "[WARNING] HF_DATASETS_CACHE variable was not set."
fi

echo "Finished setup"