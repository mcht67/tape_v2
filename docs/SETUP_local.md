# Setup [Local]

## 1. Clone repository
```bash
git clone https://github.com/mcht67/tape_v2.git
```

## 2. Update global.env

Update the project name and optionally local venv paths in global.env:
```file
# Name of the project
PROJECT_NAME=train-bird-models

########################################
##     Venvs and python paths  ##
########################################

# Local venv directories
LOCAL_COMPLETE_VENV=/complete-venv

# Local Python binaries 
LOCAL_COMPLETE_PYTHON=${LOCAL_COMPLETE_VENV}/bin/python
```

## 3. Setup virtual environments

Create venv and install dependencies from complete_requirements.txt:
```bash
source global.env
python3.12 -m venv $LOCAL_COMPLETE_VENV
$LOCAL_COMPLETE_VENV/bin/pip install -r complete_requirements.txt
```

or install the packages directly (in case requiremets are broken):
```bash
source global.env
python3.12 -m venv $LOCAL_COMPLETE_VENV
$LOCAL_COMPLETE_VENV/bin/pip install \
        datasets==3.6.0 \
        dvc \
        dvclive \
        dvc_gdrive \
        librosa \
        soundfile \
        omegaconf \
        numpy \
        tensorflow \
        tensorflow_hub \
        tensorboard \
        matplotlib \
        hydra-core \
        torch \
        torchvision \
        torchaudio \
        transformers==4.44.2 \
        seaborn \
        scikit-learn \
        psutil \
        ruamel.yaml \
        python-dotenv \
        git+https://github.com/google-research/perch-hoplite.git
```

## 4. Create local.env
Create a local.env file in the repository. You can use the local.env.template and replace placeholders with your information.
```file
# Copy this to local.env and fill in your values
# DO NOT commit local.env to git!

HUGGINGFACE_TOKEN="your_hf_token_here"
DOCKERHUB_USERNAME="your_dockerhub_username"
```

## 5. Setup dvc remote

First
[setup Google Drive Project for dvc remote](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended).

Then
[connect dvc with Google drive](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive)
like this:
```sh
mkdir secrets
dvc remote add -d myremote gdrive://YOUR_FOLDER_ID
dvc remote modify --local myremote gdrive_acknowledge_abuse true
dvc remote modify --local myremote gdrive_client_id  "actual gdrive client id"
dvc remote modify --local myremote gdrive_client_secret "actual gdrive client secret"
dvc remote modify --local myremote \
gdrive_user_credentials_file ./secrets/dvc-token.json

dvc pull
``` 
To reset authentication delete dvc-token.json. This will open authentication in the browser on the next dvc pull.

