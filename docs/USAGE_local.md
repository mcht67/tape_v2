# Usage [Local]

## Define dvc experiment configuration using hydra

Hydra automatically creates a params.yaml for every dvc experiment used by the different python scripts to extract the current experiments parameters, such as models, dataset, training parameters etc.

The hydra config is defined in [conf/config.yaml] and composed from config-files in subfolders (eg. [conf/dataset/PER.yaml]).

To change the configuration you can add new config-files and add them as defaults in [conf/config.yaml].

```bash
dvc exp run --set-param train.learning_rate=0.001
```

## Run single experiment

### DVC Pipeline

Use dvc to run the pipeline directly.
```bash
dvc exp run
```

### Workflow

Run the workflow script.
```bash
./exp_workflow.sh
```

### Workflow in docker
Run image and mount project dir.
```bash
source global.env
docker run --rm \                   
  --mount type=bind,source="$(pwd)",target=/home/app \
  $PROJECT_NAME-image \
  /home/app/exp_workflow.sh
```

### Local Run on Docker with GitHub Authentication via SSH

Create ssh-key and add it to GitHub.

Add key to ssh-agent:
```bash
ssh-add ~/.ssh/id_ed25519
```

Run workflow within docker container.
```bash
source global.env
docker run --rm \
  -v $SSH_AUTH_SOCK:$SSH_AUTH_SOCK \
  -e SSH_AUTH_SOCK=$SSH_AUTH_SOCK \
  -v $(pwd):/home/app \
  $PROJECT_NAME-image \
  bash -c "mkdir -p ~/.ssh && ssh-keyscan -H github.com >> ~/.ssh/known_hosts 2>/dev/null && ./exp_workflow.sh"
```

## Run multiple experiments on potentially multiple datasets and hyperparameter configurations

Define parameters, datasets and hyperparameters to overwrite default hydra configuration in [multi_submission.py].
All dvc experiments artifacts are stored in a folder in ./archive/ corresponding to the study name.
```python
# Study Name
study_name = 'Embeddings-Comparison'

# Huggingface dataset repository path
huggingface_path = 'mcht67/polyphonic-bird-set-with-embeddings'

# Base Config
base_config = {
    "log.study_name": study_name,
    "dataset.huggingface_path": huggingface_path,
    "train.epochs": 5,
    "train.learning_rate": 0.001,
}

# Define all lists of parameters or hydra config files [Hyperparameters]
dataset_configs = ['HSN_polyphonic']
input_features = ['audio', 'no_noise_audio']

models = [
            'SimpleMLP'
        ]

embedding_type = 'pooled' #'spatial'
[...]
```

Run multi_submission.py
```bash
source global.env
$LOCAL_COMPLETE_PYTHON multi_submission.py
```

## Checkpoints & Logs

Checkpoints and logs are stored in ./archive/. All experiment logs with the same study name are stored in one folder.

### Folder structure
archive/<study-name>/<timecode_dvc-exp-name>/
├─ checkpoints/
│   ├── epoch_weights/  
│   │   ├── epoch_05.weights.h5
│   │   └── epoch_10.weights.h5
│   ├── best/           
│   │   └── best.weights.h5
│   └── resumable_checkpoints/    
│       ├── epoch_05.keras
│       ├── epoch_10.keras
│       └── history.json
├─ logs/
│   ├── dvclive/
│   ├── train/
│   ├── validation/
│   └── params.yaml
└── metrics/

### Tensorboard

You can visualize logs with tensorboard:
```bash
tensorboard --logdir=./archive/logs/tensorboard/<study name>
```