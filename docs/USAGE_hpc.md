# Usage [HPC Cluster]
## Define dvc experiment configuration using hydra

Hydra automatically creates a params.yaml for every dvc experiment used by the different python scripts to extract the current experiments parameters, such as models, dataset, training parameters etc.

The hydra config is defined in [conf/config.yaml] and composed from config-files in subfolders (eg. [conf/dataset/PER.yaml]).

To change the configuration you can add new config-files and add them as defaults in [conf/config.yaml].

## Run single experiment

Run [build_singularity.sh] to build singularity container from docker image.
```bash
./build_singularity.sh
```

Submit dataset preparation job if using pre-computed embeddings.
```bash
sbatch prepare_dataset_job.sh
```

Submit experiment workflow job script to run experiment.
```bash
sbatch exp_workflow_job.sh
```

## Run only dataset preparation

Submit dataset preparation job with arguments
```bash
sbatch prepare_dataset_job.sh -- \
    --huggingface_path mcht67/polyphonic-bird-set-with-embeddings \
    --dataset_config HSN \
    --input_features '["audio"]' \
    --embeddings '["birdnet_V2.3"]'

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
python3 multi_submission.py
```

## Rebuild container

Rebuild the singularity container manually, if the docker image changed.
Optionally use --sif-container flag to use .sif container instead of sandbox.
```bash
./build_singularity.sh --rebuild-container 
```

## GoogleDrive reauthentication
Because authentiation is done inside the browser. You have do the authentication locally. To reset authentication delete dvc-token.json on your local machine. This will open authentication in the browser on the next dvc pull.

```bash
dvc pull
```

### Move your local authentication to the Cluster

Move your local dvc-token to the Cluster
```bash
cat > secrets/dvc-token.json << 'EOF'
# Content of your secrets/dvc-token.json
EOF
```

## Checkpoints & Logs

Checkpoints and logs are stored in ./archive/ on the HPC Cluster. All experiment logs with the same study name are stored in one folder.

### Syncing artifacts from HPC to Local Machine
 To investigate the logs it is useful to sync them to the local machine.
```bash
rsync -rv $HPC:${DEFAULT_DIR}/archive/ ./archive/
```

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

