# Implementation Details

## Python paths

## Hugging Face

### Offline usage in multi submission

Because multiple jobs possibly access the huggingface cache at the same time, HF_CACHE_OFFLINE has to be set to "1".
This is done in exp_workflow_job.sh:
```bash
singularity exec \
    --env HF_DATASETS_OFFLINE=1 \
    --bind $HF_HOME:$HF_HOME \
    --bind $DEFAULT_DIR \
    --pwd $DEFAULT_DIR \
    $PROJECT_NAME-image-latest$container_extension \
    ./exp_workflow.sh
```

To ensure the dataset cache is valid, the dataset is loaded once in prepare_dataset.py.

## DVC remote

## Docker | Singularity

## Losses & Metrics

Losses and metrics are derived from the ojectives defined in config.objectives for the current dvc experiment.
For every objective a specific loss is defined (cfg.objectives.loss). The loss config includes the tf.keras loss class as _target_ and all arguments for the initialization of the loss. 

Losses are created inside of train.py by the create_losses_from_objectives function and are each wrapped in a DynamicWeightedLoss to be ready for multi task setups.

Compile metrics and log metrics, displayed in Tensorbaords hParam tab are also created from objectives in build_metrics(). Each objective key is mapped to one or more metrics.  The mapping is hard coded. Compile metrics are passed to the tf.keras.modle. to get computed during training. Log metrics included also losses as metrics and are passed to CustomSummaryWriter.


<!-- Metrics which should be displayed in Tensorboard are defined in the log-config. They follow the scheme of the naming of metrics by tf.keras: val_{metric_name} if there is only one objective and val_{objective}_{metric_name} if there are multiple objectives. -->

## Models

### BirdSet Model Wrappers

### Perch models

### Polyphony Estimation Heads



### Multi Task Heads

### Dataset preparation
To avoid multiple experiments accessing huggingface hub at the same time an pushing new embeddings, the dataset preparation is separated from the experiment pipeline. All embeddings needed for the multi_submission setup are calculated before the experiments jobs are submitted.

### Loggging

#### History logging
- replaying back history to Tensorboard
