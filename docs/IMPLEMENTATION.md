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

## Docker | Singularityvis