# Usage [Local]

## Define dvc experiment configuration using hydra

Hydra automatically creates a params.yaml for every dvc experiment used by the different python scripts to extract the current experiments parameters, such as models, dataset, training parameters etc.

The hydra config is defined in conf/config.yaml and composed from config-files in subfolders (eg. conf/dataset/PER.yaml).

To change the configuration you can add new config-files and add them as defaults in conf/config.yaml or overwrite parameters directly when running dvc experiments:

```bash
dvc exp run --set-param train.learning_rate=0.001
```

## Run single experiment

### DVC Pipeline (Local)

Use dvc to run the pipeline directly
bash
```
dvc exp run
```

### Workflow (Local)

Run the workflow script
```bash
./exp_workflow.sh
```

### Workflow in docker (Local)

Run image and mount project dir (Local)
```bash
  docker run --rm \                   
  --mount type=bind,source="$(pwd)",target=/home/app \
  poly-birdset-image \
  /home/app/exp_workflow.sh
```
# Run multi_submission.py

Activate venv and run multi_submission.py
```bash
source venv/bin/activate
python multi_submission.py
```

## Local Run on Docker with GitHub Authentication via SSH

Create ssh-key and add it to GitHub.

Add key to ssh-agent:
```bash
ssh-add ~/.ssh/id_ed25519
```

```bash
docker run --rm \
  -v $SSH_AUTH_SOCK:$SSH_AUTH_SOCK \
  -e SSH_AUTH_SOCK=$SSH_AUTH_SOCK \
  -v $(pwd):/home/app \
  train-bird-models \
  bash -c "mkdir -p ~/.ssh && ssh-keyscan -H github.com >> ~/.ssh/known_hosts 2>/dev/null && ./exp_workflow.sh"
```