#!./venv/bin/python

# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.

import itertools
import os
import sys
import shutil
import json
import subprocess
import shlex
# import huggingface_hub

from dotenv import load_dotenv
from hydra import compose, initialize
# from omegaconf import OmegaConf
# from pathlib import Path

# Submit dataset preparation based on requested configuration
def submit_dataset_prep_job(huggingface_path, dataset_config, input_features, embeddings, recompute_embeddings=False):
    env = {
        **os.environ,
        "DEFAULT_DIR": os.getcwd(),
    }

    args = [
        "--huggingface_path", huggingface_path,
        "--dataset_config", dataset_config,
        "--input_features", json.dumps(input_features),
        "--embeddings", json.dumps(embeddings),
    ]
    if recompute_embeddings: args.append("--recompute_embeddings")
    
    try:
        subprocess.run(
            ['/usr/bin/bash', '-c', f'sbatch prepare_dataset_job.sh {" ".join(shlex.quote(a) for a in args)}'],
            env=env)
    except subprocess.CalledProcessError:
            print(f"Dataset preparation failed for {dataset_config}. Aborting experiment submission.")
            sys.exit(1)

# Submit experiment for hyperparameter combination
def submit_batch_job(arguments, exp_params, experiment_name):

    # Set dynamic parameters for the batch job as environment variables
    # But dont forget to add the os.environ to the new environment variables otherwise the PATH is not found
    env = {
        **os.environ,
        "EXP_PARAMS": exp_params,
        "EXP_NAME": experiment_name,
        "DEFAULT_DIR": os.getcwd(),
        "SYNC_INTERVAL": '0'
    }

    # For debugging and local runs
    if shutil.which('sbatch') is None:
        print("SLURM not available. Would submit job with:", exp_params)

        # Run DVC experiment directly
        cmd = './exp_workflow.sh' #'dvc exp run'
        print("Run command", cmd)
        result = subprocess.run(cmd, shell=True, env=env, stderr=subprocess.PIPE, text=True)
        print("Exit code:", result.returncode)
        print("Stderr:", result.stderr)

        # Copy logs dir to local_logs
        # shutil.copytree('logs', 'local_logs', dirs_exist_ok=True)

        # TODO: setup remote
        # print("Push to remote...")
        # subprocess.run("dvc exp push origin", shell=True)
        return
    
    # Run sbatch command with the environment variables as bash! subprocess! command (otherwise module not found)
    subprocess.run(['/usr/bin/bash', '-c', f'sbatch exp_workflow_job.sh {" ".join(arguments)}'], env=env)

def create_exp_params_str(config_dict):
    exp_params_str = ''
    for key, value in config_dict.items():
        exp_params_str += f"-S  {key}={str(value)} "
    return exp_params_str

# def create_exp_params_str(config_overwrite, config_append=None):
#     exp_params_str = ''
#     for key, value in config_overwrite.items():
#         exp_params_str += f"-S {key}={str(value)} "
#     if config_append:
#         for key, value in config_append.items():
#             exp_params_str += f"+{key}={str(value)} "
#     return exp_params_str

if __name__ == "__main__":

    arguments = sys.argv[1:]

    # ########################
    # # Python version
    # ########################

    # HANDLE IN PREPARE_DATASETS.PY

    # # Ensures base-venv is used, even if running script from another venv
    # # TODO: remove?

    # # Get docker python path
    # base_python = os.getenv('DOCKER_BASE_PYTHON')

    # Get local python path from default config if docker paths not defined
    # cfg = OmegaConf.load("params.yaml")
    # if not base_python:
    #     base_python = cfg.python.base

    # base_venv = Path(base_python).parent.parent
    # os.environ["VIRTUAL_ENV"] = str(base_venv)
    # os.environ["PATH"] = str(base_venv / "bin") + ":" + os.environ["PATH"]

    # complete_python = os.getenv('DOCKER_COMPLETE_PYTHON')
    # if not complete_python:
    #     complete_python = cfg.python.complete

    ##########################
    # Configuration
    ##########################

    # Define Experiment Name
    experiment_name = 'Embeddings-Comparison'
    huggingface_path = 'mcht67/polyphonic-bird-set-with-embeddings'

    # Define Base Config
    base_config = {
        "log.experiment_name": experiment_name,
        "dataset.huggingface_path": huggingface_path,
        
        "train.epochs": 5,
        #"train.initial_epoch": 20,
        #"train.num_batches_train": 20,
        #"train.num_batches_val": 5
        "train.learning_rate": 0.001,
        #"train.load_model_path": "model_path"
    }

    # Define all lists of parameters or config files [Hyperparameters]
    
    dataset_configs = ['HSN_polyphonic']
    input_features = ['audio', 'no_noise_audio']

    models = [
                'TemporalCNN',
                # 'SimpleMLP'
            ]

    embedding_type = 'spatial'
    embeddings = [
                    # 'EfficientNet-B1-BirdSet-XCL',
                    # 'perch_8'
                    'perch_v2_cpu'
                ]
    
    objectives = [
                    'multis_task_v1_add_event_logits',
                    # 'only_polyphony_degree'
                ]

    hyperparams = {
                    "model": models,
                    "dataset.config": dataset_configs,
                    "train.input_feature": input_features,
                    "embeddings": embeddings,   
                    "objectives": objectives                         
                }
    
    recompute_embeddings = False
    # recompute_labels = False

    # ##########################
    # # Huggingface login
    # ##########################

    # load_dotenv('local.env')
    # huggingface_hub.login(token=os.getenv('HUGGINGFACE_TOKEN'))

    ##########################
    # Prepare dataset
    ##########################

    # Collect all input features, embeddings and labels used
    # Pass params (and force_recompute?) to prepare_dataset.py

    # Init config and save for prepare_dataset.py to use 
    # [local python paths, dataset download and upload paths]
    # with initialize(config_path="conf", version_base=None):
    #     cfg = compose(config_name="config")
    # OmegaConf.save(cfg, "params.yaml")

    # Right now just embeddings are computed in prepare dataset
    if embeddings:
        for dataset_config in dataset_configs:
            submit_dataset_prep_job(huggingface_path, dataset_config, input_features, embeddings, recompute_embeddings=recompute_embeddings)

            # try:
            #     # Replace with singularity cmd
            #     cmd = [ complete_python, "prepare_dataset.py",#base_python, "prepare_dataset.py",
            #             "--huggingface_path", huggingface_path,
            #             "--dataset_config", dataset_config,
            #             "--input_features", json.dumps(input_features), 
            #             "--embeddings", json.dumps(embeddings),
            #             # "--objectives", json.dumps(objectives)
            #             ]

            #     if recompute_embeddings: cmd.append("--recompute_embeddings")
            #     #if recompute_labels: cmd.append("--recompute_labels")
            #     subprocess.run(cmd, check=True)
            # except subprocess.CalledProcessError:
            #     print(f"Dataset preparation failed for {dataset_config}. Aborting experiment submission.")
            #     sys.exit(1)

    ##########################
    # Submit experiment jobs
    ##########################
    # Add hyperparameters
    all_hyper_parameter_combinations = (dict(zip(hyperparams.keys(), values)) for values in itertools.product(*hyperparams.values()))
    print(all_hyper_parameter_combinations)
    for hyperparams_config in all_hyper_parameter_combinations:

        # Get hyperparams keys for logging purposes
        hyperparams_keys_str = ",".join(hyperparams_config.keys())
        hyperparams_keys = {"log.hyperparameters": f"[{hyperparams_keys_str}]"}

        # Update input feature name
        if 'train.input_feature' in hyperparams_config and 'embeddings' in hyperparams_config:
            hyperparams_config['train.input_feature_name'] = hyperparams_config['embeddings'] + "_" + hyperparams_config['train.input_feature'] + "_" + embedding_type + "_embeddings"

        # Create config
        config_overwrites = base_config | hyperparams_config | hyperparams_keys
        print(config_overwrites)
        
        # Submit job for every hyperparameter configuration
        exp_params = create_exp_params_str(config_overwrites)
        print("Exp params: ", exp_params)
        submit_batch_job(arguments, exp_params, experiment_name)