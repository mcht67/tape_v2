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
import yaml
import configparser
from pathlib import Path

# Submit dataset preparation based on requested configuration
def submit_dataset_prep_job(study_config, subset, dataset_config, recompute_embeddings=False, force_redownload=False):

    huggingface_path = study_config['base_config']['dataset.huggingface_path']
    input_features = study_config['hyperparams']['train.input_feature']
    embeddings = study_config['hyperparams']['embeddings']

    env = {
        **os.environ,
        "DEFAULT_DIR": os.getcwd(),
    }

    args = [
        "--huggingface_path", huggingface_path,
        "--subset", subset,
        "--dataset_config", dataset_config,
        "--input_features", json.dumps(input_features),
        "--embeddings", json.dumps(embeddings),
    ]
    if recompute_embeddings: args.append("--recompute_embeddings")
    if force_redownload: args.append("--force_redownload")

    # For debugging and local runs
    if shutil.which('sbatch') is None:
        print("SLURM not available. Would submit dataset prep job with:", args)

        # Run prepare_dataset.py directly
        cmd = ["python3", "prepare_dataset.py"] + args
        print("Run command", cmd)
        result = subprocess.run(cmd, env=env, stderr=subprocess.PIPE, text=True)
        print("Exit code:", result.returncode)
        print("Stderr:", result.stderr)
        
        return
    
    try:
        result = subprocess.run(
        ['/usr/bin/bash', '-c', f'sbatch prepare_dataset_job_cpu.sh {" ".join(shlex.quote(a) for a in args)}'],
        env=env, capture_output=True, text=True
        )
        # Output is "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]
        print(f"Dataset prep job for {dataset_config} submitted: {job_id}")
        return job_id
    except subprocess.CalledProcessError:
            print(f"Dataset preparation failed for {dataset_config}. Aborting experiment submission.")
            sys.exit(1)

# Submit experiment for hyperparameter combination
def submit_batch_job(arguments, exp_params, study_name, dependency_job_id=None, run_on_gpu=False, force_exp_rerun=False):

    # Set dynamic parameters for the batch job as environment variables
    # But dont forget to add the os.environ to the new environment variables otherwise the PATH is not found
    env = {
        **os.environ,
        "EXP_PARAMS": exp_params,
        "STUDY_NAME": study_name,
        "DEFAULT_DIR": os.getcwd(),
        "SYNC_INTERVAL": '0',
        "FORCE_EXP_RERUN": '1' if force_exp_rerun else '0'
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
        
        return

    # Run sbatch command with the environment variables as bash! subprocess! command (otherwise module not found) 
    # Run only if dataset preparation succeded otherwise abandone
    dependency_flag = f"--dependency=afterok:{dependency_job_id} --kill-on-invalid-dep=yes" if dependency_job_id else ""
    # subprocess.run(
    #     ['/usr/bin/bash', '-c', f'sbatch {dependency_flag}exp_workflow_job.sh {" ".join(arguments)}'],
    #     env=env)
    if run_on_gpu:
        result = subprocess.run(
                                ['/usr/bin/bash', '-c', f'sbatch {dependency_flag} exp_workflow_job_gpu.sh {" ".join(arguments)}'],
                                env=env, capture_output=True, text=True
                            )
    else:
        result = subprocess.run(
                                    ['/usr/bin/bash', '-c', f'sbatch {dependency_flag} exp_workflow_job_cpu.sh {" ".join(arguments)}'],
                                    env=env, capture_output=True, text=True
                                )
    if not result.returncode==0:
        print("Stderr:", result.stderr)
    else:
        submitted_job_id = result.stdout.strip().split()[-1]
        print("Experiment job submitted: ", submitted_job_id)

# def create_exp_params_str(config_dict):
#     exp_params_str = ''
#     for key, value in config_dict.items():
#         exp_params_str += f"-S  {key}={str(value)} "
#     return exp_params_str

# def create_exp_params_str(config_dict):
#     exp_params_str = ''
#     for key, value in config_dict.items():
#         if isinstance(value, list):
#             formatted = "[" + ",".join(str(v) for v in value) + "]"
#         else:
#             formatted = str(value)
#         exp_params_str += f"-S  {key}={formatted} "
#     return exp_params_str

def format_value(value):
    if isinstance(value, dict):
        items = ",".join(f"{k}:{format_value(v)}" for k, v in value.items())
        return "{" + items + "}"
    elif isinstance(value, list):
        items = ",".join(format_value(v) for v in value)
        return "[" + items + "]"
    else:
        return str(value)

def create_exp_params_str(config_dict):
    parts = []
    for key, value in config_dict.items():
        formatted = format_value(value)
        parts.append(f"-S {key}={formatted}")
    return " ".join(parts)

def submit_experiment_jobs(study_config, subset, train_config, dependency_job_id, run_on_gpu=False, force_exp_rerun=False):      

    base_config = study_config['base_config']
    hyperparams = study_config['hyperparams']
    study_name = base_config['log.study_name']
    # embedding_type = study_config['embedding_type']

    all_hyper_parameter_combinations = (dict(zip(hyperparams.keys(), values)) for values in itertools.product(*hyperparams.values()))
    for hyperparams_config in all_hyper_parameter_combinations:

        # Add train_config and subset to hyperparams
        hyperparams_config['dataset.subset'] = subset
        hyperparams_config['dataset.train_config'] = train_config

        # Get hyperparams keys for logging purposes
        hyperparams_keys_str = ",".join(hyperparams_config.keys())
        #hyperparams_keys_str = ",".join(hyperparams.keys()) #",".join(hyperparams_config.keys())
        hyperparams_keys = {"log.hyperparameters": f"[{hyperparams_keys_str}]"}

        # Update input feature name
        # TODO: make more robust for missing keys and different combinations of input features and embeddings
        if 'train.input_feature' in hyperparams_config and 'embeddings' in hyperparams_config:
            hyperparams_config['train.input_feature_name'] = hyperparams_config['embeddings'] + "_" + hyperparams_config['train.input_feature'] + "_" + study_config['base_config']['embeddings.dimension_type'] + "_embeddings"

        # Create config
        config_overwrites = base_config | hyperparams_config | hyperparams_keys
        # print("Config overwrites")
        # print(config_overwrites)
        
        # Submit job for every hyperparameter configuration
        exp_params = create_exp_params_str(config_overwrites)
        # print("Exp params: ", exp_params)
        print("Submitting experiment for dataset ", train_config, " with hyperparameters: ", hyperparams_config)
        submit_batch_job(arguments, exp_params, study_name, dependency_job_id=dependency_job_id, run_on_gpu=run_on_gpu, force_exp_rerun=force_exp_rerun)

def create_study_remote(study_name: str, base_remote: str = "base-remote"):
    local_config_path = Path(".dvc/config.local")
    global_config_path = Path(".dvc/config")
    
    local_config = configparser.RawConfigParser()
    local_config.read(local_config_path)

    global_config = configparser.RawConfigParser()
    global_config.read(global_config_path)
    
    base_section = f'remote "{base_remote}"'
    new_section = f'remote "{study_name}"'
    
    # Check if already exists
    if local_config.has_section(new_section):
        print(f"Remote '{study_name}' already exists, skipping creation.")
        global_config.set('core', 'remote', study_name)
        with open(global_config_path, 'w') as f:
            global_config.write(f)
        return

    # Get base URL and create new URL with subfolder
    base_url = local_config.get(base_section, 'url')
    
    # Create new section
    local_config.add_section(new_section)
    local_config.set(new_section, 'url', f"{base_url}/{study_name}")
    
    # Copy non-empty values from base remote
    for key in ['gdrive_acknowledge_abuse', 'gdrive_client_id', 
                'gdrive_client_secret', 'gdrive_user_credentials_file']:
        try:
            value = local_config.get(base_section, key)
            if value:
                local_config.set(new_section, key, value)
        except configparser.NoOptionError:
            pass
    
    # Set as default
    global_config.set('core', 'remote', study_name)
    
    with open(local_config_path, 'w') as f:
        local_config.write(f)
    
    with open(global_config_path, 'w') as f:
        global_config.write(f)

    print(f"Created remote '{study_name}' -> {base_url}/{study_name}")

if __name__ == "__main__":

    arguments = sys.argv[1:]

    ##########################
    # Configuration
    ##########################
    
    # Load study configuration
    study_config_path = 'study_conf/pooled_embeddings.yaml'

    with open(study_config_path) as f:
        study_config = yaml.safe_load(f)

    dataset_subsets = study_config['dataset_subsets']
    embeddings = study_config['hyperparams']['embeddings'] if 'embeddings' in study_config['hyperparams'] else None
    study_name = study_config['base_config']['log.study_name']

    # Create DVC remote for the study
    create_study_remote(study_name, base_remote="base-remote")
    
    # Dataset preparation options
    run_dataset_preparation = False
    recompute_embeddings = False
    force_redownload = False
    run_on_gpu = False
    force_exp_rerun = True

    ##########################
    # Submit jobs
    ##########################
    
    for subset in dataset_subsets:
        train_config = subset + '_polyphonic' #'_' + str(study_config['base_config']['dataset.max_polyphony'])
        # data_dir = f'{subset}/{train_config}' # as in get_data_dir function in source/utils/dataset.py
        prep_job_id = None
        if embeddings:
            if run_dataset_preparation:
                prep_job_id = submit_dataset_prep_job(study_config, subset, train_config, recompute_embeddings=recompute_embeddings, force_redownload=force_redownload)
        submit_experiment_jobs(study_config, subset, train_config, dependency_job_id=prep_job_id, run_on_gpu=run_on_gpu, force_exp_rerun=force_exp_rerun)