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

# from dotenv import load_dotenv
# from hydra import compose, initialize
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
        print(f"Dataset prep job submitted: {job_id}")
        return job_id
    except subprocess.CalledProcessError:
            print(f"Dataset preparation failed for {dataset_config}. Aborting experiment submission.")
            sys.exit(1)

# Submit experiment for hyperparameter combination
def submit_batch_job(arguments, exp_params, study_name, dependency_job_id=None):

    # Set dynamic parameters for the batch job as environment variables
    # But dont forget to add the os.environ to the new environment variables otherwise the PATH is not found
    env = {
        **os.environ,
        "EXP_PARAMS": exp_params,
        "STUDY_NAME": study_name,
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
        
        return
    
    #print("Submit experiment slurm job")

    # Run sbatch command with the environment variables as bash! subprocess! command (otherwise module not found) 
    # Run only if dataset preparation succeded otherwise abandone
    dependency_flag = f"--dependency=afterok:{dependency_job_id} " if dependency_job_id else ""
    # subprocess.run(
    #     ['/usr/bin/bash', '-c', f'sbatch {dependency_flag}exp_workflow_job.sh {" ".join(arguments)}'],
    #     env=env)

    result = subprocess.run(
                                ['/usr/bin/bash', '-c', f'sbatch {dependency_flag} --kill-on-invalid-dep=yes exp_workflow_job_cpu.sh {" ".join(arguments)}'],
                                env=env, capture_output=True, text=True
                            )
    if not result.returncode==0:
        print("Stderr:", result.stderr)
    else:
        submitted_job_id = result.stdout.strip().split()[-1]
        print("Experiment job submitted: ", submitted_job_id)

    # #print("Submit clean up job")

    # # Submit a cleanup job that cancels the pending job if the dependency fails
    # dependency_fail_flag = f"--dependency=afternotok:{dependency_job_id} " if dependency_job_id else ""
    # result = subprocess.run([
    #                     '/usr/bin/bash', '-c',
    #                     f'sbatch {dependency_fail_flag} --wrap="scancel {submitted_job_id}"'
    #                 ], env=env, capture_output=True, text=True)
    # submitted_clean_job_id = result.stdout.strip().split()[-1]
    # print("Clean up job submitted: ", submitted_clean_job_id)
    
   # subprocess.run(['/usr/bin/bash', '-c', f'sbatch exp_workflow_job.sh {" ".join(arguments)}'], env=env)

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

def submit_experiment_jobs(base_config, hyperparams, dataset_config, dependency_job_id):                                                                                                           
    all_hyper_parameter_combinations = (dict(zip(hyperparams.keys(), values)) for values in itertools.product(*hyperparams.values()))
    for hyperparams_config in all_hyper_parameter_combinations:

        # Add dataset_config to hyperparams
        hyperparams_config['dataset.config'] = dataset_config

        # Get hyperparams keys for logging purposes
        hyperparams_keys_str = ",".join(hyperparams_config.keys())
        #hyperparams_keys_str = ",".join(hyperparams.keys()) #",".join(hyperparams_config.keys())
        hyperparams_keys = {"log.hyperparameters": f"[{hyperparams_keys_str}]"}

        # Update input feature name
        if 'train.input_feature' in hyperparams_config and 'embeddings' in hyperparams_config:
            hyperparams_config['train.input_feature_name'] = hyperparams_config['embeddings'] + "_" + hyperparams_config['train.input_feature'] + "_" + embedding_type + "_embeddings"

        # Create config
        config_overwrites = base_config | hyperparams_config | hyperparams_keys
        # print("Config overwrites")
        # print(config_overwrites)
        
        # Submit job for every hyperparameter configuration
        exp_params = create_exp_params_str(config_overwrites)
        # print("Exp params: ", exp_params)
        submit_batch_job(arguments, exp_params, study_name, dependency_job_id=dependency_job_id)

if __name__ == "__main__":

    arguments = sys.argv[1:]

    ##########################
    # Configuration
    ##########################

    # Define Study name
    study_name = 'Embeddings-Comparison'
    huggingface_path = 'mcht67/Polyphonic-Bird-Set'

    # Define Base Config
    base_config = {
        "log.study_name": study_name,
        "dataset.huggingface_path": huggingface_path,
        
        "train.epochs": 5,
        #"train.initial_epoch": 20,
        #"train.num_batches_train": 20,
        #"train.num_batches_val": 5
        "train.learning_rate": 0.001,
        #"train.load_model_path": "model_path"
    }

    # Define all lists of parameters or config files [Hyperparameters]
    dataset_configs = ['HSN_polyphonic_6']
    input_features = ['audio'] #, 'no_noise_audio']

    models = [
                #'TemporalCNN',
                'SimpleMLP'
            ]

    embedding_type = 'pooled' #'spatial'
    embeddings = [
                    "birdnet_V2.3", 
                    "vggish", 
                    "perch_8",
                    "yamnet",
                    "beans_baseline",
                    "EfficientNet-B1-BirdSet-XCL",
                    "Bird-MAE-Huge",
                    "AudioProtoPNet-20-BirdSet-XCL",
                    "AST-Birdset-XCL",
                    "Wav2Vec2-Base-BirdSet-XCL",
                    #"perch_v2",
                    "perch_v2_cpu"
                ]
    
    objectives = [
                    #'multi_task_v1_add_event_logits',
                    'only_polyphony_degree'
                ]

    hyperparams = {
                    "model": models,
                    #"dataset.config": dataset_configs,
                    "train.input_feature": input_features,
                    "embeddings": embeddings,   
                    "objectives": objectives                         
                }
    
    recompute_embeddings = True
    
    for dataset_config in dataset_configs:
        prep_job_id = None
        if embeddings:
            prep_job_id = submit_dataset_prep_job(huggingface_path, dataset_config, input_features, embeddings, recompute_embeddings=recompute_embeddings)
        
        submit_experiment_jobs(base_config, hyperparams, dataset_config, dependency_job_id=prep_job_id)

  