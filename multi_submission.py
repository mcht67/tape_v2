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
# import huggingface_hub

# from source.utils.dataset import get_data_dir

# from dotenv import load_dotenv
# from hydra import compose, initialize
# from omegaconf import OmegaConf
# from pathlib import Path

# Submit dataset preparation based on requested configuration
def submit_dataset_prep_job(study_config, data_dir, dataset_config, recompute_embeddings=False, force_redownload=False):

    huggingface_path = study_config['base_config']['dataset.huggingface_path']
    input_features = study_config['hyperparams']['train.input_feature']
    embeddings = study_config['hyperparams']['embeddings']

    env = {
        **os.environ,
        "DEFAULT_DIR": os.getcwd(),
    }

    args = [
        "--huggingface_path", huggingface_path,
        "--data_dir", data_dir,
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

    # Run sbatch command with the environment variables as bash! subprocess! command (otherwise module not found) 
    # Run only if dataset preparation succeded otherwise abandone
    dependency_flag = f"--dependency=afterok:{dependency_job_id} --kill-on-invalid-dep=yes" if dependency_job_id else ""
    # subprocess.run(
    #     ['/usr/bin/bash', '-c', f'sbatch {dependency_flag}exp_workflow_job.sh {" ".join(arguments)}'],
    #     env=env)

    result = subprocess.run(
                                ['/usr/bin/bash', '-c', f'sbatch {dependency_flag} exp_workflow_job_cpu.sh {" ".join(arguments)}'],
                                env=env, capture_output=True, text=True
                            )
    if not result.returncode==0:
        print("Stderr:", result.stderr)
    else:
        submitted_job_id = result.stdout.strip().split()[-1]
        print("Experiment job submitted: ", submitted_job_id)

def create_exp_params_str(config_dict):
    exp_params_str = ''
    for key, value in config_dict.items():
        exp_params_str += f"-S  {key}={str(value)} "
    return exp_params_str

def submit_experiment_jobs(study_config, subset, train_config, dependency_job_id):      

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
        submit_batch_job(arguments, exp_params, study_name, dependency_job_id=dependency_job_id)

if __name__ == "__main__":

    arguments = sys.argv[1:]

    ##########################
    # Configuration
    ##########################

    # # Define Study name
    # study_name = 'Pooled-Embeddings-Comparison'
    # huggingface_path = 'mcht67/Polyphonic-BirdSet-train'

    # # Define Base Config
    # base_config = {
    #     "log.study_name": study_name,
    #     "dataset.huggingface_path": huggingface_path,
        
    #     "train.epochs": 50,
    #     #"train.initial_epoch": 20,
    #     #"train.num_batches_train": 20,
    #     #"train.num_batches_val": 5
    #     "train.learning_rate": 0.001,
    #     #"train.load_model_path": "model_path"
    # }

    # # Define all lists of parameters or config files [Hyperparameters]
    # dataset_configs = ['HSN_polyphonic_6']
    # input_features = ['audio'] #, 'no_noise_audio']

    # models = [
    #             #'TemporalCNN',
    #             'SimpleMLP'
    #         ]

    # embedding_type = 'pooled' #'spatial'
    # embeddings = [
    #                 "birdnet_V2.3", 
    #                 "vggish", 
    #                 "perch_8",
    #                 "yamnet",
    #                 "beans_baseline",
    #                 "EfficientNet-B1-BirdSet-XCL",
    #                 "Bird-MAE-Huge",
    #                 "AudioProtoPNet-20-BirdSet-XCL",
    #                 "AST-Birdset-XCL",
    #                 "Wav2Vec2-Base-BirdSet-XCL",
    #                 "perch_v2_cpu",
    #                 # #"perch_v2",    
    #             ]
    
    # objectives = [
    #                 #'multi_task_v1_add_event_logits',
    #                 'only_polyphony_degree'
    #             ]

    # hyperparams = {
    #                 "model": models,
    #                 #"dataset.config": dataset_configs,
    #                 "train.input_feature": input_features,
    #                 "embeddings": embeddings,   
    #                 "objectives": objectives                         
    #             }
    
    # Load study configuration
    study_config_path = 'study_conf/pooled_embeddings_comparison.yaml'

    with open(study_config_path) as f:
        study_config = yaml.safe_load(f)

    dataset_subsets = study_config['dataset_subsets']
    embeddings = study_config['hyperparams']['embeddings']
    
    # Dataset preparation options
    run_dataset_preparation = True
    recompute_embeddings = False
    force_redownload = False
    
    for subset in dataset_subsets:
        train_config = subset + '_' + study_config['base_config']['dataset.train_suffix']
        data_dir = f'{subset}/{train_config}' # as in get_data_dir function in source/utils/dataset.py
        prep_job_id = None
        if embeddings:
            if run_dataset_preparation:
                # prep_job_id = submit_dataset_prep_job(study_config['base_config']['dataset.huggingface_path'], dataset_config, study_config['hyperparams']['train.input_feature'], embeddings, recompute_embeddings=recompute_embeddings, force_redownload=force_redownload)
                prep_job_id = submit_dataset_prep_job(study_config, data_dir, train_config, recompute_embeddings=recompute_embeddings, force_redownload=force_redownload)
        # submit_experiment_jobs(study_config['base_config'], study_config['hyperparams'], dataset_config, dependency_job_id=prep_job_id)
        submit_experiment_jobs(study_config, data_dir, train_config, dependency_job_id=prep_job_id)
  