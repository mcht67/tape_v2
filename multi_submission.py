#!./venv/bin/python

# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.

import itertools
import subprocess
import os
import sys
import shutil

# Submit experiment for hyperparameter combination
def submit_batch_job(arguments, exp_params):

    # Set dynamic parameters for the batch job as environment variables
    # But dont forget to add the os.environ to the new environment variables otherwise the PATH is not found
    env = {
        **os.environ,
        "EXP_PARAMS": exp_params,
        "DEFAULT_DIR": os.getcwd(),
        "TUSTU_SYNC_INTERVAL": '0'
    }

    # For debugging and local runs
    if shutil.which('sbatch') is None:
        print("SLURM not available. Would submit job with:", exp_params)

        # Run DVC experiment directly
        cmd = 'dvc exp run $EXP_PARAMS'
        print("Run experiment", cmd)
        subprocess.run(cmd, shell=True, env=env)

        # Copy logs dir to local_logs
        # shutil.copytree('logs', 'local_logs', dirs_exist_ok=True)

        # TODO: setup remote
        # print("Push to remote...")
        # subprocess.run("dvc exp push origin", shell=True)
        return
    
    # Run sbatch command with the environment variables as bash! subprocess! command (otherwise module not found)
    subprocess.run(['/usr/bin/bash', '-c', f'sbatch slurm_job.sh {" ".join(arguments)}'], env=env)

def create_exp_params_str(config_dict):
    exp_params_str = ''
    for key, value in config_dict.items():
        exp_params_str += f"-S  {key}={str(value)} "
    return exp_params_str

if __name__ == "__main__":

    arguments = sys.argv[1:]

    # Define Experiment Name
    experiment_name = 'MultiTask_SpatialEmbeddings'

    # Define Base Config
    base_config = {
        # Define which config files are used
        #"general": general_config,
        # "dataset": 'default',
        "embeddings": 'all_embeddings',
        "model": 'TemporalCNN',
        #"objectives": objectives_config,
        "train": 'perch2_spatial_embeddings',

        # Define specific parameters
        "log.experiment_name": experiment_name,
        #"train.train_size_batches": train_size_batches
    }

    # Define all lists of parameters or config files [Hyperparameters]
    hyperparams = {'objectives': ['only_polyphony_degree', 'multi_task_v1_add_event_logits', 'multi_task_v2_add_framewise_polyphony']
                    #'train.train_size_batches': [10, 30, 76]
                    # input_feature = ['audio', 'audio_no_noise']
                    # train.learning_rate = [0.001, 0.0001]
                    # train.batch_size = [32, 128, 256]
                }

    # Iterate over all combinations of parameters
    # for params in (dict(zip(hyperparams.keys(), values)) for values in itertools.product(*hyperparams.values())):
    # #for objectives_config, train_size_batches in itertools.product(hyperparams['objectives_configs'], hyperparams['train_sizes_batches']):
    #     # Define Experiment
    #     # base_config = {
    #     #         # Define which config files are used
    #     #         #"general": general_config,
    #     #         # "dataset": 'default',
    #     #         "embeddings": 'all_embeddings',
    #     #         "model": 'TemporalCNN',
    #     #         #"objectives": objectives_config,
    #     #         "train": 'perch2_spatial_embeddings',

    #     #         # Define specific parameters
    #     #         "log.experiment_name": 'MultiTask-Perch2-Spatial-Embeddings',
    #     #         #"train.train_size_batches": train_size_batches
    #     #     }
        
    #     # Add hyperparameters
    #     config_dict.update(params)

    # Add hyperparameters
    all_hyper_parameter_combinations = (dict(zip(hyperparams.keys(), values)) for values in itertools.product(*hyperparams.values()))
    print(all_hyper_parameter_combinations)
    for hyperparams_config in all_hyper_parameter_combinations:

        # Get hyperparams keys for logging purposes
        hyperparams_keys_str = ",".join(hyperparams_config.keys())
        hyperparams_keys = {"log.hyperparameters": f"[{hyperparams_keys_str}]"}

        # Create config
        config_dict = base_config | hyperparams_config | hyperparams_keys
        
        # Submit job for every hyperparameter configuration
        exp_params = create_exp_params_str(config_dict)
        submit_batch_job(arguments, exp_params)