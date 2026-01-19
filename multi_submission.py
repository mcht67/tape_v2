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
        print(cmd)
        subprocess.run(cmd, shell=True, env=env)
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

    # Define all lists of parameters or config files
    general_configs = ['default', 'v2']
    train_epochs = ['5', '10']

    # Iterate over all combinations of parameters
    for general_config, train_epoch in itertools.product(general_configs, train_epochs):
        # Define Experiment
        config_dict = {
            # Define which config files are used
            "general": general_config,
            # "dataset": 'default',
            # "embeddings": 'default',
            # "train": 'default',

            # Define specific parameters
            "train.epochs": train_epoch
            }

        exp_params = create_exp_params_str(config_dict)

        submit_batch_job(arguments, exp_params)