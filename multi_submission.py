#!./venv/bin/python

# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.

import itertools
import subprocess
import os
import sys
import shutil
import json

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

    ##########################
    # Configuration
    ##########################

    # Define Experiment Name
    experiment_name = 'Embeddings-Comparison'

    # Define Base Config
    base_config = {
        # # Define which config files are used -> Done in hydra config
        # #"general": general_config,
        # "dataset": 'PER',
        # "labels": 'all',
        # "embeddings": 'default',
        # "model": 'SimpleMLP',
        # "objectives": 'only_polyphony_degree', #'only_polyphony_degree_class',
        # #"train": 'perch2_spatial_embeddings',

        # Define specific parameters
        "log.experiment_name": experiment_name,
        "train.epochs": 5,
        #"train.initial_epoch": 20,
        #"train.train_size_batches": 20,
        #"train.val_size_batches": 5
        "train.learning_rate": 0.001,
        #"train.load_model_path": "model_path"
    }

    # Define all lists of parameters or config files [Hyperparameters]
    input_features = ['audio', 'no_noise_audio']
    embeddings = ['perch_8'] 
    hyperparams = {
                    # "train.input_feature_name": [
                    #                             "perch_v2_cpu_audio_embeddings",
                    #                             #"perch_v2_cpu_no_noise_audio_embeddings",
                    #                             #"birdnet_V2.3_audio_embeddings",
                    #                             #"birdnet_V2.3_no_noise_audio_embeddings",
                    #                             #"vggish_audio_embeddings",
                    #                             #"vggish_no_noise_audio_embeddings",
                    #                             #"perch_8_audio_embeddings",
                    #                             #"perch_8_no_noise_audio_embeddings",
                    #                             #"yamnet_audio_embeddings",
                    #                             #"yamnet_no_noise_audio_embeddings",
                    #                             #"beans_baseline_audio_embeddings",
                    #                             #"beans_baseline_no_noise_audio_embeddings",
                    #                             "EfficientNet-B1-BirdSet-XCL_audio_pooled_embeddings",
                    #                             #"EfficientNet-B1-BirdSet-XCL_no_noise_audio_pooled_embeddings",
                    #                             #"Bird-MAE-Huge_audio_pooled_embeddings",
                    #                             #"Bird-MAE-Huge_no_noise_audio_pooled_embeddings",
                    #                             #"AudioProtoPNet-20-BirdSet-XCL_audio_pooled_embeddings",
                    #                             #"AudioProtoPNet-20-BirdSet-XCL_no_noise_audio_pooled_embeddings",
                    #                             #"AST-Birdset-XCL_audio_pooled_embeddings",
                    #                             #"AST-Birdset-XCL_no_noise_audio_pooled_embeddings",
                    #                             #"Wav2Vec2-Base-BirdSet-XCL_audio_pooled_embeddings"]#,
                    #                             #"Wav2Vec2-Base-BirdSet-XCL_no_noise_audio_pooled_embeddings"
                    #                             ],
                    "train.input_feature": input_features,
                    "embeddings": embeddings                                 
                }

    ##########################
    # Prepare dataset
    ##########################

    # Collect all input features, embeddings and labels used
    # Pass params (and force_recompute?) to prepare_dataset.py

    # in prepare_dataset.py:

    # Check if embeddings are included in dataset 
    # Compute missing embeddings

    # Check if labels are included in dataset
    # Compute missing labels

    try:
        subprocess.run(["python", "prepare_dataset.py", 
                        "--input_features", json.dumps(input_features), 
                        "--embeddings", json.dumps(embeddings)],
                        check=True)
    except subprocess.CalledProcessError:
        print("Dataset preparation failed. Aborting experiment submission.")
        sys.exit(1)


    ##########################
    # Submit jobs
    ##########################
    # Add hyperparameters
    all_hyper_parameter_combinations = (dict(zip(hyperparams.keys(), values)) for values in itertools.product(*hyperparams.values()))
    print(all_hyper_parameter_combinations)
    for hyperparams_config in all_hyper_parameter_combinations:

        # Get hyperparams keys for logging purposes
        hyperparams_keys_str = ",".join(hyperparams_config.keys())
        hyperparams_keys = {"log.hyperparameters": f"[{hyperparams_keys_str}]"}
        if 'train.input_feature' in hyperparams_config and 'embeddings' in hyperparams_config:
            hyperparams_config['train.input_feature_name'] = hyperparams_config['embeddings'] + "_" + hyperparams_config['train.input_feature'] + "_embeddings"

        # Create config
        config_dict = base_config | hyperparams_config | hyperparams_keys

        print(config_dict)
        
        # Submit job for every hyperparameter configuration
        exp_params = create_exp_params_str(config_dict)
        submit_batch_job(arguments, exp_params, experiment_name)