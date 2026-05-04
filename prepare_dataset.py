import subprocess
import argparse
import json
import os
from dotenv import load_dotenv
from omegaconf import OmegaConf
from hydra import compose, initialize
import huggingface_hub

if __name__ == "__main__":

    # Define arguments
    parser = argparse.ArgumentParser(
        description="Updatess dataset when provided with lists of input_features, embeddings and labels by computing missing ones."
    )

    parser.add_argument("--huggingface_path", type=str)
    parser.add_argument("--dataset_config", type=str)
    parser.add_argument("--input_features", type=json.loads)
    parser.add_argument("--embeddings", type=json.loads)
    # parser.add_argument("--labels", type=json.loads)
    # parser.add_argument('--objectives', type=json.loads)
    parser.add_argument('--recompute_embeddings', action='store_true')
    parser.add_argument('--force_redownload', action='store_true')
    # parser.add_argument('--recompute_labels', action='store_true')
    args = parser.parse_args()

     ##########################
    # Huggingface login
    ##########################
   
    # load_dotenv('local.env')
    # #huggingface_hub.login(token=os.getenv('HUGGINGFACE_TOKEN'))
    # huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

    # ########################
    # # Python versions
    # ########################

    # # Get docker python paths
    # base_python = os.getenv('DOCKER_BASE_PYTHON')
    # perch_python = os.getenv('DOCKER_PERCH_PYTHON')
    # train_python = os.getenv('DOCKER_TRAIN_PYTHON')
    complete_python = os.getenv(('DOCKER_COMPLETE_PYTHON'))

    # Init config and save for prepare_dataset.py to use 
    # [local python paths, dataset download and upload paths]
    with initialize(config_path="conf", version_base=None):
        cfg = compose(config_name="config")
        #print(cfg)
    # OmegaConf.save(cfg, "params.yaml")

    # # # Get local python paths from default config if docker paths not defined
    # cfg = OmegaConf.load("params.yaml")
    # if not base_python:
    #     base_python = cfg.python.base
    # if not perch_python:
    #     perch_python = cfg.python.perch
    # if not train_python:
    #     train_python = cfg.python.train
    if not complete_python:
        complete_python = cfg.python.complete

    ########################
    # Setup
    ########################
    huggingface_path = args.huggingface_path
    dataset_config = args.dataset_config
    input_features = args.input_features
    embeddings = args.embeddings
    # labels = args.labels
    # objectives = args.objectives
    recompute_embeddings = args.recompute_embeddings 
    # recompute_labels = args.recompute_labels

    # ##########################
    # # Download/Update dataset
    # ##########################

    # # Download dataset to use hf dataset offline inside the jobs, avoiding hitting rate limit on hf cache

    # cmd =   [
    #             complete_python, #base_python, 
    #             "source/load_dataset.py",
    #             "--huggingface_path", huggingface_path,
    #             "--dataset_config", dataset_config,
    #         ]
    # print(cmd)
    # subprocess.run(cmd, check=True)

    ##########################
    # Embed audio with perch
    ##########################
    embeddings_uploaded = False

    if input_features and embeddings:
        cmd = [
            complete_python,
            "source/perch_embed_audio_stream.py",
            "--huggingface_path", huggingface_path,
            "--dataset_config", dataset_config,
            "--input_features", json.dumps(input_features),
            "--embeddings", json.dumps(embeddings),
        ]
        if recompute_embeddings:
            cmd.append("--force_recompute")

        result = subprocess.run(cmd)
     
        if result.returncode == 0:
            embeddings_uploaded = True
        elif result.returncode == 2:
            print("Embeddings already exist. Perch embedding script skipped did not add embeddings.")
        else:
            raise RuntimeError(f"perch_embed_audio_stream.py failed with exit code {result.returncode}")

    ###########################
    # Embed audio with birdset
    ###########################
    if input_features and embeddings:
        cmd = [
            complete_python,
            "source/birdset_embed_audio_stream.py",
            "--huggingface_path", huggingface_path,
            "--dataset_config", dataset_config,
            "--input_features", json.dumps(input_features),
            "--embeddings", json.dumps(embeddings),
        ]
        if recompute_embeddings:
            cmd.append("--force_recompute")

        result = subprocess.run(cmd)

        if result.returncode == 0:
            embeddings_uploaded = True
        elif result.returncode == 2:
            print("Embeddings already exist. Birdset embedding script skipped did not add embeddings.")
        else:
            raise RuntimeError(f"birdset_embed_audio_stream.py failed with exit code {result.returncode}")

    ###########################
    # Load dataset
    ###########################
    if embeddings_uploaded or force_redownload:
        cmd = [
            complete_python,
            "source/load_dataset.py",
            "--huggingface_path", huggingface_path,
            "--dataset_config", dataset_config,
            "--download_mode", "force_redownload"
        ]
        subprocess.run(cmd, check=True)
    else:
        print("No embeddings uploaded in any script. Skipping forced dataset download.")

    # DO IN TRAIN 
    # #########################
    # # Add labels
    # ##########################

    # if objectives:
    #     cmd =   [
    #                 base_python, 
    #                 "source/add_labels_stream.py",
    #                 "--huggingface_path", huggingface_path,
    #                 "--dataset_config", dataset_config,
    #                 "--objectives", json.dumps(objectives), 
    #             ]
    #     if recompute_labels: cmd.append("--force_recompute")
    #     subprocess.run(cmd, check=True)

