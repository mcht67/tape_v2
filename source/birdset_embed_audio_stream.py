import os
from datasets import Audio, load_from_disk
import torch
import sys
import json
import argparse
from dotenv import load_dotenv


from integrations.birdset import load_model_configs, add_embeddings_batchwise

from utils.dataset import load_dataset_with_retry, get_local_data_dir, overwrite_dataset

def main():
    ###################################################
    # Configuration
    ###################################################

    print("Running birdset embedding script...")

    # Define arguments
    parser = argparse.ArgumentParser(
        description="Computes misssing birdset and updates dataset."
    )

    parser.add_argument("--huggingface_path", type=str)
    parser.add_argument("--dataset_config", type=str)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--input_features", type=json.loads)
    parser.add_argument("--embeddings", type=json.loads)
    parser.add_argument('--force_recompute', action='store_true')
    args = parser.parse_args()

    huggingface_path = args.huggingface_path
    dataset_config = args.dataset_config
    subset = args.subset
    input_features = args.input_features
    embeddings = args.embeddings
    force_recompute = args.force_recompute  

    # Exit script if no features or embeddings are passed
    if not embeddings or not input_features:
        print("No input features or no embeddings passed. Skipping.")
        sys.exit(2)

    batch_size = 50

    #  # Get default config
    # cfg = OmegaConf.load("params.yaml")
    # hf_download_path = cfg.dataset.huggingface.download_path
    # hf_upload_path = cfg.dataset.huggingface.upload_path

    ########################
    # Request GPU
    ########################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ########################
    # Load data
    ########################

    # Load environment variables from .env file
    load_dotenv('local.env')

    # Huggingface login
    # huggingface_hub.login(token=os.getenv('HUGGINGFACE_TOKEN'))
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

    model_configs = load_model_configs(embeddings, "conf/embeddings")

    # Filter by type "birdset"
    birdset_model_configs = {
                        key: cfg
                        for key, cfg in model_configs.items()
                        if cfg.get("type") == "birdset"
                    }
    
    if not birdset_model_configs:
        print("No birdset model configs found. Skipping.")
        sys.exit(2)

    # Load Dataset
    # Check if dataset exists locally
    train_config = subset + '_polyphonic' #'_' + str(study_config['base_config']['dataset.max_polyphony'])
    scape_test_config = subset + '_soundscape_test' #'_' + str(study_config['base_config']['dataset.max_polyphony'])

    dataset_configs = [train_config, scape_test_config] if subset!='XCM' and subset!="XCL" else [train_config]
    
    for dataset_config in dataset_configs:
        
        local_data_dir = get_local_data_dir(dataset_config=dataset_config, subset=subset)
        if not os.path.exists(local_data_dir):
            print(f"Dataset {dataset_config} not found locally. Downloading from Huggingface...")
            dataset = load_dataset_with_retry(huggingface_path, dataset_config, token=huggingface_token, download_mode='force_redownload')
        else:
            print(f"Dataset {dataset_config} found locally. Loading from disk: {local_data_dir}...")
            dataset = load_from_disk(local_data_dir)

        print({split: len(dataset[split]) for split in dataset.keys()})
        
        # ===================
        # Embed
        # ===================

        print("Start embedding...")
        # Compute embeddings
        for input_feature in input_features:
            for split in dataset.keys():     
                dataset[split] = dataset[split]
                dataset[split] = dataset[split].cast_column(input_feature, Audio())

        if force_recompute:
            print("force_recompute is set to True. Recompute all embeddings!")

        embeddings_names = []
        embeddings_added = False
        for model_key in birdset_model_configs:
            for input_feature in input_features:
                for split in dataset.keys():
                    dataset[split], embeddings_name = add_embeddings_batchwise(input_feature, 
                                                                            model_key, 
                                                                            birdset_model_configs, 
                                                                            dataset[split], 
                                                                            split, 
                                                                            force_recompute=force_recompute,
                                                                            device=device,
                                                                            batch_size=batch_size
                                                                            )
                    if embeddings_name:
                        embeddings_names.append(embeddings_name)
                        embeddings_added = True
        print("Embedding completed.")


        if embeddings_added:
            # SAVE TO DISK
            os.makedirs(local_data_dir, exist_ok=True)
            print(f"Saving dataset with embeddings to {local_data_dir}...")
            overwrite_dataset(dataset, local_data_dir, store_backup=False)
            # dataset.save_to_disk(local_data_dir)
            print("Save done.")

        # UPLOAD TO HUGGINGFACE
    #     print("Upload embeddings...")
    #     # data_dir = get_data_dir(dataset_config)
    #     commit_message = f"adds {embeddings_names} to {dataset_config}"
    #     for attempt in range(5):
    #         try:
    #             dataset.push_to_hub(huggingface_path, config_name=dataset_config, data_dir=data_dir, commit_message=commit_message, token=huggingface_token)
    #             break
    #         except Exception as e:
    #             if isinstance(e, ConnectionError) or "503" in str(e) or "504" in str(e):
    #                 print(f"Attempt {attempt+1} failed: {e}")
    #                 if attempt < 4:
    #                     time.sleep(60 * (attempt + 1))  # back-off
    #                 else:
    #                     raise
    #     print("Upload done.")
    #     print("Finished birdset embedding script.")
    #     sys.exit(0)
    if not embeddings_added:
        print("No embeddings added. Skip upload.")
        print("Finished birdset embedding script.")  
        sys.exit(2)
if __name__=="__main__":
     main()