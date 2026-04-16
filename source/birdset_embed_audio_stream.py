import os
import numpy as np
import datasets
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from datasets import load_dataset, concatenate_datasets, Audio
from functools import partial
import tempfile
import torch
import torchaudio
from hydra.utils import instantiate
import sys
import json
import argparse
from pathlib import Path

def get_embedding_keys(model_name, input_feature):
    return {
    "pooled_embeddings": model_name + "_" + input_feature + "_pooled_embeddings",
    "spatial_embeddings": model_name + "_" + input_feature + "_spatial_embeddings"
    }

def add_embeddings_batchwise(input_feature, model_key, model_configs, dataset, split_key, force_recompute=False, batch_size=100):
    
    # Set HuggingFace cache to this temporary directory
    #datasets.config.HF_DATASETS_CACHE = temp_cache_dir
    
    embeddings_keys = get_embedding_keys(model_key, input_feature).values()

    # any(example in split_key if example.get(key) is not None)
    for key in embeddings_keys:
        features = dataset.features
        boolean = key in features

    if any(key in dataset.features for key in embeddings_keys) and not force_recompute:
        print("Embedding with model", model_key, "for", input_feature, "in", split_key, "split has already been calculated, skipping.")
        return dataset, None
    
    # Instantiate model
    model_cfg = model_configs[model_key]['model_cfg']
    model = instantiate(model_cfg)

    # Auto-detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    if hasattr(model, 'to'):
        model = model.to(device)

    print("######################################################################")
    print("Embed", input_feature, "with:", model_key)
    print("######################################################################")    
        
    embedding_fn = partial(
            embed_example_batched,
            model=model,
            model_name=model_key,
            input_feature=input_feature,
            device=device
        )
    
    # Process in batches
    processed_datasets = []
    total_samples = len(dataset)

    with tempfile.TemporaryDirectory() as temp_cache_dir:
        
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            print(f"Processing batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")
            
            # Select batch
            batch_dataset = dataset.select(range(i, end_idx))
            
            # Process batch
            cache_file = os.path.join(temp_cache_dir, f"{model_key}_{input_feature}_{split_key}_batch_{i}_{end_idx}_cache.arrow")

            #batch_processed = batch_dataset.map(embedding_fn, cache_file_name=cache_file)
            # Enable batched processing
            batch_processed = batch_dataset.map(
                embedding_fn, 
                batched=True,
                batch_size=50,
                cache_file_name=cache_file
            )
            
            processed_datasets.append(batch_processed)
        
        # Concatenate all processed batches
        print(f"Concatenating {len(processed_datasets)} batches...")
        dataset = concatenate_datasets(processed_datasets)

        embeddings_name = model_key + "_" + input_feature

        pooled_embeddings_key, spatial_embeddings_key = get_embedding_keys(model_key, input_feature).values()

        print("Pooled embeddings key:", pooled_embeddings_key)
        print("Spatial embeddingskey:", spatial_embeddings_key)
        # DEBUG: print embeddings dimension
        example = dataset.take(1)
        example_embeddings = example[pooled_embeddings_key]
        embeddings_dim = np.shape(example_embeddings)
        print("Pooled embeddings dim: ", embeddings_dim)

        try:
            example_spatial_embeddings = example[spatial_embeddings_key]
            spatial_embeddings_dim = np.shape(example_spatial_embeddings)
            print("Spatial embeddings dim: ", spatial_embeddings_dim)
        except:
            pass
    
    return dataset, embeddings_name

def embed_example_batched(examples, model, model_name, input_feature, device="cpu"):
    """Process a batch of examples at once"""
    
    # Get model sampling rate
    default_sampling_rate = 22050
    if not (model_sampling_rate := model.sampling_rate):
        model_sampling_rate = default_sampling_rate
        model.set_sampling_rate(model_sampling_rate)
    
    # Get embedding keys
    embeddings_keys = get_embedding_keys(model_name, input_feature)
    pooled_embeddings_key = embeddings_keys['pooled_embeddings']
    spatial_embeddings_key = embeddings_keys['spatial_embeddings']

    # # Check if all have same sampling rate
    # sampling_rates = [int(audio['sampling_rate']) for audio in examples[input_feature]]
    
    # if len(set(sampling_rates)) == 1 and sampling_rates[0] != model_sampling_rate:
    #     # All same rate - create resampler once
    #     resample = torchaudio.transforms.Resample(
    #         orig_freq=sampling_rates[0], 
    #         new_freq=model_sampling_rate
    #     )
    # else:
    #     resample = None
    
    # # Process batch
    # batch_audio_list = []
    
    # for audio in examples[input_feature]:
    #     # Extract and resample each audio
    #     audio_array = audio['array']
    #     sampling_rate = int(audio['sampling_rate'])
    #     audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        
    #     batch_audio_list.append(audio_tensor)

    # Process each audio sample, resampling individually if needed
    batch_audio_list = []
    for audio in examples[input_feature]:
        audio_array = audio['array']
        sampling_rate = int(audio['sampling_rate'])
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)

        if sampling_rate != model_sampling_rate:
            resample = torchaudio.transforms.Resample(
                orig_freq=sampling_rate,
                new_freq=model_sampling_rate
            )
            audio_tensor = resample(audio_tensor)

        batch_audio_list.append(audio_tensor)
    
    # Pad sequences to same length (required for batching)
    max_length = max(audio.shape[-1] for audio in batch_audio_list)
    padded_batch = []
    for audio in batch_audio_list:
        if audio.shape[-1] < max_length:
            padding = max_length - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        padded_batch.append(audio)

    # Move to GPU if available
    #device = next(model.parameters()).device
    
    # Stack into batch tensor: [batch_size, time]
    audio_batch = torch.stack(padded_batch).to(device)
    
    # Process entire batch at once
    with torch.no_grad():
        outputs = model(audio_batch)
    
    # Convert outputs back to CPU and to lists
    if outputs.pooled_embeddings is not None:
        pooled_emb = outputs.pooled_embeddings.cpu().numpy()
        examples[pooled_embeddings_key] = [emb for emb in pooled_emb]
    
    if outputs.spatial_embeddings is not None:
        spatial_emb = outputs.spatial_embeddings.cpu().numpy()
        examples[spatial_embeddings_key] = [emb for emb in spatial_emb]
    
    return examples

def embed_example(example, model, model_name, input_feature, device="cpu"):

    # Get audio
    audio = example[input_feature]

    # Get model sampling rate
    default_sampling_rate = 22050
    if not (model_sampling_rate := model.sampling_rate):
        model_sampling_rate = default_sampling_rate
        model.set_sampling_rate(model_sampling_rate)

    # audio_array = audio['array']
    # sampling_rate = int(audio['sampling_rate'])

    # audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
    # resample = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=model_sampling_rate)
    # audio_resampled = resample(audio_tensor)
    
    # Resample if needed
    audio = example[input_feature]
    audio_tensor = torch.tensor(audio['array'], dtype=torch.float32)
    sampling_rate = int(audio['sampling_rate'])

    if sampling_rate != model_sampling_rate:
        resample = torchaudio.transforms.Resample(
            orig_freq=sampling_rate,
            new_freq=model_sampling_rate
        )
        audio_tensor = resample(audio_tensor)

    # Move to device and run inference
    audio_tensor = audio_tensor.to(device)

    # Get embeddings keys
    embeddings_keys = get_embedding_keys(model_name, input_feature)
    pooled_embeddings_key = embeddings_keys['pooled_embeddings']
    spatial_embeddings_key = embeddings_keys['spatial_embeddings']

    # Normalize
    # audio_resampled = audio / (np.max(np.abs(audio)) + 1e-9)

    # Embed
    with torch.no_grad():
        outputs = model(audio_resampled)
    if outputs.pooled_embeddings:
         example[pooled_embeddings_key] = outputs.pooled_embeddings.cpu().numpy()
    if outputs.spatial_embeddings:
        example[spatial_embeddings_key] = outputs.spatial_embeddings.cpu().numpy()

    return example

def load_model_configs(
                        model_keys: list[str],
                        embeddings_dir: str = "conf/embeddings",
                    ) -> dict[str, DictConfig]:
    """Load embedding configs for the given model_keys."""
    embeddings_path = Path(embeddings_dir).resolve()
    return {
        key: OmegaConf.load(embeddings_path / f"{key}.yaml")
        for key in model_keys
    }   

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
    parser.add_argument("--input_features", type=json.loads)
    parser.add_argument("--embeddings", type=json.loads)
    parser.add_argument('--force_recompute', action='store_true')
    args = parser.parse_args()

    huggingface_path = args.huggingface_path
    dataset_config = args.dataset_config
    input_features = args.input_features
    embeddings = args.embeddings
    force_recompute = args.force_recompute  

    # Exit script if no features or embeddings are passed
    if not embeddings or not input_features:
        print("No input features or no embeddings passed. Skipping.")
        sys.exit(0)

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

    model_configs = load_model_configs(embeddings, "conf/embeddings")

    # Filter by type "birdset"
    birdset_model_configs = {
                        key: cfg
                        for key, cfg in model_configs.items()
                        if cfg.get("type") == "birdset"
                    }
    
    if not birdset_model_configs:
        print("No birdset model configs found. Skipping.")
        sys.exit(0)

    # Load Dataset 
    dataset = load_dataset(huggingface_path, dataset_config)

    # Reduce dataset for testing purposes TODO: remove
    for split in dataset.keys():
        dataset[split] = dataset[split].select(range(10))
    
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
                                                                           force_recompute=force_recompute
                                                                           )
                if embeddings_name:
                    embeddings_names.append(embeddings_name)
                    embeddings_added = True
    print("Embedding completed.")

    if embeddings_added:
        print("Upload embeddings...")
        commit_message = f"adds {embeddings_names} to {dataset_config}"
        dataset.push_to_hub(huggingface_path, config_name=dataset_config, private=True, commit_message=commit_message)
        print("Upload done.")
    else:
        print("No embeddings added. Skip upload.")  

    print("Finished birdset embedding script.")

if __name__=="__main__":
     main()