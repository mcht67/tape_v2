import os
import numpy as np
import datasets
from omegaconf import OmegaConf
from hydra.utils import instantiate
from datasets import load_from_disk, concatenate_datasets, Audio
from functools import partial
import tempfile
import torch
import torchaudio
from hydra.utils import instantiate
import sys
import json
from datetime import datetime

from utils.general import store_embeddings, overwrite_dataset

def get_embedding_keys(model_name, input_feature):
    return {
    "pooled_embeddings": model_name + "_" + input_feature + "_pooled_embeddings",
    "spatial_embeddings": model_name + "_" + input_feature + "_spatial_embeddings"
    }

def add_embeddings_batchwise(input_feature, model_key, model_configs, dataset, split_key, force_recompute=False, cache_dir=None, batch_size=100, dataset_path=None, dataset_metadata_path=None):
    
    # Instantiate model
    model_cfg = model_configs[model_key]
    model = instantiate(model_cfg)

    # Get model name
    model_path = model_cfg.pretrained_model_path
    model_name = model_path.replace("DBD-research-group/", "")

    with tempfile.TemporaryDirectory() as temp_cache_dir:

        # Set HuggingFace cache to this temporary directory
        datasets.config.HF_DATASETS_CACHE = temp_cache_dir
        
        embeddings_keys = get_embedding_keys(model_name, input_feature).values()

        # any(example in split_key if example.get(key) is not None)
        for key in embeddings_keys:
            features = dataset.features
            boolean = key in features

        if any(key in dataset.features for key in embeddings_keys) and not force_recompute:
            print("Embedding with model", model_name, "for", input_feature, "in", split_key, "split has already been calculated, skipping.")
            return dataset, None

        # print("######################################################################")
        print("Embed", input_feature, "with:", model_name)
        # print("######################################################################")    
            
        embedding_fn = partial(
                embed_example_batched,
                model=model,
                model_name=model_name,
                input_feature=input_feature
            )
        
        # Process in batches
        processed_datasets = []
        total_samples = len(dataset)
        
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            print(f"Processing batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")
            
            # Select batch
            batch_dataset = dataset.select(range(i, end_idx))
            
            # Process batch
            cache_file = os.path.join(temp_cache_dir, f"{model_name}_{input_feature}_{split_key}_batch_{i}_{end_idx}_cache.arrow")

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

        embeddings_name = model_name + "_" + input_feature

        pooled_embeddings_key, spatial_embeddings_key = get_embedding_keys(model_name, input_feature).values()

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

# def add_embeddings_batchwise(input_features, models_config, dataset, split_key, cache_dir=None, batch_size=100, dataset_path=None, dataset_metadata_path=None):
    
#     embeddings_names = []

#     for model_key in models_config:

#         # Instantiate model
#         model_cfg = models_config[model_key]
#         model = instantiate(model_cfg)

#         # Get model name
#         model_path = model_cfg.pretrained_model_path
#         model_name = model_path.replace("DBD-research-group/", "")

#         for input_feature in input_features:

#             with tempfile.TemporaryDirectory() as temp_cache_dir:

#                 # Set HuggingFace cache to this temporary directory
#                 datasets.config.HF_DATASETS_CACHE = temp_cache_dir
                
#                 embeddings_keys = get_embedding_keys(model_name, input_feature).values()

#                 # any(example in split_key if example.get(key) is not None)

#                 if any(key in dataset.features for key in embeddings_keys): # TODO: skips even if spatial embeddings has been added later
#                     print("Embedding with model", model_name, "for", input_feature, "has already been calculated, skipping.")
#                     continue      

#                 print("######################################################################")
#                 print("Embed", input_feature, "with:", model_name)
#                 print("######################################################################")    
                    
#                 embedding_fn = partial(
#                         embed_example_batched,
#                         model=model,
#                         model_name=model_name,
#                         input_feature=input_feature
#                     )
                
#                 # Process in batches
#                 processed_datasets = []
#                 total_samples = len(dataset)
                
#                 for i in range(0, total_samples, batch_size):
#                     end_idx = min(i + batch_size, total_samples)
#                     print(f"Processing batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")
                    
#                     # Select batch
#                     batch_dataset = dataset.select(range(i, end_idx))
                    
#                     # Process batch
#                     cache_file = os.path.join(temp_cache_dir, f"{model_name}_{input_feature}_{split_key}_batch_{i}_{end_idx}_cache.arrow")

#                     #batch_processed = batch_dataset.map(embedding_fn, cache_file_name=cache_file)
#                     # Enable batched processing
#                     batch_processed = batch_dataset.map(
#                         embedding_fn, 
#                         batched=True,
#                         batch_size=50,
#                         cache_file_name=cache_file
#                     )
                    
#                     processed_datasets.append(batch_processed)
                
#                 # Concatenate all processed batches
#                 print(f"Concatenating {len(processed_datasets)} batches...")
#                 dataset = concatenate_datasets(processed_datasets)

#                 print("######################################################################")
#                 print("Store emmbeddings", model_name, "for", input_feature)
#                 print("######################################################################") 

#                 # Store changes
#                 if dataset_path:
#                     overwrite_dataset(dataset, dataset_path, metadata_path=dataset_metadata_path, store_backup=False)

#                 # Store metadata
#                 embeddings_name = model_name + input_feature
#                 embeddings_names.append(embeddings_name)
#                 print(embeddings_names)
#                 metadata = {
#                         "datetime": datetime.now().isoformat(),
#                         "dataset_path": dataset_path,
#                         "embeddings_added": embeddings_names
#                     }

#                 if dataset_metadata_path:
#                     metadata_dir = os.path.dirname(dataset_metadata_path)
#                     if metadata_dir:
#                         os.makedirs(metadata_dir, exist_ok=True)
#                     with open(dataset_metadata_path, "w") as f:
#                                 json.dump(metadata, f, indent=2)
            
#     return dataset

def embed_example_batched(examples, model, model_name, input_feature):
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

    # Check if all have same sampling rate
    sampling_rates = [int(audio['sampling_rate']) for audio in examples[input_feature]]
    
    if len(set(sampling_rates)) == 1 and sampling_rates[0] != model_sampling_rate:
        # All same rate - create resampler once
        resample = torchaudio.transforms.Resample(
            orig_freq=sampling_rates[0], 
            new_freq=model_sampling_rate
        )
    else:
        resample = None
    
    # Process batch
    batch_audio_list = []
    
    for audio in examples[input_feature]:
        # Extract and resample each audio
        audio_array = audio['array']
        sampling_rate = int(audio['sampling_rate'])
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        
        batch_audio_list.append(audio_tensor)
    
    # Pad sequences to same length (required for batching)
    max_length = max(audio.shape[-1] for audio in batch_audio_list)
    
    padded_batch = []
    for audio in batch_audio_list:
        if audio.shape[-1] < max_length:
            padding = max_length - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        padded_batch.append(audio)
    
    # Stack into batch tensor: [batch_size, time]
    audio_batch = torch.stack(padded_batch)
    
    # Move to GPU if available
    device = next(model.parameters()).device
    audio_batch = audio_batch.to(device)
    
    # Process entire batch at once
    with torch.no_grad():
        outputs = model(audio_batch)
    
    # Convert outputs back to CPU and to lists
    if outputs.pooled_embeddings is not None:
        pooled_emb = outputs.pooled_embeddings.cpu().numpy()
        examples[pooled_embeddings_key] = [emb for emb in pooled_emb]
    
    # if outputs.spatial_embeddings is not None:
    #     spatial_emb = outputs.spatial_embeddings.cpu().numpy()
    #     examples[spatial_embeddings_key] = [emb for emb in spatial_emb]
    
    return examples

def embed_example(example, model, model_name, input_feature):

    # Get audio
    audio = example[input_feature]

    # Resample
    default_sampling_rate = 22050
    if not (model_sampling_rate := model.sampling_rate):
        model_sampling_rate = default_sampling_rate
        model.set_sampling_rate(model_sampling_rate)

    audio_array = audio['array']
    sampling_rate = int(audio['sampling_rate'])

    audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
    resample = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=model_sampling_rate)
    audio_resampled = resample(audio_tensor)

    # Get embeddings keys
    embeddings_keys = get_embedding_keys(model_name, input_feature)
    pooled_embeddings_key = embeddings_keys['pooled_embeddings']
    spatial_embeddings_key = embeddings_keys['spatial_embeddings']

    # Normalize
    # audio_resampled = audio / (np.max(np.abs(audio)) + 1e-9)

    # Embed
    outputs = model(audio_resampled)
    if outputs.pooled_embeddings:
         example[pooled_embeddings_key] = outputs.pooled_embeddings
    if outputs.spatial_embeddings:
        example[spatial_embeddings_key] = outputs.spatial_embeddings

    return example

# def store_embeddings(dataset, dataset_path, dataset_metadata_path, embeddings_names):
#     # print("######################################################################")
#     print("Store emmbeddings", embeddings_names[-1])
#     # print("######################################################################") 

#     # Store changes
#     if dataset_path:
#         overwrite_dataset(dataset, dataset_path, metadata_path=dataset_metadata_path, store_backup=False)

#     # Store metadata
#     metadata = {
#             "datetime": datetime.now().isoformat(),
#             "dataset_path": dataset_path,
#             "embeddings_added": embeddings_names
#         }

#     if dataset_metadata_path:
#         metadata_dir = os.path.dirname(dataset_metadata_path)
#         if metadata_dir:
#             os.makedirs(metadata_dir, exist_ok=True)
#         with open(dataset_metadata_path, "w") as f:
#                     json.dump(metadata, f, indent=2)

def main():
    # with tempfile.TemporaryDirectory() as temp_cache_dir:

    # # Set HuggingFace cache to this temporary directory
    # datasets.config.HF_DATASETS_CACHE = temp_cache_dir

    ###################################################
    # Configuration
    ###################################################

    cfg = OmegaConf.load("params.yaml")

    dataset_path = cfg.path.dataset
    dataset_metadata_path = cfg.path.dataset_metadata
    embeddings_metadata_path = cfg.path.birdset_embeddings_metadata

    # Skip stage if no features or perch models are defined in embeddings config
    if not 'input_features' in cfg.embeddings or not 'perch_models' in cfg.embeddings:

        # Store metadata
        metadata = {
                "datetime": datetime.now().isoformat(),
                "dataset_path": dataset_path,
                "embeddings added": None
            }

        metadata_dir = os.path.dirname(embeddings_metadata_path)
        if metadata_dir:
            os.makedirs(metadata_dir, exist_ok=True)
        with open(embeddings_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print("No input features or no birdset models defined. Skip stage.")

        sys.exit(0)

    input_features = cfg.embeddings.input_features
    model_configs = cfg.embeddings.birdset_models
    force_recompute = cfg.embeddings.force_recompute
    

    # ===================
    # Embed
    # ===================

    # Load Dataset 
    dataset = load_from_disk(dataset_path)
    
    print("Start embedding...")
    # Compute embeddings
    for input_feature in input_features:
        for split in dataset.keys():     
            dataset[split] = dataset[split]
            dataset[split] = dataset[split].cast_column(input_feature, Audio())

    if force_recompute:
        print("force_recompute is set to True. Recompute all embeddings!")

    embeddings_names = []
    for model_key in model_configs:
        for input_feature in input_features:
            embeddings_added = False
            for split in dataset.keys():
                dataset[split], embeddings_name = add_embeddings_batchwise(input_feature, model_key, model_configs, dataset[split], split, force_recompute=force_recompute, dataset_path=dataset_path, dataset_metadata_path=dataset_metadata_path)
                if embeddings_name:
                    embeddings_names.append(embeddings_name)
                    embeddings_added = True
            try:
                if embeddings_added:
                    store_embeddings(dataset, dataset_path, embeddings_metadata_path, embeddings_names)
            except:
                subset = cfg.dataset.subset
                huggingface_user = 'mcht67'
                huggingface_dataset_name = 'polyphonic-bird-set-with-embeddings'

                huggingface_path = huggingface_user + "/" + huggingface_dataset_name

                commit_message_polyphonic = f"updates polyphonic dataset with in {subset}"
                dataset.push_to_hub(huggingface_path, config_name=subset, private=True, commit_message=commit_message_polyphonic)
            
            
            # if embeddings_added:
            #      store_embeddings(dataset, dataset_path, embeddings_metadata_path, embeddings_names)
    

            # print("######################################################################")
            # print("Store emmbeddings", embeddings_name)
            # print("######################################################################") 

            # # Store changes
            # if dataset_path:
            #     overwrite_dataset(dataset, dataset_path, metadata_path=dataset_metadata_path, store_backup=False)

            # # Store metadata
            # embeddings_names.append(embeddings_name)
            # print(embeddings_names)
            # metadata = {
            #         "datetime": datetime.now().isoformat(),
            #         "dataset_path": dataset_path,
            #         "embeddings_added": embeddings_names
            #     }

            # if dataset_metadata_path:
            #     metadata_dir = os.path.dirname(dataset_metadata_path)
            #     if metadata_dir:
            #         os.makedirs(metadata_dir, exist_ok=True)
            #     with open(dataset_metadata_path, "w") as f:
            #                 json.dump(metadata, f, indent=2)
    print("Embedding completed.")

    # # ===================
    # # Save dataset
    # # ===================
    # overwrite_dataset(dataset, dataset_path, metadata_path=dataset_metadata_path, store_backup=False)

    # model_names = []
    # for model_key in embedding_models:
    #     # Get model name
    #     model_cfg = embedding_models[model_key]
    #     model_path = model_cfg.pretrained_model_path
    #     model_names.append(model_path.replace("DBD-research-group/", ""))

    # print(model_names)

    # # Store metadata
    # metadata = {
    #         "datetime": datetime.now().isoformat(),
    #         "dataset_path": dataset_path,
    #         "embeddings_added": model_names
    #     }

    # metadata_dir = os.path.dirname(embeddings_metadata_path)
    # if metadata_dir:
    #     os.makedirs(metadata_dir, exist_ok=True)
    # with open(embeddings_metadata_path, "w") as f:
    #             json.dump(metadata, f, indent=2)

if __name__=="__main__":
     main()