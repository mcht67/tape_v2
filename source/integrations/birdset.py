import os
import numpy as np
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
import datasets
from datasets import concatenate_datasets
from functools import partial
import tempfile
import torch
import torchaudio
from hydra.utils import instantiate

from pathlib import Path

from utils.dsp import normalize_audio_array

def get_embedding_keys(model_name, input_feature):
    return {
    "pooled_embeddings": model_name + "_" + input_feature + "_pooled_embeddings",
    "spatial_embeddings": model_name + "_" + input_feature + "_spatial_embeddings"
    }

def add_embeddings_batchwise(input_feature, model_key, model_configs, dataset, split_key, force_recompute=False, batch_size=100, device=torch.device("cpu")):
    
    embeddings_keys = get_embedding_keys(model_key, input_feature).values()

    if any(key in dataset.features for key in embeddings_keys) and not force_recompute:
        print(f"Embedding with model {model_key} for {input_feature} in {split_key} split has already been calculated, skipping.")
        return dataset, None

    # Instantiate model
    model_cfg = model_configs[model_key]['model_cfg']
    model = instantiate(model_cfg)
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

    # # Single .map() call — HuggingFace handles batching and streams to Arrow cache
    # cache_file = os.path.join(
    #     datasets.config.HF_DATASETS_CACHE,
    #     f"{model_key}_{input_feature}_{split_key}_cache.arrow"
    # )
    # if force_recompute and os.path.exists(cache_file):
    #     os.remove(cache_file)

    dataset = dataset.map(
        embedding_fn,
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=not force_recompute,
        # cache_file_name=cache_file
    )

    embeddings_name = model_key + "_" + input_feature
    pooled_embeddings_key, spatial_embeddings_key = get_embedding_keys(model_key, input_feature).values()
    print("Pooled embeddings key:", pooled_embeddings_key)
    print("Spatial embeddings key:", spatial_embeddings_key)

    # Debug: print embedding dimensions
    example = dataset[0]
    embeddings_dim = np.shape(example[pooled_embeddings_key])
    print("Pooled embeddings dim:", embeddings_dim)
    try:
        spatial_embeddings_dim = np.shape(example[spatial_embeddings_key])
        print("Spatial embeddings dim:", spatial_embeddings_dim)
    except Exception:
        pass

    return dataset, embeddings_name

# def add_embeddings_batchwise(input_feature, model_key, model_configs, dataset, split_key, force_recompute=False, batch_size=100, device=torch.device("cpu")):
    
#     # Set HuggingFace cache to this temporary directory
#     #datasets.config.HF_DATASETS_CACHE = temp_cache_dir
    
#     embeddings_keys = get_embedding_keys(model_key, input_feature).values()

#     # any(example in split_key if example.get(key) is not None)
#     for key in embeddings_keys:
#         features = dataset.features
#         boolean = key in features

#     if any(key in dataset.features for key in embeddings_keys) and not force_recompute:
#         print("Embedding with model", model_key, "for", input_feature, "in", split_key, "split has already been calculated, skipping.")
#         return dataset, None
    
#     # Instantiate model
#     model_cfg = model_configs[model_key]['model_cfg']
#     model = instantiate(model_cfg)

#     # # Auto-detect device
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # print(f"Using device: {device}")

#     if hasattr(model, 'to'):
#         model = model.to(device)

#     print("######################################################################")
#     print("Embed", input_feature, "with:", model_key)
#     print("######################################################################")    
        
#     embedding_fn = partial(
#             embed_example_batched,
#             model=model,
#             model_name=model_key,
#             input_feature=input_feature,
#             device=device
#         )
    
#     # Process in batches
#     processed_datasets = []
#     total_samples = len(dataset)

#     with tempfile.TemporaryDirectory() as temp_cache_dir:
        
#         for i in range(0, total_samples, batch_size):
#             end_idx = min(i + batch_size, total_samples)
#             print(f"Processing batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")
            
#             # Select batch
#             batch_dataset = dataset.select(range(i, end_idx))
            
#             # Process batch
#             cache_file = os.path.join(temp_cache_dir, f"{model_key}_{input_feature}_{split_key}_batch_{i}_{end_idx}_cache.arrow")

#             #batch_processed = batch_dataset.map(embedding_fn, cache_file_name=cache_file)
#             # Enable batched processing
#             batch_processed = batch_dataset.map(
#                 embedding_fn, 
#                 batched=True,
#                 batch_size=50,
#                 cache_file_name=cache_file
#             )
            
#             processed_datasets.append(batch_processed)
        
#         # Concatenate all processed batches
#         print(f"Concatenating {len(processed_datasets)} batches...")
#         dataset = concatenate_datasets(processed_datasets)

#         # Load and concatenate memory-mapped shards — much cheaper
#         from datasets import load_from_disk, concatenate_datasets
#         result = concatenate_datasets([load_from_disk(p) for p in shard_paths])

#         embeddings_name = model_key + "_" + input_feature

#         pooled_embeddings_key, spatial_embeddings_key = get_embedding_keys(model_key, input_feature).values()

#         print("Pooled embeddings key:", pooled_embeddings_key)
#         print("Spatial embeddingskey:", spatial_embeddings_key)
#         # DEBUG: print embeddings dimension
#         example = dataset.take(1)
#         example_embeddings = example[pooled_embeddings_key]
#         embeddings_dim = np.shape(example_embeddings)
#         print("Pooled embeddings dim: ", embeddings_dim)

#         try:
#             example_spatial_embeddings = example[spatial_embeddings_key]
#             spatial_embeddings_dim = np.shape(example_spatial_embeddings)
#             print("Spatial embeddings dim: ", spatial_embeddings_dim)
#         except:
#             pass
    
#     return dataset, embeddings_name

def embed_example_batched(examples, model, model_name, input_feature, device=torch.device("cpu")):
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

    # Process each audio sample, resampling individually if needed
    batch_audio_list = []
    valid_indices = []  # Track which examples have valid audio

    for i, audio in enumerate(examples[input_feature]):
        audio_array = audio['array'] if audio is not None else None
        
        if audio_array is None:
            batch_audio_list.append(None)  # Placeholder
            continue
        
        valid_indices.append(i)
        sampling_rate = int(audio['sampling_rate'])
        audio_array = normalize_audio_array(audio_array)
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        
        if sampling_rate != model_sampling_rate:
            resample = torchaudio.transforms.Resample(
                orig_freq=sampling_rate,
                new_freq=model_sampling_rate
            )
            audio_tensor = resample(audio_tensor)
        
        batch_audio_list.append(audio_tensor)

    # Initialize output lists with None for entire batch
    batch_size = len(examples[input_feature])
    pooled_results = [None] * batch_size
    spatial_results = [None] * batch_size

    # Only compute embeddings if there are valid samples
    if valid_indices:
        valid_tensors = [batch_audio_list[i] for i in valid_indices]
        
        # Pad sequences to same length (required for batching)
        max_length = max(audio.shape[-1] for audio in valid_tensors)
        padded_batch = []
        for audio in valid_tensors:
            if audio.shape[-1] < max_length:
                padding = max_length - audio.shape[-1]
                audio = torch.nn.functional.pad(audio, (0, padding))
            padded_batch.append(audio)

        # Stack into batch tensor: [batch_size, time]
        audio_batch = torch.stack(padded_batch).to(device)

        # Process entire batch at once
        with torch.no_grad():
            print(f"audio dtype: {audio_batch.dtype}, device: {audio_batch.device}")
            print(f"model device: {next(model.parameters()).device}")
            outputs = model(audio_batch)

        # Place results back at the correct indices
        if outputs.pooled_embeddings is not None:
            pooled_emb = outputs.pooled_embeddings.cpu().numpy()
            for result_idx, original_idx in enumerate(valid_indices):
                pooled_results[original_idx] = pooled_emb[result_idx]

        if outputs.spatial_embeddings is not None:
            spatial_emb = outputs.spatial_embeddings.cpu().numpy()
            for result_idx, original_idx in enumerate(valid_indices):
                spatial_results[original_idx] = spatial_emb[result_idx]

    examples[pooled_embeddings_key] = pooled_results
    examples[spatial_embeddings_key] = spatial_results
    return examples

    # # Process each audio sample, resampling individually if needed
    # batch_audio_list = []
    # for audio in examples[input_feature]:
    #     audio_array = audio['array']
    #     sampling_rate = int(audio['sampling_rate'])
    #     audio_array = normalize_audio_array(audio_array)
    #     audio_tensor = torch.tensor(audio_array, dtype=torch.float32)

    #     if sampling_rate != model_sampling_rate:
    #         resample = torchaudio.transforms.Resample(
    #             orig_freq=sampling_rate,
    #             new_freq=model_sampling_rate
    #         )
    #         audio_tensor = resample(audio_tensor)

    #     batch_audio_list.append(audio_tensor)
    
    # # Pad sequences to same length (required for batching)
    # max_length = max(audio.shape[-1] for audio in batch_audio_list)
    # padded_batch = []
    # for audio in batch_audio_list:
    #     if audio.shape[-1] < max_length:
    #         padding = max_length - audio.shape[-1]
    #         audio = torch.nn.functional.pad(audio, (0, padding))
    #     padded_batch.append(audio)

    # # Move to GPU if available
    # #device = next(model.parameters()).device
    
    # # Stack into batch tensor: [batch_size, time]
    # audio_batch = torch.stack(padded_batch).to(device)
    
    # # Process entire batch at once
    # with torch.no_grad():
    #     print(f"audio dtype: {audio_batch.dtype}, device: {audio_batch.device}")
    #     print(f"model device: {next(model.parameters()).device}")
    #     outputs = model(audio_batch)
    
    # # Convert outputs back to CPU and to lists
    # if outputs.pooled_embeddings is not None:
    #     pooled_emb = outputs.pooled_embeddings.cpu().numpy()
    #     examples[pooled_embeddings_key] = [emb for emb in pooled_emb]
    
    # if outputs.spatial_embeddings is not None:
    #     spatial_emb = outputs.spatial_embeddings.cpu().numpy()
    #     examples[spatial_embeddings_key] = [emb for emb in spatial_emb]
    
    # return examples

def compute_embedding(audio, model, device=torch.device("cpu")):

    # Get model sampling rate
    default_sampling_rate = 22050
    if not (model_sampling_rate := model.sampling_rate):
        model_sampling_rate = default_sampling_rate
        model.set_sampling_rate(model_sampling_rate)

    audio_array = normalize_audio_array(audio['array'])
    audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
    sampling_rate = int(audio['sampling_rate'])

    if sampling_rate != model_sampling_rate:
        resample = torchaudio.transforms.Resample(
            orig_freq=sampling_rate,
            new_freq=model_sampling_rate
        )
        audio_tensor = resample(audio_tensor)

    # Move to device and run inference
    audio_tensor = audio_tensor.to(device)

    with torch.no_grad():
        return model(audio_tensor)
    
def embed_example(example, model, model_name, input_feature, device=torch.device("cpu")):

    # Get audio
    audio = example[input_feature]

    # Get embeddings keys
    embeddings_keys = get_embedding_keys(model_name, input_feature)
    pooled_embeddings_key = embeddings_keys['pooled_embeddings']
    spatial_embeddings_key = embeddings_keys['spatial_embeddings']

    # Early return if audio is empty
    if audio is None or audio['array'] is None:
        example[pooled_embeddings_key] = None
        example[spatial_embeddings_key] = None
        return example

    # Embed
    outputs = compute_embedding(audio, model, device)
    if outputs.pooled_embeddings:
         example[pooled_embeddings_key] = outputs.pooled_embeddings.cpu().numpy()
    if outputs.spatial_embeddings:
        example[spatial_embeddings_key] = outputs.spatial_embeddings.cpu().numpy()

    return example


# def embed_example(example, model, model_name, input_feature, device=torch.device("cpu")):

#     # Get audio
#     audio = example[input_feature]

#     # Get model sampling rate
#     default_sampling_rate = 22050
#     if not (model_sampling_rate := model.sampling_rate):
#         model_sampling_rate = default_sampling_rate
#         model.set_sampling_rate(model_sampling_rate)

#     # audio_array = audio['array']
#     # sampling_rate = int(audio['sampling_rate'])

#     # audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
#     # resample = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=model_sampling_rate)
#     # audio_resampled = resample(audio_tensor)

#     audio_array = normalize_audio_array(audio['array'])
#     audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
#     sampling_rate = int(audio['sampling_rate'])

#     if sampling_rate != model_sampling_rate:
#         resample = torchaudio.transforms.Resample(
#             orig_freq=sampling_rate,
#             new_freq=model_sampling_rate
#         )
#         audio_tensor = resample(audio_tensor)

#     # Move to device and run inference
#     audio_tensor = audio_tensor.to(device)

#     # Get embeddings keys
#     embeddings_keys = get_embedding_keys(model_name, input_feature)
#     pooled_embeddings_key = embeddings_keys['pooled_embeddings']
#     spatial_embeddings_key = embeddings_keys['spatial_embeddings']

#     # Embed
#     with torch.no_grad():
#         outputs = model(audio_tensor)
#     if outputs.pooled_embeddings:
#          example[pooled_embeddings_key] = outputs.pooled_embeddings.cpu().numpy()
#     if outputs.spatial_embeddings:
#         example[spatial_embeddings_key] = outputs.spatial_embeddings.cpu().numpy()

#     return example

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