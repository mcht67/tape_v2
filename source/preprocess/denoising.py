from IPython import display as disp
import os
import torch
from biodenoising import pretrained
from biodenoising.denoiser.dsp import convert_audio
from datasets import concatenate_datasets
from matplotlib import pyplot as plt
import numpy as np
from functools import partial

def denoise_example(example, model, denoising_key, device):
    sampling_rate = example['sampling_rate']
    audio = torch.tensor(example['audio'])
    # Add channel dim, if needed
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    wav = convert_audio(audio, sampling_rate, model.sample_rate, model.chin).to(device)
    with torch.no_grad():
        example[denoising_key] = model(wav[None])[0]
    return example

# def add_denoising_batchwise(dataset, cache_dir, denoising_key='audio_denoised', batch_size=100, recompute=False):
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')
#     modified = False
#     if recompute or denoising_key not in dataset.features:
#         model = pretrained.biodenoising16k_dns48().to(device)
#         denoising_fn = partial(
#                 denoise_example,
#                 model=model, 
#                 device=device
#             )

#          # Process in batches
#         processed_datasets = []
#         total_samples = len(dataset)
        
#         for i in range(0, total_samples, batch_size):
#             end_idx = min(i + batch_size, total_samples)
#             print(f"Processing batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")
            
#             # Select batch
#             batch_dataset = dataset.select(range(i, end_idx))
            
#             # Process batch
#             cache_file = os.path.join(cache_dir, f"{denoising_key}_batch_{i}_{end_idx}_cache.arrow")
#             batch_processed = batch_dataset.map(denoising_fn, cache_file_name=cache_file)
    
#             # Concatenate all processed batches
#             print(f"Concatenating {len(processed_datasets)} batches...")
#             dataset = concatenate_datasets(processed_datasets)
#             modified = True
    
#     return dataset, modified

def add_denoising_batchwise(dataset, cache_dir, denoising_key='audio_denoised', batch_size=100, recompute=False):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    modified = False
    if recompute or denoising_key not in dataset.features:
        model = pretrained.biodenoising16k_dns48().to(device)
        print(model)
        denoising_fn = partial(
                denoise_example,
                model=model,
                denoising_key=denoising_key, 
                device=device
            )
        return process_batchwise(dataset, denoising_fn, denoising_key, cache_dir, batch_size)
    return dataset, modified

def process_batchwise(dataset, process_fn, feature_key, cache_dir, batch_size=100):
    processed_datasets = []
    total_samples = len(dataset)
    
    for i in range(0, total_samples, batch_size):

        if i >= total_samples:
            break

        end_idx = min(i + batch_size, total_samples)
        print(f"Processing batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")
        
        # Select batch
        batch_dataset = dataset.select(range(i, end_idx))
        
        # Process batch
        cache_file = os.path.join(cache_dir, f"{feature_key}_batch_{i}_{end_idx}_cache.arrow")
        batch_processed = batch_dataset.map(process_fn, cache_file_name=cache_file)
        processed_datasets.append(batch_processed)

        # Concatenate all processed batches
        print(f"Concatenating {len(processed_datasets)} batches...")
        dataset = concatenate_datasets(processed_datasets)
        modified = True

    return dataset, modified