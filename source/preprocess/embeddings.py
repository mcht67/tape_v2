from perch_hoplite.zoo import model_configs
from omegaconf import OmegaConf
from datasets import load_from_disk, Dataset
import numpy as np
from functools import partial
from utils.dsp import resample_audio
import os
from tensorflow import squeeze
from tensorflow.math import reduce_mean
from datasets import concatenate_datasets

# Available models
# BIRDNET_V2_1 = 'birdnet_V2.1'
# BIRDNET_V2_2 = 'birdnet_V2.2'
# BIRDNET_V2_3 = 'birdnet_V2.3'
# PERCH_8 = 'perch_8'
# SURFPERCH = 'surfperch'
# VGGISH = 'vggish'
# YAMNET = 'yamnet'
# HUMPBACK = 'humpback'
# MULTISPECIES_WHALE = 'multispecies_whale'
# BEANS_BASELINE = 'beans_baseline'
# AVES = 'aves'
# PLACEHOLDER = 'placeholder'

def initialize_model(model):
    """Initialize a perch_hoplite model by running a dummy inference."""
    try:
        # Create dummy audio data (adjust size based on your model requirements)
        import numpy as np
        dummy_audio = np.zeros((1, 32000), dtype=np.float32)  # 1 second at 32kHz
        
        # Try to run embedding to initialize everything
        try:
            _ = model.embed(dummy_audio)
            print("Model initialized successfully")
        except Exception as e:
            print(f"Model initialization attempt failed: {e}")
            # Try direct TFLite initialization
            if hasattr(model, 'model'):
                model.model.allocate_tensors()
                print("Tensors allocated directly")
    except Exception as e:
        print(f"Could not initialize model: {e}")
    
    return model

def load_model_by_key(model_key):
    model_config_name = model_configs.ModelConfigName(model_key)
    preset_info = model_configs.get_preset_model_config(model_config_name)
    model = preset_info.load_model()
    sampling_rate = preset_info.model_config["sample_rate"]

    # model = initialize_model(model)

    # # Ensure TFLite model is properly initialized
    # if hasattr(model, 'model') and hasattr(model.model, 'allocate_tensors'):
    #     model.model.allocate_tensors()

    return model, sampling_rate

def embed_example(example, model, feature_key, new_feature_key, sampling_rate):

    audio = example[feature_key]
    audio = resample_audio(audio['array'], audio['sampling_rate'], sampling_rate)

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    # Get embedding
    outputs = model.embed(audio)
    embeddings = squeeze(outputs.embeddings)
    if embeddings.ndim > 1:
        embeddings = reduce_mean(embeddings, axis=0)
        print(f'Embeddings include multiple segments. Calculate mean.')
    example[new_feature_key] = embeddings

    return example


def add_embeddings(model_keys, feature_key, dataset, cache_dir, recompute=False):
    modified = False
    for model_key in model_keys:
        new_feature_key = model_key + "_embeddings"
        if recompute or new_feature_key not in dataset.features:
            model, sampling_rate = load_model_by_key(model_key)
            embedding_fn = partial(
                embed_example,
                model=model,
                feature_key=feature_key,
                new_feature_key=new_feature_key,
                sampling_rate=sampling_rate,
            )
            cache_file = os.path.join(cache_dir, f"{model_key}_cache.arrow")
            dataset = dataset.map(embedding_fn, cache_file_name=cache_file)
            modified = True
    return dataset, modified

def add_embeddings_batchwise(model_keys, feature_key, dataset, cache_dir, batch_size=100, recompute=False):
    modified = False
    
    for model_key in model_keys:
        new_feature_key = model_key + "_embeddings"
        if recompute or new_feature_key not in dataset.features:
            print(f"Processing {model_key} embeddings in batches of {batch_size}...")
            
            model, sampling_rate = load_model_by_key(model_key)
            
            embedding_fn = partial(
                embed_example,
                model=model,
                feature_key=feature_key,
                new_feature_key=new_feature_key,
                sampling_rate=sampling_rate,
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
                cache_file = os.path.join(cache_dir, f"{model_key}_batch_{i}_{end_idx}_cache.arrow")
                batch_processed = batch_dataset.map(embedding_fn, cache_file_name=cache_file)
                
                processed_datasets.append(batch_processed)
            
            # Concatenate all processed batches
            print(f"Concatenating {len(processed_datasets)} batches...")
            dataset = concatenate_datasets(processed_datasets)
            modified = True
    
    return dataset, modified

