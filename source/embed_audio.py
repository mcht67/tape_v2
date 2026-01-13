from perch_hoplite.zoo import model_configs
from omegaconf import OmegaConf
import datasets
from datasets import load_from_disk, Audio
import numpy as np
from functools import partial
from utils.dsp import resample_audio
import os
import tensorflow_hub as hub
import tensorflow as tf
tf.experimental.numpy.experimental_enable_numpy_behavior()
from tensorflow import squeeze
from tensorflow.math import reduce_mean
from datasets import concatenate_datasets
import tempfile
import argparse

from utils.general import overwrite_dataset

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

def load_perch1_model(model_key):
    model_config_name = model_configs.ModelConfigName(model_key)
    preset_info = model_configs.get_preset_model_config(model_config_name)
    model = preset_info.load_model()
    sampling_rate = preset_info.model_config["sample_rate"]
    return model, sampling_rate

def load_perch2_model(model_key):
    if model_key == 'perch_v2_cpu':
        model = hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1')
        sampling_rate = 32000
    else:
         raise Exception("This is no perch_v2 model or loading this model is not supoorted yet!")
    return model, sampling_rate

# TODO: implement
def load_birdset_model(model_key):
    raise Exception("Not implemented.")
    return model, sampling_rate

def embed_example(example, model, model_key, model_family, input_feature, sampling_rate):

    audio = example[input_feature]
    audio = resample_audio(audio['array'], audio['sampling_rate'], sampling_rate)

    embeddings_key = model_key + "_embeddings"
    spatial_embeddigns_key = model_key + "_spatial_embeddings"

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    # Get spatial embedding
    if model_family == 'perch_v1':
        example[embeddings_key] = embed_with_perch1(model, audio)
    elif model_family == 'perch_v2':
        example[embeddings_key], example[spatial_embeddigns_key]= embed_with_perch2(model, audio)
    elif model_family == 'birdset':
        example[embeddings_key] = embed_with_birdset(model, audio)
    else:
        raise Exception("Model family is not supported.")

    return example

def embed_with_perch1(model, audio):
    outputs = model.embed(audio)
    embeddings = squeeze(outputs.embeddings)
    if embeddings.ndim > 1:
        embeddings = reduce_mean(embeddings, axis=0)
        print(f'Embeddings include multiple segments. Calculate mean.')
    return embeddings

def embed_with_perch2(model, audio):
    infer_fn = model.signatures['serving_default']
    audio_batched = audio[np.newaxis, :]  # Shape: (1, 160000)
    outputs = infer_fn(inputs=audio_batched)
    # logits = outputs['label']  # (1, 14795) - classification logits
    # spectrogram = outputs['spectrogram']  # (1, 500, 128)
    spatial_embeddings = outputs['spatial_embedding']  # (1, 16, 4, 1536) (batch, time, freq embeddings)
    one_dim_embeddings = outputs['embedding']  # (1, 1536) - mean pooled
    return tf.squeeze(one_dim_embeddings), tf.squeeze(spatial_embeddings)

# TODO: implement
def embed_with_birdset(model, audio):
    raise Exception("Embedding function for birdset is not implemented.")
    return embeddings


def add_embeddings(model_family, model_keys, input_feature, dataset, cache_dir, recompute=False):
    modified = False
    for model_key in model_keys:
        #if recompute or new_feature_key not in dataset.features:
        if model_family == 'perch_v1':
            model, sampling_rate = load_perch1_model(model_key)
        elif model_family == 'perch_v2':
            model, sampling_rate = load_perch2_model(model_key)
        elif model_family == 'birdset':
            load_birdset_model = load_birdset_model(model_key)
        else:
            raise Exception("Model family unknown. Can not load model.")
        embedding_fn = partial(
            embed_example,
            model=model,
            input_feature=input_feature,
            model_key=model_key,
            sampling_rate=sampling_rate,
        )
        cache_file = os.path.join(cache_dir, f"{model_key}_cache.arrow")
        dataset = dataset.map(embedding_fn, cache_file_name=cache_file)
        modified = True
    return dataset, modified

def add_embeddings_batchwise(model_family, model_keys, input_feature, dataset, cache_dir, batch_size=100):
    for model_key in model_keys:
        embedding_key = model_key + "_" + input_feature + "_embeddings"

        if embedding_key in dataset.features:
             print("Embedding with model", model_key, "for", input_feature, "has already been calculated, skipping.")
             break
        
        print(f"Processing {model_key} embeddings in batches of {batch_size}...")

        if model_family == 'perch_v1':
            model, sampling_rate = load_perch1_model(model_key)
        elif model_family == 'perch_v2':
            model, sampling_rate = load_perch2_model(model_key)
        elif model_family == 'birdset':
            load_birdset_model = load_birdset_model(model_key)
        else:
             raise Exception("Model family unknown. Can not load model.")
        
        embedding_fn = partial(
            embed_example,
            model=model,
            input_feature=input_feature,
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
    
    return dataset

# def add_embeddings_batchwise(model_keys, feature_key, dataset, cache_dir, batch_size=100):
    
#     for model_key in model_keys:
#         new_feature_key = model_key + "_embeddings"
#         print(f"Processing {model_key} embeddings in batches of {batch_size}...")
        
#         model, sampling_rate = load_model_by_key(model_key)
        
#         embedding_fn = partial(
#             embed_example,
#             model=model,
#             feature_key=feature_key,
#             new_feature_key=new_feature_key,
#             sampling_rate=sampling_rate,
#         )
        
#         # Process in batches
#         processed_datasets = []
#         total_samples = len(dataset)
        
#         for i in range(0, total_samples, batch_size):
#             end_idx = min(i + batch_size, total_samples)
#             print(f"Processing batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")
            
#             # Select batch
#             batch_dataset = dataset.select(range(i, end_idx))
            
#             # Process batch
#             cache_file = os.path.join(cache_dir, f"{model_key}_batch_{i}_{end_idx}_cache.arrow")
#             batch_processed = batch_dataset.map(embedding_fn, cache_file_name=cache_file)
            
#             processed_datasets.append(batch_processed)
        
#         # Concatenate all processed batches
#         print(f"Concatenating {len(processed_datasets)} batches...")
#         dataset = concatenate_datasets(processed_datasets)
    
#     return dataset

def main():
    with tempfile.TemporaryDirectory() as temp_cache_dir:

        # Set HuggingFace cache to this temporary directory
        datasets.config.HF_DATASETS_CACHE = temp_cache_dir

        # ===================
        # Configuration
        # ===================

        parser = argparse.ArgumentParser()
        #parser.add_argument("--model_keys", nargs="*", type=str, default=None)
        parser.add_argument("--version", type=str, default=None)
        args = parser.parse_args()
        print(args)

        cfg = OmegaConf.load("params.yaml")
        input_feature = cfg.embeddings.input_feature
        embedding_model = cfg.embeddings.model
        dataset_path = cfg.path.dataset

        print(embedding_model)

        # perch v1 available models
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

        perch_v1_models = ['birdnet_V2.1', 'birdnet_V2.2', 'birdnet_V2.3', 'perch_8', 'surfperch', 'vggish', 'yamnet', 'humpback', 'multispecies_whale', 'beans_baseline', 'aves']
        if args.version=='perch_v1' and embedding_model not in perch_v1_models:
            print("Requested embedding is no perch_v1 model, exiting.")
            exit(0)

        perch_v2_models = ['perch_v2', 'perch_v2_cpu']
        if args.version=='perch_v2' and embedding_model not in perch_v2_models:
                    print("Requested embedding is no perch_v2 model, exiting.")
                    exit(0)
        
        birdset_models = []
        if args.version=='birdset' and embedding_model not in birdset_models:
                    print("Requested embedding is no birdset model, exiting.")
                    exit(0)

        model_keys = [embedding_model]

        # Load Dataset 
        dataset = load_from_disk(dataset_path)
        dataset = dataset.cast_column(input_feature, Audio())

        # ===================
        # Embeddings
        # ===================

        print("Start embedding...")
        # Compute embeddings
        for split in dataset.keys():
            dataset[split] = add_embeddings_batchwise(args.version, model_keys, input_feature, dataset[split], temp_cache_dir)
        print("Embedding completed.")

        # ===================
        # Save dataset
        # ===================
        overwrite_dataset(dataset, dataset_path, store_backup=False)

if __name__ == "__main__":
    main()