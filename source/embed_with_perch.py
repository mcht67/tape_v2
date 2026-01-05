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
    # model_config_name = model_configs.ModelConfigName(model_key)
    # preset_info = model_configs.get_preset_model_config(model_config_name)
    # model = preset_info.load_model()
    # sampling_rate = preset_info.model_config["sample_rate"]
    model = hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1')
    sampling_rate = 32000

    return model, sampling_rate

def embed_example(example, model, feature_key, new_feature_key, sampling_rate):

    audio = example[feature_key]
    audio = resample_audio(audio['array'], audio['sampling_rate'], sampling_rate)

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    # Get spatial embedding
    infer_fn = model.signatures['serving_default']
    audio_batched = audio[np.newaxis, :]  # Shape: (1, 160000)
    outputs = infer_fn(inputs=audio_batched)
    spatial_embedding = outputs['spatial_embedding']  # (1, 16, 4, 1536) - This is what you want!
    embedding = outputs['embedding']  # (1, 1536) - mean pooled
    # logits = outputs['label']  # (1, 14795) - classification logits
    # spectrogram = outputs['spectrogram']  # (1, 500, 128)
    example['perch_v2_cpu_spatial_embeddings'] = tf.squeeze(spatial_embedding, axis=1)

    #outputs = model.embed(audio)
    # embeddings = squeeze(outputs.embeddings)
    # if embeddings.ndim > 1:
    #     embeddings = reduce_mean(embeddings, axis=0)
    #     print(f'Embeddings include multiple segments. Calculate mean.')
    # example[new_feature_key] = embeddings

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

def add_embeddings_batchwise(model_keys, feature_key, dataset, cache_dir, batch_size=100):
    
    for model_key in model_keys:
        new_feature_key = model_key + "_embeddings"
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
    
    return dataset

def main():
    with tempfile.TemporaryDirectory() as temp_cache_dir:

        # Set HuggingFace cache to this temporary directory
        datasets.config.HF_DATASETS_CACHE = temp_cache_dir

        # ===================
        # Configuration
        # ===================

        # parser = argparse.ArgumentParser()
        # parser.add_argument("--model_keys", nargs="*", type=str, default=None)
        # parser.add_argument("--compute_embeddings", type=str, default="True")
        # args = parser.parse_args()
        # print(args)

        # if not args.model_keys or args.compute_embeddings == "False":
        #     print("No model keys passed or passed compute_embeddings='False'. Exiting.")
        #     exit(0)

        model_keys = ['perch_v2_cpu'] #= args.model_keys #cfg.embeddings.modelsm

        cfg = OmegaConf.load("params.yaml")

        feature_key = cfg.embeddings.feature
        dataset_path = cfg.paths.dataset

        # Load Dataset 
        dataset = load_from_disk(dataset_path)
        dataset = dataset.cast_column(feature_key, Audio())

        # ===================
        # Embeddings
        # ===================

        print("Start embedding...")
        # Compute embeddings
        dataset = add_embeddings_batchwise(model_keys, feature_key, dataset, temp_cache_dir)
        print("Embedding completed.")

        # ===================
        # Save dataset
        # ===================
        overwrite_dataset(dataset, dataset_path, store_backup=False)

if __name__ == "__main__":
    main()