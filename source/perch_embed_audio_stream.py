from perch_hoplite.zoo import model_configs
from omegaconf import OmegaConf
from datasets import Audio, load_dataset
import numpy as np
from functools import partial
from utils.dsp import resample_audio
import os
import tensorflow_hub as hub
import tensorflow as tf
tf.experimental.numpy.experimental_enable_numpy_behavior()

from datasets import concatenate_datasets
import json
import argparse
import sys
import tempfile


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
    if model_key=='yamnet': 
        model = hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')
        sampling_rate = 16000
    elif model_key=='vggish':
        model = hub.load('https://www.kaggle.com/models/google/vggish/TensorFlow2/vggish/1')
        sampling_rate = 16000
    else:
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

def embed_example(example, model, model_key, embedding_type, input_feature, sampling_rate, device='/CPU:0'):

    audio = example[input_feature]
    audio = resample_audio(audio['array'], audio['sampling_rate'], sampling_rate)

    embeddings_key = model_key + "_" + input_feature + "_pooled_embeddings"
    spatial_embeddings_key = model_key + "_" + input_feature + "_spatial_embeddings"

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    # Get embeddings
    if embedding_type == 'perch_v1':
        pooled_embeddings, spatial_embeddings = embed_with_perch1(model, model_key, audio, device=device)
        if spatial_embeddings is not None:
            example[spatial_embeddings_key]= spatial_embeddings
        example[embeddings_key] = pooled_embeddings
    elif embedding_type == 'perch_v2':
        example[embeddings_key], example[spatial_embeddings_key]= embed_with_perch2(model, audio, device=device)
    # elif embedding_type == 'birdset':
    #     example[embeddings_key] = embed_with_birdset(model, audio, device=device)
    else:
        raise Exception("Model family is not supported.")

    return example

def embed_with_perch1(model, model_key, audio, device='/CPU:0'):
    spatial_embeddings = None
    with tf.device(device):
        if model_key == 'yamnet':
            scores, embeddings, log_mel_spectrogram = model(audio)
        elif model_key == 'vggish':
            embeddings = model(audio)
        else:
            outputs = model.embed(audio)
            embeddings = outputs.embeddings

        if embeddings.ndim > 1:
            spatial_embeddings = embeddings
            # Average pooling
            num_dims = len(embeddings.shape)
            axes_to_reduce = list(range(num_dims - 1))
            pooled_embeddings = tf.reduce_mean(embeddings, axis=axes_to_reduce, keepdims=False)
            pooled_embeddings = tf.reshape(pooled_embeddings, [-1])  # Flatten to 1D
            print(f'Embeddings include multiple segments. Calculate mean.')
        else:
            pooled_embeddings = tf.reshape(embeddings, [-1])

        if spatial_embeddings is not None:
            spatial_embeddings = tf.squeeze(spatial_embeddings)

    return pooled_embeddings.numpy(), spatial_embeddings.numpy() if spatial_embeddings is not None else None

def embed_with_perch2(model, audio, device='/CPU:0'):
    with tf.device(device):
        infer_fn = model.signatures['serving_default']
        audio_batched = tf.constant(audio[np.newaxis, :], dtype=tf.float32)  # Shape: (1, 160000)
        outputs = infer_fn(inputs=audio_batched)

        spatial_embeddings = outputs['spatial_embedding']  # (1, 16, 4, 1536)
        one_dim_embeddings = outputs['embedding']          # (1, 1536)

    return tf.squeeze(one_dim_embeddings).numpy(), tf.squeeze(spatial_embeddings).numpy()

# def embed_with_perch1(model, model_key, audio):
#     spatial_embeddings = None

#     if model_key=='yamnet':
#         scores, embeddings, log_mel_spectrogram = model(audio)
#     elif model_key=='vggish':
#         embeddings = model(audio)
#     else:
#         outputs = model.embed(audio)
#         embeddings = outputs.embeddings

#     if embeddings.ndim > 1:
#         spatial_embeddings = embeddings

#         # Average pooling
#         num_dims = len(embeddings.shape)
#         axes_to_reduce = list(range(num_dims - 1))
#         pooled_embeddings = tf.reduce_mean(embeddings, axis=axes_to_reduce, keepdims=False)
#         pooled_embeddings = tf.reshape(pooled_embeddings, [-1])  # Flatten to 1D
#         print(f'Embeddings include multiple segments. Calculate mean.')
#     else:
#         pooled_embeddings = tf.reshape(embeddings, [-1])  

#     if spatial_embeddings is not None:
#         spatial_embeddings = tf.squeeze(spatial_embeddings)

#     return pooled_embeddings, spatial_embeddings

# def embed_with_perch2(model, audio):
#     infer_fn = model.signatures['serving_default']
#     audio_batched = audio[np.newaxis, :]  # Shape: (1, 160000)
#     outputs = infer_fn(inputs=audio_batched)
#     # logits = outputs['label']  # (1, 14795) - classification logits
#     # spectrogram = outputs['spectrogram']  # (1, 500, 128)
#     spatial_embeddings = outputs['spatial_embedding']  # (1, 16, 4, 1536) (batch, time, freq embeddings)
#     one_dim_embeddings = outputs['embedding']  # (1, 1536) - mean pooled
#     return tf.squeeze(one_dim_embeddings), tf.squeeze(spatial_embeddings)

# # TODO: implement
# def embed_with_birdset(model, audio):
#     raise Exception("Embedding function for birdset is not implemented.")
#     return embeddings

def add_embeddings(embedding_type, model_keys, input_feature, dataset, cache_dir, recompute=False):
    modified = False
    for model_key in model_keys:
        #if recompute or new_feature_key not in dataset.features:
        if embedding_type == 'perch_v1':
            model, sampling_rate = load_perch1_model(model_key)
        elif embedding_type == 'perch_v2':
            model, sampling_rate = load_perch2_model(model_key)
        elif embedding_type == 'birdset':
            model, sampling_rate = load_birdset_model(model_key)
        else:
            raise Exception("Model family unknown. Can not load model.")
        embedding_fn = partial(
            embed_example,
            model=model,
            input_feature=input_feature,
            model_key=model_key,
            embedding_type = embedding_type,
            sampling_rate=sampling_rate,
        )
        cache_file = os.path.join(cache_dir, f"{model_key}_cache.arrow")
        dataset = dataset.map(embedding_fn, cache_file_name=cache_file)
        modified = True
    return dataset, modified

def add_embeddings_batchwise(model_key, dataset_split, input_feature, dataset, force_recompute=False, batch_size=100, device='/CPU:0'):
        
    embeddings_key = model_key + "_" + input_feature + "_pooled_embeddings"
    spatial_embeddings_key =  model_key + "_" + input_feature + "_spatial_embeddings"

    if embeddings_key in dataset.features and not force_recompute:
        print("Embedding with model", model_key, "for", input_feature, "has already been calculated, skipping.")
        return dataset, None
    
    # If embedding key is not in dataset compute
    print(f"Processing {embeddings_key} in batches of {batch_size}...")

    dataset = dataset.cast_column(input_feature, Audio())
    embedding_type = get_embedding_type(model_key)

    if embedding_type == 'perch_v1':
        model, sampling_rate = load_perch1_model(model_key)
    elif embedding_type == 'perch_v2':
        model, sampling_rate = load_perch2_model(model_key)
    elif embedding_type == 'birdset':
        load_birdset_model = load_birdset_model(model_key)
    else:
        print("Model family unknown. Can not load model.")
        return dataset, None
    
    embedding_fn = partial(
        embed_example,
        model=model,
        model_key=model_key,
        embedding_type=embedding_type,
        input_feature=input_feature,
        sampling_rate=sampling_rate,
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
            cache_file = os.path.join(temp_cache_dir, f"{embeddings_key}_{dataset_split}_batch_{i}_{end_idx}_cache.arrow")
            batch_processed = batch_dataset.map(embedding_fn, cache_file_name=cache_file)
            
            processed_datasets.append(batch_processed)
    
    # Concatenate all processed batches
    print(f"Concatenating {len(processed_datasets)} batches...")
    dataset = concatenate_datasets(processed_datasets)

    # DEBUG: print embeddings dimension
    example = dataset.take(1)
    example_embeddings = example[embeddings_key]
    embeddings_dim = np.shape(example_embeddings)
    print("Pooled embeddings dim: ", embeddings_dim)

    try:
        example_spatial_embeddings = example[spatial_embeddings_key]
        spatial_embeddings_dim = np.shape(example_spatial_embeddings)
        print("Spatial embeddings dim: ", spatial_embeddings_dim)
    except:
        pass
        
    return dataset, embeddings_key

def get_embedding_type(model_key):
            # Define available models
            perch_v1_models = ['birdnet_V2.1', 'birdnet_V2.2', 'birdnet_V2.3', 'perch_8', 'surfperch', 'vggish', 'yamnet', 'humpback', 'multispecies_whale', 'beans_baseline', 'aves']
            perch_v2_models = ['perch_v2', 'perch_v2_cpu']
            birdset_models = []

            # Get embedding type [perch_v1, perch_v2, birdset]
            if model_key in perch_v1_models:
                embedding_type = 'perch_v1'
            elif model_key in perch_v2_models:
                embedding_type = 'perch_v2'
            elif model_key in birdset_models:
                embedding_type = 'birdset'
            else:
                print("Could not get embedding type. Embedding model is not supported, skipping!")
                return None
            
            print(model_key, "is a ", embedding_type, "model.")
            return embedding_type

def main():

    print("Running perch embedding script...")

    # ===================
    # Configuration
    # ===================

    # Define arguments
    parser = argparse.ArgumentParser(
        description="Computes missing perch embeddings and updates dataset."
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
    embedding_models = args.embeddings
    force_recompute = args.force_recompute

    # Exit script if no features or embeddings are passed
    if not embedding_models or not input_features:
        print("No input features or no embeddings passed. Skipping.")
        sys.exit(0)
    
    # # Get default config
    # cfg = OmegaConf.load("params.yaml")
    # hf_download_path = cfg.dataset.huggingface.download_path
    # hf_upload_path = cfg.dataset.huggingface.upload_path

    # Filter embedding models bny type "perch_v1" and "perch_v2"
    perch_embeddings_models = [key for key in embedding_models if get_embedding_type(key)=='perch_v1' or get_embedding_type(key)=='perch_v2']

    if not perch_embeddings_models:
        print("No perch model keys found. Skipping.")
        sys.exit(0)

    ########################
    # Load data
    ########################

    # Load Dataset 
    dataset = load_dataset(huggingface_path, dataset_config)

    # Reduce dataset for testing purposes TODO: remove
    for split in dataset.keys():
        dataset[split] = dataset[split].select(range(10))

    ########################
    # Request GPU
    ########################

    gpus = tf.config.list_physical_devices('GPU')
    device = '/GPU:0' if gpus else '/CPU:0'
    print(f"Using device: {device}")

    # Prevent TF from grabbing all GPU memory at once
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    

    # ===================
    # Embeddings
    # ===================

    print("Running perch embedding script...")

    # Compute embeddings
    if force_recompute:
        print("force_recompute is set to True. Recompute all embeddings!")

    embeddings_names = []
    embeddings_added = False
    for model_key in embedding_models:
        for input_feature in input_features:
            for split in dataset.keys():
                dataset[split], embeddings_name = add_embeddings_batchwise(model_key, split, input_feature, dataset[split], force_recompute=force_recompute, device=device)
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

    print("Finished embedding with perch.") 

if __name__ == "__main__":
    main()