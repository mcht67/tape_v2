from perch_hoplite.zoo import model_configs
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import tempfile
import os
from functools import partial
from datasets import Audio, concatenate_datasets

tf.experimental.numpy.experimental_enable_numpy_behavior()

from utils.dsp import resample_audio, normalize_audio_array

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
        if model_key in ['birdnet_V2.1', 'birdnet_V2.2', 'birdnet_V2.3', 'birdnet_V2.4']:
            # Set hop size to 1s for birdnet models (otherwise last 2 seconds of audio are ignored as BirdNet expects 3s of audio)
            preset_info.model_config.hop_size_s = 1.0
        model = preset_info.load_model()
        sampling_rate = preset_info.model_config["sample_rate"]
        print(f"Loaded model {model_key} with sampling rate {sampling_rate}.")
        print("model_config:", preset_info.model_config)
    return model, sampling_rate

def load_perch2_model(model_key):
    if model_key == 'perch_v2_cpu':
        model = hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1')
        sampling_rate = 32000
    else:
         raise Exception(f"Model {model_key} is not a supported perch_v2 model or loading this model is not supported yet!")
    return model, sampling_rate

# TODO: implement
def load_birdset_model(model_key):
    raise Exception("Not implemented.")
    return model, sampling_rate

def compute_embedding(audio, model, model_key, embedding_type, sampling_rate, device='/CPU:0'):

    audio_array = resample_audio(audio['array'], audio['sampling_rate'], sampling_rate)
    audio_array = normalize_audio_array(audio_array)

    # TODO: hot fix -> remove
    # # Early return if audio is empty or shorter or longar than 5s
    # if audio_array is None or len(audio_array) < 16000 or len(audio_array) > 160000:
    #     return None, None
    # Pad/Truncate audio to 5s if shorter
    # TODO: take sampling rate into account, e.g. 32kHz -> 160000 samples for 5s
    num_samples = 5 * sampling_rate  # 5 seconds worth of samples
    if len(audio_array) < num_samples:
        audio_array = np.pad(audio_array, (0, num_samples - len(audio_array)), mode='constant')
    elif len(audio_array) > num_samples:
        audio_array = audio_array[:num_samples]

    if embedding_type == 'perch_hoplite':
        pooled_embeddings, spatial_embeddings = embed_with_perch1(
            model, model_key, audio_array, device=device
        )
    elif embedding_type == 'perch_v2':
        pooled_embeddings, spatial_embeddings = embed_with_perch2(
            model, audio_array, device=device
        )
    else:
        raise Exception(
            f"Model family {embedding_type} is not supported. "
            f"Can not compute embeddings for model {model_key}."
        )

    return pooled_embeddings, spatial_embeddings


def embed_example(example, model, model_key, embedding_type, input_feature, sampling_rate, device='/CPU:0'):

    audio = example[input_feature]

    pooled_embeddings_key = model_key + "_" + input_feature + "_pooled_embeddings"
    spatial_embeddings_key = model_key + "_" + input_feature + "_spatial_embeddings"

    # Early return if audio is empty
    if audio is None or audio['array'] is None:
        example[pooled_embeddings_key] = None
        example[spatial_embeddings_key] = None
        return example

    pooled_embeddings, spatial_embeddings = compute_embedding(
        audio, model, model_key, embedding_type, sampling_rate, device=device
    )

    example[pooled_embeddings_key] = pooled_embeddings
    example[spatial_embeddings_key] = spatial_embeddings

    return example

# def embed_example(example, model, model_key, embedding_type, input_feature, sampling_rate, device='/CPU:0'):

#     audio = example[input_feature]

#     pooled_embeddings_key = model_key + "_" + input_feature + "_pooled_embeddings"
#     spatial_embeddings_key = model_key + "_" + input_feature + "_spatial_embeddings"

#     # Early return if audio is empty
#     if audio is None or audio['array'] is None:
#         example[pooled_embeddings_key] = None
#         example[spatial_embeddings_key] = None
#         return example

#     audio_array = resample_audio(audio['array'], audio['sampling_rate'], sampling_rate)

#     # Normalize
#     audio_array = normalize_audio_array(audio_array)

#     # Get embeddings
#     if embedding_type == 'perch_hoplite':
#         pooled_embeddings, spatial_embeddings = embed_with_perch1(model, model_key, audio_array, device=device)
#         # if spatial_embeddings is not None:
#         example[spatial_embeddings_key]= spatial_embeddings
#         example[pooled_embeddings_key] = pooled_embeddings
#     elif embedding_type == 'perch_v2':
#         example[pooled_embeddings_key], example[spatial_embeddings_key]= embed_with_perch2(model, audio_array, device=device)
#     # elif embedding_type == 'birdset':
#     #     example[embeddings_key] = embed_with_birdset(model, audio, device=device)
#     else:
#         raise Exception(f"Model family {embedding_type} is not supported. Can not compute embeddings for model {model_key}.")

#     return example

def embed_with_perch1(model, model_key, audio_array, device='/CPU:0'): #TODO: implement batched embed function perch hoplite provides batch_emebd()in zoo_interface.py

    # Early return if audio is empty
    if audio_array is None:
        return None, None

    num_samples = len(audio_array)

    with tf.device(device):
        if model_key == 'yamnet':
            scores, embeddings, log_mel_spectrogram = model(audio_array)
        elif model_key == 'vggish':
            embeddings = model(audio_array)
        else:
            outputs = model.embed(audio_array)
            embeddings = outputs.embeddings
            # print("embeddings:", embeddings.shape)
            # print("pooled embeddings:", outputs.pooled_embeddings(time_pooling='mean', channel_pooling='squeeze').shape)

        spatial_embeddings = None
    #     if embeddings.ndim > 1:
            
    #         dim = embeddings.shape
    #         num_dims = len(embeddings.shape)
            
    #         # Keep spatial embeddings if there are multiple segments, but not for models that already return pooled embeddings
    #         if num_dims > 1 and any(dim[x] != 1 for x in range(num_dims - 1)):
    #             spatial_embeddings = embeddings.copy()
            
    #         # Average pooling
    #         axes_to_reduce = list(range(num_dims - 1))
    #         pooled_embeddings = tf.reduce_mean(embeddings, axis=axes_to_reduce, keepdims=False)
    #         pooled_embeddings = tf.reshape(pooled_embeddings, [-1])  # Flatten to 1D
    #         # print(f'Embeddings include multiple segments. Calculate mean.')
    #     else:
    #         pooled_dim = pooled_embeddings.shape
    #         pooled_embeddings = tf.reshape(embeddings, [-1])

    #     if spatial_embeddings is not None:
    #         spatial_dim = spatial_embeddings.shape
    #         spatial_embeddings = tf.squeeze(spatial_embeddings)

    # return pooled_embeddings.numpy(), spatial_embeddings.numpy() if spatial_embeddings is not None else None
        #print(type(embeddings))
        embeddings =np.asarray(embeddings)
              
        if embeddings.ndim > 1:
            #embeddings = embeddings.numpy()  # convert once here
            
            dim = embeddings.shape
            num_dims = len(embeddings.shape)
            
            if num_dims > 1 and any(dim[x] != 1 for x in range(num_dims - 1)):
                if model_key == 'yamnet' or model_key == 'vggish' or model_key == 'beans_baseline':
                    spatial_embeddings = embeddings[:, None, :]  # (time, freq, embeddings)
                else:
                    spatial_embeddings = embeddings.copy()

            axes_to_reduce = tuple(range(num_dims - 1))
            pooled_embeddings = np.mean(embeddings, axis=axes_to_reduce)
            pooled_embeddings = pooled_embeddings.flatten()

        else:
            pooled_embeddings = embeddings.flatten()

        pooled_dim = pooled_embeddings.shape
            
        spatial_dim = None
        if spatial_embeddings is not None:
            spatial_dim = spatial_embeddings.shape
            spatial_embeddings = np.squeeze(spatial_embeddings)

    #print(f'Return pooled embeddings with dim {pooled_dim} and spatial embeddings with dim {spatial_dim} for model {model_key}.')
            
    return pooled_embeddings, spatial_embeddings if spatial_embeddings is not None else None

def embed_with_perch2(model, audio_array, device='/CPU:0'):

    # Early return if audio is empty
    if audio_array is None:
        return None, None

    with tf.device(device):
        infer_fn = model.signatures['serving_default']
        audio_batched = tf.constant(audio_array[np.newaxis, :], dtype=tf.float32)  # Shape: (1, 160000)
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
        if embedding_type == 'perch_hoplite':
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
        dataset = dataset.map(embedding_fn, cache_file_name=cache_file, num_proc=32, batched=True,
        batch_size=50)
        modified = True
    return dataset, modified

def add_embeddings_batchwise(model_key, dataset_split, input_feature, dataset, force_recompute=False, batch_size=100, device='/CPU:0'):
        
    embeddings_key = model_key + "_" + input_feature + "_pooled_embeddings"
    spatial_embeddings_key =  model_key + "_" + input_feature + "_spatial_embeddings"

    if embeddings_key in dataset.features and not force_recompute:
        print("Embedding with model", model_key, "for", input_feature, "has already been calculated, skipping.")
        return dataset, None
    
    # Get embedding type
    embedding_type = get_embedding_type(model_key)

    # Load model
    if embedding_type == 'perch_hoplite':
        model, sampling_rate = load_perch1_model(model_key)
    elif embedding_type == 'perch_v2':
        model, sampling_rate = load_perch2_model(model_key)
    elif embedding_type == 'birdset':
        model, sampling_rate = load_birdset_model(model_key)
    else:
        #print(f"Model family unknown. Can not load model {model_key}. Skipping.")
        return dataset, None
    
    # If embedding key is not in dataset compute
    print(f"Processing {dataset_split} split with {embeddings_key} in batches of {batch_size}...")

    dataset = dataset.cast_column(input_feature, Audio())
    
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
            batch_processed = batch_dataset.map(embedding_fn, keep_in_memory=True) #cache_file_name=cache_file)
            
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
    perch_hoplite_models = ['birdnet_V2.1', 'birdnet_V2.2', 'birdnet_V2.3', 'birdnet_V2.4', 'perch_8', 'surfperch', 'vggish', 'yamnet', 'humpback', 'multispecies_whale', 'beans_baseline', 'aves', 'birdaves']# added recently??? 'perch_v2', 'perch_v2_cpu'
    perch_v2_models = ['perch_v2', 'perch_v2_cpu']
    birdset_models = []

    # Get embedding type [perch_hoplite, perch_v2, birdset]
    if model_key in perch_hoplite_models:
        embedding_type = 'perch_hoplite'
    elif model_key in perch_v2_models:
        embedding_type = 'perch_v2'
    elif model_key in birdset_models:
        embedding_type = 'birdset'
    else:
        print(f"Could not get embedding type for model {model_key}. Embedding model is not supported, skipping!")
        return None
    
    print(model_key, "is a ", embedding_type, "model.")
    return embedding_type