import random
import psutil
import os
from functools import wraps
import numpy as np
import tensorflow as tf
from datasets import concatenate_datasets
import shutil
import json
from datetime import datetime

def with_random_state(func):
    """
    Decorator that allows a function to accept random_state parameter.
    The function can accept either a seed (int) or a state tuple.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract random_state from kwargs
        random_state = kwargs.pop('random_state', None)
        
        if random_state is None:
            # No state provided, call function normally
            return func(*args, **kwargs)
        
        # Save current state
        current_state = random.getstate()
        
        try:
            # Set the provided state
            if isinstance(random_state, int):
                # It's a seed
                random.seed(random_state)
            else:
                # It's a state tuple
                random.setstate(random_state)
            
            # Call the original function
            return func(*args, **kwargs)
        
        finally:
            # Restore original state
            random.setstate(current_state)
    
    return wrapper

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

def reshape_tensor_data(example, column_name, target_shape, pooling_strategy="mean", pad_value=0.0, suffix="_reshaped"):
    """
    General function to reshape any tensor data to a target shape.
    Handles both features and labels with flexible target shapes.
    Adds reshaped tensor to dataset with specified suffix added to the column name.
    
    Args:
        example: Dataset example (dict)
        column_name: Name of the column to reshape
        target_shape: Target shape as tuple, e.g. (1280,) for 1D or (64, 20) for 2D
        pooling_strategy: How to handle extra dimensions - 'mean', 'max', 'first', 'last', 'flatten'
        pad_value: Value to use for padding
    
    Returns:
        Updated example with new column_name + suffix 
    """
    # Get the raw data (could be any shape)
    data = example[column_name]
    
    # Convert to numpy for shape manipulation
    data = np.array(data, dtype=np.float32)
    original_shape = data.shape
    
    # Calculate target size
    target_size = np.prod(target_shape)
    
    # Step 1: Handle initial shape normalization
    if data.ndim == 0:
        # Scalar - convert to 1D array
        data = np.array([data])
    
    # Step 2: Reduce to appropriate dimensionality using pooling strategy
    if pooling_strategy == "flatten":
        # Simply flatten everything
        data = data.flatten()
    else:
        # Apply pooling strategies for multi-dimensional data
        while data.ndim > len(target_shape):
            if pooling_strategy == "mean":
                data = np.mean(data, axis=0)
            elif pooling_strategy == "max":
                data = np.max(data, axis=0)
            elif pooling_strategy == "first":
                data = data[0]
            elif pooling_strategy == "last":
                data = data[-1]
            elif pooling_strategy == "sum":
                data = np.sum(data, axis=0)
            else:
                # Default to mean
                data = np.mean(data, axis=0)
        
        # Handle case where we need to add dimensions
        while data.ndim < len(target_shape):
            data = np.expand_dims(data, axis=-1)
        
        # If dimensions match but shapes don't, flatten and reshape
        if data.ndim == len(target_shape) and data.shape != target_shape:
            data = data.flatten()
    
    # Step 3: Resize to target total size
    current_size = data.size
    
    if current_size < target_size:
        # Pad with specified value
        pad_size = target_size - current_size
        data = np.concatenate([data.flatten(), np.full(pad_size, pad_value)])
    elif current_size > target_size:
        # Truncate (could also use PCA, random sampling, etc.)
        data = data.flatten()[:target_size]
    else:
        # Perfect size, just flatten
        data = data.flatten()
    
    # Step 4: Reshape to target shape
    data = data.reshape(target_shape)
    
    # Update the example
    example[column_name + "_reshaped"] = data.tolist()
    
    # # Optional: Store metadata about the transformation
    # example[column_name + "_original_shape"] = list(original_shape)
    # example[column_name + "_target_shape"] = list(target_shape)
    
    return example

def reshape_tensor_data_tf(example, column_name, target_shape, pooling_strategy="mean", pad_value=0.0, suffix="_reshaped"):
    """
    TensorFlow-compatible function to reshape embeddings or other tensors in a dataset.

    Args:
        example (dict): One dataset example, with tensors.
        column_name (str): Key of the tensor to reshape.
        target_shape (tuple): Desired output shape (e.g., (1280,))
        pooling_strategy (str): "mean", "max", "first", "last", "sum", "flatten"
        pad_value (float): Value for padding if needed.
        suffix (str): Suffix for new key name.

    Returns:
        dict: Updated example with new reshaped tensor under key: column_name + suffix
    """
    data = example[column_name]  # Tensor of shape (segments, 1, 1280) or similar

    # Ensure data is float32
    data = tf.cast(data, tf.float32)

    # Step 1: Remove unnecessary singleton dimensions
    data = tf.squeeze(data, axis=1) if tf.rank(data) == 3 and data.shape[1] == 1 else data  # shape: (segments, 1280)

    # Step 2: Pooling to reduce dimensionality
    if pooling_strategy == "mean":
        pooled = tf.reduce_mean(data, axis=0)
    elif pooling_strategy == "max":
        pooled = tf.reduce_max(data, axis=0)
    elif pooling_strategy == "sum":
        pooled = tf.reduce_sum(data, axis=0)
    elif pooling_strategy == "first":
        pooled = data[0]
    elif pooling_strategy == "last":
        pooled = data[-1]
    elif pooling_strategy == "flatten":
        pooled = tf.reshape(data, [-1])  # Just flatten everything
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

    # Step 3: Reshape or pad/truncate to match target shape
    flat = tf.reshape(pooled, [-1])
    current_size = tf.shape(flat)[0]
    target_size = tf.reduce_prod(target_shape)

    def pad():
        pad_len = target_size - current_size
        return tf.concat([flat, tf.fill([pad_len], pad_value)], axis=0)

    def truncate():
        return flat[:target_size]

    output = tf.cond(current_size < target_size, pad, truncate)
    output = tf.reshape(output, target_shape)

    # Update dictionary
    example[column_name + suffix] = output
    return example

def process_in_batches(dataset, process_fn, cache_dir, prefix="", batch_size=100):
    """
    Generic batch processor for datasets.
    
    Args:
        dataset: Dataset to process.
        process_fn: Function to apply to each batch.
        cache_dir: Directory to store batch caches.
        prefix: Optional prefix for cache filenames.
        batch_size: Number of samples per batch.
    
    Returns:
        Concatenated processed dataset.
    """
    processed_datasets = []
    total_samples = len(dataset)
    
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        print(f"Processing batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}...")
        
        batch_dataset = dataset.select(range(i, end_idx))
        cache_file = os.path.join(cache_dir, f"{prefix}_batch_{i}_{end_idx}_cache.arrow")
        batch_processed = batch_dataset.map(process_fn, cache_file_name=cache_file)
        
        processed_datasets.append(batch_processed)
    
    print(f"Concatenating {len(processed_datasets)} batches...")
    return concatenate_datasets(processed_datasets)

# def overwrite_dataset(dataset, dataset_path, store_backup=True):
    
#     # Save to temporary location
#     temp_path = os.path.join(dataset_path, "_temp")
#     os.makedirs(temp_path, exist_ok=True)
    
#     try:
#         dataset.save_to_disk(temp_path)
        
#         # Load metadata json if it exists
#         metadata = None
#         metadata_path = os.path.join(dataset_path, 'dataset_metadata.json')
#         if os.path.exists(metadata_path):
#             with open(metadata_path, 'r') as f:
#                 metadata = json.load(f)
        
#         # Move old data to backup
#         if store_backup:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             backup_path = f"{dataset_path}_backup_{timestamp}"
#             if os.path.exists(dataset_path):
#                 shutil.move(dataset_path, backup_path)
#         # else:
#         #     # Remove old dataset without backup
#         #     if os.path.exists(dataset_path):
#         #         shutil.rmtree(dataset_path)
        
#         # Move temp data into place
#         shutil.move(temp_path, dataset_path)
        
#         # Restore metadata json if it existed
#         if metadata is not None:
#             with open(metadata_path, 'w') as f:
#                 json.dump(metadata, f, indent=2)
        
#         # Optionally remove backup
#         if not store_backup:
#             shutil.rmtree(backup_path)
                
#     except Exception as e:
#         # Cleanup temp on failure
#         if os.path.exists(temp_path):
#             shutil.rmtree(temp_path)
#         raise e

def overwrite_dataset(dataset, dataset_path, metadata_path=None, store_backup=True):
    # Save to temporary location
    temp_path = f"{dataset_path}_temp"
    os.makedirs(temp_path, exist_ok=True)
    dataset.save_to_disk(temp_path)

    # Load metadata json
    metadata = None
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                content = f.read()
                if content.strip():  # Check if file has content
                    metadata = json.loads(content)
                else:
                    print(f"Warning: {metadata_path} is empty, skipping metadata preservation")
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse {metadata_path}: {e}")
            print("Skipping metadata preservation")

    # Move old data to backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f'{dataset_path}_backup_{timestamp}'
    os.makedirs(backup_path, exist_ok=True)
    if os.path.exists(dataset_path):
        shutil.move(dataset_path, backup_path)

    # Move temp data into place
    shutil.move(temp_path, dataset_path)
    
    # Dump metadata json
    if metadata:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    # Optionally remove backup
    if not store_backup:
        shutil.rmtree(backup_path)

def build_event_logits(
    events,
    segment_duration_sec,
    num_event_logits,
):
    """
    Build binary temporal event labels.

    Parameters
    ----------
    events : list of tuples
        Each tuple is (start_sec, end_sec, low_freq_hz, high_freq_hz)
    segment_duration_sec : float
        Total duration of the audio segment in seconds
    num_event_logits : int
        Number of temporal bins (e.g. 16)

    Returns
    -------
    event_logits : np.ndarray, shape (num_event_logits,)
        Binary array indicating event presence per time bin
    """

    event_logits = np.zeros(num_event_logits, dtype=np.float32)

    # Length of one time bin in seconds
    bin_size = segment_duration_sec / num_event_logits

    for i in range(num_event_logits):
        bin_start = i * bin_size
        bin_end = (i + 1) * bin_size

        # Check if ANY event overlaps this time bin
        for event in events:
            event_start, event_end, _, _ = event

            # Overlap condition
            if event_start < bin_end and event_end > bin_start:
                event_logits[i] = 1.0
                break

    return event_logits

def build_framewise_polyphony(
    time_freq_bounds_per_individual,
    segment_duration_s,
    num_frames,
):
    """
    Vectorized frame-wise polyphony computation.

    Args
    ----
    time_freq_bounds_per_individual : list[list[tuple]]
        Outer list = individuals
        Inner list = events of that individual
        Event tuple = (start_time, end_time, freq_low, freq_high)

    segment_duration_s : float
        Segment duration in seconds

    num_frames : int
        Number of temporal bins (e.g. 16)

    Returns
    -------
    frame_polyphony : np.ndarray, shape (num_frames,)
    """

    # Frame boundaries
    frame_edges = np.linspace(
        0.0, segment_duration_s, num_frames + 1
    )
    frame_start = frame_edges[:-1]   # (T,)
    frame_end = frame_edges[1:]      # (T,)

    frame_polyphony = np.zeros(num_frames, dtype=np.int32)

    for individual_events in time_freq_bounds_per_individual:
        if len(individual_events) == 0:
            continue

        events = np.asarray(individual_events)

        event_start = events[:, 0][:, None]  # (E, 1)
        event_end = events[:, 1][:, None]    # (E, 1)

        # Overlap: (E, T)
        overlaps = (event_end > frame_start) & (event_start < frame_end)

        # Does this individual overlap each frame? (T,)
        individual_active = overlaps.any(axis=0)

        frame_polyphony += individual_active.astype(np.int32)

    return frame_polyphony



