from datasets import Sequence, Value
from functools import partial
import numpy as np
import time
import random
from datasets import load_dataset

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


def add_event_logits(example, num_event_logits, feature_name):
        all_events = []
        for events in example['sources_time_freq_bounds']:
            all_events.extend(events)
        segment_duration_s = example['segment_duration_s'] #num_samples_to_duration_s(segment_sum_samples, sampling_rate)
        event_logits = build_event_logits(all_events, segment_duration_s, num_event_logits)
        example[feature_name] = event_logits
        return example

def add_framewise_polyphony(example, num_frames, feature_name):
        time_freq_bounds_per_raw_file = example['sources_time_freq_bounds']
        segment_durations_s = example["segment_duration_s"]
        framewise_polyphony_array = build_framewise_polyphony(time_freq_bounds_per_raw_file, segment_durations_s, num_frames)
        example[feature_name] = framewise_polyphony_array
        return example

def add_labels(dataset, labels, time_dim=None, freq_dim=None):
    added_labels = []
    # Time dimension based labels
    if time_dim:
        # Event logits
        if 'event_logits' in labels:
            print("Add event logits...")
            feature_name = 'event_logits'
            added_labels.append(feature_name)
            num_event_logits = time_dim

            add_event_logits_fn = partial(add_event_logits, num_event_logits=num_event_logits, feature_name=feature_name)
            event_logits_feature = Sequence(Value("float32"))
            
            for split in dataset.keys():
                dataset[split] = dataset[split].map(add_event_logits_fn, keep_in_memory=False)
                dataset[split] = dataset[split].cast_column(feature_name, event_logits_feature)
            print('Done!')

        # Framewise polyphony
        if 'framewise_polyphony' in labels:
            print('Add framewise polyphony labels...')
            feature_name = 'framewise_polyphony'
            num_frames = time_dim
            added_labels.append(feature_name)

            add_framewise_polyphony_fn = partial(add_framewise_polyphony, num_frames=num_frames, feature_name=feature_name)
            framewise_polyphony_feature = Sequence(Value("float32"))

            for split in dataset.keys():
                dataset[split] = dataset[split].map(add_framewise_polyphony_fn, keep_in_memory=False)
                dataset[split] = dataset[split].cast_column(feature_name, framewise_polyphony_feature)
            print('Done!')

    return dataset, added_labels

def load_dataset_with_retry(path, config, token=None, retries=5, download_mode='reuse_cache_if_exists'):
    for attempt in range(retries):
        try:
            return load_dataset(path, config, token=token, download_mode=download_mode)
        except FileNotFoundError as e:
            if "fchmod" in str(e) and attempt < retries - 1:
                wait = random.uniform(1, 5) * (attempt + 1)
                print(f"Cache lock race, retrying in {wait:.1f}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise