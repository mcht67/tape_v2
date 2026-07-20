from datasets import Sequence, Value
from functools import partial
import numpy as np
import time
import random
from datasets import load_dataset
import traceback
from collections import Counter
import os
import shutil

def overwrite_dataset(dataset, dataset_path, store_backup=True):
    # Save to temporary location
    temp_path = dataset_path + "_temp"
    os.makedirs(temp_path, exist_ok=True)
    dataset.save_to_disk(temp_path)

    # Move old data to backup
    backup_path = dataset_path + "_backup"
    os.makedirs(backup_path, exist_ok=True)
    if os.path.exists(dataset_path):
        shutil.move(dataset_path, backup_path)

    # Move temp data into place
    shutil.move(temp_path, dataset_path)

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

    # Early exit if no events 
    if not events:
        return event_logits

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
        if 'sources_time_freq_bounds' in example and example['sources_time_freq_bounds'] is not None:
            for events in example['sources_time_freq_bounds']:
                all_events.extend(events)
            segment_duration_s = example['segment_duration_s'] #num_samples_to_duration_s(segment_sum_samples, sampling_rate)
            event_logits = build_event_logits(all_events, segment_duration_s, num_event_logits)
        else:
            event_logits = np.zeros(num_event_logits, dtype=np.float32)
        example[feature_name] = event_logits
        return example

def add_framewise_polyphony(example, num_frames, feature_name):
        if 'sources_time_freq_bounds' in example and example['sources_time_freq_bounds'] is not None:
            time_freq_bounds_per_raw_file = example['sources_time_freq_bounds']
            segment_durations_s = example["segment_duration_s"]
            framewise_polyphony_array = build_framewise_polyphony(time_freq_bounds_per_raw_file, segment_durations_s, num_frames)
        else:
            framewise_polyphony_array = np.zeros(num_frames, dtype=np.int32)
        example[feature_name] = framewise_polyphony_array
        return example

def add_species_polyphony(example, birdset_id2label, feature_name):
    counts = None

    # TODO: remove once we have birdset_id_multilabel for all datasets
    if 'birdset_code_multilabel' in example and example['birdset_code_multilabel'] is not None:
        counts = Counter(example['birdset_code_multilabel'])
    elif 'birdset_id_multilabel' in example and example['birdset_id_multilabel'] is not None:
        counts = Counter(example['birdset_id_multilabel'])
    elif 'ebird_code_multilabel' in example and example['ebird_code_multilabel'] is not None:
        counts = Counter(example['ebird_code_multilabel'])
    
    if counts is not None:
        # total_polyphony = sum(counts.values())
        labels = [counts.get(int(birdset_id), 0) for birdset_id in birdset_id2label.keys()]
        # labels[len(birdset_id2label)] = total_polyphony
    else:
        labels = [0] * (len(birdset_id2label) + 1)
        # total_polyphony = 0
        # labels[len(birdset_id2label)] = total_polyphony
    example[feature_name] = labels
    return example

# def get_birdset_id2label(dataset):

#     # Get list of birdset ids
#     dataset_split = next(iter(dataset.keys()))
#     info = dataset[dataset_split].info
#     metadata = getattr(info, "metadata", None)

#     if metadata and "birdset_id2label" in metadata:
#         return metadata["birdset_id2label"]
#     else:
#         unique_birdset_ids = set()

#         def collect_values(batch):
#             if 'birdset_id_multilabel' in batch and batch['birdset_id_multilabel'] is not None:
#                 for birdset_id in batch["birdset_id_multilabel"]:
#                     unique_birdset_ids.update(birdset_id)
#             # TODO: remove once we have birdset_id_multilabel for all datasets
#             elif 'birdset_code_multilabel' in batch and batch['birdset_code_multilabel'] is not None:
#                 for birdset_id in batch["birdset_code_multilabel"]:
#                     unique_birdset_ids.update(birdset_id)
#             else:
#                 raise ValueError("No birdset_id_multilabel or birdset_code_multilabel found in example")
#             return batch  # return unchanged

#         for split in dataset.values():
#             split.map(collect_values, batched=True, batch_size=100, num_proc=1, keep_in_memory=False)

#         print(f"Unique birdset IDs: {sorted(unique_birdset_ids)}")
#         return {birdset_id: None for birdset_id in sorted(unique_birdset_ids)}
    
def get_birdset_id2label(subset, dataset=None):
    if dataset is not None:
        if 'ebird_code_multilabel' in dataset[next(iter(dataset.keys()))].features:
                return get_birdset_id2label_from_dataset(dataset)

    # TODO remove
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv('local.env')
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
    from datasets import load_dataset
    if not subset=='XCM' and not subset=='XCL':
        ds = load_dataset("mcht67/PolyBirdMix", f'{subset}_soundscape_test', split='test_5s', token=huggingface_token)
    elif subset=='XCM' or subset=='XCL':
        ds = load_dataset("DBD-research-group/BirdSet", f'{subset}', split='train', token=huggingface_token)
    else:
        if not dataset:
            raise ValueError("Dataset must be provided for unknown subset")
        unique_birdset_ids = set()
        for split in dataset.values():
            for example in split:
                if 'birdset_id_multilabel' in example and example['birdset_id_multilabel'] is not None:
                    unique_birdset_ids.update(example['birdset_id_multilabel'])
                elif 'birdset_code_multilabel' in example and example['birdset_code_multilabel'] is not None:
                    unique_birdset_ids.update(example['birdset_code_multilabel'])
                else:
                    raise ValueError("No birdset_id_multilabel or birdset_code_multilabel found in example")
        print(f"Unique birdset IDs: {sorted(unique_birdset_ids)}")
        return {birdset_id: None for birdset_id in sorted(unique_birdset_ids)}
    
    ebird_code_class_labels = ds.features['ebird_code_multilabel'].feature.names
    print(f"Found {len(ebird_code_class_labels)} species in the dataset: {ebird_code_class_labels}")
    birdset_id2label = {birdset_id: label for birdset_id, label in enumerate(ebird_code_class_labels)}
    return birdset_id2label

from datasets import Dataset, DatasetDict

def get_birdset_id2label_from_dataset(dataset):
    if isinstance(dataset, DatasetDict):
        first_split = next(iter(dataset.keys()))
        dataset = dataset[first_split]
    elif not isinstance(dataset, Dataset):
        raise TypeError(
            f"Expected a Hugging Face Dataset or DatasetDict, got {type(dataset)}"
        )

    ebird_code_class_labels = dataset.features['ebird_code_multilabel'].feature.names
    print(f"Found {len(ebird_code_class_labels)} species in the dataset: {ebird_code_class_labels}")
    birdset_id2label = {birdset_id: label for birdset_id, label in enumerate(ebird_code_class_labels)}
    return birdset_id2label



    # dataset_split = next(iter(dataset.keys()))
    # if 'ebird_code_multilabel' in dataset[dataset_split].features:
    #     ebird_class_labels = dataset[dataset_split].features['ebird_code_multilabel'].feature.names
    #     birdset_id2label = {birdset_id: label for birdset_id, label in enumerate(ebird_class_labels)}
    #     return birdset_id2label
    # else:
    #     unique_birdset_ids = set()
    #     for split in dataset.values():
    #         for example in split:
    #             if 'birdset_id_multilabel' in example and example['birdset_id_multilabel'] is not None:
    #                 unique_birdset_ids.update(example['birdset_id_multilabel'])
    #             elif 'birdset_code_multilabel' in example and example['birdset_code_multilabel'] is not None:
    #                 unique_birdset_ids.update(example['birdset_code_multilabel'])
    #             else:
    #                 raise ValueError("No birdset_id_multilabel or birdset_code_multilabel found in example")
    #     print(f"Unique birdset IDs: {sorted(unique_birdset_ids)}")
    #     return {birdset_id: None for birdset_id in sorted(unique_birdset_ids)}

def add_labels(dataset, labels, birdset_id2label=None, time_dim=None, freq_dim=None):
    added_labels = []

    # Segment-wise polyphony based label
    if 'polyphony_reg' in labels:
        print('Add polyphony degree reg labels...')
        feature_name = 'polyphony_reg'

        for split in dataset.keys():
            dataset[split] = dataset[split].map(lambda example: {feature_name: float(example["polyphony"])},keep_in_memory=False)
        
        added_labels.append(feature_name)
        print('Done!')

    if 'polyphony_class' in labels:
        print('Add polyphony degree class labels...')
        feature_name = 'polyphony_class'

        for split in dataset.keys():
            dataset[split] = dataset[split].map(lambda example: {feature_name: int(example["polyphony"])},keep_in_memory=False)
        
        added_labels.append(feature_name)
        print('Done!')

    # Species specific polyphony
    if 'species_polyphony_reg' in labels or 'species_polyphony_class' in labels:
        feature_names = [x for x in ['species_polyphony_reg', 'species_polyphony_class'] if x in labels]

        if not birdset_id2label:
            birdset_id2label = get_birdset_id2label('', dataset=dataset)

        for feature_name in feature_names:
            # print(f'Add {feature_name} labels...')
            # add_species_polyphony_fn = partial(add_species_polyphony, birdset_id2label=birdset_id2label, feature_name=feature_name)
            # species_polyphony_feature = Sequence(Value("int32"))

            # for split in dataset.keys():
            #     dataset[split] = dataset[split].map(add_species_polyphony_fn, keep_in_memory=False)
            #     dataset[split] = dataset[split].cast_column(feature_name, species_polyphony_feature)

            for split in dataset.keys():
                dataset[split] = dataset[split].map(lambda example: {feature_name: int(example["species_polyphony"])},keep_in_memory=False)

            added_labels.append(feature_name)
            print('Done!')

    # Time dimension based labels
    if 'framewise_polyphony_reg' in labels or 'framewise_polyphony_class' in labels or 'event_logits' in labels:
        if not time_dim:
            raise ValueError("Time dimension is required for framewise polyphony and event logits")
        # Event logits
        if 'event_logits' in labels:
            print("Add event logits..."
                  )
            feature_name = 'event_logits'
            num_event_logits = time_dim

            add_event_logits_fn = partial(add_event_logits, num_event_logits=num_event_logits, feature_name=feature_name)
            event_logits_feature = Sequence(Value("int32"))
            
            for split in dataset.keys():
                dataset[split] = dataset[split].map(add_event_logits_fn, keep_in_memory=False)
                dataset[split] = dataset[split].cast_column(feature_name, event_logits_feature)
            added_labels.append(feature_name)
            print('Done!')

        # Framewise polyphony
        if 'framewise_polyphony_reg' in labels or 'framewise_polyphony_class' in labels:
            feature_names = [x for x in ['framewise_polyphony_reg', 'framewise_polyphony_class'] if x in labels]
            num_frames = time_dim

            for feature_name in feature_names:
                print(f'Add {feature_name} labels...')
                add_framewise_polyphony_fn = partial(add_framewise_polyphony, num_frames=num_frames, feature_name=feature_name)
                framewise_polyphony_feature = Sequence(Value("int32"))

                for split in dataset.keys():
                    dataset[split] = dataset[split].map(add_framewise_polyphony_fn, keep_in_memory=False)
                    dataset[split] = dataset[split].cast_column(feature_name, framewise_polyphony_feature)

                added_labels.append(feature_name)
                print('Done!')

    return dataset, added_labels

def load_dataset_with_retry(path, config, token=None, retries=5, download_mode='reuse_dataset_if_exists'):
    for attempt in range(retries):
        try:
            print("Loading dataset with download_mode:", download_mode)
            return load_dataset(path, config, token=token, download_mode=download_mode)
        except FileNotFoundError as e:
            if '.incomplete' in str(e) or 'dataset_info.json' in str(e):
                print(f"Incomplete cache detected, forcing re-download (attempt {attempt + 1})")
                download_mode = 'force_redownload'
            elif "fchmod" in traceback.format_exc() and attempt < retries - 1:
                wait = random.uniform(1, 5) * (attempt + 1)
                print(f"Cache lock race, retrying in {wait:.1f}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise

def get_data_dir(dataset_config, subset=None):
    if subset is None:
        subset = dataset_config.split('_')[0]
    return f"{subset}/{dataset_config}"

def get_local_data_dir(dataset_config=None, subset=None, data_dir=None):
    if data_dir is not None:
        return os.path.join("data", data_dir)
    if subset is None:
        subset = dataset_config.split('_')[0]
    return f"data/{subset}/{dataset_config}"

# def add_min_max_polyphony(example):
#     start_times = np.atleast_1d(np.array(example['start_time']))
#     end_times = np.atleast_1d(np.array(example['end_time']))
#     species = np.atleast_1d(example['ebird_code_multilabel']).tolist()  # list of int values

#     n_events = start_times.size

#     # --- Maximum polyphony (upper bound) ---
#     # Total number of events in the soundscape
#     max_polyphony = n_events

#     # --- Minimum polyphony (lower bound) ---
#     # Candidate 1: max overlapping events at any point in time
#     # Use an event-sweep approach: collect all start/end endpoints
#     events = []
#     for s, e in zip(start_times, end_times):
#         events.append((s, +1))  # start: +1
#         events.append((e, -1))  # end:   -1
#     # Sort by time; on ties, process ends (-1) before starts (+1)
#     events.sort(key=lambda x: (x[0], x[1]))

#     max_overlap = 0
#     current_overlap = 0
#     for _, delta in events:
#         current_overlap += delta
#         max_overlap = max(max_overlap, current_overlap)

#     # Candidate 2: number of unique species active in the soundscape
#     n_unique_species = len(set(species))

#     min_polyphony = max(max_overlap, n_unique_species)
#     max_polyphony = max(max_polyphony, min_polyphony)

#     example['min_polyphony'] = int(min_polyphony)
#     example['max_polyphony'] = int(max_polyphony)
    # return example
    
def _max_overlap(starts, ends):
    """Event-sweep max simultaneous overlap for a set of (start, end) intervals."""
    if len(starts) == 0:
        return 0
    events = []
    for s, e in zip(starts, ends):
        events.append((s, +1))  # start: +1
        events.append((e, -1))  # end:   -1
    events.sort(key=lambda x: (x[0], x[1]))  # ties: ends before starts

    max_overlap = 0
    current_overlap = 0
    for _, delta in events:
        current_overlap += delta
        max_overlap = max(max_overlap, current_overlap)
    return max_overlap


def add_min_max_polyphony(example, num_species=None):
    """
    num_species: size of the species vocabulary (e.g. len(birdset_id2label)).
        Required to produce fixed-length min/max_species_polyphony vectors
        aligned with species indices. If None, returns dicts keyed by
        species id instead (sparse — only species present in this example).
    """
    start_times = np.atleast_1d(np.array(example['start_time']))
    end_times = np.atleast_1d(np.array(example['end_time']))
    species = np.atleast_1d(example['ebird_code_multilabel']).tolist()  # list of int values
    species_arr = np.array(species)

    n_events = start_times.size

    # --- Total polyphony (unchanged) ---
    max_polyphony = n_events
    max_overlap = _max_overlap(start_times, end_times)
    n_unique_species = len(set(species))
    min_polyphony = max(max_overlap, n_unique_species)
    max_polyphony = max(max_polyphony, min_polyphony)

    example['min_polyphony'] = int(min_polyphony)
    example['max_polyphony'] = int(max_polyphony)

    # --- Per-species polyphony ---
    # min: max overlapping calls of that species at any instant (event sweep)
    # max: total number of calls of that species in the segment
    if num_species is not None:
        min_species_polyphony = np.zeros(num_species, dtype=int)
        max_species_polyphony = np.zeros(num_species, dtype=int)
    else:
        min_species_polyphony = {}
        max_species_polyphony = {}

    for sp in set(species):
        mask = species_arr == sp
        sp_starts = start_times[mask]
        sp_ends = end_times[mask]

        min_sp = _max_overlap(sp_starts, sp_ends)
        max_sp = int(mask.sum())

        min_species_polyphony[sp] = min_sp
        max_species_polyphony[sp] = max_sp

    example['min_species_polyphony'] = (
        min_species_polyphony if num_species is not None else min_species_polyphony
    )
    example['max_species_polyphony'] = (
        max_species_polyphony if num_species is not None else max_species_polyphony
    )

    return example