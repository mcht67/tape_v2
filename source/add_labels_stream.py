from omegaconf import OmegaConf, DictConfig
from datasets import load_dataset, Sequence, Value
from functools import partial
from datetime import datetime
import os
import json
import numpy as np
import argparse
from pathlib import Path

from utils.general import overwrite_dataset

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

def main():

    print("Running add labels script...")

    # ===================
    # Configuration
    # ===================

    # Define arguments
    parser = argparse.ArgumentParser(
        description="Computes missing labels and updates dataset."
    )

    parser.add_argument("--huggingface_path", type=str)
    parser.add_argument("--dataset_config", type=str)
    parser.add_argument('--objectives', type=json.loads)
    parser.add_argument('--force_recompute', action='store_true')
    args = parser.parse_args()

    huggingface_path = args.huggingface_path
    dataset_config = args.dataset_config
    force_recompute = args.force_recompute
    objectives = args.objectives

    def load_objective_configs(
                        objective_keys: list[str],
                        objectives_dir: str = "conf/objectives",
                    ) -> dict[str, DictConfig]:
        """Load objectives configs for the given model_keys."""
        embeddings_path = Path(objectives_dir).resolve()
        return {
            key: OmegaConf.load(embeddings_path / f"{key}.yaml")
            for key in objective_keys
        }
    
    objective_configs = load_objective_configs(objectives, "conf/objectives")
    all_objectives = {objective for config_file in objective_configs.values() for objective in config_file.values() if 'label' in objective}
  
    # Load Dataset 
    dataset = load_dataset(huggingface_path, dataset_config)

    # Check if any label is not in dataset features
    for split in dataset.keys():
        missing_labels = {objective['label'] for objective in all_objectives if objective['label'] not in dataset[split].features}
        if missing_labels: #any(label not in dataset[split].features for label in all_labels):

            

            added_labels = []

            # Add event logits
            if 'event_logits' in missing_labels:
                print("Add event logits...")
                feature_name = 'event_logits'
                added_labels.append(feature_name)
                num_event_logits = cfg.labels.event_logits.num_logits

                add_event_logits_fn = partial(add_event_logits, num_event_logits=num_event_logits, feature_name=feature_name)
                event_logits_feature = Sequence(Value("float32"))
                
                for split in dataset.keys():
                    dataset[split] = dataset[split].map(add_event_logits_fn, keep_in_memory=False)
                    dataset[split] = dataset[split].cast_column(feature_name, event_logits_feature)
                print('Done!')

            # Add framewise polyphony labels
            if 'framewise_polyphony' in missing_labels:
                print('Add framewise polyphony labels...')
                feature_name = 'framewise_polyphony'
                num_frames = cfg.labels.framewise_polyphony.num_frames
                added_labels.append(feature_name)

                add_framewise_polyphony_fn = partial(add_framewise_polyphony, num_frames=num_frames, feature_name=feature_name)
                framewise_polyphony_feature = Sequence(Value("float32"))

                for split in dataset.keys():
                    dataset[split] = dataset[split].map(add_framewise_polyphony_fn, keep_in_memory=False)
                    dataset[split] = dataset[split].cast_column(feature_name, framewise_polyphony_feature)
                print('Done!')

    overwrite_dataset(dataset, dataset_path, store_backup=False)

    # Store metadata
    metadata = {
            "datetime": datetime.now().isoformat(),
            "dataset_path": dataset_path,
            "added_labels": added_labels
        }

    metadata_dir = os.path.dirname(added_labels_metadata_path)
    if metadata_dir:
        os.makedirs(metadata_dir, exist_ok=True)
    with open(added_labels_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

if __name__=="__main__":
    main()