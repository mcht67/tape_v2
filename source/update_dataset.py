from omegaconf import OmegaConf
import os
from datasets import Audio, load_dataset
from dotenv import load_dotenv

"""
Build a `datasets.Dataset` (split name: "test_5s") from BirdSet's raw
event-level `test` config.

Each output row = one 5-second, non-overlapping window of one soundscape
file, with:
  - decoded audio (fixed length, zero-padded if the trailing window is short)
  - per-event annotations as PARALLEL LISTS: start_time, end_time, low_freq,
    high_freq, ebird_code (all window-relative / window-clipped)
  - ebird_code_multilabel: sorted list of unique species in the window
    ([] for segments with no bird calls)
  - min_polyphony / max_polyphony: computed PER SEGMENT

Design choices (confirmed):
  - audio is decoded and stored inline (like BirdSet's own test_5s)
  - no-call segments -> ebird_code_multilabel = []  (not a special class id)
  - per-event fields are parallel lists, not nested dicts
  - polyphony is local to each 5s window, not the full original recording

Dependencies: datasets, soundfile, numpy, librosa (only needed if source
files aren't already 32 kHz).
"""

import math
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import soundfile as sf
from datasets import Dataset, DatasetDict, Features, Value, Sequence, Audio, ClassLabel

TARGET_SR = 32_000
SEGMENT_LEN = 5.0  # seconds
MIN_OVERLAP = 0.0  # seconds; minimum event/window overlap to count an event in a segment
PAD_LAST_SEGMENT = False # zero-pad trailing short segment to a fixed length


@dataclass
class Event:
    start_time: float
    end_time: float
    low_freq: float
    high_freq: float
    ebird_code: int


def group_events_by_file(hf_dataset):
    """One pass over the raw `test` split -> {filepath: [Event, ...]}."""
    groups = defaultdict(list)
    for row in hf_dataset:
        groups[row["filepath"]].append(
            Event(
                start_time=row["start_time"],
                end_time=row["end_time"],
                low_freq=row["low_freq"],
                high_freq=row["high_freq"],
                ebird_code=row["ebird_code"],
            )
        )
    return groups


def load_full_audio(filepath, target_sr=TARGET_SR):
    """Read + resample (if needed) + downmix to mono ONCE per file."""
    audio, sr = sf.read(filepath, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa  # imported lazily; only needed for mismatched sample rates
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr


def build_segments_for_file(filepath, events, target_sr=TARGET_SR,
                             segment_len=SEGMENT_LEN, min_overlap=MIN_OVERLAP,
                             pad_last=PAD_LAST_SEGMENT):
    """Yield one dict per 5s segment for a single soundscape file."""
    audio, sr = load_full_audio(filepath, target_sr)
    duration = len(audio) / sr
    n_segments = max(1, math.ceil(duration / segment_len))
    seg_n_samples = int(round(segment_len * sr))

    for idx in range(n_segments):
        seg_start = idx * segment_len
        seg_end = min(seg_start + segment_len, duration)

        frame_start = int(round(seg_start * sr))
        frame_end = int(round(seg_end * sr))
        seg_audio = audio[frame_start:frame_end]

        if pad_last and seg_audio.shape[0] < seg_n_samples:
            seg_audio = np.pad(seg_audio, (0, seg_n_samples - seg_audio.shape[0]))

        starts, ends, lows, highs, codes = [], [], [], [], []
        labels = set()
        for ev in events:
            overlap = min(ev.end_time, seg_end) - max(ev.start_time, seg_start)
            if overlap > min_overlap:
                starts.append(float(max(ev.start_time - seg_start, 0.0)))
                ends.append(float(min(ev.end_time - seg_start, seg_end - seg_start)))
                lows.append(int(ev.low_freq))
                highs.append(int(ev.high_freq))
                codes.append(int(ev.ebird_code))
                labels.add(int(ev.ebird_code))

        yield {
            "filepath": filepath,
            "segment_start": float(seg_start),
            "segment_end": float(seg_end),
            "audio": {"array": seg_audio, "sampling_rate": sr},
            "start_time": starts,
            "end_time": ends,
            "low_freq": lows,
            "high_freq": highs,
            # "ebird_code": codes,
            "ebird_code_multilabel": codes, #sorted(labels),
        }


def segment_generator(hf_dataset, **kwargs):
    """Generator over ALL segments of ALL files -> feeds Dataset.from_generator."""
    groups = group_events_by_file(hf_dataset)
    for filepath, events in groups.items():
        yield from build_segments_for_file(filepath, events, **kwargs)

def add_min_max_polyphony(example):
    # Minimum polyphony degree: get maximum number of overlapping events at any time
    # Minimum polyphony degree: get number of species active in soundscape
    # Maximum polyphony degree: get total number of events that occur in the soundscape
    start_times = np.atleast_1d(np.array(example['start_time']))
    end_times = np.atleast_1d(np.array(example['end_time']))
    species = np.atleast_1d(example['ebird_code_multilabel']).tolist()  # list of int values

    n_events = start_times.size

    # --- Maximum polyphony (upper bound) ---
    # Total number of events in the soundscape
    max_polyphony = n_events

    # --- Minimum polyphony (lower bound) ---
    # Candidate 1: max overlapping events at any point in time
    # Use an event-sweep approach: collect all start/end endpoints
    events = []
    for s, e in zip(start_times, end_times):
        events.append((s, +1))  # start: +1
        events.append((e, -1))  # end:   -1
    # Sort by time; on ties, process ends (-1) before starts (+1)
    events.sort(key=lambda x: (x[0], x[1]))

    max_overlap = 0
    current_overlap = 0
    for _, delta in events:
        current_overlap += delta
        max_overlap = max(max_overlap, current_overlap)

    # Candidate 2: number of unique species active in the soundscape
    n_unique_species = len(set(species))

    min_polyphony = max(max_overlap, n_unique_species)
    max_polyphony = max(max_polyphony, min_polyphony)

    example['min_polyphony'] = int(min_polyphony)
    example['max_polyphony'] = int(max_polyphony)
    return example


def build_test_5s_split(hf_dataset, features=None, **segment_kwargs) -> Dataset:
    """
    Build the full `test_5s` Dataset from BirdSet's raw `test` split.

    Uses Dataset.from_generator so segments are streamed/written incrementally
    rather than held as one giant Python list in memory (important once you
    scale beyond a single test subset like HSN).
    """
    ds = Dataset.from_generator(
        lambda: segment_generator(hf_dataset, **segment_kwargs),
        features=features,
    )
    ds = ds.map(add_min_max_polyphony)
    return ds


def main():

    print("Load config")

    #######################
    # Config
    #######################

    #  Load the parameters from the config file
    cfg = OmegaConf.load("params.yaml")

    # dataset_source = cfg.dataset.source
    # dataset_subset = cfg.dataset.subset

    soundscape_test_data_path = cfg.paths.soundscape_test_data

     # Load environment variables from .env file
    load_dotenv('local.env')

    # Huggingface login
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

    ######################
    # Load dataset
    ######################

    # if dataset_subset == "XCM" or dataset_subset == "XCL":
    #     print("No soundscape data exists for this dataset subset. Skipping soundscape test data preparation.")
    #     # Create empty test directory for consistent dvc tracking
    #     os.makedirs(soundscape_test_data_path, exist_ok=True)
    #     return

    # Load soundscape data
    for dataset_subset in ['HSN', 'PER', 'NES', 'UHH', 'HSN', 'POW', 'SSW', 'SNE']:

        ###########################
        # Update soudnscape data
        ###########################

        print("Load soundscape data from hugging face hub.") 
        dataset = load_dataset("DBD-research-group/BirdSet", dataset_subset, trust_remote_code=True, token=huggingface_token)
        # train_dataset = load_dataset("DBD-research-group/BirdSet", train_subset_name, split="test", trust_remote_code=True, token=huggingface_token)
        ebird_code_class_labels = dataset['test'].features['ebird_code_multilabel'].feature
        print(ebird_code_class_labels)
        soundscape_dataset = dataset['test']

        FEATURES = Features({
            "filepath": Value("string"),
            "segment_start": Value("float32"),
            "segment_end": Value("float32"),
            "audio": Audio(sampling_rate=TARGET_SR, mono=True, decode=True),
            "start_time": Sequence(Value("float32")),
            "end_time": Sequence(Value("float32")),
            "low_freq": Sequence(Value("int64")),
            "high_freq": Sequence(Value("int64")),
            # "ebird_code_multilabel": Sequence(Value("int64")),
            "ebird_code_multilabel": Sequence(ebird_code_class_labels) #Sequence(ClassLabel(names=ebird_code_labels)),
        })


        test_5s = build_test_5s_split(soundscape_dataset, features=FEATURES, target_sr=TARGET_SR, segment_len=SEGMENT_LEN, min_overlap=MIN_OVERLAP, pad_last=PAD_LAST_SEGMENT)

        dataset_dict = DatasetDict({"test_5s": test_5s})
        print(dataset_dict)
        print(dataset_dict["test_5s"][0])

        # Push to Huggingface Hub
        config_name = dataset_subset + "_soundscape_test"
        data_dir = f"{dataset_subset}/{config_name}"
        dataset_dict.push_to_hub("mcht67/PolyBirdMix", config_name, commit_message=f"Update {config_name} dataset", data_dir=data_dir, private=True, token=os.environ.get("HUGGINGFACE_TOKEN"))

        # ###########################
        # # Update polyphonic
        # ###########################

        # poly_config = dataset_subset + "_polyphonic"
    
        # print("Loading PolyBirdMix dataset for config:", poly_config)
        # polybird_ds = load_dataset("mcht67/PolyBirdMix", poly_config)
        # ebird_code_class_labels = dataset['train'].features['ebird_code_multilabel'].feature
        # for split in polybird_ds:
        #     polybird_ds[split] = polybird_ds[split].cast_column("birdset_id_multilabel", Sequence(ebird_code_class_labels))
        #     print("Class labels for config", poly_config, "and split", split, ":", polybird_ds[split].features['birdset_id_multilabel'])
        #     polybird_ds[split] = polybird_ds[split].rename_column("birdset_id_multilabel", "ebird_code_multilabel")


if __name__ == "__main__":
    main()