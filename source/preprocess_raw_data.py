from datasets import load_dataset, Audio
import datasets
from preprocess_raw_data.source_separation import BirdMixitSeparator
import numpy as np
import tempfile

with tempfile.TemporaryDirectory() as temp_cache_dir:
    # Set HuggingFace cache to this temporary directory
    datasets.config.HF_DATASETS_CACHE = temp_cache_dir

    # Load BirdSet
    ds = load_dataset("DBD-research-group/BirdSet", "HSN_xc")
    ds = ds.cast_column("audio", Audio(sampling_rate=32000))

    separator = BirdMixitSeparator(
        model_dir="externals/bird_mixit/bird_mixit_model_checkpoints/output_sources4",
        checkpoint="externals/bird_mixit/bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090",
        num_sources=4,
        model_sr=22050,
    )

    def add_birdmixit_source(example):
        arr = example["audio"]["array"]
        sr = example["audio"]["sampling_rate"]
        separated = separator.separate_array(arr, sr)
        return {"birdmixit_source0": separated[0]}

    # Map across dataset
    ds = ds.map(add_birdmixit_source)

    separator.close()

    # ===================
    # Save dataset
    # ===================

    # if not dataset_was_modified:
    #     print('Preprocessed Dataset was not modified.')
    
    # else:
    #     # Save to temporary location
    #     temp_path = preprocessed_dataset_path + "_temp"
    #     os.makedirs(temp_path, exist_ok=True)
    #     dataset.save_to_disk(temp_path)

    #     # Move old data to backup
    #     backup_path = preprocessed_dataset_path + "_backup"
    #     if os.path.exists(preprocessed_dataset_path):
    #         shutil.move(preprocessed_dataset_path, backup_path)

    #     # Move temp data into place
    #     shutil.move(temp_path, preprocessed_dataset_path)

    #     # Optionally remove backup
    #     shutil.rmtree(backup_path)