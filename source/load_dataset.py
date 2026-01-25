from datasets import load_dataset
from omegaconf import OmegaConf
import os
from datetime import datetime
import json

from utils.general import overwrite_dataset

# Configuration
cfg = OmegaConf.load("params.yaml")

print("restart")

dataset_path = cfg.path.dataset
huggingface_path = cfg.dataset.huggingface_path
dataset_subset = cfg.dataset.subset
dataset_metadata_path = cfg.path.dataset_metadata

# Load dataset
dataset = load_dataset(huggingface_path, dataset_subset)

dataset['train'] = dataset['train']
dataset['test'] = dataset['test']
dataset['validation'] = dataset['validation']

# Store dataset
os.makedirs(dataset_path, exist_ok=True)
overwrite_dataset(dataset, dataset_path, store_backup=False)

# Store metadata
metadata = {
        "datetime": datetime.now().isoformat(),
        "huggingface_path": huggingface_path,
        "subset": dataset_subset,
        #"split": split,
        "dataset_path": dataset_path
    }

metadata_dir = os.path.dirname(dataset_metadata_path)
if metadata_dir:
    os.makedirs(metadata_dir, exist_ok=True)
with open(dataset_metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)