from datasets import load_dataset
from omegaconf import OmegaConf
import os
from datetime import datetime
import json
import huggingface_hub
from dotenv import load_dotenv

from utils.general import overwrite_dataset

# Configuration
cfg = OmegaConf.load("params.yaml")

dataset_path = cfg.path.dataset
huggingface_path = cfg.dataset.huggingface_path
dataset_subset = cfg.dataset.subset
dataset_metadata_path = cfg.path.dataset_metadata

# Huggingface login
load_dotenv('local.env')
token=os.getenv('HUGGINGFACE_TOKEN')
huggingface_hub.login(token=os.getenv('HUGGINGFACE_TOKEN'))

# Load polyphonic dataset
dataset = load_dataset(huggingface_path, dataset_subset + '_polyphonic')

path = huggingface_hub.hf_hub_download(
    repo_id=huggingface_path,
    filename=f"{dataset_subset}/metadata.json",
    repo_type="dataset"
)

metadata = None
with open(path) as f:
    metadata = json.load(f)

dataset['train'] = dataset['train'].select(range(10))
dataset['test'] = dataset['test'].select(range(1))
dataset['validation'] = dataset['validation'].select(range(1))

# Store dataset
os.makedirs(dataset_path, exist_ok=True)
overwrite_dataset(dataset, dataset_path, store_backup=False)

# Store metadata
if not metadata:
    metadata = {
        "datetime": datetime.now().isoformat(),
        "huggingface_path": huggingface_path,
        "subset": dataset_subset,
        "dataset_path": dataset_path
    }

metadata_dir = os.path.dirname(dataset_metadata_path)
if metadata_dir:
    os.makedirs(metadata_dir, exist_ok=True)
with open(dataset_metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)