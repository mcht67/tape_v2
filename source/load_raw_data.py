from datasets import load_dataset
from omegaconf import OmegaConf
import os

# Load Configuration
cfg = OmegaConf.load("params.yaml")

sampling_rate = cfg.dataset.sampling_rate

dataset_name = cfg.dataset.name
dataset_subset = cfg.dataset.subset
dataset_split = cfg.dataset.split

output_path = cfg.paths.raw_dataset
os.makedirs(output_path, exist_ok=True)

# Load raw dataset
print("Load dataset", flush=True)
raw_dataset = load_dataset(dataset_name, dataset_subset, split=dataset_split, trust_remote_code=True)

# Save raw dataset
raw_dataset.save_to_disk(output_path)
print(f"Saved raw dataset {dataset_subset} to {output_path}")