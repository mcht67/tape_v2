from datasets import load_from_disk, Audio
from omegaconf import OmegaConf

print("Start dataset upload...")

# Load the parameters from the config file
cfg = OmegaConf.load("params.yaml")
dataset_path = cfg.path.dataset
subset = cfg.dataset.subset
huggingface_user = 'mcht67'
huggingface_dataset_name = 'polyphonic-bird-set-with-embeddings'

huggingface_path = huggingface_user + "/" + huggingface_dataset_name

polyphonic_dataset = load_from_disk(dataset_path)

commit_message_polyphonic = f"updates polyphonic dataset with in {subset}"
polyphonic_dataset.push_to_hub(huggingface_path, config_name=subset, private=True, commit_message=commit_message_polyphonic)