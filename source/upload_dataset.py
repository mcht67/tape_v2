from datasets import load_from_disk, Audio
from omegaconf import OmegaConf
from dotenv import load_dotenv
import os

from utils.dataset import get_data_dir

print("Start dataset upload...")

# Load the parameters from the config file
cfg = OmegaConf.load("params.yaml")
dataset_path = cfg.path.dataset
subset = cfg.dataset.subset
huggingface_user = 'mcht67'
huggingface_dataset_name = 'polyphonic-bird-set-with-embeddings'

 # Load environment variables from .env file
load_dotenv('local.env')
token=os.getenv('HUGGINGFACE_TOKEN')

# Huggingface login
# huggingface_hub.login(token=os.getenv('HUGGINGFACE_TOKEN'))
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

huggingface_path = huggingface_user + "/" + huggingface_dataset_name

polyphonic_dataset = load_from_disk(dataset_path)

commit_message_polyphonic = f"updates polyphonic dataset [subset: {subset}]"
data_dir = get_data_dir(cfg.dataset.config)
polyphonic_dataset.push_to_hub(huggingface_path, config_name=subset, data_dir=data_dir, commit_message=commit_message_polyphonic, token=huggingface_token)