import argparse
import os
from dotenv import load_dotenv
import shutil
from datasets import load_dataset

print("Running load dataset script...")

parser = argparse.ArgumentParser(description="Downloads/Updates huggingface dataset.")
parser.add_argument("--huggingface_path", type=str, required=True)
parser.add_argument("--dataset_config", type=str, required=True)
parser.add_argument("--download_mode", type=str, default='reuse_dataset_if_exists', help="Download mode for Huggingface datasets. Default: 'reuse_cache_if_exists'. Set to 'force_redownload' to force redownloading the dataset (e.g. after updating it on the Huggingface Hub).")
args = parser.parse_args()

print(f"[INFO] Downloading {args.huggingface_path} / {args.dataset_config}")

if shutil.which('sbatch') is not None:
    load_dotenv("global.env", override=True)
    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    if not cache_dir:
        raise ValueError("HF_DATASETS_CACHE is not set — check global.env or your environment")
    print(f"[INFO] HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE', 'NOT SET')}")
    dataset = load_dataset(args.huggingface_path, args.dataset_config, cache_dir=cache_dir, download_mode=args.download_mode)
else:
    dataset = load_dataset(args.huggingface_path, args.dataset_config, download_mode=args.download_mode)

print(f"[INFO] Done. Splits: {list(dataset.keys())}")
print(f"[INFO] Sizes: { {k: len(v) for k, v in dataset.items()} }")