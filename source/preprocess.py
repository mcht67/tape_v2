from perch_hoplite.zoo import model_configs
from omegaconf import OmegaConf
from datasets import load_from_disk, Dataset
import datasets
import numpy as np
from functools import partial
import shutil
from utils.dsp import resample_audio
import os
import tempfile
from preprocess.embeddings import add_embeddings_batchwise

import hashlib
from pathlib import Path

def get_file_hash(filepath):
    """Get SHA256 hash of a file"""
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def should_recompute_feature(feature_name, code_files, dataset):
    """Check if feature should be recomputed based on code changes"""
    # Check if feature exists
    if feature_name not in dataset.columns:
        return True
    
    # Get current code hashes
    current_hashes = {str(Path(f).name): get_file_hash(f) for f in code_files}
    
    # Check stored hashes (you could store these in dataset metadata or separate file)
    stored_hash_key = f"{feature_name}_code_hashes"
    if stored_hash_key not in dataset.attrs:
        return True
    
    stored_hashes = dataset.attrs[stored_hash_key]
    
    # Compare hashes
    return current_hashes != stored_hashes

def should_recompute_step(fcode_files, dataset):

    # Get current code hashes
    current_hashes = {str(Path(f).name): get_file_hash(f) for f in code_files}
    
    # Check stored hashes (you could store these in dataset metadata or separate file)
    stored_hash_key = f"{feature_name}_code_hashes"
    if stored_hash_key not in dataset.attrs:
        return True
    
    stored_hashes = dataset.attrs[stored_hash_key]
    
    # Compare hashes
    return current_hashes != stored_hashes

def update_feature_hashes(feature_name, code_files, dataset):
    """Update stored hashes after feature computation"""
    current_hashes = {str(Path(f).name): get_file_hash(f) for f in code_files}
    dataset.attrs[f"{feature_name}_code_hashes"] = current_hashes

def get_function_file_path(func):
    """Get the file path of a function"""
    return inspect.getfile(func)

import hashlib
import json
import inspect
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

class FileCache:
    """
    A caching system for tracking when preprocessing steps need recomputation
    based on source code changes.
    """
    
    def __init__(self, cache_file: str = "cache/file_hashes.json"):
        self.cache_file = Path(cache_file)
        # Create cache directory if it doesn't exist
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from file, create new if doesn't exist"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load cache file {self.cache_file}: {e}")
                print("Creating new cache...")
        
        # Create new cache structure
        return {
            "steps": {},
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
    
    def _save_cache(self) -> None:
        """Save cache to file"""
        self.cache["metadata"]["updated_at"] = datetime.now().isoformat()
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2, sort_keys=True)
        except IOError as e:
            print(f"Warning: Could not save cache file: {e}")
    
    def should_recompute(self, step_name: str, code_files: List[str], 
                        config: Optional[Dict] = None) -> bool:
        """
        Check if step should be recomputed based on code/config changes
        
        Args:
            step_name: Name of the step
            code_files: List of code files that affect this step
            config: Optional configuration dict to include in hash
            
        Returns:
            True if step should be recomputed, False otherwise
        """
        try:
            current_hash = self._compute_hash(code_files, config)
            cached_entry = self.cache["steps"].get(step_name, {})
            cached_hash = cached_entry.get("hash")
            
            if cached_hash != current_hash:
                print(f"Hash changed for {step_name}: {cached_hash} -> {current_hash}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Warning: Error checking cache for {step_name}: {e}")
            # If we can't check cache, assume we need to recompute
            return True
    
    def mark_computed(self, step_name: str, code_files: List[str], 
                     config: Optional[Dict] = None) -> None:
        """
        Mark step as computed with current code/config hash
        
        Args:
            step_name: Name of the step
            code_files: List of code files that affect this step
            config: Optional configuration dict to include in hash
        """
        try:
            current_hash = self._compute_hash(code_files, config)
            
            self.cache["steps"][step_name] = {
                "hash": current_hash,
                "computed_at": datetime.now().isoformat(),
                "files": [str(Path(f).resolve()) for f in code_files],
                "config_included": config is not None
            }
            
            self._save_cache()
            print(f"Marked {step_name} as computed (hash: {current_hash[:8]}...)")
            
        except Exception as e:
            print(f"Warning: Could not mark {step_name} as computed: {e}")
    
    def _compute_hash(self, code_files: List[str], config: Optional[Dict] = None) -> str:
        """Compute hash for code files and optional config"""
        hasher = hashlib.sha256()
        
        # Hash code files
        for file_path in sorted(code_files):
            file_path = Path(file_path).resolve()
            if not file_path.exists():
                raise FileNotFoundError(f"Code file not found: {file_path}")
            
            with open(file_path, 'rb') as f:
                hasher.update(f.read())
        
        # Include config in hash if provided
        if config is not None:
            config_str = json.dumps(config, sort_keys=True)
            hasher.update(config_str.encode('utf-8'))
        
        return hasher.hexdigest()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get human-readable cache information"""
        steps = self.cache.get("steps", {})
        return {
            "cache_file": str(self.cache_file),
            "total_steps": len(steps),
            "steps": list(steps.keys()),
            "last_updated": self.cache.get("metadata", {}).get("updated_at"),
            "created_at": self.cache.get("metadata", {}).get("created_at")
        }
    
    def clear_step(self, step_name: str) -> bool:
        """Remove a specific step from cache"""
        if step_name in self.cache["steps"]:
            del self.cache["steps"][step_name]
            self._save_cache()
            print(f"Cleared cache for step: {step_name}")
            return True
        return False
    
    def clear_all(self) -> None:
        """Clear all cached steps"""
        self.cache["steps"] = {}
        self._save_cache()
        print("Cleared all step cache entries")
    
    def invalidate_missing_files(self) -> List[str]:
        """Remove cache entries for steps whose code files no longer exist"""
        invalidated = []
        
        for step_name, entry in list(self.cache["steps"].items()):
            files = entry.get("files", [])
            for file_path in files:
                if not Path(file_path).exists():
                    del self.cache["steps"][step_name]
                    invalidated.append(step_name)
                    print(f"Invalidated {step_name} (missing file: {file_path})")
                    break
        
        if invalidated:
            self._save_cache()
        
        return invalidated


def main():

    with tempfile.TemporaryDirectory() as temp_cache_dir:
        # Set HuggingFace cache to this temporary directory
        datasets.config.HF_DATASETS_CACHE = temp_cache_dir
        
        # ===================
        # Configuration
        # ===================

        cfg = OmegaConf.load("params.yaml")

        model_keys = cfg.embeddings.models
        feature_key = cfg.embeddings.feature

        polyphonic_dataset_path = cfg.paths.polyphonic_dataset
        preprocessed_dataset_path = cfg.paths.preprocessed_dataset

        # Load Dataset depending on state of preprocessing dataset
        if not os.path.exists(preprocessed_dataset_path):
            dataset = load_from_disk(polyphonic_dataset_path)
            os.makedirs(preprocessed_dataset_path, exist_ok=True)
        else:
          dataset = load_from_disk(preprocessed_dataset_path)
        
        dataset_was_modified = False
        
        # Initialize cache
        cache = FileCache("file_cache/preprocessing_cache.json")

        # Print cache info
        print("Cache info:", cache.get_cache_info())
        
        # Clean up any stale entries
        invalidated = cache.invalidate_missing_files()
        if invalidated:
            print(f"Invalidated steps with missing files: {invalidated}")
        
        # Define your steps with their corresponding functions
        steps = [
            {"name": "embeddings"}
        ]
        
        # ===================
        # Preprocessing
        # ===================

        # Print cache info
        print("Cache info:", cache.get_cache_info())

        # Embeddings
        embeddings_step_name = 'embeddings'

        # Check hashes for code changes
        embeddings_file = inspect.getfile(add_embeddings_batchwise)
        should_recompute = cache.should_recompute(embeddings_step_name, [embeddings_file])

        # Compute embeddings
        modified_dataset, modified = add_embeddings_batchwise(model_keys, feature_key, dataset, temp_cache_dir, recompute=should_recompute)

        if should_recompute:
            print(f"Recomputed {embeddings_step_name} due to code changes.")
            cache.mark_computed(embeddings_step_name, [embeddings_file])

        if modified:
            dataset = modified_dataset
            dataset_was_modified = True
        else:
            print("No changes in embeddings.")

        # Print cache info
        print("Cache info:", cache.get_cache_info())

        # ===================
        # Save dataset
        # ===================

        if not dataset_was_modified:
            print('Preprocessed Dataset was not modified.')
        
        else:
            # Save to temporary location
            temp_path = preprocessed_dataset_path + "_temp"
            os.makedirs(temp_path, exist_ok=True)
            dataset.save_to_disk(temp_path)

            # Move old data to backup
            backup_path = preprocessed_dataset_path + "_backup"
            if os.path.exists(preprocessed_dataset_path):
                shutil.move(preprocessed_dataset_path, backup_path)

            # Move temp data into place
            shutil.move(temp_path, preprocessed_dataset_path)

            # Optionally remove backup
            shutil.rmtree(backup_path)

if __name__ == "__main__":
    main()