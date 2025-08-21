"""This file contains the code to load the dataset using configs/config.yaml."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from configs.logger import setup_logger
from configs.config import Config

# Load global config
config = Config()
logger = setup_logger('Dataset','logs','info')

def load_custom_dataset():
    """
    Load dataset from Hugging Face based on values in config.yaml.
    
    Returns:
        List of dicts with dataset entries
    """
    dataset_name = config.get("dataset.name")
    split = config.get("dataset.split", "train")
    limit = config.get("dataset.limit", 5)

    logger.info(f"Loading dataset: {dataset_name} (split={split}) ...")
    dataset = load_dataset(dataset_name)

    # if split is wrong, pick first available
    if split not in dataset:
        logger.warning(f"Requested split '{split}' not found. Available: {list(dataset.keys())}")
        split = list(dataset.keys())[0]
        logger.info(f"Falling back to split '{split}'")

    ds_split = dataset[split]
    logger.info(f"Dataset split '{split}' loaded with {len(ds_split)} total samples")

    if limit > 0:
        logger.debug(f"Selecting first {limit} samples for preview")
        samples = ds_split.select(range(min(limit, len(ds_split))))
    else:
        samples = ds_split

    formatted = []
    for i, entry in enumerate(samples):
        logger.debug(f"Processing sample {i} (id={entry['id']})")
        formatted.append({
            "id": entry["id"],
            "image": entry["image"],  # PIL image object
            "source_prompt": entry["source_prompt"],
            "target_prompt": entry["target_prompt"],
            "edit_action": entry["edit_action"],
            "aspect_mapping": entry["aspect_mapping"],
            "blended_words": entry["blended_words"],
            "mask": entry["mask"],
            "edit_prompts": [
                entry.get("edit_1_prompt"),
                entry.get("edit_2_prompt"),
                entry.get("edit_3_prompt"),
            ]
        })

    logger.info(f"Finished processing {len(formatted)} samples")
    return formatted


if __name__ == "__main__":
    samples = load_custom_dataset()
    
    # for s in samples:
    #     logger.info(f"Sample ID: {s['id']}")
    #     logger.debug(f"Source prompt: {s['source_prompt']}")
    #     logger.debug(f"Target prompt: {s['target_prompt']}")
    #     logger.debug(f"Edit prompts: {s['edit_prompts']}")
    #     logger.debug(f"Image type: {type(s['image'])}")
