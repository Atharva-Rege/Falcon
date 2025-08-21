"""
Main pipeline entrypoint:
1. Load dataset (multiple samples).
2. Save each sample into experiments/ directory.
3. Run WatermarkAnything watermarking on them.
"""

import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import load_custom_dataset
from models.watermarking.watermark_anything import WatermarkAnythingWrapper
from configs.logger import setup_logger

logger = setup_logger("Main", "logs", "info")


def save_sample_to_experiments(sample, base_dir="experiments"):
    """
    Save a dataset sample to the experiments/ directory structure.
    """
    sample_id = f"image_{sample['id']}"
    sample_dir = os.path.join(base_dir, sample_id)
    original_dir = os.path.join(sample_dir, "original")

    os.makedirs(original_dir, exist_ok=True)

    # Save image
    img_path = os.path.join(original_dir, "image.png")
    sample["image"].save(img_path)

    # Save captions.json
    captions = {
        "original": sample["source_prompt"],
        "v1": sample["edit_prompts"][0],
        "v2": sample["edit_prompts"][1],
        "v3": sample["edit_prompts"][2],
    }
    with open(os.path.join(original_dir, "captions.json"), "w") as f:
        json.dump(captions, f, indent=4)

    logger.info(f"Saved sample {sample_id} at {sample_dir}")
    return sample_dir


def main():
    # 1. Load dataset
    samples = load_custom_dataset()
    if not samples:
        logger.error("No samples loaded from dataset")
        return

    # 2. Initialize the watermarking model ONCE before the loop for efficiency
    wam = WatermarkAnythingWrapper(model_name="wam")

    # 3. Loop through and process the first two samples
    for sample in samples[:2]:  # Using list slicing to get the first 2
        logger.info(f"Processing sample id={sample['id']}")

        # Save the current sample to the experiments/ structure
        sample_dir = save_sample_to_experiments(sample)

        # Run watermarking on the current sample
        wam.process_sample(sample_dir)


if __name__ == "__main__":
    main()