"""
Main pipeline entrypoint:
1. Load dataset (multiple samples).
2. Save each sample into experiments/ directory.
3. Run WatermarkAnything watermarking on them.
4. Edit the watermarked image using InstructPix2Pix and a caption variation.
"""

import os
import json
import sys
# Add parent directory to path to allow for imports from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import load_custom_dataset
from models.watermarking.watermark_anything import WatermarkAnythingWrapper
from models.editing.instruct_pix2pix import InstructPix2PixWrapper
from configs.logger import setup_logger

# Setup the logger for the main script
logger = setup_logger("Main", "logs", "info")


def save_sample_to_experiments(sample, base_dir="experiments"):
    """
    Save a dataset sample to the experiments/ directory structure.
    """
    # Create a unique directory name for the sample based on its ID
    sample_id = f"image_{sample['id']}"
    sample_dir = os.path.join(base_dir, sample_id)
    original_dir = os.path.join(sample_dir, "original")

    # Create the directory structure if it doesn't exist
    os.makedirs(original_dir, exist_ok=True)

    # Save the original image as a PNG file
    img_path = os.path.join(original_dir, "image.png")
    sample["image"].save(img_path)

    # Prepare and save the captions into a JSON file
    captions = {
        "original": sample["source_prompt"],
        "v1": sample["edit_prompts"][0],
        "v2": sample["edit_prompts"][1],
        "v3": sample["edit_prompts"][2],
    }
    captions_path = os.path.join(original_dir, "captions.json")
    with open(captions_path, "w") as f:
        json.dump(captions, f, indent=4)

    logger.info(f"Saved sample {sample_id} at {sample_dir}")
    return sample_dir, captions_path


def main():
    # 1. Load dataset
    samples = load_custom_dataset()
    if not samples:
        logger.error("No samples loaded from dataset")
        return

    # 2. Initialize the models ONCE before the loop for efficiency
    wam = WatermarkAnythingWrapper(model_name="wam")
    editor = InstructPix2PixWrapper(model_name="instructpix2pix")
    
    # 3. Loop through and process the first two samples
    for sample in samples[:2]:
        logger.info(f"Processing sample id={sample['id']}")

        # Save the current sample to the experiments/ structure
        sample_dir, captions_path = save_sample_to_experiments(sample)

        # 4. Run watermarking on the current sample
        # The watermarking model will save its output to the correct location
        # and we can use that path for the next step.
        wm_output_dir = wam.process_sample(sample_dir)
        wm_image_path = os.path.join(wm_output_dir, "watermarked.png")

        # 5. Load captions for the editing step
        with open(captions_path, "r") as f:
            captions = json.load(f)
        
        # We are using v1 for now, as requested
        edit_prompt = captions["v1"]

        # 6. Run image editing on the watermarked image
        # Construct the correct output path for the edited image based on the file structure
        # experiments/image_ID/watermarking/wm_model/editing/edit_model/v1/
        editing_base_dir = os.path.join(wm_output_dir, "editing", editor.model_name, "v1")
        os.makedirs(editing_base_dir, exist_ok=True)
        edited_image_path = os.path.join(editing_base_dir, "edited.png")

        logger.info(f"Editing {wm_image_path} with prompt: '{edit_prompt}'")
        editor.run_edit(
            input_image_path=wm_image_path,
            output_image_path=edited_image_path,
            edit_prompt=edit_prompt
        )

        # NOTE: The next step would be to run watermark detection on the edited image
        # and save the results, but this is not yet implemented in this script.
        # You would call a detection model wrapper here.

    logger.info("Pipeline completed successfully for selected samples.")

if __name__ == "__main__":
    main()
