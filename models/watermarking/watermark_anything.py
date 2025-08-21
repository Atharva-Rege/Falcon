"""
Wrapper for WatermarkAnything (WAM).
Integrates with the experiments/ directory structure.
"""

import os
import json
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.logger import setup_logger
from configs.config import Config

# WAM imports
from external.watermarkanything.watermark_anything.data.metrics import msg_predict_inference
from external.watermarkanything.notebooks.inference_utils import (
    load_model_from_checkpoint, default_transform, unnormalize_img, create_random_mask
)

logger = setup_logger("WAM", "logs", "debug")


class WatermarkAnythingWrapper:
    def __init__(self, model_name="wam_model1",
                 ckpt_dir="external/watermarkanything/checkpoints",
                 device=None):
        """
        Initialize WAM model.

        Args:
            model_name (str): Identifier for saving outputs.
            ckpt_dir (str): Path to WAM checkpoints (params.json + checkpoint.pth).
            device (str): "cuda" or "cpu". Defaults to CUDA if available.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        json_path = os.path.join(ckpt_dir, "params.json")
        ckpt_path = os.path.join(ckpt_dir, "wam_mit.pth")

        if not (os.path.exists(json_path) and os.path.exists(ckpt_path)):
            raise FileNotFoundError(
                f"Missing checkpoint files in {ckpt_dir}. "
                f"Expected params.json and checkpoint.pth"
            )

        logger.info("Loading WAM model...")
        self.model = load_model_from_checkpoint(json_path, ckpt_path).to(self.device).eval()
        logger.info("WAM model loaded successfully")

    def process_sample(self, sample_dir: str, wm_bits: int = 32, proportion_masked: float = 0.5):
        """
        Apply watermarking + detection for a single dataset sample.

        Args:
            sample_dir (str): Path to sample directory (e.g. experiments/image_0001/).
            wm_bits (int): Number of bits in watermark message.
            proportion_masked (float): Proportion of image to watermark.
        """
        logger.info(f"Processing sample at {sample_dir}")

        # Input: experiments/image_xxxx/original/image.png
        img_path = os.path.join(sample_dir, "original", "image.png")
        if not os.path.exists(img_path):
            logger.error(f"No input image found at {img_path}")
            return

        # Output dir: experiments/image_xxxx/watermarking/wm_model1/
        output_dir = os.path.join(sample_dir, "watermarking", self.model_name)
        os.makedirs(output_dir, exist_ok=True)

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_pt = default_transform(img).unsqueeze(0).to(self.device)

        # Random message
        wm_msg = torch.randint(0, 2, (wm_bits,)).float().to(self.device)

        # Embed watermark
        # outputs = self.model.embed(img_pt, wm_msg)
        outputs = self.model.embed(img_pt, wm_msg.unsqueeze(0))

        # Random mask
        mask = create_random_mask(img_pt, num_masks=1, mask_percentage=proportion_masked)
        img_w = outputs['imgs_w'] * mask + img_pt * (1 - mask)

        # Detect watermark
        preds = self.model.detect(img_w)["preds"]
        mask_preds = torch.sigmoid(preds[:, 0, :, :])
        bit_preds = preds[:, 1:, :, :]

        pred_message = msg_predict_inference(bit_preds, mask_preds).cpu().float()
        bit_acc = (pred_message == wm_msg.cpu()).float().mean().item()

        # Save images
        mask_preds_res = F.interpolate(
            mask_preds.unsqueeze(1),
            size=(img_pt.shape[-2], img_pt.shape[-1]),
            mode="bilinear",
            align_corners=False
        )

        save_image(unnormalize_img(img_w), os.path.join(output_dir, "watermarked.png"))
        save_image(mask_preds_res, os.path.join(output_dir, "mask_pred.png"))
        save_image(mask, os.path.join(output_dir, "mask_target.png"))

        # Save detection results
        detection_data = {
            "original_image": img_path,
            "watermarked_image": os.path.join(output_dir, "watermarked.png"),
            "predicted_message": pred_message[0].tolist(),
            "bit_accuracy": bit_acc,
            "wm_bits": wm_bits,
            "proportion_masked": proportion_masked,
            "model": self.model_name,
        }
        with open(os.path.join(output_dir, "detection.json"), "w") as f:
            json.dump(detection_data, f, indent=4)

        logger.info(f"✅ Done sample {os.path.basename(sample_dir)} | Bit Acc: {bit_acc:.3f}")


if __name__ == "__main__":
    cfg = Config()
    wam = WatermarkAnythingWrapper(model_name="wam_model1")

    # Example: process one sample
    wam.process_sample("experiments/image_0001")
