"""
Wrapper for WatermarkAnything (WAM).
Supports single and multi-watermark embedding/detection.
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
    load_model_from_checkpoint, default_transform, unnormalize_img, create_random_mask,
    multiwm_dbscan, msg2str
)

logger = setup_logger("WAM", "logs", "debug")


class WatermarkAnythingWrapper:
    def __init__(self, model_name="wam_model1", ckpt_dir="external/watermarkanything/checkpoints", device=None):
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
        Single watermarking.
        """
        logger.info(f"Processing sample at {sample_dir}")

        img_path = os.path.join(sample_dir, "original", "image.png")
        if not os.path.exists(img_path):
            logger.error(f"No input image found at {img_path}")
            return

        output_dir = os.path.join(sample_dir, "watermarking", self.model_name)
        os.makedirs(output_dir, exist_ok=True)

        img = Image.open(img_path).convert("RGB")
        img_pt = default_transform(img).unsqueeze(0).to(self.device)

        wm_msg = torch.randint(0, 2, (wm_bits,)).float().to(self.device)

        outputs = self.model.embed(img_pt, wm_msg.unsqueeze(0))

        mask = create_random_mask(img_pt, num_masks=1, mask_percentage=proportion_masked)
        img_w = outputs['imgs_w'] * mask + img_pt * (1 - mask)

        preds = self.model.detect(img_w)["preds"]
        mask_preds = torch.sigmoid(preds[:, 0, :, :])
        bit_preds = preds[:, 1:, :, :]

        pred_message = msg_predict_inference(bit_preds, mask_preds).cpu().float()
        bit_acc = (pred_message == wm_msg.cpu()).float().mean().item()

        mask_preds_res = F.interpolate(
            mask_preds.unsqueeze(1),
            size=(img_pt.shape[-2], img_pt.shape[-1]),
            mode="bilinear",
            align_corners=False
        )

        save_image(unnormalize_img(img_w), os.path.join(output_dir, "watermarked.png"))
        save_image(mask_preds_res, os.path.join(output_dir, "mask_pred.png"))
        save_image(mask, os.path.join(output_dir, "mask_target.png"))

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

    def process_sample_multi(self, sample_dir: str, num_msgs: int = 2, wm_bits: int = 32,
                             proportion_masked: float = 0.1, epsilon: float = 1, min_samples: int = 500):
        """
        Multi-watermarking with DBSCAN clustering.
        """
        logger.info(f"Processing MULTI sample at {sample_dir}")

        img_path = os.path.join(sample_dir, "original", "image.png")
        if not os.path.exists(img_path):
            logger.error(f"No input image found at {img_path}")
            return

        output_dir = os.path.join(sample_dir, f"watermarking/{self.model_name}_multi")
        os.makedirs(output_dir, exist_ok=True)

        img = Image.open(img_path).convert("RGB")
        img_pt = default_transform(img).unsqueeze(0).to(self.device)

        wm_msgs = torch.randint(0, 2, (num_msgs, wm_bits)).float().to(self.device)

        masks = create_random_mask(img_pt, num_masks=num_msgs, mask_percentage=proportion_masked)

        multi_wm_img = img_pt.clone()
        for ii in range(num_msgs):
            wm_msg, mask = wm_msgs[ii].unsqueeze(0), masks[ii]
            outputs = self.model.embed(img_pt, wm_msg)
            multi_wm_img = outputs['imgs_w'] * mask + multi_wm_img * (1 - mask)

        preds = self.model.detect(multi_wm_img)["preds"]
        mask_preds = torch.sigmoid(preds[:, 0, :, :])
        bit_preds = preds[:, 1:, :, :]

        centroids, positions = multiwm_dbscan(bit_preds, mask_preds,
                                              epsilon=epsilon, min_samples=min_samples)
        centroids_pt = torch.stack(list(centroids.values())) if centroids else torch.empty(0)

        save_image(unnormalize_img(multi_wm_img), os.path.join(output_dir, "multi_watermarked.png"))
        save_image(mask, os.path.join(output_dir, "mask_targets.png"))

        detection_data = {
            "original_image": img_path,
            "multi_watermarked_image": os.path.join(output_dir, "multi_watermarked.png"),
            "num_msgs_hidden": num_msgs,
            "num_msgs_detected": len(centroids),
            "wm_bits": wm_bits,
            "epsilon": epsilon,
            "min_samples": min_samples,
            "model": self.model_name,
            "detected_messages": []
        }

        for centroid in centroids_pt:
            msg_str = msg2str(centroid)
            bit_acc = (centroid == wm_msgs).float().mean(dim=1)
            best_acc, idx = bit_acc.max(dim=0)
            hamming = int(torch.sum(centroid != wm_msgs[idx]).item())
            detection_data["detected_messages"].append({
                "msg_str": msg_str,
                "bit_accuracy": best_acc.item(),
                "hamming_distance": f"{hamming}/{wm_bits}"
            })
            logger.info(f"Found centroid: {msg_str} | Acc: {best_acc:.3f} | Hamming: {hamming}/{wm_bits}")

        with open(os.path.join(output_dir, "detection_multi.json"), "w") as f:
            json.dump(detection_data, f, indent=4)

        logger.info(f"✅ Done MULTI sample {os.path.basename(sample_dir)} | Found {len(centroids)} msgs")


if __name__ == "__main__":
    cfg = Config()
    wam = WatermarkAnythingWrapper(model_name="wam_model1")

    # Single watermark
    wam.process_sample("experiments/image_0001")

    # Multi watermark (example with 2)
    wam.process_sample_multi("experiments/image_0001", num_msgs=2)
