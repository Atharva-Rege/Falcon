"""
This file contains the wrapper on instructpix2pix model.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from __future__ import annotations

import math
import random
import sys
import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps, ImageDraw, ImageFont
from torch import autocast

from configs.config import Config
from configs.logger import setup_logger
from external.instrctpix2pix.stable_diffusion.ldm.util import instantiate_from_config
logger = setup_logger("IPix2Pix","logs","debug")

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


class InstructPix2PixWrapper:
    def __init__(self, config_path):
        self.config = OmegaConf.load(config_path)
        pix2pix_config = self.config.instructpix2pix
        self.resolution = pix2pix_config.resolution
        self.steps = pix2pix_config.steps
        self.cfg_text = pix2pix_config.cfg_text
        self.cfg_image = pix2pix_config.cfg_image
        self.ckpt_path = pix2pix_config.ckpt_path
        self.vae_ckpt_path = pix2pix_config.vae_ckpt_path
        model_config = OmegaConf.load(pix2pix_config.config_path)
        
        # Initialize and load the model
        self.model = self._load_model_from_config(model_config, self.ckpt_path, self.vae_ckpt_path)
        self.model.eval().cuda()
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.null_token = self.model.get_learned_conditioning([""])

    def _load_model_from_config(self, config, ckpt, vae_ckpt=None, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        if vae_ckpt is not None:
            print(f"Loading VAE from {vae_ckpt}")
            vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
            sd = {
                k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
                for k, v in sd.items()
            }
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
        return model

    def _add_watermark(self, image, text):
        """Adds a text watermark to a PIL Image."""
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            font = ImageFont.load_default()
        
        text_width, text_height = draw.textsize(text, font=font)
        x = image.width - text_width - 10
        y = image.height - text_height - 10

        draw.text((x, y), text, fill=(255, 255, 255, 128), font=font)
        return image

    def run_edit(self, input_image_path, output_image_path, edit_prompt, seed=None):
        seed = random.randint(0, 100000) if seed is None else seed
        
        input_image = Image.open(input_image_path).convert("RGB")
        width, height = input_image.size
        factor = self.resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
        
        if not edit_prompt:
            watermarked_image = self._add_watermark(input_image, "No edit prompt provided")
            watermarked_image.save(output_image_path)
            return

        with torch.no_grad(), autocast("cuda"), self.model.ema_scope():
            cond = {}
            cond["c_crossattn"] = [self.model.get_learned_conditioning([edit_prompt])]
            input_image_tensor = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
            input_image_tensor = rearrange(input_image_tensor, "h w c -> 1 c h w").to(self.model.device)
            cond["c_concat"] = [self.model.encode_first_stage(input_image_tensor).mode()]

            uncond = {}
            uncond["c_crossattn"] = [self.null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = self.model_wrap.get_sigmas(self.steps)
            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": self.cfg_text,
                "image_cfg_scale": self.cfg_image,
            }
            torch.manual_seed(seed)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(self.model_wrap_cfg, z, sigmas, extra_args=extra_args)
            x = self.model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
            
            watermarked_image = self._add_watermark(edited_image, edit_prompt)
            watermarked_image.save(output_image_path)
            print(f"Edited image saved to {output_image_path}")


if __name__ == "__main__":
    editor = InstructPix2PixWrapper("config/config.yaml")
    
    editor.run_edit(
        input_image_path="/home/shreyas/Desktop/Shreyas/Projects/Falcon/experiments/image_000000000001/original/image.png",
        output_image_path="/home/shreyas/Desktop/Shreyas/Projects/Falcon/output.png",
        edit_prompt="a square cake with strawberry frosting on a plastic plate"
    )