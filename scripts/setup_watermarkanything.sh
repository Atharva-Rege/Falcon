#!/usr/bin/env bash

# setup_watermarkanything.sh
# Usage: ./setup_watermarkanything.sh [ENV_NAME]

set -e

# Get environment name from argument or use default
ENV_NAME=${1:-"Falcon"}

echo "Setting up Watermark Anything in environment: $ENV_NAME"

# Ensure we're in the correct conda environment
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# -----------------------------
# 1. Rename submodule folder if needed
# -----------------------------
if [ -d "external/watermark-anything" ]; then
    echo "Renaming watermark-anything → watermarkanything"
    mv external/watermark-anything external/watermarkanything
fi

# -----------------------------
# 2. Install WatermarkAnything dependencies
# -----------------------------
echo "Installing WatermarkAnything dependencies..."
if [ -f "external/watermarkanything/requirements.txt" ]; then
    pip install -r external/watermarkanything/requirements.txt
else
    echo "No requirements.txt for WatermarkAnything found, installing common dependencies..."
    # Install common dependencies that WatermarkAnything typically needs
    pip install opencv-python pillow numpy torch torchvision
fi

# -----------------------------
# 3. Download pretrained weights (skip if already present)
# -----------------------------
echo "Checking pretrained weights..."

CHECKPOINT_DIR="external/watermarkanything/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

WAM_MIT="$CHECKPOINT_DIR/wam_mit.pth"
WAM_COCO="$CHECKPOINT_DIR/wam_coco.pth"

if [ -f "$WAM_MIT" ]; then
    echo "wam_mit.pth already exists, skipping download"
else
    echo "Downloading wam_mit.pth..."
    wget -nc https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth -P "$CHECKPOINT_DIR/"
fi

if [ -f "$WAM_COCO" ]; then
    echo "wam_coco.pth already exists, skipping download"
else
    echo "Downloading wam_coco.pth..."
    wget -nc https://dl.fbaipublicfiles.com/watermark_anything/wam_coco.pth -P "$CHECKPOINT_DIR/"
fi

# -----------------------------
# 4. Final message
# -----------------------------
echo "Watermark Anything setup complete in environment: $ENV_NAME"