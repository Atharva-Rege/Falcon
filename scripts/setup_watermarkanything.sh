# setup_watermarkanything.sh
#!/usr/bin/env bash

# Exit on error
set -e

ENV_NAME="falcon"

echo "Setting up Watermark Anything..."

# -----------------------------
# 1. Rename submodule folder if needed
# -----------------------------
if [ -d "external/watermark-anything" ]; then
    echo "Renaming watermark-anything → watermarkanything"
    mv external/watermark-anything external/watermarkanything
fi

# -----------------------------
# 2. Ensure Conda env exists
# -----------------------------
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda env '$ENV_NAME' found"
else
    echo "Conda env '$ENV_NAME' not found. Please run main Falcon setup first."
    exit 1
fi

# -----------------------------
# 3. Install WatermarkAnything dependencies
# -----------------------------
if [ -f "external/watermarkanything/requirements.txt" ]; then
    echo "Installing WatermarkAnything requirements into $ENV_NAME..."
    conda run -n $ENV_NAME pip install -r external/watermarkanything/requirements.txt
else
    echo "No requirements.txt for WatermarkAnything found, skipping dependency install"
fi

# -----------------------------
# 4. Download pretrained weights (skip if already present)
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
# 5. Final message
# -----------------------------
echo "Watermark Anything setup complete!"
echo "To use: conda activate $ENV_NAME"
