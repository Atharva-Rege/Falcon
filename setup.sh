#!/usr/bin/env bash

# Exit on error
set -e

echo "🚀 Setting up Falcon project with a single conda environment..."

# -----------------------------
# 1. Rename watermark-anything submodule (if exists)
# -----------------------------
if [ -d "external/watermark-anything" ]; then
    echo "📂 Renaming watermark-anything → watermarkanything"
    mv external/watermark-anything external/watermarkanything
fi

# -----------------------------
# 2. Create single conda env
# -----------------------------
ENV_NAME="falcon"

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "❌ Conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

# Create env if it doesn’t exist
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "✅ Conda env '$ENV_NAME' already exists"
else
    echo "📦 Creating conda env '$ENV_NAME'..."
    conda create -y -n $ENV_NAME python=3.10.14
fi

# -----------------------------
# 3. Install core dependencies
# -----------------------------
echo "📚 Installing dependencies into $ENV_NAME..."

# PyTorch with CUDA 12.4
conda run -n $ENV_NAME conda install -y pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia

# Falcon pipeline requirements
if [ -f "requirements.txt" ]; then
    conda run -n $ENV_NAME pip install -r requirements.txt
else
    echo "⚠️ No Falcon requirements.txt found, skipping"
fi

# Watermark Anything requirements
if [ -f "external/watermarkanything/requirements.txt" ]; then
    conda run -n $ENV_NAME pip install -r external/watermarkanything/requirements.txt
else
    echo "⚠️ No watermarkanything requirements.txt found, skipping"
fi

# -----------------------------
# 4. Download pretrained weights
# -----------------------------
echo "⬇️ Downloading Watermark Anything pretrained weights..."
mkdir -p external/watermarkanything/checkpoints
wget -nc https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth -P external/watermarkanything/checkpoints/
wget -nc https://dl.fbaipublicfiles.com/watermark_anything/wam_coco.pth -P external/watermarkanything/checkpoints/

# -----------------------------
# 5. Final message
# -----------------------------
echo "🎉 Setup complete!"
echo "👉 Activate environment: conda activate $ENV_NAME"
