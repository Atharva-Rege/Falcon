#!/usr/bin/env bash

# Exit on error
set -e

echo "Running full Falcon setup..."

ENV_NAME="Falcon"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

# -----------------------------
# 1. Create Conda environment
# -----------------------------
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda env '$ENV_NAME' already exists"
else
    echo "Creating Conda environment '$ENV_NAME'..."
    conda create -n "$ENV_NAME" python=3.10 -y

    echo "Activating environment '$ENV_NAME'..."
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    # Install common dependencies that might be needed
    echo "Installing common dependencies..."
    conda install -n "$ENV_NAME" pip -y
fi

# # -----------------------------
# # 2. Setup InstructPix2Pix
# # -----------------------------
echo "------------------------------------------------------------------------------------------------------------"
cd scripts/
if [ -f "./setup_instrctpix2pix.sh" ]; then
    echo "Setting up InstructPix2Pix..."
    bash ./setup_instrctpix2pix.sh "$ENV_NAME"
else
    echo "setup_instrctpix2pix.sh not found!"
    exit 1
fi

# # -----------------------------
# # 3. Setup Watermark Anything
# # -----------------------------
# if [ -f "./setup_watermarkanything.sh" ]; then
#     echo "Setting up Watermark Anything..."
#     bash ./setup_watermarkanything.sh "$ENV_NAME"
# else
#     echo "setup_watermarkanything.sh not found!"
#     exit 1
# fi

# -----------------------------
# 4. Final message
# -----------------------------
echo "All components of Falcon are ready!"
echo "To use the environment, run: conda activate $ENV_NAME"