#!/bin/bash

# This script sets up the InstructPix2Pix model and environment.
# Usage: ./setup_instrctpix2pix.sh [ENV_NAME]

set -e

# Get environment name from argument or use default
ENV_NAME=${1:-"Falcon"}

echo "Setting up InstructPix2Pix in environment: $ENV_NAME"

# # Ensure we're in the correct conda environment
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

cd ../
# Directory renaming logic
if [ -d "external/instruct-pix2pix" ]; then
    mv "external/instruct-pix2pix" "external/instructpix2pix"
    echo "Directory 'external/instruct-pix2pix' renamed to 'external/instructpix2pix' successfully."
else
    echo "Note: Directory 'external/instruct-pix2pix' not found (may already be renamed or not needed)."
fi

# # Generate requirements.txt from pip section in environment.yaml and install
echo "Extracting pip dependencies from environment.yaml to requirements.txt..."
if [ -f "external/instructpix2pix/environment.yaml" ]; then
    # Extract pip dependencies and write to requirements.txt
    awk '/- pip:/ {flag=1; next} flag && /^ *- / {sub(/^ *- /,""); print} flag && !/^ *- / {flag=0}' external/instructpix2pix/environment.yaml > external/instructpix2pix/requirements.txt
    if [ -s "external/instructpix2pix/requirements.txt" ]; then
        pip install -r external/instructpix2pix/requirements.txt
    else
        echo "No pip dependencies found in environment.yaml."
    fi
else
    echo "Warning: environment.yaml not found in external/instructpix2pix."
fi

# # Navigate to scripts directory and download checkpoints
# cd external/instructpix2pix/scripts || { echo "Failed to enter scripts directory"; exit 1; }
# chmod +x download_checkpoints.sh

# CHECKPOINT_DIR="../checkpoints"
# CHECKPOINT_FILE="$CHECKPOINT_DIR/instruct-pix2pix-00-22000.ckpt"

# if [ -f "$CHECKPOINT_FILE" ]; then
#     echo "Checkpoints already exist at $CHECKPOINT_FILE, skipping download."
# else
#     echo "Downloading checkpoints..."
#     ./download_checkpoints.sh
# fi

# # Go to stable diffusion folder and install additional dependencies
# echo "Setting up Stable Diffusion environment dependencies..."

# cd ../stable_diffusion || { echo "Failed to enter stable_diffusion directory"; exit 1; }

# # Use the same environment, just install environment.yaml dependencies into it
# echo "Renaming environment in environment.yaml from 'ldm' to 'Falcon'..."
# sed -i 's/^name: ldm$/name: Falcon/' environment.yaml

# echo "Installing dependencies from environment.yaml into environment: $ENV_NAME"
# conda env update -n "$ENV_NAME" -f environment.yaml

# # Re-activate the same environment to ensure all paths are correctly loaded
# conda activate "$ENV_NAME"

# # Install required pip packages
# pip install transformers==4.19.2 diffusers invisible-watermark

# # Install the local package
# pip install -e .

# echo "InstructPix2Pix and Stable Diffusion setup complete in environment: $ENV_NAME"
