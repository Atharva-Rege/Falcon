# main.sh
#!/usr/bin/env bash

# Exit on error
set -e

echo "Running full Falcon setup..."

# -----------------------------
# 1. Setup InstructPix2Pix
# -----------------------------
cd scripts/
if [ -f "./setup_instrctpix2pix.sh" ]; then
    echo "Setting up InstructPix2Pix..."
    bash ./setup_instrctpix2pix.sh
else
    echo "setup_instrctpix2pix.sh not found!"
    exit 1
fi

# -----------------------------
# 2. Setup Watermark Anything
# -----------------------------
if [ -f "./setup_watermarkanything.sh" ]; then
    echo "Setting up Watermark Anything..."
    bash ./setup_watermarkanything.sh
else
    echo "setup_watermarkanything.sh not found!"
    exit 1
fi

# -----------------------------
# 3. Final message
# -----------------------------
echo "All components of Falcon are ready!"
