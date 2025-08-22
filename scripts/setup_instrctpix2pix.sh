# This script contains the code to setup instructpix2pix model.
#!/bin/bash

cd external/instrctpix2pix/scripts || { echo "Failed to enter scripts directory"; exit 1; }
chmod +x download_checkpoints.sh
CHECKPOINT_DIR="../checkpoints"
CHECKPOINT_FILE="$CHECKPOINT_DIR/instruct-pix2pix.ckpt"
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "Checkpoints already exist at $CHECKPOINT_FILE skipping download."
else
    echo "Downloading checkpoints..."
    ./download_checkpoints.sh
fi

# Optionally set executable permissions for other scripts (uncomment if needed)
# chmod +x download_data.sh download_pretrained_sd.sh

echo "Setup is ready for instrct pix2pix!"
