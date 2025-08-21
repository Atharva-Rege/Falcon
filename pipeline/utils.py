import os
import json

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def save_json(obj: dict, path: str):
    """Save dictionary as JSON."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_image_stub(path: str):
    """Stub: create an empty file as placeholder for an image."""
    with open(path, "wb") as f:
        f.write(b"")  # later this will be real image data
