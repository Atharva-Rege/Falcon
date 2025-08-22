# Falcon: Text-Based Image Editing with Watermark Robustness

## Overview

Falcon is a research project investigating the robustness of watermarks in AI-generated images when subjected to text-based editing operations. The project combines InstructPix2Pix for image editing and WatermarkAnything for watermark injection and detection, enabling a comprehensive analysis of watermark sustainability under various editing transformations.

## Key Features

- **Image Editing**
  - Text-guided image manipulation using InstructPix2Pix
  - Support for both simple and complex editing instructions
  - Configurable editing parameters for fine-tuned control

- **Watermarking**
  - Robust watermark injection using WatermarkAnything
  - Support for visible and invisible watermarks
  - Multiple watermark models (MIT and COCO variants)

- **Analysis & Detection**
  - Watermark detection post-editing
  - Quantitative robustness metrics
  - Detailed experiment tracking and visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/falcon.git
cd falcon
```

2. Run the setup script to configure all components:
```bash
bash setup.sh
```

This will:
- Create a conda environment named 'falcon'
- Set up InstructPix2Pix and download its models
- Set up WatermarkAnything and download its models

## Project Structure

```
experiments/
│
├── image_000000000001/                       # Sample 1
│   ├── original/
│   │   ├── image.png                        # Input image
│   │   └── captions.json                    # Edit instructions
│   │
│   ├── watermarking/
│   │   └── wm_model1/                       # Watermark model
│   │       ├── watermarked.png              # Watermarked image
│   │       ├── detection.json               # Initial detection results
│   │       └── editing/
│   │           └── edit_model1/             # Edit model
│   │               └── v1/                  # Edit variation
│   │                   ├── edited.png       # Edited result
│   │                   └── detection.json   # Post-edit detection
│   └── summary.json                         # Sample summary
│
└── global_summary.json                       # Experiment summary
```

## Usage

### Single Image Pipeline

Process a single image through the complete pipeline:

```bash
python pipeline/run_single.py \
  --input path/to/image.jpg \
  --edit "make it look like an oil painting" \
  --output path/to/output/
```

### Batch Processing

Process multiple images with different edit instructions:

```bash
python pipeline/run_batch.py \
  --input path/to/dataset/ \
  --instructions instructions.json \
  --output path/to/outputs/
```

### Parameter Tuning

For InstructPix2Pix editing:
- `--steps`: Number of diffusion steps (default: 100)
- `--cfg-text`: Text condition scale (default: 7.5)
- `--cfg-image`: Image condition scale (default: 1.5)

For WatermarkAnything:
- `--wm-model`: Watermark model selection (mit/coco)
- `--strength`: Watermark strength (default: 1.0)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Submit a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Citation

If you use this code in your research, please cite our work:

```bibtex
@article{your-paper,
  title={Falcon: Text-Based Image Editing with Watermark Robustness},
  author={Your Name},
  journal={arXiv preprint},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Contact

- Project Lead: [Shreyas](mailto:shreyasrd31@gmail.com)
- Project Website: [https://github.com/yourusername/falcon](https://github.com/yourusername/falcon)
## License

This project is licensed under the MIT License.

## Contact

For any questions or feedback, please contact [Shreyas](mailto:shreyasrd31@gmail.com).
