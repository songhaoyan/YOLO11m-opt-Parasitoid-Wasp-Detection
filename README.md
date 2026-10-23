# YOLO11m-opt: Parasitoid Wasp Detection and Identification

This repository provides the source code, trained model weights, and a demonstration dataset for **YOLO11m-opt**, an optimized YOLO11m-based framework for automated detection and taxonomic identification of parasitoid wasps in microscopy images.

The repository accompanies our research manuscript on automated parasitoid wasp identification using an optimized single-stage object detector.

## 1. Model Overview

YOLO11m-opt is developed based on the Ultralytics YOLO11m architecture and incorporates the following modifications:

- **P2 detection head:** An additional high-resolution detection branch with a stride of 4 for improved representation of small targets.
- **SPD-Conv:** Space-to-depth convolution for preserving and processing spatial information.
- **DySample:** Dynamic feature upsampling.
- **CBAM:** Convolutional Block Attention Module for channel and spatial attention.
- **WIoUv3:** Wise-IoU v3 bounding-box regression loss.

The model uses 105 predefined taxonomic output labels, including 38 terminal biological classes and additional higher-rank taxonomic labels.

These labels are implemented within a unified YOLO detection framework without an explicit hierarchical classification head.

## 2. Repository Structure

```text
YOLO11m-opt-Parasitoid-Wasp-Detection/
├── README.md
├── requirements.txt
├── train.py
├── reconstructed_yolo11m/
│   ├── __init__.py
│   ├── integration.py
│   ├── modules.py
│   └── wiou.py
├── models/
│   ├── yolo11m_opt.yaml
│   └── yolo11m_p2.yaml
├── weight/
│   ├── best.pt
│   └── last.pt
└── demo_dataset

The filenames shown in `demo_dataset/images/` illustrate the intended organization. The actual image filenames may differ.

## 3. Pretrained Model Weights

Two trained YOLO11m-opt model checkpoints are provided:

| File | Description |
|---|---|
| `weight/best.pt` | Model checkpoint selected according to validation performance; recommended for inference |
| `weight/last.pt` | Checkpoint saved at the end of training |

The checkpoints correspond to the optimized YOLO11m architecture with 105 taxonomic output labels.

Loading these checkpoints may require the custom modules provided in `reconstructed_yolo11m/`.

## 4. Demonstration Dataset

A small demonstration dataset is provided to allow users to explore the model and perform example predictions.

The demonstration dataset contains **38 microscopy images**, with **one image randomly selected from each of the 38 terminal biological classes** represented in the study.

The images are intended for:

- Testing model loading and inference.
- Visualizing predicted bounding boxes and taxonomic labels.
- Exploring the model's behavior on representative parasitoid wasp specimens.
- Demonstrating the basic workflow of automated parasitoid wasp identification.

### Important Notes

The demonstration dataset is a small sample of the research dataset and is provided solely for demonstration and trial use.

**It is not an independent external test dataset and must not be treated as a benchmark for reproducing the performance metrics reported in the manuscript.**

Because only one image per biological class is included, the demonstration dataset cannot adequately characterize within-class variation, model robustness, or generalization performance.

The images were randomly selected rather than selected on the basis of model prediction performance.

## 5. Full Dataset Availability

The complete microscopy dataset used for model development and evaluation is not publicly released because of research data access restrictions.

Instead, this repository provides a limited demonstration subset containing one randomly selected image from each terminal biological class.

The released demonstration subset is not sufficient to reproduce the complete training process, the experimental comparisons, or the quantitative results reported in the manuscript.

Requests for additional research data may be directed to the corresponding author and will be considered subject to applicable access restrictions.

## 6. Installation

Clone the repository:

```bash
git clone https://github.com/<YOUR_USERNAME>/YOLO11m-opt-Parasitoid-Wasp-Detection.git
cd YOLO11m-opt-Parasitoid-Wasp-Detection
```

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The framework is implemented using PyTorch and Ultralytics. Users should ensure that their PyTorch and CUDA installations are compatible with their operating system and hardware.

## 7. Example Inference

The following example demonstrates how to load the trained checkpoint and perform inference using one of the demonstration images.

```python
from ultralytics import YOLO
from reconstructed_yolo11m import register_model_modules

register_model_modules()

model = YOLO("weight/best.pt", task="detect")

results = model.predict(
    source="demo_dataset/images",
    imgsz=960,
    conf=0.25,
    save=True
)
```

Predictions are generated using the 105-label output space defined during training.

The code assumes that the custom modules can be successfully registered using the implementation supplied in this repository.

## 8. Training Configuration

| Parameter | Configuration |
|---|---|
| Model | YOLO11m-opt |
| Taxonomic output labels | 105 |
| Terminal biological classes | 38 |
| Input resolution | 960 × 960 |
| Maximum training epochs | 400 |
| Optimizer | SGD |
| Random seed | 42 |
| Detection scales | P2, P3, P4, P5 |
| Bounding-box regression loss | WIoUv3 |

Detailed computational environment information, software versions, data augmentation settings, and training hyperparameters are documented in Supplementary Table S2 of the accompanying manuscript.

The training script is provided for transparency and methodological inspection. Reproducing the original training results requires access to the full research dataset and its corresponding annotations.

## 9. Intended Use and Limitations

YOLO11m-opt is intended for research applications involving parasitoid wasp detection, preliminary specimen screening, and computer-assisted taxonomic identification.

The model operates within a predefined set of taxonomic labels and does not explicitly identify previously unseen taxa.

Its robustness across different illumination conditions, microscope configurations, geographic regions, and unseen species has not been comprehensively established.

Model predictions should not be considered a substitute for verification by qualified taxonomic experts.

## 10. Citation

If you use this repository or its pretrained model weights, please cite the accompanying research article.

The full citation and DOI will be added following publication.

## 11. Acknowledgments

This work builds upon the Ultralytics YOLO framework and incorporates existing methodological components, including SPD-Conv, DySample, CBAM, and Wise-IoU.

Please refer to the accompanying manuscript for the original references and methodological details.

## 12. License

The applicable licensing terms for the source code, trained weights, and demonstration images will be clarified in accordance with the relevant third-party software licenses and research data access conditions.
