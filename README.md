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

The GitHub repository contains the source code, model configuration files, and dependency information.

```text
YOLO11m-opt-Parasitoid-Wasp-Detection/
├── README.md
├── requirements.txt
├── train.py
├── train_rebuilt.py
├── reconstructed_yolo11m/
│   ├── __init__.py
│   ├── integration.py
│   ├── modules.py
│   └── wiou.py
└── models/
    ├── yolo11m_opt.yaml
    └── yolo11m_p2.yaml
```

The pretrained model weights and demonstration dataset are distributed separately through GitHub Releases.

After downloading and extracting these resources, the local project directory can be organized as follows:

```text
YOLO11m-opt-Parasitoid-Wasp-Detection/
├── README.md
├── requirements.txt
├── train.py
├── train_rebuilt.py
├── reconstructed_yolo11m/
├── models/
├── weights/
│   ├── best.pt
│   └── last.pt
└── demo_dataset/
    └── images/
        └── [38 demonstration images]
```

## 3. Pretrained Model Weights

Two trained YOLO11m-opt model checkpoints are available through GitHub Releases.

**[Download Pretrained Model Weights](https://github.com/songhaoyan/YOLO11m-opt-Parasitoid-Wasp-Detection/releases/tag/weights)**

| Checkpoint | Description |
|---|---|
| `best.pt` | Model checkpoint selected according to validation performance; recommended for inference |
| `last.pt` | Model checkpoint saved at the end of training |

Both checkpoints correspond to the optimized YOLO11m architecture with 105 predefined taxonomic output labels.

To use the pretrained model:

1. Open the model weights release using the link above.
2. Download `best.pt` and, optionally, `last.pt` from the release assets.
3. Create a folder named `weights` in the repository root.
4. Place the downloaded checkpoint files inside this folder.

Loading the checkpoints requires the corresponding custom modules provided in `reconstructed_yolo11m/`.

## 4. Demonstration Dataset

A demonstration dataset containing **38 microscopy images**, with **one image randomly selected from each of the 38 terminal biological classes**, is provided for trial use.

**[Download Demonstration Dataset](https://github.com/songhaoyan/YOLO11m-opt-Parasitoid-Wasp-Detection/releases/tag/demo_dataset)**

The demonstration images are distributed as `demo_dataset.zip` through GitHub Releases.

To prepare the dataset:

1. Download `demo_dataset.zip` from the release assets.
2. Extract the ZIP archive.
3. Place the extracted images in a local `demo_dataset/images/` directory.

The demonstration dataset is intended for model loading, inference, visualization of predicted bounding boxes, and exploration of the model's predictions.

### Important Notes

The demonstration dataset is provided exclusively for demonstration and trial use.

Ground-truth annotation files are not included.

**The demonstration dataset is not an independent external test dataset and must not be used as a benchmark for reproducing the quantitative performance metrics reported in the manuscript.**

Because only one image per biological class is provided, the demonstration dataset cannot adequately characterize within-class variation, model robustness, or generalization performance.

The demonstration images were randomly selected rather than selected according to prediction performance.

## 5. Full Dataset Availability

The complete microscopy dataset used for model development and evaluation is not publicly released due to research data access restrictions.

Instead, a limited demonstration subset containing one image from each of the 38 terminal biological classes is made available through GitHub Releases.

The complete research dataset, original annotations, and training, validation, and test partitions are not included.

Consequently, the released materials do not enable full reproduction of the original training experiments, experimental comparisons, or reported quantitative evaluation results.

Requests for additional research data may be directed to the corresponding author and will be considered subject to applicable access restrictions.

## 6. Installation

Clone the repository:

```bash
git clone https://github.com/songhaoyan/YOLO11m-opt-Parasitoid-Wasp-Detection.git
cd YOLO11m-opt-Parasitoid-Wasp-Detection
```

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The framework is implemented using PyTorch and Ultralytics. Users should ensure that their PyTorch and CUDA installations are compatible with their operating system and hardware.

Download the pretrained weights and demonstration images from the release links provided in Sections 3 and 4 before running inference.

## 7. Example Inference

After downloading `best.pt` and extracting the demonstration images into the directories described above, run the following Python code:

```python
from ultralytics import YOLO
from reconstructed_yolo11m import register_model_modules

register_model_modules()

model = YOLO("weights/best.pt", task="detect")

results = model.predict(
    source="demo_dataset/images",
    imgsz=960,
    conf=0.25,
    save=True
)
```

The model generates predictions within the predefined 105-label taxonomic output space.

The custom model modules must be registered before loading the checkpoint. The inference example assumes that the required dependencies have been installed and that the downloaded weights and images are located in the specified directories.

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

The training scripts are provided for methodological transparency. Dataset paths must be configured by users who have access to an appropriately annotated dataset.

Because the complete research dataset and its annotations are not publicly available, the original training and quantitative evaluation results cannot be fully reproduced using only the materials in this repository.

## 9. Intended Use and Limitations

YOLO11m-opt is intended for research applications involving parasitoid wasp detection, preliminary specimen screening, and computer-assisted taxonomic identification.

The model operates within a predefined set of taxonomic labels and does not explicitly identify previously unseen taxa.

Its robustness across different illumination conditions, microscope configurations, geographic regions, and unseen taxa has not been comprehensively established.

Model predictions should not be considered a substitute for verification by qualified taxonomic experts.

## 10. Citation

If you use this repository, its source code, or its pretrained model weights, please cite the accompanying research article.

The full citation and DOI will be added following publication.

## 11. Acknowledgments

This work builds upon the Ultralytics YOLO framework and incorporates existing methodological components, including SPD-Conv, DySample, CBAM, and Wise-IoU.

Please refer to the accompanying manuscript for the relevant original references and methodological details.

## 12. License

The applicable licensing terms for the source code, trained weights, and demonstration images will be clarified in accordance with the relevant third-party software licenses and research data access conditions.
