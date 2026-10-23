# YOLO11m-opt-Parasitoid-Wasp-Detection
# YOLO11m-opt: Parasitoid Wasp Detection and Identification

This repository provides the implementation and trained model weights of **YOLO11m-opt**, an optimized YOLO11m-based object detection framework developed for automated detection and taxonomic identification of parasitoid wasps from microscopy images.

The project accompanies the manuscript *“Automated Identification of Parasitoid Wasps for Biological Control Monitoring Using an Optimized Single-Stage Detector.”*

## 1. Model Overview

YOLO11m-opt builds upon the Ultralytics YOLO11m architecture and integrates several modifications designed to improve feature representation and bounding-box localization for small parasitoid wasps:

- **P2 detection head:** Adds a high-resolution detection branch with a stride of 4.
- **SPD-Conv:** Incorporates space-to-depth convolution for spatial feature processing.
- **DySample:** Introduces adaptive feature upsampling.
- **CBAM:** Applies channel and spatial attention to intermediate features.
- **WIoUv3:** Employs a dynamically weighted bounding-box regression loss.

The model uses **105 taxonomic output labels**, including 38 terminal biological classes and additional labels representing higher taxonomic ranks. The implementation uses a unified YOLO detection framework rather than a dedicated hierarchical classification head.

## 2. Repository Contents

```text
YOLO11m-opt-Parasitoid-Wasp-Detection/
├── README.md
├── reconstructed_yolo11m.py
├── train.py
├── models/
│   └── yolo11m_opt.yaml
└── weights/
    ├── best.pt
    └── last.pt
```

**Model weights**

- `best.pt`: Model checkpoint selected according to validation performance.
- `last.pt`: Model checkpoint saved at the final training epoch.

Both files correspond to the YOLO11m-opt architecture. For standard inference and evaluation, `best.pt` is recommended.

## 3. Training Configuration

The optimized model was configured using the following settings:

| Parameter | Value |
|---|---|
| Base architecture | YOLO11m |
| Number of output labels | 105 |
| Input image size | 960 × 960 |
| Maximum training epochs | 400 |
| Optimizer | SGD |
| Random seed | 42 |
| Bounding-box regression loss | WIoUv3 |
| Detection branches | P2, P3, P4, P5 |

Detailed software versions, computational environment information, augmentation parameters, and training hyperparameters are documented in Supplementary Table S2 of the accompanying manuscript.

## 4. Model Implementation

The custom implementations of SPD-Conv, DySample, CBAM, and WIoUv3 are provided in `reconstructed_yolo11m.py`.

The modified detection architecture is defined in `models/yolo11m_opt.yaml`.

The training procedure is provided in `train.py`.

The model depends on Ultralytics YOLO and PyTorch. Custom modules must be registered before loading model checkpoints when required by the implementation.

## 5. Dataset and Taxonomic Labels

The dataset comprises microscopy images of parasitoid wasps representing 38 terminal biological classes.

Each specimen may have annotations corresponding to multiple taxonomic ranks within the predefined 105-label space.

The model is designed for the taxa represented during training. It does not include an explicit open-set recognition mechanism, and predictions for previously unseen taxa should be interpreted cautiously.

The complete research dataset is not included in this repository. Dataset availability and access restrictions are described in the accompanying manuscript.

## 6. Intended Use and Limitations

YOLO11m-opt was developed as a research tool for parasitoid wasp detection, preliminary specimen screening, and computer-assisted taxonomic identification.

Performance under substantially different illumination conditions, microscopy equipment, geographic locations, and previously unseen taxa has not been comprehensively established.

Model predictions should be verified by qualified taxonomic experts when used for scientific identification.

## 7. Citation

If you use this implementation or its trained weights, please cite the accompanying manuscript:

*Automated Identification of Parasitoid Wasps for Biological Control Monitoring Using an Optimized Single-Stage Detector.*

Full publication details and DOI will be added when available.

## 8. Acknowledgments

This project is based on the Ultralytics YOLO framework. SPD-Conv, DySample, CBAM, and Wise-IoU are existing methods integrated and adapted for the present detection task.

Please refer to the accompanying manuscript for the relevant original references.

## 9. License

Licensing information will be provided after verification of the applicable third-party software licenses and redistribution requirements.
