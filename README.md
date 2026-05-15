# Correlation-Guided Progressive Interaction and Gated Fusion for Multi-modal Optical and SAR Remote Sensing Image Object Detection

This repository provides the official implementation for our research on multi-modal optical and SAR remote sensing image object detection. The proposed architecture is built upon the official YOLOv8 framework to achieve robust performance in complex environments.

## Usage Instructions

The implementation follows a modular design based on the Ultralytics YOLOv8 codebase. Please follow the steps below to set up and use the code.

### 1. Environment Configuration
First, download the official YOLOv8 source code from the following address: **[YOLOv8 Official Repository](https://github.com/ultralytics/ultralytics)**. Follow the official installation guide to configure the necessary Python environment and dependencies.

### 2. Implementation Details
Our method involves several key modifications to the standard YOLOv8 architecture to support multi-modal data fusion:

* **Dataset Loading Mechanism:** We modified the data pipeline to support the synchronized loading of registered optical and SAR image pairs.
* **FusionDetection Class:** A new **FusionDetection** class was implemented by extending the base **Detection** class to handle dual-stream feature processing.
* **FusionValid Class:** The **FusionValid** class was rewritten based on the standard **Validator** class to evaluate detection performance on multi-modal datasets.
* **Multi-modal Fusion Module:** This includes the source code for our proposed fusion module and the necessary logic for feature decoding.
* **Forward Propagation:** The forward function of the model was refined to integrate the fusion module and ensure correct feature flow between the heterogeneous branches.

## Dataset Access

Experiments were conducted on two primary multi-modal benchmarks. You can access the datasets via the following links:

| Dataset | Access Link |
| :--- | :--- |
| **OGSOD** | [Link to OGSOD Dataset](XXX) |
| **M4SAR** | [Link to M4SAR Dataset](XXX) |

## Code Availability

The source code for the **Multi-modal Feature Fusion Module** is currently publicly available in this repository. The remaining components of the architecture namely the full training and evaluation scripts will be made public after the paper is officially accepted by the **IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing**.

## Contact

If you have any questions or would like to discuss this research, please feel free to reach out:
* **Email:** huangzexian20@mails.ucas.ac.cn
