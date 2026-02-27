# Dermafinder AI Core
**Role:** AI Lead & Technical Pitcher
**Event:** 24-Hour Hackathon Sprint (Innovation Labs)
**Objective:** Medical-grade classification of skin lesions with focus on False Negative reduction under hardware constraints.

[![Dermafinder Demo](https://img.youtube.com/vi/5BMkCJC4iZI/maxresdefault.jpg)](https://www.youtube.com/watch?v=5BMkCJC4iZI)

## Hackathon Constraints & Performance Optimization
Built in under 24 hours with limited compute resources (Google Colab Free Tier).
* **Mixed Precision ($float16$):** Implemented to maximize GPU throughput on NVIDIA Tesla T4. By using 16-bit precision for matrix multiplications while keeping 32-bit for critical stability, training speed was increased by ~2x and VRAM usage was halved.
* **Rapid Delivery:** Managed the full AI pipeline (Data -> Architecture -> TTA) in a single-day sprint, including the final technical pitch and jury demonstration.

## Technical Problem
In clinical diagnostics, missing a malignant lesion (False Negative) is the highest risk factor. Standard models optimized for overall accuracy often fail to prioritize high-stakes classes like Melanoma.

## Dataset
The model was trained on the HAM10000 dataset (Human Against Machine with 10000 training images).
* **Source:** [Kaggle - HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
* **Scope:** Extracted and balanced 4 specific classes: BCC, BKL, MEL, and NV.
* **Pre-processing:** Images resized to 224x224 and normalized for the EfficientNetB0 architecture.

## Engineering Solutions

### 1. Real-World Robustness (Data Augmentation)
To handle low-quality user uploads (poor lighting/angles), I implemented heavy data augmentation during training:
* **Invariance Training:** Random flips, 20% rotations, and zoom distortions were applied.
* **Outcome:** Forced the model to learn morphological features of lesions rather than pixel-level orientation.

### 2. False Negative Minimization
Implemented a biased loss strategy to penalize errors on life-threatening conditions:
* **Weighted Cross-Entropy:** 3.0x weight multiplier applied to the Melanoma class.
* **Impact:** Prioritizes sensitivity (recall) for malignancy over generic accuracy.

### 3. Inference via TTA (Test Time Augmentation)
Reduced prediction variance using a 5-view consensus:
* For every upload, the engine generates 4 distorted versions of the image.
* Final diagnosis is the mathematical average of all 5 predictions.

## Tech Stack
* **Architecture:** EfficientNetB0 (Transfer Learning).
* **Framework:** TensorFlow / Keras.
* **Optimization:** Mixed Precision (float16) / XLA Compiler.

## Repository Structure
* train.py: Dual-phase training (Coarse + Fine-tuning).
* evaluate.py: TTA-based validation and confusion matrix generation.
* predict.py: CLI tool for single-image diagnosis.

## Note on Collaboration
This repository covers the AI engine. The frontend (React) and backend (Flask) were developed by the project team.
