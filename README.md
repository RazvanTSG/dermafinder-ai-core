# Dermafinder AI Core
**Role:** AI Lead & Architect  
**Objective:** Medical-grade classification of skin lesions with a focus on False Negative reduction.

## Technical Problem
In clinical diagnostics, missing a malignant lesion (False Negative) is the highest risk factor. Standard models optimized for overall accuracy often fail to prioritize high-stakes classes like Melanoma.

## Dataset
The model was trained on the **HAM10000** dataset (Human Against Machine with 10000 training images).
* **Source:** [Kaggle - HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
* **Scope:** Extracted and balanced 4 specific classes for this prototype: BCC, BKL, MEL, and NV.
* **Pre-processing:** All images were resized to 224x224 and normalized for the EfficientNetB0 architecture.

## Engineering Solutions

### 1. Handling Real-World User Input (Data Augmentation)
To ensure the model remains accurate when users upload low-quality or poorly angled photos, I implemented heavy data augmentation during training:
* **Rotations and Distortions:** The model was trained on images subjected to random flips, 20% rotations, and zooms.
* **Goal:** This forces the neural network to learn invariant features of the lesions rather than relying on the orientation or scale of the photo.



### 2. False Negative Minimization
I implemented a biased loss strategy to punish errors on life-threatening conditions more severely than others:
* **Weighted Cross-Entropy:** Applied a 3.0x weight multiplier for Melanoma.
* **Impact:** The model prioritizes sensitivity for malignant classes over general accuracy.

### 3. Inference via TTA (Test Time Augmentation)
To increase prediction stability, the system uses a 5-view consensus:
* For every user upload, the engine generates 4 distorted versions of the same image.
* The final diagnosis is the mathematical average of all 5 predictions.

## Tech Stack
* **Architecture:** EfficientNetB0 (Transfer Learning).
* **Framework:** TensorFlow / Keras.
* **Optimization:** Mixed Precision (float16) for GPU throughput.

## Repository Structure
* train.py: Dual-phase training (Coarse + Fine-tuning).
* evaluate.py: TTA-based validation and confusion matrix generation.
* predict.py: CLI tool for single-image diagnosis.

## Note on Collaboration
This repository covers the AI engine. The frontend (React) and backend (Flask) were developed by the project team.
