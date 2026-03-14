# ZooCAM Plankton Classification – 1st Place Solution

This repository contains our solution to the **3-MD-4040 2026 ZooCAM Challenge**, a private Kaggle competition organized within the **Deep Learning course at CentraleSupélec**.

Our team **Accord Italie France** achieved **1st place** on the final leaderboard.

---

# Competition Overview

The goal of the challenge was to **classify plankton organisms from images captured by a ZooCAM imaging system**.

Automatic plankton classification is a difficult task due to the large variability in morphology and imaging conditions. Modern imaging devices generate extremely large datasets that require automated analysis using machine learning and deep learning techniques.

---

# Dataset

The dataset provided for the challenge contained:

- **1,215,213 images**
- **~1,093,000 training images**
- **86 classes**

The test labels were not available to participants.

## Major challenges of the dataset

### 1. Extreme class imbalance

The dataset was **highly imbalanced**:

- Largest class: **~300,000 images**
- Smallest class: **73 images**

To address this, we used:

- **WeightedRandomSampler**
- sampling weight controlled by **α ∈ [0.25, 0.5]**

This helped reduce the dominance of the largest classes during training.

---

### 2. Highly variable image sizes

Images had **extremely heterogeneous resolutions**:

- minimum: **5 × 5**
- maximum: **1288 × 1288**

Images were therefore **rescaled to a fixed resolution during preprocessing** to allow batch training with CNN architectures.

---

# Evaluation Metric

Submissions were evaluated using **Macro F1-score**, which gives equal importance to each class, including rare plankton species.

---

# Results

Competition statistics:

- **39 participants**
- **14 teams**
- **639 total submissions**

Our team achieved:

| Leaderboard | Score |
|-------------|------|
| Public leaderboard | **0.80506** |
| Private leaderboard | **0.79164** |

🏆 **Final rank: 1st place**

Notably, we achieved this result with **only 15 submissions**, indicating strong offline validation and careful experimentation.

---

# Method

Our solution is based on **training multiple convolutional neural networks from scratch** and combining them through a weighted ensemble.

## Models

We trained the following architectures:

- **ResNet50**
- **EfficientNet-B3**
- **ConvNeXt-Tiny**

All models were trained **from scratch on the ZooCAM dataset**.

---

## Training

Training setup:

- Loss: **CrossEntropyLoss**
- **Label smoothing** between **0.05 and 0.1**
- **WeightedRandomSampler** for class imbalance
- Strong **data augmentation**
- Validation-based model selection

---

## Test Time Augmentation (TTA)

Each model prediction was improved using **Test Time Augmentation**, allowing more robust predictions by averaging outputs across multiple augmented versions of the same image.

---

# Ensemble

The final prediction is obtained through a **weighted ensemble of three models**.

For each sample we compute the **weighted sum of the logits** produced by:

- ResNet50
- EfficientNet-B3
- ConvNeXt-Tiny

The final class prediction is obtained after applying softmax to the aggregated logits.

This ensemble strategy significantly improved performance compared to individual models.

---

# Repository Structure
- analysis/
- outputs/
- src/torchtmpl/
- config-*.yaml


## analysis/

This folder contains exploratory analysis of the dataset used to better understand its characteristics before training.

The analyses include:

- computation of **dataset mean and standard deviation** (images are grayscale)
- **image size distribution analysis**
- **class distribution analysis** to highlight the strong class imbalance

These analyses helped guide preprocessing, sampling strategies, and training design.

---

## outputs/

This folder contains the **CSV prediction files** generated during inference and used for **Kaggle submissions**.

Each file corresponds to the predictions produced by a specific model or ensemble configuration.

---

## src/torchtmpl/

This directory contains the main **training and inference framework** used in the project.

### models/

Implementation of all the **deep learning architectures tested during the competition**, including the final models used in the ensemble:

- ResNet50  
- EfficientNet-B3  
- ConvNeXt-Tiny  

---

### data.py

Handles the **dataset pipeline**, including:

- dataset creation
- dataloaders
- preprocessing
- data augmentation transforms

---

### main.py

Entry points for running the **training and testing pipelines**.

These scripts load the configuration files, initialize models, and start training or inference.

---

### optim.py

Utilities for training optimization:

- **loss functions**
- **optimizers**
- **learning rate schedulers**

---

### utils.py

Training utilities including:

- single epoch training loop
- validation/testing routines
- model checkpoint saving
- logging utilities
- **Test Time Augmentation (TTA)** implementation

---

### model_committee.py

Implements the **ensemble strategy** used for the final submission.

The final prediction is obtained by computing a **weighted sum of the logits** produced by multiple models before applying softmax.

---

## config-*.yaml

Configuration files describing **training and testing setups**.

Each model has its own configuration file specifying:

- model architecture
- hyperparameters
- optimizer and scheduler
- training settings
- inference parameters

# Lien youtube
https://www.youtube.com/@GiorgioBono

### Local experimentation

For a local experimentation, you start by setting up the environment :

```
python3 -m virtualenv venv
source venv/bin/activate
python -m pip install .
```

Then you can run a training, by editing the yaml file, then 

```
python -m torchtmpl.main config-file.yaml train
```

And for testing

```
python main.py config-file.yaml test
```

# Training Results

Training and validation metrics were tracked using **Weights & Biases (wandb)**.

This allowed us to:

- monitor training dynamics
- compare models
- track hyperparameter experiments
- analyze validation performance

Some example training and validation curves are shown below.

