# WGAN-GP for Galaxy Image Generation

This project implements a Wasserstein GAN with Gradient Penalty (WGAN-GP) using TensorFlow/Keras to generate realistic galaxy images based on the Galaxy10_DECals dataset.

---

## Overview

The model consists of:

- A **Critic** (discriminator) network
- A **Generator** network
- Custom **training loop** using WGAN-GP loss
- Optional GPU support
- Callbacks for **model checkpointing**, **TensorBoard**, and **sample image generation**

---

## Dataset

This project uses the [Galaxy10_DECals](https://astronn.readthedocs.io/en/latest/galaxy10.html) dataset in `.h5` format. The dataset should be stored in:

```
/content/drive/My Drive/Sem2/Topics_in_ML/Project/Galaxy10_DECals1.h5
```

It contains:

- `images`: RGB galaxy images
- `ans`: integer labels (converted to one-hot)

---

## Setup

### Requirements

- TensorFlow 2.x
- NumPy
- h5py
- scikit-learn
- Matplotlib

Install them via pip:

```bash
pip install tensorflow numpy h5py scikit-learn matplotlib
```

---

## Usage

### 1. GPU Check and Optimizer Setup

The script checks for GPU availability and disables TensorFlow's meta optimizer for better debugging and performance consistency.

### 2. Load and Preprocess Data

- Images and labels are loaded from the `.h5` file.
- Labels are converted to one-hot encoding.
- Images are normalized to [-1, 1] and resized to 64x64.

### 3. Visualizations

Functions are provided to visualize batches of galaxy images.

### 4. Model Architecture

- The **Critic** is a 5-layer CNN that outputs a single scalar.
- The **Generator** is a deconvolutional network that maps a 128D latent vector to a 64x64x3 image.

### 5. Training

The `WGANGP` class defines the training logic, including:

- WGAN loss
- Gradient penalty
- Multiple critic steps per generator step

Training is run for `200` epochs with model checkpointing and optional TensorBoard logging.

---

## Outputs

- Trained generator and critic models are saved under:

```
/content/drive/My Drive/Sem2/Topics_in_ML/Project/models/
```

- Sample images (optional) can be saved per epoch by enabling the `ImageGenerator` callback.

---

## Customization

- Adjust model parameters like `EPOCHS`, `Z_DIM`, `LEARNING_RATE`, and others at the top of the script.
- Enable the `ImageGenerator` callback to visualize progress.

---

## TensorBoard

To visualize training metrics:

```bash
tensorboard --logdir=./logs
```

---

## Model Checkpoints

The model weights are saved after every epoch to:

```
/content/drive/My Drive/Sem2/Topics_in_ML/Project/checkpoint/
```

You can resume training by setting `LOAD_MODEL = True`.

---

## Creds

Based on the Galaxy10 dataset and the WGAN-GP model from the original paper:

> Gulrajani et al., *Improved Training of Wasserstein GANs*, 2017.
