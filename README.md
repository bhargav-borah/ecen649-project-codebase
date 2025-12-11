# ECEN 649 Project Codebase

## Overview
This repository contains the code and experiments for the ECEN 649 project, investigating how fine-tuning affects internal mechanisms in language models, specifically focusing on entity tracking. The codebase is organized into several directories corresponding to different experiments and models (Goat and Qloat).

## Directory Structure

- **`circuit_finetuning/`**: Contains code to replicate the paper sections 5.2, 5.3, and 5.4.
- **`goat-experiment-1/` & `goat-experiment-2/`**: Contains experimental setups for the Goat model at different checkpoints (1750, 1950, 2200).
- **`qloat-experiment-1/` & `qloat-experiment-2/`**: Contains experimental setups for the Qloat model at different checkpoints (1750, 1950, 2200).
- **`goat-checkpoints/` & `qloat-checkpoints/`**: **Note: These are placeholder directories.** You should download the model weights and place them here or update the paths in the scripts accordingly.

## Checkpoints
The model checkpoints used in the experiments are not included in this repository due to size constraints. You can download them from the following links:

- **Qloat Checkpoints**: [Download Link](https://drive.google.com/drive/folders/1lmkUW_rNkegkI7hfo6S2PpFua6nQF6J_?usp=sharing)
- **Goat Checkpoints**: [Download Link](https://drive.google.com/drive/folders/1lUUet6zBKXXzNMNwTAcQRitu-6wm7iS5?usp=sharing)

## Experiments
Each experiment directory (e.g., `goat_exp_1_1750`) contains subdirectories for specific methods:
- **`experiment_1/`**: Path Patching experiments.
- **`experiment_2/`**: Desiderata-based Component Masking (DCM) experiments.
- **`experiment_3/`**: Cross-Model Activation Patching (CMAP) experiments.
- **`attn_knockout/`**: Attention knockout experiments.
- **`data/`**: Dataset and utility scripts.

## Usage
Please refer to the `README.md` files within each subdirectory for specific instructions on running the experiments.
