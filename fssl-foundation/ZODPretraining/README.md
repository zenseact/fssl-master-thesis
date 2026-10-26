# ZODPretraining — Self-Supervised DINO Pre-training on Zenseact Open Dataset

This repository implements **self-supervised DINO (DIstillation with NO labels)** pre-training on the [Zenseact Open Dataset (ZOD)](https://zod.zenseact.com/) for autonomous driving perception. It supports both **centralized** (single-node multi-GPU) and **federated** (via NVIDIA FLARE) training pipelines, with extensive support for non-IID data splits and configuration tracking.

---

## Project Structure

```
ZODPretraining/
├── Dockerfile              # Container definition
└── src/
    ├── centralized_main_dino.py       # Centralized (non-federated) DINO training with DDP
    ├── config.py                      # Central hyperparameter configuration
    ├── convert_checkpoint.py          # Fixes checkpoint compatibility (numpy → tensor)
    ├── dino_wrapper.py                # Reusable DINO class and training loop for FL clients
    ├── download_DINOv1_weights.py     # Downloads official DINOv1 ResNet50 backbone weights
    ├── execute_centralized_training.sh # SLURM launcher for centralized training (4 GPUs)
    ├── execute_job_with_sampling.sh   # SLURM launcher for FL with client sampling (16 clients)
    ├── execute_job.sh                 # SLURM launcher for small-scale FL (up to 4 clients)
    ├── fedavg_with_checkpoints.py     # FedAvg controller with periodic checkpointing
    ├── fix_pth_file.py                # Prepends "backbone." to state-dict keys for alignment
    ├── job_with_sampling.py           # NVFlare job config for large client pools (sampling)
    ├── job.py                         # NVFlare job config for small-scale FL
    ├── load_zod_metadata.py           # Scans ZOD JSON metadata and caches for non-IID splits
    ├── main_dino.py                   # Federated client DINO training script
    ├── plot_training.py               # Plot training curves (loss, LR, accuracy)
    ├── train_resnet50.sh              # Quick launcher: 10-epoch ResNet50 DINO training
    ├── utils.py                       # DINO core utilities (loss, augmentations, helpers)
    ├── vision_transformer.py          # ViT architectures (tiny/small/base) + DINO head
    ├── visualize_augmentations.py     # Visualize DINO augmentation pipeline
    ├── zod_dataset.py                 # ZOD PyTorch Dataset wrapper
    ├── zod_metadata_cache.json        # Pre-computed metadata cache for non-IID splits
    └── saved_models/                  # Pre-trained checkpoints (centralized & FL)
```

---

## File Descriptions

### Training Scripts

| File | Description |
|------|-------------|
| `centralized_main_dino.py` | Standalone DINO training using multi-GPU DDP on ZOD. Supports ResNet and ViT backbones, wandb logging, and checkpointing. |
| `main_dino.py` | Federated client training script — receives global model weights via NVFlare, performs local DINO self-supervised training, and returns updated weights. Handles IID/non-IID splits, multi-GPU, and wandb logging. |
| `dino_wrapper.py` | Wraps DINO components (student/teacher, loss, training loop) into a reusable `Dino` class used by both centralized and FL clients. Hyperparameters are read from `config.py`. |

### Federated Learning (NVFlare)

| File | Description |
|------|-------------|
| `job.py` | NVFlare job definition for small-scale FL (≤4 clients, no sampling). Uses `FedAvgWithCheckpoints` with `simulator_run()`. |
| `job_with_sampling.py` | NVFlare job definition for large client pools (e.g., 16 clients) with per-round client sampling (e.g., 4 per round). Exports to a POC workspace. |
| `fedavg_with_checkpoints.py` | Custom `FedAvg` subclass that adds periodic numbered checkpoint saving every N rounds. Used by both job configurations. |

### SLURM / Batch Launchers

| File | Description |
|------|-------------|
| `execute_job.sh` | SLURM batch script for small-scale FL (4 clients, no sampling). Binds Apptainer container and runs `job.py`. |
| `execute_job_with_sampling.sh` | SLURM batch script for FL with 16 clients and sampling. Sets up NVFlare POC workspace and runs `job_with_sampling.py`. |
| `execute_centralized_training.sh` | SLURM batch script for centralized DINO training on 4 GPUs via `torch.distributed.run`. |
| `train_resnet50.sh` | Minimal shell script to run 10-epoch ResNet50 DINO training with 4 GPUs. |

### Data & Dataset

| File | Description |
|------|-------------|
| `zod_dataset.py` | PyTorch `Dataset` for ZOD — reads key camera frames, applies transforms, returns PIL images. |
| `load_zod_metadata.py` | Scans ZOD frame JSONs for metadata (country, weather, time of day, road type) and caches results for non-IID split creation. |
| `zod_metadata_cache.json` | Pre-computed metadata cache file (output of `load_zod_metadata.py`). |
| `visualize_augmentations.py` | Applies DINO augmentations (global + local crops) to a sample image and plots the results for debugging. |

### Configuration & Utilities

| File | Description |
|------|-------------|
| `config.py` | Central hyperparameter file (data paths, augmentation scales, optimizer settings, etc.). |
| `utils.py` | Core DINO utilities adapted from Facebook Research: `MultiCropWrapper`, `DINOLoss`, cosine LR schedules, `DataAugmentationDINO`, `MetricLogger`, and distributed helpers. |
| `vision_transformer.py` | ViT architectures (`vit_tiny`, `vit_small`, `vit_base`) and `DINOHead` projection head, adapted from `timm`. Used when `--arch vit_*` is selected. |

### Checkpoint Utilities

| File | Description |
|------|-------------|
| `download_DINOv1_weights.py` | Downloads official DINOv1 ResNet50 weights from torch hub and saves the backbone (stripped of classifier head). |
| `fix_pth_file.py` | Prepends `"backbone."` to all state-dict keys (excluding `fc.`), aligning centralized checkpoint format with FL pipelines. |
| `convert_checkpoint.py` | Converts numpy scalars/arrays in saved checkpoints to PyTorch tensors for cross-environment compatibility. |

### Visualization & Analysis

| File | Description |
|------|-------------|
| `plot_training.py` | Reads a JSON-lines log and generates a 3-panel plot (loss, LR, accuracy over epochs). |

---

## Container

`Dockerfile` builds on `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` (JetPack 5.1.2 / PyTorch 2.0 for Jetson AGX Orin). It installs HDF5 support and Python dependencies. The container is used via Apptainer/Singularity on the HPC cluster.

---

## Saved Models

```
saved_models/
├── centralized_checkpoints/500_epochs/     # Centralized training checkpoints
├── non-IID_split/
│   ├── country_code/                       # FL checkpoints — split by country
│   ├── road_type/                          # FL checkpoints — split by road type
│   ├── scraped_weather/                    # FL checkpoints — split by weather
│   └── time_of_day/                        # FL checkpoints — split by time of day
├── random_split/
│   ├── 100-percent/                        # FL checkpoints — 100% data, IID split
│   ├── 50-percent/                         # FL checkpoints — 50% data, IID split
│   └── 25-percent/                         # FL checkpoints — 25% data, IID split
└── public_DINO_resnet50_weights_FIXED.pth  # Pre-downloaded public DINO weights
```

---

## Quick Start

```bash
# Centralized training (ResNet50, 4 GPUs)
bash src/train_resnet50.sh

# Federated training (small-scale, 4 clients)
bash src/execute_job.sh
```

See individual script headers and `config.py` for detailed hyperparameter documentation.
