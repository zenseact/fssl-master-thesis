# Evaluation — Downstream Object Detection on ZOD

Evaluates DINO self-supervised representations from `ZODPretraining` by fine-tuning Faster R-CNN / DETR on the [Zenseact Open Dataset (ZOD)](https://zod.zenseact.com/).

---

## Structure

```
evaluation/
├── Dockerfile              # Container (same base as ZODPretraining)
├── outputs/                # Placeholder for results
└── src/
    ├── Shared files        # COCO eval, backbone loader, dataset, utils, ViT
    ├── evaluate_resnet_centralized/    # DINO (centralized, 500ep) → Faster R-CNN
    ├── evaluate_resnet_fl_4clients/    # DINO (FL, 4 clients, 500ep) → Faster R-CNN
    ├── evaluate_resnet_fl_16clients/   # DINO (FL, 16 clients, 250ep) → Faster R-CNN
    ├── evaluate_not_pretrained_resnet/ # Ablation: public DINOv1 weights (no ZOD SSL)
    ├── evaluate_short_fine_tuning/     # Ablation: short fine-tuning schedule
    ├── baseline_commercial_coco/       # COCO-pretrained Faster R-CNN (no DINO)
    ├── detr/                           # DETR with optional DINO backbone
    ├── other_experiments/vit_fasterrcnn/ # ViT-Small + Faster R-CNN head
    ├── visualization/                  # Bbox plotting & attention visualization
```

---

## Experiments

| Experiment | Backbone | Pre-training | Fine-tune | config key |
|------------|----------|-------------|-----------|------------|
| `evaluate_resnet_centralized` | ResNet-50 | DINO on ZOD (centralized, 500ep) | detection head only (frozen backbone) | `centralized` |
| `evaluate_resnet_fl_4clients` | ResNet-50 | DINO on ZOD (FL, 4 clients, 500ep) | detection head only | `fl_4clients` |
| `evaluate_resnet_fl_16clients` | ResNet-50 | DINO on ZOD (FL, 16 clients, 250ep) | detection head only | `fl_16clients` |
| `evaluate_not_pretrained_resnet` | ResNet-50 | Public DINOv1 (ImageNet) | full fine-tune | `not_pretrained` |
| `evaluate_short_fine_tuning` | ResNet-50 | Public DINOv1 (ImageNet) | full fine-tune, shorter schedule | `short_fine_tune` |
| `baseline_commercial_coco` | ResNet-50 | COCO (supervised) | full fine-tune | `coco_baseline` |
| `detr` | ResNet-50 / ViT | COCO or DINO | full fine-tune | `detr` |
| `vit_fasterrcnn` | ViT-Small | DINO on ZOD | detection head only | `vit_fasterrcnn` |

---

## Shared Files (`src/`)

| File | Description |
|------|-------------|
| `coco_evaluation.py` | COCO-format ground-truth generation, prediction evaluation, metric logging |
| `load_backbone.py` | Load ResNet-50 / ViT backbones from DINO checkpoints (supports NVFlare and DDP formats) |
| `zod_dataset.py` | Dataset classes: `ZODROIFar` (far crop), `ZODROIWide` (wide crop), `ZODRescaled` (no crop) |
| `utils.py` | Augmentations (Gaussian blur, solarization), ViT helpers, distributed training utilities |
| `vision_transformer.py` | ViT-Small / ViT-Base model definitions (adapted from `timm`) |

See `src/README.md` for detailed documentation.

---

## Dataset

- **Zenseact Open Dataset (ZOD)**, `single_frames` split
- **ROI crop**: Far (`ODFarCrop`: left=1284, top=924, 1280×384) — ~25m ahead view
- **Classes**: `Vehicle` (1), `VulnerableVehicle` (2), `Pedestrian` (3)
- **Normalization**: mean=`[0.326, 0.323, 0.330]`, std=`[0.113, 0.112, 0.117]`
- **Split**: 80/20 train/val per experiment

---

## Container

`Dockerfile` builds on `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` (JetPack 5.1.2 / PyTorch 2.0). The same container used in `ZODPretraining` supports this evaluation module.

---

## Notes

- Evaluation uses COCO-style AP metrics (AP, AP50, AP75, APs, APm, APl)
- Random seed 42 for all experiments
- See experiment subfolder READMEs for per-experiment results
