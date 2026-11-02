# Scalable Federated Learning for Autonomous Driving with Self Supervision

Code for my Master's thesis at KTH Royal Institute of Technology, carried out at [RISE Research Institutes of Sweden](https://www.ri.se/) in collaboration with [Zenseact](https://zenseact.com/), within the DREAM project.

The thesis studies whether **federated self-supervised learning (FSSL)** can replace centralized self-supervised pre-training of object detection backbones for autonomous driving. Backbones are pre-trained with [DINO](https://github.com/facebookresearch/dino) on unlabelled camera frames from the [Zenseact Open Dataset (ZOD)](https://zod.zenseact.com/), first centrally and then across a simulated vehicle fleet with [NVIDIA FLARE](https://github.com/NVIDIA/NVFlare), and are evaluated by fine-tuning a detector on ZOD.

Click [here](https://www.ri.se/sites/default/files/2026-07/Thesis__Federated_Self_Supervised_Learning.pdf) to access the full thesis report.

---

## Main results

The best federated configuration reaches **25.5% mAP** on ZOD, slightly above the centralized baseline of **24.7% mAP**, under the same end-to-end fine-tuning protocol. With a frozen backbone, which measures the learned representations more directly, the centralized model is still better (24.7% vs. 20.3–21.4%), but the gap is small.

| Phase | Setup | Best AP (%) |
|---|---|---|
| 1 | Centralized ResNet-50 + Faster R-CNN (unfrozen backbone) | 24.7 |
| 1 | Centralized ResNet-50 + Faster R-CNN (frozen backbone) | 24.7 |
| 1 | Centralized ResNet-50 + Plain-DETR | 5.03 |
| 2 | DINOv3 ViT-S + Plain-DETR on COCO | 49.8 |
| 2 | DINOv3 ViT-S + Plain-DETR on ZOD | 9.8 |
| 3 | 4 clients, ResNet-50 + Faster R-CNN (unfrozen backbone) | 25.0 |
| 3 | 4 clients, ResNet-50 + Faster R-CNN (frozen backbone) | 20.3 |
| 3 | 16 clients, ResNet-50 + Faster R-CNN (unfrozen backbone) | **25.5** |
| 3 | 16 clients, ResNet-50 + Faster R-CNN (frozen backbone) | 21.4 |

COCO-style AP@[.50:.95] on the ZOD far crop, three classes (Vehicle, VulnerableVehicle, Pedestrian). Full tables and analysis are in Chapters 4 and 5 of the thesis.

---

## Approach

The experiments are organised in three phases:

1. **CNN baseline (centralized).** Continue DINO pre-training of the public DINOv1 ResNet-50 on ZOD, then fine-tune Faster R-CNN and Plain-DETR for object detection.
2. **ViT baseline (centralized).** Fine-tune Plain-DETR with a DINOv3 ViT-Small backbone, on COCO and on ZOD.
3. **Federated SSL.** Take the best centralized setup (ResNet-50 + Faster R-CNN) and train it with FedAvg in NVIDIA FLARE:
   - 4 clients, all taking part in every round;
   - 16 clients, with 4 sampled per round;
   - IID (random) splits and non-IID splits by country, weather, time of day and road type, built from the ZOD metadata to mimic a real fleet;
   - scalability runs with 100%, 50% and 25% of the training data.

Only the anonymised (blurred) camera frames of ZOD are used. No LiDAR or radar.

---

## Repository structure

```
fssl-master-thesis/
├── fssl-foundation/
│   ├── DINOv1/            # Fork of facebookresearch/dino, used to test DINO and visualise attention
│   ├── ZODPretraining/    # DINO pre-training on ZOD: centralized (DDP) and federated (NVIDIA FLARE)
│   └── evaluation/        # Downstream object detection on ZOD (Faster R-CNN, DETR, ablations)
└── Plain-DETR/            # Fork of impiga/Plain-DETR with ZOD support and a DINOv3 ViT-S backbone
```

Each folder has its own README with file-by-file details:

- [`fssl-foundation/ZODPretraining`](fssl-foundation/ZODPretraining/README.md): pre-training scripts, NVFlare jobs, SLURM launchers, non-IID splits.
- [`fssl-foundation/evaluation`](fssl-foundation/evaluation/README.md): one folder per experiment, shared dataset and COCO-evaluation code.
- [`Plain-DETR`](Plain-DETR/README.md): the detector used in Phases 1 and 2.

---

## Setup

### Software

The FSSL code (`fssl-foundation`) was run with:

| Package | Version |
|---|---|
| Python | 3.10.13 |
| PyTorch | 2.1.0 |
| torchvision | 0.16.0 |
| NVFlare | 2.7.2 |
| NumPy | 1.24.4 |
| ZOD devkit (`zod`) | 0.5.0 |


---

## Usage

### 1. Centralized DINO pre-training

From `fssl-foundation/ZODPretraining/src`:

```bash
# Download the public DINOv1 ResNet-50 weights used as the starting point
python download_DINOv1_weights.py

# 500-epoch run on 4 GPUs (SLURM)
sbatch execute_centralized_training.sh
```

Hyperparameters are set in `config.py` and in the launch script.

### 2. Federated DINO pre-training

```bash
# 4 clients, all clients in every round (FLARE simulator)
sbatch execute_job.sh

# 16 clients, 4 sampled per round (FLARE POC mode)
sbatch execute_job_with_sampling.sh
```

Number of rounds, local epochs, number of clients, data fraction and split strategy are set at the top of `job.py` / `job_with_sampling.py`. `SPLIT_STRATEGY` is `"random"` for an IID split, or one of `"country_code"`, `"scraped_weather"`, `"time_of_day"`, `"road_type"` for non-IID splits. The metadata used for the splits is extracted with `load_zod_metadata.py`.

### 3. Object detection evaluation

Each experiment in `fssl-foundation/evaluation/src` has its own folder, `config.py` and README. For example, for the centralized backbone:

```bash
cd fssl-foundation/evaluation/src/evaluate_resnet_centralized
# set the DINO checkpoint path and options in config.py
python fasterrcnn.py   # fine-tune + evaluate
python eval.py         # evaluate a trained model and save plots
```

The same pattern applies to `evaluate_resnet_fl_4clients`, `evaluate_resnet_fl_16clients`, the ablations and the COCO baseline.

### 4. Plain-DETR on ZOD (DINOv3 ViT-S)

```bash
cd Plain-DETR
bash configs/zod_dinov3_vit_small.sh
```

---

## Notes

- Model checkpoints (`*.pth`) and training outputs are not included in this repository.
- Results were obtained on the RISE ICE cluster (Phases 1 and 2, NVIDIA GTX 1080 Ti / RTX 2080 Ti) and on the MeluXina EuroHPC supercomputer (Phase 3, NVIDIA A100).
- All evaluation experiments use random seed 42 and an 80/20 train/test split.

---

## Citation

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{baldassarre2026fssl,
  author = {Baldassarre, Francesco},
  title  = {Scalable Federated Learning for Autonomous Driving with Self Supervision},
  school = {KTH Royal Institute of Technology},
  year   = {2026},
  url    = {https://www.ri.se/sites/default/files/2026-07/Thesis__Federated_Self_Supervised_Learning.pdf}
}
```
