# Plain-DETR

By [Yutong Lin](https://github.com/impiga)\*, [Yuhui Yuan](https://github.com/PkuRainBow)\*, [Zheng Zhang](https://stupidzz.github.io/)\*, [Chen Li](https://github.com/LC-Edward), [Nanning Zheng](http://www.iair.xjtu.edu.cn/info/1046/1229.htm) and [Han Hu](https://ancientmooner.github.io/)\*

This repo is the official implementation of "[DETR Doesn’t Need Multi-Scale or Locality Design](https://openaccess.thecvf.com/content/ICCV2023/html/Lin_DETR_Does_Not_Need_Multi-Scale_or_Locality_Design_ICCV_2023_paper.html)".

## Introduction

We present an improved DETR detector that maintains a “plain” nature: using a single-scale feature map and global cross-attention calculations without specific locality constraints, in contrast to previous leading DETR-based detectors that re-introduce architectural inductive biases of multi-scale and locality into the decoder.

We show that two simple technologies are surprisingly effective within a plain design: 1) a box-to-pixel relative position bias (BoxRPB) term to guide each query to attend to the corresponding object region; 2) masked image modeling (MIM)-based backbone pre-training to help learn representation with fine-grained localization ability and to remedy dependencies on the multi-scale feature maps.

## Main Results
| BoxRPB | MIM PT. | Reparam. | AP | Paper Position | CFG | CKPT |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| ✗ | ✗ | ✗ | 37.2 | Tab2 Exp1 | [cfg](./configs/swinv2_small_sup_pt_ape.sh) | [ckpt](https://msravcghub.blob.core.windows.net/plaindetr-release/plaindetr_models/plaindetr_swinv2_small_sup_pt_ape.pth?sv=2020-04-08&st=2023-11-11T09%3A53%3A39Z&se=2049-12-31T09%3A53%3A00Z&sr=b&sp=r&sig=1ntSPDFHXBDVfYts8aUX8xTxA2kRpZyEQdwZL0tRNSk%3D)
| ✓ | ✗ | ✗ | 46.1 | Tab2 Exp2 | [cfg](./configs/swinv2_small_sup_pt_boxrpe.sh) | [ckpt](https://msravcghub.blob.core.windows.net/plaindetr-release/plaindetr_models/plaindetr_swinv2_small_sup_pt_boxrpe.pth?sv=2020-04-08&st=2023-11-11T09%3A54%3A10Z&se=2023-11-12T09%3A54%3A10Z&sr=b&sp=r&sig=yg4gw7vWX8zlOurS3x2J9%2BPwfsSaHEYYOHE4DJTWw%2BQ%3D) 
| ✓ | ✓ | ✗ | 48.7 | Tab2 Exp5 | [cfg](./configs/swinv2_small_mim_pt_boxrpe.sh) | [ckpt](https://msravcghub.blob.core.windows.net/plaindetr-release/plaindetr_models/plaindetr_swinv2_small_mim_pt_boxrpe.pth?sv=2020-04-08&st=2023-11-11T09%3A52%3A38Z&se=2049-12-31T09%3A52%3A00Z&sr=b&sp=r&sig=eX%2FNgca78ccyBhlujtCSh1BDHiPPOjjceyrMMLxKgr8%3D)
| ✓ | ✓ | ✓ | 50.9 | Tab2 Exp6 | [cfg](./configs/swinv2_small_mim_pt_boxrpe_reparam.sh) | [ckpt](https://msravcghub.blob.core.windows.net/plaindetr-release/plaindetr_models/plaindetr_swinv2_small_mim_pt_boxrpe_reparam.pth?sv=2020-04-08&st=2023-11-14T07%3A42%3A25Z&se=2049-12-31T07%3A42%3A00Z&sr=b&sp=r&sig=5r09k4tFNIO%2FURYIQ2RbjJOU7v4dWqFW1D3F%2Bdg%2FYq0%3D)

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management with Python 3.12 and PyTorch 2.10.

```bash
# install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# clone and install
git clone https://github.com/impiga/Plain-DETR.git
cd Plain-DETR
uv sync
```

To run scripts within the managed environment:

```bash
uv run main.py ...
```

## Usage

### Dataset preparation

Please download [COCO 2017 dataset](https://cocodataset.org/) and organize them as following:

```
data/
└── coco/
    ├── train2017/
    ├── val2017/
    └── annotations/
    	├── instances_train2017.json
    	└── instances_val2017.json
```

By default the training script looks for datasets under `./data`. To use a
different location pass `--data_dir`:

You can also override the COCO path directly with `--coco_path /path/to/coco`.

### Pretrained models preparation

Please run the following script to download supervised and mask-image-modeling pretrained models.

(We adopts Swin Transformer v2 as the default backbone. If you are interested in the pretraining, please refer to Swin Transformer v2 ([paper](https://arxiv.org/abs/2111.09883), [github](https://github.com/microsoft/Swin-Transformer)) and SimMIM ([paper](https://arxiv.org/abs/2111.09886), [github](https://github.com/microsoft/SimMIM)) for more details.)

```
bash scripts/prepare_pt_model.sh
```

### Training

#### Training on single node

```
GPUS_PER_NODE=<num gpus> ./scripts/run_dist_launch.sh <num gpus> <path to config file>
```

#### Training on multiple nodes

On each node, run the following script:

```
MASTER_ADDR=<master node IP address> GPUS_PER_NODE=<num gpus> NODE_RANK=<rank> ./scripts/run_dist_launch.sh <num gpus> <path to config file> 
```

### Evaluation

To evalute a plain-detr model, please run the following script:

```
 <path to config file> --eval --resume <path to plain-detr model>
```

You could also use `./scripts/run_dist_launch.sh` to evaluate a model on multiple GPUs.

## Limitation & Discussion
 - While we have eliminated multi-scale designs for the backbone output and decoder input, the generation of proposals still depends on multi-scale features.

   We have performed trials utilizing single-scale features for proposals(not included in the paper), but it led to ~1 mAP performance drop.

## Known issues

- Most of our experiments are conducted on 16 GPUs with 1 image per GPU. We have tested our released checkpoints with larger batch size and found that the performance of first three models drops significantly.

  We are now reviewing our implementation and will update our code to support larger batch size for both training and inference.

---

## Customizations for Zenseact Open Dataset (ZOD)

This fork adds ZOD support for evaluating DINOv3-pretrained backbones on autonomous driving data.

### ZOD-specific files

| File | Description |
|------|-------------|
| `plain_detr/datasets/zod.py` | ZOD dataset class — loads `single_frames` JSON, 3 classes (Pedestrian/Vehicle/VulnerableVehicle), sky/ego-vehicle removal crop, random 800×800 crop |
| `plain_detr/datasets/zod_eval.py` | COCO-format evaluator for ZOD predictions |
| `configs/zod_dinov3_vit_small.sh` | Training config: DINOv3 ViT-S backbone, batch 12, 28 workers, Box-RPE decoder |

### ZOD preprocessing pipeline

1. Original full-frame → sky crop (remove top 400px) + ego-vehicle crop (remove bottom 568px)
2. Fixed resize to 1333×800
3. Random 800×800 crop with bounding box filtering
4. Normalization: ImageNet stats (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

### Training on ZOD

```bash
# Set dataset path
export ZOD_DATA_DIR=/path/to/zod

# Run ZOD config
./scripts/run_config.sh 1 configs/zod_dinov3_vit_small.sh
```

Or directly:

```bash
uv run torchrun --standalone --nproc-per-node=1 -m plain_detr.main \
    --args.dataset_file zod \
    --args.zod_path $ZOD_DATA_DIR
```

### Visualizations

- `visualizations/ZOD/visualize_zod_crops_tensors.py` — step-by-step visualization of the ZOD transform pipeline
- `visualizations/COCO/inference.py` — COCO inference with good/bad detection examples

---

## Citing Plain-DETR

If you find Plain-DETR useful in your research, please consider citing:
```
inproceedings{lin2023detr,
  title={DETR Does Not Need Multi-Scale or Locality Design},
  author={Lin, Yutong and Yuan, Yuhui and Zhang, Zheng and Li, Chen and Zheng, Nanning and Hu, Han},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6545--6554},
  year={2023}
}
```