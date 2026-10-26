import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

import config

from load_backbone import *
from zod_dataset import *
from coco_evaluation import *


torch.manual_seed(42)


def create_fasterrcnn(checkpoint_path, num_classes=4, freeze_backbone=True):
    # 1. Load COCO pretrained model (with trained FPN and detection head)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # 2. Replace final prediction layer for 4 classes (Vehicle, VulnerableVehicle, Pedestrian + background)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)

    # 3. If checkpoint provided, replace backbone with DINO-trained ResNet
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print("Loading DINO-trained backbone from:", checkpoint_path)
        resnet_model = load_trained_resnet(checkpoint_path)

        # Freeze or unfreeze backbone based on config
        if freeze_backbone:
            print("Freezing backbone (head-only training)")
            for param in resnet_model.parameters():
                param.requires_grad = False
        else:
            print("Unfreezing backbone (end-to-end training)")

        # Define which layers from ResNet to use for FPN
        return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        in_channels_list = [256, 512, 1024, 2048]

        del resnet_model.fc
        fpn_backbone = BackboneWithFPN(resnet_model, return_layers, in_channels_list, out_channels=256)

        # Replace only the backbone, keep COCO-trained RPN and detection head
        model.backbone = fpn_backbone
    else:
        print("Using COCO pretrained backbone (no DINO checkpoint provided)")

    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def train(model, train_loader, output_path="faster_rcnn.pth", num_epochs=20, eval_interval=None, test_dataset=None, score_threshold=0.5, eval_output="eval_results.txt"):
    """trains a given faster r-cnn model for *num_epochs* and saves it to *output_path*"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("running on ", device)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.003, momentum=0.9, weight_decay=0.0005)

    model.train()

    loss_per_epoch = []
    global_progress = tqdm(range(0, num_epochs), desc=f"Training")
    for epoch in global_progress:
        epoch_loss = 0

        local_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, targets in local_progress:
            optimizer.zero_grad()

            images = list(image.to(device) for image in images)
            targets = [
                {k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)}
                for t in targets
            ]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            losses.backward()

            optimizer.step()
        try:
            print(f"loss for epoch {epoch + 1}: {epoch_loss / len(train_loader)}")
            loss_per_epoch.append(epoch_loss / len(train_loader))
        except Exception as e:
            print(e)

        if eval_interval and (epoch + 1) % eval_interval == 0 and (epoch + 1) != num_epochs and test_dataset is not None:
            model.eval()
            predictions_to_coco(test_dataset, model, score_threshold,
                                eval_output.replace(".txt", f"_epoch{epoch+1}.txt"))
            model.train()

    torch.save(model.state_dict(), output_path)


def eval(model, test_dataset, score_threshold=0.5, output_file="eval_results.txt"):
    predictions_to_coco(test_dataset, model, score_threshold, output_file)


if __name__ == "__main__":
    # Create model architecture
    model = create_fasterrcnn(config.model_checkpoint_path, num_classes=4, freeze_backbone=config.freeze_backbone)

    # Load trained weights if needed
    if config.load_full_fasterrcnn_model:
        # Load full model weights (from state dict saved during training)
        print(f"Loading trained weights from {config.output_path}")
        state_dict = torch.load(config.output_path)
        model.load_state_dict(state_dict)
    elif not config.train:
        # Create model + load weights (for backward compatibility)
        print(f"Loading trained weights from {config.output_path}")
        state_dict = torch.load(config.output_path)
        model.load_state_dict(state_dict)

    ## random weights
    # model = create_fasterrcnn(None, num_classes=4)

    ## faster r-cnn with dino vit backbone
    # vit = load_trained_vit(dino_checkpoint_path="models/vit_small_200_frames_sequences.pth")
    # backbone = ViTFeatureExtractor(vit)
    # model = create_fasterrcnn(backbone)

    transform = T.Compose(
        [
            T.Resize((1000, 1200)),  # height, width - full size for training
            T.ToTensor(),
            T.Normalize(
                mean=[0.326, 0.323, 0.330],  # ZOD normalization
                std=[0.113, 0.112, 0.117],
            ),
        ]
    )

    dataset = ZODROIFar(
        dataset_root=config.datapath,
        type="val",
        transform=transform,
        rescaled_size=(1200, 1000),
    )
    # subset_indices = list(range(50_000))
    # train_dataset = Subset(train_dataset, subset_indices)

    # val dataset
    # test_dataset = ZODROIWide(dataset_root=config.datapath, type="val", transform=transform, rescaled_size=(1200, 1000))

    # train/test split (80/20 with fixed seed for reproducibility)
    random.seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Quick test subsets (configured in config.py)
    if config.quick_train_subset:
        train_dataset = Subset(train_dataset, list(range(config.quick_train_subset)))
        print(f"QUICK TEST: Using {config.quick_train_subset} training samples")
    if config.quick_test_subset:
        test_dataset = Subset(test_dataset, list(range(config.quick_test_subset)))
        print(f"QUICK TEST: Using {config.quick_test_subset} test samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
    )

    if config.train:  # train + evaluate
        print("starting training")
        train(model, train_loader, config.output_path, config.num_epochs, config.eval_interval, test_dataset, config.score_threshold, config.eval_output)

        print("starting final evaluation")
        eval(model, test_dataset, config.score_threshold, config.eval_output)

    else:  # only evaluate (weights already loaded above)
        print("starting evaluation")
        eval(model, test_dataset, config.score_threshold, config.eval_output)
