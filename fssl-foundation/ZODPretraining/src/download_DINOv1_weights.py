import torch
print("Downloading DINOv1 weights...")
dino_pretrained = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")

# Clean the state dict
state_dict = {k: v for k, v in dino_pretrained.state_dict().items() if not k.startswith("fc.")}

torch.save(state_dict, "public_DINO_resnet50_weights.pth")
print("Saved to public_DINO_resnet50_weights.pth")