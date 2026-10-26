import torch
import numpy as np
import sys

src = sys.argv[1]
dst = sys.argv[2]

print(f"Loading checkpoint from {src}...")
ckpt = torch.load(src, map_location="cpu", weights_only=False)

print(f"Converting {len(ckpt)} keys from numpy to torch tensors...")
converted = {}
for k, v in ckpt.items():
    if isinstance(v, np.ndarray):
        converted[k] = torch.from_numpy(v)
    elif isinstance(v, (np.floating, np.integer)):
        converted[k] = torch.tensor(v.item())
    else:
        converted[k] = v

print(f"Saving converted checkpoint to {dst}...")
torch.save(converted, dst)
print("Done.")