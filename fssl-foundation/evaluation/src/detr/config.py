import os

train = True
output_path = "detr.pth"
model_checkpoint_path = os.environ.get(
    "MODEL_CHECKPOINT_PATH",
    "/root/projects/fssl-foundation/ZODPretraining/src/outputs/checkpoint.pth"
)
datapath = os.environ.get(
    "ZOD_DATA_DIR",
    "/root/zod-dataset/"
)
num_epochs = 20
score_threshold = 0.3
batch_size = 2
num_workers = 4
