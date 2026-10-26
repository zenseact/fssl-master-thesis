"""
This script uses the FedJob API from NVFlare with FedAvgWithCheckpoints.
It uses POC mode to run a federated training job.
Used for training a number of clients greater than the number of available GPUs, by sampling a subset of clients at each round.
"""


import torch
import torch.nn as nn
import utils
import os

from nvflare.job_config.api import FedJob
from fedavg_with_checkpoints import FedAvgWithCheckpoints
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.app_opt.pt.job_config.model import PTModel
from torchvision import models as torchvision_models

from vision_transformer import DINOHead

class DINOAggregatorModel(utils.MultiCropWrapper):
    def __init__(self):
        # 1. Recreate the ResNet50 backbone
        backbone = torchvision_models.__dict__["resnet50"]()
        embed_dim = backbone.fc.weight.shape[1]
        backbone.fc = nn.Identity()
        
        # 2. Recreate the DINO Head
        head = DINOHead(
            embed_dim,
            out_dim=65536,
            use_bn=False,
            norm_last_layer=False,
        )
        # 3. Pass them to the parent class
        super().__init__(backbone, head)


def main():

    # Hyperparameters
    FL_ROUNDS = 100
    LOCAL_EPOCHS = 5
    NUM_CLIENTS = 16
    CLIENTS_PER_ROUND = 4
    BATCH_SIZE_PER_GPU = 32
    BASE_LR = 0.0005
    NUM_WORKERS = 16
    DATA_FRACTION = 1.0                  # Fraction of the dataset to be used for training in total (e.g., 25% means that with 4 clients each client will use 25%/4 = 6.25% of the total dataset)
    SPLIT_STRATEGY = "country_code"      # can be "random" (for IID split) or {"country_code", "scraped_weather", "time_of_day", "road_type"} for non-IID splits based on the ZOD metadata

    # Training arguments for the local training script (main_dino.py) that will be executed on each client
    zod_data_dir = os.environ.get("ZOD_DATA_DIR")
    train_args = (
        f"--data_path {zod_data_dir} "
        f"--epochs {LOCAL_EPOCHS} "             
        f"--num_rounds {FL_ROUNDS} "
        f"--batch_size_per_gpu {BATCH_SIZE_PER_GPU} "
        f"--lr {BASE_LR} "
        f"--num_workers {NUM_WORKERS} "
        f"--data_fraction {DATA_FRACTION} "
        f"--split_strategy {SPLIT_STRATEGY} "
        f"--num_clients {NUM_CLIENTS} "
        "--arch resnet50 "
        "--global_crops_scale 0.14 1.0 "
        "--local_crops_number 8 "
        "--local_crops_scale 0.05 0.4 "
        "--out_dim 65536 "
        "--drop_path_rate 0.1 "
        "--norm_last_layer False "
        "--warmup_teacher_temp 0.04 "
        "--teacher_temp 0.04 "
        "--warmup_teacher_temp_epochs 1 "
        "--freeze_last_layer 1 "
        "--min_lr 1e-5 "
        "--warmup_epochs 1 "
        "--weight_decay 0.04 "
        "--weight_decay_end 0.4 "
        "--momentum_teacher 0.996 "
        "--clip_grad 3.0 "
        "--timestamp False "
        "--wandb False "
        "--verbose False "
    )   



    project_root = os.environ.get("PROJECT_ROOT")
    server_model = DINOAggregatorModel()
    init_ckpt_path = os.path.join(
        project_root,
        "src/saved_models/non-IID_split/country_code/16_clients/checkpoints/global_round_0050.pth"
    )


    # ------------------
    # Define the job
    # ------------------
    job = FedJob(name="ResNet_DINO_FedAvg_RandomSample", min_clients=CLIENTS_PER_ROUND)
   
   # Server definition
    controller = FedAvgWithCheckpoints(
        checkpoint_interval=5,      # save a checkpoint every 5 rounds
        checkpoint_dir=os.path.join(project_root, "src/saved_models/non-IID_split/country_code/16_clients/checkpoints"),
        num_clients=CLIENTS_PER_ROUND,  
        num_rounds=FL_ROUNDS,
        start_round=50,     # start from round 50 since we are loading a checkpoint from round 50
    )
    job.to_server(controller)

    job.to_server(PTModel(server_model, initial_ckpt=init_ckpt_path))


    # Client definition
    runner = ScriptRunner(
        script="main_dino.py",
        script_args=train_args,
        launch_external_process=True,   # run as subprocess, NOT inline
        launch_once=False,              # only spawn subprocess when sampled
    )
    job.to_clients(runner)


    # -----------------
    # Execute the Job (POC Mode)
    # -----------------
    print(f"Starting NVFlare federated training (POC mode)...")
    print(f"Total pool: {NUM_CLIENTS} clients | Sampling {CLIENTS_PER_ROUND} clients per round.")

    nvflare_job_dir = os.environ.get("NVFLARE_JOB_DIR")
    job.export_job(nvflare_job_dir)
   

if __name__ == "__main__":
    main()