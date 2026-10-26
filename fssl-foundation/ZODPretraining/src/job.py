"""
This script contains the server-side code of the FL training job.
It uses the FedJob API with FedAvgWithCheckpoints and the SimEnv execution environment.
Federated training hyperparameters can be set from here.

NOTE: this script is able to simulate a max of NUM_CLIENTS = number of available GPUs (e.g., if you have 4 GPUs, you can simulate up to 4 clients with this script, due to the SimEnv limitations).
For larger simulations, please refer to job_with_sampling.py, which uses a lower-level API that implements client sampling at each round, allowing simulating more clients than available GPUs.

NOTE 2: you can run this script via the bash file execute_job.sh
"""


import os

import torch
import torch.nn as nn
import utils

from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import ScriptRunner
from fedavg_with_checkpoints import FedAvgWithCheckpoints
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
    NUM_CLIENTS = 4
    MIN_CLIENTS = 4
    BATCH_SIZE_PER_GPU = 192
    BASE_LR = 0.0005
    NUM_WORKERS = 32
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
    )   



    project_root = os.environ.get("PROJECT_ROOT")
    server_model = DINOAggregatorModel()
    init_ckpt_path = os.path.join(
        project_root,
        "src/saved_models/non-IID_split/country_code/4_clients/checkpoints/global_round_0050.pth"
    )
    
   
    job = FedJob(name="ResNet_DINO_FedAvg_Checkpoints", min_clients=MIN_CLIENTS)

        
    controller = FedAvgWithCheckpoints(
        checkpoint_interval=5,
        checkpoint_dir=os.path.join(project_root, "src/saved_models/non-IID_split/country_code/4_clients/checkpoints"),
        num_clients=MIN_CLIENTS,
        num_rounds=FL_ROUNDS,
        start_round=50,     # start from round 50 since we are loading a checkpoint from round 50
    )
    job.to_server(controller)
    job.to_server(PTModel(server_model, initial_ckpt=init_ckpt_path))


    runner = ScriptRunner(
        script="main_dino.py",
        script_args=train_args,
        launch_external_process=True,
    )
    
    job.to_clients(runner)


    
    # # Execute the Job
    print("Starting NVFlare federated simulator...")
    job.simulator_run(
        workspace=os.path.join(project_root, f"src/nvflare_workspace_{SPLIT_STRATEGY}_4_clients"),
        n_clients=NUM_CLIENTS,
        gpu="0,1,2,3",
    )
    



if __name__ == "__main__":
    main()