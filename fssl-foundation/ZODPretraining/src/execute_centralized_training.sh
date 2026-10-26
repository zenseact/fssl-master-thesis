#!/bin/bash -l
#SBATCH --job-name=Centralized_dino
#SBATCH --account=<project_account>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --qos=default
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --output=<project_root>/src/centralized_%j.out
#SBATCH --error=<project_root>/src/centralized_%j.err

# ---- Site-specific configuration (edit or set env vars before running) ----
: "${PROJECT_ROOT:=/}"
: "${ZOD_DATA_DIR:=/}"
: "${CONTAINER_PATH:=/}"

echo "Loading Apptainer module..."
module load Apptainer

echo "Starting centralized DINO training on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

cd "$PROJECT_ROOT/src"

# Train DINO ResNet50 on ZOD with 4 GPUs using Apptainer
apptainer exec --nv \
    --bind /mnt/tier2,/project/home \
    "$CONTAINER_PATH" \
    python -m torch.distributed.run --nproc_per_node=4 centralized_main_dino.py \
        --data_path "$ZOD_DATA_DIR" \
        --epochs 500 \
        --arch resnet50 \
        --output_dir "$PROJECT_ROOT/src/saved_models/centralized_checkpoints/500_epochs" \
        --saveckp_freq 25 \
        --batch_size_per_gpu 192 \
        --lr 0.0005 \
        --out_dim 65536 \
        --drop_path_rate 0.1 \
        --norm_last_layer False \
        --warmup_teacher_temp 0.04 \
        --teacher_temp 0.04 \
        --warmup_teacher_temp_epochs 1 \
        --freeze_last_layer 1 \
        --min_lr 1e-6 \
        --warmup_epochs 10 \
        --weight_decay 0.04 \
        --weight_decay_end 0.4 \
        --momentum_teacher 0.996 \
        --clip_grad 3.0 \
        --use_fp16 True \
        --num_workers 32 \
        --global_crops_scale 0.14 1.0 \
        --local_crops_number 8 \
        --local_crops_scale 0.05 0.4 \
        --wandb False

echo "Training completed at $(date)"