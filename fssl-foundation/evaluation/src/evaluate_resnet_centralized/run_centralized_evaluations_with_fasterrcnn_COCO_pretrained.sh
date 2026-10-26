#!/bin/bash -l
#SBATCH --job-name=cent-eval_COCO_FasterRCNN
#SBATCH --account=<project_account>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --qos=default
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=<project_root>/evaluation/src/evaluate_resnet_centralized/slurm_logs/COCO_run-%j.out
#SBATCH --error=<project_root>/evaluation/src/evaluate_resnet_centralized/slurm_logs/COCO_run-%j.err

# ---- Site-specific configuration (edit or set env vars before running) ----
: "${PROJECT_ROOT:=/}"
: "${CONTAINER_PATH:=/}"

module load Apptainer

echo "Starting training + evaluation with Apptainer at $(date)"
apptainer exec --nv \
    --bind /mnt/tier2,/project/home \
    "$CONTAINER_PATH" \
    python "$PROJECT_ROOT/evaluation/src/evaluate_resnet_centralized/evaluate_model.py"

echo "Job $SLURM_JOB_ID completed at $(date)"