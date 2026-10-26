#!/bin/bash -l
#SBATCH --job-name=16-clients-evaluation_COCO_FasterRCNN
#SBATCH --account=<project_account>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --qos=default
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=<project_root>/evaluation/src/evaluate_resnet_fl_16clients/slurm_logs/COCO_run-%j.out
#SBATCH --error=<project_root>/evaluation/src/evaluate_resnet_fl_16clients/slurm_logs/COCO_run-%j.err

# ---- Site-specific configuration (edit or set env vars before running) ----
: "${PROJECT_ROOT:=/}"
: "${CONTAINER_PATH:=/}"

echo "Task 1: Fine-tune of a Faster R-CNN with ZOD pretrained backbone and COCO pretrained heads"
echo "Task 2: Evaluation of the model on ZOD val set"

module load Apptainer

echo "Starting training + evaluation with Apptainer at $(date)"
apptainer exec --nv \
    --bind /mnt/tier2,/project/home \
    "$CONTAINER_PATH" \
    python "$PROJECT_ROOT/evaluation/src/evaluate_resnet_fl_16clients/evaluate_model.py"

echo "Job $SLURM_JOB_ID completed at $(date)"