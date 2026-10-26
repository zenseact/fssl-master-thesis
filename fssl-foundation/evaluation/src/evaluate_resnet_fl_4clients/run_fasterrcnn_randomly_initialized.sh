#!/bin/bash -l
#SBATCH --job-name=Evaluation_random_FasterRCNN
#SBATCH --account=<project_account>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --qos=default
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=<project_root>/evaluation/src/baseline_resnet_fasterrcnn/slurm_logs/random_run-%j.out
#SBATCH --error=<project_root>/evaluation/src/baseline_resnet_fasterrcnn/slurm_logs/random_run-%j.err

# ---- Site-specific configuration (edit or set env vars before running) ----
: "${PROJECT_ROOT:=/}"
: "${CONTAINER_PATH:=/}"

echo "Task 1: Fine-tune of a Faster R-CNN with ZOD pretrained backbone and randomly initialized heads"
echo "Task 2: Evaluation of the model on ZOD val set"

module load Apptainer

echo "Starting training + evaluation with Apptainer at $(date)"
apptainer exec --nv \
    --bind /mnt/tier2,/project/home \
    "$CONTAINER_PATH" \
    python "$PROJECT_ROOT/evaluation/src/baseline_resnet_fasterrcnn/fasterrcnn.py"

echo "Job $SLURM_JOB_ID completed at $(date)"