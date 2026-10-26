#!/bin/bash -l
#SBATCH --job-name=ZOD_Federated
#SBATCH --account=<project_account>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --qos=default
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --output=<project_root>/src/slurm_logs/FSSL-4_clients-run-%j.out
#SBATCH --error=<project_root>/src/slurm_logs/FSSL-4_clients-run-%j.err

# ---- Site-specific configuration (edit or set env vars before running) ----
: "${PROJECT_ROOT:=/}"
: "${ZOD_DATA_DIR:=/}"
: "${CONTAINER_PATH:=/}"

echo "Starting job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "NOTE: to use wandb for monitoring the training, set the argument --wandb True in the job.py script"

module load Apptainer

echo "Executing the training job with Apptainer at $(date)"
apptainer exec --nv \
    --bind /mnt/tier2,/project/home \
    "$CONTAINER_PATH" \
    python "$PROJECT_ROOT/src/job.py"

echo "Job $SLURM_JOB_ID completed at $(date)"