#!/bin/bash -l
#SBATCH --job-name=ZOD_Federated
#SBATCH --account=<project_account>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --qos=default
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --output=<project_root>/src/slurm_logs/FSSL-16_clients-run-%j.out
#SBATCH --error=<project_root>/src/slurm_logs/FSSL-16_clients-run-%j.err

# ---- Site-specific configuration (edit or set env vars before running) ----
: "${PROJECT_ROOT:=/}"
: "${ZOD_DATA_DIR:=/}"
: "${CONTAINER_PATH:=/}"
: "${NVFLARE_POC_WORKSPACE:=/}"
: "${NVFLARE_JOB_DIR:=/}"

# Export for use inside the container heredoc
export PROJECT_ROOT ZOD_DATA_DIR NVFLARE_POC_WORKSPACE NVFLARE_JOB_DIR

module load Apptainer 2>/dev/null || true

echo "Job started on $(hostname) at $(date)"

apptainer exec --nv \
    --bind /mnt/tier2,/project/home \
    "$CONTAINER_PATH" \
    bash << EOF

set -e
cd "\$PROJECT_ROOT/src"

echo "=== 1. Cleanup ==="
rm -rf ~/.nvflare \
       "\$NVFLARE_POC_WORKSPACE" \
       "\$NVFLARE_JOB_DIR"

echo "=== 2. Export Job ==="
python job_with_sampling.py

echo "=== 3. Prepare POC ==="
yes | nvflare poc prepare -n 16

echo "=== 4. Start POC (background, no admin console) ==="
nvflare poc start -ex admin@nvidia.com -gpu 0 1 2 3 &
echo "Waiting 90s for server and 8 clients to come up..."
sleep 90

echo "=== 5. Submit Job ==="
JOB_FOLDER=\$(ls -d "\$NVFLARE_JOB_DIR"/*/ | head -n 1)
echo "Submitting: \$JOB_FOLDER"
nvflare job submit -j "\$JOB_FOLDER"

echo "=== 6. Waiting for training (sleeping for 47h, can be adjusted) ==="
sleep 169200

echo "=== 7. Stop POC ==="
nvflare poc stop

EOF

echo "Job finished at $(date)"