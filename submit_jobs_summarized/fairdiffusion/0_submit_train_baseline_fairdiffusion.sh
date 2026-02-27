#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs_summarized/logs_fairdiffusion/0_train_baseline_fairdiffusion/training_%j.out
#SBATCH --error=logs_summarized/logs_fairdiffusion/0_train_baseline_fairdiffusion/training_%j.err
#SBATCH --time=7-00:00:00          # 7 days (adjust as needed)
#SBATCH --nodes=1                  # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4         # CPUs per task (adjust based on your needs)
#SBATCH --gres=gpu:a100:6               # 6 A100 GPUs (matching your training setup)
#SBATCH --mem=90gb                    # Use all available memory on the node
#SBATCH --partition=precisionhealth      # Partition name (adjust to your cluster's partition)
#SBATCH --account=precisionhealth      # Account name (adjust to your cluster's account)
#SBATCH --mail-type=ALL             # Email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mahmoud.ibrahim@vito.be  # Your email (adjust as needed)

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Load modules if needed (uncomment and adjust for your cluster)
# module load cuda/11.8
# module load python/3.10

module load CUDA

# Initialize conda for non-interactive SLURM (source activate often fails in batch)
if [ -n "$CONDA_EXE" ]; then
    source "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
fi
conda activate roentgen || { echo "ERROR: conda activate roentgen failed"; exit 1; }

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Verify environment (fail early if torch/accelerate missing)
echo "Python: $(which python)"
echo "Python version: $(python --version)"
python -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print("Number of GPUs:", torch.cuda.device_count())' || { echo "ERROR: torch not found in roentgen env"; exit 1; }
command -v accelerate >/dev/null 2>&1 || { echo "ERROR: accelerate not found. Install with: pip install accelerate"; exit 1; }

# Set environment variables for distributed training
# For single-node multi-GPU, we don't need InfiniBand - use local GPU interconnects
# Note: CUDA_VISIBLE_DEVICES is set by SLURM when --gres is used, but we can override if needed
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NCCL_DEBUG=WARN  # Set to INFO for debugging, WARN for less verbose
export NCCL_IB_DISABLE=1  # Disable InfiniBand (not needed for single-node)
export NCCL_P2P_DISABLE=0  # Enable P2P (NVLink/PCIe) for single-node
export NCCL_SHM_DISABLE=0  # Enable shared memory
# Don't set NCCL_SOCKET_IFNAME for single-node - let NCCL auto-detect
# export NCCL_SOCKET_IFNAME=ib0  # Only needed for multi-node with InfiniBand


# Navigate to project directory
cd /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2

# Create logs directory if it doesn't exist
mkdir -p logs_summarized/logs_fairdiffusion/0_train_baseline_fairdiffusion

# Config file path (adjust as needed)
CONFIG_FILE="configs_summarized/fairdiffusion/0_train_baseline_fairdiffusion.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Launch training with accelerate
echo "=========================================="
echo "Starting training..."
echo "Config file: $CONFIG_FILE"
echo "=========================================="

accelerate launch \
    --num_processes=6 \
    --multi_gpu \
    --mixed_precision bf16 \
    roentgenv2/train_code/train.py \
    --config_file="$CONFIG_FILE"

# Capture exit code
EXIT_CODE=$?

echo "=========================================="
echo "Training finished with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

# Exit with the same code as the training script
exit $EXIT_CODE
