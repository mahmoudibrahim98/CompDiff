#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs_summarized/logs_v0/0_train_baseline/training_%j.out
#SBATCH --error=logs_summarized/logs_v0/0_train_baseline/training_%j.err
#SBATCH --time=7-00:00:00          # 7 days (adjust as needed)
#SBATCH --nodes=1                  # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4         # CPUs per task (adjust based on your needs)
#SBATCH --gres=gpu:6               # 6 GPUs (matching your training setup)
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
source activate roentgen

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Set environment variables for distributed training
# For single-node multi-GPU, we don't need InfiniBand - use local GPU interconnects
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,
export NCCL_DEBUG=WARN  # Set to INFO for debugging, WARN for less verbose
export NCCL_IB_DISABLE=1  # Disable InfiniBand (not needed for single-node)
export NCCL_P2P_DISABLE=0  # Enable P2P (NVLink/PCIe) for single-node
export NCCL_SHM_DISABLE=0  # Enable shared memory
# Don't set NCCL_SOCKET_IFNAME for single-node - let NCCL auto-detect
# export NCCL_SOCKET_IFNAME=ib0  # Only needed for multi-node with InfiniBand


# Navigate to project directory
cd /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2

# Create logs directory if it doesn't exist
mkdir -p logs_summarized/logs_v0/0_train_baseline

# Config file path (adjust as needed)
CONFIG_FILE="configs_summarized/v0/0_train_baseline.yaml"

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
