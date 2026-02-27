#!/bin/bash
#SBATCH --job-name=test_validation
#SBATCH --output=logs_summarized/logs_v0/0_train_baseline/test_%j.out
#SBATCH --error=logs_summarized/logs_v0/0_train_baseline/test_%j.err
#SBATCH --time=7-00:00:00          # 7 days (test can take a while for multiple checkpoints)
#SBATCH --nodes=1                  # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # CPUs per task
#SBATCH --gres=gpu:a100:1               # 8 GPUs for distributed test validation
#SBATCH --mem=150gb                # Memory for test validation (8 processes × models + metrics)
#SBATCH --partition=precisionhealth      # Partition name (adjust to your cluster's partition)
#SBATCH --account=precisionhealth      # Account name (adjust to your cluster's account)
#SBATCH --mail-type=ALL             # Email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mahmoud.ibrahim@vito.be  # Your email (adjust as needed)

# Usage:
#   sbatch SCRIPT_NAME [checkpoints]
#
# Arguments:
#   checkpoints (optional): Comma-separated list of checkpoint numbers (default: 8500,10000)
#                           Example: 30000,50000,70000
#
# Examples:
#   # Use default checkpoints from script creation
#   sbatch SCRIPT_NAME
#
#   # Specify different checkpoints
#   sbatch SCRIPT_NAME 30000,50000,70000

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Load modules
module load CUDA
source activate roentgen

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Verify environment
echo "=========================================="
echo "Environment Setup:"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "=========================================="

# Set environment variables for multi-GPU distributed test validation
# SLURM automatically sets CUDA_VISIBLE_DEVICES when --gres=gpu is used
# Don't override it - let SLURM handle GPU allocation
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1  # Disable InfiniBand (not needed for single-node)
export NCCL_P2P_DISABLE=0  # Enable P2P (NVLink/PCIe) for single-node
export NCCL_SHM_DISABLE=0  # Enable shared memory

# Create logs directory if it doesn't exist
mkdir -p logs_summarized/logs_v0/0_train_baseline

# Navigate to project directory
cd /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2

# Config file path (based on experiment name)
CONFIG_FILE="configs_summarized/v0/0_train_baseline.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Available config files:"
    ls -1 configs_summarized/VERSION/*.yaml 2>/dev/null || echo "  No config files found in configs_summarized/VERSION/"
    exit 1
fi

# Get checkpoints (use provided or default from script creation)
CHECKPOINTS_ARG="${1:-10000}"

# Launch test validation
echo "=========================================="
echo "Starting test validation..."
echo "Config file: $CONFIG_FILE"
echo "Checkpoints: $CHECKPOINTS_ARG"
echo "Note: test_dir must be specified in config file"
echo "=========================================="

# Build accelerate launch command
ACCELERATE_CMD="accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision bf16 \
    roentgenv2/train_code/run_validation_monitor_debug.py \
    --config_file=\"$CONFIG_FILE\" \
    --test_mode \
    --checkpoints=\"$CHECKPOINTS_ARG\" \
    --load_images_from_dir"

# Run test validation with all allocated GPUs
# Using 8 GPUs for faster test validation (distributes image generation across GPUs)
echo "Running command:"
echo "$ACCELERATE_CMD"
echo "=========================================="
eval $ACCELERATE_CMD

# Capture exit code
EXIT_CODE=$?

echo "=========================================="
echo "Test validation finished"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

# Exit with the same code as the test script
exit $EXIT_CODE
