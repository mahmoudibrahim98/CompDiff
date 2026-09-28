#!/bin/bash
#SBATCH --job-name=generate_train_monitor
#SBATCH --output=logs_summarized/logs_v0/0_train_baseline/generate_train_%j.out
#SBATCH --error=logs_summarized/logs_v0/0_train_baseline/generate_train_%j.err
#SBATCH --time=7-00:00:00          # 2 days (validation can run longer)
#SBATCH --nodes=1                  # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # CPUs per task
#SBATCH --gres=gpu:rtx4090:8               # GPUs for distributed validation
#SBATCH --mem=256gb                # Memory for validation (8 processes × models + metrics)
#SBATCH --partition=ai4health      # Partition name (adjust to your cluster's partition)
#SBATCH --account=ai4health      # Account name (adjust to your cluster's account)
#SBATCH --mail-type=ALL             # Email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mahmoud.ibrahim@vito.be  # Your email (adjust as needed)

# Usage:
#   sbatch submit_generate_train.sh [checkpoints] [continuous] [check_interval] [manifest_file] [load_images_flag]
#
# Arguments:
#   checkpoints (optional): Comma-separated list of checkpoint numbers (default: 10000)
#                           Example: 30000,50000,70000
#   continuous (optional): Set to "1" or "true" for continuous monitoring, "0" or "false" for one-time processing (default: 0)
#   check_interval (optional): Check for new checkpoints every N seconds (default: 300)
#   manifest_file (optional): Manifest file name (default: train_manifest.json)
#   load_images_flag (optional): Set to "1" or "true" to load pre-generated images from train_images directory
#
# Examples:
#   # One-time mode: Process specific checkpoints once
#   sbatch submit_generate_train.sh 30000,50000,70000 0
#
#   # Continuous mode: Monitor for checkpoints matching the list
#   sbatch submit_generate_train.sh 30000,50000,70000 1

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

# Set environment variables for multi-GPU distributed validation
# SLURM automatically sets CUDA_VISIBLE_DEVICES when --gres=gpu is used
# Don't override it - let SLURM handle GPU allocation
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1  # Disable InfiniBand (not needed for single-node)
export NCCL_P2P_DISABLE=0  # Enable P2P (NVLink/PCIe) for single-node
export NCCL_SHM_DISABLE=0  # Enable shared memory

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to project directory
cd /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2

# Config file path (from configs_summarized)
CONFIG_FILE="configs_summarized/v0/0_train_baseline.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Available config files:"
    ls -1 configs_summarized/*.yaml 2>/dev/null || echo "  No config files found in configs_summarized/"
    exit 1
fi

# Parse arguments
CHECKPOINTS_ARG="${1:-10000}"
CONTINUOUS_FLAG="${2:-0}"
CHECK_INTERVAL="${3:-300}"  # Default: 5 minutes
MANIFEST_FILE="${4:-train_manifest.json}"
LOAD_IMAGES_FLAG="${5:-0}"  # Optional: set to "1" or "true" to load pre-generated images

# Determine if continuous mode is enabled
if [ "$CONTINUOUS_FLAG" = "1" ] || [ "$CONTINUOUS_FLAG" = "true" ] || [ "$CONTINUOUS_FLAG" = "True" ]; then
    CONTINUOUS_MODE="Continuous monitoring"
    TRAIN_CONTINUOUS_FLAG="--train_continuous"
else
    CONTINUOUS_MODE="One-time processing"
    TRAIN_CONTINUOUS_FLAG=""
fi

# Launch validation monitor
echo "=========================================="
echo "Starting TRAIN MODE monitoring..."
echo "Config file: $CONFIG_FILE"
echo "Target checkpoints: $CHECKPOINTS_ARG"
echo "Mode: $CONTINUOUS_MODE"
if [ "$CONTINUOUS_MODE" = "Continuous monitoring" ]; then
    echo "Check interval: ${CHECK_INTERVAL}s"
fi
echo "Manifest file: $MANIFEST_FILE"
if [ "$LOAD_IMAGES_FLAG" = "1" ] || [ "$LOAD_IMAGES_FLAG" = "true" ]; then
    echo "Mode: Loading pre-generated images from train_images directory (skipping generation)"
else
    echo "Mode: Generating images from checkpoints on train dataset"
fi
echo "=========================================="

# Build accelerate launch command
ACCELERATE_CMD="accelerate launch \
    --num_processes=8 \
    --num_machines=1 \
    --multi_gpu \
    --mixed_precision bf16 \
    roentgenv2/train_code/run_validation_monitor_debug.py \
    --config_file=\"$CONFIG_FILE\" \
    --train_mode \
    --checkpoints=\"$CHECKPOINTS_ARG\" \
    --check_interval=\"$CHECK_INTERVAL\" \
    --manifest_file=\"$MANIFEST_FILE\""

# Add continuous flag if enabled
if [ -n "$TRAIN_CONTINUOUS_FLAG" ]; then
    ACCELERATE_CMD="$ACCELERATE_CMD $TRAIN_CONTINUOUS_FLAG"
fi

# Add load_images_from_dir flag if provided
if [ "$LOAD_IMAGES_FLAG" = "1" ] || [ "$LOAD_IMAGES_FLAG" = "true" ]; then
    ACCELERATE_CMD="$ACCELERATE_CMD --load_images_from_dir"
fi

# Run validation with all allocated GPUs
# Using 8 GPUs for faster validation (distributes image generation across GPUs)
# Note: When loading from directory, GPUs are only used for metrics computation
eval $ACCELERATE_CMD

# Capture exit code
EXIT_CODE=$?

echo "=========================================="
echo "Generate train monitoring finished"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

# Exit with the same code as the validation script
exit $EXIT_CODE
