#!/bin/bash
#SBATCH --job-name=validation_monitor
#SBATCH --output=logs_summarized/logs_fairdiffusion/0_train_baseline_fairdiffusion/validation_%j.out
#SBATCH --error=logs_summarized/logs_fairdiffusion/0_train_baseline_fairdiffusion/validation_%j.err
#SBATCH --time=7-00:00:00          # 2 days (validation can run longer)
#SBATCH --nodes=1                  # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # CPUs per task
#SBATCH --gres=gpu:a5000:8               # GPUs for distributed validation
#SBATCH --mem=256gb                # Memory for validation (8 processes × models + metrics)
#SBATCH --partition=ai4health      # Partition name (adjust to your cluster's partition)
#SBATCH --account=ai4health      # Account name (adjust to your cluster's account)
#SBATCH --mail-type=ALL             # Email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mahmoud.ibrahim@vito.be  # Your email (adjust as needed)

# Usage:
#   sbatch submit_validation.sh [config_file] [check_interval] [manifest_file] [load_images_flag]
#
# Arguments:
#   config_file (optional): Path to config file (default: configs_summarized/test_config.yaml)
#   check_interval (optional): Check for new checkpoints every N seconds (default: 1800)
#   manifest_file (optional): Manifest file name (default: validation_manifest.json)
#   load_images_flag (optional): Set to "1" or "true" to load pre-generated images from validation_images directory
#
# Examples:
#   # Normal mode: Monitor checkpoints and generate images
#   sbatch submit_validation.sh configs_summarized/my_config.yaml
#
#   # Load pre-generated images from validation_images directory
#   sbatch submit_validation.sh configs_summarized/my_config.yaml 1800 validation_manifest.json 1

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
echo "=========================================="
echo "Environment Setup:"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
python -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print("Number of GPUs:", torch.cuda.device_count())' || { echo "ERROR: torch not found in roentgen env"; exit 1; }
command -v accelerate >/dev/null 2>&1 || { echo "ERROR: accelerate not found. Install with: pip install accelerate"; exit 1; }
echo "=========================================="

# Set environment variables for multi-GPU distributed validation
# SLURM automatically sets CUDA_VISIBLE_DEVICES when --gres=gpu is used
# Don't override it - let SLURM handle GPU allocation
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1  # Disable InfiniBand (not needed for single-node)
export NCCL_P2P_DISABLE=0  # Enable P2P (NVLink/PCIe) for single-node
export NCCL_SHM_DISABLE=0  # Enable shared memory

# Create logs directory if it doesn't exist
mkdir -p logs_summarized

# Navigate to project directory
cd /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2

# Config file path (adjust as needed)
CONFIG_FILE="${1:-configs_summarized/fairdiffusion/0_train_baseline_fairdiffusion.yaml}"  # Use first argument or default

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Available config files:"
    ls -1 configs_summarized/*.yaml 2>/dev/null || echo "  No config files found in configs_summarized/"
    exit 1
fi

# Optional: Check interval, manifest file, load images flag, and stop-at-step arguments
CHECK_INTERVAL="${2:-300}"  # Default: 30 minutes
MANIFEST_FILE="${3:-validation_manifest.json}"
LOAD_IMAGES_FLAG="${4:-0}"  # Optional: set to "1" or "true" to load pre-generated images
STOP_AT_STEP="${5:-}"      # Optional: if set, stop validation after this global step

# Launch validation monitor
echo "=========================================="
echo "Starting validation monitoring..."
echo "Config file: $CONFIG_FILE"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "Manifest file: $MANIFEST_FILE"
if [ "$LOAD_IMAGES_FLAG" = "1" ] || [ "$LOAD_IMAGES_FLAG" = "true" ]; then
    echo "Mode: Loading pre-generated images from validation_images directory (skipping generation)"
else
    echo "Mode: Generating images from checkpoints"
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
    --check_interval=\"$CHECK_INTERVAL\" \
    --manifest_file=\"$MANIFEST_FILE\""

# Add stop_at_step if provided
if [ -n "$STOP_AT_STEP" ]; then
    ACCELERATE_CMD="$ACCELERATE_CMD --stop_at_step=\"$STOP_AT_STEP\""
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
echo "Validation monitoring finished"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

# Exit with the same code as the validation script
exit $EXIT_CODE
