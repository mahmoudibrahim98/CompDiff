#!/bin/bash
#SBATCH --job-name=test_validation
#SBATCH --output=logs_summarized/logs_v7/6_train_hcn_age_from_promt/test_%j.out
#SBATCH --error=logs_summarized/logs_v7/6_train_hcn_age_from_promt/test_%j.err
#SBATCH --time=7-00:00:00          # 7 days (test can take a while for multiple checkpoints)
#SBATCH --nodes=1                  # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # CPUs per task
#SBATCH --gres=gpu:rtx4090:8               # GPUs for distributed test validation
#SBATCH --mem=256gb                # Memory for test validation (8 processes × models + metrics)
#SBATCH --partition=ai4health      # Partition name (adjust to your cluster's partition)
#SBATCH --account=ai4health      # Account name (adjust to your cluster's account)
#SBATCH --mail-type=ALL             # Email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mahmoud.ibrahim@vito.be  # Your email (adjust as needed)

# Usage:
#   sbatch SCRIPT_NAME [checkpoints] [continuous] [check_interval] [manifest_file]
#
# Arguments:
#   checkpoints (optional): Comma-separated list of checkpoint numbers (default: 20000)
#                           Example: 30000,50000,70000
#   continuous (optional): Set to "1" or "true" for continuous monitoring, "0" or "false" for one-time processing (default: 0)
#   check_interval (optional): Check for new checkpoints every N seconds (default: 300, only used in continuous mode)
#   manifest_file (optional): Manifest file name (default: test_manifest.json)
#
# Examples:
#   # Use default checkpoints from script creation (one-time mode)
#   sbatch SCRIPT_NAME
#
#   # Specify different checkpoints (one-time mode)
#   sbatch SCRIPT_NAME 30000,50000,70000
#
#   # Continuous monitoring mode
#   sbatch SCRIPT_NAME 30000,50000,70000 1
#
#   # Continuous monitoring with custom check interval
#   sbatch SCRIPT_NAME 30000,50000,70000 1 600

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
mkdir -p logs_summarized/logs_v7/6_train_hcn_age_from_promt

# Navigate to project directory
cd /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2

# Config file path (based on experiment name)
CONFIG_FILE="configs_summarized/v7/6_train_hcn_age_from_promt.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Available config files:"
    ls -1 configs_summarized/VERSION/*.yaml 2>/dev/null || echo "  No config files found in configs_summarized/VERSION/"
    exit 1
fi

# Get checkpoints (use provided or default from script creation)
CHECKPOINTS_ARG="${1:-20000}"
CONTINUOUS_FLAG="${2:-0}"
CHECK_INTERVAL="${3:-300}"  # Default: 5 minutes
MANIFEST_FILE="${4:-test_manifest.json}"

# Determine if continuous mode is enabled
if [ "$CONTINUOUS_FLAG" = "1" ] || [ "$CONTINUOUS_FLAG" = "true" ] || [ "$CONTINUOUS_FLAG" = "0" ]; then
    CONTINUOUS_MODE="Continuous monitoring"
    TEST_CONTINUOUS_FLAG="--test_continuous"
else
    CONTINUOUS_MODE="One-time processing"
    TEST_CONTINUOUS_FLAG=""
fi

# Launch test validation
echo "=========================================="
echo "Starting test validation..."
echo "Config file: $CONFIG_FILE"
echo "Target checkpoints: $CHECKPOINTS_ARG"
echo "Mode: $CONTINUOUS_MODE"
if [ "$CONTINUOUS_MODE" = "Continuous monitoring" ]; then
    echo "Check interval: ${CHECK_INTERVAL}s"
fi
echo "Manifest file: $MANIFEST_FILE"
echo "Note: test_dir must be specified in config file"
echo "=========================================="

# Build accelerate launch command
ACCELERATE_CMD="accelerate launch \
    --num_processes=8 \
    --num_machines=1 \
    --multi_gpu \
    --mixed_precision bf16 \
    roentgenv2/train_code/run_validation_monitor_debug.py \
    --config_file=\"$CONFIG_FILE\" \
    --test_mode \
    --checkpoints=\"$CHECKPOINTS_ARG\" \
    --check_interval=\"$CHECK_INTERVAL\" \
    --manifest_file=\"$MANIFEST_FILE\" \
    --load_images_from_dir"

# Add continuous flag if enabled
if [ -n "$TEST_CONTINUOUS_FLAG" ]; then
    ACCELERATE_CMD="$ACCELERATE_CMD $TEST_CONTINUOUS_FLAG"
fi

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
