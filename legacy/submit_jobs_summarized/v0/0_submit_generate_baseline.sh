#!/bin/bash
#SBATCH --job-name=generate_synthetic
#SBATCH --output=logs_summarized/logs_v0/0_train_baseline/generate_synthetic_%j.out
#SBATCH --error=logs_summarized/logs_v0/0_train_baseline/generate_synthetic_%j.err
#SBATCH --time=7-00:00:00          # 7 days (generation can take a while for large datasets)
#SBATCH --nodes=1                  # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # CPUs per task
#SBATCH --gres=gpu:a5000:8               # GPUs for distributed generation
#SBATCH --mem=256gb                # Memory for generation (8 processes × models)
#SBATCH --partition=ai4health      # Partition name (adjust to your cluster's partition)
#SBATCH --account=ai4health      # Account name (adjust to your cluster's account)
#SBATCH --mail-type=ALL             # Email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mahmoud.ibrahim@vito.be  # Your email (adjust as needed)

# Usage:
#   sbatch SCRIPT_NAME [checkpoint_path] [output_dir] [max_samples]
#
# Arguments:
#   checkpoint_path (optional): Path to checkpoint directory (default: outputs_summarized/v0/0_train_baseline/CHECKPOINT)
#   output_dir (optional): Output directory for synthetic images (default: synthetic_datasets/VERSION/EXPERIMENT_ID/CHECKPOINT)
#   max_samples (optional): Maximum number of samples to generate (optional)
#
# Examples:
#   # Use default checkpoint and output paths
#   sbatch SCRIPT_NAME
#
#   # Specify checkpoint
#   sbatch SCRIPT_NAME checkpoint-30000
#
#   # Specify checkpoint and output directory
#   sbatch SCRIPT_NAME checkpoint-30000 ./my_synthetic_output
#
#   # Specify all parameters
#   sbatch SCRIPT_NAME checkpoint-30000 ./my_synthetic_output 10000

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

# Set environment variables for multi-GPU distributed generation
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

# Default checkpoint (from script creation or use latest)
DEFAULT_CHECKPOINT_NAME="checkpoint-10000"
DEFAULT_CHECKPOINT_PATH="outputs_summarized/v0/0_train_baseline/checkpoint-10000"
DEFAULT_OUTPUT_DIR="synthetic_datasets/v0/0_train_baseline/train-10000-copy3"

# Get checkpoint path (use provided or default)
CHECKPOINT_ARG="${1:-}"
OUTPUT_DIR_ARG="${2:-}"
MAX_SAMPLES="${3:-}"

# Determine checkpoint path
if [ -n "$CHECKPOINT_ARG" ]; then
    # User provided checkpoint - construct full path if relative
    if [[ "$CHECKPOINT_ARG" =~ ^/ ]]; then
        # Absolute path - use as-is
        CHECKPOINT_PATH="$CHECKPOINT_ARG"
    else
        # Relative path - ensure it has "checkpoint-" prefix
        if [[ "$CHECKPOINT_ARG" =~ ^checkpoint- ]]; then
            CHECKPOINT_NAME="$CHECKPOINT_ARG"
        else
            # Add "checkpoint-" prefix if missing (e.g., "10000" -> "checkpoint-10000")
            CHECKPOINT_NAME="checkpoint-${CHECKPOINT_ARG}"
        fi
        CHECKPOINT_PATH="outputs_summarized/v0/0_train_baseline/$CHECKPOINT_NAME"
    fi
else
    # Use default checkpoint
    CHECKPOINT_PATH="$DEFAULT_CHECKPOINT_PATH"
fi

# Determine output directory
if [ -n "$OUTPUT_DIR_ARG" ]; then
    OUTPUT_DIR="$OUTPUT_DIR_ARG"
else
    OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
fi

# Launch synthetic dataset generation
echo "=========================================="
echo "Starting synthetic dataset generation..."
echo "Config file: $CONFIG_FILE"
echo "Checkpoint path: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max samples: $MAX_SAMPLES"
fi
echo "=========================================="

# Build accelerate launch command
ACCELERATE_CMD="accelerate launch \
    --num_processes=8 \
    --num_machines=1 \
    --multi_gpu \
    --mixed_precision bf16 \
    roentgenv2/inference_code/generate_synthetic_dataset.py \
    --config_file=\"$CONFIG_FILE\" \
    --checkpoint_path=\"$CHECKPOINT_PATH\" \
    --output_dir=\"$OUTPUT_DIR\""

if [ -n "$MAX_SAMPLES" ]; then
    ACCELERATE_CMD="$ACCELERATE_CMD --max_samples=\"$MAX_SAMPLES\""
fi

# Add --merge_csv flag if enabled
MERGE_CSV_FLAG="1"
if [ "$MERGE_CSV_FLAG" = "1" ] || [ "$MERGE_CSV_FLAG" = "true" ]; then
    ACCELERATE_CMD="$ACCELERATE_CMD --merge_csv"
    echo "CSV Mode: Will merge all GPU CSV files after generation"
fi

# Run generation with all allocated GPUs
# Using 8 GPUs for faster generation (distributes image generation across GPUs)
echo "Running command:"
echo "$ACCELERATE_CMD"
echo "=========================================="
eval $ACCELERATE_CMD

# Capture exit code
EXIT_CODE=$?

echo "=========================================="
echo "Synthetic dataset generation finished"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

# Exit with the same code as the generation script
exit $EXIT_CODE
