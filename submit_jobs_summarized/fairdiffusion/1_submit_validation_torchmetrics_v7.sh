#!/bin/bash
#SBATCH --job-name=validation_torchmetrics_v7
#SBATCH --output=logs_summarized/logs_fairdiffusion/1_train_hcn_fairdiffusion/validation_torchmetrics_v7_%j.out
#SBATCH --error=logs_summarized/logs_fairdiffusion/1_train_hcn_fairdiffusion/validation_torchmetrics_v7_%j.err
#SBATCH --time=2-00:00:00          # 2 days (should be faster since images are pre-generated)
#SBATCH --nodes=1                  # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # CPUs per task
#SBATCH --gres=gpu:rtx4090:1      # Only need 1 GPU for metrics computation (images already generated)
#SBATCH --mem=64gb                # Less memory needed (no image generation)
#SBATCH --partition=ai4health      # Partition name (adjust to your cluster's partition)
#SBATCH --account=ai4health      # Account name (adjust to your cluster's account)
#SBATCH --mail-type=ALL             # Email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mahmoud.ibrahim@vito.be  # Your email (adjust as needed)

# Usage:
#   sbatch 1_submit_validation_torchmetrics_v7.sh [v7_output_dir] [config_file] [check_interval] [manifest_file]
#
# Arguments:
#   v7_output_dir (required): Path to v7 output directory containing validation_images/step_* directories
#   config_file (optional): Path to config file (default: configs_summarized/fairdiffusion/1_train_hcn_fairdiffusion.yaml)
#   check_interval (optional): Check for new step directories every N seconds (default: 60)
#   manifest_file (optional): Manifest file name (default: validation_torchmetrics_manifest.json)
#
# Examples:
#   # Validate images from v7 output directory
#   sbatch 1_submit_validation_torchmetrics_v7.sh outputs_summarized/v7/0_train_hcn/outputs
#
#   # With custom config and check interval
#   sbatch 1_submit_validation_torchmetrics_v7.sh outputs_summarized/v7/0_train_hcn/outputs configs_summarized/fairdiffusion/1_train_hcn_fairdiffusion.yaml 120

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
echo "torchmetrics available: $(python -c 'import torchmetrics; print(torchmetrics.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
echo "=========================================="

# Check if torchmetrics is installed
python -c "from torchmetrics.image.fid import FrechetInceptionDistance" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: torchmetrics is not installed or not available"
    echo "Please install with: pip install torchmetrics"
    exit 1
fi

# Set environment variables
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# Create logs directory if it doesn't exist
mkdir -p logs_summarized

# Navigate to project directory
cd /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2

# V7 output directory (required argument)
V7_OUTPUT_DIR="${1:-}"
if [ -z "$V7_OUTPUT_DIR" ]; then
    echo "ERROR: V7 output directory is required as first argument"
    echo "Usage: sbatch 1_submit_validation_torchmetrics_v7.sh <v7_output_dir> [config_file] [check_interval] [manifest_file]"
    echo ""
    echo "Example:"
    echo "  sbatch 1_submit_validation_torchmetrics_v7.sh outputs_summarized/v7/0_train_hcn/outputs"
    exit 1
fi

# Check if v7 output directory exists
if [ ! -d "$V7_OUTPUT_DIR" ]; then
    echo "ERROR: V7 output directory not found: $V7_OUTPUT_DIR"
    echo "Available directories:"
    ls -d outputs_summarized/v7/*/outputs 2>/dev/null || echo "  No v7 output directories found"
    exit 1
fi

# Check if validation_images directory exists in v7 output
V7_VALIDATION_IMAGES_DIR="$V7_OUTPUT_DIR/validation_images"
if [ ! -d "$V7_VALIDATION_IMAGES_DIR" ]; then
    echo "ERROR: validation_images directory not found in: $V7_OUTPUT_DIR"
    echo "Expected: $V7_VALIDATION_IMAGES_DIR"
    echo "Available subdirectories:"
    ls -1 "$V7_OUTPUT_DIR" 2>/dev/null || echo "  Directory is empty or inaccessible"
    exit 1
fi

# Count step directories
STEP_COUNT=$(find "$V7_VALIDATION_IMAGES_DIR" -maxdepth 1 -type d -name "step_*" | wc -l)
echo "Found $STEP_COUNT step directories in $V7_VALIDATION_IMAGES_DIR"

# Config file path (optional, defaults to fairdiffusion config)
CONFIG_FILE="${2:-configs_summarized/fairdiffusion/1_train_hcn_fairdiffusion.yaml}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Available config files:"
    ls -1 configs_summarized/fairdiffusion/*.yaml 2>/dev/null || echo "  No config files found"
    exit 1
fi

# Optional: Check interval, manifest file arguments
CHECK_INTERVAL="${3:-60}"  # Default: 60 seconds (faster since we're just loading images)
MANIFEST_FILE="${4:-validation_torchmetrics_manifest.json}"

# Create a temporary config file that points to v7 output directory
# We need to override output_dir in the config to point to v7 output
TEMP_CONFIG=$(mktemp /tmp/torchmetrics_config_XXXXXX.yaml)
cp "$CONFIG_FILE" "$TEMP_CONFIG"

# Update output_dir in temp config to point to v7 output directory
# Using sed to replace output_dir line
sed -i "s|^output_dir:.*|output_dir: \"$V7_OUTPUT_DIR\"|" "$TEMP_CONFIG"

echo "=========================================="
echo "Starting torchmetrics validation on pre-generated images..."
echo "V7 Output Directory: $V7_OUTPUT_DIR"
echo "Validation Images Directory: $V7_VALIDATION_IMAGES_DIR"
echo "Step Directories Found: $STEP_COUNT"
echo "Config file: $CONFIG_FILE"
echo "Temp config (with overridden output_dir): $TEMP_CONFIG"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "Manifest file: $MANIFEST_FILE"
echo "Mode: Loading pre-generated images (skipping generation)"
echo "=========================================="

# Build accelerate launch command
# Note: Only need 1 GPU for metrics computation (images are already generated)
ACCELERATE_CMD="accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision bf16 \
    roentgenv2/train_code/run_validation_monitor_torchmetrics.py \
    --config_file=\"$TEMP_CONFIG\" \
    --check_interval=\"$CHECK_INTERVAL\" \
    --manifest_file=\"$MANIFEST_FILE\" \
    --load_images_from_dir"

# Run validation
echo "Launching torchmetrics validation..."
eval $ACCELERATE_CMD

# Capture exit code
EXIT_CODE=$?

# Clean up temp config file
rm -f "$TEMP_CONFIG"

echo "=========================================="
echo "Torchmetrics validation finished"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

# Exit with the same code as the validation script
exit $EXIT_CODE
