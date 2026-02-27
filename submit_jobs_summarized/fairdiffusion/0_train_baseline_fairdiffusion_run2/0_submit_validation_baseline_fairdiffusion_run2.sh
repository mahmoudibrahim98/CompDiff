#!/bin/bash
#SBATCH --job-name=val_fairdiffusion_run2
#SBATCH --output=logs_summarized/logs_fairdiffusion/0_train_baseline_fairdiffusion_run2/validation_%j.out
#SBATCH --error=logs_summarized/logs_fairdiffusion/0_train_baseline_fairdiffusion_run2/validation_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx4090:6
#SBATCH --mem=256gb
#SBATCH --partition=ai4health
#SBATCH --account=ai4health
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mahmoud.ibrahim@vito.be

# Usage:
#   sbatch SCRIPT [config_file] [check_interval] [manifest_file] [load_images_flag] [stop_at_step]

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

module load CUDA
source activate roentgen

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "=========================================="
echo "Environment Setup:"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "=========================================="

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

mkdir -p logs_summarized/logs_fairdiffusion/0_train_baseline_fairdiffusion_run2

cd /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2

CONFIG_FILE="${1:-configs_summarized/fairdiffusion/0_train_baseline_fairdiffusion_run2.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

CHECK_INTERVAL="${2:-300}"
MANIFEST_FILE="${3:-validation_manifest.json}"
LOAD_IMAGES_FLAG="${4:-0}"
STOP_AT_STEP="${5:-}"

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

ACCELERATE_CMD="accelerate launch \
    --num_processes=6 \
    --num_machines=1 \
    --multi_gpu \
    --mixed_precision bf16 \
    roentgenv2/train_code/run_validation_monitor_debug.py \
    --config_file=\"$CONFIG_FILE\" \
    --check_interval=\"$CHECK_INTERVAL\" \
    --manifest_file=\"$MANIFEST_FILE\""

if [ -n "$STOP_AT_STEP" ]; then
    ACCELERATE_CMD="$ACCELERATE_CMD --stop_at_step=\"$STOP_AT_STEP\""
fi

if [ "$LOAD_IMAGES_FLAG" = "1" ] || [ "$LOAD_IMAGES_FLAG" = "true" ]; then
    ACCELERATE_CMD="$ACCELERATE_CMD --load_images_from_dir"
fi

eval $ACCELERATE_CMD

EXIT_CODE=$?

echo "=========================================="
echo "Validation monitoring finished"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
