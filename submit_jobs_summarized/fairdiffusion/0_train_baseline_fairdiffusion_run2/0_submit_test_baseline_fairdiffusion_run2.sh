#!/bin/bash
#SBATCH --job-name=test_fairdiffusion_run2
#SBATCH --output=logs_summarized/logs_fairdiffusion/0_train_baseline_fairdiffusion_run2/test_%j.out
#SBATCH --error=logs_summarized/logs_fairdiffusion/0_train_baseline_fairdiffusion_run2/test_%j.err
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
#   sbatch SCRIPT [checkpoints] [continuous] [check_interval] [manifest_file]
#
# Arguments:
#   checkpoints (optional): Comma-separated checkpoint numbers (default: 7500)
#   continuous (optional): "1" for continuous monitoring, "0" for one-time (default: 0)
#   check_interval (optional): Seconds between checks (default: 300)
#   manifest_file (optional): Manifest file name (default: test_manifest.json)
#
# Examples:
#   sbatch SCRIPT                             # default checkpoint, one-time
#   sbatch SCRIPT 20000,25000                 # specific checkpoints
#   sbatch SCRIPT 20000,25000 1              # continuous monitoring
#   sbatch SCRIPT 20000,25000 1 600          # with custom interval

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

CONFIG_FILE="configs_summarized/fairdiffusion/0_train_baseline_fairdiffusion_run2.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

CHECKPOINTS_ARG="${1:-7500}"
CONTINUOUS_FLAG="${2:-0}"
CHECK_INTERVAL="${3:-300}"
MANIFEST_FILE="${4:-test_manifest.json}"

if [ "$CONTINUOUS_FLAG" = "1" ] || [ "$CONTINUOUS_FLAG" = "true" ]; then
    CONTINUOUS_MODE="Continuous monitoring"
    TEST_CONTINUOUS_FLAG="--test_continuous"
else
    CONTINUOUS_MODE="One-time processing"
    TEST_CONTINUOUS_FLAG=""
fi

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

ACCELERATE_CMD="accelerate launch \
    --num_processes=6 \
    --num_machines=1 \
    --multi_gpu \
    --mixed_precision bf16 \
    roentgenv2/train_code/run_validation_monitor_debug.py \
    --config_file=\"$CONFIG_FILE\" \
    --test_mode \
    --checkpoints=\"$CHECKPOINTS_ARG\" \
    --check_interval=\"$CHECK_INTERVAL\" \
    --manifest_file=\"$MANIFEST_FILE\""

if [ -n "$TEST_CONTINUOUS_FLAG" ]; then
    ACCELERATE_CMD="$ACCELERATE_CMD $TEST_CONTINUOUS_FLAG"
fi

echo "Running command:"
echo "$ACCELERATE_CMD"
echo "=========================================="
eval $ACCELERATE_CMD

EXIT_CODE=$?

echo "=========================================="
echo "Test validation finished"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
