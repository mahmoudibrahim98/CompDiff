#!/bin/bash
#SBATCH --job-name=generate_fairdiffusion_run2
#SBATCH --output=logs_summarized/logs_fairdiffusion/0_train_baseline_fairdiffusion_run2/generate_synthetic_%j.out
#SBATCH --error=logs_summarized/logs_fairdiffusion/0_train_baseline_fairdiffusion_run2/generate_synthetic_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a5000:8
#SBATCH --mem=230gb
#SBATCH --partition=ai4health
#SBATCH --account=ai4health
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mahmoud.ibrahim@vito.be

# Usage:
#   sbatch SCRIPT [checkpoint_path] [output_dir] [max_samples]

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

DEFAULT_CHECKPOINT_NAME="checkpoint-7500"
DEFAULT_CHECKPOINT_PATH="outputs_summarized/fairdiffusion/0_train_baseline_fairdiffusion_run2/checkpoint-7500"
DEFAULT_OUTPUT_DIR="synthetic_datasets/fairdiffusion/0_train_baseline_fairdiffusion_run2/train-7500"

CHECKPOINT_ARG="${1:-}"
OUTPUT_DIR_ARG="${2:-}"
MAX_SAMPLES="${3:-}"

if [ -n "$CHECKPOINT_ARG" ]; then
    if [[ "$CHECKPOINT_ARG" =~ ^/ ]]; then
        CHECKPOINT_PATH="$CHECKPOINT_ARG"
    else
        if [[ "$CHECKPOINT_ARG" =~ ^checkpoint- ]]; then
            CHECKPOINT_NAME="$CHECKPOINT_ARG"
        else
            CHECKPOINT_NAME="checkpoint-${CHECKPOINT_ARG}"
        fi
        CHECKPOINT_PATH="outputs_summarized/fairdiffusion/0_train_baseline_fairdiffusion_run2/$CHECKPOINT_NAME"
    fi
else
    CHECKPOINT_PATH="$DEFAULT_CHECKPOINT_PATH"
fi

if [ -n "$OUTPUT_DIR_ARG" ]; then
    OUTPUT_DIR="$OUTPUT_DIR_ARG"
else
    OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
fi

echo "=========================================="
echo "Starting synthetic dataset generation..."
echo "Config file: $CONFIG_FILE"
echo "Checkpoint path: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max samples: $MAX_SAMPLES"
fi
echo "=========================================="

ACCELERATE_CMD="accelerate launch \
    --num_processes=6 \
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

ACCELERATE_CMD="$ACCELERATE_CMD --merge_csv"
echo "CSV Mode: Will merge all GPU CSV files after generation"

echo "Running command:"
echo "$ACCELERATE_CMD"
echo "=========================================="
eval $ACCELERATE_CMD

EXIT_CODE=$?

echo "=========================================="
echo "Synthetic dataset generation finished"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
