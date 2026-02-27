#!/bin/bash
#SBATCH --job-name=downstream_1b_fd
#SBATCH --output=logs_summarized/downstream_eval/model_1b_fairdiffusion/model_1b_fairdiffusion_%j.out
#SBATCH --error=logs_summarized/downstream_eval/model_1b_fairdiffusion/model_1b_fairdiffusion_%j.err
#SBATCH --time=2-00:00:00          # 2 days (adjust as needed)
#SBATCH --nodes=1                  # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4         # CPUs for data loading
#SBATCH --gres=gpu:rtx4090:1               # 1 GPU for classifier training
#SBATCH --mem=64gb                 # Memory (adjust based on batch size)
#SBATCH --partition=ai4health # Partition name (adjust to your cluster's partition)
#SBATCH --account=ai4health  # Account name (adjust to your cluster's account)
#SBATCH --mail-type=ALL            # Email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mahmoud.ibrahim@vito.be  # Your email (adjust as needed)

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

# Use roentgen env Python by explicit path (avoids PATH picking up project user_monitor/venv on compute nodes)
if [ -x "$HOME/.conda/envs/roentgen/bin/python" ]; then
    PYTHON_CMD="$HOME/.conda/envs/roentgen/bin/python"
elif [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
else
    # Initialize conda and activate roentgen
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
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
fi
export LD_LIBRARY_PATH="${CONDA_PREFIX:-$HOME/.conda/envs/roentgen}/lib:$LD_LIBRARY_PATH"

# Verify environment (fail early if torch missing)
echo "Python: $PYTHON_CMD"
echo "Python version: $($PYTHON_CMD --version)"
$PYTHON_CMD -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print("Number of GPUs:", torch.cuda.device_count())' || { echo "ERROR: torch not found in roentgen env"; exit 1; }

# Navigate to project directory
cd /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2

# Create logs directory if it doesn't exist
mkdir -p logs_summarized/downstream_eval/model_1b_fairdiffusion

# Config file path (FairDiffusion 0_train_baseline_fairdiffusion)
CONFIG_FILE="configs_summarized/downstream_eval/model_1b_fairdiffusion.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Launch training
echo "=========================================="
echo "Starting downstream classifier training (Model 1b FairDiffusion)..."
echo "Config file: $CONFIG_FILE"
echo "=========================================="

# Check if checkpoint exists - if yes, skip training and only evaluate
CHECKPOINT_PATH=""
CHECKPOINT_PATH1="outputs_summarized/downstream_eval/model_1b_fairdiffusion/checkpoint_best.pth"
CHECKPOINT_PATH2="outputs/downstream_eval/model_1b_fairdiffusion/checkpoint_best.pth"
if [ -f "$CHECKPOINT_PATH1" ]; then
    CHECKPOINT_PATH="$CHECKPOINT_PATH1"
elif [ -f "$CHECKPOINT_PATH2" ]; then
    CHECKPOINT_PATH="$CHECKPOINT_PATH2"
fi

if [ -n "$CHECKPOINT_PATH" ] && [ -f "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint found: $CHECKPOINT_PATH"
    echo "Skipping training, running evaluation only..."
    $PYTHON_CMD scripts/train_downstream_classifier.py \
        --config "$CONFIG_FILE" \
        --skip_training \
        --n_bootstrap 1000 \
        --random_seed 42
else
    echo "No checkpoint found, starting training..."
    $PYTHON_CMD scripts/train_downstream_classifier.py \
        --config "$CONFIG_FILE" \
        --n_bootstrap 1000 \
        --random_seed 42
fi

# Capture exit code
EXIT_CODE=$?

echo "=========================================="
echo "Training finished with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

# Exit with the same code as the training script
exit $EXIT_CODE
