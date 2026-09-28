#!/bin/bash
#SBATCH --job-name=downstream_5a
#SBATCH --output=logs_summarized/downstream_eval/model_5a/model_5a_%j.out
#SBATCH --error=logs_summarized/downstream_eval/model_5a/model_5a_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx4090:1               # 1 GPU for classifier training
#SBATCH --mem=64gb                 # Memory (adjust based on batch size)
#SBATCH --partition=ai4health # Partition name (adjust to your cluster's partition)
#SBATCH --account=ai4health  # Account name (adjust to your cluster's account)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mahmoud.ibrahim@vito.be

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

module load CUDA

# Use roentgen env Python by explicit path (avoids PATH picking up project user_monitor/venv on compute nodes)
if [ -x "$HOME/.conda/envs/roentgen/bin/python" ]; then
    PYTHON_CMD="$HOME/.conda/envs/roentgen/bin/python"
elif [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
else
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
$PYTHON_CMD -c 'import torch' || { echo "ERROR: torch not found in roentgen env"; exit 1; }

echo "Python: $PYTHON_CMD"
echo "Python version: $($PYTHON_CMD --version)"
echo "CUDA available: $($PYTHON_CMD -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $($PYTHON_CMD -c 'import torch; print(torch.cuda.device_count())')"

cd /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2

mkdir -p logs_summarized/downstream_eval

CONFIG_FILE="configs_summarized/downstream_eval/model_5a_synthetic_test_v0.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "Starting downstream classifier training (Model 5a)..."
echo "Config file: $CONFIG_FILE"
echo "=========================================="

CHECKPOINT_PATH_1="outputs/downstream_eval/model_5a_v0/checkpoint_best.pth"
CHECKPOINT_PATH_2="outputs_summarized/downstream_eval/model_5a_v0/checkpoint_best.pth"

if [ -f "$CHECKPOINT_PATH_1" ] || [ -f "$CHECKPOINT_PATH_2" ]; then
    echo "Checkpoint found for model_5a. Skipping training, running evaluation only..."
    $PYTHON_CMD scripts/train_downstream_classifier.py \
        --config "$CONFIG_FILE" \
        --skip_training \
        --n_bootstrap 1000 \
        --random_seed 42
else
    echo "No checkpoint found for model_5a, starting training..."
    $PYTHON_CMD scripts/train_downstream_classifier.py \
        --config "$CONFIG_FILE" \
        --n_bootstrap 1000 \
        --random_seed 42
fi

EXIT_CODE=$?

echo "=========================================="
echo "Training finished with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE






