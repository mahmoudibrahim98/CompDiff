#!/bin/bash
#SBATCH --job-name=downstream_1a
#SBATCH --output=logs_summarized/downstream_eval/model_1a/model_1a_%j.out
#SBATCH --error=logs_summarized/downstream_eval/model_1a/model_1a_%j.err
#SBATCH --time=2-00:00:00          # 2 days (adjust as needed)
#SBATCH --nodes=1                  # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # CPUs for data loading
#SBATCH --gres=gpu:a100:1               # 1 GPU for classifier training
#SBATCH --mem=64gb                 # Memory (adjust based on batch size)
#SBATCH --partition=precisionhealth # Partition name (adjust to your cluster's partition)
#SBATCH --account=precisionhealth  # Account name (adjust to your cluster's account)
#SBATCH --mail-type=ALL            # Email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mahmoud.ibrahim@vito.be  # Your email (adjust as neededs)

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
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Navigate to project directory
cd /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2

# Create logs directory if it doesn't exist
mkdir -p logs_summarized/downstream_eval

# Config file path
CONFIG_FILE="configs_summarized/downstream_eval/model_1a_real_only.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Launch training
echo "=========================================="
echo "Starting downstream classifier training (Model 1a)..."
echo "Config file: $CONFIG_FILE"
echo "=========================================="

# Check if checkpoint exists - if yes, skip training and only evaluate
CHECKPOINT_PATH="outputs_summarized/downstream_eval/model_1a/checkpoint_best.pth"
if [ -f "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint found: $CHECKPOINT_PATH"
    echo "Skipping training, running evaluation only..."
    python scripts/train_downstream_classifier.py \
        --config "$CONFIG_FILE" \
        --skip_training \
        --n_bootstrap 1000 \
        --random_seed 42
else
    echo "No checkpoint found, starting training..."
    python scripts/train_downstream_classifier.py \
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

