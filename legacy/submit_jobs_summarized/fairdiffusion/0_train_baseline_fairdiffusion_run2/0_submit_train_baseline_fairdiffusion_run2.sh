#!/bin/bash
#SBATCH --job-name=train_fairdiffusion_run2
#SBATCH --output=logs_summarized/logs_fairdiffusion/0_train_baseline_fairdiffusion_run2/training_%j.out
#SBATCH --error=logs_summarized/logs_fairdiffusion/0_train_baseline_fairdiffusion_run2/training_%j.err
#SBATCH --time=7-00:00:00          # 7 days (adjust as needed)
#SBATCH --nodes=1                  # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4         # CPUs per task (adjust based on your needs)
#SBATCH --gres=gpu:6               # GPUs (matching your training setup)
#SBATCH --mem=90gb                    # Use all available memory on the node
#SBATCH --partition=precisionhealth      # Partition name (adjust to your cluster's partition)
#SBATCH --account=precisionhealth      # Account name (adjust to your cluster's account)
#SBATCH --mail-type=ALL             # Email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mahmoud.ibrahim@vito.be  # Your email (adjust as needed)

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

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Set environment variables for distributed training
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# Navigate to project directory
cd /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2

# Create logs directory if it doesn't exist
mkdir -p logs_summarized/logs_fairdiffusion/0_train_baseline_fairdiffusion_run2

# Config file path
CONFIG_FILE="configs_summarized/fairdiffusion/0_train_baseline_fairdiffusion_run2.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Launch training with accelerate
echo "=========================================="
echo "Starting training..."
echo "Config file: $CONFIG_FILE"
echo "=========================================="

accelerate launch \
    --num_processes=6 \
    --multi_gpu \
    --mixed_precision bf16 \
    roentgenv2/train_code/train.py \
    --config_file="$CONFIG_FILE"

# Capture exit code
EXIT_CODE=$?

echo "=========================================="
echo "Training finished with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
