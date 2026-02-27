#!/bin/bash
#SBATCH --job-name=downstream_3b_v0
#SBATCH --output=logs_summarized/downstream_eval/model_3b_v0/model_3b_v0_%j.out
#SBATCH --error=logs_summarized/downstream_eval/model_3b_v0/model_3b_v0_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1               # 1 GPU for classifier training
#SBATCH --mem=64gb                 # Memory (adjust based on batch size)
#SBATCH --partition=precisionhealth # Partition name (adjust to your cluster's partition)
#SBATCH --account=precisionhealth  # Account name (adjust to your cluster's account)
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
source activate roentgen

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

cd /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2

mkdir -p logs_summarized/downstream_eval/model_3b_v0

CONFIG_FILE="configs_summarized/downstream_eval/model_3b_pretrain_finetune_v0.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "Starting downstream classifier training (Model 3b v0 - ImageNet initialized)..."
echo "Config file: $CONFIG_FILE"
echo "=========================================="

# Strategy 3b uses two-phase training (ImageNet pretrain on synthetic + finetune on real)
# The final checkpoint is in the finetune directory with subset_fraction
CHECKPOINT_PATH_1="outputs/downstream_eval/model_3b_v0_finetune_1.0/checkpoint_best.pth"
CHECKPOINT_PATH_2="outputs_summarized/downstream_eval/model_3b_v0_finetune_1.0/checkpoint_best.pth"

if [ -f "$CHECKPOINT_PATH_1" ] || [ -f "$CHECKPOINT_PATH_2" ]; then
    echo "Checkpoint found for model_3b_v0_finetune_1.0. Skipping training, running evaluation only..."
    python scripts/train_downstream_classifier.py \
        --config "$CONFIG_FILE" \
        --skip_training \
        --n_bootstrap 1000 \
        --random_seed 42
else
    echo "No checkpoint found for model_3b_v0_finetune_1.0, starting training..."
    python scripts/train_downstream_classifier.py \
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
