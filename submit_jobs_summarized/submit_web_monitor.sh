#!/bin/bash
#SBATCH --job-name=web_monitor
#SBATCH --output=logs_summarized/web_monitor/web_monitor_%j.out
#SBATCH --error=logs_summarized/web_monitor/web_monitor_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --partition=ai4health
#SBATCH --account=ai4health
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mahmoud.ibrahim@vito.be

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Navigate to project root
cd /home/vito/ibrahimm/projects/AI4Health/notebooks/ibrahimm/Generative-Models/images/Chest_XRay/RoentGen-v2

# Create log directory
mkdir -p logs_summarized/web_monitor

# Navigate to web_monitor directory
cd web_monitor

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.installed
fi

# Set Flask environment variables (can be overridden)
export FLASK_HOST=${FLASK_HOST:-0.0.0.0}
export FLASK_PORT=${FLASK_PORT:-56000}
export FLASK_DEBUG=${FLASK_DEBUG:-False}

echo "=========================================="
echo "Starting Web Monitor..."
echo "Host: $FLASK_HOST"
echo "Port: $FLASK_PORT"
echo "Debug: $FLASK_DEBUG"
echo "Access the dashboard at: http://localhost:$FLASK_PORT"
if [ "$FLASK_HOST" = "0.0.0.0" ]; then
    echo "Or from other machines: http://$(hostname -I | awk '{print $1}'):$FLASK_PORT"
fi
echo "=========================================="

# Run the Flask application
python app.py

EXIT_CODE=$?

echo "=========================================="
echo "Web Monitor finished with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE




