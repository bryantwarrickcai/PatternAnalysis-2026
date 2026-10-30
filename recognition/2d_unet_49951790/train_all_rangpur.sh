#!/bin/bash
#SBATCH --job-name=comp3710_train
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# We use 'set -uo pipefail' instead of 'set -e' so we can catch errors 
# and print custom messages before the script exits.
set -uo pipefail

# 1. Echo context for debugging (as recommended by the Rangpur guide)
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

# 2. Activate the environment
source $HOME/miniconda3/bin/activate
conda activate comp3710_env

# 3. Detect Python executable
if command -v python &> /dev/null; then
    PYTHON_BIN="python"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
else
    echo "ERROR: Neither 'python' nor 'python3' could be found in PATH."
    exit 1
fi

echo "Using Python binary: $PYTHON_BIN"
$PYTHON_BIN -c "import torch; print('Torch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"

# 4. Output directory for models and graphs
export RUN_DIR=$HOME/comp3710_runs
mkdir -p "$RUN_DIR"

echo "================================================="
echo "Starting U-Net Training..."
echo "================================================="
$PYTHON_BIN train.py \
    --model unet \
    --out-dir   "$RUN_DIR/runs_unet_full" \
    --epochs 50 --batch-size 16 --base-features 32 \
    --device cuda --num-workers 4 --seed 42

echo "================================================="
echo "Starting Baseline Training..."
echo "================================================="
$PYTHON_BIN train.py \
    --model baseline \
    --out-dir   "$RUN_DIR/runs_baseline_full" \
    --epochs 50 --batch-size 32 --base-features 16 \
    --device cuda --num-workers 4 --seed 42

echo "================================================="
echo "Running Inference & Generating Overlays..."
echo "================================================="
$PYTHON_BIN predict.py \
    --checkpoint "$RUN_DIR/runs_unet_full/best_model.pt" \
    --split test --device cuda --batch-size 32 \
    --save-overlays --num-overlays 30

$PYTHON_BIN predict.py \
    --checkpoint "$RUN_DIR/runs_baseline_full/best_model.pt" \
    --split test --device cuda --batch-size 32 \
    --save-overlays --num-overlays 30

echo "================================================="
echo "Running Failure Case Autopsy..."
echo "================================================="
$PYTHON_BIN failure_analysis.py \
    --csv "$RUN_DIR/runs_unet_full/test_metrics.csv" \
    --checkpoint "$RUN_DIR/runs_unet_full/best_model.pt" \
    --device cuda --top-k 5 \
    --out-dir "$RUN_DIR/runs_unet_full/failure_autopsy"

$PYTHON_BIN failure_analysis.py \
    --csv "$RUN_DIR/runs_baseline_full/test_metrics.csv" \
    --checkpoint "$RUN_DIR/runs_baseline_full/best_model.pt" \
    --device cuda --top-k 5 \
    --out-dir "$RUN_DIR/runs_baseline_full/failure_autopsy"

echo "================================================="
echo "Profiling Resources..."
echo "================================================="
$PYTHON_BIN profile_model.py --device cuda --iterations 100 \
    | tee "$RUN_DIR/profile_output.txt"

echo "Pipeline Complete! Finished at $(date)"
