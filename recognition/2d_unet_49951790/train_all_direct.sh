#!/bin/bash
# ==============================================================================
# train_all_direct.sh
# Use this script for a direct-execution Linux GPU environment.
# (For the Rangpur HPC cluster using Conda and Slurm, use train_all_rangpur.sh)
# ==============================================================================

set -uo pipefail

# 1. Detect Python executable (python vs python3)
if command -v python &> /dev/null; then
    PYTHON_BIN="python"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
else
    echo "ERROR: Neither 'python' nor 'python3' could be found in PATH."
    exit 1
fi

# 2. Setup output directory relative to the project folder
RUN_DIR="./runs_full"
mkdir -p "$RUN_DIR"

echo "================================================="
echo "Environment Check:"
echo "Using Python binary: $PYTHON_BIN"
$PYTHON_BIN -c "import torch; print('Torch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"
echo "================================================="

if ! $PYTHON_BIN -c "import torch" &> /dev/null; then
    echo "ERROR: PyTorch is not installed in this Python environment."
    exit 1
fi

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

echo "Pipeline Complete!"
