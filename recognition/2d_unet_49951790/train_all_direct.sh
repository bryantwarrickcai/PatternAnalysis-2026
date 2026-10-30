#!/bin/bash
# ==============================================================================
# train_all_direct.sh
# Use this script for a direct-execution Linux GPU environment.
# (For the Rangpur HPC cluster using Conda and Slurm, use train_all_rangpur.sh)
# ==============================================================================

set -euo pipefail

# 1. Setup output directory relative to the project folder
RUN_DIR="./runs_full"
mkdir -p "$RUN_DIR"

echo "================================================="
echo "Environment Check:"
which python
python -c "import torch; print('Torch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"
echo "================================================="

echo "================================================="
echo "Starting U-Net Training..."
echo "================================================="
python train.py \
    --model unet \
    --out-dir   "$RUN_DIR/runs_unet_full" \
    --epochs 50 --batch-size 16 --base-features 32 \
    --device cuda --num-workers 4 --seed 42

echo "================================================="
echo "Starting Baseline Training..."
echo "================================================="
python train.py \
    --model baseline \
    --out-dir   "$RUN_DIR/runs_baseline_full" \
    --epochs 50 --batch-size 32 --base-features 16 \
    --device cuda --num-workers 4 --seed 42

echo "================================================="
echo "Running Inference & Generating Overlays..."
echo "================================================="
python predict.py \
    --checkpoint "$RUN_DIR/runs_unet_full/best_model.pt" \
    --split test --device cuda --batch-size 32 \
    --save-overlays --num-overlays 30

python predict.py \
    --checkpoint "$RUN_DIR/runs_baseline_full/best_model.pt" \
    --split test --device cuda --batch-size 32 \
    --save-overlays --num-overlays 30

echo "================================================="
echo "Running Failure Case Autopsy..."
echo "================================================="
python failure_analysis.py \
    --csv "$RUN_DIR/runs_unet_full/test_metrics.csv" \
    --checkpoint "$RUN_DIR/runs_unet_full/best_model.pt" \
    --device cuda --top-k 5 \
    --out-dir "$RUN_DIR/runs_unet_full/failure_autopsy"

python failure_analysis.py \
    --csv "$RUN_DIR/runs_baseline_full/test_metrics.csv" \
    --checkpoint "$RUN_DIR/runs_baseline_full/best_model.pt" \
    --device cuda --top-k 5 \
    --out-dir "$RUN_DIR/runs_baseline_full/failure_autopsy"

echo "================================================="
echo "Profiling Resources..."
echo "================================================="
python profile_model.py --device cuda --iterations 100 \
    | tee "$RUN_DIR/profile_output.txt"

echo "Pipeline Complete!"
