"""
utils.py
========
Shared helpers for the 2D U-Net prostate segmentation project

This file will contain small, reusable utilities used across
train.py and predict.py, e.g.:
- Dice and IoU metric functions (per-class + macro)
- Dice loss
- Plotting helpers for training curves and overlays

Kept deliberately small; anything model-specific lives in modules.py.
"""
