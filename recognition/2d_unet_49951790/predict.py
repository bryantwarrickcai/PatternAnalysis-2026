"""
predict.py
==========
Inference and visualization for the 2D U-Net

This file will contain:
- Loading a trained checkpoint
- Running inference over the held-out test split
- Per-class Dice / IoU reporting
- 4-panel overlay visualizations (image | GT | pred | error)
- Per-slice metrics exported to CSV for downstream failure analysis

NumPy is allowed here for visualization only (per course spec).
"""
