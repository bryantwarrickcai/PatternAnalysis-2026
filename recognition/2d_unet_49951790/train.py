"""
train.py
========
Training and validation entrypoint for the 2D U-Net

This file will contain:
- Loss composition (Dice + CrossEntropy)
- Optimizer and LR schedule setup
- Train / validation epoch loops
- Best-model checkpointing (by validation Dice)
- Loss / Dice curve plotting into runs/
- argparse CLI when executed as __main__
"""
