"""
modules.py
==========
Model components for the 2D U-Net prostate segmentation task

This file will contain all PyTorch nn.Module definitions only:
- DoubleConv, Down, Up building blocks
- The full 2D UNet(in_channels, out_channels, base_features) class

No dataset loading, no training loop, no NumPy usage — model code only.
"""
