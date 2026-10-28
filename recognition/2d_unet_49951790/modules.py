"""
modules.py
==========
Model components for the 2D U-Net prostate segmentation task

Contains only PyTorch nn.Module definitions — no dataset loading, no
training loop, no NumPy. Training entrypoint lives in train.py;
dataloading lives in dataset.py.

Reference:
    Ronneberger, Fischer & Brox, "U-Net: Convolutional Networks for
    Biomedical Image Segmentation", MICCAI 2015.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (Conv3x3 -> BatchNorm -> ReLU) repeated twice.

    ``padding=1`` and ``kernel_size=3`` keep the spatial size unchanged,
    so a DoubleConv block is shape-preserving in (H, W).

    ``bias=False`` on both convs because BatchNorm absorbs the bias.
    Optional spatial dropout (``Dropout2d``) is off by default and can
    be enabled from the U-Net constructor for regularisation.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        mid_ch: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.insert(3, nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Encoder step: 2x2 max pool (stride 2) followed by a DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """
    Decoder step: transposed-conv upsample, concatenate the skip, DoubleConv.

    ``forward(x, skip)``:
        x    : (N, in_ch,  H,   W  ) — output of the previous decoder block
                                       (or the bottleneck for the first call)
        skip : (N, skip_ch, 2H, 2W ) — matching encoder feature map

    Output is (N, out_ch, 2H, 2W). If the upsampled tensor and the skip
    differ by one pixel in either dimension (possible with odd input
    resolutions or non-power-of-two downsampling), the upsampled tensor
    is bilinearly resized to the skip size before concatenation. This is
    the same guard used by many U-Net implementations and costs nothing
    on even-sized inputs like 256x128.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
