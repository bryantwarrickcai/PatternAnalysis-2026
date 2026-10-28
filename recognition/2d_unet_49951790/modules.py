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


class UNet(nn.Module):
    """
    Standard 2D U-Net for semantic segmentation.

    Layout (four encoder levels plus a bottleneck):
        inc        DoubleConv(in_ch        -> base,     H,     W)
        down1      Down(base               -> 2*base)    H/2,   W/2
        down2      Down(2*base             -> 4*base)    H/4,   W/4
        down3      Down(4*base             -> 8*base)    H/8,   W/8
        down4      Down(8*base             -> 16*base)   H/16,  W/16   <- bottleneck
        up1        Up(16*base, 8*base      -> 8*base)    H/8,   W/8
        up2        Up( 8*base, 4*base      -> 4*base)    H/4,   W/4
        up3        Up( 4*base, 2*base      -> 2*base)    H/2,   W/2
        up4        Up( 2*base, 1*base      -> 1*base)    H,     W
        outc       1x1 Conv(base -> out_ch)                H,     W   <- logits

    The final 1x1 conv returns **raw logits** — no softmax/sigmoid. The
    loss function in train.py combines softmax cross-entropy with a
    Dice term, so applying softmax here would double-apply the
    nonlinearity. The ``predict.py`` script applies ``argmax`` over
    channels to obtain a per-pixel class map.

    H and W should each be divisible by 16 so that four downsamples are
    exact and no spatial rounding occurs before the upsampling path.
    The real HipMRI slices are 256x128, both divisible by 16.

    Parameters
    ----------
    in_channels   : number of input channels (1 for grayscale MRI slices).
    out_channels  : number of output classes (6 for HipMRI).
    base_features : channel width at the first level (default 32).
                    Peak U-Net width is ``16 * base_features`` (512 here).
    dropout       : spatial dropout applied inside every DoubleConv
                    (0.0 = off).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 6,
        base_features: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        b = base_features
        self.inc = DoubleConv(in_channels, b, dropout=dropout)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        self.down4 = Down(b * 8, b * 16)
        self.up1 = Up(b * 16, b * 8, b * 8)
        self.up2 = Up(b * 8, b * 4, b * 4)
        self.up3 = Up(b * 4, b * 2, b * 2)
        self.up4 = Up(b * 2, b, b)
        self.outc = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)             # (N, base,     H,    W)
        x2 = self.down1(x1)          # (N, 2*base,   H/2,  W/2)
        x3 = self.down2(x2)          # (N, 4*base,   H/4,  W/4)
        x4 = self.down3(x3)          # (N, 8*base,   H/8,  W/8)
        x5 = self.down4(x4)          # (N, 16*base,  H/16, W/16)  bottleneck
        x = self.up1(x5, x4)         # (N, 8*base,   H/8,  W/8)
        x = self.up2(x,  x3)         # (N, 4*base,   H/4,  W/4)
        x = self.up3(x,  x2)         # (N, 2*base,   H/2,  W/2)
        x = self.up4(x,  x1)         # (N, base,     H,    W)
        return self.outc(x)          # (N, out_ch,   H,    W)
