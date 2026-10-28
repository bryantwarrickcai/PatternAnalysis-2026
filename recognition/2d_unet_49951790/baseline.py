"""
baseline.py
===========
Simple CNN baseline for the HipMRI prostate segmentation task

Purpose
-------
The spec asks for "a simple baseline model ... to establish an initial
performance benchmark." We compare the 2D U-Net against a **fully
convolutional network without skip connections**.

Why no skip connections specifically?
-------------------------------------
The central claim of the U-Net paper is that skip connections fuse
high-resolution spatial detail (from the encoder) with low-resolution
semantic context (from the bottleneck), which is precisely what medical
boundary segmentation requires. A baseline *without* skip connections
tests this hypothesis directly:

- If the U-Net significantly beats this baseline on prostate Dice,
  the skip connections are demonstrably valuable.
- If not, then the added complexity of the U-Net is not justified for
  this Easy-difficulty task — an honest engineering finding either way.

The baseline still has an encoder/decoder structure (to see global
context) and uses the same loss, optimiser, augmentation, and splits as
the U-Net. Only the skip connections are missing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dataset import NUM_CLASSES


class BaselineCNN(nn.Module):
    """
    Simple encoder-decoder CNN with NO skip connections.

    Architecture:
        in  (N, 1, H, W)
        down1  Conv3x3 stride 2 + BN + ReLU -> (N, base,     H/2,  W/2)
        down2  Conv3x3 stride 2 + BN + ReLU -> (N, 2*base,   H/4,  W/4)
        up1    ConvT 2x2 stride 2 + BN + ReLU -> (N, base,   H/2,  W/2)
        up2    ConvT 2x2 stride 2 + BN + ReLU -> (N, base,   H,    W)
        outc   Conv1x1                        -> (N, out_ch, H,    W)

    This is the smallest reasonable baseline: it must learn to
    reconstruct spatial detail purely from the compressed bottleneck,
    which it fundamentally cannot do for thin structures like the
    prostate boundary. That is the point.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = NUM_CLASSES,
        base_features: int = 16,
    ) -> None:
        super().__init__()
        b = base_features

        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, b, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(b, b * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(b * 2),
            nn.ReLU(inplace=True),
        )

        # Decoder (no skips)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(b, b, kernel_size=2, stride=2),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )

        # Output head
        self.outc = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down1(x)
        x = self.down2(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.outc(x)


def build_baseline(
    in_channels: int = 1,
    out_channels: int = NUM_CLASSES,
    base_features: int = 16,
) -> BaselineCNN:
    """Convenience constructor (mirrors ``train.build_model``)."""
    return BaselineCNN(
        in_channels=in_channels,
        out_channels=out_channels,
        base_features=base_features,
    )


def _main() -> None:
    """Standalone smoke test — print shapes and parameter count."""
    import argparse

    parser = argparse.ArgumentParser(description="Baseline CNN smoke test.")
    parser.add_argument("--h", type=int, default=256)
    parser.add_argument("--w", type=int, default=160)
    parser.add_argument("--base-features", type=int, default=16)
    args = parser.parse_args()

    model = build_baseline(base_features=args.base_features).eval()
    x = torch.randn(1, 1, args.h, args.w)
    with torch.no_grad():
        y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BaselineCNN(base={args.base_features})")
    print(f"  input  : {tuple(x.shape)}")
    print(f"  output : {tuple(y.shape)}")
    print(f"  params : {n_params:,}")


if __name__ == "__main__":
    _main()
