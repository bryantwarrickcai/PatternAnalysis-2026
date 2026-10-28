"""
check_model.py
==============
Standalone smoke test for the 2D U-Net in ``modules.py``.

Purpose
-------
Prove, without needing any data, that:
  - ``modules.py`` imports cleanly.
  - ``UNet`` instantiates at the project's real configuration
    (``in_channels=1``, ``out_channels=6``, ``base_features=32``).
  - A forward pass on a 256x128 dummy tensor returns the expected shape.
  - A forward pass on a binary (``out_channels=1``) configuration,
    used later by the baseline model, also returns the expected shape.
  - Every parameter is trainable (``requires_grad=True``).

This is not a unit test in the strict sense — it has no fixtures and
no parametrization. It is a hard-fail sanity check meant to be run by
hand or from CI before training:

    python check_model.py

The script exits with code 0 on success and code 1 on any failure.
"""

from __future__ import annotations

import sys

import torch

from modules import UNet


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def forward_shape_check(
    in_channels: int,
    out_channels: int,
    base_features: int,
    h: int,
    w: int,
) -> None:
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_features=base_features,
    ).eval()

    x = torch.randn(1, in_channels, h, w)
    with torch.no_grad():
        y = model(x)

    expected = (1, out_channels, h, w)
    if y.shape != expected:
        raise AssertionError(
            f"forward shape mismatch: expected {expected}, got {tuple(y.shape)}"
        )

    params = count_trainable_parameters(model)
    print(
        f"  UNet(in={in_channels}, out={out_channels}, base={base_features}) "
        f"input={tuple(x.shape)} -> output={tuple(y.shape)}  "
        f"params={params:,}"
    )


def all_parameters_trainable() -> None:
    model = UNet(in_channels=1, out_channels=6, base_features=32)
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    if frozen:
        raise AssertionError(f"unexpectedly frozen parameters: {frozen}")
    print("  all parameters are trainable")


def main() -> int:
    print("check_model.py — 2D U-Net smoke test")
    try:
        # Project configuration (multi-class HipMRI)
        forward_shape_check(
            in_channels=1, out_channels=6, base_features=32, h=256, w=128
        )
        # Baseline configuration (binary prostate, used in Phase 5)
        forward_shape_check(
            in_channels=1, out_channels=1, base_features=16, h=256, w=128
        )
        all_parameters_trainable()
    except AssertionError as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1
    print("\nall model checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
