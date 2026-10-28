"""
train.py
========
Training and validation entrypoint for the 2D U-Net

Current commit (Step 3.2) adds the training *setup* only:

- CombinedLoss   : alpha * Dice + (1 - alpha) * CrossEntropy
- Adam optimizer : configurable learning rate
- ReduceLROnPlateau : tracks validation Dice, halves LR after plateau
- Patient-leakage startup guard from dataset.assert_no_patient_overlap
- Full argparse CLI
- Prints a setup summary then exits (no training loop yet)

The training / validation loop arrives in Step 3.3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import (
    NUM_CLASSES,
    SPLITS,
    assert_no_patient_overlap,
    get_dataloaders,
)
from modules import UNet
from utils import dice_loss


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class CombinedLoss(nn.Module):
    """
    Weighted combination of soft Dice and cross-entropy.

        loss = dice_weight * dice_loss(logits, mask_onehot)
             + ce_weight   * CE(logits, mask_int)

    Design notes
    ------------
    - Both terms operate on the same ``logits`` (no activation applied
      in the model). ``dice_loss`` internally applies softmax/sigmoid.
    - The Dice term uses ``ignore_absent=True`` (default), so slices
      with no prostate contribute nothing to the prostate channel —
      critical given that many coronal HipMRI slices have no prostate.
    - The CE term uses ``ignore_index=255`` and a defensive mapping
      (``_safe_int_target``) that converts any out-of-range integer
      label to that ignore value, so the loss cannot produce NaN
      if a future data revision introduces unexpected labels.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        ignore_index: int = 255,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.ignore_index = int(ignore_index)
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=self.ignore_index
        )

    def _safe_int_target(
        self, mask_int: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        """Map any out-of-range integer label to ``ignore_index``."""
        valid = (mask_int >= 0) & (mask_int < num_classes)
        return torch.where(
            valid,
            mask_int,
            torch.full_like(mask_int, self.ignore_index),
        )

    def forward(
        self,
        logits: torch.Tensor,
        mask_onehot: torch.Tensor,
        mask_int: torch.Tensor,
    ) -> torch.Tensor:
        dice = dice_loss(logits, mask_onehot, from_logits=True)
        ce = self.ce(
            logits, self._safe_int_target(mask_int, num_classes=logits.shape[1])
        )
        return self.dice_weight * dice + self.ce_weight * ce


def build_loss(args: argparse.Namespace, device: torch.device) -> CombinedLoss:
    """Construct :class:`CombinedLoss` from CLI arguments, with validation."""
    if args.dice_weight < 0 or args.ce_weight < 0:
        raise ValueError(
            f"loss weights must be >= 0, got "
            f"dice_weight={args.dice_weight}, ce_weight={args.ce_weight}"
        )
    if args.dice_weight + args.ce_weight <= 0:
        raise ValueError("dice_weight + ce_weight must be > 0")

    class_weights = None
    if args.ce_class_weights:
        values = [float(v) for v in args.ce_class_weights.split(",")]
        if len(values) != NUM_CLASSES:
            raise ValueError(
                f"--ce-class-weights must have {NUM_CLASSES} values "
                f"(comma-separated), got {len(values)}"
            )
        class_weights = torch.tensor(values, dtype=torch.float32, device=device)

    return CombinedLoss(
        dice_weight=args.dice_weight,
        ce_weight=args.ce_weight,
        ignore_index=args.ignore_index,
        class_weights=class_weights,
    )


# ---------------------------------------------------------------------------
# Model, optimizer, scheduler builders
# ---------------------------------------------------------------------------
def build_model(args: argparse.Namespace) -> UNet:
    return UNet(
        in_channels=1,
        out_channels=NUM_CLASSES,
        base_features=args.base_features,
        dropout=args.dropout,
    )


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=args.lr)


def build_scheduler(
    optimizer: optim.Optimizer, args: argparse.Namespace
) -> optim.lr_scheduler.ReduceLROnPlateau | None:
    if args.no_scheduler:
        return None
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=1e-6,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2D U-Net training setup for HipMRI prostate segmentation."
    )
    # Data
    p.add_argument(
        "--data-root", type=Path,
        default=Path("HipMRI_Study_open/keras_slices_data"),
        help="Folder containing keras_slices_* subfolders.",
    )
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (0 is safest on Windows).")
    p.add_argument("--no-augment", action="store_true",
                   help="Disable training augmentation.")
    p.add_argument("--norm-method", choices=("zscore", "minmax"),
                   default="zscore", help="Image normalization method.")
    # Model
    p.add_argument("--base-features", type=int, default=32,
                   help="First-level channel width of the U-Net.")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Spatial dropout inside every DoubleConv (0 disables).")
    # Optimization
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dice-weight", type=float, default=0.5)
    p.add_argument("--ce-weight", type=float, default=0.5)
    p.add_argument("--ignore-index", type=int, default=255)
    p.add_argument(
        "--ce-class-weights", type=str, default=None,
        help=(
            f"Comma-separated per-class CE weights, {NUM_CLASSES} values "
            "(e.g. '1,0.1,2,4,4,8'). Default: None (uniform)."
        ),
    )
    # Scheduler
    p.add_argument("--no-scheduler", action="store_true")
    p.add_argument("--scheduler-factor", type=float, default=0.5)
    p.add_argument("--scheduler-patience", type=int, default=5)
    # Reproducibility & device
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    # Output
    p.add_argument("--out-dir", type=Path, default=Path("runs"),
                   help="Where checkpoints and curves will be written (Step 3.4).")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Setup entrypoint (training loop in Step 3.3)
# ---------------------------------------------------------------------------
def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _seed_everything(args.seed)
    device = resolve_device(args.device)

    print("=" * 62)
    print(" 2D U-Net — HipMRI prostate segmentation")
    print("=" * 62)
    print(f" Torch {torch.__version__} | device = {device}")
    if device.type == "cuda":
        print(f" GPU   {torch.cuda.get_device_name(device)}")
    print()

    # ---- Patient-level split audit (hard safety guard) -------------------
    print(f"Auditing patient-level splits in: {args.data_root}")
    audit = assert_no_patient_overlap(args.data_root)
    for split in SPLITS:
        print(
            f"  {split:9s}  {audit.pair_counts[split]:5d} slices  "
            f"{len(audit.patient_ids[split]):2d} patients"
        )
    print("  -> CLEAN (no patient appears in more than one split)")
    print()

    # ---- Model -----------------------------------------------------------
    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: UNet(in=1, out={NUM_CLASSES}, base={args.base_features}, "
          f"dropout={args.dropout})")
    print(f"       {n_params:,} trainable parameters")
    print()

    # ---- Loss / optimizer / scheduler ------------------------------------
    loss_fn = build_loss(args, device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)

    print(f"Loss: {args.dice_weight:.2f}*Dice + {args.ce_weight:.2f}*CE "
          f"(ignore_index={args.ignore_index}, "
          f"ce_class_weights={'set' if args.ce_class_weights else 'none'})")
    print(f"Optimizer: Adam(lr={args.lr})")
    if scheduler is not None:
        print(
            f"Scheduler: ReduceLROnPlateau(mode=max, factor="
            f"{args.scheduler_factor}, patience={args.scheduler_patience})"
        )
    else:
        print("Scheduler: disabled")
    print()

    # ---- Data loaders ----------------------------------------------------
    print(f"Building DataLoaders (batch_size={args.batch_size}, "
          f"num_workers={args.num_workers}, augment={not args.no_augment})")
    loaders = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_augment=not args.no_augment,
        norm_method=args.norm_method,
        seed=args.seed,
        pin_memory=(device.type == "cuda"),
    )
    for split, loader in loaders.items():
        print(
            f"  {split:9s}  {len(loader.dataset):5d} slices  "
            f"{len(loader):4d} batches"
        )
    print()

    print(f"Output directory (created on first checkpoint): {args.out_dir}")
    print()
    print("Setup complete. The training loop arrives in Step 3.3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
