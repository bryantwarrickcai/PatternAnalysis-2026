"""
train.py
========
Training and validation entrypoint for the 2D U-Net

Contains:
- CombinedLoss   : alpha * Dice + (1 - alpha) * CrossEntropy
- train_one_epoch / validate  : metric computation per epoch
- fit            : full training loop with best-checkpoint selection
- CLI (parse_args + main)

Metrics per epoch
-----------------
- loss                   : weighted combined loss
- dice_macro             : mean Dice over all classes (batch-weighted)
- dice_macro_present     : mean Dice over classes present in any batch
- dice_prostate_all      : mean prostate Dice over ALL validation slices
- dice_prostate_present  : mean prostate Dice over slices where the
                           prostate is actually present (assignment target)

Best checkpoint is selected by ``dice_prostate_present``, matching the
spec's target metric. The full history (per epoch) is written to
``<out_dir>/history.json`` for plotting in Step 3.4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from baseline import build_baseline
from dataset import (
    NUM_CLASSES,
    PROSTATE_LABEL,
    SPLITS,
    assert_no_patient_overlap,
    get_dataloaders,
)
from modules import UNet
from utils import dice_loss, dice_score


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class CombinedLoss(nn.Module):
    """
    Weighted combination of soft Dice and cross-entropy.

        loss = dice_weight * dice_loss(logits, mask_onehot)
             + ce_weight   * CE(logits, mask_int)
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
        valid = (mask_int >= 0) & (mask_int < num_classes)
        return torch.where(
            valid, mask_int, torch.full_like(mask_int, self.ignore_index)
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
def build_model(args: argparse.Namespace) -> nn.Module:
    if getattr(args, "model", "unet") == "baseline":
        return build_baseline(
            in_channels=1,
            out_channels=NUM_CLASSES,
            base_features=args.base_features,
        )
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
# Per-epoch metric helpers
# ---------------------------------------------------------------------------
def _prostate_dice_per_slice(
    logits: torch.Tensor,
    prostate: torch.Tensor,
    smooth: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Dice on the prostate channel for every slice in the batch.

    Parameters
    ----------
    logits   : (N, C, H, W) raw logits.
    prostate : (N, 1, H, W) binary prostate target (float 0/1).

    Returns
    -------
    dice_values  : (N,) Dice per slice, range ~[0, 1].
    is_present   : (N,) bool, True where the target has at least 1 prostate pixel.
    """
    probs = torch.softmax(logits, dim=1)[:, PROSTATE_LABEL]   # (N, H, W)
    target = prostate.squeeze(1)                              # (N, H, W)
    inter = (probs * target).sum(dim=(1, 2))                  # (N,)
    union = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))    # (N,)
    dice = (2.0 * inter + smooth) / (union + smooth)          # (N,)
    is_present = target.sum(dim=(1, 2)) > 0                   # (N,)
    return dice, is_present


def _init_metric_accumulators(num_classes: int, device: torch.device) -> dict:
    return {
        "loss_sum": 0.0,
        "n_samples": 0,
        "dice_per_class_sum": torch.zeros(num_classes, device=device),
        "dice_macro_present_sum": 0.0,
        "prostate_dice_parts": [],
        "prostate_present_parts": [],
    }


def _finalize_metrics(
    acc: dict, num_classes: int
) -> dict:
    n = max(acc["n_samples"], 1)
    dice_per_class = (acc["dice_per_class_sum"] / n).tolist()
    dice_macro = sum(dice_per_class) / num_classes
    dice_macro_present = acc["dice_macro_present_sum"] / n

    ps_dice = (
        torch.cat(acc["prostate_dice_parts"])
        if acc["prostate_dice_parts"]
        else torch.zeros(1)
    )
    ps_present = (
        torch.cat(acc["prostate_present_parts"])
        if acc["prostate_present_parts"]
        else torch.zeros(1, dtype=torch.bool)
    )
    dice_prostate_all = float(ps_dice.mean().item())
    dice_prostate_present = (
        float(ps_dice[ps_present].mean().item()) if ps_present.any() else 0.0
    )

    return {
        "loss": acc["loss_sum"] / n,
        "dice_macro": dice_macro,
        "dice_macro_present": dice_macro_present,
        "dice_per_class": dice_per_class,
        "dice_prostate_all": dice_prostate_all,
        "dice_prostate_present": dice_prostate_present,
        "n_slices": int(acc["n_samples"]),
        "n_prostate_present": int(ps_present.sum().item()),
    }


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int = 1,
    epoch_total: int = 1,
) -> dict:
    """One full pass over the training loader. Returns metric dict."""
    model.train()
    acc = _init_metric_accumulators(NUM_CLASSES, device)
    pbar = tqdm(loader, desc=f"train {epoch}/{epoch_total}", leave=False)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        onehot = batch["mask_onehot"].to(device, non_blocking=True)
        mask_int = batch["mask_int"].to(device, non_blocking=True)
        prostate = batch["prostate"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, onehot, mask_int)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        acc["loss_sum"] += loss.item() * bs
        acc["n_samples"] += bs

        with torch.no_grad():
            d = dice_score(logits.detach(), onehot)
            acc["dice_per_class_sum"] += (
                torch.tensor(d["per_class"], device=device) * bs
            )
            acc["dice_macro_present_sum"] += d["macro_present"] * bs

            ps_dice, ps_present = _prostate_dice_per_slice(
                logits.detach(), prostate
            )
            acc["prostate_dice_parts"].append(ps_dice.cpu())
            acc["prostate_present_parts"].append(ps_present.cpu())

        pbar.set_postfix(
            loss=f"{acc['loss_sum'] / acc['n_samples']:.4f}"
        )

    return _finalize_metrics(acc, NUM_CLASSES)


def validate(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int = 1,
    epoch_total: int = 1,
) -> dict:
    """Full pass over a validation/test loader with no gradients."""
    model.eval()
    acc = _init_metric_accumulators(NUM_CLASSES, device)
    pbar = tqdm(loader, desc=f"val   {epoch}/{epoch_total}", leave=False)

    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            onehot = batch["mask_onehot"].to(device, non_blocking=True)
            mask_int = batch["mask_int"].to(device, non_blocking=True)
            prostate = batch["prostate"].to(device, non_blocking=True)

            logits = model(images)
            loss = loss_fn(logits, onehot, mask_int)

            bs = images.size(0)
            acc["loss_sum"] += loss.item() * bs
            acc["n_samples"] += bs

            d = dice_score(logits, onehot)
            acc["dice_per_class_sum"] += (
                torch.tensor(d["per_class"], device=device) * bs
            )
            acc["dice_macro_present_sum"] += d["macro_present"] * bs

            ps_dice, ps_present = _prostate_dice_per_slice(logits, prostate)
            acc["prostate_dice_parts"].append(ps_dice.cpu())
            acc["prostate_present_parts"].append(ps_present.cpu())

            pbar.set_postfix(
                loss=f"{acc['loss_sum'] / acc['n_samples']:.4f}"
            )

    return _finalize_metrics(acc, NUM_CLASSES)


def _args_to_jsonable(args: argparse.Namespace) -> dict:
    """
    Convert an argparse Namespace to a JSON-safe dict of primitives.

    Necessary because we store ``vars(args)`` inside the checkpoint,
    and PyTorch 2.6 changed ``torch.load`` to default ``weights_only=True``,
    which rejects ``pathlib.Path`` (and other non-allowlisted classes).
    Converting Paths to strings keeps the checkpoint loadable with the
    safe default.
    """
    out: dict = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
        else:
            # Fallback for any exotic type a future flag might introduce.
            out[key] = str(value)
    return out


def plot_history(
    history: dict,
    out_dir: Path | str,
    best_epoch: int | None = None,
) -> list[Path]:
    """
    Render training curves as PNGs.

    Files written
    -------------
    - ``loss_curve.png``   : train vs. validate loss, with best-epoch marker
    - ``dice_curves.png``  : three panels (macro-present, prostate-present,
                             prostate-all), each train vs. validate
    - ``lr_curve.png``     : learning-rate schedule (log y-axis)

    Skips gracefully if matplotlib is not installed — training still
    works; only the plots are lost.

    Parameters
    ----------
    history     : dict with keys "train", "validate", "lr" (per-epoch lists).
    out_dir     : destination folder (created if missing).
    best_epoch  : optional 1-based epoch number to mark with a dashed line.
                  If None, derived from validate["dice_prostate_present"].

    Returns the list of PNG paths actually written (empty if matplotlib
    is unavailable).
    """
    try:
        import matplotlib
        # Force a non-interactive backend so savefig works headless
        # (Rangpur HPC, CI). If matplotlib was already imported, force=True
        # safely reconfigures. We never call plt.show(), so Agg is enough.
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "  [plot] matplotlib not installed — skipping curve plots. "
            "Install with: pip install matplotlib"
        )
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = history["train"]
    val = history["validate"]
    epochs = list(range(1, len(train) + 1))
    lr = history.get("lr", [])

    if best_epoch is None and val:
        prost_series = [m["dice_prostate_present"] for m in val]
        best_epoch = int(
            max(range(len(prost_series)), key=lambda i: prost_series[i])
        ) + 1

    written: list[Path] = []

    # --- Loss ---
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(epochs, [m["loss"] for m in train], marker="o", label="train")
    ax.plot(epochs, [m["loss"] for m in val], marker="s", label="validate")
    if best_epoch is not None:
        ax.axvline(
            best_epoch, color="green", linestyle="--", alpha=0.6,
            label=f"best epoch ({best_epoch})",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Combined loss (0.5 · Dice + 0.5 · CE)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = out_dir / "loss_curve.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # --- Dice (3 panels) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    panels = [
        ("dice_macro_present",   "macro Dice\n(present classes)"),
        ("dice_prostate_present", "prostate Dice\n(present slices)"),
        ("dice_prostate_all",    "prostate Dice\n(all slices)"),
    ]
    for ax, (key, title) in zip(axes, panels):
        ax.plot(epochs, [m[key] for m in train], marker="o", label="train")
        ax.plot(epochs, [m[key] for m in val], marker="s", label="validate")
        if best_epoch is not None:
            ax.axvline(
                best_epoch, color="green", linestyle="--", alpha=0.6,
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel("Dice")
        ax.set_title(title)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "dice_curves.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # --- LR (optional) ---
    if lr:
        fig, ax = plt.subplots(figsize=(7, 3.5), dpi=150)
        ax.plot(epochs, lr, marker="o")
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.set_title("Learning-rate schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / "lr_curve.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    return written


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------
def fit(
    model: nn.Module,
    loaders: dict,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    """
    Run the full training loop.

    Saves:
      <out_dir>/best_model.pt   whenever ``dice_prostate_present`` improves
      <out_dir>/history.json    per-epoch metric history (for Step 3.4)
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, list] = {"train": [], "validate": [], "lr": []}
    best_prostate_dice = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(
            model, loaders["train"], loss_fn, optimizer, device,
            epoch=epoch, epoch_total=args.epochs,
        )
        val_m = validate(
            model, loaders["validate"], loss_fn, device,
            epoch=epoch, epoch_total=args.epochs,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(val_m["dice_prostate_present"])

        history["train"].append(train_m)
        history["validate"].append(val_m)
        history["lr"].append(current_lr)

        improved = val_m["dice_prostate_present"] > best_prostate_dice
        marker = " *" if improved else ""
        if improved:
            best_prostate_dice = val_m["dice_prostate_present"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_metrics": val_m,
                    "args": _args_to_jsonable(args),
                    "model_type": args.model,
                },
                out_dir / "best_model.pt",
            )

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss={train_m['loss']:.4f} "
            f"dice_pres={train_m['dice_macro_present']:.4f} "
            f"prost_pres={train_m['dice_prostate_present']:.4f} "
            f"({train_m['n_prostate_present']}/{train_m['n_slices']}) | "
            f"val loss={val_m['loss']:.4f} "
            f"dice_pres={val_m['dice_macro_present']:.4f} "
            f"prost_pres={val_m['dice_prostate_present']:.4f} "
            f"({val_m['n_prostate_present']}/{val_m['n_slices']}) | "
            f"lr={current_lr:.2e}{marker}"
        )

    history_path = out_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    plot_paths: list[Path] = []
    if not getattr(args, "no_plot", False):
        print()
        print("Rendering training curves...")
        plot_paths = plot_history(history, out_dir, best_epoch=best_epoch)
        for p in plot_paths:
            print(f"  wrote {p}")

    return {
        "best_prostate_dice": best_prostate_dice,
        "best_epoch": best_epoch,
        "history_path": str(history_path),
        "checkpoint_path": str(out_dir / "best_model.pt"),
        "plot_paths": [str(p) for p in plot_paths],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2D U-Net training for HipMRI prostate segmentation."
    )
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
                   default="zscore")
    p.add_argument(
        "--target-size", type=int, nargs=2, default=[256, 160],
        metavar=("H", "W"),
        help="Pad/crop all slices to this size before batching. "
             "Both values should be divisible by 16 for a clean U-Net.",
    )
    p.add_argument(
        "--limit-train", type=int, default=None,
        help="Use only the first N training slices (fast smoke test).",
    )
    p.add_argument(
        "--limit-val", type=int, default=None,
        help="Use only the first N validation slices (fast smoke test).",
    )
    p.add_argument("--base-features", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dice-weight", type=float, default=0.5)
    p.add_argument("--ce-weight", type=float, default=0.5)
    p.add_argument("--ignore-index", type=int, default=255)
    p.add_argument("--ce-class-weights", type=str, default=None)
    p.add_argument("--no-scheduler", action="store_true")
    p.add_argument("--scheduler-factor", type=float, default=0.5)
    p.add_argument("--scheduler-patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--out-dir", type=Path, default=Path("runs"))
    p.add_argument(
        "--no-plot", action="store_true",
        help="Skip rendering training curves at the end of fit().",
    )
    p.add_argument(
        "--model", choices=("unet", "baseline"), default="unet",
        help="Model architecture: 'unet' (default) or 'baseline'."
    )
    return p.parse_args(argv)


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.epochs < 1:
        raise SystemExit(f"--epochs must be >= 1, got {args.epochs}")

    _seed_everything(args.seed)
    device = resolve_device(args.device)

    print("=" * 62)
    print(" 2D U-Net — HipMRI prostate segmentation")
    print("=" * 62)
    print(f" Torch {torch.__version__} | device = {device}")
    if device.type == "cuda":
        print(f" GPU   {torch.cuda.get_device_name(device)}")
    print()

    print(f"Auditing patient-level splits in: {args.data_root}")
    audit = assert_no_patient_overlap(args.data_root)
    for split in SPLITS:
        print(
            f"  {split:9s}  {audit.pair_counts[split]:5d} slices  "
            f"{len(audit.patient_ids[split]):2d} patients"
        )
    print("  -> CLEAN")
    print()

    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.model == "baseline":
        print(f"Model: BaselineCNN(base={args.base_features}) | {n_params:,} params")
    else:
        print(f"Model: UNet(base={args.base_features}, dropout={args.dropout}) "
              f"| {n_params:,} params")

    loss_fn = build_loss(args, device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)

    print(f"Loss: {args.dice_weight:.2f}*Dice + {args.ce_weight:.2f}*CE")
    print(f"Optimizer: Adam(lr={args.lr}) | scheduler: "
          f"{'off' if scheduler is None else 'plateau'}")
    print()

    loaders = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_augment=not args.no_augment,
        norm_method=args.norm_method,
        seed=args.seed,
        pin_memory=(device.type == "cuda"),
        target_size=tuple(args.target_size),
        limit_train=args.limit_train,
        limit_val=args.limit_val,
    )
    for split, loader in loaders.items():
        print(f"  {split:9s}  {len(loader.dataset):5d} slices  "
              f"{len(loader):4d} batches")
    print()

    print(f"Training for {args.epochs} epoch(s); out_dir = {args.out_dir}")
    print()
    result = fit(model, loaders, loss_fn, optimizer, scheduler, device, args)

    print()
    print("=" * 62)
    print("Training complete")
    print(f"  Best epoch                : {result['best_epoch']}")
    print(f"  Best dice_prostate_present: {result['best_prostate_dice']:.4f}")
    print(f"  Checkpoint                : {result['checkpoint_path']}")
    print(f"  History                   : {result['history_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
