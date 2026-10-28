"""
predict.py
==========
Inference and evaluation for the 2D U-Net

Loads a trained checkpoint, evaluates on a held-out split, and writes
per-slice metrics to CSV. A per-class summary is printed to stdout.

The CSV written by this script is the input to the failure-case
autopsy (Step 5.3) — it contains enough detail to rank slices by Dice
and pick the worst 3-5 for qualitative analysis.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from dataset import (
    NUM_CLASSES,
    PROSTATE_LABEL,
    SPLITS,
    get_dataloaders,
)
from modules import UNet
from utils import dice_score, iou_score


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_checkpoint(
    path: Path | str, device: torch.device
) -> tuple[dict, dict]:
    """
    Load a checkpoint saved by ``train.py``'s ``fit()`` function.

    Uses ``weights_only=True`` (the PyTorch 2.6+ default). The
    checkpoint only contains primitives + tensors, so this is safe
    and portable across OSes (no ``Path`` objects were pickled).

    Returns
    -------
    model_state : dict
        The model ``state_dict``.
    saved_args : dict
        The ``vars(args)`` dict from the training run (JSON-safe primitives).
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "model_state" not in ckpt:
        raise KeyError(
            f"Checkpoint {path} is missing 'model_state'. "
            f"Keys present: {list(ckpt.keys())}"
        )
    return ckpt["model_state"], ckpt.get("args", {})


def build_model_from_checkpoint(
    ckpt: dict, device: torch.device
) -> UNet:
    """Instantiate ``UNet`` with the same architecture used during training."""
    saved_args = ckpt.get("args", {})
    model = UNet(
        in_channels=1,
        out_channels=NUM_CLASSES,
        base_features=int(saved_args.get("base_features", 32)),
        dropout=float(saved_args.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-slice metrics
# ---------------------------------------------------------------------------
def predict_split(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    smooth: float = 1.0,
) -> list[dict]:
    """
    Run inference on a loader and return per-slice metrics.

    Each row is a dict with:
      key               : str   e.g. "case_003_week_2_slice_15"
      dice_prostate     : float soft Dice on the prostate channel
      iou_prostate      : float soft IoU on the prostate channel
      dice_macro        : float mean Dice over all 6 classes (soft)
      iou_macro         : float mean IoU over all 6 classes (soft)
      tp                : int   true-positive pixels (prostate, thr=0.5)
      fp                : int   false-positive pixels
      fn                : int   false-negative pixels
      prostate_present  : int   1 if the target has any prostate pixel, else 0
    """
    model.eval()
    rows: list[dict] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="predict", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            onehot = batch["mask_onehot"].to(device, non_blocking=True)
            prostate = batch["prostate"].to(device, non_blocking=True)
            keys = batch["key"]

            logits = model(images)

            # --- Soft per-class metrics ---
            d_all = dice_score(logits, onehot)
            i_all = iou_score(logits, onehot)

            # --- Hard per-slice prostate metrics ---
            probs_prost = torch.softmax(logits, dim=1)[:, PROSTATE_LABEL]  # (N, H, W)
            pred_hard = (probs_prost > 0.5).float()
            target_hard = prostate.squeeze(1)                              # (N, H, W)

            # --- Aggregate per-slice ---
            bs = images.size(0)
            for i in range(bs):
                p = pred_hard[i]
                t = target_hard[i]
                tp = float((p * t).sum().item())
                fp = float((p * (1.0 - t)).sum().item())
                fn = float(((1.0 - p) * t).sum().item())
                union = tp + fp + fn
                dice_prost = (2.0 * tp + smooth) / (p.sum().item() + t.sum().item() + smooth)
                iou_prost = (tp + smooth) / (union + smooth)

                rows.append({
                    "key": str(keys[i]),
                    "dice_prostate": float(dice_prost),
                    "iou_prostate": float(iou_prost),
                    "dice_macro": float(d_all["macro"]),
                    "iou_macro": float(i_all["macro"]),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "prostate_present": int(t.sum().item() > 0),
                })

    return rows


# ---------------------------------------------------------------------------
# CSV / summary
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "key", "dice_prostate", "iou_prostate", "dice_macro", "iou_macro",
    "tp", "fp", "fn", "prostate_present",
]


def write_csv(rows: list[dict], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-slice metrics into overall numbers for printing."""
    n = len(rows)
    if n == 0:
        return {"n": 0}

    prost_present = [r for r in rows if r["prostate_present"] == 1]
    n_present = len(prost_present)
    return {
        "n": n,
        "n_present": n_present,
        "dice_prostate_all": sum(r["dice_prostate"] for r in rows) / n,
        "dice_prostate_present": (
            sum(r["dice_prostate"] for r in prost_present) / n_present
            if n_present else 0.0
        ),
        "iou_prostate_all": sum(r["iou_prostate"] for r in rows) / n,
        "iou_prostate_present": (
            sum(r["iou_prostate"] for r in prost_present) / n_present
            if n_present else 0.0
        ),
        "dice_macro": sum(r["dice_macro"] for r in rows) / n,
        "iou_macro": sum(r["iou_macro"] for r in rows) / n,
        "total_tp": sum(r["tp"] for r in rows),
        "total_fp": sum(r["fp"] for r in rows),
        "total_fn": sum(r["fn"] for r in rows),
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
        description="Run inference with a trained 2D U-Net and save per-slice metrics."
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to best_model.pt (e.g. runs/best_model.pt).")
    p.add_argument(
        "--data-root", type=Path,
        default=Path("HipMRI_Study_open/keras_slices_data"),
        help="Folder containing keras_slices_* subfolders.",
    )
    p.add_argument("--split", choices=SPLITS, default="test",
                   help="Split to evaluate (train/validate/test).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--target-size", type=int, nargs=2, default=[256, 160],
                   metavar=("H", "W"),
                   help="Must match the training target size (default 256 160).")
    p.add_argument("--norm-method", choices=("zscore", "minmax"),
                   default="zscore")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="CSV path for per-slice metrics. "
                        "Default: <checkpoint_dir>/<split>_metrics.csv")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)

    if not args.checkpoint.is_file():
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = args.checkpoint.parent / f"{args.split}_metrics.csv"

    print("=" * 62)
    print(f" Inference on split = {args.split}")
    print("=" * 62)
    print(f" Checkpoint : {args.checkpoint}")
    print(f" Device     : {device}")
    print()

    # --- Model ---
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False) \
        if False else None  # placeholder; we use load_checkpoint below
    model_state, saved_args = load_checkpoint(args.checkpoint, device)
    model = UNet(
        in_channels=1,
        out_channels=NUM_CLASSES,
        base_features=int(saved_args.get("base_features", 32)),
        dropout=float(saved_args.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f" Model      : UNet(base={saved_args.get('base_features', 32)}) "
          f"| {n_params:,} params")

    # --- Data ---
    loaders = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_augment=False,
        norm_method=args.norm_method,
        seed=None,
        pin_memory=(device.type == "cuda"),
        target_size=tuple(args.target_size),
    )
    loader = loaders[args.split]
    print(f" Data       : {len(loader.dataset)} slices  "
          f"{len(loader)} batches  (target_size={tuple(args.target_size)})")
    print()

    # --- Inference ---
    rows = predict_split(model, loader, device)
    summary = summarize(rows)

    write_csv(rows, out_csv)
    print(f" Wrote per-slice metrics -> {out_csv}")
    print()

    # --- Summary ---
    print("Summary")
    print("-" * 62)
    print(f"  slices                 : {summary['n']}")
    print(f"  slices with prostate   : {summary['n_present']}")
    print()
    print(f"  Dice    prostate (all) : {summary['dice_prostate_all']:.4f}")
    print(f"  Dice    prostate (pres): {summary['dice_prostate_present']:.4f}")
    print(f"  IoU     prostate (all) : {summary['iou_prostate_all']:.4f}")
    print(f"  IoU     prostate (pres): {summary['iou_prostate_present']:.4f}")
    print()
    print(f"  Dice    macro (mean)   : {summary['dice_macro']:.4f}")
    print(f"  IoU     macro (mean)   : {summary['iou_macro']:.4f}")
    print()
    print(f"  prostate TP total      : {summary['total_tp']:,}")
    print(f"  prostate FP total      : {summary['total_fp']:,}")
    print(f"  prostate FN total      : {summary['total_fn']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
