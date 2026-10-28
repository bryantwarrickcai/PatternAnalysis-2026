"""
failure_analysis.py
===================
Failure-case autopsy for the 2D U-Net

Reads the ``test_metrics.csv`` produced by ``predict.py``, identifies the
slices with the worst prostate Dice (where the prostate is actually
present), and generates 4-panel visual overlays for those specific slices.

Use this script to answer the Open Research Dilemma:
    - Do the failures occur at the base/apex boundaries?
    - Is the model predicting prostate where there is none (false positives)?
    - Is it missing the prostate entirely (false negatives)?
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import HipMRI2DDataset, SPLITS
from predict import (
    build_model_from_checkpoint,
    load_checkpoint,
    resolve_device,
    save_overlays,
)


def load_metrics(csv_path: Path | str) -> list[dict]:
    """Read the per-slice metrics CSV into a list of dicts."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_worst_cases(
    rows: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Filter to slices that actually contain the prostate, then sort by
    Dice score from lowest (worst) to highest (best).
    """
    # CSV values are strings; convert to correct types
    present_rows = []
    for r in rows:
        if int(r["prostate_present"]) == 1:
            r["dice_prostate"] = float(r["dice_prostate"])
            r["iou_prostate"] = float(r["iou_prostate"])
            r["tp"] = int(r["tp"])
            r["fp"] = int(r["fp"])
            r["fn"] = int(r["fn"])
            present_rows.append(r)

    if not present_rows:
        print("WARNING: No slices in this CSV have prostate_present == 1.")
        return []

    # Sort by Dice ascending (worst first)
    present_rows.sort(key=lambda x: x["dice_prostate"])
    return present_rows[:top_k]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find and visualize the worst prostate segmentation failures."
    )
    parser.add_argument(
        "--csv", type=Path, required=True,
        help="Path to the test_metrics.csv file."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to best_model.pt (needed to generate overlays)."
    )
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of worst cases to report (default: 5).")
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("HipMRI_Study_open/keras_slices_data")
    )
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--target-size", type=int, nargs=2, default=[256, 160],
                        metavar=("H", "W"))
    parser.add_argument("--norm-method", choices=("zscore", "minmax"),
                        default="zscore")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Where to save overlays (default: alongside the CSV)."
    )
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.csv.is_file():
        print(f"error: CSV not found: {args.csv}", file=sys.stderr)
        return 1

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.csv.parent / "failure_autopsy"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    # 1. Find the worst cases
    rows = load_metrics(args.csv)
    worst_cases = find_worst_cases(rows, top_k=args.top_k)

    if not worst_cases:
        return 1

    # 2. Print the diagnostic table
    print("=" * 78)
    print(f" Failure Case Autopsy — {args.top_k} worst slices with prostate present")
    print("=" * 78)
    print(f"{'Rank':<5} | {'Key':<30} | {'Dice':<6} | {'IoU':<6} | {'TP':<5} | {'FP':<5} | {'FN':<5}")
    print("-" * 78)
    for i, r in enumerate(worst_cases, 1):
        print(f"{i:<5} | {r['key']:<30} | {r['dice_prostate']:.3f}  | "
              f"{r['iou_prostate']:.3f}  | {r['tp']:<5} | {r['fp']:<5} | {r['fn']:<5}")
    print()

    # 3. Instantiate model and dataset
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_model_from_checkpoint(ckpt, device)

    # We need the dataset to get the images for these specific slices
    dataset = HipMRI2DDataset(
        args.data_root,
        split=args.split,
        augment=False,
        norm_method=args.norm_method,
        target_size=tuple(args.target_size),
    )

    # Map keys back to their dataset indices
    key_to_idx = {pair.key.__str__(): i for i, pair in enumerate(dataset.pairs)}
    target_indices = []
    for r in worst_cases:
        key = r["key"]
        if key in key_to_idx:
            target_indices.append(key_to_idx[key])
        else:
            print(f"WARNING: Key '{key}' from CSV not found in dataset.")

    if not target_indices:
        print("error: None of the worst case keys were found in the dataset.")
        return 1

    # 4. Create a subset dataloader for ONLY these slices
    subset = Subset(dataset, target_indices)
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False)

    # 5. Generate overlays
    print(f" Generating {len(target_indices)} overlay images in {out_dir}...")
    saved_paths = save_overlays(
        model=model,
        loader=loader,
        device=device,
        out_dir=out_dir,
        num_overlays=len(target_indices),
    )
    
    print(f" Done. Wrote {len(saved_paths)} images.")
    print()
    print("Next steps for your report:")
    print("1. Open the PNGs in the output directory.")
    print("2. Identify if the failure is at the base/apex (boundary failure),")
    print("   or if the model is confusing rectum/bladder for prostate (FP).")
    print("3. Reference these specific filenames in your README.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
