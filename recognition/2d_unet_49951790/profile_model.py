"""
profile_model.py
================
Resource profiling for the 2D U-Net and the Baseline CNN

Measures:
- Total trainable parameters
- Peak GPU VRAM (MB) — skipped if CUDA is unavailable
- Mean inference latency (ms) over a batch of 1
- Maximum inference throughput (FPS) over a larger batch

Outputs a formatted Markdown table ready for the README resource
profiling section.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from baseline import build_baseline
from dataset import NUM_CLASSES
from modules import UNet


def get_peak_vram_mb() -> float:
    """Returns peak GPU memory allocated in MB, or 0.0 if CUDA is unavailable."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def reset_vram_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def measure_latency(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    warmup: int = 10,
    iterations: int = 50,
) -> tuple[float, float]:
    """
    Measures average latency (ms) and throughput (FPS) for a single forward pass.

    Returns
    -------
    latency_ms : float
    throughput_fps : float
    """
    model.eval()
    model.to(device)
    x = input_tensor.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timing
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    total_time_s = end - start
    latency_ms = (total_time_s / iterations) * 1000.0
    throughput_fps = iterations / total_time_s
    return latency_ms, throughput_fps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile parameters, VRAM, and latency for U-Net vs. Baseline."
    )
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for latency measurement (default: 1).")
    parser.add_argument("--target-size", type=int, nargs=2, default=[256, 160],
                        metavar=("H", "W"), help="Input resolution.")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of forward passes to average over.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    H, W = args.target_size
    input_tensor = torch.randn(args.batch_size, 1, H, W)

    print("=" * 70)
    print(f" Resource Profiling | Device: {device} | Input: {args.batch_size}x1x{H}x{W}")
    print("=" * 70)
    print()

    results = []

    # 1. U-Net (default architecture)
    reset_vram_stats()
    unet = UNet(in_channels=1, out_channels=NUM_CLASSES, base_features=32)
    unet_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    unet_latency, unet_fps = measure_latency(
        unet, input_tensor, device, iterations=args.iterations
    )
    unet_vram = get_peak_vram_mb()
    results.append({
        "Model": "U-Net",
        "Params": unet_params,
        "VRAM (MB)": unet_vram,
        "Latency (ms)": unet_latency,
        "FPS": unet_fps,
    })

    # 2. Baseline CNN (default architecture)
    reset_vram_stats()
    baseline = build_baseline(in_channels=1, out_channels=NUM_CLASSES, base_features=16)
    baseline_params = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    baseline_latency, baseline_fps = measure_latency(
        baseline, input_tensor, device, iterations=args.iterations
    )
    baseline_vram = get_peak_vram_mb()
    results.append({
        "Model": "Baseline CNN",
        "Params": baseline_params,
        "VRAM (MB)": baseline_vram,
        "Latency (ms)": baseline_latency,
        "FPS": baseline_fps,
    })

    # 3. Print results
    print("Markdown Table for README:")
    print()
    print("| Model | Parameters | Peak VRAM (MB) | Latency (ms) | Throughput (FPS) |")
    print("|---|---|---|---|---|")
    for r in results:
        vram_str = f"{r['VRAM (MB)']:.1f}" if r['VRAM (MB)'] > 0 else "N/A (CPU)"
        print(f"| {r['Model']} | {r['Params']:,} | {vram_str} | "
              f"{r['Latency (ms)']:.2f} | {r['FPS']:.1f} |")
    print()

    # 4. Print summary statistics
    print("Summary:")
    print(f"- U-Net is {unet_params / baseline_params:.1f}x larger than the baseline.")
    if device.type == "cuda":
        print(f"- U-Net uses {unet_vram / baseline_vram:.1f}x more VRAM.")
    print(f"- U-Net latency is {unet_latency / baseline_latency:.1f}x slower.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
