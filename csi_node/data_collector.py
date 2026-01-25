"""Data collection tool for pose classifier training.

Collects labeled CSI data for training the pose classifier.
Run this tool to create training datasets for different poses.

Usage:
    # Interactive collection mode
    python -m csi_node.data_collector --output data/training.npz

    # Collect specific pose
    python -m csi_node.data_collector --pose standing --samples 100 --output data/standing.npz

    # Merge multiple datasets
    python -m csi_node.data_collector --merge data/standing.npz data/crouching.npz --output data/combined.npz
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from collections import deque
from typing import List, Dict, Any, Optional
import numpy as np

from . import utils
from .pose_classifier import extract_features, LABELS, LABEL_MAP


def collect_window(
    log_path: Path,
    window_size: float = 3.0,
    timeout: float = 30.0,
) -> Optional[np.ndarray]:
    """Collect a single window of CSI data.

    Args:
        log_path: Path to CSI log file
        window_size: Window duration in seconds
        timeout: Maximum wait time

    Returns:
        CSI window array or None if timeout
    """
    if not log_path.exists():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        return None

    buffer: deque = deque()
    start_time = time.time()

    with open(log_path, "r") as f:
        # Seek to end to get new data
        f.seek(0, 2)

        while time.time() - start_time < timeout:
            line = f.readline()
            if not line:
                time.sleep(0.05)
                continue

            pkt = utils.parse_csi_line(line)
            if pkt is None:
                continue

            buffer.append(pkt)

            # Keep only recent packets
            now = pkt["ts"]
            while buffer and now - buffer[0]["ts"] > window_size:
                buffer.popleft()

            # Check if we have enough data
            if len(buffer) >= 10 and buffer[-1]["ts"] - buffer[0]["ts"] >= window_size * 0.8:
                # Stack CSI data
                csi_list = [p["csi"] for p in buffer if p.get("csi") is not None]
                if csi_list:
                    return np.stack(csi_list, axis=0)

    return None


def collect_sample(
    log_path: Path,
    label: int,
    window_size: float = 3.0,
) -> Optional[Dict[str, Any]]:
    """Collect a single labeled sample.

    Args:
        log_path: Path to CSI log file
        label: Pose label index
        window_size: Window duration

    Returns:
        Dictionary with features and label, or None on failure
    """
    csi_window = collect_window(log_path, window_size)
    if csi_window is None:
        return None

    # Extract features
    features = extract_features(csi_window)

    return {
        "features": features,
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_packets": len(csi_window),
    }


def interactive_collect(
    log_path: Path,
    output_path: Path,
    samples_per_pose: int = 50,
    window_size: float = 3.0,
) -> None:
    """Interactive data collection with prompts.

    Args:
        log_path: Path to CSI log file
        output_path: Output .npz file path
        samples_per_pose: Number of samples per pose class
        window_size: Window duration
    """
    print("=== CSI Pose Data Collector ===")
    print(f"Log file: {log_path}")
    print(f"Output: {output_path}")
    print(f"Samples per pose: {samples_per_pose}")
    print(f"Window size: {window_size}s")
    print("")

    all_features: List[np.ndarray] = []
    all_labels: List[int] = []

    for label_idx, label_name in enumerate(LABELS[:3]):  # Standing, Crouching, Prone
        print(f"\n--- Collecting {label_name} samples ---")
        print(f"Please assume the {label_name} position.")
        input("Press Enter when ready...")

        collected = 0
        while collected < samples_per_pose:
            print(f"  Collecting sample {collected + 1}/{samples_per_pose}...", end=" ", flush=True)

            sample = collect_sample(log_path, label_idx, window_size)
            if sample is None:
                print("TIMEOUT - retrying")
                continue

            all_features.append(sample["features"])
            all_labels.append(sample["label"])
            collected += 1
            print("OK")

            # Brief pause between samples
            time.sleep(0.5)

        print(f"  Completed {label_name}: {collected} samples")

    # Save dataset
    X = np.array(all_features)
    y = np.array(all_labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, X=X, y=y, labels=LABELS[:3])

    print(f"\nDataset saved to {output_path}")
    print(f"  Total samples: {len(X)}")
    print(f"  Feature shape: {X.shape}")
    print(f"  Labels: {np.bincount(y)}")


def collect_single_pose(
    log_path: Path,
    pose: str,
    samples: int,
    output_path: Path,
    window_size: float = 3.0,
) -> None:
    """Collect samples for a single pose.

    Args:
        log_path: Path to CSI log file
        pose: Pose name (standing/crouching/prone)
        samples: Number of samples to collect
        output_path: Output .npz file path
        window_size: Window duration
    """
    pose_upper = pose.upper()
    if pose_upper not in LABEL_MAP:
        print(f"Unknown pose: {pose}. Valid: {list(LABEL_MAP.keys())}", file=sys.stderr)
        sys.exit(1)

    label_idx = LABEL_MAP[pose_upper]

    print(f"Collecting {samples} samples for {pose_upper}...")
    print(f"Please assume the {pose_upper} position.")
    input("Press Enter when ready...")

    all_features: List[np.ndarray] = []
    collected = 0

    while collected < samples:
        print(f"  Sample {collected + 1}/{samples}...", end=" ", flush=True)

        sample = collect_sample(log_path, label_idx, window_size)
        if sample is None:
            print("TIMEOUT - retrying")
            continue

        all_features.append(sample["features"])
        collected += 1
        print("OK")
        time.sleep(0.3)

    X = np.array(all_features)
    y = np.full(len(X), label_idx)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, X=X, y=y, pose=pose_upper)

    print(f"\nSaved {len(X)} samples to {output_path}")


def merge_datasets(
    input_paths: List[Path],
    output_path: Path,
) -> None:
    """Merge multiple .npz datasets into one.

    Args:
        input_paths: List of input .npz files
        output_path: Output .npz file path
    """
    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for path in input_paths:
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            continue

        data = np.load(path)
        X = data["X"]
        y = data["y"]

        all_X.append(X)
        all_y.append(y)
        print(f"  Loaded {len(X)} samples from {path}")

    if not all_X:
        print("No data to merge!", file=sys.stderr)
        sys.exit(1)

    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, X=X_combined, y=y_combined, labels=LABELS)

    print(f"\nMerged dataset saved to {output_path}")
    print(f"  Total samples: {len(X_combined)}")
    print(f"  Labels distribution: {np.bincount(y_combined.astype(int))}")


def main() -> None:
    """CLI entry point for data collector."""
    parser = argparse.ArgumentParser(
        description="CSI Pose Data Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--log",
        type=str,
        default="./data/csi_raw.log",
        help="CSI log file path",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/training.npz",
        help="Output dataset path",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=50,
        help="Samples per pose (or total for single pose)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=3.0,
        help="Window size in seconds",
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--pose",
        type=str,
        default=None,
        choices=["standing", "crouching", "prone", "sitting", "walking"],
        help="Collect single pose type",
    )
    mode_group.add_argument(
        "--merge",
        nargs="+",
        type=str,
        default=None,
        help="Merge multiple .npz files",
    )

    args = parser.parse_args()

    if args.merge:
        merge_datasets(
            [Path(p) for p in args.merge],
            Path(args.output),
        )
    elif args.pose:
        collect_single_pose(
            Path(args.log),
            args.pose,
            args.samples,
            Path(args.output),
            args.window,
        )
    else:
        interactive_collect(
            Path(args.log),
            Path(args.output),
            args.samples,
            args.window,
        )


if __name__ == "__main__":
    main()
