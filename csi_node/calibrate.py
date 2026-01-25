"""Calibrate RSSI→distance parameters for the log-distance model.

Usage examples:

1) From two logs captured at known distances (recommended):

   python -m csi_node.calibrate \
       --log1 data/csi_raw_d1.log --d1 1.0 \
       --log2 data/csi_raw_d2.log --d2 3.0 \
       --config csi_node/config.yaml

2) From direct RSSI values (median of logs, etc.):

   python -m csi_node.calibrate --rssi1 -48 --d1 1.0 --rssi2 -60 --d2 5.0

This estimates:
   n   = (rssi1 - rssi2) / (10 * log10(d2/d1))
   txp = rssi1 + 10 * n * log10(d1)

and optionally writes them to a YAML config file.
"""

from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from datetime import datetime
from statistics import median
from typing import Tuple, Optional

import yaml  # type: ignore


def estimate_from_pairs(rssi1: float, d1: float, rssi2: float, d2: float) -> Tuple[float, float]:
    """Estimate path loss parameters from two RSSI measurements.

    The log-distance path loss model is:
        RSSI = tx_power - 10 * n * log10(d)

    Where:
        tx_power = RSSI at 1 meter (reference)
        n = path loss exponent (typically 2.0-4.0 for indoor)
        d = distance in meters

    Args:
        rssi1: RSSI at distance d1 (dB)
        d1: First distance in meters
        rssi2: RSSI at distance d2 (dB)
        d2: Second distance in meters

    Returns:
        Tuple of (tx_power_dbm, path_loss_exponent)

    Raises:
        ValueError: If distances are invalid or RSSI relationship is inverted
    """
    if d1 <= 0 or d2 <= 0 or d1 == d2:
        raise ValueError("Distances must be positive and distinct")

    # Ensure d1 < d2 for consistent calculation
    if d1 > d2:
        d1, d2 = d2, d1
        rssi1, rssi2 = rssi2, rssi1

    denom = 10.0 * math.log10(d2 / d1)
    if abs(denom) < 1e-9:
        raise ValueError("Distances too close for stable estimation")

    n = (rssi1 - rssi2) / denom

    # Validate path loss exponent
    if n < 0:
        # This indicates RSSI increased with distance, which is physically impossible
        # in normal conditions. Common causes:
        # 1. RSSI values are inverted (positive instead of negative)
        # 2. Measurement error
        # 3. Multipath constructive interference at far distance
        print(
            f"WARNING: Calculated path_loss_exponent is negative ({n:.3f}).\n"
            f"  This indicates RSSI increased with distance, which is unusual.\n"
            f"  Possible causes:\n"
            f"    - RSSI scale is inverted (check dat_rssi_offset)\n"
            f"    - Measurements taken in different conditions\n"
            f"    - Severe multipath interference\n"
            f"  Using absolute value: {abs(n):.3f}",
            file=sys.stderr,
        )
        n = abs(n)

    if n < 1.0:
        print(
            f"WARNING: path_loss_exponent {n:.3f} is below free-space minimum (2.0).\n"
            f"  This is unusual. Consider recalibrating.",
            file=sys.stderr,
        )
    elif n > 5.0:
        print(
            f"WARNING: path_loss_exponent {n:.3f} is very high (typical: 2.0-4.0).\n"
            f"  This indicates heavy obstruction. Verify calibration conditions.",
            file=sys.stderr,
        )

    txp = rssi1 + 10.0 * n * math.log10(d1)

    # Validate tx_power
    if txp > 0:
        print(
            f"WARNING: tx_power_dbm {txp:.1f} is positive, which is unusual.\n"
            f"  Typical RSSI at 1m is -20 to -50 dBm.",
            file=sys.stderr,
        )
    elif txp < -80:
        print(
            f"WARNING: tx_power_dbm {txp:.1f} is very low.\n"
            f"  This may indicate excessive range during calibration.",
            file=sys.stderr,
        )

    return txp, n


def _avg_rssi_from_log(path: Path) -> Optional[float]:
    vals = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rssi = obj.get("rssi")
            if not isinstance(rssi, list) or len(rssi) < 2:
                continue
            try:
                avg = (float(rssi[0]) + float(rssi[1])) / 2.0
            except (TypeError, ValueError):
                continue
            vals.append(avg)
    if not vals:
        return None
    return median(vals)


def write_config(cfg_path: Path, txp: float, n: float) -> None:
    data = yaml.safe_load(open(cfg_path))
    data["tx_power_dbm"] = float(txp)
    data["path_loss_exponent"] = float(n)
    # Mark calibration metadata for UI
    data["calibrated"] = True
    data["calibrated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate RSSI→distance parameters")
    ap.add_argument("--log1", type=str, default=None, help="JSONL with per-packet rssi (data/csi_raw.log)")
    ap.add_argument("--d1", type=float, default=None, help="distance for log1 (m)")
    ap.add_argument("--log2", type=str, default=None, help="second JSONL log path")
    ap.add_argument("--d2", type=float, default=None, help="distance for log2 (m)")
    ap.add_argument("--rssi1", type=float, default=None, help="RSSI at d1 (dB)")
    ap.add_argument("--rssi2", type=float, default=None, help="RSSI at d2 (dB)")
    ap.add_argument("--config", type=str, default=None, help="write results into YAML config (e.g., csi_node/config.yaml)")
    args = ap.parse_args()

    # Prefer logs if provided; else fall back to explicit RSSI values
    r1 = args.rssi1
    r2 = args.rssi2

    if args.log1:
        val = _avg_rssi_from_log(Path(args.log1))
        if val is None:
            raise SystemExit(f"No usable rssi values found in {args.log1}")
        r1 = val
    if args.log2:
        val = _avg_rssi_from_log(Path(args.log2))
        if val is None:
            raise SystemExit(f"No usable rssi values found in {args.log2}")
        r2 = val

    if r1 is None or r2 is None or args.d1 is None or args.d2 is None:
        raise SystemExit("Provide either --log1/--d1 and --log2/--d2, or --rssi1/--d1 and --rssi2/--d2")

    txp, n = estimate_from_pairs(float(r1), float(args.d1), float(r2), float(args.d2))
    print(f"Estimated tx_power_dbm: {txp:.2f} dBm at 1 m")
    print(f"Estimated path_loss_exponent: {n:.3f}")

    if args.config:
        cfgp = Path(args.config)
        write_config(cfgp, txp, n)
        print(f"Wrote calibration to {cfgp}")


if __name__ == "__main__":
    main()
