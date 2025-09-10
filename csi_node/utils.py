"""Utility functions for CSI processing."""
from __future__ import annotations
import json
import os
import fcntl
import tempfile
import time
import logging
from pathlib import Path
import numpy as np
from typing import List, Optional
from scipy.signal import savgol_filter as _savgol_filter


def moving_median(data: np.ndarray, window: int) -> np.ndarray:
    """Compute moving median along the first axis."""
    if window <= 1:
        return data
    padded = np.pad(data, ((window - 1, 0), (0, 0)), mode="edge")
    return np.array([
        np.median(padded[i:i + window], axis=0) for i in range(data.shape[0])
    ])


def savgol(data: np.ndarray, window: int, poly: int) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    return _savgol_filter(data, window, poly, axis=0)


def compute_pca(matrix: np.ndarray) -> np.ndarray:
    """Return eigenvalues sorted descending using SVD."""
    if matrix.ndim != 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    vals = (s ** 2) / max(matrix.shape[0] - 1, 1)
    return vals

def safe_csv_append(path: str, row: List) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = ",".join(map(str, row)) + "\n"
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        with os.fdopen(tmp_fd, "w") as tmp:
            tmp.write(line)
            tmp.flush()
            os.fsync(tmp.fileno())
        with open(path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            with open(tmp_path) as tmp:
                f.write(tmp.read())
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            logging.warning("Failed to remove temporary file %s", tmp_path)


def rotate_file(path: str, max_bytes: int) -> None:
    if not os.path.exists(path):
        return
    if os.path.getsize(path) < max_bytes:
        return
    for i in range(5, 0, -1):
        src = f"{path}.{i}"
        dst = f"{path}.{i+1}"
        if os.path.exists(src):
            os.rename(src, dst)
    os.rename(path, f"{path}.1")


def wait_for_file(path: Path | str, timeout: float = 5.0, interval: float = 0.5) -> bool:
    """Poll for a file to appear up to ``timeout`` seconds."""
    p = Path(path)
    end = time.time() + timeout
    while time.time() < end:
        if p.exists():
            return True
        time.sleep(interval)
    return p.exists()


def parse_csi_line(line: str) -> Optional[dict]:
    """Parse a single FeitCSI JSON line.

    Expected format::

        {"ts": 0.0, "rssi": [-40, -42], "csi": [[...], [...]]}

    ``rssi`` is expected to contain two values (one per chain) and will be
    returned as ``List[float]``. ``None`` is returned if parsing fails.
    """
    line = line.strip()
    if not line:
        return None
    try:
        pkt = json.loads(line)
    except json.JSONDecodeError:
        return None

    try:
        pkt["csi"] = np.array(pkt.get("csi", []), dtype=float)

        raw_rssi = pkt.get("rssi")
        if not isinstance(raw_rssi, (list, tuple)) or len(raw_rssi) != 2:
            return None
        pkt["rssi"] = [float(v) for v in raw_rssi]

        pkt["ts"] = float(pkt.get("ts", 0.0))
    except (TypeError, ValueError):
        return None

    return pkt
