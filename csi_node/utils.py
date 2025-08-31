"""Utility functions for CSI processing."""
from __future__ import annotations
import json
import os
import fcntl
import tempfile
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
    os.remove(tmp_path)


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


def parse_csi_line(line: str) -> Optional[dict]:
    """Parse a single FeitCSI JSON line.

    Expected format:
    {"ts": 0.0, "rssi": [-40, -42], "csi": [[...],[...]]}
    """
    line = line.strip()
    if not line:
        return None
    try:
        pkt = json.loads(line)
        pkt["csi"] = np.array(pkt.get("csi", []), dtype=float)
        pkt["rssi"] = pkt.get("rssi")
        pkt["ts"] = float(pkt.get("ts", 0.0))
        return pkt
    except json.JSONDecodeError:
        return None
