"""Utility functions for CSI processing."""
from __future__ import annotations
import json
import os
import sys
import tempfile

# Cross-platform file locking
if sys.platform == 'win32':
    import msvcrt
    def _lock_file(f):
        """Lock file on Windows."""
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
    def _unlock_file(f):
        """Unlock file on Windows."""
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass  # May fail if not locked
else:
    import fcntl
    def _lock_file(f):
        """Lock file on Unix."""
        fcntl.flock(f, fcntl.LOCK_EX)
    def _unlock_file(f):
        """Unlock file on Unix."""
        fcntl.flock(f, fcntl.LOCK_UN)
import time
import logging
import traceback
from pathlib import Path
import numpy as np
from typing import List, Optional, Dict, Any


class RunLogManager:
    """Manage per-run logging with crash-safe metadata."""

    def __init__(
        self,
        out_path: str | Path,
        rotation_bytes: int = 1_048_576,
        rotation_seconds: float | None = None,
        retention: int = 5,
        run_prefix: str = "run",
    ) -> None:
        self.rotation_bytes = max(int(rotation_bytes), 0)
        self.rotation_seconds = float(rotation_seconds or 0.0)
        self.retention = max(int(retention), 0)
        target = Path(out_path).expanduser()
        if not target.is_absolute():
            target = (Path.cwd() / target).resolve()
        base_dir = target.parent
        base_dir.mkdir(parents=True, exist_ok=True)
        runs_root = base_dir / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        candidate = runs_root / f"{run_prefix}-{stamp}"
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = runs_root / f"{run_prefix}-{stamp}-{suffix:02d}"
        candidate.mkdir(parents=True, exist_ok=False)
        self.run_dir = candidate
        self.base_name = target.name
        self.log_path = self.run_dir / self.base_name
        self.symlink_path = target
        self.fatal_path = self.run_dir / "fatal.json"
        self._mirror_path: Path | None = None
        self._rotated: List[Path] = []
        self.last_entry: Optional[Dict[str, Any]] = None
        self.last_frame: Optional[Dict[str, Any]] = None
        self.last_packet: Optional[Dict[str, Any]] = None
        self._fatal_written = False
        self._file_started_at = time.time()
        self._install_symlink()

    def _install_symlink(self) -> None:
        """Point the legacy output path at the run-specific log."""

        self._mirror_path = None
        try:
            if self.symlink_path.exists() or self.symlink_path.is_symlink():
                if self.symlink_path.is_dir():
                    # Do not replace directories with symlinks
                    return
                self.symlink_path.unlink()
        except OSError:
            return
        try:
            self.symlink_path.parent.mkdir(parents=True, exist_ok=True)
            self.symlink_path.symlink_to(self.log_path)
        except OSError:
            # Fall back to creating an empty file if symlinks are unavailable
            try:
                with open(self.symlink_path, "w"):
                    pass
            except OSError:
                pass
            else:
                self._mirror_path = self.symlink_path

    def append(
        self,
        entry: Dict[str, Any],
        frame: Optional[Dict[str, Any]] = None,
        latest_packet: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a JSONL entry and track crash metadata."""

        self._rotate_for_time()
        created = not self.log_path.exists()
        payload = self._sanitize(entry)
        safe_json_append(str(self.log_path), payload)
        if created:
            self._file_started_at = time.time()
        self.last_entry = payload
        if frame is not None:
            self.last_frame = self._sanitize(frame)
        if latest_packet is not None:
            self.last_packet = self._sanitize(latest_packet)
        if self._mirror_path and self._mirror_path != self.log_path:
            try:
                if self.rotation_bytes > 0:
                    rotate_file(str(self._mirror_path), self.rotation_bytes)
                safe_json_append(str(self._mirror_path), payload)
            except Exception:
                pass
        self._rotate_for_size()

    def write_fatal(self, exc: BaseException) -> Path:
        """Persist fatal crash metadata exactly once."""

        if self._fatal_written:
            return self.fatal_path
        stack = traceback.format_exception(exc.__class__, exc, exc.__traceback__)
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "exception": {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "stack": stack,
            },
            "last_entry": self.last_entry,
            "last_frame": self.last_frame,
            "last_packet": self.last_packet,
            "log_path": str(self.log_path),
            "run_dir": str(self.run_dir),
        }
        self._atomic_write_json(self.fatal_path, payload)
        self._fatal_written = True
        return self.fatal_path

    # Internal helpers -------------------------------------------------

    def _rotate_for_time(self) -> None:
        if self.rotation_seconds <= 0:
            return
        if not self.log_path.exists():
            return
        age = time.time() - self._file_started_at
        if age >= self.rotation_seconds and self.log_path.stat().st_size > 0:
            self._rotate("time")

    def _rotate_for_size(self) -> None:
        if self.rotation_bytes <= 0:
            return
        if not self.log_path.exists():
            return
        if self.log_path.stat().st_size >= self.rotation_bytes:
            self._rotate("size")

    def _rotate(self, reason: str) -> None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        rotated = self.run_dir / f"{self.base_name}.{reason}-{timestamp}"
        counter = 1
        while rotated.exists():
            counter += 1
            rotated = self.run_dir / f"{self.base_name}.{reason}-{timestamp}-{counter}"
        try:
            self.log_path.replace(rotated)
        except FileNotFoundError:
            return
        self._rotated.append(rotated)
        self._file_started_at = time.time()
        self._prune_rotated()

    def _prune_rotated(self) -> None:
        if self.retention <= 0:
            return
        excess = len(self._rotated) - self.retention
        while excess > 0 and self._rotated:
            oldest = self._rotated.pop(0)
            try:
                oldest.unlink()
            except OSError:
                pass
            excess -= 1

    def _sanitize(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
        if isinstance(value, dict):
            return {k: self._sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize(v) for v in value]
        return value

    def _atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent))
        try:
            with os.fdopen(fd, "w") as tmp:
                json.dump(payload, tmp, indent=2, sort_keys=True)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass



def moving_median(data: np.ndarray, window: int) -> np.ndarray:
    """Compute moving median along the first axis."""
    if window <= 1:
        return data
    padded = np.pad(data, ((window - 1, 0), (0, 0)), mode="edge")
    return np.array([
        np.median(padded[i:i + window], axis=0) for i in range(data.shape[0])
    ])


def savgol(data: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Apply Savitzky–Golay filter if SciPy is available, else return input.

    SciPy is an optional dependency. To keep import-time light for offline
    tests, we import on demand and gracefully fall back when unavailable.
    """
    try:
        from scipy.signal import savgol_filter as _savgol_filter  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return data
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
            _lock_file(f)
            with open(tmp_path) as tmp:
                f.write(tmp.read())
            f.flush()
            os.fsync(f.fileno())
            _unlock_file(f)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            logging.warning("Failed to remove temporary file %s", tmp_path)


def safe_json_append(path: str, obj: Dict) -> None:
    """Atomically append JSON object ``obj`` as one line to ``path``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(obj) + "\n"
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        with os.fdopen(tmp_fd, "w") as tmp:
            tmp.write(line)
            tmp.flush()
            os.fsync(tmp.fileno())
        with open(path, "a") as f:
            _lock_file(f)
            with open(tmp_path) as tmp:
                f.write(tmp.read())
            f.flush()
            os.fsync(f.fileno())
            _unlock_file(f)
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


def rssi_to_distance(rssi: float, tx_power: float = -40.0, n: float = 2.0) -> float:
    """Estimate distance in meters from RSSI using a log-distance path loss model.

    ``tx_power`` is the expected RSSI (dBm) at 1 m and ``n`` the path loss
    exponent. This is a crude estimate suitable only for coarse demo output.
    """
    try:
        return float(10 ** ((tx_power - rssi) / (10 * n)))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return float("nan")
