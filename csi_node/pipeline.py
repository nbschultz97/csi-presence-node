"""Realtime CSI presence pipeline.

FeitCSI's command line tool streams CSI frames to a JSON lines log. This
module tails that log, performs windowed feature extraction and emits
standardised JSON results with presence, direction and pose estimates.
"""
import time
import json
import yaml
import sys
import os
from pathlib import Path
from collections import deque
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from . import utils
from .pose import PoseEstimator

CAPTURE_EXIT_CODE = 2
STALE_THRESHOLD = 5.0
_ERR_MSG = (
    "CSI capture not running or log idle. Start scripts/10_csi_capture.sh and retry."
)


def _capture_fail() -> None:
    print(_ERR_MSG, file=sys.stderr)
    sys.exit(CAPTURE_EXIT_CODE)


def _check_log_fresh(path: Path) -> None:
    if not path.exists() or time.time() - path.stat().st_mtime > STALE_THRESHOLD:
        _capture_fail()


class CSILogHandler(FileSystemEventHandler):
    def __init__(self, path: Path, buffer, process_cb):
        if not path.exists():
            raise FileNotFoundError(_ERR_MSG)
        self.path = str(path)
        self.buffer = buffer
        self.process_cb = process_cb
        self._fp = open(self.path, "r")
        # Start tailing from end of current file contents
        self._fp.seek(0, 2)
        stat = os.fstat(self._fp.fileno())
        self._inode = stat.st_ino
        self._size = stat.st_size

    def on_modified(self, event):
        if event.src_path != self.path:
            return

        try:
            stat = os.stat(self.path)
            inode, size = stat.st_ino, stat.st_size
        except FileNotFoundError:
            return

        if inode != self._inode or size < self._size:
            # Log rotated or truncated: reopen from start
            try:
                self._fp.close()
            except Exception:
                pass
            try:
                self._fp = open(self.path, "r")
                self._fp.seek(0)
                stat = os.fstat(self._fp.fileno())
                self._inode = stat.st_ino
                self._size = 0
            except FileNotFoundError:
                return
        else:
            self._size = size

        while True:
            line = self._fp.readline()
            if not line:
                break
            pkt = utils.parse_csi_line(line)
            if pkt:
                self.buffer.append(pkt)
                self.process_cb()
        self._size = self._fp.tell()


def compute_window(buffer, start_ts, end_ts, baseline, cfg, pose_estimator=None):
    window = [p for p in buffer if start_ts <= p["ts"] <= end_ts]
    if not window:
        return None

    dropped = 0
    valid = []
    expected_shape = baseline.shape if baseline is not None else None
    for pkt in window:
        csi = pkt.get("csi")
        if csi is None or csi.size == 0:
            dropped += 1
            continue
        if expected_shape is None:
            expected_shape = csi.shape
        if csi.shape != expected_shape:
            dropped += 1
            continue
        valid.append(pkt)

    if not valid:
        if dropped and cfg.get("log_dropped", False):
            print(
                f"Dropped {dropped} packets with empty or mismatched CSI",
                file=sys.stderr,
            )
        return None

    if dropped and cfg.get("log_dropped", False):
        print(
            f"Dropped {dropped} packets with empty or mismatched CSI",
            file=sys.stderr,
        )

    amps = np.stack([p["csi"] for p in valid], axis=0)
    if baseline is not None:
        if baseline.shape != amps.shape[1:]:
            msg = (
                f"Baseline shape {baseline.shape} does not match "
                f"amplitude shape {amps.shape[1:]}"
            )
            raise ValueError(msg)
        amps = amps - baseline
    amps = amps.reshape(amps.shape[0], -1)
    var = float(np.var(amps))
    pca1 = float(utils.compute_pca(amps)[0])
    pose, conf = "unknown", 0.0
    if pose_estimator is not None:
        pose, conf = pose_estimator.predict(amps)
    rssi0 = rssi1 = float("nan")
    direction = "C"
    rssis = [p.get("rssi") for p in valid if p.get("rssi")]
    if rssis and all(len(r) >= 2 for r in rssis):
        r0_vals = [r[0] for r in rssis]
        r1_vals = [r[1] for r in rssis]
        rssi0 = float(np.mean(r0_vals))
        rssi1 = float(np.mean(r1_vals))
        diff = rssi0 - rssi1
        delta = cfg["rssi_delta"]
        if diff > delta:
            direction = "L"
        elif diff < -delta:
            direction = "R"
    presence = int(var > cfg["variance_threshold"] or pca1 > cfg["pca_threshold"])
    return {
        "presence": presence,
        "direction": direction,
        "pose": pose,
        "confidence": conf,
        "var": var,
        "pca1": pca1,
        "rssi0": rssi0,
        "rssi1": rssi1,
        "ts": end_ts,
    }


def run(config_path: str = "csi_node/config.yaml") -> None:
    cfg = yaml.safe_load(open(config_path))
    buffer = deque()
    baseline = None
    if Path(cfg["baseline_file"]).exists():
        baseline = np.load(cfg["baseline_file"])["mean"]
    last_emit = 0.0
    pose_estimator = PoseEstimator()

    def process():
        nonlocal last_emit
        now = buffer[-1]["ts"]
        while buffer and now - buffer[0]["ts"] > cfg["window_size"]:
            buffer.popleft()
        if now - last_emit < cfg["window_hop"]:
            return
        last_emit = now
        start = now - cfg["window_size"]
        result = compute_window(buffer, start, now, baseline, cfg, pose_estimator)
        if result:
            iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(result["ts"]))
            row = [
                iso,
                int(result["ts"] * 1000),
                result["presence"],
                result["direction"],
                f"{result['var']:.3f}",
                f"{result['pca1']:.3f}",
                f"{result['rssi0']:.1f}",
                f"{result['rssi1']:.1f}",
                int(cfg["window_size"] * 1000),
            ]
            utils.rotate_file(cfg["output_file"], cfg["rotation_max_bytes"])
            utils.safe_csv_append(cfg["output_file"], row)

            # Emit standardised JSON for realtime consumers.
            out = {
                "timestamp": iso,
                "presence": bool(result["presence"]),
                "pose": result["pose"],
                "direction": result["direction"],
                "confidence": float(result["confidence"]),
            }
            print(json.dumps(out))
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"session_{time.strftime('%Y%m%d')}.json"
            with open(log_file, "a") as fp:
                fp.write(json.dumps(out) + "\n")

    observer = Observer()
    log_path = Path(cfg["log_file"])
    wait = cfg.get("log_wait", 5.0)
    if not log_path.exists() and not utils.wait_for_file(log_path, wait):
        _capture_fail()
    _check_log_fresh(log_path)
    handler = CSILogHandler(log_path, buffer, process)
    observer.schedule(handler, str(log_path.parent), recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def run_offline(log_path: str, cfg: dict):
    buffer = deque()
    baseline = None
    if Path(cfg["baseline_file"]).exists():
        baseline = np.load(cfg["baseline_file"])["mean"]
    rows = []
    pose_estimator = PoseEstimator()
    log_path = Path(log_path)
    wait = cfg.get("log_wait", 5.0)
    if not log_path.exists() and not utils.wait_for_file(log_path, wait):
        print(_ERR_MSG, file=sys.stderr)
        raise FileNotFoundError(_ERR_MSG)
    with open(log_path, "r") as f:
        for line in f:
            pkt = utils.parse_csi_line(line)
            if not pkt:
                continue
            buffer.append(pkt)
            now = pkt["ts"]
            while buffer and now - buffer[0]["ts"] > cfg["window_size"]:
                buffer.popleft()
            if len(buffer) < 1:
                continue
            start = now - cfg["window_size"]
            result = compute_window(buffer, start, now, baseline, cfg, pose_estimator)
            if result:
                rows.append(result)
    import pandas as pd
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    run()
