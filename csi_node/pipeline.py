"""Realtime CSI presence pipeline with optional pose classifier and curses TUI."""
import argparse
import time
import yaml
import sys
import os
import json
from pathlib import Path
from collections import deque
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Event, Thread

from . import utils
from .pose_classifier import PoseClassifier  # pose classifier for skeletal state
from . import tui as tui_mod  # curses dashboard hooks
from . import replay as replay_mod

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


def compute_window(buffer, start_ts, end_ts, baseline, cfg):
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
    abs_amps = np.abs(amps)
    mean_mag = float(np.mean(abs_amps))
    std_mag = float(np.std(abs_amps))
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
    avg_rssi = (rssi0 + rssi1) / 2.0
    distance = utils.rssi_to_distance(avg_rssi)
    presence = int(var > cfg["variance_threshold"] or pca1 > cfg["pca_threshold"])
    return {
        "presence": presence,
        "direction": direction,
        "var": var,
        "pca1": pca1,
        "rssi0": rssi0,
        "rssi1": rssi1,
        "distance": distance,
        "pose_feat": np.array([mean_mag, std_mag]),
        "ts": end_ts,
    }


def run_demo(
    pose: bool = False,
    tui: bool = False,
    replay_path: str | None = None,
    source=None,
    window: float = 3.0,
    out: str = "data/presence_log.jsonl",
    speed: float = 1.0,
) -> None:
    """Run realtime or replay pipeline with optional pose and TUI."""
    cfg = yaml.safe_load(open("csi_node/config.yaml"))
    cfg["window_size"] = window
    cfg["output_file"] = out
    buffer = deque()
    baseline = None
    if Path(cfg["baseline_file"]).exists():
        baseline = np.load(cfg["baseline_file"])["mean"]

    pose_clf: PoseClassifier | None = None
    if pose:
        try:
            pose_clf = PoseClassifier("models/wipose.joblib")
        except Exception as exc:  # pragma: no cover - classifier optional
            print(f"Pose classifier init failed: {exc}", file=sys.stderr)

    alpha = 0.2
    presence_ema = 0.0
    pose_ema = 0.0
    last_dir = "C"
    l_cnt = r_cnt = 0

    state = {
        "presence": "NO",
        "presence_conf": 0.0,
        "direction": "C",
        "rssi_delta": 0.0,
        "pose": "N/A",
        "pose_conf": 0.0,
    }
    stop = Event()
    tui_thread = None
    if tui:
        mode = "LIVE (FeitCSI)" if replay_path is None else "REPLAY"
        tui_thread = Thread(
            target=tui_mod.run,
            args=(state, stop, out, mode, replay_path),
            daemon=True,
        )
        tui_thread.start()

    def handle(result: dict) -> None:
        nonlocal presence_ema, pose_ema, last_dir, l_cnt, r_cnt
        raw = 1.0 if result["presence"] else 0.0
        presence_ema = alpha * raw + (1 - alpha) * presence_ema
        diff = result["rssi0"] - result["rssi1"]
        delta = cfg["rssi_delta"]
        if diff > delta:
            l_cnt += 1
            r_cnt = 0
        elif diff < -delta:
            r_cnt += 1
            l_cnt = 0
        else:
            l_cnt = r_cnt = 0
        if l_cnt >= 3:
            last_dir = "L"
        elif r_cnt >= 3:
            last_dir = "R"
        pose_label = "N/A"
        pose_conf = 0.0
        if pose_clf is not None:
            pose_label, conf = pose_clf.predict(result["pose_feat"])
            pose_ema = alpha * conf + (1 - alpha) * pose_ema
            pose_conf = pose_ema
        state.update(
            presence="YES" if result["presence"] else "NO",
            presence_conf=presence_ema,
            direction=last_dir,
            rssi_delta=diff,
            pose=pose_label,
            pose_conf=pose_conf,
        )
        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(result["ts"]))
        entry = {
            "timestamp": iso,
            "presence": bool(result["presence"]),
            "pose": pose_label.lower(),
            "direction": {"L": "left", "R": "right", "C": "center"}[last_dir],
            "distance_m": float(result["distance"]),
            "confidence": presence_ema,
        }
        print(json.dumps(entry))
        utils.rotate_file(cfg["output_file"], cfg["rotation_max_bytes"])
        utils.safe_json_append(cfg["output_file"], entry)

    def process() -> None:
        now = buffer[-1]["ts"]
        while buffer and now - buffer[0]["ts"] > cfg["window_size"]:
            buffer.popleft()
        start = now - cfg["window_size"]
        result = compute_window(buffer, start, now, baseline, cfg)
        if result:
            handle(result)

    if source is not None:
        for pkt in source:
            if stop.is_set():
                break
            buffer.append(pkt)
            process()
    elif replay_path:
        for pkt in replay_mod.replay(replay_path, speed):
            if stop.is_set():
                break
            buffer.append(pkt)
            process()
    else:
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
            while not stop.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        observer.stop()
        observer.join()

    stop.set()
    if tui_thread:
        tui_thread.join()


def run(config_path: str = "csi_node/config.yaml") -> None:
    cfg = yaml.safe_load(open(config_path))
    buffer = deque()
    baseline = None
    if Path(cfg["baseline_file"]).exists():
        baseline = np.load(cfg["baseline_file"])["mean"]
    last_emit = 0.0

    def process():
        nonlocal last_emit
        now = buffer[-1]["ts"]
        while buffer and now - buffer[0]["ts"] > cfg["window_size"]:
            buffer.popleft()
        if now - last_emit < cfg["window_hop"]:
            return
        last_emit = now
        start = now - cfg["window_size"]
        result = compute_window(buffer, start, now, baseline, cfg)
        if result:
            iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(result["ts"]))
            entry = {
                "timestamp": iso,
                "presence": bool(result["presence"]),
                "pose": "n/a",
                "direction": {"L": "left", "R": "right", "C": "center"}[result["direction"]],
                "distance_m": float(result["distance"]),
                "confidence": float(result["presence"]),
            }
            utils.rotate_file(cfg["output_file"], cfg["rotation_max_bytes"])
            utils.safe_json_append(cfg["output_file"], entry)

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
            result = compute_window(buffer, start, now, baseline, cfg)
            if result:
                rows.append(result)
    import pandas as pd
    df = pd.DataFrame(rows)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="CSI presence pipeline")
    parser.add_argument("--pose", action="store_true", help="enable pose classifier")
    parser.add_argument("--tui", action="store_true", help="launch curses UI")
    parser.add_argument("--replay", type=str, default=None, help="replay log file")
    parser.add_argument("--iface", type=str, default=None, help="live capture interface")
    parser.add_argument("--window", type=float, default=3.0, help="window size (s)")
    parser.add_argument("--out", type=str, default="data/presence_log.jsonl", help="output JSONL")
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed factor")
    args = parser.parse_args()

    src = None
    if args.iface:
        try:
            from . import feitcsi
            src = feitcsi.live_stream(args.iface)
        except Exception as exc:  # pragma: no cover - hardware optional
            print(f"FeitCSI live stream unavailable: {exc}", file=sys.stderr)

    run_demo(
        pose=args.pose,
        tui=args.tui,
        replay_path=args.replay,
        source=src,
        window=args.window,
        out=args.out,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
