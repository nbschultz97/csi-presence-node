"""Realtime CSI presence pipeline with optional pose classifier and curses TUI."""
import argparse
import time
import yaml
import sys
import os
import json
import math
from pathlib import Path
from collections import deque
from typing import Optional, Tuple
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Event, Thread

from node.aoa import aoa_deg
from node.range import AlphaBetaFilter, distance_from_rss

from . import utils
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


def _channel_to_freq_mhz(ch: int) -> float:
    if 1 <= ch <= 13:
        return 2407 + ch * 5
    if ch == 14:
        return 2484
    return 5000 + ch * 5


def _center_frequency_hz(cfg: dict) -> float:
    try:
        ch = int(cfg.get("channel", 36))
    except Exception:
        ch = 36
    return _channel_to_freq_mhz(ch) * 1e6


def _split_rx_chains(csi) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    arr = np.asarray(csi)
    if arr.size == 0:
        return None, None
    if arr.ndim >= 1 and arr.shape[-1] == 2 and not np.iscomplexobj(arr):
        arr = arr[..., 0] + 1j * arr[..., 1]
    arr = np.asarray(arr, dtype=np.complex128)
    if arr.ndim <= 1:
        return None, None
    if arr.shape[0] >= 2:
        return arr[0].reshape(-1), arr[1].reshape(-1)
    if arr.shape[-1] >= 2:
        return arr[..., 0].reshape(-1), arr[..., 1].reshape(-1)
    return None, None


class CSILogHandler(FileSystemEventHandler):
    def __init__(self, path: Path, buffer, process_cb, pkt_cb=None):
        if not path.exists():
            raise FileNotFoundError(_ERR_MSG)
        self.path = str(path)
        self.buffer = buffer
        self.process_cb = process_cb
        self.pkt_cb = pkt_cb
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
                if self.pkt_cb:
                    try:
                        self.pkt_cb()
                    except Exception:
                        pass
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
    diff = float("nan")
    rssis = [p.get("rssi") for p in valid if p.get("rssi")]
    if rssis and all(len(r) >= 2 for r in rssis):
        r0_vals = [r[0] for r in rssis]
        r1_vals = [r[1] for r in rssis]
        rssi0 = float(np.mean(r0_vals))
        rssi1 = float(np.mean(r1_vals))
        diff = rssi0 - rssi1
    avg_vals = [v for v in (rssi0, rssi1) if math.isfinite(v)]
    avg_rssi = float(np.mean(avg_vals)) if avg_vals else float("nan")
    txp = float(cfg.get("tx_power_dbm", -40.0))
    ple = float(cfg.get("path_loss_exponent", 2.0))
    intercept = float(cfg.get("path_loss_c", txp))
    distance = distance_from_rss(avg_rssi, ple, intercept, txp)
    spacing_cm = float(cfg.get("antenna_spacing_cm", 6.25))
    spacing_m = spacing_cm / 100.0 if spacing_cm > 0 else 0.0
    theta_offset = float(cfg.get("theta_offset_deg", 0.0))
    freq_hz = _center_frequency_hz(cfg)
    aoa_samples: list[float] = []
    if spacing_m > 0 and freq_hz > 0:
        for pkt in valid:
            c1, c2 = _split_rx_chains(pkt.get("csi"))
            if c1 is None or c2 is None:
                continue
            ang = aoa_deg(c1, c2, freq_hz, spacing_m, theta_offset=theta_offset)
            if math.isfinite(ang):
                aoa_samples.append(float(ang))
    bearing_deg = float(np.median(aoa_samples)) if aoa_samples else float("nan")
    aoa_deadzone = float(cfg.get("aoa_deadzone_deg", 7.5))
    direction = "C"
    if math.isfinite(bearing_deg):
        if bearing_deg < -aoa_deadzone:
            direction = "L"
        elif bearing_deg > aoa_deadzone:
            direction = "R"
    elif math.isfinite(diff):
        delta = float(cfg.get("rssi_delta", 3.5))
        if diff > delta:
            direction = "L"
        elif diff < -delta:
            direction = "R"
    presence = int(var > cfg["variance_threshold"] or pca1 > cfg["pca_threshold"])
    return {
        "presence": presence,
        "direction": direction,
        "rssi_delta": diff,
        "var": var,
        "pca1": pca1,
        "rssi0": rssi0,
        "rssi1": rssi1,
        "distance": distance,
        "distance_raw": distance,
        "bearing_deg": bearing_deg,
        "bearing_label": {"L": "left", "R": "right", "C": "center"}[direction],
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
    log_override: str | None = None,
) -> None:
    """Run realtime or replay pipeline with optional pose and TUI."""
    # Load config relative to this file to avoid CWD issues
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    cfg["window_size"] = window
    cfg["output_file"] = out
    if log_override:
        cfg["log_file"] = log_override
    buffer = deque()
    baseline = None
    if Path(cfg["baseline_file"]).exists():
        baseline = np.load(cfg["baseline_file"])["mean"]

    pose_clf = None
    if pose:
        try:
            # Lazy import to avoid scikit-learn dependency unless requested
            from .pose_classifier import PoseClassifier
            pose_clf = PoseClassifier("models/wipose.joblib")
        except Exception as exc:  # pragma: no cover - classifier optional
            print(f"Pose classifier init failed: {exc}", file=sys.stderr)

    range_filter = AlphaBetaFilter(
        alpha=float(cfg.get("range_alpha", 0.65)),
        beta=float(cfg.get("range_beta", 0.15)),
        dt=float(cfg.get("range_dt", cfg.get("window_hop", 0.5))),
    )
    alpha = 0.2
    presence_ema = 0.0
    pose_ema = 0.0
    last_dir = "C"
    l_cnt = r_cnt = 0

    state = {
        "presence": "NO",
        "presence_conf": 0.0,
        "direction": "C",
        "bearing_deg": 0.0,
        "bearing_label": "center",
        "rssi_delta": 0.0,
        "distance_raw": float("nan"),
        "distance_filtered": float("nan"),
        "pose": "N/A",
        "pose_conf": 0.0,
    }
    stop = Event()
    tui_thread = None
    # Track whether any frames have arrived, to surface clearer diagnostics
    first_frame_seen = Event()
    if tui:
        mode = "LIVE (FeitCSI)" if replay_path is None else "REPLAY"
        tui_thread = Thread(
            target=tui_mod.run,
            args=(state, stop, out, mode, replay_path),
            daemon=True,
        )
        tui_thread.start()

    # Emit a helpful status and enforce a timeout if no frames are seen
    wait_timeout = float(cfg.get("frames_wait_timeout", 10.0))
    print("[status] Waiting for frames…", file=sys.stderr)

    def _timeout_watchdog():
        deadline = time.time() + wait_timeout
        while time.time() < deadline and not first_frame_seen.is_set() and not stop.is_set():
            time.sleep(0.2)
        if not first_frame_seen.is_set() and not stop.is_set():
            print(
                f"[error] No CSI frames received within {wait_timeout:.0f}s. "
                "Ensure FeitCSI is installed, privileges are granted, and the NIC is free.",
                file=sys.stderr,
            )
            sys.exit(3)

    Thread(target=_timeout_watchdog, daemon=True).start()

    def handle(result: dict) -> None:
        nonlocal presence_ema, pose_ema, last_dir, l_cnt, r_cnt
        raw = 1.0 if result["presence"] else 0.0
        presence_ema = alpha * raw + (1 - alpha) * presence_ema
        diff = float(result.get("rssi_delta", float("nan")))
        bearing = float(result.get("bearing_deg", float("nan")))
        aoa_delta = float(cfg.get("aoa_deadzone_deg", 7.5))
        rssi_thresh = float(cfg.get("rssi_delta", 3.5))
        if math.isfinite(bearing):
            if bearing < -aoa_delta:
                l_cnt += 1
                r_cnt = 0
            elif bearing > aoa_delta:
                r_cnt += 1
                l_cnt = 0
            else:
                l_cnt = r_cnt = 0
        elif math.isfinite(diff):
            if diff > rssi_thresh:
                l_cnt += 1
                r_cnt = 0
            elif diff < -rssi_thresh:
                r_cnt += 1
                l_cnt = 0
            else:
                l_cnt = r_cnt = 0
        else:
            l_cnt = r_cnt = 0
        if l_cnt >= 2:
            last_dir = "L"
        elif r_cnt >= 2:
            last_dir = "R"
        elif math.isfinite(bearing) and abs(bearing) <= aoa_delta:
            last_dir = "C"
        elif math.isfinite(diff) and abs(diff) <= rssi_thresh:
            last_dir = "C"
        pose_label = "N/A"
        pose_conf = 0.0
        if pose_clf is not None:
            pose_label, conf = pose_clf.predict(result["pose_feat"])
            pose_ema = alpha * conf + (1 - alpha) * pose_ema
            pose_conf = pose_ema
        filtered_distance = range_filter.update(result.get("distance"), result.get("ts"))
        state.update(
            presence="YES" if result["presence"] else "NO",
            presence_conf=presence_ema,
            direction=last_dir,
            bearing_deg=bearing,
            bearing_label={"L": "left", "R": "right", "C": "center"}[last_dir],
            rssi_delta=diff,
            distance_raw=result.get("distance"),
            distance_filtered=filtered_distance,
            pose=pose_label,
            pose_conf=pose_conf,
        )

        def _finite_or_none(val: float) -> Optional[float]:
            try:
                fval = float(val)
            except (TypeError, ValueError):
                return None
            return fval if math.isfinite(fval) else None

        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(result["ts"]))
        entry = {
            "timestamp": iso,
            "presence": bool(result["presence"]),
            "pose": pose_label.lower(),
            "direction": {"L": "left", "R": "right", "C": "center"}[last_dir],
            "bearing_label": {"L": "left", "R": "right", "C": "center"}[last_dir],
            "bearing_deg": _finite_or_none(bearing),
            "distance_m": _finite_or_none(filtered_distance),
            "distance_raw_m": _finite_or_none(result.get("distance")),
            "confidence": presence_ema,
        }
        print(json.dumps(entry))
        utils.rotate_file(cfg["output_file"], cfg["rotation_max_bytes"])
        utils.safe_json_append(cfg["output_file"], entry)

    def process() -> None:
        now = buffer[-1]["ts"]
        if not first_frame_seen.is_set():
            first_frame_seen.set()
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
            if not first_frame_seen.is_set():
                first_frame_seen.set()
            process()
    elif replay_path:
        for pkt in replay_mod.replay(replay_path, speed):
            if stop.is_set():
                break
            buffer.append(pkt)
            if not first_frame_seen.is_set():
                first_frame_seen.set()
            process()
    else:
        observer = Observer()
        log_path = Path(cfg["log_file"])
        wait = cfg.get("log_wait", 5.0)
        if not log_path.exists() and not utils.wait_for_file(log_path, wait):
            _capture_fail()
        _check_log_fresh(log_path)
        handler = CSILogHandler(log_path, buffer, process, pkt_cb=lambda: first_frame_seen.set())
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


def run(config_path: str | None = None) -> None:
    if config_path is None:
        config_path = str(Path(__file__).resolve().parent / "config.yaml")
    cfg = yaml.safe_load(open(config_path))
    buffer = deque()
    baseline = None
    if Path(cfg["baseline_file"]).exists():
        baseline = np.load(cfg["baseline_file"])["mean"]
    range_filter = AlphaBetaFilter(
        alpha=float(cfg.get("range_alpha", 0.65)),
        beta=float(cfg.get("range_beta", 0.15)),
        dt=float(cfg.get("range_dt", cfg.get("window_hop", 0.5))),
    )
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
            raw_distance = result.get("distance")
            filtered_distance = range_filter.update(raw_distance, result.get("ts"))
            bearing = float(result.get("bearing_deg", float("nan")))
            if isinstance(raw_distance, (int, float)) and math.isfinite(raw_distance):
                raw_out = float(raw_distance)
            else:
                raw_out = None
            iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(result["ts"]))
            entry = {
                "timestamp": iso,
                "presence": bool(result["presence"]),
                "pose": "n/a",
                "direction": {"L": "left", "R": "right", "C": "center"}[result["direction"]],
                "bearing_label": {"L": "left", "R": "right", "C": "center"}[result["direction"]],
                "bearing_deg": bearing if math.isfinite(bearing) else None,
                "distance_m": filtered_distance if math.isfinite(filtered_distance) else None,
                "distance_raw_m": raw_out,
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
    range_filter = AlphaBetaFilter(
        alpha=float(cfg.get("range_alpha", 0.65)),
        beta=float(cfg.get("range_beta", 0.15)),
        dt=float(cfg.get("range_dt", cfg.get("window_hop", 0.5))),
    )
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
                raw_distance = result.get("distance")
                filtered_distance = range_filter.update(raw_distance, result.get("ts"))
                result["distance_filtered"] = filtered_distance
                bearing = float(result.get("bearing_deg", float("nan")))
                result["bearing_label"] = {"L": "left", "R": "right", "C": "center"}[result["direction"]]
                if not math.isfinite(bearing):
                    result["bearing_deg"] = float("nan")
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
    parser.add_argument("--log", type=str, default=None, help="override path to input JSONL log")
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
        log_override=args.log,
    )


if __name__ == "__main__":
    main()
