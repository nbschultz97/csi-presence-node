"""Realtime CSI presence pipeline."""
import time
import yaml
from pathlib import Path
from collections import deque
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from . import utils


class CSILogHandler(FileSystemEventHandler):
    def __init__(self, path: Path, buffer, process_cb):
        self.path = str(path)
        self.buffer = buffer
        self.process_cb = process_cb
        self._fp = open(self.path, "r")
        self._fp.seek(0, 2)

    def on_modified(self, event):
        if event.src_path != self.path:
            return
        while True:
            line = self._fp.readline()
            if not line:
                break
            pkt = utils.parse_csi_line(line)
            if pkt:
                self.buffer.append(pkt)
                self.process_cb()


def compute_window(buffer, start_ts, end_ts, baseline, cfg):
    window = [p for p in buffer if start_ts <= p["ts"] <= end_ts]
    if not window:
        return None
    amps = np.stack([p["csi"] for p in window], axis=0)
    if baseline is not None:
        amps = amps - baseline
    amps = amps.reshape(amps.shape[0], -1)
    var = float(np.var(amps))
    pca1 = float(utils.compute_pca(amps)[0])
    rssi0 = rssi1 = float("nan")
    direction = "C"
    rssis = [p.get("rssi") for p in window if p.get("rssi")]
    if rssis:
        rssi0 = float(np.mean([r[0] for r in rssis]))
        rssi1 = float(np.mean([r[1] for r in rssis]))
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

    observer = Observer()
    handler = CSILogHandler(Path(cfg["log_file"]), buffer, process)
    observer.schedule(handler, str(Path(cfg["log_file"]).parent), recursive=False)
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


if __name__ == "__main__":
    run()
