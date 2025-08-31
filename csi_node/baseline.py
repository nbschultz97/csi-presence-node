"""Record empty-room baseline for CSI amplitudes."""
import argparse
import sys
import time
import numpy as np
from pathlib import Path
from . import utils


def record(log_path: Path, duration: float, outfile: Path, wait: float = 5.0) -> None:
    """Capture a baseline CSI sample when the log file is available."""
    log_path = Path(log_path)
    if not log_path.exists():
        if not utils.wait_for_file(log_path, wait):
            print(
                f"ERROR: log file {log_path} not found. Run scripts/10_csi_capture.sh first.",
                file=sys.stderr,
            )
            sys.exit(1)
    start = time.time()
    amps = []
    with open(log_path, "r") as f:
        while time.time() - start < duration:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            pkt = utils.parse_csi_line(line)
            if pkt and pkt["csi"].size:
                amps.append(pkt["csi"])
    if not amps:
        raise RuntimeError("no CSI captured for baseline")
    mean_amp = np.mean(np.stack(amps, axis=0), axis=0)
    np.savez(outfile, mean=mean_amp)
    print(f"Saved baseline to {outfile}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="./data/csi_raw.log")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--out", default="./data/baseline.npz")
    ap.add_argument("--wait", type=float, default=5.0, help="seconds to wait for log file")
    args = ap.parse_args()
    record(Path(args.log), args.duration, Path(args.out), wait=args.wait)


if __name__ == "__main__":
    main()
