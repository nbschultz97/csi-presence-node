"""Record empty-room baseline for CSI amplitudes."""
import argparse
import time
import numpy as np
from pathlib import Path
from . import utils


def record(log_path: Path, duration: float, outfile: Path) -> None:
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
    args = ap.parse_args()
    record(Path(args.log), args.duration, Path(args.out))


if __name__ == "__main__":
    main()
