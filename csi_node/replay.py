from __future__ import annotations
"""Offline replay of CSI log files yielding packets like the live parser."""

import time
from pathlib import Path
from typing import Iterator

from . import utils


def replay(path: str, speed: float = 1.0) -> Iterator[dict]:
    """Yield packets from ``path`` at a given speed factor."""
    p = Path(path)
    with open(p, "r") as f:
        prev_ts = None
        for line in f:
            pkt = utils.parse_csi_line(line)
            if not pkt:
                continue
            if prev_ts is not None and speed > 0:
                delay = (pkt["ts"] - prev_ts) / speed
                if delay > 0:
                    time.sleep(delay)
            prev_ts = pkt["ts"]
            yield pkt


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Replay CSI log file")
    parser.add_argument("path", help="log file")
    parser.add_argument("--speed", type=float, default=1.0, help="speed factor")
    args = parser.parse_args()
    for _ in replay(args.path, args.speed):
        pass


if __name__ == "__main__":
    main()
