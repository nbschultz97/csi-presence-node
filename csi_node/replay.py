from __future__ import annotations
"""Offline replay of CSI log files yielding packets like the live parser."""

import base64
import os
import tempfile
import time
from pathlib import Path
from typing import Iterator

from . import utils


def decode_b64_capture(src: Path) -> Path:
    """Decode ``src`` (base64 text) to a temporary file and return its path."""
    data = base64.b64decode(src.read_text())
    fd, tmp = tempfile.mkstemp(suffix=".dat")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return Path(tmp)


def replay(path: str, speed: float = 1.0) -> Iterator[dict]:
    """Yield packets from ``path`` at a given speed factor."""
    p = Path(path)
    tmp: Path | None = None
    if p.suffix == ".b64":
        tmp = decode_b64_capture(p)
        p = tmp

    def _iter() -> Iterator[dict]:
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

    try:
        yield from _iter()
    finally:
        if tmp and tmp.exists():
            tmp.unlink()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Replay CSI log file")
    parser.add_argument("--file", required=True, help="log file (.log or .b64)")
    parser.add_argument("--speed", type=float, default=1.0, help="speed factor")
    args = parser.parse_args()
    for _ in replay(args.file, args.speed):
        pass


if __name__ == "__main__":
    main()
