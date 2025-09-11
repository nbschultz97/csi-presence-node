from __future__ import annotations
"""Thin wrappers around FeitCSI's Python ``CSIExtractor`` interface."""

import time
from typing import Iterator
import numpy as np


def _frame_to_pkt(frame) -> dict:
    """Convert a CSIExtractor frame to pipeline packet dict."""
    ts = getattr(frame, "timestamp", time.time())
    rssi = getattr(frame, "rssi", [float("nan"), float("nan")])
    csi = np.array(getattr(frame, "csi", []), dtype=float)
    return {"ts": float(ts), "rssi": rssi, "csi": csi}


def live_stream(iface: str) -> Iterator[dict]:
    """Yield packets from ``iface`` using FeitCSI's ``CSIExtractor``.

    Raises ``RuntimeError`` if the module is unavailable. This function is only
    exercised on systems with FeitCSI installed and an Intel AX210 NIC.
    """
    try:
        from feitcsi import CSIExtractor  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("CSIExtractor not available") from exc

    extractor = CSIExtractor(iface)  # pragma: no cover - hardware dependant
    for frame in extractor:  # pragma: no cover - hardware dependant
        yield _frame_to_pkt(frame)


def ftm_stream(path: str) -> Iterator[dict]:
    """Yield packets from a saved ``.ftm`` log via ``CSIExtractor``."""
    try:
        from feitcsi import CSIExtractor  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("CSIExtractor not available") from exc

    extractor = CSIExtractor(path)  # pragma: no cover
    for frame in extractor:  # pragma: no cover
        yield _frame_to_pkt(frame)
