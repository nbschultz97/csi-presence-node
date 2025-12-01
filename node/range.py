"""Range estimation primitives using log-distance path-loss."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np


def fit_pathloss(
    baseline: Union[float, Tuple[float, float]],
    points: Sequence[Tuple[float, float]] | None = None,
) -> Tuple[float, float]:
    """Fit ``n`` and ``C`` for ``Pr = C - 10 n log10(d)``."""
    data: list[Tuple[float, float]] = []
    if isinstance(baseline, (tuple, list)) and len(baseline) >= 2:
        d0 = float(baseline[0])
        pr0 = float(baseline[1])
        data.append((d0, pr0))
    else:
        data.append((1.0, float(baseline)))
    if points:
        for d, pr in points:
            data.append((float(d), float(pr)))
    arr = np.array(data, dtype=float)
    mask = (arr[:, 0] > 0) & np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 0])
    arr = arr[mask]
    if arr.shape[0] < 2:
        raise ValueError("Need at least two (distance, RSSI) pairs")
    logd = np.log10(arr[:, 0])
    y = arr[:, 1]
    a = np.vstack([logd, np.ones_like(logd)]).T
    slope, intercept = np.linalg.lstsq(a, y, rcond=None)[0]
    n = -slope / 10.0
    C = intercept
    return float(n), float(C)


def distance_from_rss(Pr: float, n: float, C: float, Pt: Optional[float]) -> float:
    """Return distance for observed ``Pr`` given model parameters."""
    try:
        pr = float(Pr)
        n = float(n)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(pr) or not math.isfinite(n) or n <= 0:
        return float("nan")
    intercept = None
    for candidate in (C, Pt):
        try:
            val = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            intercept = val
            break
    if intercept is None:
        return float("nan")
    exponent = (intercept - pr) / (10.0 * n)
    return float(10 ** exponent)


@dataclass
class AlphaBetaFilter:
    """1D alpha-beta tracker for smooth range output."""

    alpha: float = 0.65
    beta: float = 0.15
    dt: float = 0.5

    _x: Optional[float] = None
    _v: float = 0.0
    _last_ts: Optional[float] = None

    def update(self, measurement: Optional[float], timestamp: Optional[float] = None) -> float:
        if measurement is None or not math.isfinite(measurement):
            return float(self._x) if self._x is not None else float("nan")
        meas = float(measurement)
        if self._x is None:
            self._x = meas
            self._v = 0.0
            self._last_ts = timestamp
            return meas
        dt = self.dt
        if timestamp is not None and self._last_ts is not None:
            raw_dt = timestamp - self._last_ts
            if raw_dt > 0:
                dt = max(0.05, min(raw_dt, 2.0))
        self._last_ts = timestamp
        x_pred = self._x + self._v * dt
        r = meas - x_pred
        self._x = x_pred + self.alpha * r
        self._v = self._v + (self.beta * r) / dt
        return float(self._x)

    def reset(self) -> None:
        self._x = None
        self._v = 0.0
        self._last_ts = None
