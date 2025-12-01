"""Angle-of-arrival helpers for two-element linear arrays."""
from __future__ import annotations

import math
from typing import Iterable, Union

import numpy as np

_SPEED_OF_LIGHT = 299_792_458.0  # m/s


def _to_complex(arr: Union[np.ndarray, Iterable]) -> np.ndarray:
    data = np.asarray(arr)
    if data.size == 0:
        return np.array([], dtype=np.complex128)
    if np.iscomplexobj(data):
        return data.astype(np.complex128, copy=False)
    if data.ndim >= 1 and data.shape[-1] == 2:
        real = data[..., 0]
        imag = data[..., 1]
        data = real + 1j * imag
        return np.asarray(data, dtype=np.complex128)
    return data.astype(np.complex128)


def aoa_deg(
    csi1: Iterable,
    csi2: Iterable,
    freqs_hz: Union[Iterable, float],
    d_m: float,
    theta_offset: float = 0.0,
) -> float:
    """Return the calibrated arrival angle in degrees for two CSI streams."""
    if d_m <= 0:
        return float("nan")
    v1 = _to_complex(csi1).ravel()
    v2 = _to_complex(csi2).ravel()
    if v1.size == 0 or v2.size == 0:
        return float("nan")
    n = min(v1.size, v2.size)
    v1 = v1[:n]
    v2 = v2[:n]

    freqs = np.asarray(freqs_hz, dtype=float).ravel()
    if freqs.size == 0:
        return float("nan")
    if freqs.size == 1:
        freqs = np.full(n, float(freqs[0]))
    elif freqs.size != n:
        freqs = np.resize(freqs, n)

    mask = (
        np.isfinite(v1)
        & np.isfinite(v2)
        & np.isfinite(freqs)
        & (np.abs(v1) > 0)
        & (np.abs(v2) > 0)
        & (freqs > 0)
    )
    if not np.any(mask):
        return float("nan")
    v1 = v1[mask]
    v2 = v2[mask]
    freqs = freqs[mask]

    phase_diff = np.angle(v1 * np.conj(v2))
    if phase_diff.size > 1:
        phase_diff = np.unwrap(phase_diff)
    lam = _SPEED_OF_LIGHT / freqs
    sin_theta = phase_diff * lam / (2.0 * math.pi * d_m)
    weights = np.abs(v1) * np.abs(v2)
    weights = np.where(weights > 0, weights, 1.0)
    sin_est = np.average(sin_theta, weights=weights)
    sin_clamped = float(np.clip(sin_est, -1.0, 1.0))
    theta = math.degrees(math.asin(sin_clamped))
    theta -= float(theta_offset)
    if not math.isfinite(theta):
        return float("nan")
    theta = ((theta + 180.0) % 360.0) - 180.0
    theta = max(min(theta, 89.9), -89.9)
    return float(theta)


def calibrate_theta_offset(samples: Iterable[float]) -> float:
    vals = np.asarray(list(samples), dtype=float)
    if vals.size == 0:
        return 0.0
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    radians = np.deg2rad(vals)
    vec = np.exp(1j * radians)
    mean = np.mean(vec)
    if mean == 0:
        return 0.0
    angle = math.degrees(math.atan2(mean.imag, mean.real))
    return float(angle)
