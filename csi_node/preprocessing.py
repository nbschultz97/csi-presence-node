"""Signal conditioning for CSI data: Hampel outlier filter + Butterworth bandpass.

Applied per-subcarrier before windowed feature extraction to remove impulse
noise and isolate the human-motion frequency band (~0.1–10 Hz).
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.signal import butter, sosfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def hampel_filter(
    x: np.ndarray,
    window_size: int = 5,
    n_sigma: float = 3.0,
) -> np.ndarray:
    """Hampel outlier rejection along axis 0.

    Replaces samples that deviate more than ``n_sigma`` MADs from the local
    median with the median value.  Operates independently on each column
    (subcarrier).

    Args:
        x: 2-D array of shape (time_steps, features).
        window_size: One-sided window length for local median.
        n_sigma: Threshold in units of MAD (median absolute deviation).

    Returns:
        Filtered copy of *x* with outliers replaced.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    n = x.shape[0]
    k = 1.4826  # consistency constant for Gaussian MAD → σ
    half = window_size

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        med = np.median(x[lo:hi], axis=0)
        mad = k * np.median(np.abs(x[lo:hi] - med), axis=0)
        mask = np.abs(x[i] - med) > n_sigma * np.maximum(mad, 1e-10)
        x[i] = np.where(mask, med, x[i])
    return x


def butterworth_bandpass(
    x: np.ndarray,
    fs: float,
    low: float = 0.1,
    high: float = 10.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter along axis 0.

    Isolates the human-motion frequency band.  Falls back to returning *x*
    unchanged if SciPy is not installed or the array is too short.

    Args:
        x: 2-D array (time_steps, features).
        fs: Sampling rate in Hz.
        low: Low cutoff frequency in Hz.
        high: High cutoff frequency in Hz.
        order: Filter order.

    Returns:
        Bandpass-filtered copy of *x*.
    """
    if not SCIPY_AVAILABLE:
        return x

    nyq = fs / 2.0
    # Clamp to valid Nyquist range
    low_norm = max(low / nyq, 1e-5)
    high_norm = min(high / nyq, 0.9999)
    if low_norm >= high_norm or x.shape[0] < 3 * order:
        return x  # Too few samples or invalid band

    sos = butter(order, [low_norm, high_norm], btype="band", output="sos")
    return sosfilt(sos, x, axis=0).astype(x.dtype)


def condition_signal(
    csi: np.ndarray,
    fs: float = 30.0,
    hampel_window: int = 5,
    hampel_sigma: float = 3.0,
    bp_low: float = 0.1,
    bp_high: float = 10.0,
) -> np.ndarray:
    """Full signal conditioning pipeline: Hampel → Butterworth bandpass.

    Args:
        csi: 2-D array (time_steps, features) of CSI amplitudes.
        fs: Approximate sampling rate (Hz).
        hampel_window: Hampel filter one-sided window.
        hampel_sigma: Hampel outlier threshold (MAD units).
        bp_low: Bandpass low cutoff (Hz).
        bp_high: Bandpass high cutoff (Hz).

    Returns:
        Conditioned CSI array (same shape).
    """
    if csi.ndim == 1:
        csi = csi.reshape(-1, 1)
    out = hampel_filter(csi, window_size=hampel_window, n_sigma=hampel_sigma)
    out = butterworth_bandpass(out, fs=fs, low=bp_low, high=bp_high)
    return out
