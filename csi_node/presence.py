"""Advanced presence detection engine for through-wall sensing.

Combines multiple detection methods for robust presence detection:
1. Energy-based: Total CSI energy compared to empty-room baseline
2. Variance-based: Temporal variance of CSI (human motion causes fluctuation)
3. Spectral-based: Frequency content changes when humans are present
4. Adaptive thresholding: Learns environment baseline automatically

Designed for through-wall scenarios where signal is attenuated and noisy.
"""
from __future__ import annotations

import time
import json
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class PresenceState:
    """Current presence detection state for dashboard/streaming."""
    present: bool = False
    confidence: float = 0.0
    method: str = "none"  # which detector triggered
    energy_ratio: float = 0.0
    variance_ratio: float = 0.0
    spectral_ratio: float = 0.0
    movement: str = "none"  # none, stationary, moving, breathing
    movement_intensity: float = 0.0
    direction: str = "center"
    distance_m: float = 0.0
    timestamp: float = 0.0
    packets_per_sec: float = 0.0
    # Calibration state
    calibrated: bool = False
    baseline_energy: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class AdaptivePresenceDetector:
    """Multi-method presence detector with adaptive thresholds.

    Uses a fusion of energy, variance, and spectral methods to detect
    human presence through walls. Adapts to environment automatically
    during a calibration phase.

    Usage:
        detector = AdaptivePresenceDetector()
        detector.calibrate_start()  # Begin 30s empty-room calibration
        # ... feed CSI frames during calibration ...
        detector.calibrate_finish()

        # Normal operation
        state = detector.update(csi_amplitudes)
        if state.present:
            print(f"Human detected! Confidence: {state.confidence:.0%}")
    """

    def __init__(
        self,
        # Energy detector params
        energy_threshold_factor: float = 2.5,
        # Variance detector params
        variance_threshold_factor: float = 3.0,
        # Spectral detector params (breathing/micro-motion)
        spectral_threshold_factor: float = 2.0,
        breathing_band: tuple[float, float] = (0.1, 0.5),  # Hz
        motion_band: tuple[float, float] = (0.5, 5.0),  # Hz
        # Fusion params
        ema_alpha: float = 0.3,
        presence_threshold: float = 0.5,
        # Window params
        window_seconds: float = 3.0,
        sample_rate_hz: float = 30.0,
        # History for dashboard
        history_seconds: float = 60.0,
    ):
        self.energy_threshold_factor = energy_threshold_factor
        self.variance_threshold_factor = variance_threshold_factor
        self.spectral_threshold_factor = spectral_threshold_factor
        self.breathing_band = breathing_band
        self.motion_band = motion_band
        self.ema_alpha = ema_alpha
        self.presence_threshold = presence_threshold
        self.window_seconds = window_seconds
        self.sample_rate_hz = sample_rate_hz

        # Baseline (calibrated empty room)
        self._baseline_energy: float = 0.0
        self._baseline_variance: float = 0.0
        self._baseline_spectral: float = 0.0
        self._calibrated = False
        self._calibrating = False
        self._cal_samples: list[np.ndarray] = []

        # Running state
        self._ema_presence = 0.0
        self._ema_energy = 0.0
        self._ema_variance = 0.0
        self._prev_amps: Optional[np.ndarray] = None
        self._frame_buffer: deque = deque()
        self._frame_times: deque = deque()

        # History for dashboard graphs
        max_hist = int(history_seconds * 2)  # ~2 entries/sec
        self._history: deque[PresenceState] = deque(maxlen=max_hist)
        self._state = PresenceState()

    @property
    def state(self) -> PresenceState:
        return self._state

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def history(self) -> list[PresenceState]:
        return list(self._history)

    def calibrate_start(self) -> None:
        """Begin calibration — collect empty-room samples."""
        self._calibrating = True
        self._cal_samples = []

    def calibrate_finish(self) -> bool:
        """Finish calibration and compute baseline thresholds.

        Returns True if calibration succeeded (enough samples).
        """
        self._calibrating = False
        if len(self._cal_samples) < 10:
            return False

        # Stack all calibration samples
        all_amps = np.concatenate(self._cal_samples, axis=0)

        # Energy baseline
        self._baseline_energy = float(np.mean(np.sum(all_amps ** 2, axis=-1)))

        # Variance baseline (compute variance across sliding windows)
        win_size = min(int(self.sample_rate_hz * self.window_seconds), len(all_amps))
        if win_size > 1:
            variances = []
            for i in range(0, len(all_amps) - win_size, max(1, win_size // 2)):
                chunk = all_amps[i:i + win_size]
                variances.append(float(np.var(chunk)))
            self._baseline_variance = float(np.mean(variances)) if variances else 0.01
        else:
            self._baseline_variance = 0.01

        # Spectral baseline
        if len(all_amps) > 32:
            self._baseline_spectral = self._compute_spectral_energy(all_amps)
        else:
            self._baseline_spectral = 0.01

        # Avoid division by zero
        self._baseline_energy = max(self._baseline_energy, 1e-6)
        self._baseline_variance = max(self._baseline_variance, 1e-6)
        self._baseline_spectral = max(self._baseline_spectral, 1e-6)

        self._calibrated = True
        self._cal_samples = []
        return True

    def auto_calibrate(self, duration_samples: int = 100) -> None:
        """Quick auto-calibrate from first N samples (no human present assumed)."""
        self._cal_auto_remaining = duration_samples
        self._calibrating = True
        self._cal_samples = []

    def set_profile(self, profile: dict) -> None:
        """Apply a detection profile (through_wall, same_room, etc.)."""
        for key, value in profile.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_subcarrier_variance(self) -> list[float]:
        """Return per-subcarrier variance from recent frames (for heatmap)."""
        if len(self._frame_buffer) < 5:
            return []
        window = np.array(list(self._frame_buffer)[-30:])
        if window.ndim < 2:
            return []
        var = np.var(window, axis=0)
        max_var = np.max(var) if np.max(var) > 0 else 1.0
        return (var / max_var).tolist()

    def save_calibration(self, path: str | Path) -> None:
        """Save calibration to file."""
        data = {
            "baseline_energy": self._baseline_energy,
            "baseline_variance": self._baseline_variance,
            "baseline_spectral": self._baseline_spectral,
            "calibrated": self._calibrated,
            "timestamp": time.time(),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load_calibration(self, path: str | Path) -> bool:
        """Load calibration from file. Returns True if successful."""
        try:
            data = json.loads(Path(path).read_text())
            self._baseline_energy = data["baseline_energy"]
            self._baseline_variance = data["baseline_variance"]
            self._baseline_spectral = data["baseline_spectral"]
            self._calibrated = data.get("calibrated", True)
            return True
        except Exception:
            return False

    def update(self, amps: np.ndarray, rssi: Optional[list[float]] = None,
               timestamp: Optional[float] = None) -> PresenceState:
        """Process a new CSI amplitude frame and return presence state.

        Args:
            amps: CSI amplitude array, shape (subcarriers,) or (chains, subcarriers).
                  Flattened internally.
            rssi: Optional RSSI values [rssi0, rssi1]
            timestamp: Optional timestamp (defaults to time.time())

        Returns:
            Updated PresenceState
        """
        ts = timestamp or time.time()
        amps = amps.flatten().astype(np.float64)

        # Track frame rate
        self._frame_times.append(ts)
        while self._frame_times and ts - self._frame_times[0] > 2.0:
            self._frame_times.popleft()
        pps = len(self._frame_times) / max(ts - self._frame_times[0], 0.1) if len(self._frame_times) > 1 else 0

        # Calibration mode
        if self._calibrating:
            self._cal_samples.append(amps.reshape(1, -1))
            if hasattr(self, '_cal_auto_remaining'):
                self._cal_auto_remaining -= 1
                if self._cal_auto_remaining <= 0:
                    self.calibrate_finish()
                    del self._cal_auto_remaining

        # Add to frame buffer
        self._frame_buffer.append(amps)
        buf_max = int(self.sample_rate_hz * self.window_seconds * 1.5)
        while len(self._frame_buffer) > buf_max:
            self._frame_buffer.popleft()

        # Need enough frames for analysis
        min_frames = max(int(self.sample_rate_hz * 0.5), 5)
        if len(self._frame_buffer) < min_frames:
            self._state = PresenceState(
                timestamp=ts,
                packets_per_sec=pps,
                calibrated=self._calibrated,
                baseline_energy=self._baseline_energy,
            )
            return self._state

        # Get window of recent frames
        window = np.array(list(self._frame_buffer))

        # --- Energy detection ---
        energy = float(np.mean(np.sum(window ** 2, axis=-1)))
        if self._calibrated:
            energy_ratio = energy / self._baseline_energy
        else:
            # Without calibration, use running EMA as pseudo-baseline
            self._ema_energy = 0.95 * self._ema_energy + 0.05 * energy if self._ema_energy > 0 else energy
            energy_ratio = energy / max(self._ema_energy, 1e-6)

        energy_vote = 1.0 if energy_ratio > self.energy_threshold_factor else 0.0

        # --- Variance detection ---
        variance = float(np.var(window))
        if self._calibrated:
            var_ratio = variance / self._baseline_variance
        else:
            self._ema_variance = 0.95 * self._ema_variance + 0.05 * variance if self._ema_variance > 0 else variance
            var_ratio = variance / max(self._ema_variance, 1e-6)

        var_vote = 1.0 if var_ratio > self.variance_threshold_factor else 0.0

        # --- Spectral detection (breathing/micro-motion) ---
        spectral_ratio = 0.0
        spectral_vote = 0.0
        if len(window) >= 32:
            spec_energy = self._compute_spectral_energy(window)
            if self._calibrated:
                spectral_ratio = spec_energy / self._baseline_spectral
            else:
                spectral_ratio = 1.0  # Can't compare without baseline
            spectral_vote = 1.0 if spectral_ratio > self.spectral_threshold_factor else 0.0

        # --- Movement classification ---
        movement = "none"
        movement_intensity = 0.0
        if self._prev_amps is not None and len(window) > 1:
            frame_delta = float(np.mean(np.abs(window[-1] - self._prev_amps)))
            window_var = float(np.var(np.diff(window, axis=0)))

            if window_var > (self._baseline_variance * 10 if self._calibrated else 5.0):
                movement = "moving"
                movement_intensity = min(window_var / max(self._baseline_variance * 10, 1.0), 1.0)
            elif var_ratio > self.variance_threshold_factor * 0.5:
                movement = "breathing" if spectral_vote > 0 else "stationary"
                movement_intensity = min(var_ratio / self.variance_threshold_factor, 1.0)

        self._prev_amps = window[-1].copy()

        # --- Fusion: weighted vote ---
        # Energy and variance are primary; spectral is supplementary
        raw_score = 0.5 * energy_vote + 0.35 * var_vote + 0.15 * spectral_vote

        # Determine which method triggered
        if energy_vote > 0 and var_vote > 0:
            method = "energy+variance"
        elif energy_vote > 0:
            method = "energy"
        elif var_vote > 0:
            method = "variance"
        elif spectral_vote > 0:
            method = "spectral"
        else:
            method = "none"

        # EMA smoothing
        self._ema_presence = self.ema_alpha * raw_score + (1 - self.ema_alpha) * self._ema_presence
        present = self._ema_presence > self.presence_threshold

        # Direction from RSSI
        direction = "center"
        if rssi and len(rssi) >= 2:
            diff = rssi[0] - rssi[1]
            if diff > 2.0:
                direction = "left"
            elif diff < -2.0:
                direction = "right"

        self._state = PresenceState(
            present=present,
            confidence=self._ema_presence,
            method=method,
            energy_ratio=energy_ratio,
            variance_ratio=var_ratio,
            spectral_ratio=spectral_ratio,
            movement=movement if present else "none",
            movement_intensity=movement_intensity if present else 0.0,
            direction=direction,
            distance_m=0.0,  # TODO: integrate RSSI distance
            timestamp=ts,
            packets_per_sec=pps,
            calibrated=self._calibrated,
            baseline_energy=self._baseline_energy,
        )

        self._history.append(self._state)
        return self._state

    def _compute_spectral_energy(self, window: np.ndarray) -> float:
        """Compute energy in human-relevant frequency bands."""
        try:
            # Average across subcarriers first
            signal = np.mean(window, axis=-1) if window.ndim > 1 else window
            n = len(signal)
            if n < 8:
                return 0.0

            fft = np.fft.rfft(signal - np.mean(signal))
            power = np.abs(fft) ** 2
            freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate_hz)

            # Energy in breathing band (0.1-0.5 Hz)
            breath_mask = (freqs >= self.breathing_band[0]) & (freqs <= self.breathing_band[1])
            # Energy in motion band (0.5-5.0 Hz)
            motion_mask = (freqs >= self.motion_band[0]) & (freqs <= self.motion_band[1])

            breath_energy = float(np.sum(power[breath_mask])) if np.any(breath_mask) else 0.0
            motion_energy = float(np.sum(power[motion_mask])) if np.any(motion_mask) else 0.0

            return breath_energy + motion_energy
        except Exception:
            return 0.0

    def get_dashboard_data(self) -> dict:
        """Get data formatted for the web dashboard."""
        history = self.history[-120:]  # Last 60 seconds at ~2/sec

        return {
            "current": self._state.to_dict(),
            "history": {
                "timestamps": [h.timestamp for h in history],
                "confidence": [h.confidence for h in history],
                "energy_ratio": [h.energy_ratio for h in history],
                "variance_ratio": [h.variance_ratio for h in history],
                "spectral_ratio": [h.spectral_ratio for h in history],
                "movement_intensity": [h.movement_intensity for h in history],
            },
            "calibration": {
                "calibrated": self._calibrated,
                "baseline_energy": self._baseline_energy,
                "baseline_variance": self._baseline_variance,
            },
        }
