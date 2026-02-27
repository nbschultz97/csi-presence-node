"""Breathing and micro-movement detection from WiFi CSI data.

Detects respiratory patterns and subtle body movements by analyzing
low-frequency periodic components in CSI amplitude and phase data.
Uses Doppler analysis in the 0.1-0.5 Hz band (6-30 breaths/min).

Key capabilities:
- Breathing rate estimation (BPM) via FFT peak detection
- Breathing confidence scoring
- Micro-movement detection (fidgeting, swaying)
- Apnea/absence-of-breathing alerting
- Multi-person breathing separation (experimental)
- Integration with AdaptivePresenceDetector

Patent alignment: Implements the Doppler-based breathing detection
described in provisional patent claims for passive WiFi CSI sensing.

Usage:
    detector = BreathingDetector(sample_rate_hz=30.0)
    for csi_frame in stream:
        result = detector.update(csi_frame)
        if result.breathing_detected:
            print(f"Breathing at {result.rate_bpm:.1f} BPM, "
                  f"confidence {result.confidence:.0%}")
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

try:
    from scipy.signal import butter, sosfilt, find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# Normal adult breathing: 12-20 BPM = 0.2-0.33 Hz
# Extended range: 6-30 BPM = 0.1-0.5 Hz (covers sleep, exercise, children)
MIN_BREATHING_HZ = 0.1   # 6 BPM
MAX_BREATHING_HZ = 0.5   # 30 BPM
NORMAL_BREATHING_HZ = (0.2, 0.33)  # 12-20 BPM

# Micro-movement band: 0.5-2.0 Hz (fidgeting, swaying, typing)
MIN_MICRO_HZ = 0.5
MAX_MICRO_HZ = 2.0


@dataclass
class BreathingState:
    """Current breathing detection state."""
    breathing_detected: bool = False
    rate_bpm: float = 0.0
    rate_hz: float = 0.0
    confidence: float = 0.0
    breathing_amplitude: float = 0.0
    # Micro-movement
    micro_movement_detected: bool = False
    micro_movement_intensity: float = 0.0
    # Apnea detection
    apnea_alert: bool = False
    seconds_since_breath: float = 0.0
    # Multi-person (experimental)
    estimated_persons: int = 0
    secondary_rate_bpm: float = 0.0
    # Metadata
    timestamp: float = 0.0
    window_seconds: float = 0.0
    spectral_snr_db: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class BreathingDetector:
    """Detect breathing and micro-movements from CSI amplitude data.

    Analyzes the low-frequency periodic content of CSI signals to detect
    respiratory patterns. Works by:
    1. Accumulating CSI frames in a sliding window
    2. Extracting the mean amplitude signal across subcarriers
    3. Bandpass filtering to isolate the breathing band (0.1-0.5 Hz)
    4. FFT-based peak detection for rate estimation
    5. Confidence scoring based on spectral SNR and peak prominence

    Args:
        sample_rate_hz: CSI frame rate (packets/sec).
        window_seconds: Analysis window length. Longer = better frequency
            resolution but slower response. 15-30s recommended.
        min_bpm: Minimum breathing rate to detect (BPM).
        max_bpm: Maximum breathing rate to detect (BPM).
        confidence_threshold: Minimum confidence to report detection.
        apnea_timeout_s: Seconds without detected breathing before apnea alert.
        subcarrier_weights: Optional per-subcarrier weights. If None, uses
            variance-based automatic weighting (subcarriers with more
            breathing-band variance get higher weight).
    """

    def __init__(
        self,
        sample_rate_hz: float = 30.0,
        window_seconds: float = 20.0,
        min_bpm: float = 6.0,
        max_bpm: float = 30.0,
        confidence_threshold: float = 0.3,
        apnea_timeout_s: float = 30.0,
        subcarrier_weights: Optional[np.ndarray] = None,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.window_seconds = window_seconds
        self.min_hz = min_bpm / 60.0
        self.max_hz = max_bpm / 60.0
        self.confidence_threshold = confidence_threshold
        self.apnea_timeout_s = apnea_timeout_s
        self.subcarrier_weights = subcarrier_weights

        # Frame buffer
        max_frames = int(sample_rate_hz * window_seconds * 1.2)
        self._buffer: deque[np.ndarray] = deque(maxlen=max_frames)
        self._timestamps: deque[float] = deque(maxlen=max_frames)

        # State tracking
        self._state = BreathingState()
        self._last_breath_time: Optional[float] = None
        self._rate_history: deque[float] = deque(maxlen=10)  # smooth rate
        self._history: deque[BreathingState] = deque(maxlen=200)

        # Baseline for micro-movement (learned from first N frames)
        self._baseline_micro_energy: float = 0.0
        self._baseline_frames: int = 0
        self._baseline_target: int = int(sample_rate_hz * 5)  # 5s baseline

    @property
    def state(self) -> BreathingState:
        return self._state

    @property
    def history(self) -> list[BreathingState]:
        return list(self._history)

    @property
    def min_frames_needed(self) -> int:
        """Minimum frames needed for a meaningful FFT analysis."""
        # Need at least 2 full breathing cycles at the lowest frequency
        min_period = 1.0 / max(self.min_hz, 0.05)
        return int(self.sample_rate_hz * min_period * 2)

    def reset(self) -> None:
        """Clear all state and start fresh."""
        self._buffer.clear()
        self._timestamps.clear()
        self._last_breath_time = None
        self._rate_history.clear()
        self._history.clear()
        self._baseline_micro_energy = 0.0
        self._baseline_frames = 0
        self._state = BreathingState()

    def update(self, amps: np.ndarray, timestamp: Optional[float] = None) -> BreathingState:
        """Process a new CSI frame and update breathing state.

        Args:
            amps: CSI amplitude array, shape (subcarriers,) or (chains, subcarriers).
            timestamp: Optional timestamp (defaults to time.time()).

        Returns:
            Updated BreathingState.
        """
        ts = timestamp or time.time()
        amps = np.asarray(amps, dtype=np.float64).flatten()

        self._buffer.append(amps)
        self._timestamps.append(ts)

        n_frames = len(self._buffer)
        min_needed = self.min_frames_needed

        if n_frames < min_needed:
            self._state = BreathingState(
                timestamp=ts,
                window_seconds=n_frames / max(self.sample_rate_hz, 1),
            )
            return self._state

        # Build analysis window
        window = np.array(list(self._buffer))
        actual_window_s = n_frames / max(self.sample_rate_hz, 1)

        # Step 1: Compute weighted mean signal across subcarriers
        signal = self._extract_breathing_signal(window)

        # Step 2: Bandpass filter for breathing band
        filtered = self._bandpass_filter(signal, self.min_hz, self.max_hz)

        # Step 3: FFT analysis for rate estimation
        rate_hz, confidence, snr_db, amplitude = self._estimate_rate(filtered)
        rate_bpm = rate_hz * 60.0

        # Step 4: Micro-movement detection (higher frequency band)
        micro_detected, micro_intensity = self._detect_micro_movement(signal)

        # Step 5: Update baseline during initial period
        if self._baseline_frames < self._baseline_target:
            micro_energy = self._compute_band_energy(
                signal, MIN_MICRO_HZ, MAX_MICRO_HZ
            )
            alpha = 0.9
            self._baseline_micro_energy = (
                alpha * self._baseline_micro_energy + (1 - alpha) * micro_energy
                if self._baseline_frames > 0
                else micro_energy
            )
            self._baseline_frames += 1

        # Step 6: Multi-person breathing separation (experimental)
        n_persons, secondary_bpm = self._detect_multiple_breathers(filtered)

        # Step 7: Smooth rate with history
        breathing_detected = confidence >= self.confidence_threshold
        if breathing_detected:
            self._rate_history.append(rate_bpm)
            self._last_breath_time = ts
            if len(self._rate_history) > 1:
                rate_bpm = float(np.median(list(self._rate_history)))
                rate_hz = rate_bpm / 60.0

        # Step 8: Apnea detection
        apnea_alert = False
        seconds_since = 0.0
        if self._last_breath_time is not None:
            seconds_since = ts - self._last_breath_time
            if seconds_since > self.apnea_timeout_s:
                apnea_alert = True
        elif n_frames > min_needed * 2:
            # Never detected a breath but have enough data
            apnea_alert = not breathing_detected

        self._state = BreathingState(
            breathing_detected=breathing_detected,
            rate_bpm=rate_bpm if breathing_detected else 0.0,
            rate_hz=rate_hz if breathing_detected else 0.0,
            confidence=confidence,
            breathing_amplitude=amplitude,
            micro_movement_detected=micro_detected,
            micro_movement_intensity=micro_intensity,
            apnea_alert=apnea_alert,
            seconds_since_breath=seconds_since,
            estimated_persons=n_persons if breathing_detected else 0,
            secondary_rate_bpm=secondary_bpm,
            timestamp=ts,
            window_seconds=actual_window_s,
            spectral_snr_db=snr_db,
        )

        self._history.append(self._state)
        return self._state

    def _extract_breathing_signal(self, window: np.ndarray) -> np.ndarray:
        """Extract the primary breathing signal from multi-subcarrier data.

        Uses variance-based weighting: subcarriers with more low-frequency
        variance contribute more to the output signal. This naturally
        selects the subcarriers most affected by breathing.
        """
        if window.ndim == 1:
            return window - np.mean(window)

        if self.subcarrier_weights is not None:
            weights = self.subcarrier_weights
            if len(weights) != window.shape[1]:
                weights = np.ones(window.shape[1])
        else:
            # Automatic: weight by variance in breathing band
            # Simple proxy: overall temporal variance per subcarrier
            var_per_sub = np.var(window, axis=0)
            total_var = np.sum(var_per_sub)
            if total_var > 0:
                weights = var_per_sub / total_var
            else:
                weights = np.ones(window.shape[1]) / window.shape[1]

        # Weighted average across subcarriers
        signal = window @ weights
        # Remove DC (mean)
        signal = signal - np.mean(signal)
        return signal

    def _bandpass_filter(
        self, signal: np.ndarray, low_hz: float, high_hz: float
    ) -> np.ndarray:
        """Apply bandpass filter to isolate a frequency band."""
        if not SCIPY_AVAILABLE or len(signal) < 20:
            return signal

        nyq = self.sample_rate_hz / 2.0
        low_norm = max(low_hz / nyq, 1e-5)
        high_norm = min(high_hz / nyq, 0.9999)

        if low_norm >= high_norm:
            return signal

        try:
            sos = butter(3, [low_norm, high_norm], btype="band", output="sos")
            return sosfilt(sos, signal)
        except Exception:
            return signal

    def _estimate_rate(
        self, filtered_signal: np.ndarray
    ) -> tuple[float, float, float, float]:
        """Estimate breathing rate from filtered signal using FFT.

        Returns:
            (rate_hz, confidence, snr_db, amplitude)
        """
        n = len(filtered_signal)
        if n < 16:
            return 0.0, 0.0, 0.0, 0.0

        # Apply Hanning window to reduce spectral leakage
        windowed = filtered_signal * np.hanning(n)

        # FFT
        fft_vals = np.fft.rfft(windowed)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate_hz)

        # Mask to breathing band
        mask = (freqs >= self.min_hz) & (freqs <= self.max_hz)
        if not np.any(mask):
            return 0.0, 0.0, 0.0, 0.0

        band_power = power[mask]
        band_freqs = freqs[mask]

        if len(band_power) == 0 or np.max(band_power) == 0:
            return 0.0, 0.0, 0.0, 0.0

        # Find peak
        peak_idx = int(np.argmax(band_power))
        peak_freq = band_freqs[peak_idx]
        peak_power = band_power[peak_idx]

        # Amplitude: sqrt of peak power, normalized
        amplitude = float(np.sqrt(peak_power) * 2.0 / n)

        # SNR: peak power vs mean of non-peak power in band
        non_peak = np.delete(band_power, peak_idx)
        noise_floor = float(np.mean(non_peak)) if len(non_peak) > 0 else 1e-10
        noise_floor = max(noise_floor, 1e-10)
        snr = peak_power / noise_floor
        snr_db = float(10 * np.log10(max(snr, 1e-10)))

        # Confidence from SNR and peak prominence
        # SNR > 10 dB is strong; > 5 dB is moderate; < 3 dB is weak
        if snr_db > 10:
            confidence = min(0.95, 0.7 + (snr_db - 10) * 0.025)
        elif snr_db > 5:
            confidence = 0.4 + (snr_db - 5) * 0.06
        elif snr_db > 3:
            confidence = 0.2 + (snr_db - 3) * 0.1
        else:
            confidence = max(0.0, snr_db * 0.05)

        # Boost confidence if rate is in normal range
        if NORMAL_BREATHING_HZ[0] <= peak_freq <= NORMAL_BREATHING_HZ[1]:
            confidence = min(1.0, confidence * 1.15)

        # Use scipy find_peaks for prominence-based validation if available
        if SCIPY_AVAILABLE and len(band_power) > 3:
            try:
                peaks, props = find_peaks(
                    band_power, prominence=np.max(band_power) * 0.1
                )
                if len(peaks) == 0:
                    confidence *= 0.5  # No clear peak
                elif len(peaks) == 1:
                    confidence = min(1.0, confidence * 1.1)  # Clean single peak
                # Multiple peaks reduce confidence (ambiguous)
                elif len(peaks) > 3:
                    confidence *= 0.7
            except Exception:
                pass

        confidence = float(np.clip(confidence, 0.0, 1.0))

        return float(peak_freq), confidence, snr_db, amplitude

    def _detect_micro_movement(
        self, signal: np.ndarray
    ) -> tuple[bool, float]:
        """Detect micro-movements (fidgeting, swaying) in 0.5-2.0 Hz band."""
        n = len(signal)
        if n < 32:
            return False, 0.0

        energy = self._compute_band_energy(signal, MIN_MICRO_HZ, MAX_MICRO_HZ)

        if self._baseline_micro_energy > 0 and self._baseline_frames >= self._baseline_target:
            ratio = energy / max(self._baseline_micro_energy, 1e-10)
            detected = ratio > 3.0
            intensity = float(np.clip((ratio - 1.0) / 10.0, 0.0, 1.0))
        else:
            # No baseline yet — use absolute threshold
            detected = energy > 0.1
            intensity = float(np.clip(energy, 0.0, 1.0))

        return detected, intensity

    def _compute_band_energy(
        self, signal: np.ndarray, low_hz: float, high_hz: float
    ) -> float:
        """Compute total spectral energy in a frequency band."""
        n = len(signal)
        if n < 8:
            return 0.0

        fft_vals = np.fft.rfft(signal * np.hanning(n))
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate_hz)

        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if not np.any(mask):
            return 0.0

        return float(np.sum(power[mask])) / n

    def _detect_multiple_breathers(
        self, filtered_signal: np.ndarray
    ) -> tuple[int, float]:
        """Attempt to detect multiple breathing rates (experimental).

        Looks for multiple distinct peaks in the breathing frequency band.

        Returns:
            (estimated_persons, secondary_rate_bpm)
        """
        n = len(filtered_signal)
        if n < 64 or not SCIPY_AVAILABLE:
            return 0, 0.0

        windowed = filtered_signal * np.hanning(n)
        fft_vals = np.fft.rfft(windowed)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate_hz)

        mask = (freqs >= self.min_hz) & (freqs <= self.max_hz)
        band_power = power[mask]
        band_freqs = freqs[mask]

        if len(band_power) < 5:
            return 0, 0.0

        try:
            # Find peaks with minimum separation of 0.05 Hz (~3 BPM)
            min_dist = max(1, int(0.05 / (freqs[1] - freqs[0]))) if len(freqs) > 1 else 1
            peaks, props = find_peaks(
                band_power,
                prominence=np.max(band_power) * 0.15,
                distance=min_dist,
            )

            if len(peaks) == 0:
                return 0, 0.0
            elif len(peaks) == 1:
                return 1, 0.0
            else:
                # Sort by power (descending)
                sorted_peaks = sorted(peaks, key=lambda i: band_power[i], reverse=True)
                primary_freq = band_freqs[sorted_peaks[0]]
                secondary_freq = band_freqs[sorted_peaks[1]]

                # Only count as separate person if rates differ by > 2 BPM
                if abs(primary_freq - secondary_freq) * 60 > 2.0:
                    return min(len(sorted_peaks), 3), float(secondary_freq * 60.0)
                else:
                    return 1, 0.0
        except Exception:
            return 0, 0.0

    def get_dashboard_data(self) -> dict:
        """Get data formatted for web dashboard visualization."""
        recent = list(self._history)[-60:]
        return {
            "current": self._state.to_dict(),
            "history": {
                "timestamps": [h.timestamp for h in recent],
                "rate_bpm": [h.rate_bpm for h in recent],
                "confidence": [h.confidence for h in recent],
                "amplitude": [h.breathing_amplitude for h in recent],
                "micro_intensity": [h.micro_movement_intensity for h in recent],
            },
        }
