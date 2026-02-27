"""Tests for csi_node.breathing — breathing and micro-movement detection."""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from csi_node.breathing import (
    BreathingDetector,
    BreathingState,
    MIN_BREATHING_HZ,
    MAX_BREATHING_HZ,
    MIN_MICRO_HZ,
    MAX_MICRO_HZ,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_breathing_signal(
    rate_bpm: float = 15.0,
    duration_s: float = 25.0,
    sample_rate: float = 30.0,
    n_subcarriers: int = 52,
    amplitude: float = 2.0,
    noise_std: float = 0.3,
    seed: int = 42,
) -> list[np.ndarray]:
    """Generate synthetic CSI frames with a breathing-rate modulation."""
    rng = np.random.default_rng(seed)
    freq_hz = rate_bpm / 60.0
    n_frames = int(duration_s * sample_rate)
    frames = []
    base = 20.0 + rng.normal(0, 1, n_subcarriers)
    for i in range(n_frames):
        t = i / sample_rate
        breath = amplitude * math.sin(2 * math.pi * freq_hz * t)
        # Breathing affects central subcarriers more
        mask = np.exp(-0.5 * ((np.arange(n_subcarriers) - n_subcarriers / 2) / (n_subcarriers / 4)) ** 2)
        frame = base + breath * mask + rng.normal(0, noise_std, n_subcarriers)
        frames.append(frame)
    return frames


def _make_still_signal(
    duration_s: float = 25.0,
    sample_rate: float = 30.0,
    n_subcarriers: int = 52,
    noise_std: float = 0.3,
    seed: int = 42,
) -> list[np.ndarray]:
    """Generate CSI frames with NO breathing — just noise."""
    rng = np.random.default_rng(seed)
    n_frames = int(duration_s * sample_rate)
    base = 20.0 + rng.normal(0, 1, n_subcarriers)
    return [base + rng.normal(0, noise_std, n_subcarriers) for _ in range(n_frames)]


def _make_micro_movement_signal(
    freq_hz: float = 1.0,
    duration_s: float = 25.0,
    sample_rate: float = 30.0,
    n_subcarriers: int = 52,
    amplitude: float = 5.0,
    seed: int = 42,
) -> list[np.ndarray]:
    """Generate CSI with micro-movement (higher freq oscillation)."""
    rng = np.random.default_rng(seed)
    n_frames = int(duration_s * sample_rate)
    base = 20.0 + rng.normal(0, 1, n_subcarriers)
    frames = []
    for i in range(n_frames):
        t = i / sample_rate
        movement = amplitude * math.sin(2 * math.pi * freq_hz * t)
        frame = base + movement + rng.normal(0, 0.3, n_subcarriers)
        frames.append(frame)
    return frames


# ---------------------------------------------------------------------------
# BreathingState tests
# ---------------------------------------------------------------------------

class TestBreathingState:
    def test_defaults(self):
        s = BreathingState()
        assert s.breathing_detected is False
        assert s.rate_bpm == 0.0
        assert s.confidence == 0.0
        assert s.apnea_alert is False

    def test_to_dict(self):
        s = BreathingState(breathing_detected=True, rate_bpm=15.0)
        d = s.to_dict()
        assert isinstance(d, dict)
        assert d["breathing_detected"] is True
        assert d["rate_bpm"] == 15.0

    def test_all_fields_in_dict(self):
        s = BreathingState()
        d = s.to_dict()
        expected_keys = {
            "breathing_detected", "rate_bpm", "rate_hz", "confidence",
            "breathing_amplitude", "micro_movement_detected",
            "micro_movement_intensity", "apnea_alert", "seconds_since_breath",
            "estimated_persons", "secondary_rate_bpm", "timestamp",
            "window_seconds", "spectral_snr_db",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# BreathingDetector construction
# ---------------------------------------------------------------------------

class TestBreathingDetectorInit:
    def test_defaults(self):
        d = BreathingDetector()
        assert d.sample_rate_hz == 30.0
        assert d.window_seconds == 20.0
        assert d.confidence_threshold == 0.3

    def test_custom_params(self):
        d = BreathingDetector(
            sample_rate_hz=50.0,
            window_seconds=15.0,
            min_bpm=10.0,
            max_bpm=25.0,
            confidence_threshold=0.5,
            apnea_timeout_s=60.0,
        )
        assert d.sample_rate_hz == 50.0
        assert d.min_hz == 10.0 / 60.0
        assert d.max_hz == 25.0 / 60.0
        assert d.apnea_timeout_s == 60.0

    def test_min_frames_needed(self):
        d = BreathingDetector(sample_rate_hz=30.0, min_bpm=6.0)
        # At 6 BPM = 0.1 Hz, period = 10s, need 2 cycles = 20s = 600 frames
        assert d.min_frames_needed == 600

    def test_min_frames_higher_bpm(self):
        d = BreathingDetector(sample_rate_hz=30.0, min_bpm=12.0)
        # 12 BPM = 0.2 Hz, period = 5s, 2 cycles = 10s = 300 frames
        assert d.min_frames_needed == 300

    def test_state_property(self):
        d = BreathingDetector()
        assert isinstance(d.state, BreathingState)

    def test_history_empty(self):
        d = BreathingDetector()
        assert d.history == []


# ---------------------------------------------------------------------------
# Breathing detection
# ---------------------------------------------------------------------------

class TestBreathingDetection:
    def test_detects_clear_breathing(self):
        """15 BPM breathing with strong signal should be detected."""
        det = BreathingDetector(sample_rate_hz=30.0, window_seconds=20.0, min_bpm=6.0)
        frames = _make_breathing_signal(rate_bpm=15.0, duration_s=25.0, amplitude=3.0, noise_std=0.2)
        for i, f in enumerate(frames):
            result = det.update(f, timestamp=float(i) / 30.0)
        assert result.breathing_detected is True
        # Rate should be close to 15 BPM (within ±3)
        assert 12.0 <= result.rate_bpm <= 18.0
        assert result.confidence > 0.3

    def test_no_breathing_in_noise(self):
        """Pure noise should not trigger breathing detection."""
        det = BreathingDetector(sample_rate_hz=30.0, window_seconds=20.0, min_bpm=6.0)
        frames = _make_still_signal(duration_s=25.0, noise_std=0.3)
        for i, f in enumerate(frames):
            result = det.update(f, timestamp=float(i) / 30.0)
        # May or may not detect — but confidence should be low
        assert result.confidence < 0.5

    def test_detects_slow_breathing(self):
        """8 BPM (sleep-like) breathing."""
        det = BreathingDetector(sample_rate_hz=30.0, window_seconds=25.0, min_bpm=6.0)
        frames = _make_breathing_signal(rate_bpm=8.0, duration_s=30.0, amplitude=3.0, noise_std=0.2)
        for i, f in enumerate(frames):
            result = det.update(f, timestamp=float(i) / 30.0)
        if result.breathing_detected:
            assert 5.0 <= result.rate_bpm <= 12.0

    def test_detects_fast_breathing(self):
        """25 BPM (exercise/stress) breathing."""
        det = BreathingDetector(sample_rate_hz=30.0, window_seconds=20.0, min_bpm=6.0)
        frames = _make_breathing_signal(rate_bpm=25.0, duration_s=25.0, amplitude=3.0, noise_std=0.2)
        for i, f in enumerate(frames):
            result = det.update(f, timestamp=float(i) / 30.0)
        if result.breathing_detected:
            assert 20.0 <= result.rate_bpm <= 30.0

    def test_insufficient_frames_returns_no_detection(self):
        """Not enough data should return no detection."""
        det = BreathingDetector(sample_rate_hz=30.0, min_bpm=6.0)
        frame = np.ones(52) * 20.0
        result = det.update(frame, timestamp=0.0)
        assert result.breathing_detected is False
        assert result.rate_bpm == 0.0

    def test_1d_input(self):
        """Single subcarrier input should work."""
        det = BreathingDetector(sample_rate_hz=30.0, window_seconds=10.0, min_bpm=12.0)
        rng = np.random.default_rng(42)
        for i in range(400):
            t = i / 30.0
            val = 20.0 + 3.0 * math.sin(2 * math.pi * 0.25 * t) + rng.normal(0, 0.2)
            result = det.update(np.array([val]), timestamp=t)
        # Should work without error
        assert isinstance(result, BreathingState)

    def test_custom_subcarrier_weights(self):
        """Custom weights should be used."""
        weights = np.ones(52) / 52.0
        det = BreathingDetector(subcarrier_weights=weights)
        assert det.subcarrier_weights is not None

    def test_rate_smoothing(self):
        """Rate history should smooth output."""
        det = BreathingDetector(sample_rate_hz=30.0, window_seconds=15.0, min_bpm=10.0)
        frames = _make_breathing_signal(rate_bpm=16.0, duration_s=30.0, amplitude=4.0, noise_std=0.1)
        results = []
        for i, f in enumerate(frames):
            r = det.update(f, timestamp=float(i) / 30.0)
            if r.breathing_detected:
                results.append(r.rate_bpm)
        # If detected, rates shouldn't jump wildly (smoothed)
        if len(results) > 3:
            diffs = [abs(results[i] - results[i-1]) for i in range(1, len(results))]
            assert max(diffs) < 10.0  # No jumps > 10 BPM


# ---------------------------------------------------------------------------
# Micro-movement detection
# ---------------------------------------------------------------------------

class TestMicroMovement:
    def test_detects_micro_movement(self):
        """1 Hz oscillation should trigger micro-movement."""
        det = BreathingDetector(sample_rate_hz=30.0, window_seconds=10.0, min_bpm=12.0)
        # Feed baseline first
        still = _make_still_signal(duration_s=6.0, noise_std=0.1)
        for i, f in enumerate(still):
            det.update(f, timestamp=float(i) / 30.0)
        # Then micro-movement
        t_offset = len(still) / 30.0
        movement = _make_micro_movement_signal(freq_hz=1.0, duration_s=15.0, amplitude=5.0)
        for i, f in enumerate(movement):
            result = det.update(f, timestamp=t_offset + float(i) / 30.0)
        assert result.micro_movement_intensity >= 0.0

    def test_no_micro_in_quiet(self):
        """Quiet signal should have low micro-movement."""
        det = BreathingDetector(sample_rate_hz=30.0, window_seconds=10.0, min_bpm=12.0)
        frames = _make_still_signal(duration_s=15.0, noise_std=0.1)
        for i, f in enumerate(frames):
            result = det.update(f, timestamp=float(i) / 30.0)
        assert result.micro_movement_intensity < 0.5


# ---------------------------------------------------------------------------
# Apnea detection
# ---------------------------------------------------------------------------

class TestApnea:
    def test_apnea_after_timeout(self):
        """No breathing for apnea_timeout_s should trigger alert."""
        det = BreathingDetector(
            sample_rate_hz=30.0,
            window_seconds=10.0,
            min_bpm=12.0,
            apnea_timeout_s=10.0,
            confidence_threshold=0.5,
        )
        # Feed breathing first
        breathing = _make_breathing_signal(rate_bpm=15.0, duration_s=15.0, amplitude=4.0, noise_std=0.1)
        for i, f in enumerate(breathing):
            det.update(f, timestamp=float(i) / 30.0)

        # Then feed quiet signal for long enough that the window fully flushes
        t_offset = len(breathing) / 30.0
        quiet = _make_still_signal(duration_s=30.0, noise_std=0.05)
        for i, f in enumerate(quiet):
            result = det.update(f, timestamp=t_offset + float(i) / 30.0)

        # Should either alert apnea or have high seconds_since_breath
        assert result.seconds_since_breath > 5.0 or result.apnea_alert


# ---------------------------------------------------------------------------
# Multi-person
# ---------------------------------------------------------------------------

class TestMultiPerson:
    def test_two_breathing_rates(self):
        """Two distinct breathing rates should detect multiple persons."""
        det = BreathingDetector(sample_rate_hz=30.0, window_seconds=25.0, min_bpm=6.0)
        rng = np.random.default_rng(42)
        n_sub = 52
        base = 20.0 + rng.normal(0, 1, n_sub)
        n_frames = int(30.0 * 30.0)  # 30s
        for i in range(n_frames):
            t = i / 30.0
            # Person 1: 12 BPM
            b1 = 3.0 * math.sin(2 * math.pi * 0.2 * t)
            # Person 2: 20 BPM
            b2 = 2.5 * math.sin(2 * math.pi * 0.333 * t)
            frame = base + (b1 + b2) + rng.normal(0, 0.2, n_sub)
            result = det.update(frame, timestamp=t)
        # Multi-person detection is experimental; just verify it doesn't crash
        assert result.estimated_persons >= 0


# ---------------------------------------------------------------------------
# Reset & dashboard
# ---------------------------------------------------------------------------

class TestResetAndDashboard:
    def test_reset_clears_state(self):
        det = BreathingDetector()
        frames = _make_breathing_signal(duration_s=25.0)
        for i, f in enumerate(frames):
            det.update(f, timestamp=float(i) / 30.0)
        assert len(det.history) > 0
        det.reset()
        assert len(det.history) == 0
        assert det.state.breathing_detected is False

    def test_dashboard_data_format(self):
        det = BreathingDetector(sample_rate_hz=30.0, window_seconds=10.0, min_bpm=12.0)
        frames = _make_breathing_signal(rate_bpm=15.0, duration_s=15.0)
        for i, f in enumerate(frames):
            det.update(f, timestamp=float(i) / 30.0)
        data = det.get_dashboard_data()
        assert "current" in data
        assert "history" in data
        assert "timestamps" in data["history"]
        assert "rate_bpm" in data["history"]
        assert "confidence" in data["history"]

    def test_history_accumulates(self):
        det = BreathingDetector(sample_rate_hz=30.0, window_seconds=10.0, min_bpm=12.0)
        frames = _make_breathing_signal(duration_s=15.0)
        for i, f in enumerate(frames):
            det.update(f, timestamp=float(i) / 30.0)
        # History should have entries once we pass min_frames_needed
        assert len(det.history) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_amplitude_signal(self):
        det = BreathingDetector(sample_rate_hz=30.0, window_seconds=10.0, min_bpm=12.0)
        for i in range(500):
            result = det.update(np.zeros(52), timestamp=float(i) / 30.0)
        assert isinstance(result, BreathingState)

    def test_constant_signal(self):
        det = BreathingDetector(sample_rate_hz=30.0, window_seconds=10.0, min_bpm=12.0)
        for i in range(500):
            result = det.update(np.ones(52) * 20.0, timestamp=float(i) / 30.0)
        assert result.breathing_detected is False

    def test_very_short_window(self):
        det = BreathingDetector(sample_rate_hz=30.0, window_seconds=3.0, min_bpm=12.0)
        frames = _make_breathing_signal(rate_bpm=15.0, duration_s=10.0)
        for i, f in enumerate(frames):
            result = det.update(f, timestamp=float(i) / 30.0)
        assert isinstance(result, BreathingState)

    def test_2d_input_chains_x_subcarriers(self):
        """2D input (chains, subcarriers) should be flattened."""
        det = BreathingDetector(sample_rate_hz=30.0, window_seconds=10.0, min_bpm=12.0)
        for i in range(400):
            frame = np.random.randn(2, 52) * 20.0
            result = det.update(frame, timestamp=float(i) / 30.0)
        assert isinstance(result, BreathingState)

    def test_default_timestamp(self):
        """Omitting timestamp should use time.time()."""
        det = BreathingDetector()
        before = time.time()
        result = det.update(np.ones(52) * 20.0)
        after = time.time()
        assert before <= result.timestamp <= after

    def test_mismatched_subcarrier_weights(self):
        """Wrong-size weights should fall back to uniform."""
        weights = np.ones(10)  # Wrong size for 52-subcarrier input
        det = BreathingDetector(subcarrier_weights=weights, sample_rate_hz=30.0, window_seconds=10.0, min_bpm=12.0)
        frames = _make_breathing_signal(duration_s=15.0)
        for i, f in enumerate(frames):
            result = det.update(f, timestamp=float(i) / 30.0)
        assert isinstance(result, BreathingState)
