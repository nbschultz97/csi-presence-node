"""Tests for the advanced presence detection engine."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

from csi_node.presence import AdaptivePresenceDetector, PresenceState


class TestPresenceState:
    def test_to_dict(self):
        s = PresenceState(present=True, confidence=0.8)
        d = s.to_dict()
        assert d["present"] is True
        assert d["confidence"] == 0.8

    def test_to_json(self):
        s = PresenceState(present=False)
        j = s.to_json()
        parsed = json.loads(j)
        assert parsed["present"] is False


class TestAdaptivePresenceDetector:
    def test_init_defaults(self):
        d = AdaptivePresenceDetector()
        assert not d.calibrated
        assert d.state.present is False

    def test_update_returns_state(self):
        d = AdaptivePresenceDetector()
        amps = np.random.randn(64)
        state = d.update(amps)
        assert isinstance(state, PresenceState)
        assert state.timestamp > 0

    def test_needs_min_frames(self):
        d = AdaptivePresenceDetector(sample_rate_hz=10)
        # Single frame shouldn't trigger detection
        state = d.update(np.zeros(64))
        assert state.present is False
        assert state.confidence == 0.0

    def test_calibration_flow(self):
        d = AdaptivePresenceDetector(sample_rate_hz=10)
        d.calibrate_start()

        # Feed quiet samples
        for _ in range(50):
            d.update(np.random.randn(64) * 0.1)

        success = d.calibrate_finish()
        assert success
        assert d.calibrated
        assert d._baseline_energy > 0

    def test_calibration_needs_enough_samples(self):
        d = AdaptivePresenceDetector()
        d.calibrate_start()
        # Only 5 samples
        for _ in range(5):
            d.update(np.random.randn(64))
        assert not d.calibrate_finish()

    def test_auto_calibrate(self):
        d = AdaptivePresenceDetector(sample_rate_hz=10)
        d.auto_calibrate(20)
        for _ in range(25):
            d.update(np.random.randn(64) * 0.1)
        assert d.calibrated

    def test_detects_presence_after_calibration(self):
        d = AdaptivePresenceDetector(
            sample_rate_hz=10,
            energy_threshold_factor=2.0,
            variance_threshold_factor=2.0,
            ema_alpha=0.8,
            presence_threshold=0.3,
        )

        # Calibrate on quiet signal
        d.calibrate_start()
        for _ in range(50):
            d.update(np.random.randn(64) * 0.1)
        d.calibrate_finish()

        # Feed loud signal (simulating human presence)
        for _ in range(30):
            state = d.update(np.random.randn(64) * 10.0)

        # Should detect presence (loud signal >> baseline)
        assert state.confidence > 0.3

    def test_save_load_calibration(self):
        d1 = AdaptivePresenceDetector()
        d1.calibrate_start()
        for _ in range(20):
            d1.update(np.random.randn(64) * 0.5)
        d1.calibrate_finish()

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        d1.save_calibration(path)

        d2 = AdaptivePresenceDetector()
        assert not d2.calibrated
        assert d2.load_calibration(path)
        assert d2.calibrated
        assert d2._baseline_energy == d1._baseline_energy

        Path(path).unlink()

    def test_history_accumulates(self):
        d = AdaptivePresenceDetector(sample_rate_hz=10)
        for _ in range(20):
            d.update(np.random.randn(64))
        assert len(d.history) > 0

    def test_dashboard_data_format(self):
        d = AdaptivePresenceDetector(sample_rate_hz=10)
        for _ in range(20):
            d.update(np.random.randn(64))
        data = d.get_dashboard_data()
        assert "current" in data
        assert "history" in data
        assert "calibration" in data
        assert "confidence" in data["history"]

    def test_movement_classification(self):
        d = AdaptivePresenceDetector(sample_rate_hz=10, ema_alpha=0.9, presence_threshold=0.2)
        d.calibrate_start()
        for _ in range(30):
            d.update(np.random.randn(64) * 0.1)
        d.calibrate_finish()

        # Feed high-variance signal
        for _ in range(30):
            state = d.update(np.random.randn(64) * 20.0)

        # Movement should be detected (either moving or stationary, not none when present)
        if state.present:
            assert state.movement != "none"

    def test_2d_input(self):
        """Test with 2D CSI input (chains x subcarriers)."""
        d = AdaptivePresenceDetector(sample_rate_hz=10)
        amps = np.random.randn(2, 64)
        state = d.update(amps)
        assert isinstance(state, PresenceState)

    def test_direction_from_rssi(self):
        d = AdaptivePresenceDetector(sample_rate_hz=10)
        # Feed enough frames first
        for _ in range(10):
            d.update(np.random.randn(64))

        state = d.update(np.random.randn(64), rssi=[-40, -50])
        assert state.direction == "left"

        state = d.update(np.random.randn(64), rssi=[-50, -40])
        assert state.direction == "right"
