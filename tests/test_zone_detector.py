"""Tests for multi-zone presence detection."""
import numpy as np
import pytest
from csi_node.zone_detector import MultiZoneDetector, ZoneState, MultiZoneState


class TestMultiZoneDetector:
    def test_init_defaults(self):
        d = MultiZoneDetector()
        assert d.n_zones == 3
        assert d.zone_names == ["near", "mid", "far"]

    def test_update_returns_state(self):
        d = MultiZoneDetector()
        amps = np.random.rand(52)
        state = d.update(amps)
        assert isinstance(state, MultiZoneState)
        assert len(state.zones) == 3
        assert all(isinstance(z, ZoneState) for z in state.zones)

    def test_calibration_flow(self):
        d = MultiZoneDetector()
        d.calibrate_start()
        assert d._calibrating

        # Feed enough samples
        for _ in range(20):
            d.update(np.random.rand(52) * 10)

        ok = d.calibrate_finish()
        assert ok
        assert d._calibrated

    def test_calibration_too_few_samples(self):
        d = MultiZoneDetector()
        d.calibrate_start()
        for _ in range(5):
            d.update(np.random.rand(52))
        ok = d.calibrate_finish()
        assert not ok

    def test_detects_energy_change(self):
        d = MultiZoneDetector(energy_threshold=2.0, variance_threshold=3.0)
        d.calibrate_start()

        # Calibrate with low-energy signal
        for _ in range(20):
            d.update(np.ones(52) * 5)
        d.calibrate_finish()

        # Feed high-energy signal to "near" zone (first third of subcarriers)
        for _ in range(30):
            amps = np.ones(52) * 5
            amps[:17] *= 5  # Boost near zone
            state = d.update(amps)

        # Near zone should have highest confidence
        assert state.zones[0].confidence > state.zones[2].confidence

    def test_to_dict(self):
        d = MultiZoneDetector()
        state = d.update(np.random.rand(52))
        d_dict = state.to_dict()
        assert "zones" in d_dict
        assert "primary_zone" in d_dict
        assert len(d_dict["zones"]) == 3

    def test_custom_zones(self):
        d = MultiZoneDetector(n_zones=2, zone_names=["close", "far"])
        state = d.update(np.random.rand(52))
        assert len(state.zones) == 2
        assert state.zones[0].name == "close"
        assert state.zones[1].name == "far"
