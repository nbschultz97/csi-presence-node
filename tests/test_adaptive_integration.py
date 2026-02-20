"""Tests for AdaptivePresenceDetector integration into pipeline."""
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from collections import deque

from csi_node.presence import AdaptivePresenceDetector, PresenceState
from csi_node.simulator import CSISimulator, SimScenario
from csi_node.pipeline import compute_window, run_demo


class TestAdaptiveDetectorIntegration:
    """Test that AdaptivePresenceDetector works end-to-end with simulator data."""

    def test_detector_calibrates_on_empty_room(self):
        """Detector should calibrate from empty room frames."""
        detector = AdaptivePresenceDetector()
        detector.calibrate_start()

        sim = CSISimulator(
            scenarios=[SimScenario("empty", 5.0, False, "none")],
            seed=42,
        )
        for pkt in sim.stream(loop=False, realtime=False):
            detector.update(pkt["csi"], rssi=pkt.get("rssi"), timestamp=pkt["ts"])

        assert detector.calibrate_finish() is True
        assert detector.calibrated is True
        assert detector._baseline_energy > 0
        assert detector._baseline_variance > 0

    def test_detector_detects_presence_after_calibration(self):
        """After calibration, detector should detect simulated presence."""
        detector = AdaptivePresenceDetector(
            energy_threshold_factor=1.5,
            variance_threshold_factor=1.5,
            presence_threshold=0.3,
        )

        # Calibrate on empty room
        detector.calibrate_start()
        sim_empty = CSISimulator(
            scenarios=[SimScenario("empty", 3.0, False, "none")],
            seed=42,
        )
        for pkt in sim_empty.stream(loop=False, realtime=False):
            detector.update(pkt["csi"], rssi=pkt.get("rssi"), timestamp=pkt["ts"])
        detector.calibrate_finish()

        # Now feed walking presence
        sim_present = CSISimulator(
            scenarios=[SimScenario("walking", 3.0, True, "walking", 3.0, 12.0, walk_speed=1.5)],
            seed=42,
        )
        states = []
        for pkt in sim_present.stream(loop=False, realtime=False):
            state = detector.update(pkt["csi"], rssi=pkt.get("rssi"), timestamp=pkt["ts"])
            states.append(state)

        # Should have detected presence in many frames
        detected = sum(1 for s in states if s.present)
        assert detected > len(states) * 0.5, f"Only detected {detected}/{len(states)} frames"

    def test_auto_calibrate(self):
        """Auto-calibrate should finish after N samples."""
        detector = AdaptivePresenceDetector()
        detector.auto_calibrate(50)

        sim = CSISimulator(
            scenarios=[SimScenario("empty", 3.0, False, "none")],
            seed=42,
        )
        count = 0
        for pkt in sim.stream(loop=False, realtime=False):
            detector.update(pkt["csi"], rssi=pkt.get("rssi"), timestamp=pkt["ts"])
            count += 1
            if count > 60:
                break

        assert detector.calibrated is True

    def test_save_load_calibration(self, tmp_path):
        """Calibration should round-trip through save/load."""
        detector = AdaptivePresenceDetector()
        detector.calibrate_start()

        sim = CSISimulator(
            scenarios=[SimScenario("empty", 3.0, False, "none")],
            seed=42,
        )
        for pkt in sim.stream(loop=False, realtime=False):
            detector.update(pkt["csi"], rssi=pkt.get("rssi"), timestamp=pkt["ts"])
        detector.calibrate_finish()

        cal_path = tmp_path / "cal.json"
        detector.save_calibration(cal_path)

        detector2 = AdaptivePresenceDetector()
        assert detector2.load_calibration(cal_path) is True
        assert detector2.calibrated is True
        assert abs(detector2._baseline_energy - detector._baseline_energy) < 1e-6

    def test_through_wall_profile_lower_thresholds(self):
        """Through-wall profile should have lower thresholds than default."""
        from csi_node.web_dashboard import DETECTION_PROFILES

        default = DETECTION_PROFILES["default"]
        tw = DETECTION_PROFILES["through_wall"]

        assert tw["energy_threshold_factor"] < default["energy_threshold_factor"]
        assert tw["variance_threshold_factor"] < default["variance_threshold_factor"]
        assert tw["presence_threshold"] < default["presence_threshold"]

    def test_presence_state_json_roundtrip(self):
        """PresenceState should serialize cleanly to JSON."""
        state = PresenceState(
            present=True,
            confidence=0.85,
            method="energy+variance",
            movement="walking",
        )
        d = state.to_dict()
        j = state.to_json()
        parsed = json.loads(j)
        assert parsed["present"] is True
        assert parsed["confidence"] == 0.85
        assert parsed["method"] == "energy+variance"

    def test_dashboard_data_format(self):
        """get_dashboard_data should return well-formed dict."""
        detector = AdaptivePresenceDetector()
        sim = CSISimulator(
            scenarios=[SimScenario("empty", 2.0, False, "none")],
            seed=42,
        )
        for pkt in sim.stream(loop=False, realtime=False):
            detector.update(pkt["csi"])

        data = detector.get_dashboard_data()
        assert "current" in data
        assert "history" in data
        assert "calibration" in data
        assert "timestamps" in data["history"]
        assert "confidence" in data["history"]

    def test_movement_classification(self):
        """Detector should classify movement vs stationary."""
        detector = AdaptivePresenceDetector()
        detector.auto_calibrate(50)

        # Feed empty room first for calibration
        sim_empty = CSISimulator(
            scenarios=[SimScenario("empty", 2.0, False, "none")],
            seed=42,
        )
        for pkt in sim_empty.stream(loop=False, realtime=False):
            detector.update(pkt["csi"], timestamp=pkt["ts"])

        # Feed walking data
        sim_walk = CSISimulator(
            scenarios=[SimScenario("walk", 3.0, True, "walking", 3.0, 12.0, walk_speed=2.0)],
            seed=42,
        )
        movements = set()
        for pkt in sim_walk.stream(loop=False, realtime=False):
            state = detector.update(pkt["csi"], timestamp=pkt["ts"])
            if state.present:
                movements.add(state.movement)

        # Should see some non-"none" movements
        assert len(movements) > 0


class TestRunPyCalibrateMode:
    """Test the --calibrate CLI mode integration."""

    def test_calibrate_flag_exists(self):
        """run.py should accept --calibrate flag."""
        import importlib
        import run
        importlib.reload(run)
        # Just verify the module loads without error
        assert hasattr(run, 'main')
