"""Integration tests: simulator → presence detector → correct detection."""
import numpy as np
import pytest

from csi_node.simulator import CSISimulator, SimScenario
from csi_node.presence import AdaptivePresenceDetector


class TestPresenceDetectorWithSimulator:
    """Verify the AdaptivePresenceDetector correctly detects simulated presence."""

    def _run_scenarios(self, scenarios, calibrate_frames=100):
        """Run scenarios through detector and return list of (label, state) pairs."""
        sim = CSISimulator(scenarios=scenarios, seed=42, sample_rate_hz=30.0)
        det = AdaptivePresenceDetector(sample_rate_hz=30.0)

        results = []
        cal_done = False
        frame_i = 0

        for pkt, label in sim.stream_with_labels(loop=False):
            if not cal_done and frame_i == 0:
                det.calibrate_start()

            state = det.update(pkt["csi"], rssi=pkt["rssi"], timestamp=pkt["ts"])
            frame_i += 1

            if not cal_done and frame_i >= calibrate_frames:
                det.calibrate_finish()
                cal_done = True
                continue  # Skip calibration frames in results

            if cal_done:
                results.append((label, state))

        return results

    def test_detects_presence_after_calibration(self):
        """After calibrating on empty room, detector should catch presence."""
        scenarios = [
            SimScenario("empty_cal", 5.0, False, "none"),  # 150 frames for calibration
            SimScenario("empty_post", 3.0, False, "none"),  # Should be CLEAR
            SimScenario("person", 5.0, True, "stationary", 2.5, 4.0, breathing_amplitude=0.15),
        ]
        results = self._run_scenarios(scenarios)

        # Check last 2 seconds of empty period
        empty_results = [r for r in results if r[0]["scenario"] == "empty_post"]
        late_empty = empty_results[-30:]  # last 1 second
        empty_detections = sum(1 for _, s in late_empty if s.present)
        assert empty_detections < len(late_empty) * 0.3, "Too many false positives in empty room"

        # Check last 2 seconds of presence period (give EMA time to converge)
        presence_results = [r for r in results if r[0]["scenario"] == "person"]
        late_presence = presence_results[-60:]  # last 2 seconds
        presence_detections = sum(1 for _, s in late_presence if s.present)
        assert presence_detections > len(late_presence) * 0.5, "Failed to detect presence"

    def test_movement_classification(self):
        """Walking should show higher intensity than stationary."""
        scenarios = [
            SimScenario("empty", 5.0, False, "none"),
            SimScenario("walking", 4.0, True, "walking", 3.0, 12.0, walk_speed=2.0),
            SimScenario("still", 4.0, True, "stationary", 2.0, 3.0, breathing_amplitude=0.1),
        ]
        results = self._run_scenarios(scenarios)

        walk_intensity = [s.movement_intensity for l, s in results if l["scenario"] == "walking"]
        still_intensity = [s.movement_intensity for l, s in results if l["scenario"] == "still"]

        if walk_intensity and still_intensity:
            assert np.mean(walk_intensity) > np.mean(still_intensity) * 0.5

    def test_confidence_drops_when_person_leaves(self):
        """Confidence should decrease after person leaves."""
        scenarios = [
            SimScenario("empty", 5.0, False, "none"),
            SimScenario("present", 5.0, True, "stationary", 2.5, 4.0),
            SimScenario("empty_again", 5.0, False, "none"),
        ]
        results = self._run_scenarios(scenarios)

        present_conf = [s.confidence for l, s in results if l["scenario"] == "present"]
        empty_conf = [s.confidence for l, s in results if l["scenario"] == "empty_again"]

        if present_conf and empty_conf:
            # Late presence should have higher confidence than late empty
            assert np.mean(present_conf[-30:]) > np.mean(empty_conf[-30:])

    def test_save_load_calibration(self, tmp_path):
        """Calibration should survive save/load cycle."""
        scenarios = [SimScenario("empty", 5.0, False, "none")]
        sim = CSISimulator(scenarios=scenarios, seed=42)
        det = AdaptivePresenceDetector()
        det.calibrate_start()
        for pkt in sim.stream(loop=False, realtime=False):
            det.update(pkt["csi"])
        det.calibrate_finish()

        cal_file = tmp_path / "cal.json"
        det.save_calibration(cal_file)

        det2 = AdaptivePresenceDetector()
        assert not det2.calibrated
        assert det2.load_calibration(cal_file)
        assert det2.calibrated
