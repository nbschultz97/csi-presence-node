"""Tests for new features: profiles, SSE, recording, heatmap, zone viz."""
import json
import numpy as np
import pytest

from csi_node.presence import AdaptivePresenceDetector, PresenceState
from csi_node.web_dashboard import DASHBOARD_HTML, DETECTION_PROFILES


class TestDetectionProfiles:
    """Test detection profile switching."""

    def test_profiles_exist(self):
        assert "default" in DETECTION_PROFILES
        assert "through_wall" in DETECTION_PROFILES
        assert "same_room" in DETECTION_PROFILES
        assert "high_sensitivity" in DETECTION_PROFILES

    def test_through_wall_more_sensitive(self):
        tw = DETECTION_PROFILES["through_wall"]
        df = DETECTION_PROFILES["default"]
        assert tw["energy_threshold_factor"] < df["energy_threshold_factor"]
        assert tw["variance_threshold_factor"] < df["variance_threshold_factor"]
        assert tw["presence_threshold"] < df["presence_threshold"]

    def test_same_room_less_sensitive(self):
        sr = DETECTION_PROFILES["same_room"]
        df = DETECTION_PROFILES["default"]
        assert sr["energy_threshold_factor"] > df["energy_threshold_factor"]
        assert sr["variance_threshold_factor"] > df["variance_threshold_factor"]

    def test_set_profile(self):
        d = AdaptivePresenceDetector()
        d.set_profile(DETECTION_PROFILES["through_wall"])
        assert d.energy_threshold_factor == 1.8
        assert d.variance_threshold_factor == 2.0

    def test_profile_keys_valid(self):
        d = AdaptivePresenceDetector()
        for name, profile in DETECTION_PROFILES.items():
            for key in profile:
                assert hasattr(d, key), f"Profile {name} has invalid key: {key}"


class TestSubcarrierHeatmap:
    """Test per-subcarrier variance computation for heatmap."""

    def test_empty_buffer_returns_empty(self):
        d = AdaptivePresenceDetector()
        assert d.get_subcarrier_variance() == []

    def test_heatmap_with_data(self):
        d = AdaptivePresenceDetector()
        rng = np.random.default_rng(42)
        for _ in range(30):
            d.update(rng.normal(20, 3, 52))
        heatmap = d.get_subcarrier_variance()
        assert len(heatmap) == 52
        assert all(0 <= v <= 1.0 for v in heatmap)
        assert max(heatmap) == pytest.approx(1.0, abs=0.01)

    def test_heatmap_detects_active_subcarriers(self):
        d = AdaptivePresenceDetector()
        rng = np.random.default_rng(42)
        for _ in range(30):
            amps = np.ones(52) * 20.0
            # Make subcarriers 20-30 highly variable
            amps[20:30] += rng.normal(0, 10, 10)
            d.update(amps)
        heatmap = d.get_subcarrier_variance()
        # Active subcarriers should have higher variance
        active_avg = np.mean(heatmap[20:30])
        quiet_avg = np.mean(heatmap[:10])
        assert active_avg > quiet_avg


class TestDashboardEnhancements:
    """Test dashboard HTML has new features."""

    def test_has_sse_connection(self):
        assert "EventSource" in DASHBOARD_HTML
        assert "/api/stream" in DASHBOARD_HTML

    def test_has_heatmap(self):
        assert "heatmap" in DASHBOARD_HTML
        assert "heatmap-bar" in DASHBOARD_HTML

    def test_has_zone_visualization(self):
        assert "zone-viz" in DASHBOARD_HTML
        assert "zone-sensor" in DASHBOARD_HTML
        assert "zone-target" in DASHBOARD_HTML
        assert "WALL" in DASHBOARD_HTML

    def test_has_recording_controls(self):
        assert "toggleRecording" in DASHBOARD_HTML
        assert "/api/record" in DASHBOARD_HTML
        assert "badge-rec" in DASHBOARD_HTML

    def test_has_profile_selector(self):
        assert "profile-select" in DASHBOARD_HTML
        assert "through_wall" in DASHBOARD_HTML
        assert "setProfile" in DASHBOARD_HTML

    def test_has_uptime_counter(self):
        assert "uptime" in DASHBOARD_HTML
        assert "updateUptime" in DASHBOARD_HTML

    def test_has_calibration_progress(self):
        assert "cal-progress" in DASHBOARD_HTML
        assert "cal-fill" in DASHBOARD_HTML

    def test_has_toolbar(self):
        assert "toolbar" in DASHBOARD_HTML
        assert "Calibrate" in DASHBOARD_HTML

    def test_has_spectral_in_chart(self):
        assert "spectral" in DASHBOARD_HTML.lower()
        assert "'Spectral'" in DASHBOARD_HTML


class TestPresenceDetectorEnhancements:
    """Test presence detector new methods."""

    def test_set_profile_preserves_calibration(self):
        d = AdaptivePresenceDetector()
        d._calibrated = True
        d._baseline_energy = 100.0
        d.set_profile({"energy_threshold_factor": 1.5})
        assert d._calibrated is True
        assert d._baseline_energy == 100.0
        assert d.energy_threshold_factor == 1.5

    def test_presence_state_to_dict(self):
        s = PresenceState(present=True, confidence=0.8, method="energy")
        d = s.to_dict()
        assert d["present"] is True
        assert d["confidence"] == 0.8

    def test_presence_state_to_json(self):
        s = PresenceState(present=True, confidence=0.8)
        j = json.loads(s.to_json())
        assert j["present"] is True
