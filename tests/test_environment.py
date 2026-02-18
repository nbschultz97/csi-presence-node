"""Tests for environment profile management."""
import json
import tempfile
from pathlib import Path

from csi_node.environment import EnvironmentManager
from csi_node.presence import AdaptivePresenceDetector


class TestEnvironmentManager:
    def test_save_and_load(self, tmp_path):
        mgr = EnvironmentManager(tmp_path)
        detector = AdaptivePresenceDetector()
        # Fake calibration
        detector._baseline_energy = 42.0
        detector._baseline_variance = 3.14
        detector._baseline_spectral = 1.5
        detector._calibrated = True

        path = mgr.save("test_env", detector, description="Test", wall_type="drywall")
        assert path.exists()

        # Load into a fresh detector
        new_detector = AdaptivePresenceDetector()
        assert mgr.load("test_env", new_detector)
        assert new_detector._baseline_energy == 42.0
        assert new_detector._baseline_variance == 3.14
        assert new_detector._calibrated is True

    def test_list_profiles(self, tmp_path):
        mgr = EnvironmentManager(tmp_path)
        detector = AdaptivePresenceDetector()
        detector._baseline_energy = 10.0
        detector._baseline_variance = 1.0
        detector._baseline_spectral = 0.5
        detector._calibrated = True

        mgr.save("env_a", detector, description="Room A")
        mgr.save("env_b", detector, description="Room B", wall_type="concrete")

        profiles = mgr.list_profiles()
        assert len(profiles) == 2
        names = [p["name"] for p in profiles]
        assert "env_a" in names
        assert "env_b" in names

    def test_delete(self, tmp_path):
        mgr = EnvironmentManager(tmp_path)
        detector = AdaptivePresenceDetector()
        detector._baseline_energy = 1.0
        detector._baseline_variance = 1.0
        detector._baseline_spectral = 1.0
        detector._calibrated = True

        mgr.save("to_delete", detector)
        assert mgr.delete("to_delete")
        assert not mgr.delete("to_delete")  # Already gone

    def test_load_nonexistent(self, tmp_path):
        mgr = EnvironmentManager(tmp_path)
        detector = AdaptivePresenceDetector()
        assert not mgr.load("nope", detector)

    def test_get(self, tmp_path):
        mgr = EnvironmentManager(tmp_path)
        detector = AdaptivePresenceDetector()
        detector._baseline_energy = 5.0
        detector._baseline_variance = 2.0
        detector._baseline_spectral = 1.0
        detector._calibrated = True
        mgr.save("my_env", detector, wall_type="wood")

        data = mgr.get("my_env")
        assert data is not None
        assert data["wall_type"] == "wood"
        assert data["baseline_energy"] == 5.0

        assert mgr.get("nonexistent") is None

    def test_settings_preserved(self, tmp_path):
        mgr = EnvironmentManager(tmp_path)
        detector = AdaptivePresenceDetector(
            energy_threshold_factor=1.8,
            variance_threshold_factor=2.0,
            presence_threshold=0.35,
        )
        detector._baseline_energy = 1.0
        detector._baseline_variance = 1.0
        detector._baseline_spectral = 1.0
        detector._calibrated = True

        mgr.save("custom", detector)

        new = AdaptivePresenceDetector()
        mgr.load("custom", new)
        assert new.energy_threshold_factor == 1.8
        assert new.variance_threshold_factor == 2.0
        assert new.presence_threshold == 0.35
