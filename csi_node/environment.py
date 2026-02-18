"""Environment profile management for multi-site calibration.

Stores and loads calibration profiles per-environment, so you can quickly
switch between demo sites without recalibrating each time.

Profiles are saved as JSON in data/environments/<name>.json.

Usage:
    from csi_node.environment import EnvironmentManager

    mgr = EnvironmentManager()
    mgr.save("hill_afb_demo", detector)
    mgr.load("hill_afb_demo", detector)
    mgr.list_profiles()  # ["hill_afb_demo", "office_test", ...]
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "environments"


class EnvironmentManager:
    """Manage calibration profiles for different demo environments."""

    def __init__(self, profiles_dir: Optional[str | Path] = None):
        self.profiles_dir = Path(profiles_dir) if profiles_dir else DEFAULT_DIR
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def list_profiles(self) -> list[dict]:
        """List all saved environment profiles with metadata."""
        profiles = []
        for f in sorted(self.profiles_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                profiles.append({
                    "name": f.stem,
                    "description": data.get("description", ""),
                    "wall_type": data.get("wall_type", "unknown"),
                    "calibrated_at": data.get("calibrated_at", ""),
                    "baseline_energy": data.get("baseline_energy", 0),
                })
            except Exception:
                profiles.append({"name": f.stem, "description": "corrupt"})
        return profiles

    def save(
        self,
        name: str,
        detector,
        description: str = "",
        wall_type: str = "unknown",
        notes: str = "",
    ) -> Path:
        """Save current detector calibration as an environment profile.

        Args:
            name: Profile name (used as filename)
            detector: AdaptivePresenceDetector with calibration data
            description: Human-readable description of the environment
            wall_type: Wall material (drywall, concrete, wood, etc.)
            notes: Additional notes

        Returns:
            Path to the saved profile file
        """
        profile = {
            "name": name,
            "description": description,
            "wall_type": wall_type,
            "notes": notes,
            "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "baseline_energy": detector._baseline_energy,
            "baseline_variance": detector._baseline_variance,
            "baseline_spectral": detector._baseline_spectral,
            "calibrated": detector._calibrated,
            "settings": {
                "energy_threshold_factor": detector.energy_threshold_factor,
                "variance_threshold_factor": detector.variance_threshold_factor,
                "spectral_threshold_factor": detector.spectral_threshold_factor,
                "presence_threshold": detector.presence_threshold,
                "ema_alpha": detector.ema_alpha,
                "breathing_band": list(detector.breathing_band),
                "motion_band": list(detector.motion_band),
            },
        }
        path = self.profiles_dir / f"{name}.json"
        path.write_text(json.dumps(profile, indent=2))
        return path

    def load(self, name: str, detector) -> bool:
        """Load an environment profile into a detector.

        Args:
            name: Profile name to load
            detector: AdaptivePresenceDetector to configure

        Returns:
            True if loaded successfully
        """
        path = self.profiles_dir / f"{name}.json"
        if not path.exists():
            return False

        try:
            data = json.loads(path.read_text())

            # Restore calibration baselines
            detector._baseline_energy = data["baseline_energy"]
            detector._baseline_variance = data["baseline_variance"]
            detector._baseline_spectral = data["baseline_spectral"]
            detector._calibrated = data.get("calibrated", True)

            # Restore detection settings
            settings = data.get("settings", {})
            for key, value in settings.items():
                if key == "breathing_band":
                    detector.breathing_band = tuple(value)
                elif key == "motion_band":
                    detector.motion_band = tuple(value)
                elif hasattr(detector, key):
                    setattr(detector, key, value)

            return True
        except Exception:
            return False

    def delete(self, name: str) -> bool:
        """Delete an environment profile."""
        path = self.profiles_dir / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def get(self, name: str) -> Optional[dict]:
        """Get profile data without loading into a detector."""
        path = self.profiles_dir / f"{name}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
