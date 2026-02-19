"""Multi-zone presence detection using subcarrier grouping.

Splits the subcarrier array into spatial zones based on frequency
sensitivity patterns. Different subcarriers respond differently to
objects at different distances/angles, enabling coarse localization.

This is a simplified approach that works for demos — real deployment
would use CSI phase + AoA for proper localization.

Zones:
    - near:  Subcarriers most sensitive to near-wall activity (low-index)
    - mid:   Mid-range subcarriers
    - far:   High-index subcarriers, more sensitive to far-field changes
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from collections import deque
from typing import Optional


@dataclass
class ZoneState:
    """Detection state for a single zone."""
    name: str
    active: bool = False
    energy: float = 0.0
    variance: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MultiZoneState:
    """Combined multi-zone detection state."""
    zones: list[ZoneState]
    primary_zone: str = "none"  # Zone with highest confidence
    total_confidence: float = 0.0
    occupancy_count: int = 0  # Estimated number of occupied zones

    def to_dict(self) -> dict:
        return {
            "zones": [z.to_dict() for z in self.zones],
            "primary_zone": self.primary_zone,
            "total_confidence": self.total_confidence,
            "occupancy_count": self.occupancy_count,
        }


class MultiZoneDetector:
    """Detect presence across multiple spatial zones.

    Splits subcarrier array into N zones and runs independent
    energy/variance detection on each zone.
    """

    def __init__(
        self,
        n_zones: int = 3,
        zone_names: Optional[list[str]] = None,
        energy_threshold: float = 2.0,
        variance_threshold: float = 3.0,
        ema_alpha: float = 0.25,
        window_size: int = 30,
    ):
        self.n_zones = n_zones
        self.zone_names = zone_names or ["near", "mid", "far"][:n_zones]
        self.energy_threshold = energy_threshold
        self.variance_threshold = variance_threshold
        self.ema_alpha = ema_alpha

        # Per-zone state
        self._baseline_energy = [0.0] * n_zones
        self._baseline_variance = [0.0] * n_zones
        self._ema_confidence = [0.0] * n_zones
        self._calibrated = False

        # Frame buffer per zone
        self._buffers: list[deque] = [deque(maxlen=window_size) for _ in range(n_zones)]
        self._cal_samples: list[list] = [[] for _ in range(n_zones)]
        self._calibrating = False

    def _split_subcarriers(self, amps: np.ndarray) -> list[np.ndarray]:
        """Split flattened amplitude array into zone chunks."""
        n = len(amps)
        chunk = n // self.n_zones
        zones = []
        for i in range(self.n_zones):
            start = i * chunk
            end = start + chunk if i < self.n_zones - 1 else n
            zones.append(amps[start:end])
        return zones

    def calibrate_start(self) -> None:
        self._calibrating = True
        self._cal_samples = [[] for _ in range(self.n_zones)]

    def calibrate_finish(self) -> bool:
        self._calibrating = False
        for i in range(self.n_zones):
            if len(self._cal_samples[i]) < 10:
                return False
            data = np.array(self._cal_samples[i])
            self._baseline_energy[i] = max(float(np.mean(np.sum(data ** 2, axis=-1))), 1e-6)
            self._baseline_variance[i] = max(float(np.var(data)), 1e-6)
        self._calibrated = True
        self._cal_samples = [[] for _ in range(self.n_zones)]
        return True

    def update(self, amps: np.ndarray) -> MultiZoneState:
        """Process a CSI frame and return multi-zone state."""
        amps = amps.flatten().astype(np.float64)
        zone_amps = self._split_subcarriers(amps)

        zones: list[ZoneState] = []

        for i, za in enumerate(zone_amps):
            self._buffers[i].append(za)

            if self._calibrating:
                self._cal_samples[i].append(za)

            # Energy
            energy = float(np.sum(za ** 2))
            if self._calibrated:
                energy_ratio = energy / self._baseline_energy[i]
            else:
                energy_ratio = 1.0

            # Variance (need buffer)
            variance = 0.0
            var_ratio = 1.0
            if len(self._buffers[i]) >= 5:
                window = np.array(list(self._buffers[i]))
                variance = float(np.var(window))
                if self._calibrated:
                    var_ratio = variance / self._baseline_variance[i]

            # Vote
            score = 0.0
            if energy_ratio > self.energy_threshold:
                score += 0.6
            if var_ratio > self.variance_threshold:
                score += 0.4

            # EMA smooth
            self._ema_confidence[i] = (
                self.ema_alpha * score +
                (1 - self.ema_alpha) * self._ema_confidence[i]
            )

            active = self._ema_confidence[i] > 0.4

            zones.append(ZoneState(
                name=self.zone_names[i],
                active=active,
                energy=energy_ratio,
                variance=var_ratio,
                confidence=self._ema_confidence[i],
            ))

        # Determine primary zone
        best_idx = max(range(self.n_zones), key=lambda j: zones[j].confidence)
        primary = zones[best_idx].name if zones[best_idx].active else "none"
        total_conf = max(z.confidence for z in zones)
        occupied = sum(1 for z in zones if z.active)

        return MultiZoneState(
            zones=zones,
            primary_zone=primary,
            total_confidence=total_conf,
            occupancy_count=occupied,
        )
