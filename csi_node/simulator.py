"""Synthetic CSI data generator for demos and testing.

Generates realistic WiFi CSI amplitude patterns that mimic:
- Empty room (baseline noise)
- Human presence (energy shift + variance increase)
- Movement (large variance, rapid changes)
- Breathing (periodic low-frequency modulation)
- Walk-in / walk-out transitions

Usage:
    from csi_node.simulator import CSISimulator
    sim = CSISimulator()
    for pkt in sim.stream():
        process(pkt)  # same format as real CSI packets
"""
from __future__ import annotations

import time
import math
import random
from typing import Iterator, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class SimScenario:
    """A segment of simulated activity."""
    name: str
    duration_s: float
    presence: bool
    movement: str  # "none", "walking", "stationary", "breathing"
    energy_factor: float = 1.0  # multiplier on baseline energy
    variance_factor: float = 1.0  # multiplier on baseline variance
    breathing_amplitude: float = 0.0
    walk_speed: float = 0.0  # subcarrier drift rate


# Pre-built demo scenarios
DEMO_SCENARIOS = [
    SimScenario("empty_room", 8.0, False, "none", 1.0, 1.0),
    SimScenario("person_enters", 3.0, True, "walking", 2.8, 8.0, walk_speed=2.0),
    SimScenario("standing_still", 6.0, True, "stationary", 2.2, 3.5, breathing_amplitude=0.15),
    SimScenario("walking_around", 5.0, True, "walking", 3.0, 12.0, walk_speed=1.5),
    SimScenario("standing_still_2", 8.0, True, "stationary", 2.0, 3.0, breathing_amplitude=0.12),
    SimScenario("person_leaves", 3.0, True, "walking", 2.5, 7.0, walk_speed=2.5),
    SimScenario("empty_again", 8.0, False, "none", 1.0, 1.0),
    SimScenario("two_people_enter", 3.0, True, "walking", 3.5, 15.0, walk_speed=3.0),
    SimScenario("two_stationary", 6.0, True, "stationary", 2.8, 4.5, breathing_amplitude=0.2),
    SimScenario("one_walks", 5.0, True, "walking", 3.2, 10.0, walk_speed=1.0),
    SimScenario("all_leave", 3.0, True, "walking", 2.5, 8.0, walk_speed=2.0),
    SimScenario("empty_final", 6.0, False, "none", 1.0, 1.0),
]


# Through-wall demo — attenuated signals, longer pauses for narration
THROUGH_WALL_SCENARIOS = [
    SimScenario("calibrating", 10.0, False, "none", 1.0, 1.0),
    SimScenario("empty_baseline", 12.0, False, "none", 1.0, 1.0),
    SimScenario("approach_wall", 4.0, True, "walking", 1.8, 5.0, walk_speed=1.2),
    SimScenario("near_wall_standing", 10.0, True, "stationary", 1.6, 2.5, breathing_amplitude=0.10),
    SimScenario("slow_movement", 8.0, True, "walking", 1.9, 6.0, walk_speed=0.8),
    SimScenario("breathing_only", 12.0, True, "breathing", 1.4, 1.8, breathing_amplitude=0.18),
    SimScenario("walk_across_room", 6.0, True, "walking", 2.2, 9.0, walk_speed=1.5),
    SimScenario("crouch_stationary", 8.0, True, "stationary", 1.3, 2.0, breathing_amplitude=0.08),
    SimScenario("stand_and_walk", 5.0, True, "walking", 2.0, 7.0, walk_speed=1.8),
    SimScenario("person_exits", 4.0, True, "walking", 1.7, 5.0, walk_speed=2.0),
    SimScenario("room_empty_verify", 15.0, False, "none", 1.0, 1.0),
    SimScenario("second_person_enters", 3.0, True, "walking", 2.0, 6.0, walk_speed=1.5),
    SimScenario("second_standing", 8.0, True, "stationary", 1.5, 2.2, breathing_amplitude=0.12),
    SimScenario("second_exits", 3.0, True, "walking", 1.8, 5.0, walk_speed=2.0),
    SimScenario("final_empty", 10.0, False, "none", 1.0, 1.0),
]


class CSISimulator:
    """Generate synthetic CSI packets that look realistic for demos.

    Produces packets in the same format as utils.parse_csi_line() output:
    {"ts": float, "csi": np.ndarray, "rssi": [float, float]}
    """

    def __init__(
        self,
        n_subcarriers: int = 52,  # Standard 20MHz 802.11n
        sample_rate_hz: float = 30.0,
        base_amplitude: float = 20.0,
        noise_floor: float = 2.0,
        scenarios: Optional[list[SimScenario]] = None,
        through_wall: bool = False,
        seed: Optional[int] = None,
    ):
        self.n_sub = n_subcarriers
        self.sample_rate = sample_rate_hz
        self.base_amp = base_amplitude
        self.noise_floor = noise_floor
        if scenarios:
            self.scenarios = scenarios
        elif through_wall:
            self.scenarios = THROUGH_WALL_SCENARIOS
        else:
            self.scenarios = DEMO_SCENARIOS
        self._rng = np.random.default_rng(seed)

        # Generate a stable "room profile" — each subcarrier has slightly
        # different baseline amplitude (like a real multipath environment)
        self._room_profile = self.base_amp + self._rng.normal(0, 3.0, self.n_sub)
        self._room_profile = np.clip(self._room_profile, 5.0, 50.0)

        # Phase accumulator for walking simulation
        self._walk_phase = 0.0

    def _generate_frame(
        self,
        t: float,
        scenario: SimScenario,
        scenario_t: float,
    ) -> dict:
        """Generate one CSI packet for the given scenario state."""
        amps = self._room_profile.copy()

        if scenario.presence:
            # Energy shift — human body attenuates/reflects multipath
            amps *= scenario.energy_factor

            # Add variance (human-caused fluctuation)
            var_noise = self._rng.normal(0, scenario.variance_factor, self.n_sub)
            amps += var_noise

            # Breathing modulation (0.2-0.4 Hz sinusoidal)
            if scenario.breathing_amplitude > 0:
                breath_freq = 0.3  # ~18 breaths/min
                breath = scenario.breathing_amplitude * self.base_amp * \
                    math.sin(2 * math.pi * breath_freq * t)
                # Breathing affects central subcarriers more
                breath_mask = np.exp(-0.5 * ((np.arange(self.n_sub) - self.n_sub / 2) / (self.n_sub / 4)) ** 2)
                amps += breath * breath_mask

            # Walking — subcarrier pattern shifts over time (Doppler-like)
            if scenario.walk_speed > 0:
                self._walk_phase += scenario.walk_speed * (1.0 / self.sample_rate)
                shift = int(self._walk_phase) % self.n_sub
                walk_pattern = 5.0 * np.sin(2 * np.pi * np.arange(self.n_sub) / self.n_sub * 3 + self._walk_phase)
                amps += walk_pattern

        # Always add sensor noise
        amps += self._rng.normal(0, self.noise_floor, self.n_sub)
        amps = np.clip(amps, 0.1, 100.0)

        # RSSI simulation
        mean_amp = float(np.mean(amps))
        rssi_base = -40 + (mean_amp - self.base_amp) * 0.5
        rssi = [
            rssi_base + self._rng.normal(0, 1.5),
            rssi_base + self._rng.normal(0, 1.5),
        ]
        # Add directional offset when moving
        if scenario.presence and scenario.walk_speed > 0:
            direction_offset = 3.0 * math.sin(self._walk_phase * 0.5)
            rssi[0] += direction_offset
            rssi[1] -= direction_offset

        return {
            "ts": t,
            "csi": amps.astype(np.float64),
            "rssi": rssi,
        }

    def stream(self, loop: bool = True, realtime: bool = True) -> Iterator[dict]:
        """Yield synthetic CSI packets following the scenario sequence.

        Applies smooth crossfade transitions between scenarios so detection
        metrics don't jump abruptly (which looks unrealistic in demos).

        Args:
            loop: If True, loop scenarios forever. If False, play once.
            realtime: If True, sleep between packets. If False, yield instantly.
        """
        t = time.time()
        dt = 1.0 / self.sample_rate
        # Number of frames to crossfade between scenarios
        fade_frames = int(self.sample_rate * 1.0)  # 1-second crossfade

        while True:
            for s_idx, scenario in enumerate(self.scenarios):
                n_frames = int(scenario.duration_s * self.sample_rate)
                # Get the previous scenario for fade-in blending
                prev_scenario = self.scenarios[s_idx - 1] if s_idx > 0 else scenario

                for i in range(n_frames):
                    scenario_t = i * dt
                    # Crossfade: blend previous scenario into current at the start
                    if i < fade_frames and s_idx > 0:
                        alpha = i / fade_frames  # 0→1 over fade period
                        pkt_new = self._generate_frame(t, scenario, scenario_t)
                        pkt_old = self._generate_frame(t, prev_scenario, scenario_t)
                        # Blend CSI amplitudes
                        blended_csi = (1 - alpha) * pkt_old["csi"] + alpha * pkt_new["csi"]
                        blended_rssi = [
                            (1 - alpha) * pkt_old["rssi"][0] + alpha * pkt_new["rssi"][0],
                            (1 - alpha) * pkt_old["rssi"][1] + alpha * pkt_new["rssi"][1],
                        ]
                        pkt = {"ts": t, "csi": blended_csi, "rssi": blended_rssi}
                    else:
                        pkt = self._generate_frame(t, scenario, scenario_t)
                    yield pkt
                    t += dt
                    if realtime:
                        time.sleep(dt)

            if not loop:
                break

    def stream_with_labels(self, loop: bool = False) -> Iterator[tuple[dict, dict]]:
        """Yield (packet, label_dict) pairs for testing/validation."""
        t = time.time()
        dt = 1.0 / self.sample_rate

        while True:
            for scenario in self.scenarios:
                n_frames = int(scenario.duration_s * self.sample_rate)
                for i in range(n_frames):
                    scenario_t = i * dt
                    pkt = self._generate_frame(t, scenario, scenario_t)
                    label = {
                        "scenario": scenario.name,
                        "presence": scenario.presence,
                        "movement": scenario.movement,
                    }
                    yield pkt, label
                    t += dt

            if not loop:
                break


def generate_sample_log(path: str, duration_s: float = 60.0, seed: int = 42) -> None:
    """Generate a sample CSI log file for replay demos."""
    import json
    from pathlib import Path

    sim = CSISimulator(seed=seed)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    t = 1000000.0  # Arbitrary start timestamp
    dt = 1.0 / sim.sample_rate
    total_frames = int(duration_s * sim.sample_rate)

    with open(p, 'w') as f:
        frame_i = 0
        for scenario in sim.scenarios:
            n_frames = int(scenario.duration_s * sim.sample_rate)
            for i in range(n_frames):
                if frame_i >= total_frames:
                    return
                pkt = sim._generate_frame(t, scenario, i * dt)
                # Write as JSONL (same format as parse_csi_line expects)
                entry = {
                    "ts": t,
                    "csi": pkt["csi"].tolist(),
                    "rssi": pkt["rssi"],
                }
                f.write(json.dumps(entry) + "\n")
                t += dt
                frame_i += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate sample CSI data")
    parser.add_argument("--output", default="data/demo_csi.log", help="Output file")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds")
    args = parser.parse_args()
    generate_sample_log(args.output, args.duration)
    print(f"Generated {args.output}")
