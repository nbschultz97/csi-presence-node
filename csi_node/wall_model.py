"""Wall attenuation and multipath propagation models for realistic CSI simulation.

Models frequency-dependent signal loss through common building materials,
Rician/Rayleigh fading channels, and multipath interference patterns.

References:
- ITU-R P.2040-1: Effects of building materials on indoor propagation
- IEEE 802.11bf WiFi Sensing standard (draft)
- Bahl & Padmanabhan (2000): RADAR indoor propagation model
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class WallMaterial:
    """Frequency-dependent wall attenuation model.

    Attenuation (dB) = base_loss_db + freq_factor * (freq_ghz - 2.4)
    Based on ITU-R P.2040-1 measured values.
    """
    name: str
    base_loss_db: float       # Loss at 2.4 GHz (dB)
    freq_factor: float        # Additional dB per GHz above 2.4
    thickness_cm: float       # Typical thickness
    phase_shift_rad: float    # Phase distortion per pass

    def attenuation_db(self, freq_ghz: float = 2.4) -> float:
        return self.base_loss_db + self.freq_factor * (freq_ghz - 2.4)


# Common building materials with measured attenuation values
MATERIALS = {
    "drywall": WallMaterial("Drywall (1/2\")", 2.5, 0.8, 1.27, 0.3),
    "drywall_double": WallMaterial("Drywall (double)", 5.0, 1.5, 2.54, 0.6),
    "concrete_thin": WallMaterial("Concrete (4\")", 12.0, 3.0, 10.16, 1.2),
    "concrete_thick": WallMaterial("Concrete (8\")", 18.0, 5.0, 20.32, 2.0),
    "brick": WallMaterial("Brick (single)", 8.0, 2.5, 10.0, 0.9),
    "wood": WallMaterial("Wood frame", 3.5, 1.0, 10.0, 0.4),
    "glass": WallMaterial("Glass (single pane)", 2.0, 0.5, 0.6, 0.15),
    "metal": WallMaterial("Metal/foil barrier", 25.0, 8.0, 0.3, 3.14),
    "cinder_block": WallMaterial("Cinder block", 10.0, 3.5, 20.0, 1.0),
    "none": WallMaterial("No wall (free space)", 0.0, 0.0, 0.0, 0.0),
}


@dataclass
class MultipathComponent:
    """A single multipath reflection/path."""
    delay_ns: float        # Propagation delay (nanoseconds)
    attenuation_db: float  # Path loss (dB)
    phase_offset: float    # Phase offset (radians)
    doppler_hz: float      # Doppler shift from moving reflector (Hz)


@dataclass
class ChannelModel:
    """Indoor wireless channel model with wall penetration and multipath.

    Produces per-subcarrier amplitude/phase that varies realistically
    based on environment geometry and wall materials.
    """
    walls: list[WallMaterial] = field(default_factory=lambda: [MATERIALS["drywall"]])
    distance_m: float = 5.0
    freq_ghz: float = 2.4
    n_subcarriers: int = 52
    bandwidth_mhz: float = 20.0
    rician_k: float = 3.0      # Rician K-factor (dB); higher = stronger LOS
    n_multipaths: int = 8       # Number of multipath components
    rng: Optional[np.random.Generator] = None

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng()
        self._multipaths = self._generate_multipaths()
        self._subcarrier_freqs = self._compute_subcarrier_freqs()

    def _compute_subcarrier_freqs(self) -> np.ndarray:
        """Compute center frequency of each OFDM subcarrier."""
        center = self.freq_ghz * 1e9
        spacing = (self.bandwidth_mhz * 1e6) / self.n_subcarriers
        offsets = (np.arange(self.n_subcarriers) - self.n_subcarriers / 2) * spacing
        return center + offsets

    def _generate_multipaths(self) -> list[MultipathComponent]:
        """Generate random multipath components for the environment."""
        paths = []
        for i in range(self.n_multipaths):
            # Delay: first paths arrive sooner, later ones bounce more
            delay = self.rng.exponential(scale=30.0) + 5.0 * (i + 1)
            # Attenuation: increases with delay (longer path = more loss)
            atten = 3.0 * math.log2(delay / 5.0 + 1) + self.rng.normal(0, 2.0)
            phase = self.rng.uniform(0, 2 * math.pi)
            doppler = self.rng.normal(0, 0.5)  # Small ambient Doppler
            paths.append(MultipathComponent(delay, max(atten, 0.5), phase, doppler))
        return paths

    @property
    def total_wall_loss_db(self) -> float:
        """Total attenuation through all walls."""
        return sum(w.attenuation_db(self.freq_ghz) for w in self.walls)

    @property
    def total_phase_shift(self) -> float:
        """Total phase distortion from wall penetration."""
        return sum(w.phase_shift_rad for w in self.walls)

    def free_space_loss_db(self) -> float:
        """Free-space path loss (Friis equation)."""
        wavelength = 0.3 / self.freq_ghz  # c / f in meters
        if self.distance_m < 0.1:
            return 0.0
        return 20 * math.log10(4 * math.pi * self.distance_m / wavelength)

    def channel_response(self, t: float = 0.0) -> np.ndarray:
        """Compute complex channel frequency response H(f) at time t.

        Returns per-subcarrier complex gains incorporating:
        - Free space path loss
        - Wall attenuation (per material, frequency-dependent)
        - Rician fading (LOS + scattered components)
        - Multipath frequency-selective fading
        - Time-varying Doppler from multipath components

        Args:
            t: Time in seconds (for Doppler evolution)

        Returns:
            Complex ndarray of shape (n_subcarriers,)
        """
        freqs = self._subcarrier_freqs

        # LOS component (Rician)
        k_linear = 10 ** (self.rician_k / 10)
        los_power = math.sqrt(k_linear / (k_linear + 1))
        scatter_power = math.sqrt(1 / (k_linear + 1))

        # LOS: uniform phase across subcarriers + wall phase shift
        total_loss_db = self.free_space_loss_db() + self.total_wall_loss_db
        # Convert to linear amplitude
        linear_atten = 10 ** (-total_loss_db / 20)

        los_phase = self.total_phase_shift
        h_los = los_power * linear_atten * np.exp(1j * los_phase) * np.ones(self.n_subcarriers)

        # Scattered (NLOS) components from multipath
        h_nlos = np.zeros(self.n_subcarriers, dtype=complex)
        for mp in self._multipaths:
            mp_atten = 10 ** (-mp.attenuation_db / 20)
            # Frequency-selective phase from delay
            phase_per_freq = -2 * math.pi * mp.delay_ns * 1e-9 * freqs
            # Time-varying Doppler
            doppler_phase = 2 * math.pi * mp.doppler_hz * t
            total_phase = phase_per_freq + mp.phase_offset + doppler_phase
            h_nlos += mp_atten * np.exp(1j * total_phase)

        h_nlos *= scatter_power * linear_atten

        # Add Rayleigh scatter (small random component)
        rayleigh = (self.rng.normal(0, 0.05, self.n_subcarriers) +
                    1j * self.rng.normal(0, 0.05, self.n_subcarriers))

        h_total = h_los + h_nlos + rayleigh
        return h_total

    def amplitude_response(self, t: float = 0.0) -> np.ndarray:
        """Get per-subcarrier amplitude |H(f)| at time t.

        Returns float64 ndarray of shape (n_subcarriers,).
        """
        return np.abs(self.channel_response(t))


@dataclass
class HumanBodyModel:
    """Model of human body's effect on WiFi CSI.

    A human body reflects, absorbs, and scatters WiFi signals.
    The effect varies by body position, movement, and distance.
    """
    # Cross-section area affects reflection strength
    cross_section_m2: float = 0.5  # Standing adult ~0.5m²
    # Water content affects absorption at 2.4GHz
    absorption_db: float = 3.0
    # Reflection coefficient (body is a partial reflector)
    reflection_coeff: float = 0.4
    # Breathing parameters
    breath_rate_hz: float = 0.3    # ~18 breaths/min
    breath_depth_cm: float = 1.5   # Chest displacement
    # Heart rate (for micro-motion, very subtle)
    heart_rate_hz: float = 1.2     # ~72 bpm

    def body_doppler(self, velocity_ms: float, freq_ghz: float = 2.4) -> float:
        """Doppler shift from body movement."""
        wavelength = 0.3 / freq_ghz
        return 2 * velocity_ms / wavelength

    def breathing_modulation(self, t: float, n_subcarriers: int = 52) -> np.ndarray:
        """Per-subcarrier amplitude modulation from breathing.

        Breathing causes ~1-2cm chest displacement which creates
        micro-Doppler in the CSI. Effect is strongest on subcarriers
        where the body is in the Fresnel zone.
        """
        # Breathing is quasi-sinusoidal with harmonics
        breath = (math.sin(2 * math.pi * self.breath_rate_hz * t) +
                  0.3 * math.sin(4 * math.pi * self.breath_rate_hz * t) +
                  0.1 * math.sin(6 * math.pi * self.breath_rate_hz * t))

        # Scale by chest displacement -> phase change
        wavelength_cm = 30.0 / 2.4  # ~12.5cm at 2.4GHz
        phase_change = 4 * math.pi * self.breath_depth_cm * breath / wavelength_cm

        # Fresnel zone effect: middle subcarriers affected more
        idx = np.arange(n_subcarriers)
        fresnel_weight = np.exp(-0.5 * ((idx - n_subcarriers / 2) / (n_subcarriers / 3)) ** 2)

        return self.reflection_coeff * phase_change * fresnel_weight

    def walking_doppler_pattern(
        self, t: float, speed_ms: float, n_subcarriers: int = 52, freq_ghz: float = 2.4
    ) -> np.ndarray:
        """Frequency-selective Doppler pattern from walking.

        Walking creates periodic limb motion that produces
        a characteristic micro-Doppler signature across subcarriers.
        """
        # Torso Doppler (constant during movement)
        torso_doppler = self.body_doppler(speed_ms, freq_ghz)

        # Limb swing: sinusoidal, ~2x step frequency (~2Hz walking)
        step_freq = speed_ms * 1.5  # Approximate step frequency
        limb_doppler = torso_doppler * 0.5 * math.sin(2 * math.pi * step_freq * t)

        # Frequency-selective: each subcarrier sees slightly different Doppler
        idx = np.arange(n_subcarriers)
        subcarrier_variation = np.sin(2 * math.pi * idx / n_subcarriers * 2 + torso_doppler * t)

        return (torso_doppler + limb_doppler) * subcarrier_variation * self.reflection_coeff


@dataclass
class InterferenceModel:
    """Model environmental RF interference.

    Includes: neighboring WiFi APs, microwave ovens, Bluetooth,
    and burst interference from other devices.
    """
    # Background interference level (dB above noise floor)
    background_level_db: float = -3.0
    # Probability of burst interference per frame
    burst_probability: float = 0.02
    # Burst duration in frames
    burst_duration_frames: int = 5
    # Burst power (dB above background)
    burst_power_db: float = 10.0
    # Neighboring AP interference (periodic beacon)
    neighbor_ap_interval_ms: float = 102.4  # Beacon interval
    neighbor_ap_power_db: float = 5.0

    def __post_init__(self):
        self._burst_countdown = 0
        self._burst_pattern: Optional[np.ndarray] = None

    def interference(
        self, t: float, n_subcarriers: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate interference pattern for one frame.

        Returns additive interference in amplitude units.
        """
        # Background noise (colored — correlated across adjacent subcarriers)
        bg_linear = 10 ** (self.background_level_db / 20)
        white = rng.normal(0, bg_linear, n_subcarriers)
        # Simple 3-tap moving average for colored noise
        kernel = np.array([0.25, 0.5, 0.25])
        colored = np.convolve(white, kernel, mode='same')

        # Burst interference
        burst = np.zeros(n_subcarriers)
        if self._burst_countdown > 0:
            burst = self._burst_pattern * 10 ** (self.burst_power_db / 20)
            self._burst_countdown -= 1
        elif rng.random() < self.burst_probability:
            # New burst: affects random subset of subcarriers
            affected = rng.choice(n_subcarriers, size=n_subcarriers // 4, replace=False)
            self._burst_pattern = np.zeros(n_subcarriers)
            self._burst_pattern[affected] = rng.uniform(0.5, 1.0, len(affected))
            self._burst_countdown = self.burst_duration_frames
            burst = self._burst_pattern * 10 ** (self.burst_power_db / 20)

        # Neighbor AP beacon
        beacon = np.zeros(n_subcarriers)
        beacon_phase = (t * 1000) % self.neighbor_ap_interval_ms
        if beacon_phase < 1.0:  # ~1ms beacon duration
            beacon_linear = 10 ** (self.neighbor_ap_power_db / 20)
            beacon = beacon_linear * np.ones(n_subcarriers) * rng.uniform(0.8, 1.2)

        return colored + burst + beacon
