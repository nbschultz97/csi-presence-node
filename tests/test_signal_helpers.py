import math

import numpy as np
import pytest

from node.aoa import aoa_deg, calibrate_theta_offset
from node.range import AlphaBetaFilter, distance_from_rss, fit_pathloss


def test_aoa_deg_nominal() -> None:
    freq = 5.18e9
    spacing = 0.06
    theta_deg = 20.0
    wavelength = 299_792_458.0 / freq
    phase_diff = 2.0 * math.pi * spacing * math.sin(math.radians(theta_deg)) / wavelength
    csi1 = np.ones(30, dtype=np.complex128)
    csi2 = np.exp(-1j * phase_diff) * np.ones_like(csi1)
    angle = aoa_deg(csi1, csi2, freq, spacing, theta_offset=0.0)
    assert angle == pytest.approx(theta_deg, abs=0.5)


def test_calibrate_theta_offset_wraps() -> None:
    samples = [359.0, -1.0, 2.0, -3.0]
    offset = calibrate_theta_offset(samples)
    assert offset == pytest.approx(0.0, abs=1.0)


def test_fit_pathloss_and_distance() -> None:
    baseline = (1.0, -40.0)
    points = [(2.0, -47.0), (3.0, -51.5)]
    n, c = fit_pathloss(baseline, points)
    assert n > 0
    assert -60.0 < c < -30.0
    dist = distance_from_rss(-47.0, n, c, baseline[1])
    assert dist == pytest.approx(2.0, rel=0.2)


def test_alpha_beta_filter_smoothing() -> None:
    filt = AlphaBetaFilter(alpha=0.5, beta=0.1, dt=1.0)
    out1 = filt.update(1.0, timestamp=0.0)
    out2 = filt.update(2.0, timestamp=1.0)
    out3 = filt.update(3.0, timestamp=2.0)
    assert out1 == pytest.approx(1.0)
    assert out2 < 2.0
    assert out3 < 3.0 and out3 > out2
