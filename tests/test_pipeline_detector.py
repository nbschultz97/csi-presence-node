"""Tests for csi_node.pipeline — PresenceDetector and compute_window."""
from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from csi_node.pipeline import PresenceDetector, compute_window


# ── PresenceDetector ───────────────────────────────────────────────

class TestPresenceDetector:
    def test_initial_state_absent(self):
        det = PresenceDetector(var_threshold=5.0)
        result = det.update(0.1, 0.01)
        assert result["present"] is False
        assert result["confidence"] < 0.5

    def test_high_variance_triggers_presence(self):
        det = PresenceDetector(var_threshold=5.0, ema_alpha=1.0)
        result = det.update(10.0, 0.0)
        assert result["present"] is True
        assert result["confidence"] == 1.0

    def test_high_pca_triggers_presence(self):
        det = PresenceDetector(pca_threshold=1.0, ema_alpha=1.0)
        result = det.update(0.0, 5.0)
        assert result["present"] is True

    def test_ema_smoothing(self):
        det = PresenceDetector(var_threshold=5.0, ema_alpha=0.3)
        # Feed one high reading then lows — confidence should decay
        det.update(100.0, 0.0)
        c1 = det.update(0.0, 0.0)["confidence"]
        c2 = det.update(0.0, 0.0)["confidence"]
        assert c2 < c1

    def test_var_ratio(self):
        det = PresenceDetector(var_threshold=4.0)
        result = det.update(8.0, 0.0)
        assert result["var_ratio"] == pytest.approx(2.0)

    def test_set_baseline(self):
        det = PresenceDetector(var_threshold=100.0)
        det.set_baseline(2.0)
        # Threshold should be max(2*3, 1) = 6
        assert det.var_threshold == pytest.approx(6.0)

    def test_set_baseline_minimum(self):
        det = PresenceDetector()
        det.set_baseline(0.0)
        assert det.var_threshold >= 1.0


# ── compute_window ─────────────────────────────────────────────────

def _make_pkt(ts, csi_shape=(2, 64), rssi=(-40, -42)):
    """Create a synthetic CSI packet."""
    rng = np.random.default_rng(int(ts * 1000) % 2**31)
    return {
        "ts": ts,
        "csi": rng.standard_normal(csi_shape),
        "rssi": list(rssi),
    }


@pytest.fixture
def default_cfg():
    return {
        "variance_threshold": 5.0,
        "pca_threshold": 1.0,
        "rssi_delta": 3.0,
        "tx_power_dbm": -40.0,
        "path_loss_exponent": 2.0,
        "movement_threshold": 2.0,
        "log_dropped": False,
        "min_conditioning_samples": 999,  # disable conditioning for test speed
        "sample_rate_hz": 30.0,
    }


class TestComputeWindow:
    def test_empty_buffer_returns_none(self, default_cfg):
        buf = deque()
        assert compute_window(buf, 0, 3, None, default_cfg) is None

    def test_no_packets_in_range(self, default_cfg):
        buf = deque([_make_pkt(10.0)])
        assert compute_window(buf, 0, 3, None, default_cfg) is None

    def test_basic_output_keys(self, default_cfg):
        buf = deque([_make_pkt(t * 0.1) for t in range(30)])
        result = compute_window(buf, 0, 3, None, default_cfg)
        assert result is not None
        for key in ("presence", "direction", "movement", "var", "pca1", "distance", "ts"):
            assert key in result

    def test_baseline_subtracted(self, default_cfg):
        pkts = [_make_pkt(t * 0.1) for t in range(30)]
        buf1 = deque(pkts)
        buf2 = deque(pkts)
        baseline = np.zeros((2, 64))
        r1 = compute_window(buf1, 0, 3, None, default_cfg)
        r2 = compute_window(buf2, 0, 3, baseline, default_cfg)
        # With zero baseline both should be the same
        assert r1["var"] == pytest.approx(r2["var"], rel=1e-6)

    def test_movement_detection(self, default_cfg):
        buf = deque([_make_pkt(t * 0.1) for t in range(30)])
        result = compute_window(buf, 0, 3, None, default_cfg, prev_var=0.0)
        # Variance should be non-trivial so movement should be "moving" vs near-zero prev
        assert result["movement"] in ("stationary", "moving")

    def test_direction_from_rssi(self, default_cfg):
        # All packets with strong left RSSI
        pkts = [_make_pkt(t * 0.1, rssi=(-30, -50)) for t in range(30)]
        buf = deque(pkts)
        default_cfg["rssi_delta"] = 3.0
        result = compute_window(buf, 0, 3, None, default_cfg)
        assert result["direction"] == "L"

    def test_dropped_packets_with_empty_csi(self, default_cfg):
        pkts = [_make_pkt(t * 0.1) for t in range(20)]
        # Add a packet with empty CSI
        pkts.append({"ts": 1.5, "csi": np.array([]), "rssi": [-40, -42]})
        buf = deque(pkts)
        result = compute_window(buf, 0, 3, None, default_cfg)
        assert result is not None
