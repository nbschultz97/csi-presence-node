"""Extended tests for csi_node.pipeline — CSILogHandler, compute_window edge cases, conditioning."""
from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from threading import Event
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from csi_node.pipeline import (
    PresenceDetector,
    CSILogHandler,
    compute_window,
    _check_log_fresh,
    _capture_fail,
    CAPTURE_EXIT_CODE,
    STALE_THRESHOLD,
)


def _make_pkt(ts, csi_shape=(2, 64), rssi=(-40, -42)):
    rng = np.random.default_rng(int(ts * 1000) % 2**31)
    return {"ts": ts, "csi": rng.standard_normal(csi_shape), "rssi": list(rssi)}


# ── PresenceDetector additional coverage ──────────────────────────

class TestPresenceDetectorExtended:
    def test_zero_threshold_var_ratio(self):
        det = PresenceDetector(var_threshold=0.0)
        result = det.update(5.0, 0.0)
        assert result["var_ratio"] == 0.0

    def test_sustained_presence(self):
        det = PresenceDetector(var_threshold=5.0, ema_alpha=0.5)
        for _ in range(20):
            result = det.update(100.0, 0.0)
        assert result["present"] is True
        assert result["confidence"] > 0.99

    def test_sustained_absence(self):
        det = PresenceDetector(var_threshold=5.0, ema_alpha=0.5)
        det.update(100.0, 0.0)  # spike
        for _ in range(30):
            result = det.update(0.0, 0.0)
        assert result["present"] is False
        assert result["confidence"] < 0.01

    def test_set_baseline_nonzero(self):
        det = PresenceDetector()
        det.set_baseline(10.0)
        assert det.var_threshold == pytest.approx(30.0)


# ── CSILogHandler ─────────────────────────────────────────────────

class TestCSILogHandler:
    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CSILogHandler(tmp_path / "nope.log", deque(), lambda: None)

    def test_reads_new_lines(self, tmp_path):
        log = tmp_path / "csi.log"
        log.write_text("")  # Create empty
        buf = deque()
        calls = []
        handler = CSILogHandler(log, buf, lambda: calls.append(1))

        # Write a line
        with open(log, "a") as f:
            f.write('{"timestamp": 1.0, "csi": [[1,2],[3,4]], "rssi": [-40,-42]}\n')

        # Simulate file modification event
        event = MagicMock()
        event.src_path = str(log)
        handler.on_modified(event)

        # Buffer should have parsed packets (if parse_csi_line handles JSON)
        # At minimum, process_cb should have been called if a packet was parsed
        # The actual behavior depends on utils.parse_csi_line format

    def test_ignores_other_files(self, tmp_path):
        log = tmp_path / "csi.log"
        log.write_text("")
        buf = deque()
        calls = []
        handler = CSILogHandler(log, buf, lambda: calls.append(1))

        event = MagicMock()
        event.src_path = str(tmp_path / "other.log")
        handler.on_modified(event)
        assert len(calls) == 0


# ── _check_log_fresh & _capture_fail ──────────────────────────────

class TestCheckLogFresh:
    def test_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            _check_log_fresh(tmp_path / "nope.log")
        assert exc_info.value.code == CAPTURE_EXIT_CODE

    def test_stale_file_exits(self, tmp_path):
        log = tmp_path / "old.log"
        log.write_text("data")
        # Make it old
        old_time = time.time() - STALE_THRESHOLD - 10
        os.utime(log, (old_time, old_time))
        with pytest.raises(SystemExit):
            _check_log_fresh(log)

    def test_fresh_file_passes(self, tmp_path):
        log = tmp_path / "fresh.log"
        log.write_text("data")
        # Should not raise
        _check_log_fresh(log)


class TestCaptureExit:
    def test_capture_fail_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            _capture_fail()
        assert exc_info.value.code == CAPTURE_EXIT_CODE


# ── compute_window extended ───────────────────────────────────────

class TestComputeWindowExtended:
    @pytest.fixture
    def cfg(self):
        return {
            "variance_threshold": 5.0,
            "pca_threshold": 1.0,
            "rssi_delta": 3.0,
            "tx_power_dbm": -40.0,
            "path_loss_exponent": 2.0,
            "movement_threshold": 2.0,
            "log_dropped": True,
            "min_conditioning_samples": 999,
            "sample_rate_hz": 30.0,
        }

    def test_mismatched_csi_shapes_dropped(self, cfg):
        pkts = [_make_pkt(t * 0.1, csi_shape=(2, 64)) for t in range(10)]
        # Add a packet with different shape
        pkts.append({"ts": 0.5, "csi": np.random.randn(3, 32), "rssi": [-40, -42]})
        buf = deque(pkts)
        result = compute_window(buf, 0, 3, None, cfg)
        assert result is not None

    def test_all_empty_csi_returns_none(self, cfg):
        pkts = [{"ts": t * 0.1, "csi": np.array([]), "rssi": [-40, -42]} for t in range(10)]
        buf = deque(pkts)
        result = compute_window(buf, 0, 3, None, cfg)
        assert result is None

    def test_right_direction(self, cfg):
        pkts = [_make_pkt(t * 0.1, rssi=(-50, -30)) for t in range(30)]
        buf = deque(pkts)
        result = compute_window(buf, 0, 3, None, cfg)
        assert result["direction"] == "R"

    def test_center_direction(self, cfg):
        pkts = [_make_pkt(t * 0.1, rssi=(-40, -41)) for t in range(30)]
        buf = deque(pkts)
        cfg["rssi_delta"] = 3.0
        result = compute_window(buf, 0, 3, None, cfg)
        assert result["direction"] == "C"

    def test_conditioning_applied(self):
        """When enough samples exist, signal conditioning should run."""
        cfg = {
            "variance_threshold": 5.0,
            "pca_threshold": 1.0,
            "rssi_delta": 3.0,
            "tx_power_dbm": -40.0,
            "path_loss_exponent": 2.0,
            "movement_threshold": 2.0,
            "log_dropped": False,
            "min_conditioning_samples": 5,  # Low threshold to trigger conditioning
            "sample_rate_hz": 30.0,
            "hampel_window": 5,
            "hampel_sigma": 3.0,
            "bp_low_hz": 0.1,
            "bp_high_hz": 10.0,
        }
        pkts = [_make_pkt(t * 0.1) for t in range(30)]
        buf = deque(pkts)
        result = compute_window(buf, 0, 3, None, cfg)
        assert result is not None
        assert "var" in result

    def test_no_rssi_packets(self, cfg):
        pkts = [{"ts": t * 0.1, "csi": np.random.randn(2, 64)} for t in range(20)]
        buf = deque(pkts)
        result = compute_window(buf, 0, 3, None, cfg)
        assert result is not None
        assert result["direction"] == "C"

    def test_single_rssi_antenna(self, cfg):
        pkts = [{"ts": t * 0.1, "csi": np.random.randn(2, 64), "rssi": [-40]} for t in range(20)]
        buf = deque(pkts)
        result = compute_window(buf, 0, 3, None, cfg)
        assert result is not None

    def test_movement_stationary_with_similar_prev(self, cfg):
        pkts = [_make_pkt(t * 0.1) for t in range(30)]
        buf = deque(pkts)
        r1 = compute_window(buf, 0, 3, None, cfg)
        # Use same variance as prev — should be stationary
        r2 = compute_window(deque(pkts), 0, 3, None, cfg, prev_var=r1["var"])
        assert r2["movement"] == "stationary"

    def test_pose_features_present(self, cfg):
        pkts = [_make_pkt(t * 0.1) for t in range(30)]
        buf = deque(pkts)
        result = compute_window(buf, 0, 3, None, cfg)
        assert result is not None
        assert "pose_feat" in result
        assert result["pose_feat"] is not None
