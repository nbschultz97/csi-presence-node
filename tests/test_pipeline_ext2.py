"""Extended pipeline tests — target uncovered lines in pipeline.py."""
from __future__ import annotations

import json
import os
import sys
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
    _capture_fail,
    _check_log_fresh,
    _ERR_MSG,
    CAPTURE_EXIT_CODE,
    STALE_THRESHOLD,
    run_demo,
)


# ── PresenceDetector ─────────────────────────────────────────────────


class TestPresenceDetector:
    def test_initial_state_no_presence(self):
        det = PresenceDetector()
        r = det.update(0.0, 0.0)
        assert r["present"] is False
        assert r["confidence"] == pytest.approx(0.0)

    def test_high_variance_triggers_presence(self):
        det = PresenceDetector(var_threshold=1.0, ema_alpha=1.0)
        r = det.update(10.0, 0.0)
        assert r["present"] is True
        assert r["confidence"] == pytest.approx(1.0)

    def test_high_pca_triggers_presence(self):
        det = PresenceDetector(pca_threshold=1.0, ema_alpha=1.0)
        r = det.update(0.0, 5.0)
        assert r["present"] is True

    def test_ema_smoothing(self):
        det = PresenceDetector(var_threshold=1.0, ema_alpha=0.5)
        det.update(10.0, 0.0)  # raw=1, ema=0.5
        r = det.update(0.0, 0.0)  # raw=0, ema=0.25
        assert r["present"] is False
        assert 0.0 < r["confidence"] < 1.0

    def test_set_baseline(self):
        det = PresenceDetector()
        det.set_baseline(2.0)
        assert det.var_threshold == pytest.approx(6.0)  # 3x baseline

    def test_set_baseline_minimum(self):
        det = PresenceDetector()
        det.set_baseline(0.1)
        assert det.var_threshold == pytest.approx(1.0)  # min(0.3, 1.0) = 1.0

    def test_var_ratio_zero_threshold(self):
        det = PresenceDetector(var_threshold=0.0)
        r = det.update(5.0, 0.0)
        assert r["var_ratio"] == pytest.approx(0.0)


# ── CSILogHandler ────────────────────────────────────────────────────


class TestCSILogHandler:
    def _make_log(self, tmp_path, lines=None):
        log = tmp_path / "csi.log"
        if lines:
            log.write_text("\n".join(lines) + "\n")
        else:
            log.write_text("")
        return log

    def _make_pkt_line(self, ts=1.0):
        csi = np.random.randn(2, 4).tolist()
        return json.dumps({"ts": ts, "rssi": [-40, -42], "csi": csi})

    def test_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CSILogHandler(tmp_path / "nope.log", deque(), lambda: None)

    def test_reads_new_lines(self, tmp_path):
        log = self._make_log(tmp_path)
        buf = deque()
        processed = []
        handler = CSILogHandler(log, buf, lambda: processed.append(1))
        # Append line to file
        with open(log, "a") as f:
            f.write(self._make_pkt_line(1.0) + "\n")
        # Simulate watchdog event
        event = MagicMock()
        event.src_path = str(log)
        handler.on_modified(event)
        assert len(buf) == 1
        assert len(processed) == 1

    def test_ignores_wrong_path(self, tmp_path):
        log = self._make_log(tmp_path)
        buf = deque()
        handler = CSILogHandler(log, buf, lambda: None)
        event = MagicMock()
        event.src_path = str(tmp_path / "other.log")
        handler.on_modified(event)
        assert len(buf) == 0

    def test_skips_invalid_lines(self, tmp_path):
        log = self._make_log(tmp_path)
        buf = deque()
        handler = CSILogHandler(log, buf, lambda: None)
        with open(log, "a") as f:
            f.write("not json\n")
        event = MagicMock()
        event.src_path = str(log)
        handler.on_modified(event)
        assert len(buf) == 0

    def test_pkt_callback(self, tmp_path):
        log = self._make_log(tmp_path)
        buf = deque()
        pkt_calls = []
        handler = CSILogHandler(log, buf, lambda: None, pkt_cb=lambda: pkt_calls.append(1))
        with open(log, "a") as f:
            f.write(self._make_pkt_line() + "\n")
        event = MagicMock()
        event.src_path = str(log)
        handler.on_modified(event)
        assert len(pkt_calls) == 1

    def test_pkt_callback_exception_swallowed(self, tmp_path):
        log = self._make_log(tmp_path)
        buf = deque()

        def bad_cb():
            raise RuntimeError("oops")

        handler = CSILogHandler(log, buf, lambda: None, pkt_cb=bad_cb)
        with open(log, "a") as f:
            f.write(self._make_pkt_line() + "\n")
        event = MagicMock()
        event.src_path = str(log)
        handler.on_modified(event)  # Should not raise
        assert len(buf) == 1

    def test_handles_file_deleted_during_stat(self, tmp_path):
        log = self._make_log(tmp_path)
        buf = deque()
        handler = CSILogHandler(log, buf, lambda: None)
        # Simulate FileNotFoundError during os.stat by patching
        event = MagicMock()
        event.src_path = str(log)
        with patch("os.stat", side_effect=FileNotFoundError):
            handler.on_modified(event)  # Should not raise
        assert len(buf) == 0


# ── compute_window edge cases ────────────────────────────────────────


class TestComputeWindowEdgeCases:
    def _cfg(self):
        return {
            "rssi_delta": 3.0,
            "variance_threshold": 5.0,
            "pca_threshold": 1.0,
            "movement_threshold": 2.0,
            "tx_power_dbm": -40.0,
            "path_loss_exponent": 2.0,
            "min_conditioning_samples": 15,
            "sample_rate_hz": 30.0,
            "log_dropped": False,
        }

    def _make_pkts(self, n=5, ts_start=0.0, shape=(2, 4)):
        pkts = []
        for i in range(n):
            pkts.append({
                "ts": ts_start + i * 0.1,
                "rssi": [-40, -42],
                "csi": np.random.randn(*shape),
            })
        return pkts

    def test_empty_window(self):
        buf = deque(self._make_pkts(5, ts_start=0.0))
        result = compute_window(buf, 100.0, 200.0, None, self._cfg())
        assert result is None

    def test_all_empty_csi(self):
        pkts = [{"ts": i * 0.1, "rssi": [-40, -42], "csi": np.array([])} for i in range(3)]
        buf = deque(pkts)
        result = compute_window(buf, 0.0, 0.3, None, self._cfg())
        assert result is None

    def test_mismatched_shapes_dropped(self):
        cfg = self._cfg()
        cfg["log_dropped"] = True
        pkts = [
            {"ts": 0.0, "rssi": [-40, -42], "csi": np.ones((2, 4))},
            {"ts": 0.1, "rssi": [-40, -42], "csi": np.ones((3, 5))},  # different shape
            {"ts": 0.2, "rssi": [-40, -42], "csi": np.ones((2, 4))},
        ]
        buf = deque(pkts)
        result = compute_window(buf, 0.0, 0.3, None, cfg)
        assert result is not None

    def test_with_baseline_subtraction(self):
        shape = (2, 4)
        pkts = self._make_pkts(5, shape=shape)
        baseline = np.ones(shape)
        buf = deque(pkts)
        result = compute_window(buf, 0.0, 0.5, baseline, self._cfg())
        assert result is not None

    def test_with_prev_var_movement(self):
        pkts = self._make_pkts(5, shape=(2, 4))
        buf = deque(pkts)
        result = compute_window(buf, 0.0, 0.5, None, self._cfg(), prev_var=0.0)
        assert result is not None
        assert result["movement"] in ("stationary", "moving")

    def test_direction_left(self):
        cfg = self._cfg()
        cfg["rssi_delta"] = 1.0
        pkts = [{"ts": i * 0.1, "rssi": [-30, -45], "csi": np.ones((2, 4))} for i in range(3)]
        buf = deque(pkts)
        result = compute_window(buf, 0.0, 0.3, None, cfg)
        assert result["direction"] == "L"

    def test_direction_right(self):
        cfg = self._cfg()
        cfg["rssi_delta"] = 1.0
        pkts = [{"ts": i * 0.1, "rssi": [-45, -30], "csi": np.ones((2, 4))} for i in range(3)]
        buf = deque(pkts)
        result = compute_window(buf, 0.0, 0.3, None, cfg)
        assert result["direction"] == "R"

    def test_no_rssi(self):
        pkts = [{"ts": i * 0.1, "csi": np.ones((2, 4))} for i in range(3)]
        buf = deque(pkts)
        result = compute_window(buf, 0.0, 0.3, None, self._cfg())
        assert result is not None
        assert result["direction"] == "C"

    def test_conditioning_skipped_short_window(self):
        """Windows shorter than min_conditioning_samples skip signal conditioning."""
        cfg = self._cfg()
        cfg["min_conditioning_samples"] = 100  # Much larger than our window
        pkts = self._make_pkts(3, shape=(2, 4))
        buf = deque(pkts)
        result = compute_window(buf, 0.0, 0.3, None, cfg)
        assert result is not None

    def test_baseline_shape_mismatch_ignored(self):
        """Baseline with wrong shape is silently ignored."""
        pkts = self._make_pkts(3, shape=(2, 4))
        baseline = np.ones((5, 10))  # Wrong shape
        buf = deque(pkts)
        result = compute_window(buf, 0.0, 0.3, baseline, self._cfg())
        assert result is not None


# ── _capture_fail and _check_log_fresh ───────────────────────────────


class TestCaptureChecks:
    def test_capture_fail_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            _capture_fail()
        assert exc_info.value.code == CAPTURE_EXIT_CODE

    def test_check_log_fresh_missing_file(self, tmp_path):
        with pytest.raises(SystemExit):
            _check_log_fresh(tmp_path / "nope.log")

    def test_check_log_fresh_stale(self, tmp_path):
        log = tmp_path / "old.log"
        log.write_text("data")
        # Make it look old
        old_time = time.time() - STALE_THRESHOLD - 10
        os.utime(log, (old_time, old_time))
        with pytest.raises(SystemExit):
            _check_log_fresh(log)

    def test_check_log_fresh_ok(self, tmp_path):
        log = tmp_path / "fresh.log"
        log.write_text("data")
        # Should not raise
        _check_log_fresh(log)


# ── run_demo with source iterator ────────────────────────────────────


class TestRunDemoSource:
    def _make_pkts(self, n=20):
        for i in range(n):
            yield {
                "ts": i * 0.1,
                "rssi": [-40, -42],
                "csi": np.random.randn(2, 64),
            }

    def test_run_demo_with_source(self, tmp_path):
        """run_demo with a source iterator processes packets."""
        out = str(tmp_path / "out.jsonl")
        # This should run through all packets and exit
        run_demo(
            source=self._make_pkts(30),
            window=1.0,
            out=out,
            through_wall=False,
        )

    def test_run_demo_through_wall(self, tmp_path):
        out = str(tmp_path / "out.jsonl")
        run_demo(
            source=self._make_pkts(30),
            window=1.0,
            out=out,
            through_wall=True,
        )
