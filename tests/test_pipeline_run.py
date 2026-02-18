"""Tests for pipeline.run_demo, run_offline, run, main — the big uncovered functions."""
from __future__ import annotations

import json
import sys
import time
from collections import deque
from io import StringIO
from pathlib import Path
from threading import Event
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from csi_node.pipeline import (
    compute_window,
    run_demo,
    run_offline,
    main,
    CSILogHandler,
)


def _make_pkt(ts, csi_shape=(2, 64), rssi=(-40, -42)):
    rng = np.random.default_rng(int(ts * 1000) % 2**31)
    return {"ts": ts, "csi": rng.standard_normal(csi_shape), "rssi": list(rssi)}


@pytest.fixture
def minimal_cfg():
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
        "window_size": 1.0,
        "window_hop": 0.5,
        "baseline_file": "nonexistent_baseline.npz",
        "log_file": "nonexistent_log.csv",
        "output_file": "test_output.jsonl",
        "rotation_max_bytes": 1048576,
        "rotation_interval_seconds": 0,
        "rotation_retention": 5,
        "log_wait": 0.01,
        "udp_enabled": False,
        "atak_enabled": False,
        "frames_wait_timeout": 0.5,
    }


class TestRunDemo:
    """Test run_demo with a synthetic source iterator."""

    def test_source_pipeline(self, tmp_path, minimal_cfg):
        """run_demo with a source iterator processes packets and writes JSONL."""
        out_file = tmp_path / "output.jsonl"
        minimal_cfg["output_file"] = str(out_file)
        minimal_cfg["baseline_file"] = str(tmp_path / "no_baseline.npz")

        pkts = [_make_pkt(t * 0.05) for t in range(60)]

        with patch("csi_node.pipeline.yaml.safe_load", return_value=minimal_cfg), \
             patch("csi_node.pipeline.config_validator.validate_config") as mock_val:
            mock_val.return_value = MagicMock(valid=True, errors=[], warnings=[])

            captured = StringIO()
            with patch("sys.stdout", captured):
                run_demo(source=pkts, window=1.0, out=str(out_file))

        output = captured.getvalue()
        lines = [l for l in output.strip().split("\n") if l]
        assert len(lines) > 0
        entry = json.loads(lines[0])
        assert "presence" in entry
        assert "movement" in entry
        assert "direction" in entry

    def test_source_with_pose(self, tmp_path, minimal_cfg):
        """run_demo with pose=True loads the classifier."""
        out_file = tmp_path / "output.jsonl"
        minimal_cfg["output_file"] = str(out_file)
        minimal_cfg["baseline_file"] = str(tmp_path / "no_baseline.npz")

        pkts = [_make_pkt(t * 0.05) for t in range(60)]

        with patch("csi_node.pipeline.yaml.safe_load", return_value=minimal_cfg), \
             patch("csi_node.pipeline.config_validator.validate_config") as mock_val:
            mock_val.return_value = MagicMock(valid=True, errors=[], warnings=[])

            captured = StringIO()
            with patch("sys.stdout", captured):
                run_demo(source=pkts, pose=True, window=1.0, out=str(out_file))

        output = captured.getvalue()
        lines = [l for l in output.strip().split("\n") if l]
        assert len(lines) > 0
        entry = json.loads(lines[0])
        assert "pose" in entry

    def test_invalid_config_prints_warnings(self, tmp_path, minimal_cfg):
        """run_demo prints config errors and warnings to stderr."""
        out_file = tmp_path / "output.jsonl"
        minimal_cfg["output_file"] = str(out_file)
        minimal_cfg["baseline_file"] = str(tmp_path / "no_baseline.npz")
        pkts = [_make_pkt(t * 0.05) for t in range(40)]

        with patch("csi_node.pipeline.yaml.safe_load", return_value=minimal_cfg), \
             patch("csi_node.pipeline.config_validator.validate_config") as mock_val:
            mock_val.return_value = MagicMock(
                valid=False,
                errors=["bad config field"],
                warnings=["consider adjusting threshold"],
            )

            captured_err = StringIO()
            with patch("sys.stderr", captured_err), \
                 patch("sys.stdout", StringIO()):
                run_demo(source=pkts, window=1.0, out=str(out_file))

            err_output = captured_err.getvalue()
            assert "bad config field" in err_output
            assert "consider adjusting threshold" in err_output

    def test_udp_streamer_init(self, tmp_path, minimal_cfg):
        """run_demo initializes UDP streamer when enabled."""
        out_file = tmp_path / "output.jsonl"
        minimal_cfg["output_file"] = str(out_file)
        minimal_cfg["baseline_file"] = str(tmp_path / "no_baseline.npz")
        minimal_cfg["udp_enabled"] = True
        minimal_cfg["udp_host"] = "127.0.0.1"
        minimal_cfg["udp_port"] = 9999

        pkts = [_make_pkt(t * 0.05) for t in range(40)]

        mock_streamer = MagicMock()
        with patch("csi_node.pipeline.yaml.safe_load", return_value=minimal_cfg), \
             patch("csi_node.pipeline.config_validator.validate_config") as mock_val, \
             patch("csi_node.udp_streamer.UDPStreamer.from_config", return_value=mock_streamer):
            mock_val.return_value = MagicMock(valid=True, errors=[], warnings=[])

            with patch("sys.stdout", StringIO()):
                run_demo(source=pkts, window=1.0, out=str(out_file))

        assert mock_streamer.send.called


class TestRunOffline:
    """Test run_offline (batch processing from log file)."""

    def test_processes_log_file(self, tmp_path, minimal_cfg):
        """run_offline reads a CSI log and returns a DataFrame."""
        log_file = tmp_path / "test.csv"
        pkts = [_make_pkt(t * 0.05) for t in range(50)]
        log_file.write_text("\n".join(["line"] * 50))

        minimal_cfg["baseline_file"] = str(tmp_path / "no_baseline.npz")
        minimal_cfg["log_wait"] = 0.01

        with patch("csi_node.pipeline.utils.parse_csi_line") as mock_parse, \
             patch("csi_node.pipeline.utils.wait_for_file", return_value=True):
            call_idx = [0]
            def side_effect(line):
                if call_idx[0] < len(pkts):
                    pkt = pkts[call_idx[0]]
                    call_idx[0] += 1
                    return pkt
                return None
            mock_parse.side_effect = side_effect

            df = run_offline(str(log_file), minimal_cfg)

        assert len(df) > 0
        assert "presence" in df.columns

    def test_missing_log_raises(self, tmp_path, minimal_cfg):
        """run_offline raises FileNotFoundError for missing log."""
        minimal_cfg["baseline_file"] = str(tmp_path / "no_baseline.npz")
        minimal_cfg["log_wait"] = 0.01

        with patch("csi_node.pipeline.utils.wait_for_file", return_value=False):
            with pytest.raises(FileNotFoundError):
                run_offline(str(tmp_path / "missing.log"), minimal_cfg)


class TestComputeWindowEdgeCases:
    """Additional edge cases for compute_window."""

    def test_mismatched_csi_shapes_dropped(self):
        cfg = {
            "variance_threshold": 5.0, "pca_threshold": 1.0,
            "rssi_delta": 3.0, "tx_power_dbm": -40.0,
            "path_loss_exponent": 2.0, "movement_threshold": 2.0,
            "log_dropped": True, "min_conditioning_samples": 999,
            "sample_rate_hz": 30.0,
        }
        pkts = [_make_pkt(t * 0.1, csi_shape=(2, 64)) for t in range(20)]
        for t in range(20, 25):
            pkts.append(_make_pkt(t * 0.1, csi_shape=(2, 32)))
        buf = deque(pkts)
        captured = StringIO()
        with patch("sys.stderr", captured):
            result = compute_window(buf, 0, 3, None, cfg)
        assert result is not None
        assert "Dropped" in captured.getvalue()

    def test_all_csi_empty_returns_none(self):
        cfg = {
            "variance_threshold": 5.0, "pca_threshold": 1.0,
            "rssi_delta": 3.0, "tx_power_dbm": -40.0,
            "path_loss_exponent": 2.0, "movement_threshold": 2.0,
            "log_dropped": True, "min_conditioning_samples": 999,
            "sample_rate_hz": 30.0,
        }
        pkts = [{"ts": t * 0.1, "csi": np.array([]), "rssi": [-40, -42]} for t in range(20)]
        buf = deque(pkts)
        result = compute_window(buf, 0, 3, None, cfg)
        assert result is None

    def test_no_rssi_data(self):
        cfg = {
            "variance_threshold": 5.0, "pca_threshold": 1.0,
            "rssi_delta": 3.0, "tx_power_dbm": -40.0,
            "path_loss_exponent": 2.0, "movement_threshold": 2.0,
            "log_dropped": False, "min_conditioning_samples": 999,
            "sample_rate_hz": 30.0,
        }
        rng = np.random.default_rng(42)
        pkts = [{"ts": t * 0.1, "csi": rng.standard_normal((2, 64)), "rssi": None} for t in range(20)]
        buf = deque(pkts)
        result = compute_window(buf, 0, 3, None, cfg)
        assert result is not None
        assert result["direction"] == "C"

    def test_conditioning_applied(self):
        cfg = {
            "variance_threshold": 5.0, "pca_threshold": 1.0,
            "rssi_delta": 3.0, "tx_power_dbm": -40.0,
            "path_loss_exponent": 2.0, "movement_threshold": 2.0,
            "log_dropped": False, "min_conditioning_samples": 5,
            "sample_rate_hz": 30.0, "hampel_window": 5,
            "hampel_sigma": 3.0, "bp_low_hz": 0.1, "bp_high_hz": 10.0,
        }
        pkts = [_make_pkt(t * 0.1) for t in range(30)]
        buf = deque(pkts)
        result = compute_window(buf, 0, 3, None, cfg)
        assert result is not None

    def test_right_direction(self):
        cfg = {
            "variance_threshold": 5.0, "pca_threshold": 1.0,
            "rssi_delta": 3.0, "tx_power_dbm": -40.0,
            "path_loss_exponent": 2.0, "movement_threshold": 2.0,
            "log_dropped": False, "min_conditioning_samples": 999,
            "sample_rate_hz": 30.0,
        }
        pkts = [_make_pkt(t * 0.1, rssi=(-50, -30)) for t in range(30)]
        buf = deque(pkts)
        result = compute_window(buf, 0, 3, None, cfg)
        assert result["direction"] == "R"

    def test_stationary_movement(self):
        cfg = {
            "variance_threshold": 5.0, "pca_threshold": 1.0,
            "rssi_delta": 3.0, "tx_power_dbm": -40.0,
            "path_loss_exponent": 2.0, "movement_threshold": 2.0,
            "log_dropped": False, "min_conditioning_samples": 999,
            "sample_rate_hz": 30.0,
        }
        pkts = [_make_pkt(t * 0.1) for t in range(30)]
        buf = deque(pkts)
        result = compute_window(buf, 0, 3, None, cfg)
        result2 = compute_window(deque(pkts), 0, 3, None, cfg, prev_var=result["var"])
        assert result2["movement"] == "stationary"

    def test_none_csi_packets_dropped(self):
        cfg = {
            "variance_threshold": 5.0, "pca_threshold": 1.0,
            "rssi_delta": 3.0, "tx_power_dbm": -40.0,
            "path_loss_exponent": 2.0, "movement_threshold": 2.0,
            "log_dropped": False, "min_conditioning_samples": 999,
            "sample_rate_hz": 30.0,
        }
        pkts = [_make_pkt(t * 0.1) for t in range(20)]
        pkts.append({"ts": 1.5, "csi": None, "rssi": [-40, -42]})
        buf = deque(pkts)
        result = compute_window(buf, 0, 3, None, cfg)
        assert result is not None

    def test_baseline_shape_mismatch_ignored(self):
        cfg = {
            "variance_threshold": 5.0, "pca_threshold": 1.0,
            "rssi_delta": 3.0, "tx_power_dbm": -40.0,
            "path_loss_exponent": 2.0, "movement_threshold": 2.0,
            "log_dropped": False, "min_conditioning_samples": 999,
            "sample_rate_hz": 30.0,
        }
        pkts = [_make_pkt(t * 0.1) for t in range(30)]
        buf = deque(pkts)
        # Wrong baseline shape
        baseline = np.zeros((3, 32))
        result = compute_window(buf, 0, 3, baseline, cfg)
        assert result is not None


class TestCSILogHandlerRotation:
    """Test CSILogHandler file rotation handling."""

    def test_handler_creates_from_path(self, tmp_path):
        """CSILogHandler accepts Path objects."""
        log_file = tmp_path / "csi.log"
        log_file.write_text("test data\n")

        buffer = deque()
        handler = CSILogHandler(log_file, buffer, lambda: None)
        assert handler is not None

    def test_handler_file_not_found(self, tmp_path):
        """CSILogHandler handles missing file."""
        log_file = tmp_path / "missing.log"
        buffer = deque()
        with pytest.raises((FileNotFoundError, OSError)):
            CSILogHandler(log_file, buffer, lambda: None)

    def test_handler_ignores_other_files(self, tmp_path):
        """Handler ignores events for other files."""
        log_file = tmp_path / "csi.log"
        log_file.write_text("")

        buffer = deque()
        handler = CSILogHandler(log_file, buffer, lambda: None)

        event = MagicMock()
        event.src_path = str(tmp_path / "other.log")
        handler.on_modified(event)  # Should not raise


class TestMain:
    """Test CLI main() entry point."""

    def test_main_no_args(self):
        with patch("sys.argv", ["pipeline"]), \
             patch("csi_node.pipeline.run_demo") as mock_run:
            mock_run.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                main()
            mock_run.assert_called_once()

    def test_main_with_replay(self):
        with patch("sys.argv", ["pipeline", "--replay", "test.log"]), \
             patch("csi_node.pipeline.run_demo") as mock_run:
            mock_run.return_value = None
            main()
            _, kwargs = mock_run.call_args
            assert kwargs["replay_path"] == "test.log"

    def test_main_with_all_options(self):
        with patch("sys.argv", [
            "pipeline", "--pose", "--tui", "--replay", "data.log",
            "--window", "5.0", "--out", "out.jsonl", "--speed", "2.0",
            "--log", "override.csv"
        ]), patch("csi_node.pipeline.run_demo") as mock_run:
            mock_run.return_value = None
            main()
            _, kwargs = mock_run.call_args
            assert kwargs["pose"] is True
            assert kwargs["tui"] is True
            assert kwargs["replay_path"] == "data.log"
            assert kwargs["window"] == 5.0
            assert kwargs["speed"] == 2.0
            assert kwargs["log_override"] == "override.csv"

    def test_main_with_iface(self):
        with patch("sys.argv", ["pipeline", "--iface", "wlan0"]), \
             patch("csi_node.pipeline.run_demo") as mock_run, \
             patch("csi_node.feitcsi.live_stream", return_value=iter([])):
            mock_run.return_value = None
            main()
            _, kwargs = mock_run.call_args
            assert kwargs["source"] is not None
