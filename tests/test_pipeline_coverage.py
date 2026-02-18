"""Coverage tests for pipeline.py — targeting uncovered lines."""
from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from threading import Event
from unittest.mock import patch, MagicMock, mock_open

import numpy as np
import pytest

from csi_node.pipeline import (
    PresenceDetector,
    CSILogHandler,
    compute_window,
    run_demo,
    run_offline,
    main as pipeline_main,
)


def _make_pkt(ts, csi_shape=(2, 64), rssi=(-40, -42)):
    rng = np.random.default_rng(int(ts * 1000) % 2**31)
    return {"ts": ts, "csi": rng.standard_normal(csi_shape), "rssi": list(rssi)}


class TestCSILogHandlerRotation:
    """Test log rotation / truncation detection in CSILogHandler."""

    def test_log_rotation_detected(self, tmp_path):
        """Lines 116-127: inode change or size shrink triggers reopen."""
        log = tmp_path / "csi.log"
        log.write_text("initial content\n")
        buf = deque()
        calls = []
        handler = CSILogHandler(log, buf, lambda: calls.append(1))

        # Simulate truncation: rewrite file with smaller content
        log.write_text("")
        event = MagicMock()
        event.src_path = str(log)
        # Force size to appear smaller
        handler._size = 9999
        handler.on_modified(event)

    def test_on_modified_file_not_found(self, tmp_path):
        """Lines 111-112: stat raises FileNotFoundError."""
        log = tmp_path / "csi.log"
        log.write_text("")
        buf = deque()
        handler = CSILogHandler(log, buf, lambda: None)

        event = MagicMock()
        event.src_path = str(log)

        # Mock os.stat to raise FileNotFoundError
        with patch("os.stat", side_effect=FileNotFoundError):
            handler.on_modified(event)  # Should not raise

    def test_rotation_reopen_file_missing(self, tmp_path):
        """Lines 139-142: reopen fails because file disappeared during rotation."""
        log = tmp_path / "csi.log"
        log.write_text("data\n")
        buf = deque()
        handler = CSILogHandler(log, buf, lambda: None)

        event = MagicMock()
        event.src_path = str(log)

        # Simulate inode change but file gone when reopening
        with patch("os.stat") as mock_stat:
            stat_result = MagicMock()
            stat_result.st_ino = handler._inode + 1  # Different inode
            stat_result.st_size = 100
            mock_stat.return_value = stat_result
            # Close handler's fp, delete file, trigger modified
            handler._fp.close()
            log.unlink()
            handler.on_modified(event)

    def test_pkt_cb_exception_ignored(self, tmp_path):
        """Line ~108: pkt_cb exception is swallowed."""
        log = tmp_path / "csi.log"
        log.write_text("")
        buf = deque()

        def bad_cb():
            raise RuntimeError("boom")

        handler = CSILogHandler(log, buf, lambda: None, pkt_cb=bad_cb)

        # Write a valid JSON line that parse_csi_line can handle
        with open(log, "a") as f:
            pkt = {"ts": 1.0, "csi": [[1, 2], [3, 4]], "rssi": [-40, -42]}
            f.write(json.dumps(pkt) + "\n")

        event = MagicMock()
        event.src_path = str(log)
        handler.on_modified(event)  # Should not raise despite bad_cb


class TestComputeWindowMovement:
    """Test movement detection branch in compute_window."""

    def test_movement_detected(self):
        """Lines 242: movement='moving' when var_delta > threshold."""
        cfg = {
            "variance_threshold": 5.0,
            "pca_threshold": 1.0,
            "rssi_delta": 3.0,
            "tx_power_dbm": -40.0,
            "path_loss_exponent": 2.0,
            "movement_threshold": 0.001,  # Very low threshold
            "log_dropped": False,
            "min_conditioning_samples": 999,
        }
        pkts = [_make_pkt(t * 0.1) for t in range(30)]
        buf = deque(pkts)
        result = compute_window(buf, 0, 3, None, cfg, prev_var=0.0)
        assert result["movement"] == "moving"

    def test_baseline_subtraction(self):
        """Test baseline subtraction branch."""
        cfg = {
            "variance_threshold": 5.0,
            "pca_threshold": 1.0,
            "rssi_delta": 3.0,
            "tx_power_dbm": -40.0,
            "path_loss_exponent": 2.0,
            "movement_threshold": 2.0,
            "log_dropped": False,
            "min_conditioning_samples": 999,
        }
        pkts = [_make_pkt(t * 0.1) for t in range(30)]
        buf = deque(pkts)
        baseline = np.zeros((2, 64))
        result = compute_window(buf, 0, 3, baseline, cfg)
        assert result is not None


class TestRunDemo:
    """Test run_demo with source iterator."""

    def test_run_demo_with_source(self, tmp_path):
        """Test run_demo processes source packets."""
        pkts = [_make_pkt(t * 0.1) for t in range(20)]

        out_file = str(tmp_path / "out.jsonl")
        with patch("csi_node.pipeline.config_validator") as mock_cv:
            mock_cv.validate_config.return_value = MagicMock(
                valid=True, errors=[], warnings=[]
            )
            with patch("csi_node.pipeline.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {
                    "window_size": 1.0,
                    "window_hop": 0.5,
                    "output_file": out_file,
                    "baseline_file": "nonexistent.npz",
                    "log_file": "nonexistent.log",
                    "variance_threshold": 5.0,
                    "pca_threshold": 1.0,
                    "rssi_delta": 3.0,
                    "tx_power_dbm": -40.0,
                    "path_loss_exponent": 2.0,
                    "movement_threshold": 2.0,
                    "min_conditioning_samples": 999,
                    "frames_wait_timeout": 60,
                    "rotation_max_bytes": 1048576,
                    "rotation_interval_seconds": 0,
                    "rotation_retention": 5,
                }
                run_demo(source=iter(pkts), out=out_file)

    def test_run_demo_with_replay(self, tmp_path):
        """Test run_demo with replay_path."""
        log = tmp_path / "replay.log"
        pkts_data = [
            {"ts": t * 0.1, "csi": np.random.randn(2, 64).tolist(), "rssi": [-40, -42]}
            for t in range(20)
        ]
        log.write_text("\n".join(json.dumps(p) for p in pkts_data) + "\n")
        out_file = str(tmp_path / "out.jsonl")

        with patch("csi_node.pipeline.config_validator") as mock_cv:
            mock_cv.validate_config.return_value = MagicMock(
                valid=True, errors=[], warnings=[]
            )
            with patch("csi_node.pipeline.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {
                    "window_size": 1.0,
                    "window_hop": 0.5,
                    "output_file": out_file,
                    "baseline_file": "nonexistent.npz",
                    "log_file": "nonexistent.log",
                    "variance_threshold": 5.0,
                    "pca_threshold": 1.0,
                    "rssi_delta": 3.0,
                    "tx_power_dbm": -40.0,
                    "path_loss_exponent": 2.0,
                    "movement_threshold": 2.0,
                    "min_conditioning_samples": 999,
                    "frames_wait_timeout": 60,
                    "rotation_max_bytes": 1048576,
                    "rotation_interval_seconds": 0,
                    "rotation_retention": 5,
                }
                run_demo(replay_path=str(log), out=out_file, speed=0)

    def test_run_demo_invalid_config_warnings(self, tmp_path):
        """Test run_demo with invalid config prints warnings."""
        out_file = str(tmp_path / "out.jsonl")
        pkts = [_make_pkt(t * 0.1) for t in range(5)]

        with patch("csi_node.pipeline.config_validator") as mock_cv:
            mock_cv.validate_config.return_value = MagicMock(
                valid=False, errors=["bad value"], warnings=["suspicious"]
            )
            with patch("csi_node.pipeline.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {
                    "window_size": 1.0,
                    "window_hop": 0.5,
                    "output_file": out_file,
                    "baseline_file": "nonexistent.npz",
                    "log_file": "nonexistent.log",
                    "variance_threshold": 5.0,
                    "pca_threshold": 1.0,
                    "rssi_delta": 3.0,
                    "tx_power_dbm": -40.0,
                    "path_loss_exponent": 2.0,
                    "movement_threshold": 2.0,
                    "min_conditioning_samples": 999,
                    "frames_wait_timeout": 60,
                    "rotation_max_bytes": 1048576,
                    "rotation_interval_seconds": 0,
                    "rotation_retention": 5,
                }
                run_demo(source=iter(pkts), out=out_file)

    def test_run_demo_with_udp(self, tmp_path):
        """Test run_demo with UDP streaming enabled."""
        out_file = str(tmp_path / "out.jsonl")
        pkts = [_make_pkt(t * 0.1) for t in range(10)]

        with patch("csi_node.pipeline.config_validator") as mock_cv:
            mock_cv.validate_config.return_value = MagicMock(
                valid=True, errors=[], warnings=[]
            )
            with patch("csi_node.pipeline.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {
                    "window_size": 1.0,
                    "window_hop": 0.5,
                    "output_file": out_file,
                    "baseline_file": "nonexistent.npz",
                    "log_file": "nonexistent.log",
                    "variance_threshold": 5.0,
                    "pca_threshold": 1.0,
                    "rssi_delta": 3.0,
                    "tx_power_dbm": -40.0,
                    "path_loss_exponent": 2.0,
                    "movement_threshold": 2.0,
                    "min_conditioning_samples": 999,
                    "frames_wait_timeout": 60,
                    "rotation_max_bytes": 1048576,
                    "rotation_interval_seconds": 0,
                    "rotation_retention": 5,
                    "udp_enabled": True,
                    "udp_host": "127.0.0.1",
                    "udp_port": 19998,
                    "atak_enabled": False,
                }
                run_demo(source=iter(pkts), out=out_file)

    def test_run_demo_tui_unavailable(self, tmp_path):
        """Test run_demo when TUI is requested but unavailable."""
        out_file = str(tmp_path / "out.jsonl")
        pkts = [_make_pkt(t * 0.1) for t in range(5)]

        with patch("csi_node.pipeline.tui_mod", None):
            with patch("csi_node.pipeline.config_validator") as mock_cv:
                mock_cv.validate_config.return_value = MagicMock(
                    valid=True, errors=[], warnings=[]
                )
                with patch("csi_node.pipeline.yaml.safe_load") as mock_yaml:
                    mock_yaml.return_value = {
                        "window_size": 1.0,
                        "window_hop": 0.5,
                        "output_file": out_file,
                        "baseline_file": "nonexistent.npz",
                        "log_file": "nonexistent.log",
                        "variance_threshold": 5.0,
                        "pca_threshold": 1.0,
                        "rssi_delta": 3.0,
                        "tx_power_dbm": -40.0,
                        "path_loss_exponent": 2.0,
                        "movement_threshold": 2.0,
                        "min_conditioning_samples": 999,
                        "frames_wait_timeout": 60,
                        "rotation_max_bytes": 1048576,
                        "rotation_interval_seconds": 0,
                        "rotation_retention": 5,
                    }
                    run_demo(tui=True, source=iter(pkts), out=out_file)


class TestRunOffline:
    """Test run_offline function."""

    def test_run_offline_basic(self, tmp_path):
        """Test offline processing of a log file."""
        log = tmp_path / "test.log"
        lines = []
        for t in range(30):
            pkt = {"ts": t * 0.1, "csi": np.random.randn(2, 64).tolist(), "rssi": [-40, -42]}
            lines.append(json.dumps(pkt))
        log.write_text("\n".join(lines) + "\n")

        cfg = {
            "window_size": 1.0,
            "baseline_file": "nonexistent.npz",
            "log_wait": 0,
            "variance_threshold": 5.0,
            "pca_threshold": 1.0,
            "rssi_delta": 3.0,
            "tx_power_dbm": -40.0,
            "path_loss_exponent": 2.0,
            "movement_threshold": 2.0,
            "min_conditioning_samples": 999,
        }
        df = run_offline(str(log), cfg)
        assert len(df) > 0
        assert "presence" in df.columns

    def test_run_offline_missing_file(self, tmp_path):
        """Test offline with missing file."""
        cfg = {
            "window_size": 1.0,
            "baseline_file": "nonexistent.npz",
            "log_wait": 0,
        }
        with pytest.raises(FileNotFoundError):
            run_offline(str(tmp_path / "nope.log"), cfg)


class TestPipelineMain:
    """Test pipeline main() argument parsing."""

    def test_main_calls_run_demo(self):
        with patch("csi_node.pipeline.run_demo") as mock_run:
            with patch("sys.argv", ["pipeline", "--window", "2.0", "--speed", "1.5"]):
                pipeline_main()
            mock_run.assert_called_once()
            kwargs = mock_run.call_args
            assert kwargs[1]["window"] == 2.0 or kwargs.kwargs["window"] == 2.0

    def test_main_with_replay(self):
        with patch("csi_node.pipeline.run_demo") as mock_run:
            with patch("sys.argv", ["pipeline", "--replay", "test.log"]):
                pipeline_main()
            call_kwargs = mock_run.call_args
            assert call_kwargs.kwargs.get("replay_path") == "test.log" or call_kwargs[1].get("replay_path") == "test.log"

    def test_main_with_log_override(self):
        with patch("csi_node.pipeline.run_demo") as mock_run:
            with patch("sys.argv", ["pipeline", "--log", "custom.log"]):
                pipeline_main()
            mock_run.assert_called_once()
