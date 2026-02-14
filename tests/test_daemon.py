"""Tests for csi_node.daemon — logging, signal handling, environment checks."""
from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock, patch, mock_open

import pytest
import yaml

from csi_node import daemon


# ── setup_logging ──────────────────────────────────────────────────

class TestSetupLogging:
    def test_returns_logger(self):
        logger = daemon.setup_logging(syslog=False, log_file=None)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "csi-presence"
        # cleanup
        logger.handlers.clear()

    def test_console_handler_always_added(self):
        logger = daemon.setup_logging(syslog=False, log_file=None)
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        logger.handlers.clear()

    def test_log_file_creates_handler(self, tmp_path):
        log_file = str(tmp_path / "sub" / "daemon.log")
        logger = daemon.setup_logging(syslog=False, log_file=log_file)
        from logging.handlers import RotatingFileHandler
        assert any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
        logger.handlers.clear()

    def test_log_level_propagated(self):
        logger = daemon.setup_logging(syslog=False, log_file=None, level=logging.DEBUG)
        assert logger.level == logging.DEBUG
        logger.handlers.clear()

    def test_syslog_skipped_on_non_linux(self):
        with patch.object(sys, "platform", "win32"):
            logger = daemon.setup_logging(syslog=True, log_file=None)
            from logging.handlers import SysLogHandler
            assert not any(isinstance(h, SysLogHandler) for h in logger.handlers)
            logger.handlers.clear()


# ── signal_handler ─────────────────────────────────────────────────

class TestSignalHandler:
    def setup_method(self):
        daemon._shutdown_event = Event()
        daemon._logger = logging.getLogger("test-daemon-sig")

    def test_sets_shutdown_event(self):
        assert not daemon._shutdown_event.is_set()
        daemon.signal_handler(signal.SIGINT, None)
        assert daemon._shutdown_event.is_set()

    def test_works_without_logger(self):
        daemon._logger = None
        daemon.signal_handler(signal.SIGINT, None)
        assert daemon._shutdown_event.is_set()


# ── validate_environment ───────────────────────────────────────────

class TestValidateEnvironment:
    def test_returns_true(self):
        daemon._logger = logging.getLogger("test-env")
        assert daemon.validate_environment() is True

    def test_works_without_logger(self):
        daemon._logger = None
        assert daemon.validate_environment() is True


# ── run_daemon ─────────────────────────────────────────────────────

class TestRunDaemon:
    def test_missing_config_returns_1(self, tmp_path):
        result = daemon.run_daemon(str(tmp_path / "nonexistent.yaml"))
        assert result == 1

    def test_invalid_config_returns_1(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"output_file": ""}))  # will fail validation
        with patch("csi_node.daemon.config_validator") as mock_cv:
            mock_result = MagicMock()
            mock_result.valid = False
            mock_result.errors = ["bad config"]
            mock_cv.validate_config.return_value = mock_result
            result = daemon.run_daemon(str(cfg_path), foreground=True)
        assert result == 1

    def test_valid_config_runs_pipeline(self, tmp_path):
        cfg = {
            "output_file": str(tmp_path / "out.log"),
            "debug": False,
            "window_size": 3.0,
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(cfg))

        mock_validation = MagicMock()
        mock_validation.valid = True
        mock_validation.errors = []
        mock_validation.warnings = []

        with patch("csi_node.daemon.config_validator") as mock_cv, \
             patch("csi_node.daemon.pipeline") as mock_pipe, \
             patch("csi_node.daemon.utils") as mock_utils:
            mock_cv.validate_config.return_value = mock_validation
            # Make pipeline exit, then set shutdown to break loop
            def run_and_stop(**kwargs):
                daemon._shutdown_event.set()
            mock_pipe.run_demo.side_effect = run_and_stop
            mock_utils.RunLogManager.return_value = MagicMock(log_path=tmp_path / "out.log")

            daemon._shutdown_event = Event()
            result = daemon.run_daemon(str(cfg_path), foreground=True)
        assert result == 0

    def test_pipeline_crash_restarts(self, tmp_path):
        cfg = {
            "output_file": str(tmp_path / "out.log"),
            "max_restarts": 2,
            "restart_delay": 0,
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(cfg))

        mock_validation = MagicMock()
        mock_validation.valid = True
        mock_validation.errors = []
        mock_validation.warnings = []

        call_count = 0
        def crash_then_stop(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                daemon._shutdown_event.set()
            raise RuntimeError("crash")

        with patch("csi_node.daemon.config_validator") as mock_cv, \
             patch("csi_node.daemon.pipeline") as mock_pipe, \
             patch("csi_node.daemon.utils") as mock_utils:
            mock_cv.validate_config.return_value = mock_validation
            mock_pipe.run_demo.side_effect = crash_then_stop
            mock_utils.RunLogManager.return_value = MagicMock(log_path=tmp_path / "out.log")

            daemon._shutdown_event = Event()
            result = daemon.run_daemon(str(cfg_path), foreground=True)
        assert result == 1  # max restarts exceeded


# ── main (CLI) ─────────────────────────────────────────────────────

class TestMain:
    def test_env_config_override(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "env_config.yaml"
        cfg_path.write_text(yaml.dump({"output_file": "x.log"}))
        monkeypatch.setenv("CSI_NODE_CONFIG", str(cfg_path))
        monkeypatch.setattr(sys, "argv", ["daemon"])
        with patch("csi_node.daemon.run_daemon", return_value=0) as mock_run, \
             pytest.raises(SystemExit) as exc:
            daemon.main()
        assert mock_run.call_args[1]["config_path"] == str(cfg_path) or \
               mock_run.call_args[0][0] == str(cfg_path)
