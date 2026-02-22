"""Coverage tests for daemon.py — targeting uncovered lines."""
import signal
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pytest
import yaml

import csi_node.daemon as daemon_mod


class TestSetupLogging:
    def test_basic_logger(self):
        logger = daemon_mod.setup_logging(syslog=False, log_file=None)
        assert isinstance(logger, logging.Logger)

    def test_with_log_file(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        logger = daemon_mod.setup_logging(syslog=False, log_file=log_file)
        assert isinstance(logger, logging.Logger)

    def test_debug_level(self):
        logger = daemon_mod.setup_logging(syslog=False, log_file=None, level=logging.DEBUG)
        assert logger.level == logging.DEBUG


class TestSignalHandler:
    def test_signal_sets_event(self):
        daemon_mod._shutdown_event.clear()
        daemon_mod._logger = MagicMock()
        daemon_mod.signal_handler(signal.SIGINT, None)
        assert daemon_mod._shutdown_event.is_set()
        daemon_mod._shutdown_event.clear()

    def test_signal_no_logger(self):
        daemon_mod._shutdown_event.clear()
        daemon_mod._logger = None
        daemon_mod.signal_handler(signal.SIGINT, None)
        assert daemon_mod._shutdown_event.is_set()
        daemon_mod._shutdown_event.clear()


class TestValidateEnvironment:
    def test_always_true(self):
        daemon_mod._logger = MagicMock()
        assert daemon_mod.validate_environment() is True

    def test_no_logger(self):
        daemon_mod._logger = None
        assert daemon_mod.validate_environment() is True


class TestRunDaemon:
    def test_missing_config(self, tmp_path):
        result = daemon_mod.run_daemon(str(tmp_path / "nope.yaml"))
        assert result == 1

    def test_invalid_config(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"output_file": "/tmp/test.log"}))
        with patch("csi_node.daemon.config_validator") as mock_cv:
            mock_result = MagicMock()
            mock_result.valid = False
            mock_result.errors = ["Missing required field"]
            mock_cv.validate_config.return_value = mock_result
            result = daemon_mod.run_daemon(str(cfg_path), foreground=True)
        assert result == 1
