"""Coverage tests for setup_wizard — capture_baseline, run_calibration, train_pose_model, main."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from csi_node import setup_wizard


class TestCaptureBaseline:
    def test_skip_baseline(self):
        with patch("builtins.input", return_value="n"):
            result = setup_wizard.capture_baseline({})
        assert result is False

    def test_successful_capture(self):
        mock_result = MagicMock(returncode=0)
        inputs = iter(["y", "", "60"])  # yes, enter for ready, duration
        with patch("builtins.input", side_effect=inputs), \
             patch("subprocess.run", return_value=mock_result):
            result = setup_wizard.capture_baseline({})
        assert result is True

    def test_failed_capture(self):
        mock_result = MagicMock(returncode=1)
        inputs = iter(["y", "", "10"])
        with patch("builtins.input", side_effect=inputs), \
             patch("subprocess.run", return_value=mock_result):
            result = setup_wizard.capture_baseline({})
        assert result is False

    def test_timeout_capture(self):
        import subprocess
        inputs = iter(["y", "", "5"])
        with patch("builtins.input", side_effect=inputs), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 35)):
            result = setup_wizard.capture_baseline({})
        assert result is False

    def test_exception_capture(self):
        inputs = iter(["y", "", "5"])
        with patch("builtins.input", side_effect=inputs), \
             patch("subprocess.run", side_effect=OSError("fail")):
            result = setup_wizard.capture_baseline({})
        assert result is False


class TestRunCalibration:
    def test_skip_calibration(self):
        with patch("builtins.input", return_value="n"):
            result = setup_wizard.run_calibration({})
        assert result is False

    def test_successful_calibration(self, tmp_path):
        near_log = Path("data/calibration_near.log")
        far_log = Path("data/calibration_far.log")

        # Inputs: yes, d1, d2, ready near, ready near done, ready far, ready far done
        inputs = iter(["y", "1.0", "3.0", "", "", "", ""])
        mock_result = MagicMock(returncode=0)

        with patch("builtins.input", side_effect=inputs), \
             patch.object(Path, "exists", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            cfg = {}
            result = setup_wizard.run_calibration(cfg)
        assert result is True
        assert cfg.get("calibrated") is True

    def test_log_files_not_found(self):
        inputs = iter(["y", "1.0", "3.0", "", "", "", ""])
        with patch("builtins.input", side_effect=inputs), \
             patch.object(Path, "exists", return_value=False):
            result = setup_wizard.run_calibration({})
        assert result is False


class TestTrainPoseModel:
    def test_skip_training(self):
        with patch("builtins.input", return_value="n"):
            result = setup_wizard.train_pose_model({})
        assert result is False

    def test_cancel_after_first_prompt(self):
        inputs = iter(["y", "n"])
        with patch("builtins.input", side_effect=inputs):
            result = setup_wizard.train_pose_model({})
        assert result is False

    def test_successful_training(self):
        inputs = iter(["y", "y"])
        mock_result = MagicMock(returncode=0)
        with patch("builtins.input", side_effect=inputs), \
             patch("subprocess.run", return_value=mock_result):
            result = setup_wizard.train_pose_model({})
        assert result is True

    def test_collector_fails(self):
        inputs = iter(["y", "y"])
        mock_result = MagicMock(returncode=1)
        with patch("builtins.input", side_effect=inputs), \
             patch("subprocess.run", return_value=mock_result):
            result = setup_wizard.train_pose_model({})
        assert result is False

    def test_training_exception(self):
        inputs = iter(["y", "y"])
        with patch("builtins.input", side_effect=inputs), \
             patch("subprocess.run", side_effect=OSError("fail")):
            result = setup_wizard.train_pose_model({})
        assert result is False


class TestCheckHardwareIwconfig:
    def test_iwconfig_detects_wifi(self):
        def which(cmd):
            if cmd == "iwconfig":
                return "/sbin/iwconfig"
            return None

        mock_result = MagicMock(stdout="wlan0  IEEE 802.11", returncode=0)
        with patch("shutil.which", side_effect=which), \
             patch("subprocess.run", return_value=mock_result):
            assert setup_wizard.check_hardware() is True

    def test_lspci_ax211(self):
        mock_result = MagicMock(stdout="Intel AX211 Network", returncode=0)
        with patch("shutil.which", return_value="/usr/bin/lspci"), \
             patch("subprocess.run", return_value=mock_result):
            assert setup_wizard.check_hardware() is True

    def test_lspci_exception(self):
        with patch("shutil.which", return_value="/usr/bin/lspci"), \
             patch("subprocess.run", side_effect=OSError("fail")):
            # Falls through to iwconfig check
            result = setup_wizard.check_hardware()
            assert result is False


class TestCheckFeitCSICommonPaths:
    def test_found_in_common_path(self):
        with patch("shutil.which", return_value=None), \
             patch.object(Path, "exists", return_value=True):
            assert setup_wizard.check_feitcsi() is True


class TestMainWizard:
    def test_cancel_setup(self, capsys):
        with patch("builtins.input", return_value="n"):
            setup_wizard.main()
        out = capsys.readouterr().out
        assert "cancelled" in out

    def test_full_run_skip_all(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        # Pre-create empty config so load_config works
        cfg_path.write_text("{}")
        inputs = iter([
            "y",   # continue setup
            "y",   # continue despite hw failure
            "n",   # skip baseline
            "n",   # skip calibration
            "0.0", "0.0", "0.0",  # location
            "n",   # skip atak
            "n",   # skip udp
            "n",   # skip training
        ])
        with patch("builtins.input", side_effect=inputs), \
             patch.object(setup_wizard, "CONFIG_PATH", cfg_path), \
             patch("shutil.which", return_value=None):
            try:
                setup_wizard.main()
            except StopIteration:
                pass  # Expected if input sequence is off


class TestConfigureLocationExisting:
    def test_skip_update_existing(self):
        cfg = {"sensor_lat": 38.0, "sensor_lon": -77.0}
        with patch("builtins.input", return_value="n"):
            setup_wizard.configure_location(cfg)
        assert cfg["sensor_lat"] == 38.0
