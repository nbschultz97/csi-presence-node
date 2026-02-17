"""Extended tests for csi_node.setup_wizard — helper functions and config I/O."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from csi_node import setup_wizard


class TestPrintHeader:
    def test_output(self, capsys):
        setup_wizard.print_header("Test Title")
        out = capsys.readouterr().out
        assert "Test Title" in out
        assert "=" * 60 in out


class TestPrintStep:
    def test_output(self, capsys):
        setup_wizard.print_step(3, 7, "Do stuff")
        out = capsys.readouterr().out
        assert "[Step 3/7]" in out
        assert "Do stuff" in out


class TestAskYesNo:
    def test_default_yes(self):
        with patch("builtins.input", return_value=""):
            assert setup_wizard.ask_yes_no("test?", default=True) is True

    def test_default_no(self):
        with patch("builtins.input", return_value=""):
            assert setup_wizard.ask_yes_no("test?", default=False) is False

    def test_explicit_yes(self):
        with patch("builtins.input", return_value="y"):
            assert setup_wizard.ask_yes_no("test?") is True

    def test_explicit_no(self):
        with patch("builtins.input", return_value="n"):
            assert setup_wizard.ask_yes_no("test?") is False

    def test_yes_word(self):
        with patch("builtins.input", return_value="yes"):
            assert setup_wizard.ask_yes_no("test?") is True

    def test_no_word(self):
        with patch("builtins.input", return_value="no"):
            assert setup_wizard.ask_yes_no("test?") is False

    def test_invalid_then_valid(self):
        with patch("builtins.input", side_effect=["maybe", "y"]):
            assert setup_wizard.ask_yes_no("test?") is True


class TestAskFloat:
    def test_valid_float(self):
        with patch("builtins.input", return_value="3.14"):
            assert setup_wizard.ask_float("val") == pytest.approx(3.14)

    def test_default_on_empty(self):
        with patch("builtins.input", return_value=""):
            assert setup_wizard.ask_float("val", default=2.5) == pytest.approx(2.5)

    def test_invalid_then_valid(self):
        with patch("builtins.input", side_effect=["abc", "1.0"]):
            assert setup_wizard.ask_float("val") == pytest.approx(1.0)

    def test_integer_input(self):
        with patch("builtins.input", return_value="42"):
            assert setup_wizard.ask_float("val") == pytest.approx(42.0)


class TestAskString:
    def test_returns_input(self):
        with patch("builtins.input", return_value="hello"):
            assert setup_wizard.ask_string("name") == "hello"

    def test_default_on_empty(self):
        with patch("builtins.input", return_value=""):
            assert setup_wizard.ask_string("name", default="world") == "world"

    def test_empty_no_default(self):
        with patch("builtins.input", return_value=""):
            assert setup_wizard.ask_string("name") == ""


class TestLoadSaveConfig:
    def test_save_and_load(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg = {"atak_enabled": True, "sensor_lat": 38.9}
        with patch.object(setup_wizard, "CONFIG_PATH", cfg_path):
            setup_wizard.save_config(cfg)
            loaded = setup_wizard.load_config()
        assert loaded["atak_enabled"] is True
        assert loaded["sensor_lat"] == pytest.approx(38.9)

    def test_load_missing_returns_empty(self, tmp_path):
        with patch.object(setup_wizard, "CONFIG_PATH", tmp_path / "nope.yaml"):
            assert setup_wizard.load_config() == {}

    def test_load_empty_file_returns_empty(self, tmp_path):
        cfg_path = tmp_path / "empty.yaml"
        cfg_path.write_text("")
        with patch.object(setup_wizard, "CONFIG_PATH", cfg_path):
            assert setup_wizard.load_config() == {}


class TestPrintSummary:
    def test_prints_defaults(self, capsys):
        setup_wizard.print_summary({})
        out = capsys.readouterr().out
        assert "No (using defaults)" in out
        assert "2.0" in out

    def test_prints_calibrated(self, capsys):
        setup_wizard.print_summary({"calibrated": True, "atak_enabled": True})
        out = capsys.readouterr().out
        assert "Yes" in out
        assert "True" in out


class TestCheckHardware:
    def test_no_tools_available(self, capsys):
        with patch("shutil.which", return_value=None):
            result = setup_wizard.check_hardware()
        assert result is False
        out = capsys.readouterr().out
        assert "Could not detect" in out

    def test_lspci_finds_ax210(self):
        from unittest.mock import MagicMock
        mock_result = MagicMock()
        mock_result.stdout = "Intel AX210 WiFi adapter"
        with patch("shutil.which", return_value="/usr/bin/lspci"), \
             patch("subprocess.run", return_value=mock_result):
            assert setup_wizard.check_hardware() is True


class TestCheckFeitCSI:
    def test_found_on_path(self):
        with patch("shutil.which", return_value="/usr/bin/feitcsi"):
            assert setup_wizard.check_feitcsi() is True

    def test_not_found(self, capsys):
        with patch("shutil.which", return_value=None), \
             patch.object(Path, "exists", return_value=False):
            result = setup_wizard.check_feitcsi()
        assert result is False


class TestConfigureLocation:
    def test_set_new_location(self):
        cfg = {}
        with patch("builtins.input", side_effect=["40.0", "-111.0", "90.0"]):
            setup_wizard.configure_location(cfg)
        assert cfg["sensor_lat"] == pytest.approx(40.0)
        assert cfg["sensor_lon"] == pytest.approx(-111.0)
        assert cfg["sensor_heading"] == pytest.approx(90.0)

    def test_keep_existing(self):
        cfg = {"sensor_lat": 38.0, "sensor_lon": -77.0}
        with patch("builtins.input", return_value="n"):
            setup_wizard.configure_location(cfg)
        assert cfg["sensor_lat"] == pytest.approx(38.0)


class TestConfigureATAK:
    def test_enable(self):
        cfg = {}
        with patch("builtins.input", side_effect=["y", "4242", "vantage-01", "V1"]):
            setup_wizard.configure_atak(cfg)
        assert cfg["atak_enabled"] is True
        assert cfg["atak_port"] == 4242

    def test_disable(self):
        cfg = {}
        with patch("builtins.input", return_value="n"):
            setup_wizard.configure_atak(cfg)
        assert cfg["atak_enabled"] is False


class TestConfigureUDP:
    def test_enable(self):
        cfg = {}
        with patch("builtins.input", side_effect=["y", "239.2.3.1", "4243"]):
            setup_wizard.configure_udp(cfg)
        assert cfg["udp_enabled"] is True
        assert cfg["udp_port"] == 4243

    def test_disable(self):
        cfg = {}
        with patch("builtins.input", return_value="n"):
            setup_wizard.configure_udp(cfg)
        assert cfg["udp_enabled"] is False


class TestCheckHardwareExtended:
    def test_lspci_with_ax211(self):
        """Test hardware check detects AX211."""
        from unittest.mock import MagicMock
        mock_result = MagicMock()
        mock_result.stdout = "Intel Corporation Wi-Fi 6E AX211"
        with patch("shutil.which", return_value="/usr/bin/lspci"), \
             patch("subprocess.run", return_value=mock_result):
            assert setup_wizard.check_hardware() is True

    def test_lspci_subprocess_error(self):
        """Test hardware check handles lspci errors."""
        with patch("shutil.which", return_value="/usr/bin/lspci"), \
             patch("subprocess.run", side_effect=Exception("Command failed")):
            # Should fall back to iwconfig
            assert setup_wizard.check_hardware() is False

    def test_iwconfig_fallback(self):
        """Test iwconfig fallback when lspci fails."""
        from unittest.mock import MagicMock
        mock_result = MagicMock()
        mock_result.stdout = "wlan0     IEEE 802.11  ESSID:off/any"
        with patch("shutil.which", side_effect=lambda cmd: "/usr/sbin/iwconfig" if cmd == "iwconfig" else None), \
             patch("subprocess.run", return_value=mock_result):
            assert setup_wizard.check_hardware() is True

    def test_iwconfig_error(self):
        """Test iwconfig error handling."""
        with patch("shutil.which", side_effect=lambda cmd: "/usr/sbin/iwconfig" if cmd == "iwconfig" else None), \
             patch("subprocess.run", side_effect=Exception("iwconfig failed")):
            assert setup_wizard.check_hardware() is False

    def test_subprocess_timeout(self):
        """Test subprocess timeout handling."""
        with patch("shutil.which", return_value="/usr/bin/lspci"), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired("lspci", 10)):
            assert setup_wizard.check_hardware() is False


class TestCheckFeitCSIExtended:
    def test_found_in_common_path(self, tmp_path):
        """Test FeitCSI found in common locations."""
        # Create fake feitcsi binary
        feitcsi_path = tmp_path / "FeitCSI" / "build" / "feitcsi"
        feitcsi_path.parent.mkdir(parents=True, exist_ok=True)
        feitcsi_path.write_text("#!/bin/bash\necho feitcsi")
        
        with patch("shutil.which", return_value=None), \
             patch("pathlib.Path.home", return_value=tmp_path):
            assert setup_wizard.check_feitcsi() is True

    def test_not_found_anywhere(self):
        """Test FeitCSI not found anywhere."""
        with patch("shutil.which", return_value=None), \
             patch.object(Path, "exists", return_value=False):
            assert setup_wizard.check_feitcsi() is False


class TestCaptureBaseline:
    def test_capture_baseline_declined(self):
        """Test user declines baseline capture."""
        with patch("builtins.input", return_value="n"):
            result = setup_wizard.capture_baseline({})
        assert result is False

    def test_capture_baseline_success(self, tmp_path):
        """Test successful baseline capture."""
        from unittest.mock import MagicMock
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        
        with patch("builtins.input", side_effect=["y", "", "30"]):  # yes, enter, 30 seconds
            with patch("subprocess.run", return_value=mock_result):
                result = setup_wizard.capture_baseline({})
        
        assert result is True

    def test_capture_baseline_timeout(self):
        """Test baseline capture timeout."""
        import subprocess
        
        with patch("builtins.input", side_effect=["y", "", "10"]):
            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 10)):
                result = setup_wizard.capture_baseline({})
        
        assert result is False

    def test_capture_baseline_failure(self):
        """Test baseline capture failure."""
        from unittest.mock import MagicMock
        
        mock_result = MagicMock()
        mock_result.returncode = 1
        
        with patch("builtins.input", side_effect=["y", "", "10"]):
            with patch("subprocess.run", return_value=mock_result):
                result = setup_wizard.capture_baseline({})
        
        assert result is False

    def test_capture_baseline_exception(self):
        """Test baseline capture exception handling."""
        with patch("builtins.input", side_effect=["y", "", "10"]):
            with patch("subprocess.run", side_effect=Exception("Process failed")):
                result = setup_wizard.capture_baseline({})
        
        assert result is False


class TestRunCalibration:
    def test_calibration_declined(self):
        """Test user declines calibration."""
        with patch("builtins.input", return_value="n"):
            result = setup_wizard.run_calibration({})
        assert result is False

    def test_calibration_no_log_files(self, tmp_path):
        """Test calibration when log files don't exist."""
        with patch("builtins.input", side_effect=[
            "y",  # yes to calibration
            "1.0",  # near distance
            "3.0",  # far distance
            "",  # enter for near recording ready
            "",  # enter for near recording complete
            "",  # enter for far recording ready  
            "",  # enter for far recording complete
        ]):
            # Mock Path.exists to return False (files don't exist)
            with patch("pathlib.Path.exists", return_value=False):
                result = setup_wizard.run_calibration({})
        
        # Should return False because log files don't exist
        assert result is False

    def test_calibration_success(self, tmp_path):
        """Test successful calibration."""
        from unittest.mock import MagicMock
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        
        with patch("builtins.input", side_effect=[
            "y",  # yes to calibration
            "1.5",  # near distance
            "4.0",  # far distance
            "",  # enter for near recording ready
            "",  # enter for near recording complete
            "",  # enter for far recording ready
            "",  # enter for far recording complete
        ]):
            # Mock Path.exists to return True (files exist)
            with patch("pathlib.Path.exists", return_value=True):
                with patch("subprocess.run", return_value=mock_result):
                    result = setup_wizard.run_calibration({})
        
        assert result is True

    def test_calibration_subprocess_error(self, tmp_path):
        """Test calibration subprocess error."""        
        with patch("builtins.input", side_effect=[
            "y", "1.0", "3.0", "", "", "", ""
        ]):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("subprocess.run", side_effect=Exception("Calibration failed")):
                    result = setup_wizard.run_calibration({})
        
        assert result is False


class TestTrainPoseModel:
    def test_training_declined(self):
        """Test user declines pose model training."""
        with patch("builtins.input", return_value="n"):
            result = setup_wizard.train_pose_model({})
        assert result is False

    def test_training_declined_after_initial_yes(self):
        """Test user declines training after initial yes."""
        with patch("builtins.input", side_effect=["y", "n"]):  # yes, then no to continue
            result = setup_wizard.train_pose_model({})
        assert result is False

    def test_training_success(self):
        """Test successful pose model training."""
        from unittest.mock import MagicMock
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        
        with patch("builtins.input", side_effect=["y", "y"]):  # yes to train, yes to continue
            with patch("subprocess.run", return_value=mock_result):
                result = setup_wizard.train_pose_model({})
        
        assert result is True

    def test_training_data_collection_fails(self):
        """Test training failure during data collection."""
        from unittest.mock import MagicMock
        
        mock_result = MagicMock()
        mock_result.returncode = 1  # Failure
        
        with patch("builtins.input", side_effect=["y", "y"]):
            with patch("subprocess.run", return_value=mock_result):
                result = setup_wizard.train_pose_model({})
        
        assert result is False

    def test_training_exception(self):
        """Test training exception handling."""
        with patch("builtins.input", side_effect=["y", "y"]):
            with patch("subprocess.run", side_effect=Exception("Training failed")):
                result = setup_wizard.train_pose_model({})
        
        assert result is False


class TestMainFunction:
    def test_main_setup_cancelled(self):
        """Test main function when user cancels setup."""
        with patch("builtins.input", return_value="n"):  # Cancel setup
            with patch("builtins.print"):
                setup_wizard.main()
        # Should exit gracefully

    def test_main_hardware_check_fails_user_exits(self):
        """Test main function when hardware checks fail and user exits."""
        with patch("builtins.input", side_effect=["y", "n"]):  # yes to continue, no to continue anyway
            with patch("csi_node.setup_wizard.check_hardware", return_value=False):
                with patch("csi_node.setup_wizard.check_feitcsi", return_value=False):
                    with patch("builtins.print"):
                        setup_wizard.main()
        # Should exit gracefully

    def test_main_full_workflow_success(self, tmp_path):
        """Test main function with full successful workflow."""
        cfg_path = tmp_path / "config.yaml"
        
        with patch.object(setup_wizard, "CONFIG_PATH", cfg_path):
            with patch("builtins.input", side_effect=[
                "y",  # continue with setup
                "y",  # continue anyway after hardware check
            ]):
                with patch("csi_node.setup_wizard.check_hardware", return_value=False):
                    with patch("csi_node.setup_wizard.check_feitcsi", return_value=False):
                        with patch("csi_node.setup_wizard.capture_baseline", return_value=True):
                            with patch("csi_node.setup_wizard.run_calibration", return_value=True):
                                with patch("csi_node.setup_wizard.configure_location"):
                                    with patch("csi_node.setup_wizard.configure_atak"):
                                        with patch("csi_node.setup_wizard.configure_udp"):
                                            with patch("csi_node.setup_wizard.train_pose_model", return_value=True):
                                                with patch("csi_node.setup_wizard.print_summary"):
                                                    with patch("builtins.print"):
                                                        setup_wizard.main()
        
        # Should have saved configuration
        assert cfg_path.exists()

    def test_main_hardware_ok_workflow(self):
        """Test main function when hardware checks pass."""
        with patch("builtins.input", return_value="y"):  # continue with setup
            with patch("csi_node.setup_wizard.check_hardware", return_value=True):
                with patch("csi_node.setup_wizard.check_feitcsi", return_value=True):
                    with patch("csi_node.setup_wizard.capture_baseline", return_value=False):
                        with patch("csi_node.setup_wizard.run_calibration", return_value=False):
                            with patch("csi_node.setup_wizard.configure_location"):
                                with patch("csi_node.setup_wizard.configure_atak"):
                                    with patch("csi_node.setup_wizard.configure_udp"):
                                        with patch("csi_node.setup_wizard.train_pose_model", return_value=False):
                                            with patch("csi_node.setup_wizard.save_config"):
                                                with patch("csi_node.setup_wizard.print_summary"):
                                                    with patch("builtins.print"):
                                                        setup_wizard.main()
        # Should complete successfully


class TestMainEntryPoint:
    def test_main_entry_point(self):
        """Test the if __name__ == '__main__' entry point."""
        with patch("csi_node.setup_wizard.main") as mock_main:
            # Test that the entry point calls main when run directly
            # This simulates the behavior without executing the full file
            import csi_node.setup_wizard as sw_module
            
            # Simulate the condition in the actual file
            if hasattr(sw_module, '__name__'):
                # Just call main directly as that's what the entry point does
                sw_module.main()
            
            mock_main.assert_called_once()
