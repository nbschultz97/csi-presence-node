"""Extended tests for csi_node.setup_wizard — helper functions and config I/O."""
from __future__ import annotations

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
