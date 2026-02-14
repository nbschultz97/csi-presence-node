"""Tests for csi_node.setup_wizard — config helpers and UI functions."""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from csi_node import setup_wizard


# ── print helpers ──────────────────────────────────────────────────

class TestPrintHelpers:
    def test_print_header(self, capsys):
        setup_wizard.print_header("Test")
        out = capsys.readouterr().out
        assert "Test" in out
        assert "=" in out

    def test_print_step(self, capsys):
        setup_wizard.print_step(1, 7, "Hardware")
        out = capsys.readouterr().out
        assert "[Step 1/7]" in out
        assert "Hardware" in out


# ── ask_yes_no ─────────────────────────────────────────────────────

class TestAskYesNo:
    def test_yes(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "y")
        assert setup_wizard.ask_yes_no("ok?") is True

    def test_no(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "n")
        assert setup_wizard.ask_yes_no("ok?") is False

    def test_empty_uses_default_true(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        assert setup_wizard.ask_yes_no("ok?", default=True) is True

    def test_empty_uses_default_false(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        assert setup_wizard.ask_yes_no("ok?", default=False) is False

    def test_invalid_then_valid(self, monkeypatch):
        responses = iter(["maybe", "yes"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        assert setup_wizard.ask_yes_no("ok?") is True


# ── ask_float ──────────────────────────────────────────────────────

class TestAskFloat:
    def test_valid_float(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "3.14")
        assert setup_wizard.ask_float("val") == pytest.approx(3.14)

    def test_empty_uses_default(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        assert setup_wizard.ask_float("val", default=2.5) == 2.5

    def test_invalid_then_valid(self, monkeypatch):
        responses = iter(["abc", "1.0"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        assert setup_wizard.ask_float("val") == 1.0


# ── ask_string ─────────────────────────────────────────────────────

class TestAskString:
    def test_returns_input(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "hello")
        assert setup_wizard.ask_string("name") == "hello"

    def test_empty_uses_default(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        assert setup_wizard.ask_string("name", "world") == "world"


# ── load_config / save_config ─────────────────────────────────────

class TestConfigIO:
    def test_load_missing_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(setup_wizard, "CONFIG_PATH", tmp_path / "nope.yaml")
        assert setup_wizard.load_config() == {}

    def test_roundtrip(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "config.yaml"
        monkeypatch.setattr(setup_wizard, "CONFIG_PATH", cfg_path)
        setup_wizard.save_config({"sensor_lat": 38.9})
        loaded = setup_wizard.load_config()
        assert loaded["sensor_lat"] == pytest.approx(38.9)

    def test_load_empty_file(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("")
        monkeypatch.setattr(setup_wizard, "CONFIG_PATH", cfg_path)
        assert setup_wizard.load_config() == {}


# ── check_hardware ─────────────────────────────────────────────────

class TestCheckHardware:
    def test_ax210_detected_via_lspci(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/lspci" if cmd == "lspci" else None)
        mock_result = MagicMock(stdout="Intel AX210 Network", returncode=0)
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_result)
        assert setup_wizard.check_hardware() is True

    def test_no_tools_available(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: None)
        assert setup_wizard.check_hardware() is False


# ── check_feitcsi ──────────────────────────────────────────────────

class TestCheckFeitcsi:
    def test_found_via_which(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/local/bin/feitcsi")
        assert setup_wizard.check_feitcsi() is True

    def test_not_found(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda cmd: None)
        # Also patch Path.exists to return False for common locations
        monkeypatch.setattr(Path, "exists", lambda self: False)
        assert setup_wizard.check_feitcsi() is False


# ── configure_location ─────────────────────────────────────────────

class TestConfigureLocation:
    def test_sets_coordinates(self, monkeypatch):
        responses = iter(["38.8977", "-77.0365", "90.0"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        cfg = {}
        setup_wizard.configure_location(cfg)
        assert cfg["sensor_lat"] == pytest.approx(38.8977)
        assert cfg["sensor_lon"] == pytest.approx(-77.0365)


# ── configure_atak ─────────────────────────────────────────────────

class TestConfigureAtak:
    def test_enable(self, monkeypatch):
        responses = iter(["y", "4242", "vantage-001", "VANTAGE-1"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        cfg = {}
        setup_wizard.configure_atak(cfg)
        assert cfg["atak_enabled"] is True
        assert cfg["atak_port"] == 4242

    def test_disable(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "n")
        cfg = {}
        setup_wizard.configure_atak(cfg)
        assert cfg["atak_enabled"] is False


# ── configure_udp ──────────────────────────────────────────────────

class TestConfigureUdp:
    def test_enable(self, monkeypatch):
        responses = iter(["y", "239.2.3.1", "4243"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        cfg = {}
        setup_wizard.configure_udp(cfg)
        assert cfg["udp_enabled"] is True

    def test_disable(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "n")
        cfg = {}
        setup_wizard.configure_udp(cfg)
        assert cfg["udp_enabled"] is False


# ── print_summary ──────────────────────────────────────────────────

class TestPrintSummary:
    def test_no_crash_on_empty_config(self, capsys):
        setup_wizard.print_summary({})
        out = capsys.readouterr().out
        assert "Calibrated" in out
