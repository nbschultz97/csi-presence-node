"""Extended coverage tests for config_validator.py."""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from csi_node.config_validator import (
    validate_config,
    validate_config_file,
    get_config_with_defaults,
    print_validation_report,
    ValidationResult,
    CONFIG_SCHEMA,
)


class TestValidateConfigExtended:
    def test_type_mismatch_string_for_int(self):
        cfg = {"channel": "not_an_int"}
        result = validate_config(cfg)
        assert not result.valid
        assert any("channel" in e for e in result.errors)

    def test_type_mismatch_auto_fix(self):
        cfg = {"channel": "bad"}
        result = validate_config(cfg, auto_fix=True)
        assert result.fixed_values["channel"] == 36

    def test_above_max(self):
        cfg = {"rotation_max_bytes": 10**12}
        result = validate_config(cfg)
        assert not result.valid

    def test_above_max_auto_fix(self):
        cfg = {"rotation_max_bytes": 10**12}
        result = validate_config(cfg, auto_fix=True)
        assert result.fixed_values["rotation_max_bytes"] == 1048576

    def test_below_min_auto_fix(self):
        cfg = {"window_size": 0.001}
        result = validate_config(cfg, auto_fix=True)
        assert result.fixed_values["window_size"] == 1.0

    def test_high_path_loss_warning(self):
        cfg = {"path_loss_exponent": 5.0}
        result = validate_config(cfg)
        assert result.valid
        assert any("high" in w.lower() for w in result.warnings)

    def test_very_low_tx_power_warning(self):
        cfg = {"tx_power_dbm": -90.0}
        result = validate_config(cfg)
        assert result.valid
        assert any("low" in w.lower() for w in result.warnings)

    def test_calibrated_default_values_warning(self):
        cfg = {"calibrated": True, "path_loss_exponent": 2.0, "tx_power_dbm": -40.0}
        result = validate_config(cfg)
        assert result.valid
        assert any("defaults" in w.lower() for w in result.warnings)

    def test_missing_baseline_file_warning(self):
        cfg = {"baseline_file": "/nonexistent/baseline.npz"}
        result = validate_config(cfg)
        assert any("baseline" in w.lower() for w in result.warnings)

    def test_bool_field_valid(self):
        cfg = {"udp_enabled": True, "atak_enabled": False}
        result = validate_config(cfg)
        assert result.valid

    def test_string_field_valid(self):
        cfg = {"sensor_uid": "my-sensor", "udp_host": "10.0.0.1"}
        result = validate_config(cfg)
        assert result.valid

    def test_udp_port_out_of_range(self):
        cfg = {"udp_port": 99999}
        result = validate_config(cfg)
        assert not result.valid

    def test_sensor_lat_out_of_range(self):
        cfg = {"sensor_lat": 100.0}
        result = validate_config(cfg)
        assert not result.valid


class TestPrintValidationReport:
    def test_valid_no_issues(self, capsys):
        result = ValidationResult(valid=True, warnings=[], errors=[], fixed_values={})
        print_validation_report(result)
        out = capsys.readouterr().out
        assert "VALID" in out
        assert "No issues" in out

    def test_invalid_with_errors(self, capsys):
        result = ValidationResult(
            valid=False, warnings=["warn1"], errors=["err1"], fixed_values={"key": "val"}
        )
        print_validation_report(result, verbose=True)
        out = capsys.readouterr().out
        assert "INVALID" in out
        assert "err1" in out
        assert "warn1" in out
        assert "key" in out

    def test_verbose_false_hides_fixes(self, capsys):
        result = ValidationResult(
            valid=False, warnings=[], errors=["err"], fixed_values={"k": "v"}
        )
        print_validation_report(result, verbose=False)
        out = capsys.readouterr().out
        assert "k" not in out


class TestValidateConfigFileExtended:
    def test_write_fixes_no_fixed_values(self, tmp_path):
        """write_fixes=True but no fixes needed."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"channel": 36}))
        result = validate_config_file(str(cfg_path), auto_fix=True, write_fixes=True)
        assert result.valid


class TestConfigSchema:
    def test_all_schema_entries_have_defaults(self):
        for key, schema in CONFIG_SCHEMA.items():
            assert "default" in schema, f"{key} missing default"
            assert "type" in schema, f"{key} missing type"
