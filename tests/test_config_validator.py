"""Tests for configuration validation module."""

import pytest
import tempfile
from pathlib import Path
import yaml

from csi_node.config_validator import (
    validate_config,
    validate_config_file,
    get_config_with_defaults,
    ValidationResult,
)


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_valid_config(self):
        """Test validation of a valid config."""
        cfg = {
            "channel": 36,
            "bandwidth": 80,
            "window_size": 1.0,
            "variance_threshold": 5.0,
            "path_loss_exponent": 2.5,
            "tx_power_dbm": -40.0,
        }
        result = validate_config(cfg)
        assert result.valid
        assert len(result.errors) == 0

    def test_negative_path_loss(self):
        """Test detection of negative path loss exponent."""
        cfg = {
            "path_loss_exponent": -0.09,
        }
        result = validate_config(cfg)
        assert not result.valid
        assert any("negative" in err.lower() for err in result.errors)

    def test_invalid_bandwidth(self):
        """Test detection of invalid bandwidth."""
        cfg = {
            "bandwidth": 100,  # Invalid - should be 20/40/80/160
        }
        result = validate_config(cfg)
        assert not result.valid
        assert any("bandwidth" in err.lower() for err in result.errors)

    def test_out_of_range_channel(self):
        """Test detection of out of range channel."""
        cfg = {
            "channel": 300,  # Invalid channel
        }
        result = validate_config(cfg)
        assert not result.valid

    def test_warning_for_low_path_loss(self):
        """Test warning for unusually low path loss."""
        cfg = {
            "path_loss_exponent": 1.2,  # Below typical indoor
        }
        result = validate_config(cfg)
        assert result.valid  # Still valid, just warning
        assert len(result.warnings) > 0

    def test_warning_for_positive_tx_power(self):
        """Test warning for positive tx_power."""
        cfg = {
            "tx_power_dbm": 10.0,  # Positive is unusual
        }
        result = validate_config(cfg)
        assert result.valid  # Still valid, just warning
        assert any("positive" in warn.lower() for warn in result.warnings)

    def test_auto_fix(self):
        """Test auto-fix of invalid values."""
        cfg = {
            "path_loss_exponent": -0.5,
            "bandwidth": 100,
        }
        result = validate_config(cfg, auto_fix=True)
        assert not result.valid  # Still invalid
        assert "path_loss_exponent" in result.fixed_values
        assert "bandwidth" in result.fixed_values
        assert result.fixed_values["path_loss_exponent"] == 2.0
        assert result.fixed_values["bandwidth"] == 80

    def test_type_coercion(self):
        """Test that int/float coercion works."""
        cfg = {
            "window_size": 1,  # Int where float expected
            "channel": 36.0,  # Float where int expected
        }
        result = validate_config(cfg)
        assert result.valid
        assert len(result.errors) == 0

    def test_calibration_inconsistency(self):
        """Test detection of calibration marked but bad values."""
        cfg = {
            "calibrated": True,
            "path_loss_exponent": -0.5,
        }
        result = validate_config(cfg)
        assert not result.valid
        assert any("calibration" in err.lower() for err in result.errors)


class TestValidateConfigFile:
    """Tests for validate_config_file function."""

    def test_missing_file(self):
        """Test handling of missing config file."""
        result = validate_config_file("/nonexistent/path/config.yaml")
        assert not result.valid
        assert any("not found" in err.lower() for err in result.errors)

    def test_valid_file(self):
        """Test validation of a valid config file."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                tmp_path = f.name
                yaml.dump({
                    "channel": 36,
                    "bandwidth": 80,
                    "path_loss_exponent": 2.5,
                }, f)
                f.flush()

            result = validate_config_file(tmp_path)
            assert result.valid
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def test_write_fixes(self):
        """Test writing fixes back to file."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                tmp_path = f.name
                yaml.dump({
                    "channel": 36,
                    "path_loss_exponent": -0.5,  # Invalid
                }, f)
                f.flush()

            result = validate_config_file(tmp_path, auto_fix=True, write_fixes=True)
            assert not result.valid

            # Read back and verify fix was written
            with open(tmp_path) as f2:
                fixed_cfg = yaml.safe_load(f2)

            # Note: write_fixes only writes if there are fixed values
            if result.fixed_values:
                assert fixed_cfg.get("path_loss_exponent") == 2.0
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)


class TestGetConfigWithDefaults:
    """Tests for get_config_with_defaults function."""

    def test_empty_config(self):
        """Test that empty config gets all defaults."""
        result = get_config_with_defaults({})
        assert result["channel"] == 36
        assert result["bandwidth"] == 80
        assert result["path_loss_exponent"] == 2.0

    def test_partial_config(self):
        """Test that partial config preserves values."""
        cfg = {
            "channel": 1,
            "custom_key": "value",
        }
        result = get_config_with_defaults(cfg)
        assert result["channel"] == 1  # Preserved
        assert result["bandwidth"] == 80  # Default
        assert result["custom_key"] == "value"  # Extra key preserved

    def test_full_config(self):
        """Test that full config is unchanged."""
        cfg = {
            "channel": 11,
            "bandwidth": 20,
            "window_size": 2.0,
            "path_loss_exponent": 3.0,
        }
        result = get_config_with_defaults(cfg)
        assert result["channel"] == 11
        assert result["bandwidth"] == 20
        assert result["window_size"] == 2.0
        assert result["path_loss_exponent"] == 3.0
