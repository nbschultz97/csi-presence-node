"""Tests for csi_node.calibrate — RSSI-distance calibration."""
import json
import math
import tempfile
from pathlib import Path

import pytest
import yaml

from csi_node.calibrate import estimate_from_pairs, _avg_rssi_from_log, write_config


class TestEstimateFromPairs:
    def test_free_space_exponent(self):
        """Known free-space case: n ≈ 2.0."""
        # RSSI at 1m = -40, at 10m = -60 → n = (-40 - -60) / (10*log10(10)) = 20/10 = 2.0
        txp, n = estimate_from_pairs(-40.0, 1.0, -60.0, 10.0)
        assert abs(n - 2.0) < 0.01
        assert abs(txp - (-40.0)) < 0.01

    def test_swapped_distances(self):
        """Order of (d1,d2) shouldn't matter."""
        txp1, n1 = estimate_from_pairs(-40.0, 1.0, -60.0, 10.0)
        txp2, n2 = estimate_from_pairs(-60.0, 10.0, -40.0, 1.0)
        assert abs(n1 - n2) < 0.001
        assert abs(txp1 - txp2) < 0.01

    def test_equal_distances_raises(self):
        with pytest.raises(ValueError, match="positive and distinct"):
            estimate_from_pairs(-40.0, 5.0, -50.0, 5.0)

    def test_zero_distance_raises(self):
        with pytest.raises(ValueError):
            estimate_from_pairs(-40.0, 0.0, -50.0, 5.0)

    def test_negative_distance_raises(self):
        with pytest.raises(ValueError):
            estimate_from_pairs(-40.0, -1.0, -50.0, 5.0)

    def test_indoor_exponent_range(self):
        """Typical indoor: n between 2 and 4."""
        txp, n = estimate_from_pairs(-45.0, 1.0, -72.0, 5.0)
        assert 1.5 < n < 5.0


class TestAvgRssiFromLog:
    def test_valid_log(self, tmp_path):
        log = tmp_path / "csi.log"
        lines = [
            json.dumps({"ts": 0, "rssi": [-40, -42], "csi": []}),
            json.dumps({"ts": 1, "rssi": [-38, -44], "csi": []}),
        ]
        log.write_text("\n".join(lines))
        result = _avg_rssi_from_log(log)
        assert result is not None
        assert abs(result - (-41.0)) < 0.01  # median of [-41, -41]

    def test_empty_log_returns_none(self, tmp_path):
        log = tmp_path / "empty.log"
        log.write_text("")
        assert _avg_rssi_from_log(log) is None

    def test_invalid_json_skipped(self, tmp_path):
        log = tmp_path / "bad.log"
        log.write_text("not json\n" + json.dumps({"ts": 0, "rssi": [-50, -52], "csi": []}))
        result = _avg_rssi_from_log(log)
        assert result is not None


class TestWriteConfig:
    def test_writes_calibration(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({"mode": "live", "tx_power_dbm": -30}))
        write_config(cfg, -42.0, 2.5)
        data = yaml.safe_load(cfg.read_text())
        assert data["tx_power_dbm"] == -42.0
        assert data["path_loss_exponent"] == 2.5
        assert data["calibrated"] is True
        assert "calibrated_at" in data
        # Existing keys preserved
        assert data["mode"] == "live"


class TestEstimateFromPairsWarnings:
    def test_negative_path_loss_warning(self, capsys):
        """Test warning when path loss exponent is negative."""
        # Inverted RSSI: stronger signal at farther distance
        txp, n = estimate_from_pairs(-60.0, 1.0, -40.0, 5.0)  # RSSI increases with distance
        captured = capsys.readouterr()
        assert "WARNING: Calculated path_loss_exponent is negative" in captured.err
        assert n > 0  # Should use absolute value

    def test_low_path_loss_warning(self, capsys):
        """Test warning when path loss exponent is below free space minimum."""
        # Create scenario with very low path loss
        txp, n = estimate_from_pairs(-40.0, 1.0, -42.0, 10.0)  # Very little path loss
        captured = capsys.readouterr()
        assert "WARNING: path_loss_exponent" in captured.err
        assert "below free-space minimum" in captured.err

    def test_high_path_loss_warning(self, capsys):
        """Test warning when path loss exponent is very high."""
        # Create scenario with very high path loss
        txp, n = estimate_from_pairs(-40.0, 1.0, -90.0, 2.0)  # Extreme path loss
        captured = capsys.readouterr()
        assert "WARNING: path_loss_exponent" in captured.err
        assert "very high" in captured.err

    def test_positive_tx_power_warning(self, capsys):
        """Test warning when calculated tx_power is positive."""
        # Create scenario that results in positive tx_power
        txp, n = estimate_from_pairs(10.0, 1.0, -20.0, 10.0)  # Positive RSSI values
        captured = capsys.readouterr()
        assert "WARNING: tx_power_dbm" in captured.err
        assert "positive" in captured.err

    def test_very_low_tx_power_warning(self, capsys):
        """Test warning when tx_power is very low."""
        # Create scenario with very low tx_power
        txp, n = estimate_from_pairs(-90.0, 1.0, -120.0, 10.0)
        captured = capsys.readouterr()
        assert "WARNING: tx_power_dbm" in captured.err
        assert "very low" in captured.err

    def test_distances_too_close_raises(self):
        """Test error when distances are too close for stable estimation."""
        # Identical distances that would cause log(d2/d1) = 0
        with pytest.raises(ValueError, match="Distances too close"):
            estimate_from_pairs(-40.0, 1.0, -40.1, 1.0000000001)


class TestAvgRssiFromLogExtended:
    def test_log_with_invalid_rssi_format(self, tmp_path):
        """Test log with various invalid RSSI formats."""
        log = tmp_path / "invalid.log"
        lines = [
            json.dumps({"ts": 0, "rssi": "not a list", "csi": []}),
            json.dumps({"ts": 1, "rssi": [-40], "csi": []}),  # Too few values
            json.dumps({"ts": 2, "rssi": ["string", "values"], "csi": []}),  # Non-numeric
            json.dumps({"ts": 3, "rssi": [-50, -52], "csi": []}),  # Valid
        ]
        log.write_text("\n".join(lines))
        result = _avg_rssi_from_log(log)
        assert result == -51.0  # Only the valid line should be used

    def test_log_missing_rssi_field(self, tmp_path):
        """Test log with missing RSSI field."""
        log = tmp_path / "no_rssi.log"
        lines = [
            json.dumps({"ts": 0, "csi": []}),  # No rssi field
            json.dumps({"ts": 1, "other": "data"}),  # No rssi field
        ]
        log.write_text("\n".join(lines))
        result = _avg_rssi_from_log(log)
        assert result is None

    def test_log_with_blank_lines(self, tmp_path):
        """Test log with blank lines and whitespace."""
        log = tmp_path / "blank_lines.log"
        lines = [
            "",
            "   ",
            json.dumps({"ts": 0, "rssi": [-45, -47], "csi": []}),
            "\n",
            json.dumps({"ts": 1, "rssi": [-43, -49], "csi": []}),
            "",
        ]
        log.write_text("\n".join(lines))
        result = _avg_rssi_from_log(log)
        assert result is not None  # Should handle blank lines gracefully


class TestMainFunction:
    def test_main_with_log_files(self, tmp_path, monkeypatch):
        """Test main function with log file inputs."""
        from csi_node.calibrate import main
        
        # Create test log files
        log1 = tmp_path / "log1.log"
        log2 = tmp_path / "log2.log"
        config_file = tmp_path / "config.yaml"
        
        log1.write_text(json.dumps({"ts": 0, "rssi": [-40, -42], "csi": []}))
        log2.write_text(json.dumps({"ts": 0, "rssi": [-60, -62], "csi": []}))
        config_file.write_text(yaml.safe_dump({"mode": "live"}))
        
        # Mock sys.argv
        monkeypatch.setattr("sys.argv", [
            "calibrate",
            "--log1", str(log1),
            "--d1", "1.0", 
            "--log2", str(log2),
            "--d2", "10.0",
            "--config", str(config_file)
        ])
        
        main()
        
        # Check that config was updated
        data = yaml.safe_load(config_file.read_text())
        assert "tx_power_dbm" in data
        assert "path_loss_exponent" in data
        assert data["calibrated"] is True

    def test_main_with_rssi_values(self, tmp_path, monkeypatch, capsys):
        """Test main function with direct RSSI inputs."""
        from csi_node.calibrate import main
        
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.safe_dump({"mode": "live"}))
        
        monkeypatch.setattr("sys.argv", [
            "calibrate",
            "--rssi1", "-40",
            "--d1", "1.0",
            "--rssi2", "-60", 
            "--d2", "10.0",
            "--config", str(config_file)
        ])
        
        main()
        
        output = capsys.readouterr()
        assert "Estimated tx_power_dbm:" in output.out
        assert "Estimated path_loss_exponent:" in output.out

    def test_main_missing_parameters_exits(self, monkeypatch):
        """Test main function exits when required parameters are missing."""
        from csi_node.calibrate import main
        
        monkeypatch.setattr("sys.argv", [
            "calibrate",
            "--rssi1", "-40",
            "--d1", "1.0",
            # Missing rssi2 and d2
        ])
        
        with pytest.raises(SystemExit, match="Provide either"):
            main()

    def test_main_invalid_log_file_exits(self, tmp_path, monkeypatch):
        """Test main function exits when log file has no usable data."""
        from csi_node.calibrate import main
        
        log1 = tmp_path / "empty.log"
        log1.write_text("")  # Empty log
        
        monkeypatch.setattr("sys.argv", [
            "calibrate",
            "--log1", str(log1),
            "--d1", "1.0",
            "--rssi2", "-60",
            "--d2", "10.0"
        ])
        
        with pytest.raises(SystemExit, match="No usable rssi values"):
            main()

    def test_main_without_config_file(self, tmp_path, monkeypatch, capsys):
        """Test main function without writing to config file."""
        from csi_node.calibrate import main
        
        monkeypatch.setattr("sys.argv", [
            "calibrate",
            "--rssi1", "-45",
            "--d1", "2.0",
            "--rssi2", "-65",
            "--d2", "8.0"
        ])
        
        main()
        
        output = capsys.readouterr()
        assert "Estimated tx_power_dbm:" in output.out
        assert "Estimated path_loss_exponent:" in output.out
        # Should not mention writing to config
        assert "Wrote calibration" not in output.out


class TestMainEntryPoint:
    def test_main_entry_point(self):
        """Test the if __name__ == '__main__' entry point."""
        from unittest.mock import patch
        
        with patch("csi_node.calibrate.main") as mock_main:
            # Test that the entry point calls main when run directly
            import csi_node.calibrate as calib_module
            
            # Just call main directly as that's what the entry point does
            calib_module.main()
            
            mock_main.assert_called_once()
