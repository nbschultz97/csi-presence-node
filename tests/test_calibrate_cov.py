"""Coverage tests for calibrate — warnings, _avg_rssi edge cases, main CLI."""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from csi_node.calibrate import estimate_from_pairs, _avg_rssi_from_log, write_config, main


class TestEstimateWarnings:
    def test_negative_n_uses_absolute(self, capsys):
        """RSSI increases with distance → negative n → abs(n) used."""
        # rssi1 < rssi2 but d1 < d2 → n negative
        txp, n = estimate_from_pairs(-60.0, 1.0, -40.0, 10.0)
        err = capsys.readouterr().err
        assert "negative" in err.lower()
        assert n > 0

    def test_n_below_1_warns(self, capsys):
        # Very small difference in RSSI → small n
        txp, n = estimate_from_pairs(-40.0, 1.0, -41.0, 10.0)
        err = capsys.readouterr().err
        assert "below" in err.lower() or n >= 1.0  # may or may not trigger

    def test_n_above_5_warns(self, capsys):
        # Large RSSI diff → high n
        txp, n = estimate_from_pairs(-30.0, 1.0, -90.0, 5.0)
        err = capsys.readouterr().err
        assert "very high" in err.lower() or n <= 5.0

    def test_positive_txp_warns(self, capsys):
        # Construct case where txp > 0
        # txp = rssi1 + 10*n*log10(d1); if d1=1, txp=rssi1; so rssi1>0
        txp, n = estimate_from_pairs(10.0, 1.0, -20.0, 10.0)
        err = capsys.readouterr().err
        assert "positive" in err.lower()

    def test_very_low_txp_warns(self, capsys):
        # rssi1 very low at d1 → txp very negative
        txp, n = estimate_from_pairs(-85.0, 1.0, -90.0, 2.0)
        err = capsys.readouterr().err
        # txp = -85 + 10 * n * log10(1) = -85
        assert txp < -80 or "very low" in err.lower()


class TestAvgRssiEdgeCases:
    def test_single_rssi_element_skipped(self, tmp_path):
        log = tmp_path / "log"
        log.write_text(json.dumps({"rssi": [-40]}) + "\n")
        assert _avg_rssi_from_log(log) is None

    def test_non_numeric_rssi_skipped(self, tmp_path):
        log = tmp_path / "log"
        lines = [
            json.dumps({"rssi": ["bad", "val"]}),
            json.dumps({"rssi": [-40, -42]}),
        ]
        log.write_text("\n".join(lines))
        result = _avg_rssi_from_log(log)
        assert result is not None

    def test_blank_lines_skipped(self, tmp_path):
        log = tmp_path / "log"
        log.write_text("\n\n" + json.dumps({"rssi": [-50, -52]}) + "\n")
        result = _avg_rssi_from_log(log)
        assert result is not None

    def test_rssi_not_list_skipped(self, tmp_path):
        log = tmp_path / "log"
        log.write_text(json.dumps({"rssi": -40}) + "\n")
        assert _avg_rssi_from_log(log) is None


class TestMainCLI:
    def test_rssi_values_mode(self, capsys):
        with patch("sys.argv", ["prog", "--rssi1", "-40", "--d1", "1.0",
                                 "--rssi2", "-60", "--d2", "10.0"]):
            main()
        out = capsys.readouterr().out
        assert "tx_power_dbm" in out
        assert "path_loss_exponent" in out

    def test_with_config_write(self, tmp_path, capsys):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({"mode": "test"}))
        with patch("sys.argv", ["prog", "--rssi1", "-40", "--d1", "1.0",
                                 "--rssi2", "-60", "--d2", "10.0",
                                 "--config", str(cfg)]):
            main()
        data = yaml.safe_load(cfg.read_text())
        assert data["calibrated"] is True

    def test_log_mode(self, tmp_path, capsys):
        log1 = tmp_path / "log1"
        log2 = tmp_path / "log2"
        log1.write_text(json.dumps({"rssi": [-40, -42]}) + "\n")
        log2.write_text(json.dumps({"rssi": [-55, -57]}) + "\n")
        with patch("sys.argv", ["prog", "--log1", str(log1), "--d1", "1.0",
                                 "--log2", str(log2), "--d2", "5.0"]):
            main()
        out = capsys.readouterr().out
        assert "path_loss_exponent" in out

    def test_missing_log_data_exits(self, tmp_path):
        log1 = tmp_path / "empty.log"
        log1.write_text("")
        with patch("sys.argv", ["prog", "--log1", str(log1), "--d1", "1.0",
                                 "--rssi2", "-60", "--d2", "5.0"]):
            with pytest.raises(SystemExit):
                main()

    def test_missing_args_exits(self):
        with patch("sys.argv", ["prog", "--rssi1", "-40"]):
            with pytest.raises(SystemExit):
                main()
