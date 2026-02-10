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
