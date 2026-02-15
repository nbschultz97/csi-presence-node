"""Extended tests for csi_node.data_collector — collect_window, collect_sample."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from csi_node.data_collector import collect_window, collect_sample, merge_datasets


class TestCollectWindow:
    def _write_csi_log(self, path: Path, n_lines: int = 30, interval: float = 0.05):
        """Write fake CSI log lines for testing."""
        from csi_node import utils
        lines = []
        base_ts = time.time()
        for i in range(n_lines):
            ts = base_ts + i * interval
            # Format matching utils.parse_csi_line expectations
            # We'll write lines that parse_csi_line can handle
            csi_data = np.random.randn(2, 64).tolist()
            line = json.dumps({
                "timestamp": ts,
                "csi": csi_data,
                "rssi": [-40, -42],
            })
            lines.append(line + "\n")
        path.write_text("".join(lines))

    def test_missing_log_returns_none(self, tmp_path):
        result = collect_window(tmp_path / "nonexistent.log", window_size=1.0, timeout=0.1)
        assert result is None

    def test_empty_log_returns_none_on_timeout(self, tmp_path):
        log = tmp_path / "empty.log"
        log.write_text("")
        result = collect_window(log, window_size=1.0, timeout=0.15)
        assert result is None


class TestCollectSample:
    def test_returns_none_when_no_data(self, tmp_path):
        from unittest.mock import patch
        log = tmp_path / "empty.log"
        log.write_text("")
        # Mock collect_window to return None immediately (avoid timeout)
        with patch("csi_node.data_collector.collect_window", return_value=None):
            result = collect_sample(log, label=0, window_size=1.0)
        assert result is None

    def test_returns_features_when_data_exists(self, tmp_path):
        from unittest.mock import patch
        log = tmp_path / "data.log"
        log.write_text("")
        fake_csi = np.random.randn(20, 128)
        with patch("csi_node.data_collector.collect_window", return_value=fake_csi):
            result = collect_sample(log, label=1, window_size=1.0)
        assert result is not None
        assert result["label"] == 1
        assert result["features"] is not None
        assert "timestamp" in result


class TestMergeDatasetsEdgeCases:
    def test_single_file(self, tmp_path):
        f1 = tmp_path / "a.npz"
        out = tmp_path / "out.npz"
        np.savez(f1, X=np.ones((4, 3)), y=np.array([0, 1, 0, 1]))
        merge_datasets([f1], out)
        data = np.load(out)
        assert data["X"].shape == (4, 3)

    def test_output_in_new_subdir(self, tmp_path):
        f1 = tmp_path / "a.npz"
        np.savez(f1, X=np.ones((2, 3)), y=np.array([0, 1]))
        out = tmp_path / "sub" / "deep" / "merged.npz"
        merge_datasets([f1], out)
        assert out.exists()

    def test_label_distribution_preserved(self, tmp_path):
        f1 = tmp_path / "a.npz"
        f2 = tmp_path / "b.npz"
        np.savez(f1, X=np.ones((3, 5)), y=np.array([0, 0, 0]))
        np.savez(f2, X=np.zeros((2, 5)), y=np.array([1, 1]))
        out = tmp_path / "out.npz"
        merge_datasets([f1, f2], out)
        data = np.load(out)
        counts = np.bincount(data["y"].astype(int))
        assert counts[0] == 3
        assert counts[1] == 2
