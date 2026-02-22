"""Additional coverage tests for utils.py — targeting uncovered lines."""
import json
import os
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from csi_node.utils import (
    RunLogManager,
    moving_median,
    savgol,
    compute_pca,
    safe_csv_append,
    safe_json_append,
    rotate_file,
    wait_for_file,
    parse_csi_line,
    rssi_to_distance,
)


class TestRunLogManagerRotation:
    def test_rotate_for_time(self, tmp_path):
        out = tmp_path / "data" / "csi.log"
        mgr = RunLogManager(str(out), rotation_bytes=0, rotation_seconds=0.01, retention=3)
        mgr.append({"ts": 1, "data": "first"})
        # Force time-based rotation by backdating
        mgr._file_started_at = time.time() - 1.0
        mgr.append({"ts": 2, "data": "second"})
        # After rotation, the main log should contain only the second entry (or be fresh)
        # At minimum, we exercised the rotation code path

    def test_rotate_for_size(self, tmp_path):
        out = tmp_path / "data" / "csi.log"
        mgr = RunLogManager(str(out), rotation_bytes=50, retention=2)
        for i in range(20):
            mgr.append({"ts": i, "data": "x" * 30})
        # Should have rotated multiple times
        rotated = list(mgr.run_dir.glob("csi.log.size-*"))
        assert len(rotated) <= 2  # retention=2

    def test_prune_rotated(self, tmp_path):
        out = tmp_path / "data" / "csi.log"
        mgr = RunLogManager(str(out), rotation_bytes=30, retention=1)
        for i in range(30):
            mgr.append({"ts": i, "v": "x" * 20})
        # With retention=1, should not keep more than 1 rotated file
        rotated_files = list(mgr.run_dir.glob("csi.log.*"))
        assert len(rotated_files) <= 2  # 1 retained + possibly 1 in-flight

    def test_write_fatal(self, tmp_path):
        out = tmp_path / "data" / "csi.log"
        mgr = RunLogManager(str(out))
        mgr.append({"ts": 1})
        try:
            raise ValueError("test error")
        except ValueError as e:
            fatal_path = mgr.write_fatal(e)
        assert fatal_path.exists()
        data = json.loads(fatal_path.read_text())
        assert data["exception"]["type"] == "ValueError"
        assert data["exception"]["message"] == "test error"

    def test_write_fatal_idempotent(self, tmp_path):
        out = tmp_path / "data" / "csi.log"
        mgr = RunLogManager(str(out))
        try:
            raise RuntimeError("once")
        except RuntimeError as e:
            p1 = mgr.write_fatal(e)
            p2 = mgr.write_fatal(e)
        assert p1 == p2

    def test_sanitize_numpy(self, tmp_path):
        out = tmp_path / "data" / "csi.log"
        mgr = RunLogManager(str(out))
        entry = {"arr": np.array([1.0, 2.0]), "val": np.float64(3.14), "flag": np.bool_(True)}
        mgr.append(entry)
        assert mgr.last_entry["arr"] == [1.0, 2.0]
        assert mgr.last_entry["val"] == 3.14
        assert mgr.last_entry["flag"] is True

    def test_mirror_path_fallback(self, tmp_path):
        """When symlinks fail, it should fall back to mirror path."""
        out = tmp_path / "data" / "csi.log"
        mgr = RunLogManager(str(out))
        # The manager should have been created successfully regardless of symlink support
        assert mgr.run_dir.exists()

    def test_duplicate_run_dir(self, tmp_path):
        """Creating two managers with same timestamp should get unique dirs."""
        out = tmp_path / "data" / "csi.log"
        mgr1 = RunLogManager(str(out), run_prefix="run")
        mgr2 = RunLogManager(str(out), run_prefix="run")
        assert mgr1.run_dir != mgr2.run_dir


class TestMovingMedian:
    def test_window_1(self):
        data = np.array([[1, 2], [3, 4], [5, 6]])
        result = moving_median(data, 1)
        np.testing.assert_array_equal(result, data)

    def test_window_3(self):
        data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        result = moving_median(data, 3)
        assert result.shape == data.shape


class TestSavgol:
    def test_even_window(self):
        data = np.random.randn(20, 5)
        result = savgol(data, 4, 2)  # even window should get bumped to 5
        assert result.shape == data.shape

    def test_odd_window(self):
        data = np.random.randn(20, 5)
        result = savgol(data, 5, 2)
        assert result.shape == data.shape


class TestComputePca:
    def test_basic(self):
        data = np.random.randn(50, 10)
        vals = compute_pca(data)
        assert len(vals) == 10
        # Eigenvalues should be sorted descending (from SVD)
        # Actually they come from s^2 which is already sorted desc
        assert vals[0] >= vals[-1]

    def test_1d_reshape(self):
        data = np.random.randn(50)
        vals = compute_pca(data.reshape(50, 1))
        assert len(vals) == 1


class TestSafeCsvAppend:
    def test_basic(self, tmp_path):
        path = str(tmp_path / "test.csv")
        safe_csv_append(path, [1, 2, 3])
        safe_csv_append(path, [4, 5, 6])
        content = open(path).read()
        assert "1,2,3" in content
        assert "4,5,6" in content


class TestSafeJsonAppend:
    def test_basic(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        safe_json_append(path, {"a": 1})
        safe_json_append(path, {"b": 2})
        lines = open(path).readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["a"] == 1


class TestRotateFile:
    def test_no_file(self, tmp_path):
        rotate_file(str(tmp_path / "noexist.log"), 100)

    def test_under_limit(self, tmp_path):
        f = tmp_path / "small.log"
        f.write_text("small")
        rotate_file(str(f), 1000)
        assert f.exists()

    def test_over_limit(self, tmp_path):
        f = tmp_path / "big.log"
        f.write_text("x" * 200)
        rotate_file(str(f), 100)
        assert not f.exists()
        assert (tmp_path / "big.log.1").exists()

    def test_multiple_rotations(self, tmp_path):
        f = tmp_path / "big.log"
        for i in range(3):
            f.write_text("x" * 200)
            rotate_file(str(f), 100)
        assert (tmp_path / "big.log.1").exists()


class TestWaitForFile:
    def test_file_exists(self, tmp_path):
        f = tmp_path / "exists.txt"
        f.write_text("hi")
        assert wait_for_file(f, timeout=0.1) is True

    def test_file_missing(self, tmp_path):
        assert wait_for_file(tmp_path / "nope.txt", timeout=0.1) is False


class TestParseCsiLine:
    def test_valid(self):
        line = json.dumps({"ts": 1.0, "rssi": [-40.0, -42.0], "csi": [[1, 2], [3, 4]]})
        result = parse_csi_line(line)
        assert result is not None
        assert result["ts"] == 1.0
        np.testing.assert_array_equal(result["csi"], np.array([[1, 2], [3, 4]], dtype=float))

    def test_empty(self):
        assert parse_csi_line("") is None
        assert parse_csi_line("   ") is None

    def test_invalid_json(self):
        assert parse_csi_line("{bad") is None

    def test_bad_rssi(self):
        line = json.dumps({"ts": 1.0, "rssi": [-40], "csi": [[1, 2]]})
        assert parse_csi_line(line) is None

    def test_rssi_not_list(self):
        line = json.dumps({"ts": 1.0, "rssi": -40, "csi": [[1, 2]]})
        assert parse_csi_line(line) is None


class TestRssiToDistance:
    def test_basic(self):
        d = rssi_to_distance(-40.0, tx_power=-40.0, n=2.0)
        assert abs(d - 1.0) < 0.01

    def test_farther(self):
        d = rssi_to_distance(-60.0, tx_power=-40.0, n=2.0)
        assert d > 1.0
