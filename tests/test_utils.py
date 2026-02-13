"""Tests for csi_node.utils — RunLogManager, math helpers, file I/O."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from csi_node.utils import (
    RunLogManager,
    compute_pca,
    moving_median,
    parse_csi_line,
    rotate_file,
    rssi_to_distance,
    safe_csv_append,
    safe_json_append,
    savgol,
)


# ── moving_median ──────────────────────────────────────────────────

class TestMovingMedian:
    def test_window_1_returns_identity(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = moving_median(data, 1)
        np.testing.assert_array_equal(result, data)

    def test_window_3(self):
        data = np.array([[1.0], [10.0], [3.0], [8.0], [2.0]])
        result = moving_median(data, 3)
        assert result.shape == data.shape
        # Last element: median of [3, 8, 2] = 3
        assert result[-1, 0] == 3.0

    def test_output_shape_matches_input(self):
        data = np.random.randn(20, 5)
        result = moving_median(data, 5)
        assert result.shape == data.shape


# ── savgol ─────────────────────────────────────────────────────────

class TestSavgol:
    def test_returns_same_shape(self):
        data = np.random.randn(50, 4)
        result = savgol(data, 7, 2)
        assert result.shape == data.shape

    def test_even_window_adjusted(self):
        """Even window should be bumped to odd internally."""
        data = np.random.randn(50, 4)
        result = savgol(data, 6, 2)
        assert result.shape == data.shape


# ── compute_pca ────────────────────────────────────────────────────

class TestComputePCA:
    def test_returns_eigenvalues_descending(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((30, 5))
        vals = compute_pca(data)
        assert len(vals) == 5
        assert all(vals[i] >= vals[i + 1] - 1e-10 for i in range(len(vals) - 1))

    def test_1d_input_reshaped(self):
        data = np.random.randn(10, 1)
        vals = compute_pca(data)
        assert vals.shape == (1,)


# ── safe_csv_append ────────────────────────────────────────────────

class TestSafeCsvAppend:
    def test_appends_row(self, tmp_path):
        p = str(tmp_path / "out.csv")
        safe_csv_append(p, [1, 2, "hello"])
        safe_csv_append(p, [3, 4, "world"])
        lines = Path(p).read_text().strip().splitlines()
        assert len(lines) == 2
        assert lines[0] == "1,2,hello"


# ── safe_json_append ──────────────────────────────────────────────

class TestSafeJsonAppend:
    def test_appends_json_lines(self, tmp_path):
        p = str(tmp_path / "out.jsonl")
        safe_json_append(p, {"a": 1})
        safe_json_append(p, {"b": 2})
        lines = Path(p).read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[1]) == {"b": 2}


# ── rotate_file ───────────────────────────────────────────────────

class TestRotateFile:
    def test_rotates_when_over_limit(self, tmp_path):
        p = tmp_path / "log.txt"
        p.write_text("x" * 200)
        rotate_file(str(p), 100)
        assert not p.exists()
        assert (tmp_path / "log.txt.1").exists()

    def test_no_rotate_when_under_limit(self, tmp_path):
        p = tmp_path / "log.txt"
        p.write_text("short")
        rotate_file(str(p), 1000)
        assert p.exists()

    def test_nonexistent_file_noop(self, tmp_path):
        rotate_file(str(tmp_path / "nope.txt"), 100)


# ── parse_csi_line ────────────────────────────────────────────────

class TestParseCsiLine:
    def test_valid_line(self):
        line = json.dumps({"ts": 1.5, "rssi": [-40, -42], "csi": [[1, 2], [3, 4]]})
        pkt = parse_csi_line(line)
        assert pkt is not None
        assert pkt["ts"] == 1.5
        assert pkt["rssi"] == [-40.0, -42.0]
        np.testing.assert_array_equal(pkt["csi"], np.array([[1, 2], [3, 4]], dtype=float))

    def test_empty_line(self):
        assert parse_csi_line("") is None
        assert parse_csi_line("   ") is None

    def test_invalid_json(self):
        assert parse_csi_line("{bad json}") is None

    def test_missing_rssi(self):
        line = json.dumps({"ts": 0, "csi": [[1, 2]]})
        assert parse_csi_line(line) is None

    def test_single_rssi_rejected(self):
        line = json.dumps({"ts": 0, "rssi": [-40], "csi": [[1, 2]]})
        assert parse_csi_line(line) is None


# ── rssi_to_distance ──────────────────────────────────────────────

class TestRssiToDistance:
    def test_at_tx_power_returns_1m(self):
        assert rssi_to_distance(-40.0, tx_power=-40.0) == pytest.approx(1.0)

    def test_weaker_signal_farther(self):
        d1 = rssi_to_distance(-50.0)
        d2 = rssi_to_distance(-60.0)
        assert d2 > d1 > 0


# ── RunLogManager ─────────────────────────────────────────────────

class TestRunLogManager:
    def test_creates_run_dir(self, tmp_path):
        out = tmp_path / "data" / "log.jsonl"
        mgr = RunLogManager(str(out))
        assert mgr.run_dir.exists()
        assert mgr.run_dir.parent.name == "runs"

    def test_append_writes_jsonl(self, tmp_path):
        out = tmp_path / "data" / "log.jsonl"
        mgr = RunLogManager(str(out))
        mgr.append({"val": 42})
        lines = mgr.log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["val"] == 42

    def test_sanitize_numpy(self, tmp_path):
        out = tmp_path / "data" / "log.jsonl"
        mgr = RunLogManager(str(out))
        entry = {"arr": np.array([1, 2, 3]), "f": np.float64(3.14), "b": np.bool_(True)}
        sanitized = mgr._sanitize(entry)
        assert sanitized["arr"] == [1, 2, 3]
        assert isinstance(sanitized["f"], float)
        assert sanitized["b"] is True

    def test_write_fatal(self, tmp_path):
        out = tmp_path / "data" / "log.jsonl"
        mgr = RunLogManager(str(out))
        try:
            raise ValueError("boom")
        except ValueError as exc:
            path = mgr.write_fatal(exc)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["exception"]["type"] == "ValueError"
        assert "boom" in data["exception"]["message"]

    def test_write_fatal_only_once(self, tmp_path):
        out = tmp_path / "data" / "log.jsonl"
        mgr = RunLogManager(str(out))
        exc = RuntimeError("x")
        mgr.write_fatal(exc)
        # Modify the fatal file to check it's NOT overwritten
        mgr.fatal_path.write_text("sentinel")
        mgr.write_fatal(exc)
        assert mgr.fatal_path.read_text() == "sentinel"

    def test_rotation_by_size(self, tmp_path):
        out = tmp_path / "data" / "log.jsonl"
        mgr = RunLogManager(str(out), rotation_bytes=100)
        # Write enough to trigger rotation
        for i in range(20):
            mgr.append({"i": i, "payload": "x" * 50})
        # At least one rotated file should exist
        rotated = list(mgr.run_dir.glob("*.size-*"))
        assert len(rotated) >= 1
