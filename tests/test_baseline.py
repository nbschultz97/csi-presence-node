"""Tests for csi_node.baseline — empty-room baseline recording."""
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from csi_node.baseline import _check_log_fresh, record, CAPTURE_EXIT_CODE


class TestCheckLogFresh:
    def test_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            _check_log_fresh(tmp_path / "nonexistent.log")
        assert exc_info.value.code == CAPTURE_EXIT_CODE

    def test_stale_file_exits(self, tmp_path):
        f = tmp_path / "old.log"
        f.write_text("data")
        # Make mtime old
        import os
        os.utime(f, (time.time() - 60, time.time() - 60))
        with pytest.raises(SystemExit):
            _check_log_fresh(f)

    def test_fresh_file_passes(self, tmp_path):
        f = tmp_path / "fresh.log"
        f.write_text("data")
        # Should not raise
        _check_log_fresh(f)


class TestRecord:
    def test_no_csi_raises(self, tmp_path):
        """If log has no valid CSI lines, should raise RuntimeError."""
        log = tmp_path / "csi.log"
        log.write_text("garbage\n")
        out = tmp_path / "baseline.npz"
        with pytest.raises((RuntimeError, SystemExit)):
            record(log, duration=0.1, outfile=out, wait=0.1)

    def test_missing_log_exits(self, tmp_path):
        out = tmp_path / "baseline.npz"
        with pytest.raises(SystemExit):
            record(tmp_path / "missing.log", duration=1.0, outfile=out, wait=0.1)

    def test_successful_baseline(self, tmp_path):
        """Write a log with valid CSI data, record should produce .npz."""
        log = tmp_path / "csi.log"
        lines = []
        for i in range(20):
            pkt = {"ts": float(i) * 0.05, "rssi": [-40, -42], "csi": [1.0, 2.0, 3.0]}
            lines.append(json.dumps(pkt))
        log.write_text("\n".join(lines) + "\n")

        out = tmp_path / "baseline.npz"
        record(log, duration=0.2, outfile=out, wait=0.1)
        assert out.exists()
        data = np.load(out)
        assert "mean" in data
        assert data["mean"].shape == (3,)
