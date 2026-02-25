"""Tests for replay.main() CLI entry point."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from csi_node.replay import main


def _make_log(tmp_path, n=5):
    log = tmp_path / "test.log"
    lines = []
    for i in range(n):
        lines.append(json.dumps({
            "ts": i * 0.1,
            "rssi": [-40, -42],
            "csi": np.random.randn(2, 4).tolist(),
        }))
    log.write_text("\n".join(lines) + "\n")
    return log


class TestReplayMain:
    def test_main_runs(self, tmp_path):
        log = _make_log(tmp_path)
        with patch.object(sys, "argv", ["replay", "--file", str(log), "--speed", "0"]):
            main()

    def test_main_with_speed(self, tmp_path):
        log = _make_log(tmp_path, n=3)
        with patch("time.sleep"):
            with patch.object(sys, "argv", ["replay", "--file", str(log), "--speed", "100"]):
                main()
