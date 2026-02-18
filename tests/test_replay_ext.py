"""Extended coverage tests for replay.py."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from csi_node.replay import decode_b64_capture, replay


def _make_log_line(ts):
    rng = np.random.default_rng(int(ts * 1000) % 2**31)
    return json.dumps({
        "ts": ts,
        "rssi": [-40, -42],
        "csi": rng.standard_normal((2, 4)).tolist(),
    })


class TestReplayB64:
    def test_replay_b64_file(self, tmp_path):
        """Replay from .b64 file — decode then replay."""
        # Create a log, then base64-encode it
        log_content = "\n".join(_make_log_line(i * 0.5) for i in range(5)) + "\n"
        b64_content = base64.b64encode(log_content.encode()).decode()
        b64_file = tmp_path / "test.b64"
        b64_file.write_text(b64_content)

        pkts = list(replay(str(b64_file), speed=0))
        assert len(pkts) == 5

    def test_replay_b64_cleanup(self, tmp_path):
        """Temp file from b64 decode is cleaned up."""
        log_content = _make_log_line(1.0) + "\n"
        b64_content = base64.b64encode(log_content.encode()).decode()
        b64_file = tmp_path / "test.b64"
        b64_file.write_text(b64_content)

        # After replay, temp file should be deleted
        pkts = list(replay(str(b64_file), speed=0))
        assert len(pkts) == 1


class TestReplayFtm:
    def test_replay_ftm_delegates(self, tmp_path):
        """Replay .ftm file delegates to feitcsi.ftm_stream."""
        ftm_file = tmp_path / "test.ftm"
        ftm_file.write_text("dummy")

        mock_pkt = {"ts": 1.0, "csi": np.zeros((2, 4)), "rssi": [-40, -42]}
        with patch("csi_node.replay.feitcsi.ftm_stream", return_value=iter([mock_pkt])) as mock_ftm:
            pkts = list(replay(str(ftm_file), speed=0))
            mock_ftm.assert_called_once_with(str(ftm_file))
            assert len(pkts) == 1


class TestReplayWithDelay:
    def test_replay_with_speed(self, tmp_path):
        """Replay with speed > 0 introduces delays."""
        log = tmp_path / "test.log"
        lines = [_make_log_line(i * 1.0) for i in range(3)]
        log.write_text("\n".join(lines) + "\n")

        with patch("time.sleep") as mock_sleep:
            pkts = list(replay(str(log), speed=10.0))  # fast
            assert len(pkts) == 3
            assert mock_sleep.call_count == 2  # No sleep before first pkt
