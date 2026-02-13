"""Tests for csi_node.replay — replay and decode_b64_capture."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from csi_node.replay import decode_b64_capture, replay


def _make_log_line(ts, rssi=(-40, -42), csi_shape=(2, 4)):
    rng = np.random.default_rng(int(ts * 1000) % 2**31)
    return json.dumps({
        "ts": ts,
        "rssi": list(rssi),
        "csi": rng.standard_normal(csi_shape).tolist(),
    })


class TestReplay:
    def test_replays_packets(self, tmp_path):
        log = tmp_path / "test.log"
        lines = [_make_log_line(i * 0.1) for i in range(5)]
        log.write_text("\n".join(lines) + "\n")

        pkts = list(replay(str(log), speed=0))  # speed=0 → no sleep
        assert len(pkts) == 5
        assert pkts[0]["ts"] == pytest.approx(0.0)

    def test_skips_invalid_lines(self, tmp_path):
        log = tmp_path / "test.log"
        log.write_text("not json\n" + _make_log_line(1.0) + "\n")
        pkts = list(replay(str(log), speed=0))
        assert len(pkts) == 1

    def test_empty_file(self, tmp_path):
        log = tmp_path / "empty.log"
        log.write_text("")
        assert list(replay(str(log), speed=0)) == []


class TestDecodeB64Capture:
    def test_decodes_to_temp_file(self, tmp_path):
        raw = b"hello world binary data"
        b64_file = tmp_path / "cap.b64"
        b64_file.write_text(base64.b64encode(raw).decode())

        result = decode_b64_capture(b64_file)
        try:
            assert result.exists()
            assert result.read_bytes() == raw
        finally:
            result.unlink(missing_ok=True)
