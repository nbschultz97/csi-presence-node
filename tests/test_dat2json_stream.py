"""Tests for scripts/dat2json_stream.py — binary .dat parsing and streaming."""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts/ to path so we can import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import dat2json_stream


# ── parse_frame ────────────────────────────────────────────────────

class TestParseFrame:
    def _make_frame(self, n_rx=2, n_sub=56):
        """Build a minimal valid frame payload.
        
        parse_frame expects needed = 2 * 2 * 56 * 2 = 448 bytes of IQ data
        (2 rx * 56 subcarriers, each with int8 real + int8 imag, but the outer
        loop is 2*56=112 and each entry reads 2 bytes => 224 bytes minimum for
        the magnitude extraction, but `needed` checks for 448).
        """
        header = b"\x00" * 18  # 18-byte header
        # needed = 2 * 2 * 56 * 2 = 448 bytes; loop reads 2*56 pairs of 2 bytes
        iq = b""
        for i in range(2 * 56):
            iq += struct.pack("<bb", 3, 4)  # real=3, imag=4 -> mag=5.0
        # Pad to meet the `needed` check (448 bytes)
        iq = iq.ljust(2 * 2 * 56 * 2, b"\x00")
        return header + iq

    def test_valid_frame(self):
        frame = self._make_frame()
        result = dat2json_stream.parse_frame(frame)
        assert result is not None
        assert len(result) == 2
        assert len(result[0]) == 56
        assert len(result[1]) == 56
        # Each mag should be sqrt(9+16) = 5.0
        assert abs(result[0][0] - 5.0) < 0.01

    def test_too_short_returns_none(self):
        assert dat2json_stream.parse_frame(b"\x00" * 10) is None

    def test_insufficient_csi_returns_none(self):
        # 18-byte header but no CSI data
        assert dat2json_stream.parse_frame(b"\x00" * 18) is None


# ── _rssi_from_csi ────────────────────────────────────────────────

class TestRssiFromCsi:
    def test_basic_rssi(self):
        csi = [[5.0] * 56, [5.0] * 56]
        rssi = dat2json_stream._rssi_from_csi(csi)
        assert len(rssi) == 2
        # 20*log10(5) ≈ 13.98
        assert abs(rssi[0] - 20 * math.log10(5.0)) < 0.01

    def test_zero_amplitude(self):
        csi = [[0.0] * 56, [0.0] * 56]
        rssi = dat2json_stream._rssi_from_csi(csi)
        assert len(rssi) == 2
        # Should not crash; uses eps

    def test_offset_env(self, monkeypatch):
        monkeypatch.setenv("DAT_RSSI_OFFSET", "-50")
        csi = [[5.0] * 56, [5.0] * 56]
        rssi = dat2json_stream._rssi_from_csi(csi)
        expected = 20 * math.log10(5.0) - 50
        assert abs(rssi[0] - expected) < 0.01

    def test_bad_input_returns_default(self):
        rssi = dat2json_stream._rssi_from_csi("garbage")
        assert rssi == [-40.0, -40.0]


# ── try_read ───────────────────────────────────────────────────────

class TestTryRead:
    def test_full_read(self, tmp_path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"\x01\x02\x03\x04")
        with open(p, "rb") as f:
            data = dat2json_stream.try_read(f, 4)
        assert data == b"\x01\x02\x03\x04"

    def test_short_read_rewinds(self, tmp_path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"\x01\x02")
        with open(p, "rb") as f:
            data = dat2json_stream.try_read(f, 4)
            assert len(data) < 4
            assert f.tell() == 0  # rewound


# ── stream (integration) ──────────────────────────────────────────

class TestStream:
    def _make_dat_file(self, path):
        """Write a single valid frame to a .dat file with length prefix."""
        header = b"\x00" * 18
        iq = b""
        for _ in range(2 * 56):
            iq += struct.pack("<bb", 3, 4)
        iq = iq.ljust(2 * 2 * 56 * 2, b"\x00")
        payload = header + iq
        with open(path, "wb") as f:
            f.write(struct.pack("<H", len(payload)))
            f.write(payload)

    def test_stream_produces_jsonl(self, tmp_path):
        dat_path = tmp_path / "input.dat"
        out_path = tmp_path / "output.log"
        self._make_dat_file(dat_path)

        # Patch time.sleep to avoid blocking, and make stream exit after one frame
        # by making the second try_read always return short data, then raising KeyboardInterrupt
        original_try_read = dat2json_stream.try_read
        call_count = 0

        def limited_try_read(f, n):
            nonlocal call_count
            call_count += 1
            if call_count > 3:  # after reading header + payload, stop
                raise KeyboardInterrupt
            return original_try_read(f, n)

        with patch.object(dat2json_stream, "try_read", side_effect=limited_try_read), \
             pytest.raises(KeyboardInterrupt):
            dat2json_stream.stream(str(dat_path), str(out_path))

        assert out_path.exists()
        lines = out_path.read_text().strip().split("\n")
        assert len(lines) >= 1
        pkt = json.loads(lines[0])
        assert "ts" in pkt
        assert "rssi" in pkt
        assert "csi" in pkt

    def test_missing_input_exits(self, tmp_path):
        """stream() should exit if input file doesn't appear within timeout."""
        with patch("time.sleep"), \
             patch("time.time", side_effect=[0, 0, 13]):  # simulate timeout
            with pytest.raises(SystemExit):
                dat2json_stream.stream(
                    str(tmp_path / "nope.dat"),
                    str(tmp_path / "out.log"),
                )
