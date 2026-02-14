"""Tests for csi_node.feitcsi — frame conversion and stream wrappers."""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from csi_node.feitcsi import _frame_to_pkt, live_stream, ftm_stream


# ── _frame_to_pkt ─────────────────────────────────────────────────

class TestFrameToPkt:
    def test_basic_conversion(self):
        frame = SimpleNamespace(
            timestamp=1000.5,
            rssi=[-30.0, -35.0],
            csi=[1.0, 2.0, 3.0],
        )
        pkt = _frame_to_pkt(frame)
        assert pkt["ts"] == 1000.5
        assert pkt["rssi"] == [-30.0, -35.0]
        np.testing.assert_array_equal(pkt["csi"], [1.0, 2.0, 3.0])

    def test_missing_attrs_uses_defaults(self):
        frame = object()  # no attributes
        pkt = _frame_to_pkt(frame)
        assert isinstance(pkt["ts"], float)
        assert len(pkt["rssi"]) == 2
        assert np.isnan(pkt["rssi"][0])
        assert len(pkt["csi"]) == 0

    def test_csi_is_numpy_array(self):
        frame = SimpleNamespace(timestamp=0, rssi=[0, 0], csi=[1, 2])
        pkt = _frame_to_pkt(frame)
        assert isinstance(pkt["csi"], np.ndarray)
        assert pkt["csi"].dtype == float


# ── live_stream / ftm_stream ──────────────────────────────────────

class TestLiveStream:
    def test_raises_without_feitcsi(self):
        with pytest.raises(RuntimeError, match="CSIExtractor not available"):
            next(live_stream("wlan0"))


class TestFtmStream:
    def test_raises_without_feitcsi(self):
        with pytest.raises(RuntimeError, match="CSIExtractor not available"):
            next(ftm_stream("/fake/path.ftm"))
