"""Tests for csi_node.udp_streamer — UDP/ATAK streaming."""
import socket
from unittest.mock import patch, MagicMock

import pytest

from csi_node.udp_streamer import UDPStreamer, UDPConfig


class TestUDPStreamerInit:
    def test_default_creation(self):
        s = UDPStreamer()
        assert s.host == "239.2.3.1"
        assert s.port == 4243
        assert s.atak_enabled is False
        s.close()

    def test_from_config(self):
        cfg = {
            "udp_host": "192.168.1.255",
            "udp_port": 5000,
            "atak_enabled": False,
            "sensor_uid": "test-001",
        }
        s = UDPStreamer.from_config(cfg)
        assert s.host == "192.168.1.255"
        assert s.port == 5000
        assert s.sensor_uid == "test-001"
        s.close()


class TestIsMulticast:
    def test_multicast_addresses(self):
        assert UDPStreamer._is_multicast("224.0.0.1") is True
        assert UDPStreamer._is_multicast("239.2.3.1") is True

    def test_non_multicast(self):
        assert UDPStreamer._is_multicast("192.168.1.1") is False
        assert UDPStreamer._is_multicast("10.0.0.1") is False

    def test_invalid_address(self):
        assert UDPStreamer._is_multicast("not-an-ip") is False


class TestSendJson:
    def test_send_increments_counter(self):
        s = UDPStreamer(host="127.0.0.1", port=19999)
        result = s.send_json({"test": True})
        assert result is True
        assert s.get_stats()["json_sent"] == 1
        s.close()

    def test_send_entry(self):
        s = UDPStreamer(host="127.0.0.1", port=19999)
        entry = {
            "timestamp": "2026-01-01T00:00:00",
            "presence": True,
            "pose": "standing",
            "direction": "left",
            "distance_m": 2.5,
            "confidence": 0.85,
        }
        assert s.send(entry) is True
        stats = s.get_stats()
        assert stats["json_sent"] == 1
        assert stats["errors"] == 0
        s.close()


class TestStats:
    def test_initial_stats_zero(self):
        s = UDPStreamer()
        stats = s.get_stats()
        assert stats == {"json_sent": 0, "atak_sent": 0, "errors": 0}
        s.close()


class TestUpdatePosition:
    def test_updates_coords(self):
        s = UDPStreamer()
        s.update_position(40.0, -111.0, 90.0)
        assert s.sensor_lat == 40.0
        assert s.sensor_lon == -111.0
        assert s.sensor_heading == 90.0
        s.close()


class TestUDPConfig:
    def test_defaults(self):
        cfg = UDPConfig()
        assert cfg.enabled is False
        assert cfg.host == "239.2.3.1"
        assert cfg.port == 4243
