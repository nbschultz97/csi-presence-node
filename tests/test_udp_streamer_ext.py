"""Extended coverage tests for udp_streamer.py."""
from __future__ import annotations

import socket
from unittest.mock import patch, MagicMock

import pytest

from csi_node.udp_streamer import UDPStreamer, UDPConfig, create_streamer_from_yaml


class TestUDPStreamerSendErrors:
    def test_send_json_socket_error(self):
        """send_json returns False and increments errors on socket failure."""
        s = UDPStreamer(host="127.0.0.1", port=19999)
        # Replace the socket with a mock
        mock_sock = MagicMock()
        mock_sock.sendto.side_effect = socket.error("fail")
        s._json_socket = mock_sock
        result = s.send_json({"test": True})
        assert result is False
        assert s.get_stats()["errors"] == 1
        s.close()

    def test_send_with_atak(self):
        """send() with ATAK enabled calls atak streamer."""
        s = UDPStreamer(host="127.0.0.1", port=19999)
        s.atak_enabled = True
        mock_atak = MagicMock()
        mock_atak.send_presence.return_value = True
        s._atak_streamer = mock_atak

        entry = {"presence": True, "direction": "left", "distance_m": 2.5,
                 "confidence": 0.85, "pose": "standing"}
        result = s.send(entry)
        assert result is True
        assert s.get_stats()["atak_sent"] == 1
        mock_atak.send_presence.assert_called_once()
        s.close()

    def test_send_atak_failure(self):
        """ATAK send failure increments errors."""
        s = UDPStreamer(host="127.0.0.1", port=19999)
        s.atak_enabled = True
        mock_atak = MagicMock()
        mock_atak.send_presence.return_value = False
        s._atak_streamer = mock_atak

        entry = {"presence": True, "direction": "left", "distance_m": 2.5,
                 "confidence": 0.85, "pose": "standing"}
        s.send(entry)
        assert s.get_stats()["errors"] == 1
        s.close()


class TestSendSensorStatus:
    def test_no_atak_streamer(self):
        s = UDPStreamer(host="127.0.0.1", port=19999)
        assert s.send_sensor_status("active") is True
        s.close()

    def test_with_atak_streamer(self):
        s = UDPStreamer(host="127.0.0.1", port=19999)
        mock_atak = MagicMock()
        mock_atak.send_sensor_position.return_value = True
        s._atak_streamer = mock_atak
        assert s.send_sensor_status("active") is True
        s.close()


class TestUpdatePositionWithAtak:
    def test_updates_atak_streamer(self):
        s = UDPStreamer(host="127.0.0.1", port=19999)
        mock_atak = MagicMock()
        s._atak_streamer = mock_atak
        s.update_position(40.0, -111.0, 90.0)
        mock_atak.update_position.assert_called_once_with(40.0, -111.0, 0.0, 90.0)
        s.close()


class TestFromConfigFull:
    def test_all_config_keys(self):
        cfg = {
            "udp_host": "10.0.0.1",
            "udp_port": 5555,
            "atak_enabled": False,
            "atak_port": 4242,
            "sensor_uid": "s1",
            "sensor_callsign": "SENSOR-1",
            "sensor_lat": 40.0,
            "sensor_lon": -111.0,
            "sensor_heading": 45.0,
            "udp_ttl": 2,
        }
        s = UDPStreamer.from_config(cfg)
        assert s.host == "10.0.0.1"
        assert s.port == 5555
        assert s.ttl == 2
        assert s.sensor_heading == 45.0
        s.close()


class TestNonMulticastInit:
    def test_unicast_no_multicast_opt(self):
        """Non-multicast address should not set multicast TTL."""
        s = UDPStreamer(host="192.168.1.1", port=5000)
        s.close()


class TestCreateStreamerFromYaml:
    def test_missing_file(self, tmp_path):
        result = create_streamer_from_yaml(str(tmp_path / "nope.yaml"))
        assert result is None

    def test_udp_disabled(self, tmp_path):
        import yaml
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"udp_enabled": False}))
        result = create_streamer_from_yaml(str(cfg_path))
        assert result is None

    def test_udp_enabled(self, tmp_path):
        import yaml
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({
            "udp_enabled": True,
            "udp_host": "127.0.0.1",
            "udp_port": 19997,
        }))
        result = create_streamer_from_yaml(str(cfg_path))
        assert result is not None
        result.close()


class TestClose:
    def test_close_handles_errors(self):
        s = UDPStreamer(host="127.0.0.1", port=19999)
        s._json_socket.close()  # Close once
        s.close()  # Close again — should not raise
