"""Tests for uncovered atak.py code — send_presence, send_sensor_position, send_presence_cot, close."""
from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
import socket

from csi_node.atak import (
    ATAKStreamer,
    SensorPosition,
    DetectedTarget,
    send_presence_cot,
)


class TestATAKStreamerSendPresence(unittest.TestCase):
    """Cover ATAKStreamer.send_presence (lines 410-460)."""

    def _make_streamer(self):
        with patch("socket.socket"):
            s = ATAKStreamer(
                sensor_uid="test-001",
                sensor_callsign="TEST",
                sensor_lat=40.0,
                sensor_lon=-111.0,
                broadcast_addr="127.0.0.1",
                port=14999,
            )
        return s

    def test_send_presence_detected(self):
        s = self._make_streamer()
        s._socket = MagicMock()
        result = s.send_presence(presence=True, direction="left", distance_m=3.0, confidence=0.9, pose="standing")
        self.assertTrue(result)
        s._socket.sendto.assert_called_once()

    def test_send_presence_not_detected(self):
        s = self._make_streamer()
        s._socket = MagicMock()
        result = s.send_presence(presence=False)
        self.assertTrue(result)

    def test_send_presence_target_uid_lifecycle(self):
        """UID should be created on presence start and cleared on absence."""
        s = self._make_streamer()
        s._socket = MagicMock()

        # No presence initially
        self.assertFalse(s._last_presence)

        # Presence starts — new UID
        s.send_presence(presence=True, direction="center", distance_m=5.0, confidence=0.8)
        uid1 = s._current_target_uid
        self.assertIsNotNone(uid1)

        # Presence continues — same UID
        s.send_presence(presence=True, direction="right", distance_m=4.0, confidence=0.7)
        self.assertEqual(s._current_target_uid, uid1)

        # Presence ends
        s.send_presence(presence=False)
        self.assertIsNone(s._current_target_uid)

        # New presence — new UID
        s.send_presence(presence=True, direction="left", distance_m=6.0, confidence=0.6)
        self.assertNotEqual(s._current_target_uid, uid1)

    def test_send_sensor_position(self):
        s = self._make_streamer()
        s._socket = MagicMock()
        result = s.send_sensor_position(status="standby")
        self.assertTrue(result)

    def test_send_raw_failure(self):
        s = self._make_streamer()
        s._socket = MagicMock()
        s._socket.sendto.side_effect = socket.error("fail")
        result = s.send_raw("<xml/>")
        self.assertFalse(result)

    def test_close(self):
        s = self._make_streamer()
        s._socket = MagicMock()
        s.close()
        s._socket.close.assert_called_once()

    def test_close_exception(self):
        s = self._make_streamer()
        s._socket = MagicMock()
        s._socket.close.side_effect = Exception("already closed")
        s.close()  # Should not raise

    def test_update_position(self):
        s = self._make_streamer()
        s.update_position(lat=41.0, lon=-112.0, hae=1500.0, heading=90.0)
        self.assertEqual(s.sensor_pos.lat, 41.0)
        self.assertEqual(s.sensor_pos.lon, -112.0)
        self.assertEqual(s.sensor_heading, 90.0)

    def test_is_multicast(self):
        self.assertTrue(ATAKStreamer._is_multicast("239.2.3.1"))
        self.assertTrue(ATAKStreamer._is_multicast("224.0.0.1"))
        self.assertFalse(ATAKStreamer._is_multicast("192.168.1.1"))
        self.assertFalse(ATAKStreamer._is_multicast("invalid"))
        self.assertFalse(ATAKStreamer._is_multicast(""))


class TestSendPresenceCot(unittest.TestCase):
    """Cover the convenience send_presence_cot function (lines 501-520)."""

    @patch("csi_node.atak.ATAKStreamer")
    def test_send_presence_cot_success(self, MockStreamer):
        instance = MockStreamer.return_value
        instance.send_presence.return_value = True

        result = send_presence_cot(
            presence=True,
            direction="center",
            distance_m=5.0,
            confidence=0.85,
            pose="standing",
            sensor_lat=40.0,
            sensor_lon=-111.0,
            broadcast_addr="127.0.0.1",
            port=15000,
        )

        self.assertTrue(result)
        instance.send_presence.assert_called_once()
        instance.close.assert_called_once()

    @patch("csi_node.atak.ATAKStreamer")
    def test_send_presence_cot_failure_still_closes(self, MockStreamer):
        instance = MockStreamer.return_value
        instance.send_presence.side_effect = Exception("network error")

        with self.assertRaises(Exception):
            send_presence_cot(
                presence=False,
                direction="left",
                distance_m=10.0,
                confidence=0.3,
            )

        instance.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
