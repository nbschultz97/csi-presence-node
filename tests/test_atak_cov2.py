"""Tests for csi_node.atak — coverage for lines 70, 524-588."""
from __future__ import annotations
import unittest
from unittest.mock import patch, MagicMock
import math

from csi_node.atak import (
    SensorPosition, DetectedTarget, ATAKStreamer,
    _calculate_target_position, direction_to_bearing,
    generate_cot_event, generate_presence_cot, generate_sensor_cot,
    send_presence_cot,
)


class TestSensorPosition(unittest.TestCase):
    def test_defaults(self):
        sp = SensorPosition()
        self.assertEqual(sp.lat, 0.0)
        self.assertEqual(sp.lon, 0.0)
        self.assertEqual(sp.hae, 0.0)

    def test_custom(self):
        sp = SensorPosition(lat=40.0, lon=-74.0, hae=100.0)
        self.assertEqual(sp.hae, 100.0)


class TestDetectedTarget(unittest.TestCase):
    def test_defaults(self):
        dt = DetectedTarget()
        self.assertFalse(dt.presence)
        self.assertEqual(dt.direction, "center")
        self.assertTrue(dt.uid.startswith("target-"))

    def test_unique_uids(self):
        t1, t2 = DetectedTarget(), DetectedTarget()
        self.assertNotEqual(t1.uid, t2.uid)


class TestCalculateTargetPosition(unittest.TestCase):
    def test_zero_distance(self):
        lat, lon = _calculate_target_position(40.0, -74.0, 0.0, 0.0)
        self.assertAlmostEqual(lat, 40.0, places=5)
        self.assertAlmostEqual(lon, -74.0, places=5)

    def test_northward(self):
        lat, lon = _calculate_target_position(40.0, -74.0, 0.0, 100.0)
        self.assertGreater(lat, 40.0)
        self.assertAlmostEqual(lon, -74.0, places=3)

    def test_eastward(self):
        lat, lon = _calculate_target_position(40.0, -74.0, 90.0, 100.0)
        self.assertGreater(lon, -74.0)


class TestDirectionToBearing(unittest.TestCase):
    def test_center(self):
        self.assertEqual(direction_to_bearing("center"), 0.0)

    def test_left(self):
        self.assertEqual(direction_to_bearing("left"), 315.0)

    def test_right(self):
        self.assertEqual(direction_to_bearing("right"), 45.0)

    def test_with_heading(self):
        self.assertAlmostEqual(direction_to_bearing("center", 90.0), 90.0)


class TestGenerateCoT(unittest.TestCase):
    def test_generate_cot_event(self):
        xml = generate_cot_event("uid-1", "a-f-G", 40.0, -74.0, callsign="TEST")
        self.assertIn("uid-1", xml)
        self.assertIn("TEST", xml)
        self.assertIn("40.000000", xml)

    def test_generate_presence_cot(self):
        sp = SensorPosition(lat=40.0, lon=-74.0)
        target = DetectedTarget(presence=True, confidence=0.9, distance_m=5.0)
        xml = generate_presence_cot("sensor-1", "VANTAGE", sp, target)
        self.assertIn("DETECTED", xml)

    def test_generate_presence_cot_no_presence(self):
        sp = SensorPosition(lat=40.0, lon=-74.0)
        target = DetectedTarget(presence=False)
        xml = generate_presence_cot("sensor-1", "VANTAGE", sp, target)
        self.assertIn("CLEAR", xml)

    def test_generate_sensor_cot(self):
        sp = SensorPosition(lat=40.0, lon=-74.0)
        xml = generate_sensor_cot("sensor-1", "VANTAGE", sp, status="active")
        self.assertIn("ACTIVE", xml)
        self.assertIn("sensor-1", xml)


class TestATAKStreamer(unittest.TestCase):
    def test_init(self):
        streamer = ATAKStreamer(
            sensor_uid="test-001", sensor_callsign="TEST",
            sensor_lat=40.0, sensor_lon=-74.0,
        )
        self.assertEqual(streamer.sensor_uid, "test-001")
        self.assertEqual(streamer.port, 4242)
        streamer.close()

    def test_is_multicast(self):
        self.assertTrue(ATAKStreamer._is_multicast("239.2.3.1"))
        self.assertFalse(ATAKStreamer._is_multicast("192.168.1.1"))
        self.assertFalse(ATAKStreamer._is_multicast("invalid"))

    def test_update_position(self):
        streamer = ATAKStreamer("uid", "CS", sensor_lat=0, sensor_lon=0)
        streamer.update_position(lat=40.0, lon=-74.0, hae=100.0, heading=90.0)
        self.assertEqual(streamer.sensor_pos.lat, 40.0)
        self.assertEqual(streamer.sensor_heading, 90.0)
        streamer.close()

    @patch.object(ATAKStreamer, 'send_raw', return_value=True)
    def test_send_presence(self, mock_send):
        streamer = ATAKStreamer("uid", "CS", sensor_lat=40.0, sensor_lon=-74.0)
        result = streamer.send_presence(presence=True, direction="left", distance_m=3.0, confidence=0.8)
        self.assertTrue(result)
        mock_send.assert_called_once()
        streamer.close()

    @patch.object(ATAKStreamer, 'send_raw', return_value=True)
    def test_send_sensor_position(self, mock_send):
        streamer = ATAKStreamer("uid", "CS")
        result = streamer.send_sensor_position(status="standby")
        self.assertTrue(result)
        streamer.close()

    @patch.object(ATAKStreamer, 'send_raw', return_value=True)
    def test_send_presence_uid_continuity(self, mock_send):
        """Target UID stays the same while presence is active."""
        streamer = ATAKStreamer("uid", "CS")
        streamer.send_presence(presence=True, confidence=0.9)
        uid1 = streamer._current_target_uid
        streamer.send_presence(presence=True, confidence=0.8)
        self.assertEqual(streamer._current_target_uid, uid1)
        streamer.send_presence(presence=False)
        self.assertIsNone(streamer._current_target_uid)
        streamer.close()


class TestSendPresenceCoT(unittest.TestCase):
    @patch.object(ATAKStreamer, 'send_raw', return_value=True)
    def test_one_shot_send(self, mock_send):
        result = send_presence_cot(
            presence=True, direction="center", distance_m=5.0,
            confidence=0.8, sensor_lat=40.0, sensor_lon=-74.0,
        )
        self.assertTrue(result)


class TestMainDemo(unittest.TestCase):
    """Test the __main__ block (lines 524-588)."""

    def test_main_generates_xml(self):
        """Running as script should generate CoT XML."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "csi_node.atak", "--lat", "40.0", "--lon", "-74.0"],
            capture_output=True, text=True, timeout=10,
            cwd=str(__import__('pathlib').Path(__file__).resolve().parent.parent),
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Sensor CoT", result.stdout)
        self.assertIn("Presence CoT", result.stdout)

    def test_main_with_presence(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "csi_node.atak", "--presence", "--direction", "left", "--distance", "3.0"],
            capture_output=True, text=True, timeout=10,
            cwd=str(__import__('pathlib').Path(__file__).resolve().parent.parent),
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("DETECTED", result.stdout)


if __name__ == "__main__":
    unittest.main()
