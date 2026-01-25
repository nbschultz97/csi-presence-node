"""Tests for ATAK CoT integration module."""

import pytest
import math
from csi_node.atak import (
    ATAKStreamer,
    SensorPosition,
    DetectedTarget,
    generate_cot_event,
    generate_presence_cot,
    generate_sensor_cot,
    direction_to_bearing,
    _calculate_target_position,
)


class TestDirectionToBearing:
    """Tests for direction_to_bearing function."""

    def test_center_no_heading(self):
        """Center direction with no heading should be 0."""
        assert direction_to_bearing("center", 0.0) == 0.0

    def test_left_no_heading(self):
        """Left direction with no heading should be 315 (-45)."""
        assert direction_to_bearing("left", 0.0) == 315.0

    def test_right_no_heading(self):
        """Right direction with no heading should be 45."""
        assert direction_to_bearing("right", 0.0) == 45.0

    def test_center_with_heading(self):
        """Center with 90 heading should be 90."""
        assert direction_to_bearing("center", 90.0) == 90.0

    def test_left_with_heading(self):
        """Left with 90 heading should be 45."""
        assert direction_to_bearing("left", 90.0) == 45.0

    def test_wraparound(self):
        """Test wraparound at 360 degrees."""
        bearing = direction_to_bearing("right", 350.0)
        assert 0 <= bearing < 360
        assert bearing == 35.0


class TestCalculateTargetPosition:
    """Tests for target position calculation."""

    def test_north_projection(self):
        """Projecting north should increase latitude."""
        sensor_lat, sensor_lon = 40.0, -74.0
        target_lat, target_lon = _calculate_target_position(
            sensor_lat, sensor_lon, bearing_deg=0.0, distance_m=1000
        )
        assert target_lat > sensor_lat
        assert abs(target_lon - sensor_lon) < 0.001

    def test_east_projection(self):
        """Projecting east should increase longitude."""
        sensor_lat, sensor_lon = 40.0, -74.0
        target_lat, target_lon = _calculate_target_position(
            sensor_lat, sensor_lon, bearing_deg=90.0, distance_m=1000
        )
        assert target_lon > sensor_lon
        assert abs(target_lat - sensor_lat) < 0.01

    def test_short_distance(self):
        """Short distance projection should be close to sensor."""
        sensor_lat, sensor_lon = 40.0, -74.0
        target_lat, target_lon = _calculate_target_position(
            sensor_lat, sensor_lon, bearing_deg=45.0, distance_m=10
        )
        # 10m is very close, should be within ~0.0001 degrees
        assert abs(target_lat - sensor_lat) < 0.001
        assert abs(target_lon - sensor_lon) < 0.001


class TestGenerateCoTEvent:
    """Tests for CoT XML generation."""

    def test_basic_event(self):
        """Generate a basic CoT event."""
        xml = generate_cot_event(
            uid="test-001",
            event_type="a-f-G-E-S",
            lat=40.7128,
            lon=-74.0060,
        )
        assert '<?xml version="1.0"' in xml
        assert 'uid="test-001"' in xml
        assert 'type="a-f-G-E-S"' in xml
        assert 'lat="40.712800"' in xml
        assert 'lon="-74.006000"' in xml

    def test_event_with_callsign(self):
        """Generate event with callsign."""
        xml = generate_cot_event(
            uid="test-002",
            event_type="a-f-G-E-S",
            lat=40.0,
            lon=-74.0,
            callsign="TEST-UNIT",
        )
        assert 'callsign="TEST-UNIT"' in xml

    def test_event_with_remarks(self):
        """Generate event with remarks."""
        xml = generate_cot_event(
            uid="test-003",
            event_type="a-f-G-E-S",
            lat=40.0,
            lon=-74.0,
            remarks="Test remarks here",
        )
        assert "<remarks>Test remarks here</remarks>" in xml


class TestGeneratePresenceCoT:
    """Tests for presence detection CoT generation."""

    def test_presence_detected(self):
        """Generate CoT for presence detection."""
        sensor_pos = SensorPosition(lat=40.7128, lon=-74.0060)
        target = DetectedTarget(
            presence=True,
            direction="left",
            distance_m=5.0,
            confidence=0.85,
            pose="standing",
        )
        xml = generate_presence_cot(
            sensor_uid="vantage-001",
            sensor_callsign="VANTAGE-1",
            sensor_pos=sensor_pos,
            target=target,
        )
        assert "DETECTED" in xml
        assert "LEFT" in xml
        assert "5.0m" in xml
        assert "85%" in xml
        assert "STANDING" in xml

    def test_no_presence(self):
        """Generate CoT when no presence detected."""
        sensor_pos = SensorPosition(lat=40.7128, lon=-74.0060)
        target = DetectedTarget(
            presence=False,
            direction="center",
            distance_m=0.0,
            confidence=0.1,
        )
        xml = generate_presence_cot(
            sensor_uid="vantage-001",
            sensor_callsign="VANTAGE-1",
            sensor_pos=sensor_pos,
            target=target,
        )
        assert "CLEAR" in xml


class TestGenerateSensorCoT:
    """Tests for sensor position CoT generation."""

    def test_sensor_position(self):
        """Generate sensor position CoT."""
        sensor_pos = SensorPosition(lat=40.7128, lon=-74.0060)
        xml = generate_sensor_cot(
            sensor_uid="vantage-001",
            sensor_callsign="VANTAGE-1",
            sensor_pos=sensor_pos,
            status="active",
        )
        assert 'uid="vantage-001"' in xml
        assert 'callsign="VANTAGE-1"' in xml
        assert "ACTIVE" in xml


class TestATAKStreamer:
    """Tests for ATAKStreamer class."""

    def test_init(self):
        """Test streamer initialization."""
        streamer = ATAKStreamer(
            sensor_uid="test-001",
            sensor_callsign="TEST",
            sensor_lat=40.0,
            sensor_lon=-74.0,
        )
        assert streamer.sensor_uid == "test-001"
        assert streamer.sensor_callsign == "TEST"
        streamer.close()

    def test_update_position(self):
        """Test position update."""
        streamer = ATAKStreamer(
            sensor_uid="test-001",
            sensor_callsign="TEST",
        )
        streamer.update_position(lat=41.0, lon=-75.0, heading=90.0)
        assert streamer.sensor_pos.lat == 41.0
        assert streamer.sensor_pos.lon == -75.0
        assert streamer.sensor_heading == 90.0
        streamer.close()

    def test_multicast_detection(self):
        """Test multicast address detection."""
        assert ATAKStreamer._is_multicast("239.2.3.1") is True
        assert ATAKStreamer._is_multicast("224.0.0.1") is True
        assert ATAKStreamer._is_multicast("192.168.1.1") is False
        assert ATAKStreamer._is_multicast("10.0.0.1") is False

    def test_send_presence_tracking(self):
        """Test that target UID is tracked across calls."""
        streamer = ATAKStreamer(
            sensor_uid="test-001",
            sensor_callsign="TEST",
            broadcast_addr="127.0.0.1",  # Use localhost to avoid actual send
        )
        # First presence should create new UID
        streamer.send_presence(presence=True, direction="left", distance_m=5.0, confidence=0.8)
        first_uid = streamer._current_target_uid
        assert first_uid is not None

        # Continued presence should keep same UID
        streamer.send_presence(presence=True, direction="left", distance_m=5.0, confidence=0.8)
        assert streamer._current_target_uid == first_uid

        # No presence should clear UID
        streamer.send_presence(presence=False, direction="center", distance_m=0.0, confidence=0.1)
        assert streamer._current_target_uid is None

        # New presence should get new UID
        streamer.send_presence(presence=True, direction="right", distance_m=3.0, confidence=0.9)
        assert streamer._current_target_uid != first_uid

        streamer.close()
