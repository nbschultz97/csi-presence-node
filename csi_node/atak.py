"""ATAK Cursor-on-Target (CoT) integration for CSI presence streaming.

This module provides CoT XML generation and UDP streaming for ATAK integration.
CoT is the standard protocol used by TAK (Team Awareness Kit) products for
sharing situational awareness data.

Usage:
    from csi_node.atak import ATAKStreamer

    streamer = ATAKStreamer(
        sensor_uid="vantage-001",
        sensor_callsign="VANTAGE-1",
        sensor_lat=40.7128,
        sensor_lon=-74.0060,
        broadcast_addr="239.2.3.1",  # ATAK multicast
        port=4242
    )

    # Stream presence detection
    streamer.send_presence(
        presence=True,
        direction="left",
        distance_m=2.5,
        confidence=0.85,
        pose="standing"
    )
"""

from __future__ import annotations
import socket
import time
import math
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
from dataclasses import dataclass, field


# Standard ATAK CoT event types
COT_TYPE_SENSOR = "a-f-G-E-S"  # Friendly ground equipment sensor
COT_TYPE_HOSTILE = "a-h-G"     # Hostile ground (detected person)
COT_TYPE_UNKNOWN = "a-u-G"     # Unknown ground
COT_TYPE_NEUTRAL = "a-n-G"     # Neutral ground


@dataclass
class SensorPosition:
    """Geographic position of the sensor."""
    lat: float = 0.0
    lon: float = 0.0
    hae: float = 0.0  # Height above ellipsoid (meters)
    ce: float = 10.0  # Circular error (meters)
    le: float = 10.0  # Linear error (meters)


@dataclass
class DetectedTarget:
    """Represents a detected target for CoT generation."""
    presence: bool = False
    direction: str = "center"  # "left", "right", "center"
    distance_m: float = 0.0
    confidence: float = 0.0
    pose: str = "unknown"  # "standing", "crouching", "prone", "unknown"
    uid: str = field(default_factory=lambda: f"target-{uuid.uuid4().hex[:8]}")


def _iso_timestamp(dt: Optional[datetime] = None) -> str:
    """Generate ISO 8601 timestamp for CoT."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _calculate_target_position(
    sensor_lat: float,
    sensor_lon: float,
    bearing_deg: float,
    distance_m: float
) -> Tuple[float, float]:
    """Calculate target lat/lon from sensor position, bearing, and distance.

    Uses the Haversine formula inverse to project a point at given
    bearing and distance from the sensor.

    Args:
        sensor_lat: Sensor latitude in degrees
        sensor_lon: Sensor longitude in degrees
        bearing_deg: Bearing from sensor to target in degrees (0=N, 90=E)
        distance_m: Distance to target in meters

    Returns:
        Tuple of (target_lat, target_lon) in degrees
    """
    R = 6371000  # Earth radius in meters

    lat1 = math.radians(sensor_lat)
    lon1 = math.radians(sensor_lon)
    bearing = math.radians(bearing_deg)

    # Angular distance
    d = distance_m / R

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d) +
        math.cos(lat1) * math.sin(d) * math.cos(bearing)
    )

    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2)
    )

    return math.degrees(lat2), math.degrees(lon2)


def direction_to_bearing(direction: str, sensor_heading: float = 0.0) -> float:
    """Convert relative direction to absolute bearing.

    Args:
        direction: "left", "right", or "center"
        sensor_heading: Sensor's heading in degrees (0=N, 90=E)

    Returns:
        Absolute bearing in degrees
    """
    offsets = {
        "left": -45.0,
        "right": 45.0,
        "center": 0.0,
        "l": -45.0,
        "r": 45.0,
        "c": 0.0,
    }
    offset = offsets.get(direction.lower(), 0.0)
    return (sensor_heading + offset) % 360.0


def generate_cot_event(
    uid: str,
    event_type: str,
    lat: float,
    lon: float,
    hae: float = 0.0,
    ce: float = 10.0,
    le: float = 10.0,
    callsign: str = "",
    remarks: str = "",
    stale_seconds: float = 30.0,
    how: str = "m-g",  # machine-generated
) -> str:
    """Generate a CoT XML event string.

    Args:
        uid: Unique identifier for this event
        event_type: CoT event type (e.g., "a-f-G-E-S")
        lat: Latitude in degrees
        lon: Longitude in degrees
        hae: Height above ellipsoid in meters
        ce: Circular error in meters
        le: Linear error in meters
        callsign: Display name/callsign
        remarks: Additional remarks text
        stale_seconds: Seconds until event becomes stale
        how: How the event was generated (m-g = machine generated)

    Returns:
        CoT XML string
    """
    now = datetime.now(timezone.utc)
    time_str = _iso_timestamp(now)
    start_str = time_str
    stale = now + timedelta(seconds=stale_seconds)
    stale_str = _iso_timestamp(stale)

    # Build the CoT XML
    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<event version="2.0" uid="{uid}" type="{event_type}" time="{time_str}" start="{start_str}" stale="{stale_str}" how="{how}">
    <point lat="{lat:.6f}" lon="{lon:.6f}" hae="{hae:.1f}" ce="{ce:.1f}" le="{le:.1f}"/>
    <detail>
        <contact callsign="{callsign}"/>
        <remarks>{remarks}</remarks>
        <precisionlocation altsrc="DTED0"/>
    </detail>
</event>'''
    return xml


def generate_presence_cot(
    sensor_uid: str,
    sensor_callsign: str,
    sensor_pos: SensorPosition,
    target: DetectedTarget,
    sensor_heading: float = 0.0,
) -> str:
    """Generate CoT XML for a presence detection event.

    Args:
        sensor_uid: Unique ID of the sensor
        sensor_callsign: Display name of sensor
        sensor_pos: Sensor geographic position
        target: Detected target information
        sensor_heading: Sensor heading in degrees (0=N)

    Returns:
        CoT XML string for the detection
    """
    # Calculate target position from direction and distance
    bearing = direction_to_bearing(target.direction, sensor_heading)
    target_lat, target_lon = _calculate_target_position(
        sensor_pos.lat,
        sensor_pos.lon,
        bearing,
        target.distance_m if target.distance_m > 0 else 5.0  # Default 5m if unknown
    )

    # Determine event type based on presence
    event_type = COT_TYPE_UNKNOWN if target.presence else COT_TYPE_SENSOR

    # Build remarks with detection details
    remarks_parts = [
        f"Presence: {'DETECTED' if target.presence else 'CLEAR'}",
        f"Direction: {target.direction.upper()}",
        f"Distance: {target.distance_m:.1f}m",
        f"Confidence: {target.confidence:.0%}",
        f"Pose: {target.pose.upper()}",
        f"Source: {sensor_callsign}",
    ]
    remarks = " | ".join(remarks_parts)

    # Circular error increases with distance and decreases with confidence
    ce = max(2.0, target.distance_m * 0.3 * (1.0 - target.confidence * 0.5))

    return generate_cot_event(
        uid=target.uid,
        event_type=event_type,
        lat=target_lat,
        lon=target_lon,
        hae=sensor_pos.hae,
        ce=ce,
        le=ce,
        callsign=f"{sensor_callsign}-TGT",
        remarks=remarks,
        stale_seconds=10.0,  # Presence data goes stale quickly
        how="m-g",
    )


def generate_sensor_cot(
    sensor_uid: str,
    sensor_callsign: str,
    sensor_pos: SensorPosition,
    status: str = "active",
) -> str:
    """Generate CoT XML for the sensor itself (SA update).

    Args:
        sensor_uid: Unique ID of the sensor
        sensor_callsign: Display name of sensor
        sensor_pos: Sensor geographic position
        status: Sensor status ("active", "standby", "offline")

    Returns:
        CoT XML string for the sensor position
    """
    remarks = f"Vantage CSI Sensor | Status: {status.upper()}"

    return generate_cot_event(
        uid=sensor_uid,
        event_type=COT_TYPE_SENSOR,
        lat=sensor_pos.lat,
        lon=sensor_pos.lon,
        hae=sensor_pos.hae,
        ce=sensor_pos.ce,
        le=sensor_pos.le,
        callsign=sensor_callsign,
        remarks=remarks,
        stale_seconds=60.0,  # Sensor position can be stale longer
        how="m-g",
    )


class ATAKStreamer:
    """UDP streamer for sending CoT events to ATAK.

    Supports both unicast and multicast transmission. The default
    multicast address 239.2.3.1:4242 is the standard ATAK SA multicast.

    Example:
        streamer = ATAKStreamer(
            sensor_uid="vantage-001",
            sensor_callsign="VANTAGE-1",
            sensor_lat=40.7128,
            sensor_lon=-74.0060,
        )

        # Send presence detection
        streamer.send_presence(
            presence=True,
            direction="left",
            distance_m=2.5,
            confidence=0.85,
        )

        # Clean up
        streamer.close()
    """

    DEFAULT_MULTICAST = "239.2.3.1"
    DEFAULT_PORT = 4242

    def __init__(
        self,
        sensor_uid: str,
        sensor_callsign: str,
        sensor_lat: float = 0.0,
        sensor_lon: float = 0.0,
        sensor_hae: float = 0.0,
        sensor_heading: float = 0.0,
        broadcast_addr: str = DEFAULT_MULTICAST,
        port: int = DEFAULT_PORT,
        ttl: int = 1,
    ):
        """Initialize the ATAK streamer.

        Args:
            sensor_uid: Unique identifier for the sensor
            sensor_callsign: Display name for the sensor
            sensor_lat: Sensor latitude in degrees
            sensor_lon: Sensor longitude in degrees
            sensor_hae: Sensor height above ellipsoid in meters
            sensor_heading: Sensor heading in degrees (0=N, 90=E)
            broadcast_addr: IP address for UDP broadcast (default: ATAK multicast)
            port: UDP port (default: 4242)
            ttl: Multicast TTL (default: 1 for local network)
        """
        self.sensor_uid = sensor_uid
        self.sensor_callsign = sensor_callsign
        self.sensor_pos = SensorPosition(
            lat=sensor_lat,
            lon=sensor_lon,
            hae=sensor_hae,
        )
        self.sensor_heading = sensor_heading
        self.broadcast_addr = broadcast_addr
        self.port = port
        self.ttl = ttl

        # Track the current target UID for continuity
        self._current_target_uid: Optional[str] = None
        self._last_presence = False

        # Create UDP socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Configure for multicast if using multicast address
        if self._is_multicast(broadcast_addr):
            self._socket.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_MULTICAST_TTL,
                ttl
            )

    @staticmethod
    def _is_multicast(addr: str) -> bool:
        """Check if address is in multicast range (224.0.0.0 - 239.255.255.255)."""
        try:
            parts = [int(p) for p in addr.split(".")]
            return 224 <= parts[0] <= 239
        except (ValueError, IndexError):
            return False

    def update_position(
        self,
        lat: float,
        lon: float,
        hae: float = 0.0,
        heading: float = 0.0,
    ) -> None:
        """Update the sensor's position.

        Args:
            lat: New latitude in degrees
            lon: New longitude in degrees
            hae: New height above ellipsoid in meters
            heading: New heading in degrees
        """
        self.sensor_pos.lat = lat
        self.sensor_pos.lon = lon
        self.sensor_pos.hae = hae
        self.sensor_heading = heading

    def send_raw(self, xml: str) -> bool:
        """Send raw CoT XML over UDP.

        Args:
            xml: CoT XML string to send

        Returns:
            True if send succeeded, False otherwise
        """
        try:
            data = xml.encode("utf-8")
            self._socket.sendto(data, (self.broadcast_addr, self.port))
            return True
        except socket.error:
            return False

    def send_presence(
        self,
        presence: bool,
        direction: str = "center",
        distance_m: float = 5.0,
        confidence: float = 0.5,
        pose: str = "unknown",
    ) -> bool:
        """Send a presence detection event to ATAK.

        Args:
            presence: Whether a person is detected
            direction: Direction relative to sensor ("left", "right", "center")
            distance_m: Estimated distance in meters
            confidence: Detection confidence (0.0 - 1.0)
            pose: Detected pose ("standing", "crouching", "prone", "unknown")

        Returns:
            True if send succeeded, False otherwise
        """
        # Generate new target UID when presence starts, keep it while active
        if presence and not self._last_presence:
            self._current_target_uid = f"vantage-tgt-{uuid.uuid4().hex[:8]}"
        elif not presence:
            self._current_target_uid = None

        self._last_presence = presence

        target = DetectedTarget(
            presence=presence,
            direction=direction,
            distance_m=distance_m,
            confidence=confidence,
            pose=pose,
            uid=self._current_target_uid or f"vantage-tgt-{uuid.uuid4().hex[:8]}",
        )

        xml = generate_presence_cot(
            sensor_uid=self.sensor_uid,
            sensor_callsign=self.sensor_callsign,
            sensor_pos=self.sensor_pos,
            target=target,
            sensor_heading=self.sensor_heading,
        )

        return self.send_raw(xml)

    def send_sensor_position(self, status: str = "active") -> bool:
        """Send sensor position/status update to ATAK.

        Args:
            status: Sensor status ("active", "standby", "offline")

        Returns:
            True if send succeeded, False otherwise
        """
        xml = generate_sensor_cot(
            sensor_uid=self.sensor_uid,
            sensor_callsign=self.sensor_callsign,
            sensor_pos=self.sensor_pos,
            status=status,
        )
        return self.send_raw(xml)

    def close(self) -> None:
        """Close the UDP socket."""
        try:
            self._socket.close()
        except Exception:
            pass


# Convenience function for one-shot sending
def send_presence_cot(
    presence: bool,
    direction: str,
    distance_m: float,
    confidence: float,
    pose: str = "unknown",
    sensor_lat: float = 0.0,
    sensor_lon: float = 0.0,
    sensor_heading: float = 0.0,
    broadcast_addr: str = ATAKStreamer.DEFAULT_MULTICAST,
    port: int = ATAKStreamer.DEFAULT_PORT,
) -> bool:
    """One-shot function to send a presence CoT event.

    Creates a temporary streamer, sends the event, and cleans up.
    For repeated sending, use ATAKStreamer directly.

    Returns:
        True if send succeeded, False otherwise
    """
    streamer = ATAKStreamer(
        sensor_uid=f"vantage-{uuid.uuid4().hex[:6]}",
        sensor_callsign="VANTAGE",
        sensor_lat=sensor_lat,
        sensor_lon=sensor_lon,
        sensor_heading=sensor_heading,
        broadcast_addr=broadcast_addr,
        port=port,
    )
    try:
        return streamer.send_presence(
            presence=presence,
            direction=direction,
            distance_m=distance_m,
            confidence=confidence,
            pose=pose,
        )
    finally:
        streamer.close()


if __name__ == "__main__":
    # Demo: generate sample CoT XML
    import argparse

    parser = argparse.ArgumentParser(description="ATAK CoT generator demo")
    parser.add_argument("--lat", type=float, default=40.7128, help="Sensor latitude")
    parser.add_argument("--lon", type=float, default=-74.0060, help="Sensor longitude")
    parser.add_argument("--heading", type=float, default=0.0, help="Sensor heading (degrees)")
    parser.add_argument("--presence", action="store_true", help="Simulate presence detected")
    parser.add_argument("--direction", type=str, default="center", help="Direction (left/right/center)")
    parser.add_argument("--distance", type=float, default=5.0, help="Distance in meters")
    parser.add_argument("--send", action="store_true", help="Actually send via UDP")
    parser.add_argument("--addr", type=str, default="239.2.3.1", help="Broadcast address")
    parser.add_argument("--port", type=int, default=4242, help="UDP port")
    args = parser.parse_args()

    sensor_pos = SensorPosition(lat=args.lat, lon=args.lon)
    target = DetectedTarget(
        presence=args.presence,
        direction=args.direction,
        distance_m=args.distance,
        confidence=0.85,
        pose="standing",
    )

    # Generate sensor CoT
    sensor_xml = generate_sensor_cot(
        sensor_uid="vantage-demo-001",
        sensor_callsign="VANTAGE-DEMO",
        sensor_pos=sensor_pos,
    )
    print("=== Sensor CoT ===")
    print(sensor_xml)
    print()

    # Generate presence CoT
    presence_xml = generate_presence_cot(
        sensor_uid="vantage-demo-001",
        sensor_callsign="VANTAGE-DEMO",
        sensor_pos=sensor_pos,
        target=target,
        sensor_heading=args.heading,
    )
    print("=== Presence CoT ===")
    print(presence_xml)

    if args.send:
        print(f"\nSending to {args.addr}:{args.port}...")
        streamer = ATAKStreamer(
            sensor_uid="vantage-demo-001",
            sensor_callsign="VANTAGE-DEMO",
            sensor_lat=args.lat,
            sensor_lon=args.lon,
            sensor_heading=args.heading,
            broadcast_addr=args.addr,
            port=args.port,
        )
        streamer.send_sensor_position()
        streamer.send_presence(
            presence=args.presence,
            direction=args.direction,
            distance_m=args.distance,
            confidence=0.85,
            pose="standing",
        )
        streamer.close()
        print("Sent!")
