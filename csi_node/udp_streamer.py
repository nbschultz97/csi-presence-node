"""UDP streaming module for broadcasting presence data to network clients.

Provides both raw JSON streaming and ATAK CoT integration.

Usage:
    from csi_node.udp_streamer import UDPStreamer

    streamer = UDPStreamer(
        host="239.2.3.1",  # Multicast address
        port=4243,         # JSON stream port
        atak_enabled=True,
        atak_port=4242,
    )

    # Stream a presence result
    streamer.send({
        "timestamp": "2024-01-01T00:00:00",
        "presence": True,
        "pose": "standing",
        "direction": "left",
        "distance_m": 2.5,
        "confidence": 0.85,
    })

    streamer.close()
"""

from __future__ import annotations
import socket
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import ATAK module for CoT integration
try:
    from . import atak
    ATAK_AVAILABLE = True
except ImportError:
    ATAK_AVAILABLE = False


@dataclass
class UDPConfig:
    """Configuration for UDP streaming."""
    enabled: bool = False
    host: str = "239.2.3.1"  # Default: ATAK multicast
    port: int = 4243         # JSON stream port (separate from ATAK)
    atak_enabled: bool = False
    atak_port: int = 4242    # Standard ATAK SA port
    sensor_uid: str = "vantage-001"
    sensor_callsign: str = "VANTAGE-1"
    sensor_lat: float = 0.0
    sensor_lon: float = 0.0
    sensor_heading: float = 0.0
    ttl: int = 1             # Multicast TTL


class UDPStreamer:
    """UDP streamer for broadcasting presence data.

    Supports two modes:
    1. Raw JSON streaming: Broadcasts the presence JSON to a configurable
       address/port for custom integrations.
    2. ATAK CoT streaming: Broadcasts Cursor-on-Target XML to ATAK clients
       for tactical map integration.

    Both modes can be enabled simultaneously on different ports.
    """

    def __init__(
        self,
        host: str = "239.2.3.1",
        port: int = 4243,
        atak_enabled: bool = False,
        atak_port: int = 4242,
        sensor_uid: str = "vantage-001",
        sensor_callsign: str = "VANTAGE-1",
        sensor_lat: float = 0.0,
        sensor_lon: float = 0.0,
        sensor_heading: float = 0.0,
        ttl: int = 1,
    ):
        """Initialize the UDP streamer.

        Args:
            host: Broadcast/multicast address for JSON stream
            port: Port for JSON stream
            atak_enabled: Enable ATAK CoT streaming
            atak_port: Port for ATAK CoT (default: 4242)
            sensor_uid: Unique identifier for sensor (used in ATAK)
            sensor_callsign: Display name for sensor (used in ATAK)
            sensor_lat: Sensor latitude (used in ATAK)
            sensor_lon: Sensor longitude (used in ATAK)
            sensor_heading: Sensor heading in degrees (used in ATAK)
            ttl: Multicast TTL
        """
        self.host = host
        self.port = port
        self.atak_enabled = atak_enabled and ATAK_AVAILABLE
        self.atak_port = atak_port
        self.sensor_uid = sensor_uid
        self.sensor_callsign = sensor_callsign
        self.sensor_lat = sensor_lat
        self.sensor_lon = sensor_lon
        self.sensor_heading = sensor_heading
        self.ttl = ttl

        # Statistics
        self._json_sent = 0
        self._atak_sent = 0
        self._errors = 0

        # Create JSON streaming socket
        self._json_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if self._is_multicast(host):
            self._json_socket.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_MULTICAST_TTL,
                ttl
            )

        # Create ATAK streamer if enabled
        self._atak_streamer: Optional[atak.ATAKStreamer] = None
        if self.atak_enabled:
            self._atak_streamer = atak.ATAKStreamer(
                sensor_uid=sensor_uid,
                sensor_callsign=sensor_callsign,
                sensor_lat=sensor_lat,
                sensor_lon=sensor_lon,
                sensor_heading=sensor_heading,
                broadcast_addr=host,
                port=atak_port,
                ttl=ttl,
            )

    @staticmethod
    def _is_multicast(addr: str) -> bool:
        """Check if address is in multicast range."""
        try:
            parts = [int(p) for p in addr.split(".")]
            return 224 <= parts[0] <= 239
        except (ValueError, IndexError):
            return False

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "UDPStreamer":
        """Create streamer from configuration dictionary.

        Expected config keys:
            udp_enabled: bool
            udp_host: str
            udp_port: int
            atak_enabled: bool
            atak_port: int
            sensor_uid: str
            sensor_callsign: str
            sensor_lat: float
            sensor_lon: float
            sensor_heading: float
        """
        return cls(
            host=cfg.get("udp_host", "239.2.3.1"),
            port=cfg.get("udp_port", 4243),
            atak_enabled=cfg.get("atak_enabled", False),
            atak_port=cfg.get("atak_port", 4242),
            sensor_uid=cfg.get("sensor_uid", "vantage-001"),
            sensor_callsign=cfg.get("sensor_callsign", "VANTAGE-1"),
            sensor_lat=cfg.get("sensor_lat", 0.0),
            sensor_lon=cfg.get("sensor_lon", 0.0),
            sensor_heading=cfg.get("sensor_heading", 0.0),
            ttl=cfg.get("udp_ttl", 1),
        )

    def update_position(
        self,
        lat: float,
        lon: float,
        heading: float = 0.0,
    ) -> None:
        """Update sensor position for ATAK integration.

        Args:
            lat: New latitude in degrees
            lon: New longitude in degrees
            heading: New heading in degrees
        """
        self.sensor_lat = lat
        self.sensor_lon = lon
        self.sensor_heading = heading
        if self._atak_streamer:
            self._atak_streamer.update_position(lat, lon, 0.0, heading)

    def send_json(self, data: Dict[str, Any]) -> bool:
        """Send raw JSON data over UDP.

        Args:
            data: Dictionary to serialize and send

        Returns:
            True if send succeeded
        """
        try:
            payload = json.dumps(data).encode("utf-8")
            self._json_socket.sendto(payload, (self.host, self.port))
            self._json_sent += 1
            return True
        except socket.error:
            self._errors += 1
            return False

    def send(self, entry: Dict[str, Any]) -> bool:
        """Send presence data via JSON and optionally ATAK.

        Args:
            entry: Presence detection entry with keys:
                - timestamp: ISO 8601 timestamp
                - presence: bool
                - pose: str
                - direction: str ("left", "right", "center")
                - distance_m: float
                - confidence: float

        Returns:
            True if at least one send succeeded
        """
        json_ok = self.send_json(entry)

        atak_ok = True
        if self._atak_streamer and self.atak_enabled:
            atak_ok = self._atak_streamer.send_presence(
                presence=entry.get("presence", False),
                direction=entry.get("direction", "center"),
                distance_m=entry.get("distance_m", 5.0),
                confidence=entry.get("confidence", 0.5),
                pose=entry.get("pose", "unknown"),
            )
            if atak_ok:
                self._atak_sent += 1
            else:
                self._errors += 1

        return json_ok or atak_ok

    def send_sensor_status(self, status: str = "active") -> bool:
        """Send sensor status update via ATAK.

        Args:
            status: Sensor status ("active", "standby", "offline")

        Returns:
            True if send succeeded (or ATAK disabled)
        """
        if not self._atak_streamer:
            return True
        return self._atak_streamer.send_sensor_position(status)

    def get_stats(self) -> Dict[str, int]:
        """Get streaming statistics.

        Returns:
            Dictionary with json_sent, atak_sent, and errors counts
        """
        return {
            "json_sent": self._json_sent,
            "atak_sent": self._atak_sent,
            "errors": self._errors,
        }

    def close(self) -> None:
        """Close all sockets and clean up."""
        try:
            self._json_socket.close()
        except Exception:
            pass
        if self._atak_streamer:
            self._atak_streamer.close()


def create_streamer_from_yaml(config_path: str) -> Optional[UDPStreamer]:
    """Create a UDPStreamer from a YAML config file if UDP is enabled.

    Args:
        config_path: Path to config.yaml

    Returns:
        UDPStreamer if udp_enabled is True, None otherwise
    """
    import yaml
    from pathlib import Path

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        return None

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if not cfg.get("udp_enabled", False):
        return None

    return UDPStreamer.from_config(cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UDP Streamer test")
    parser.add_argument("--host", type=str, default="239.2.3.1", help="Broadcast address")
    parser.add_argument("--port", type=int, default=4243, help="JSON port")
    parser.add_argument("--atak", action="store_true", help="Enable ATAK")
    parser.add_argument("--lat", type=float, default=40.7128, help="Sensor latitude")
    parser.add_argument("--lon", type=float, default=-74.0060, help="Sensor longitude")
    args = parser.parse_args()

    streamer = UDPStreamer(
        host=args.host,
        port=args.port,
        atak_enabled=args.atak,
        sensor_lat=args.lat,
        sensor_lon=args.lon,
    )

    # Send test data
    test_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "presence": True,
        "pose": "standing",
        "direction": "left",
        "distance_m": 2.5,
        "confidence": 0.85,
    }

    print(f"Sending to {args.host}:{args.port}")
    if streamer.send(test_entry):
        print("Send succeeded!")
    else:
        print("Send failed!")

    print(f"Stats: {streamer.get_stats()}")
    streamer.close()
