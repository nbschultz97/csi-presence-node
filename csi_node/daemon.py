"""Daemon mode for csi-presence-node.

Provides a long-running service suitable for systemd deployment.
Includes signal handling, graceful shutdown, and syslog integration.

Usage:
    # Run as daemon
    python -m csi_node.daemon

    # Run with specific config
    python -m csi_node.daemon --config /etc/csi-presence/config.yaml

    # Run in foreground (for debugging)
    python -m csi_node.daemon --foreground
"""

from __future__ import annotations
import argparse
import logging
import logging.handlers
import os
import signal
import sys
import time
from pathlib import Path
from threading import Event
from typing import Optional

import yaml

from . import pipeline
from . import config_validator
from . import utils


# Daemon state
_shutdown_event = Event()
_logger: Optional[logging.Logger] = None


def setup_logging(
    syslog: bool = True,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure logging for daemon operation.

    Args:
        syslog: Enable syslog output
        log_file: Optional file path for logging
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger("csi-presence")
    logger.setLevel(level)

    # Console handler (always available)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(console)

    # Syslog handler (Linux only)
    if syslog and sys.platform.startswith("linux"):
        try:
            syslog_handler = logging.handlers.SysLogHandler(
                address="/dev/log",
                facility=logging.handlers.SysLogHandler.LOG_DAEMON
            )
            syslog_handler.setLevel(level)
            syslog_handler.setFormatter(logging.Formatter(
                "csi-presence[%(process)d]: %(message)s"
            ))
            logger.addHandler(syslog_handler)
        except Exception:
            pass  # Syslog not available

    # File handler (optional)
    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            ))
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create file handler: {e}")

    return logger


def signal_handler(signum: int, frame) -> None:
    """Handle shutdown signals gracefully."""
    global _logger, _shutdown_event
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    if _logger:
        _logger.info(f"Received signal {sig_name}, initiating shutdown...")
    _shutdown_event.set()


def validate_environment() -> bool:
    """Check that the environment is suitable for daemon operation.

    Returns:
        True if environment is valid
    """
    # Check for required binaries
    feitcsi_paths = [
        "/usr/local/bin/feitcsi",
        "/usr/bin/feitcsi",
        Path.home() / "FeitCSI" / "feitcsi",
    ]
    feitcsi_found = any(Path(p).exists() for p in feitcsi_paths)

    if not feitcsi_found:
        if _logger:
            _logger.warning("FeitCSI binary not found in standard locations")
        # Not fatal - might be using different capture method

    # Check for debugfs (required for CSI capture)
    debugfs_path = Path("/sys/kernel/debug/iwlwifi")
    if sys.platform.startswith("linux") and not debugfs_path.exists():
        if _logger:
            _logger.warning(
                "iwlwifi debugfs not found. Mount debugfs or load iwlwifi module."
            )

    return True


def run_daemon(
    config_path: str,
    interface: Optional[str] = None,
    foreground: bool = False,
) -> int:
    """Run the CSI presence daemon.

    Args:
        config_path: Path to config.yaml
        interface: Network interface for capture (optional)
        foreground: Run in foreground mode (no daemonization)

    Returns:
        Exit code (0 = success, non-zero = error)
    """
    global _logger, _shutdown_event

    # Load configuration
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Setup logging
    log_file = cfg.get("daemon_log_file", "/var/log/csi-presence/daemon.log")
    if foreground:
        log_file = None
    _logger = setup_logging(
        syslog=not foreground,
        log_file=log_file,
        level=logging.DEBUG if cfg.get("debug", False) else logging.INFO,
    )

    _logger.info("CSI Presence Node daemon starting...")
    _logger.info(f"Config: {config_path}")

    # Validate configuration
    validation = config_validator.validate_config(cfg)
    if not validation.valid:
        for err in validation.errors:
            _logger.error(f"Config error: {err}")
        return 1
    for warn in validation.warnings:
        _logger.warning(f"Config warning: {warn}")

    # Validate environment
    if not validate_environment():
        return 1

    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal_handler)

    # Initialize UDP streamer if enabled
    udp_streamer = None
    if cfg.get("udp_enabled", False):
        try:
            from . import udp_streamer as udp_mod
            udp_streamer = udp_mod.UDPStreamer.from_config(cfg)
            _logger.info(f"UDP streaming to {cfg.get('udp_host')}:{cfg.get('udp_port')}")
            if cfg.get("atak_enabled", False):
                _logger.info(f"ATAK CoT streaming to port {cfg.get('atak_port')}")
        except Exception as exc:
            _logger.error(f"Failed to initialize UDP streamer: {exc}")

    # Create run log manager
    log_manager = utils.RunLogManager(
        cfg["output_file"],
        rotation_bytes=int(cfg.get("rotation_max_bytes", 1_048_576)),
        rotation_seconds=float(cfg.get("rotation_interval_seconds", 0.0)),
        retention=int(cfg.get("rotation_retention", 5)),
    )
    _logger.info(f"Output log: {log_manager.log_path}")

    # Main daemon loop
    _logger.info("Daemon started successfully")
    restart_count = 0
    max_restarts = cfg.get("max_restarts", 10)
    restart_delay = cfg.get("restart_delay", 5.0)

    while not _shutdown_event.is_set():
        try:
            _logger.info("Starting capture pipeline...")

            # Run the pipeline
            pipeline.run_demo(
                pose=cfg.get("pose_enabled", False),
                tui=False,  # No TUI in daemon mode
                replay_path=None,  # Live capture only
                source=None,
                window=cfg.get("window_size", 3.0),
                out=str(log_manager.log_path),
            )

            # If we reach here, pipeline exited cleanly
            _logger.info("Pipeline exited cleanly")

        except KeyboardInterrupt:
            _logger.info("Keyboard interrupt received")
            break

        except Exception as exc:
            restart_count += 1
            _logger.error(f"Pipeline crashed: {exc}")

            if restart_count >= max_restarts:
                _logger.error(f"Max restarts ({max_restarts}) exceeded, giving up")
                return 1

            if not _shutdown_event.is_set():
                _logger.info(f"Restarting in {restart_delay}s (attempt {restart_count}/{max_restarts})")
                time.sleep(restart_delay)

    # Cleanup
    _logger.info("Shutting down...")
    if udp_streamer:
        udp_streamer.close()
    _logger.info("Daemon stopped")

    return 0


def main() -> None:
    """CLI entry point for daemon."""
    parser = argparse.ArgumentParser(
        description="CSI Presence Node Daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run daemon with default config
    python -m csi_node.daemon

    # Run with custom config
    python -m csi_node.daemon --config /etc/csi-presence/config.yaml

    # Run in foreground for debugging
    python -m csi_node.daemon --foreground

    # Specify capture interface
    python -m csi_node.daemon --iface wlan0
""",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--iface",
        "-i",
        type=str,
        default=None,
        help="Network interface for capture",
    )
    parser.add_argument(
        "--foreground",
        "-f",
        action="store_true",
        help="Run in foreground (no daemonization)",
    )

    # Check for environment variable override
    env_config = os.environ.get("CSI_NODE_CONFIG")
    args = parser.parse_args()

    if env_config and not any("--config" in arg for arg in sys.argv):
        args.config = env_config

    exit_code = run_daemon(
        config_path=args.config,
        interface=args.iface,
        foreground=args.foreground,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
