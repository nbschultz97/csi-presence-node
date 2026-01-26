#!/usr/bin/env python3
"""CSI Presence Node - Entry Point

Usage:
    # First-time setup (RECOMMENDED for new installs)
    python run.py --setup

    # Live capture with TUI
    python run.py --iface wlan0 --tui --pose

    # Replay mode
    python run.py --replay data/sample_csi.b64 --tui

    # Daemon mode (for systemd)
    python run.py --daemon

    # Validate configuration
    python run.py --validate

    # Train pose model
    python run.py --train-pose
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="CSI Presence Node",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Run as daemon (for systemd)",
    )
    mode_group.add_argument(
        "--validate", "-V",
        action="store_true",
        help="Validate configuration and exit",
    )
    mode_group.add_argument(
        "--train-pose",
        action="store_true",
        help="Train pose classifier",
    )
    mode_group.add_argument(
        "--setup",
        action="store_true",
        help="Run guided setup wizard (recommended for first-time setup)",
    )

    # Pipeline arguments (when not in special mode)
    parser.add_argument("--pose", action="store_true", help="Enable pose classifier")
    parser.add_argument("--tui", action="store_true", help="Launch curses UI")
    parser.add_argument("--replay", type=str, default=None, help="Replay log file")
    parser.add_argument("--iface", type=str, default=None, help="Live capture interface")
    parser.add_argument("--window", type=float, default=3.0, help="Window size (s)")
    parser.add_argument("--out", type=str, default="data/presence_log.jsonl", help="Output JSONL")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed factor")
    parser.add_argument("--log", type=str, default=None, help="Override input log path")
    parser.add_argument("--config", "-c", type=str, default=None, help="Config file path")

    args = parser.parse_args()

    # Handle special modes
    if args.daemon:
        from csi_node.daemon import main as daemon_main
        daemon_main()
        return

    if args.validate:
        from csi_node.config_validator import main as validate_main
        sys.argv = ["config_validator", args.config or "csi_node/config.yaml"]
        validate_main()
        return

    if args.train_pose:
        from csi_node.pose_classifier import main as pose_main
        sys.argv = ["pose_classifier", "--train"]
        pose_main()
        return

    if args.setup:
        from csi_node.setup_wizard import main as setup_main
        setup_main()
        return

    # Normal pipeline mode
    from csi_node.pipeline import run_demo

    src = None
    if args.iface:
        try:
            from csi_node import feitcsi
            src = feitcsi.live_stream(args.iface)
        except Exception as exc:
            print(f"FeitCSI live stream unavailable: {exc}", file=sys.stderr)

    run_demo(
        pose=args.pose,
        tui=args.tui,
        replay_path=args.replay,
        source=src,
        window=args.window,
        out=args.out,
        speed=args.speed,
        log_override=args.log,
    )


if __name__ == "__main__":
    main()
