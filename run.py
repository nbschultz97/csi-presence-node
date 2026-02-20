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
import os
import argparse

# Fix Windows console encoding for emoji output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")


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
    mode_group.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch web dashboard (http://localhost:8088)",
    )
    mode_group.add_argument(
        "--demo",
        action="store_true",
        help="Launch demo mode — simulated CSI data + web dashboard (no hardware needed)",
    )
    mode_group.add_argument(
        "--preflight",
        action="store_true",
        help="Run pre-flight checks to verify demo readiness",
    )
    mode_group.add_argument(
        "--calibrate",
        action="store_true",
        help="Run 30-second empty-room calibration for presence detection",
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
    parser.add_argument("--port", type=int, default=8088, help="Dashboard HTTP port")
    parser.add_argument("--through-wall", action="store_true",
                        help="Use through-wall detection profile and scenarios (lower thresholds, attenuated signals)")

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

    if args.preflight:
        from csi_node.preflight import main as preflight_main
        sys.exit(preflight_main())

    if args.calibrate:
        from csi_node.presence import AdaptivePresenceDetector
        from csi_node.environment import EnvironmentManager
        from pathlib import Path
        import time
        import numpy as np

        detector = AdaptivePresenceDetector(sample_rate_hz=30.0)
        detector.calibrate_start()

        print("\n  📐 VANTAGE CALIBRATION")
        print("  Ensure the detection area is EMPTY.")
        print("  Collecting 30 seconds of baseline data...\n")

        # Use simulator for calibration if no log file available
        log_path = args.log
        if log_path:
            from csi_node import utils
            lp = Path(log_path)
            if not lp.exists():
                print(f"  ❌ Log file not found: {log_path}")
                sys.exit(1)
            # Tail log file for 30 seconds
            import json as json_mod
            start = time.time()
            count = 0
            with open(lp, "r") as f:
                f.seek(0, 2)
                while time.time() - start < 30:
                    line = f.readline()
                    if not line:
                        time.sleep(0.03)
                        continue
                    pkt = utils.parse_csi_line(line)
                    if pkt and pkt.get("csi") is not None:
                        detector.update(pkt["csi"], rssi=pkt.get("rssi"), timestamp=pkt.get("ts"))
                        count += 1
                        if count % 30 == 0:
                            elapsed = time.time() - start
                            print(f"  {elapsed:.0f}s / 30s  ({count} frames)", end="\r")
        else:
            from csi_node.simulator import CSISimulator, SimScenario
            empty = [SimScenario("calibration", 30.0, False, "none", 1.0, 1.0)]
            sim = CSISimulator(scenarios=empty)
            count = 0
            start = time.time()
            for pkt in sim.stream(loop=False, realtime=True):
                detector.update(pkt["csi"], rssi=pkt.get("rssi"), timestamp=pkt.get("ts"))
                count += 1
                if count % 30 == 0:
                    elapsed = time.time() - start
                    print(f"  {elapsed:.0f}s / 30s  ({count} frames)", end="\r")

        print()
        success = detector.calibrate_finish()
        if success:
            cal_dir = Path(__file__).resolve().parent / "data"
            cal_dir.mkdir(parents=True, exist_ok=True)
            cal_path = cal_dir / "calibration.json"
            detector.save_calibration(cal_path)
            print(f"  ✅ Calibration saved to {cal_path}")
            print(f"  Baseline energy:   {detector._baseline_energy:.2f}")
            print(f"  Baseline variance: {detector._baseline_variance:.6f}")
            print(f"  Baseline spectral: {detector._baseline_spectral:.6f}")
            print(f"  Frames collected:  {count}")

            # Optionally save as named environment profile
            env_name = input("\n  Save as environment profile? (name or Enter to skip): ").strip()
            if env_name:
                wall = input("  Wall type (drywall/concrete/wood/none): ").strip() or "unknown"
                mgr = EnvironmentManager()
                path = mgr.save(env_name, detector, wall_type=wall)
                print(f"  ✅ Profile saved: {path}")
        else:
            print("  ❌ Calibration failed — not enough samples collected.")
            sys.exit(1)
        print()
        return

    if args.demo:
        from csi_node.web_dashboard import run_dashboard
        tw = args.through_wall
        print("\n  🎯 VANTAGE DEMO MODE" + (" — THROUGH-WALL" if tw else ""))
        print("  Generating synthetic CSI data — no hardware required.")
        if tw:
            print("  Through-wall profile: lower thresholds, attenuated signals, longer scenarios")
        print("  Scenarios: empty room → person enters → movement → breathing → exits")
        print(f"  Dashboard: http://localhost:{args.port}\n")
        run_dashboard(port=args.port, simulate=True, through_wall=tw)
        return

    if args.dashboard:
        from csi_node.web_dashboard import run_dashboard
        run_dashboard(
            port=args.port,
            replay_path=args.replay,
            log_path=args.log,
            speed=args.speed,
            through_wall=args.through_wall,
        )
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
        through_wall=args.through_wall,
    )


if __name__ == "__main__":
    main()
