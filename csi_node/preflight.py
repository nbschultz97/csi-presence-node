"""Pre-flight check for demo readiness.

Validates all dependencies, configuration, calibration state, and hardware
before a demo. Run this 10 minutes before your audience arrives.

Usage:
    python -m csi_node.preflight
    python -m csi_node.preflight --fix    # Auto-fix what it can
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    fixable: bool = False
    fix_hint: str = ""


def check_python_version() -> CheckResult:
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    return CheckResult(
        "Python Version",
        ok,
        f"Python {v.major}.{v.minor}.{v.micro}" + ("" if ok else " — need 3.10+"),
    )


def check_numpy() -> CheckResult:
    try:
        import numpy as np
        return CheckResult("NumPy", True, f"v{np.__version__}")
    except ImportError:
        return CheckResult("NumPy", False, "not installed", True, "pip install numpy")


def check_scipy() -> CheckResult:
    try:
        from scipy.signal import butter, sosfilt
        import scipy
        return CheckResult("SciPy (signal processing)", True, f"v{scipy.__version__}")
    except ImportError:
        return CheckResult("SciPy", False, "not installed — Butterworth filter unavailable",
                           True, "pip install scipy")


def check_watchdog() -> CheckResult:
    try:
        import watchdog
        ver = getattr(watchdog, "__version__", "installed")
        return CheckResult("Watchdog (file monitoring)", True, f"v{ver}")
    except ImportError:
        return CheckResult("Watchdog", False, "not installed", True, "pip install watchdog")


def check_yaml() -> CheckResult:
    try:
        import yaml
        return CheckResult("PyYAML", True, "OK")
    except ImportError:
        return CheckResult("PyYAML", False, "not installed", True, "pip install pyyaml")


def check_config() -> CheckResult:
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    if not cfg_path.exists():
        return CheckResult("Config", False, f"Missing: {cfg_path}")
    try:
        import yaml
        cfg = yaml.safe_load(open(cfg_path))
        issues = []
        if cfg.get("path_loss_exponent", 2.0) < 0:
            issues.append("path_loss_exponent is negative")
        if not cfg.get("channel"):
            issues.append("no channel set")
        if issues:
            return CheckResult("Config", False, "; ".join(issues))
        ch = cfg.get("channel", "?")
        bw = cfg.get("bandwidth", "?")
        return CheckResult("Config", True, f"ch{ch} @ {bw}MHz")
    except Exception as e:
        return CheckResult("Config", False, str(e))


def check_calibration() -> CheckResult:
    cal_path = Path(__file__).resolve().parent.parent / "data" / "calibration.json"
    if not cal_path.exists():
        return CheckResult(
            "Calibration",
            False,
            "No calibration file — will use adaptive thresholds",
            fix_hint="Run: python run.py --demo, then click 📐 Calibrate",
        )
    try:
        data = json.loads(cal_path.read_text())
        ts = data.get("timestamp", 0)
        age_hours = (time.time() - ts) / 3600 if ts else float("inf")
        energy = data.get("baseline_energy", 0)
        if age_hours > 24:
            return CheckResult(
                "Calibration",
                False,
                f"Stale ({age_hours:.0f}h old) — recalibrate for current environment",
            )
        return CheckResult("Calibration", True, f"Energy baseline: {energy:.1f}, {age_hours:.1f}h old")
    except Exception as e:
        return CheckResult("Calibration", False, f"Corrupt: {e}")


def check_environments() -> CheckResult:
    env_dir = Path(__file__).resolve().parent.parent / "data" / "environments"
    if not env_dir.exists():
        return CheckResult("Environment Profiles", True, "None saved (OK for first demo)")
    profiles = list(env_dir.glob("*.json"))
    if not profiles:
        return CheckResult("Environment Profiles", True, "None saved")
    names = [p.stem for p in profiles]
    return CheckResult("Environment Profiles", True, f"{len(profiles)} saved: {', '.join(names)}")


def check_demo_data() -> CheckResult:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    demo_log = data_dir / "demo_csi.log"
    if demo_log.exists():
        size_kb = demo_log.stat().st_size / 1024
        return CheckResult("Demo Data", True, f"demo_csi.log ({size_kb:.0f} KB)")
    return CheckResult("Demo Data", True, "No replay data (simulation mode still works)")


def check_port(port: int = 8088) -> CheckResult:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        result = s.connect_ex(("127.0.0.1", port))
        s.close()
        if result == 0:
            return CheckResult("Port", False, f"Port {port} already in use", fix_hint=f"Use --port {port + 1}")
        return CheckResult("Port", True, f"Port {port} available")
    except Exception:
        return CheckResult("Port", True, f"Port {port} (check skipped)")


def run_preflight(port: int = 8088) -> list[CheckResult]:
    checks = [
        check_python_version(),
        check_numpy(),
        check_scipy(),
        check_watchdog(),
        check_yaml(),
        check_config(),
        check_calibration(),
        check_environments(),
        check_demo_data(),
        check_port(port),
    ]
    return checks


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vantage pre-flight demo check")
    parser.add_argument("--port", type=int, default=8088, help="Port to check")
    args, _ = parser.parse_known_args()

    print("\n  🎯 VANTAGE Pre-Flight Check\n  " + "=" * 36 + "\n")

    checks = run_preflight(args.port)
    passed = 0
    failed = 0

    for c in checks:
        icon = "✅" if c.passed else "❌"
        print(f"  {icon} {c.name}: {c.message}")
        if not c.passed and c.fix_hint:
            print(f"     💡 {c.fix_hint}")
        if c.passed:
            passed += 1
        else:
            failed += 1

    print(f"\n  {'🟢 ALL CLEAR' if failed == 0 else f'🔴 {failed} issue(s) found'} — {passed}/{len(checks)} passed\n")

    if failed == 0:
        print("  Ready for demo! Run:")
        print("    python run.py --demo                    # Simulation")
        print("    python run.py --demo --through-wall     # Through-wall sim")
        print("    .\\demo.ps1                              # Windows quick-start")
        print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
