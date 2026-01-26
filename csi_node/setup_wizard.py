"""Guided setup wizard for CSI Presence Node.

Walks users through all required configuration steps:
1. Hardware check (Intel AX210 detection)
2. FeitCSI verification
3. Baseline capture
4. Distance calibration
5. GPS/location configuration
6. ATAK integration setup
7. Pose model training (optional)

Usage:
    python -m csi_node.setup_wizard
"""

from __future__ import annotations

import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional
import yaml


CONFIG_PATH = Path(__file__).parent / "config.yaml"


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def print_step(num: int, total: int, desc: str) -> None:
    """Print step indicator."""
    print(f"\n[Step {num}/{total}] {desc}")
    print("-" * 40)


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Ask a yes/no question."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        response = input(prompt + suffix).strip().lower()
        if response == "":
            return default
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'")


def ask_float(prompt: str, default: Optional[float] = None) -> float:
    """Ask for a float value."""
    suffix = f" [{default}]: " if default is not None else ": "
    while True:
        response = input(prompt + suffix).strip()
        if response == "" and default is not None:
            return default
        try:
            return float(response)
        except ValueError:
            print("Please enter a valid number")


def ask_string(prompt: str, default: str = "") -> str:
    """Ask for a string value."""
    suffix = f" [{default}]: " if default else ": "
    response = input(prompt + suffix).strip()
    return response if response else default


def load_config() -> dict:
    """Load current config."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(cfg: dict) -> None:
    """Save config to file."""
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  Config saved to {CONFIG_PATH}")


def check_hardware() -> bool:
    """Check for Intel AX210 NIC."""
    print("Checking for Intel AX210 WiFi adapter...")

    # Try lspci on Linux
    if shutil.which("lspci"):
        try:
            result = subprocess.run(
                ["lspci", "-nn"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if "AX210" in result.stdout or "AX211" in result.stdout:
                print("  [OK] Intel AX210/AX211 detected")
                return True
        except Exception:
            pass

    # Try iwconfig
    if shutil.which("iwconfig"):
        try:
            result = subprocess.run(
                ["iwconfig"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if "wlan" in result.stdout or "wl" in result.stdout:
                print("  [OK] WiFi interface detected")
                return True
        except Exception:
            pass

    print("  [!] Could not detect Intel AX210")
    print("      Make sure you're running on the target hardware (GPD Win 4)")
    return False


def check_feitcsi() -> bool:
    """Check for FeitCSI installation."""
    print("Checking for FeitCSI...")

    feitcsi_path = shutil.which("feitcsi")
    if feitcsi_path:
        print(f"  [OK] FeitCSI found at {feitcsi_path}")
        return True

    # Check common locations
    common_paths = [
        "/usr/local/bin/feitcsi",
        "/usr/bin/feitcsi",
        Path.home() / "FeitCSI" / "build" / "feitcsi",
    ]

    for path in common_paths:
        if Path(path).exists():
            print(f"  [OK] FeitCSI found at {path}")
            return True

    print("  [!] FeitCSI not found")
    print("      Run ./setup.sh to build FeitCSI, or install manually")
    return False


def capture_baseline(cfg: dict) -> bool:
    """Guide user through baseline capture."""
    print("A baseline captures the 'empty room' WiFi signature.")
    print("This improves presence detection accuracy.\n")

    if not ask_yes_no("Capture baseline now?"):
        print("  Skipping baseline capture")
        return False

    print("\n  1. Make sure the room is EMPTY (no people or pets)")
    print("  2. Keep doors/windows closed")
    print("  3. Turn off fans or moving objects")
    input("\nPress Enter when the room is ready...")

    duration = int(ask_float("Baseline duration in seconds", 60))

    print(f"\n  Capturing baseline for {duration} seconds...")
    print("  Keep the room still!\n")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "csi_node.baseline", "--duration", str(duration)],
            timeout=duration + 30
        )
        if result.returncode == 0:
            print("  [OK] Baseline captured successfully")
            return True
        else:
            print("  [!] Baseline capture failed")
            return False
    except subprocess.TimeoutExpired:
        print("  [!] Baseline capture timed out")
        return False
    except Exception as e:
        print(f"  [!] Error: {e}")
        return False


def run_calibration(cfg: dict) -> bool:
    """Guide user through distance calibration."""
    print("Distance calibration fits a path-loss model to your environment.")
    print("You'll need to record CSI at two known distances.\n")

    if not ask_yes_no("Run distance calibration now?"):
        print("  Skipping calibration")
        print("  Using default path_loss_exponent=2.0 (may be inaccurate)")
        return False

    print("\n  Calibration requires two short recordings:")
    print("  - One with a person at a NEAR distance (e.g., 1 meter)")
    print("  - One with a person at a FAR distance (e.g., 3 meters)")
    print("\n  Each recording should be ~10-20 seconds with the person standing still.")

    d1 = ask_float("Near distance in meters", 1.0)
    d2 = ask_float("Far distance in meters", 3.0)

    # Record near distance
    print(f"\n  Recording NEAR distance ({d1}m)...")
    print(f"  Position someone exactly {d1} meters from the sensor.")
    input("Press Enter when ready...")

    near_log = Path("data/calibration_near.log")
    print("  Recording for 15 seconds... (person should stand still)")
    # In a real scenario, this would capture CSI data
    # For now, we'll check if capture is running

    print("\n  [!] Manual step required:")
    print(f"      1. Start CSI capture (python run.py --iface wlan0)")
    print(f"      2. Have someone stand at {d1}m for 15 seconds")
    print(f"      3. Save the log as: {near_log}")
    input("Press Enter when near recording is complete...")

    # Record far distance
    print(f"\n  Recording FAR distance ({d2}m)...")
    print(f"  Position someone exactly {d2} meters from the sensor.")
    input("Press Enter when ready...")

    far_log = Path("data/calibration_far.log")
    print("\n  [!] Manual step required:")
    print(f"      1. Have someone stand at {d2}m for 15 seconds")
    print(f"      2. Save the log as: {far_log}")
    input("Press Enter when far recording is complete...")

    # Run calibration
    if near_log.exists() and far_log.exists():
        print("\n  Running calibration...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "csi_node.calibrate",
                "--d1", str(d1), "--d2", str(d2),
                "--log1", str(near_log), "--log2", str(far_log)
            ], timeout=60)
            if result.returncode == 0:
                print("  [OK] Calibration complete")
                cfg["calibrated"] = True
                return True
        except Exception as e:
            print(f"  [!] Calibration error: {e}")
    else:
        print("  [!] Log files not found. Skipping calibration.")

    return False


def configure_location(cfg: dict) -> None:
    """Configure sensor GPS location."""
    print("GPS coordinates are needed for ATAK map integration.")
    print("You can get coordinates from Google Maps or a GPS app.\n")

    current_lat = cfg.get("sensor_lat", 0.0)
    current_lon = cfg.get("sensor_lon", 0.0)

    if current_lat != 0.0 or current_lon != 0.0:
        print(f"  Current location: {current_lat}, {current_lon}")
        if not ask_yes_no("Update location?", default=False):
            return

    cfg["sensor_lat"] = ask_float("Sensor latitude (e.g., 38.8977)", current_lat)
    cfg["sensor_lon"] = ask_float("Sensor longitude (e.g., -77.0365)", current_lon)
    cfg["sensor_heading"] = ask_float("Sensor heading in degrees (0=North)",
                                       cfg.get("sensor_heading", 0.0))

    print(f"  [OK] Location set to {cfg['sensor_lat']}, {cfg['sensor_lon']}")


def configure_atak(cfg: dict) -> None:
    """Configure ATAK integration."""
    print("ATAK integration streams detections to TAK clients.")
    print("Uses Cursor-on-Target (CoT) XML over UDP multicast.\n")

    enable = ask_yes_no("Enable ATAK streaming?", default=False)
    cfg["atak_enabled"] = enable

    if enable:
        cfg["atak_port"] = int(ask_float("ATAK port", cfg.get("atak_port", 4242)))
        cfg["sensor_uid"] = ask_string("Sensor UID", cfg.get("sensor_uid", "vantage-001"))
        cfg["sensor_callsign"] = ask_string("Sensor callsign",
                                            cfg.get("sensor_callsign", "VANTAGE-1"))
        print("  [OK] ATAK enabled")
    else:
        print("  ATAK disabled")


def configure_udp(cfg: dict) -> None:
    """Configure UDP streaming."""
    print("UDP streaming broadcasts JSON detections on the network.\n")

    enable = ask_yes_no("Enable UDP streaming?", default=False)
    cfg["udp_enabled"] = enable

    if enable:
        cfg["udp_host"] = ask_string("UDP broadcast address",
                                     cfg.get("udp_host", "239.2.3.1"))
        cfg["udp_port"] = int(ask_float("UDP port", cfg.get("udp_port", 4243)))
        print("  [OK] UDP streaming enabled")
    else:
        print("  UDP streaming disabled")


def train_pose_model(cfg: dict) -> bool:
    """Guide user through pose model training."""
    print("The pose classifier detects standing/crouching/prone positions.")
    print("Training with real data improves accuracy.\n")

    if not ask_yes_no("Train pose model now?", default=False):
        print("  Using default toy model (less accurate)")
        return False

    print("\n  Training requires collecting labeled samples.")
    print("  You'll assume each pose while the system records CSI data.")
    print("  This takes about 5-10 minutes.\n")

    if not ask_yes_no("Continue with training?"):
        return False

    try:
        result = subprocess.run([
            sys.executable, "-m", "csi_node.data_collector",
            "--output", "data/training.npz"
        ])
        if result.returncode == 0:
            print("\n  Training classifier...")
            result = subprocess.run([
                sys.executable, "run.py", "--train-pose"
            ])
            if result.returncode == 0:
                print("  [OK] Pose model trained successfully")
                return True
    except Exception as e:
        print(f"  [!] Training error: {e}")

    return False


def print_summary(cfg: dict) -> None:
    """Print configuration summary."""
    print_header("Configuration Summary")

    print(f"  Calibrated:      {'Yes' if cfg.get('calibrated') else 'No (using defaults)'}")
    print(f"  Path loss exp:   {cfg.get('path_loss_exponent', 2.0)}")
    print(f"  TX power (dBm):  {cfg.get('tx_power_dbm', -40.0)}")
    print()
    print(f"  Sensor location: {cfg.get('sensor_lat', 0.0)}, {cfg.get('sensor_lon', 0.0)}")
    print(f"  Sensor heading:  {cfg.get('sensor_heading', 0.0)}°")
    print()
    print(f"  ATAK enabled:    {cfg.get('atak_enabled', False)}")
    print(f"  UDP enabled:     {cfg.get('udp_enabled', False)}")

    print("\n" + "-" * 40)
    print("To start the sensor:")
    print("  python run.py --iface wlan0 --pose --tui")
    print()
    print("To run as a service:")
    print("  sudo systemctl start csi-presence-node")


def main() -> None:
    """Run the setup wizard."""
    print_header("CSI Presence Node - Setup Wizard")

    print("This wizard will guide you through the setup process.")
    print("You can skip any step and run it later.\n")

    if not ask_yes_no("Continue with setup?"):
        print("Setup cancelled.")
        return

    cfg = load_config()
    total_steps = 7

    # Step 1: Hardware check
    print_step(1, total_steps, "Hardware Check")
    hw_ok = check_hardware()

    # Step 2: FeitCSI check
    print_step(2, total_steps, "FeitCSI Check")
    feit_ok = check_feitcsi()

    if not hw_ok or not feit_ok:
        print("\n[!] Hardware requirements not met.")
        print("    Setup can continue, but live capture won't work.")
        if not ask_yes_no("Continue anyway?"):
            return

    # Step 3: Baseline capture
    print_step(3, total_steps, "Baseline Capture")
    capture_baseline(cfg)

    # Step 4: Distance calibration
    print_step(4, total_steps, "Distance Calibration")
    run_calibration(cfg)

    # Step 5: Location configuration
    print_step(5, total_steps, "Sensor Location")
    configure_location(cfg)

    # Step 6: ATAK/UDP configuration
    print_step(6, total_steps, "Network Streaming")
    configure_atak(cfg)
    configure_udp(cfg)

    # Step 7: Pose model training (optional)
    print_step(7, total_steps, "Pose Model Training (Optional)")
    train_pose_model(cfg)

    # Save configuration
    print("\nSaving configuration...")
    save_config(cfg)

    # Summary
    print_summary(cfg)

    print_header("Setup Complete!")
    print("Run 'python run.py --validate' to verify configuration.")


if __name__ == "__main__":
    main()
