"""Configuration validation for csi-presence-node.

Validates config.yaml parameters and provides warnings for anomalous values.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import yaml


@dataclass
class ValidationResult:
    """Result of a configuration validation."""
    valid: bool
    warnings: List[str]
    errors: List[str]
    fixed_values: Dict[str, Any]


# Valid ranges for configuration parameters
CONFIG_SCHEMA = {
    # Hardware settings
    "channel": {
        "type": int,
        "min": 1,
        "max": 196,  # WiFi channels 1-14 (2.4GHz), 36-196 (5GHz)
        "default": 36,
    },
    "bandwidth": {
        "type": int,
        "allowed": [20, 40, 80, 160],
        "default": 80,
    },
    # Processing parameters
    "window_size": {
        "type": float,
        "min": 0.1,
        "max": 30.0,
        "default": 1.0,
        "unit": "seconds",
    },
    "window_hop": {
        "type": float,
        "min": 0.05,
        "max": 10.0,
        "default": 0.5,
        "unit": "seconds",
    },
    "variance_threshold": {
        "type": float,
        "min": 0.0,
        "max": 1000.0,
        "default": 5.0,
    },
    "pca_threshold": {
        "type": float,
        "min": 0.0,
        "max": 1000.0,
        "default": 1.0,
    },
    "rssi_delta": {
        "type": float,
        "min": 0.0,
        "max": 30.0,
        "default": 2.0,
        "unit": "dB",
    },
    # Calibration parameters
    "tx_power_dbm": {
        "type": float,
        "min": -100.0,
        "max": 30.0,
        "default": -40.0,
        "unit": "dBm",
    },
    "path_loss_exponent": {
        "type": float,
        "min": 1.0,  # Free space minimum
        "max": 6.0,  # Dense urban maximum
        "default": 2.0,
        "warning_if_negative": True,
        "critical": True,  # This is critical for distance estimation
    },
    # Logging
    "rotation_max_bytes": {
        "type": int,
        "min": 0,
        "max": 1024 * 1024 * 1024,  # 1GB max
        "default": 1048576,
    },
    "rotation_interval_seconds": {
        "type": float,
        "min": 0,
        "max": 86400,  # 24 hours
        "default": 600,
    },
    "rotation_retention": {
        "type": int,
        "min": 0,
        "max": 100,
        "default": 5,
    },
    # UDP streaming (new)
    "udp_enabled": {
        "type": bool,
        "default": False,
    },
    "udp_host": {
        "type": str,
        "default": "239.2.3.1",
    },
    "udp_port": {
        "type": int,
        "min": 1,
        "max": 65535,
        "default": 4243,
    },
    "atak_enabled": {
        "type": bool,
        "default": False,
    },
    "atak_port": {
        "type": int,
        "min": 1,
        "max": 65535,
        "default": 4242,
    },
    "sensor_uid": {
        "type": str,
        "default": "vantage-001",
    },
    "sensor_callsign": {
        "type": str,
        "default": "VANTAGE-1",
    },
    "sensor_lat": {
        "type": float,
        "min": -90.0,
        "max": 90.0,
        "default": 0.0,
    },
    "sensor_lon": {
        "type": float,
        "min": -180.0,
        "max": 180.0,
        "default": 0.0,
    },
    "sensor_heading": {
        "type": float,
        "min": 0.0,
        "max": 360.0,
        "default": 0.0,
    },
}


def validate_config(cfg: Dict[str, Any], auto_fix: bool = False) -> ValidationResult:
    """Validate a configuration dictionary.

    Args:
        cfg: Configuration dictionary (from config.yaml)
        auto_fix: If True, attempt to fix invalid values

    Returns:
        ValidationResult with validity status, warnings, and errors
    """
    warnings: List[str] = []
    errors: List[str] = []
    fixed: Dict[str, Any] = {}

    for key, schema in CONFIG_SCHEMA.items():
        if key not in cfg:
            continue  # Optional keys are fine

        value = cfg[key]
        expected_type = schema["type"]

        # Type check
        if expected_type == float and isinstance(value, int):
            value = float(value)  # Allow int -> float
        elif expected_type == int and isinstance(value, float) and value == int(value):
            value = int(value)  # Allow float -> int if whole number

        if not isinstance(value, expected_type):
            errors.append(
                f"{key}: expected {expected_type.__name__}, got {type(value).__name__}"
            )
            if auto_fix:
                fixed[key] = schema["default"]
            continue

        # Range check for numbers
        if expected_type in (int, float):
            min_val = schema.get("min")
            max_val = schema.get("max")
            allowed = schema.get("allowed")

            if allowed is not None and value not in allowed:
                errors.append(f"{key}: {value} not in allowed values {allowed}")
                if auto_fix:
                    fixed[key] = schema["default"]
                continue

            if min_val is not None and value < min_val:
                # Special case: negative path loss exponent is a common calibration error
                if key == "path_loss_exponent" and value < 0:
                    errors.append(
                        f"{key}: {value} is NEGATIVE. Path loss exponent must be "
                        f"positive (typically 2.0-4.0 for indoor). This will cause "
                        f"distance estimation to fail. Recalibrate or set manually."
                    )
                else:
                    errors.append(f"{key}: {value} below minimum {min_val}")

                if auto_fix:
                    fixed[key] = schema["default"]
                continue

            if max_val is not None and value > max_val:
                errors.append(f"{key}: {value} above maximum {max_val}")
                if auto_fix:
                    fixed[key] = schema["default"]
                continue

            # Warnings for suspicious but not invalid values
            if key == "path_loss_exponent":
                if value < 1.5:
                    warnings.append(
                        f"{key}: {value} is unusually low (typical: 2.0-4.0). "
                        f"This may indicate calibration issues."
                    )
                elif value > 4.5:
                    warnings.append(
                        f"{key}: {value} is unusually high (typical: 2.0-4.0). "
                        f"This may indicate heavy obstruction or calibration issues."
                    )

            if key == "tx_power_dbm":
                if value > 0:
                    warnings.append(
                        f"{key}: {value} dBm is positive, which is unusual for "
                        f"RSSI at 1m. Typical values are -20 to -50 dBm."
                    )
                elif value < -80:
                    warnings.append(
                        f"{key}: {value} dBm is very low. This may indicate "
                        f"calibration was done at excessive range."
                    )

    # Check for required files
    baseline_file = cfg.get("baseline_file", "")
    if baseline_file and not Path(baseline_file).exists():
        warnings.append(f"baseline_file: '{baseline_file}' does not exist")

    # Check calibration consistency
    if cfg.get("calibrated", False):
        ple = cfg.get("path_loss_exponent", 2.0)
        txp = cfg.get("tx_power_dbm", -40.0)
        if ple < 0:
            errors.append(
                "Calibration is marked complete but path_loss_exponent is negative. "
                "Distance estimation will be incorrect. Recalibrate."
            )
        if ple == 2.0 and txp == -40.0:
            warnings.append(
                "Calibration is marked complete but values are defaults. "
                "Consider recalibrating for your environment."
            )

    valid = len(errors) == 0
    return ValidationResult(
        valid=valid,
        warnings=warnings,
        errors=errors,
        fixed_values=fixed,
    )


def validate_config_file(
    path: str | Path,
    auto_fix: bool = False,
    write_fixes: bool = False,
) -> ValidationResult:
    """Validate a config.yaml file.

    Args:
        path: Path to config.yaml
        auto_fix: If True, compute fixed values
        write_fixes: If True and auto_fix, write fixes back to file

    Returns:
        ValidationResult
    """
    path = Path(path)
    if not path.exists():
        return ValidationResult(
            valid=False,
            warnings=[],
            errors=[f"Config file not found: {path}"],
            fixed_values={},
        )

    with open(path) as f:
        cfg = yaml.safe_load(f)

    result = validate_config(cfg, auto_fix=auto_fix)

    if write_fixes and result.fixed_values:
        cfg.update(result.fixed_values)
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    return result


def print_validation_report(result: ValidationResult, verbose: bool = True) -> None:
    """Print a human-readable validation report."""
    if result.valid:
        print("Configuration: VALID")
    else:
        print("Configuration: INVALID")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for err in result.errors:
            print(f"  [ERROR] {err}")

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warn in result.warnings:
            print(f"  [WARN] {warn}")

    if result.fixed_values and verbose:
        print(f"\nSuggested fixes:")
        for key, val in result.fixed_values.items():
            print(f"  {key}: {val}")

    if not result.errors and not result.warnings:
        print("No issues found.")


def get_config_with_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return config with defaults filled in for missing keys.

    Args:
        cfg: Partial configuration dictionary

    Returns:
        Configuration with all defaults applied
    """
    result = {}
    for key, schema in CONFIG_SCHEMA.items():
        if key in cfg:
            result[key] = cfg[key]
        else:
            result[key] = schema["default"]
    # Preserve any extra keys not in schema
    for key in cfg:
        if key not in result:
            result[key] = cfg[key]
    return result


def main() -> None:
    """CLI entry point for config validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate csi-presence-node config")
    parser.add_argument(
        "config",
        nargs="?",
        default="csi_node/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix invalid values",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write fixes back to config file (requires --fix)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print if there are issues",
    )
    args = parser.parse_args()

    result = validate_config_file(
        args.config,
        auto_fix=args.fix,
        write_fixes=args.write and args.fix,
    )

    if not args.quiet or not result.valid or result.warnings:
        print_validation_report(result, verbose=args.fix)

    sys.exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()
