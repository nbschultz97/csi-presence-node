"""Environment doctor for the CSI presence demo.

Run ``python -m csi_node.doctor`` (or ``scripts/doctor.sh``) to verify that the
system is ready for FeitCSI capture. Designed for edge devices where sudo may be
restricted; reports actionable steps without mutating the system.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

CFG_PATH = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_FEITCSI_BIN = Path(os.environ.get("FEITCSI_BIN", "/usr/local/bin/feitcsi"))


@dataclass
class CheckResult:
    name: str
    ok: Optional[bool]
    detail: str
    hint: str | None = None


def _symbol(ok: Optional[bool]) -> str:
    if ok is True:
        return "[ok ]"
    if ok is False:
        return "[FAIL]"
    return "[warn]"


def _print_result(res: CheckResult) -> None:
    msg = f"{_symbol(res.ok)} {res.name}: {res.detail}"
    print(msg)
    if res.hint:
        print(f"       hint: {res.hint}")


def _check_python() -> CheckResult:
    ok = sys.version_info >= (3, 9)
    return CheckResult(
        "python",
        ok,
        f"{sys.version_info.major}.{sys.version_info.minor}",
        hint="Use Python 3.9+ (activate .venv if present)." if not ok else None,
    )


def _check_cfg(path: Path) -> CheckResult:
    if not path.exists():
        return CheckResult("config", False, f"missing {path}")
    try:
        yaml.safe_load(open(path, "r"))
        return CheckResult("config", True, "loaded", None)
    except Exception as exc:  # pragma: no cover - defensive
        return CheckResult("config", False, f"failed to parse: {exc}")


def _check_feitcsi(bin_path: Path) -> CheckResult:
    if not bin_path.exists():
        return CheckResult(
            "feitcsi",
            False,
            f"{bin_path} not found",
            hint="Build/install FeitCSI and set FEITCSI_BIN if installed elsewhere.",
        )
    if not os.access(bin_path, os.X_OK):
        return CheckResult(
            "feitcsi",
            False,
            f"{bin_path} is not executable",
            hint="Run chmod +x or reinstall FeitCSI.",
        )

    getcap = shutil.which("getcap")
    if not getcap:
        return CheckResult(
            "feitcsi", None, "getcap unavailable; capabilities not verified", "Install libcap2-bin."
        )
    try:
        out = subprocess.check_output([getcap, str(bin_path)], text=True).strip()
    except subprocess.CalledProcessError:
        return CheckResult(
            "feitcsi",
            None,
            "getcap failed; cannot verify capabilities",
            hint="Reinstall libcap2-bin or run sudo setcap cap_net_admin,cap_net_raw+eip <path>",
        )
    needs_caps = "cap_net_admin" in out and "cap_net_raw" in out
    return CheckResult(
        "feitcsi",
        needs_caps,
        "capabilities ok" if needs_caps else "capabilities missing",
        hint="Run scripts/preflight_root.sh or sudo setcap cap_net_admin,cap_net_raw+eip"
        f" {bin_path}",
    )


def _check_debugfs() -> CheckResult:
    debugfs = Path("/sys/kernel/debug")
    if not debugfs.exists():
        return CheckResult("debugfs", False, "missing /sys/kernel/debug mount")
    if not os.path.ismount(debugfs):
        return CheckResult(
            "debugfs",
            None,
            "not mounted",
            hint="Run scripts/preflight_root.sh (requires sudo) to mount debugfs.",
        )
    alt = list(Path("/sys/kernel/debug/ieee80211").glob("**/iwlwifi"))
    if not Path("/sys/kernel/debug/iwlwifi").exists() and alt:
        return CheckResult(
            "debugfs",
            None,
            f"iwlwifi debug path only under {alt[0]}",
            hint="Link the iwlwifi debug dir into /sys/kernel/debug/iwlwifi for FeitCSI",
        )
    return CheckResult("debugfs", True, "mounted")


def _check_iface(name: Optional[str]) -> CheckResult:
    if not name:
        return CheckResult("iface", None, "skipped (pass --iface to verify)")
    path = Path(f"/sys/class/net/{name}")
    if path.exists():
        return CheckResult("iface", True, f"{name} present")
    return CheckResult("iface", False, f"{name} not found", hint="List interfaces with ip link show")


def _check_logs(cfg: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    log_file = Path(cfg.get("log_file", "data/csi_raw.log"))
    out_file = Path(cfg.get("output_file", "data/presence_log.jsonl"))
    for label, path in (("input log", log_file), ("output", out_file)):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if path.parent.exists() and os.access(path.parent, os.W_OK):
            results.append(CheckResult(label, True, f"{path.parent} writable"))
        else:
            results.append(
                CheckResult(
                    label,
                    False,
                    f"{path.parent} not writable",
                    hint="Fix permissions or override paths via --log/--out",
                )
            )
    baseline = Path(cfg.get("baseline_file", "data/baseline.npz"))
    if baseline.exists():
        results.append(CheckResult("baseline", True, f"found {baseline}"))
    else:
        results.append(
            CheckResult("baseline", None, f"not found ({baseline})", hint="Optional: run python -m csi_node.baseline")
        )
    return results


def run_doctor(iface: Optional[str]) -> int:
    results: list[CheckResult] = []
    cfg_result = _check_cfg(CFG_PATH)
    results.append(_check_python())
    results.append(cfg_result)

    cfg: dict
    if cfg_result.ok:
        cfg = yaml.safe_load(open(CFG_PATH, "r"))
    else:
        cfg = {}

    results.append(_check_feitcsi(DEFAULT_FEITCSI_BIN))
    results.append(_check_debugfs())
    results.append(_check_iface(iface))
    results.extend(_check_logs(cfg))

    failures = 0
    warnings = 0
    for res in results:
        _print_result(res)
        if res.ok is False:
            failures += 1
        elif res.ok is None:
            warnings += 1

    if failures:
        print(f"\n{failures} check(s) failed. Fix the above and rerun.")
    elif warnings:
        print(f"\nAll mandatory checks passed with {warnings} warning(s).")
    else:
        print("\nEnvironment looks ready. Run ./scripts/demo.sh to verify capture.")

    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="CSI demo environment doctor")
    parser.add_argument("--iface", help="Verify that the interface exists", default=None)
    args = parser.parse_args()
    sys.exit(run_doctor(args.iface))


if __name__ == "__main__":
    main()
