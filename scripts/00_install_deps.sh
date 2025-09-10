#!/usr/bin/env bash
set -euo pipefail

# Install Python dependencies. Uses per-user installs (`pip --user`) by default
# so it plays nicely with PEP 668 `EXTERNALLY-MANAGED` Python builds. When a
# virtual environment is active (`$VIRTUAL_ENV` is set), the `--user` flag is
# dropped so packages land inside the venv.
PYTHON=${PYTHON:-python3}
PIP_FLAGS=(--upgrade --user)
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PIP_FLAGS=(--upgrade)
elif ${PYTHON} - <<'PY' >/dev/null 2>&1; then
import sysconfig, pathlib, sys
marker = pathlib.Path(sysconfig.get_path("purelib")).with_name("EXTERNALLY-MANAGED")
sys.exit(0 if marker.exists() else 1)
PY
  echo "Detected PEP 668 EXTERNALLY-MANAGED; using --user install" >&2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/../requirements.txt"

WHEEL_DIR_ENV=${WHEEL_DIR:-}
WHEEL_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --wheel-dir)
      WHEEL_DIR="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [--wheel-dir DIR]" >&2
      exit 1
      ;;
  esac
done
WHEEL_DIR=${WHEEL_DIR:-$WHEEL_DIR_ENV}
OFFLINE=${OFFLINE:-0}
if [[ -n "${WHEEL_DIR}" ]]; then
  if [[ -d "${WHEEL_DIR}" ]]; then
    PIP_FLAGS+=(--no-index --find-links "$WHEEL_DIR")
  else
    echo "Warning: wheel directory '${WHEEL_DIR}' not found" >&2
    if [[ "${OFFLINE}" != "0" ]]; then
      echo "Offline mode set; aborting" >&2
      exit 1
    else
      echo "Falling back to PyPI" >&2
    fi
  fi
fi

${PYTHON} -m pip install "${PIP_FLAGS[@]}" -r "$REQ_FILE" >/dev/null

echo "Kernel: $(uname -r)"
if lsmod | grep -q iwlwifi; then
  echo "iwlwifi module loaded"
else
  echo "iwlwifi module not loaded; run 'modprobe iwlwifi'" >&2
fi
