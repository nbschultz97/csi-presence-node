#!/usr/bin/env bash
set -euo pipefail

# Install Python dependencies. If executed inside a virtual environment
# (VIRTUAL_ENV is set), install packages into it; otherwise fall back to a
# per-user installation via --user.
PYTHON=${PYTHON:-python3}
PIP_FLAGS=(--upgrade)
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  PIP_FLAGS=(--user "${PIP_FLAGS[@]}")
fi
${PYTHON} -m pip install "${PIP_FLAGS[@]}" numpy scipy pandas csikit watchdog pyyaml >/dev/null

echo "Kernel: $(uname -r)"
if lsmod | grep -q iwlwifi; then
  echo "iwlwifi module loaded"
else
  echo "iwlwifi module not loaded; run 'modprobe iwlwifi'" >&2
fi
