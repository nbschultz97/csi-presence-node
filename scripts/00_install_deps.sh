#!/usr/bin/env bash
set -euo pipefail

# Install Python dependencies
PYTHON=${PYTHON:-python3}
${PYTHON} -m pip install --user --upgrade numpy scipy pandas csikit watchdog pyyaml >/dev/null

echo "Kernel: $(uname -r)"
if lsmod | grep -q iwlwifi; then
  echo "iwlwifi module loaded"
else
  echo "iwlwifi module not loaded; run 'modprobe iwlwifi'" >&2
fi
