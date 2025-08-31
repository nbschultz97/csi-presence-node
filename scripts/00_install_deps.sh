#!/usr/bin/env bash
set -euo pipefail

# Install Python dependencies. Installs into a virtual environment when
# VIRTUAL_ENV is set or a PEP 668 `EXTERNALLY-MANAGED` marker is present.
# Otherwise falls back to per-user installation via --user.
PYTHON=${PYTHON:-python3}
PIP_FLAGS=(--upgrade)
if [[ -z "${VIRTUAL_ENV:-}" ]] && ! ${PYTHON} - <<'PY' >/dev/null 2>&1
import sysconfig, pathlib, sys
marker = pathlib.Path(sysconfig.get_path("purelib")).with_name("EXTERNALLY-MANAGED")
sys.exit(0 if marker.exists() else 1)
PY
then
  PIP_FLAGS=(--user "${PIP_FLAGS[@]}")
fi
${PYTHON} -m pip install "${PIP_FLAGS[@]}" numpy scipy pandas csikit watchdog pyyaml >/dev/null

echo "Kernel: $(uname -r)"
if lsmod | grep -q iwlwifi; then
  echo "iwlwifi module loaded"
else
  echo "iwlwifi module not loaded; run 'modprobe iwlwifi'" >&2
fi
