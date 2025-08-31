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
${PYTHON} -m pip install "${PIP_FLAGS[@]}" numpy scipy pandas csikit watchdog pyyaml >/dev/null

echo "Kernel: $(uname -r)"
if lsmod | grep -q iwlwifi; then
  echo "iwlwifi module loaded"
else
  echo "iwlwifi module not loaded; run 'modprobe iwlwifi'" >&2
fi
