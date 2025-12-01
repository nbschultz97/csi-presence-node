#!/usr/bin/env bash
set -euo pipefail

# Lightweight environment doctor. Run without arguments for generic checks or
# pass --iface wlan0 to confirm the interface exists.
#
# Usage: ./scripts/doctor.sh [--iface wlan0]

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

if [[ -d "$REPO_ROOT/.venv" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

python -m csi_node.doctor "$@"
