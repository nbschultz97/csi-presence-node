#!/usr/bin/env bash
set -euo pipefail

# Launch the CSI Presence GUI with the project's virtualenv if present.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec python -m csi_node.gui

