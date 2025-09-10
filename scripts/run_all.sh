#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
cd "$REPO_ROOT"

LOG_FILE="${REPO_ROOT}/data/csi_raw.log"

cleanup() {
  if [[ -n "${CAPTURE_PID:-}" ]] && kill -0 "$CAPTURE_PID" 2>/dev/null; then
    echo "Stopping capture..."
    kill "$CAPTURE_PID"
    wait "$CAPTURE_PID" 2>/dev/null || true
  fi
}
trap 'cleanup; exit 1' INT TERM
trap cleanup EXIT

"$SCRIPT_DIR/10_csi_capture.sh" "$@" &
CAPTURE_PID=$!

echo "Waiting for $LOG_FILE..."
while [[ ! -f "$LOG_FILE" ]]; do
  sleep 1
done

python3 -m csi_node.baseline
python3 -m csi_node.pipeline
