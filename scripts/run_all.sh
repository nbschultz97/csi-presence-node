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
while [[ ! -s "$LOG_FILE" ]]; do
  if ! kill -0 "$CAPTURE_PID" 2>/dev/null; then
    echo "FeitCSI exited before log was ready" >&2
    wait "$CAPTURE_PID" 2>/dev/null || true
    exit 1
  fi
  sleep 1
done

set +e
python3 -m csi_node.baseline
BASELINE_EXIT=$?
python3 -m csi_node.pipeline
PIPELINE_EXIT=$?
set -e

if [[ $BASELINE_EXIT -ne 0 ]]; then
  echo "Baseline exited with code $BASELINE_EXIT" >&2
fi
if [[ $PIPELINE_EXIT -ne 0 ]]; then
  echo "Pipeline exited with code $PIPELINE_EXIT" >&2
fi

if [[ $BASELINE_EXIT -ne 0 ]]; then
  exit "$BASELINE_EXIT"
elif [[ $PIPELINE_EXIT -ne 0 ]]; then
  exit "$PIPELINE_EXIT"
fi
