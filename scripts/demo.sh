#!/bin/bash
set -e

# Quick verification run. Captures CSI for 10s and prints JSON outputs.
# Usage: ./scripts/demo.sh [--with-pose] [--replay LOG]

WITH_POSE=0
REPLAY=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --with-pose)
      WITH_POSE=1
      shift
      ;;
    --replay)
      REPLAY="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

OUT="data/presence_log.jsonl"
mkdir -p data
POSE_FLAG=""
if [[ $WITH_POSE -eq 1 ]]; then
  POSE_FLAG="--pose"
fi

if [[ -n "$REPLAY" ]]; then
  python run.py $POSE_FLAG --tui --replay "$REPLAY" --out "$OUT"
else
  BIN="${FEITCSI_BIN:-feitcsi}"
  if ! command -v "$BIN" >/dev/null 2>&1; then
    echo "FeitCSI binary '$BIN' not found" >&2
    exit 1
  fi
  echo "Starting FeitCSI for 10s..."
  timeout 12 scripts/10_csi_capture.sh >/dev/null 2>&1 &
  CAP_PID=$!
  sleep 2
  timeout 10 python run.py $POSE_FLAG --tui --out "$OUT" || true
  wait $CAP_PID || true
fi

echo "Last log entries:"
tail -n 5 "$OUT" || true
