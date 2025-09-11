#!/bin/bash
set -e

# Example: ./scripts/demo.sh --with-pose --replay data/sample_csi.b64

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

OUT="data/presence_log.csv"
mkdir -p data

POSE_FLAG=""
if [[ $WITH_POSE -eq 1 ]]; then
  POSE_FLAG="--pose"
fi

if [[ -n "$REPLAY" ]]; then
  python -m csi_node.pipeline $POSE_FLAG --tui --replay "$REPLAY" --out "$OUT"
else
  python -m csi_node.pipeline $POSE_FLAG --tui --window 3.0 --out "$OUT"
fi
