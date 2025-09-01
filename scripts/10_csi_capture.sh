#!/usr/bin/env bash
set -euo pipefail

FEITCSI_BIN=${FEITCSI_BIN:-feitcsi}

if ! { [ -x "$FEITCSI_BIN" ] || command -v "$FEITCSI_BIN" >/dev/null 2>&1; }; then
  echo "FeitCSI binary not found; set FEITCSI_BIN=/path/to/feitcsi" >&2
  exit 1
fi

CHANNEL=${1:-36}
WIDTH=${2:-80}
CODING=${3:-${FEITCSI_CODING:-BCC}}
LOG=./data/csi_raw.log

mkdir -p ./data

echo "Starting FeitCSI capture on channel $CHANNEL width $WIDTH MHz coding $CODING"
# Placeholder command; replace with actual FeitCSI binary if needed
$FEITCSI_BIN -c "$CHANNEL" -w "$WIDTH" --coding "$CODING" -o "$LOG"
