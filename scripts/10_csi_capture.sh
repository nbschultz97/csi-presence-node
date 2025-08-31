#!/usr/bin/env bash
set -euo pipefail

CHANNEL=${1:-36}
WIDTH=${2:-80}
LOG=./data/csi_raw.log
FEITCSI_BIN=${FEITCSI_BIN:-feitcsi}

mkdir -p ./data

echo "Starting FeitCSI capture on channel $CHANNEL width $WIDTH MHz"
# Placeholder command; replace with actual FeitCSI binary if needed
$FEITCSI_BIN -c "$CHANNEL" -w "$WIDTH" -o "$LOG"
