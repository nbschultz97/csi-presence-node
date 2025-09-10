#!/usr/bin/env bash
set -euo pipefail

FEITCSI_BIN=${FEITCSI_BIN:-feitcsi}

if ! { [ -x "$FEITCSI_BIN" ] || command -v "$FEITCSI_BIN" >/dev/null 2>&1; }; then
  echo "FeitCSI binary not found; set FEITCSI_BIN=/path/to/feitcsi" >&2
  exit 1
fi

channel_to_freq() {
  local ch=$1
  local freq
  if (( ch >= 1 && ch <= 13 )); then
    freq=$((2407 + ch * 5))
  elif (( ch == 14 )); then
    freq=2484
  else
    freq=$((5000 + ch * 5))
  fi
  echo "$freq"
}

CHANNEL=${1:-36}
WIDTH=${2:-80}
CODING=${3:-${FEITCSI_CODING:-BCC}}
CODING=${CODING^^}
LOG=./data/csi_raw.log
STDLOG=./data/feitcsi.log

mkdir -p ./data

freq=$(channel_to_freq "$CHANNEL")
echo "Starting FeitCSI capture on channel $CHANNEL (${freq} MHz) width $WIDTH MHz coding $CODING" | tee -a "$STDLOG"

tmp_err=$(mktemp)
(
  "$FEITCSI_BIN" -f "$freq" -w "$WIDTH" --coding "$CODING" -o "$LOG" \
    2> >(tee "$tmp_err" | tee -a "$STDLOG" >&2) | tee -a "$STDLOG"
) &
feitcsi_pid=$!

sleep 1
if [[ ! -s "$LOG" ]]; then
  echo "FeitCSI failed to produce $LOG" >&2
  cat "$tmp_err" >&2 || true
  kill "$feitcsi_pid" 2>/dev/null || true
  wait "$feitcsi_pid" 2>/dev/null || true
  rm -f "$tmp_err"
  exit 1
fi

rm -f "$tmp_err"
wait "$feitcsi_pid"
