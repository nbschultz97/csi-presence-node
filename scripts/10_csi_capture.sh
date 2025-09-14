#!/usr/bin/env bash
set -euo pipefail

FEITCSI_BIN=${FEITCSI_BIN:-/usr/local/bin/feitcsi}

echo "FEITCSI_BIN is set to '$FEITCSI_BIN'"

if [ -e "$FEITCSI_BIN" ]; then
  if [ ! -x "$FEITCSI_BIN" ]; then
    echo "FeitCSI binary exists at '$FEITCSI_BIN' but is not executable" >&2
    echo "Fix permissions or set FEITCSI_BIN to an executable binary" >&2
    exit 1
  fi
elif bin_path=$(command -v "$FEITCSI_BIN" 2>/dev/null); then
  FEITCSI_BIN="$bin_path"
  echo "Resolved FeitCSI binary to '$FEITCSI_BIN'"
else
  echo "FeitCSI binary not found or not executable (FEITCSI_BIN='$FEITCSI_BIN')" >&2
  echo "Run 'which feitcsi' or install it at /usr/local/bin/feitcsi" >&2
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

DATA_DIR=./data
DAT=$DATA_DIR/csi_raw.dat
JSONL=$DATA_DIR/csi_raw.log
STDLOG=$DATA_DIR/feitcsi.log

mkdir -p "$DATA_DIR"

freq=$(channel_to_freq "$CHANNEL")
echo "Starting FeitCSI capture on channel $CHANNEL (${freq} MHz) width $WIDTH MHz coding $CODING → $DAT" | tee -a "$STDLOG"

# Hint if permissions are likely to fail
if [[ ! -r /sys/kernel/debug/iwlwifi && ! -L /sys/kernel/debug/iwlwifi ]]; then
  echo "[warn] /sys/kernel/debug/iwlwifi not readable; run scripts/preflight_root.sh with sudo/pkexec" | tee -a "$STDLOG"
fi

# Start FeitCSI writing .dat (may require sudo/pkexec). Pipe stderr to log.
set +e
"$FEITCSI_BIN" -f "$freq" -w "$WIDTH" --coding "$CODING" -o "$DAT" 2>>"$STDLOG" &
FEIT_PID=$!
set -e

# Wait for .dat to appear and be non-empty
for i in {1..24}; do
  if [[ -s "$DAT" ]]; then
    break
  fi
  sleep 0.5
done
if [[ ! -s "$DAT" ]]; then
  echo "[error] $DAT not created or still empty; FeitCSI may require root. Try: sudo bash scripts/preflight_root.sh, then rerun this script with sudo." | tee -a "$STDLOG" >&2
  exit 1
fi

echo "Converting $DAT → $JSONL (streaming)" | tee -a "$STDLOG"
python3 "$(dirname "$0")/dat2json_stream.py" --in "$DAT" --out "$JSONL" &
CONV_PID=$!

echo "Press Ctrl+C to stop. Logs: $STDLOG"
trap 'kill $FEIT_PID $CONV_PID 2>/dev/null || true; wait $FEIT_PID $CONV_PID 2>/dev/null || true' INT TERM
wait $FEIT_PID
kill $CONV_PID 2>/dev/null || true
wait $CONV_PID 2>/dev/null || true
