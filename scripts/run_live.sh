#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data"
CSI_DAT="$DATA_DIR/csi_raw.dat"
CSI_LOG="$DATA_DIR/csi_raw.log"
STDLOG="$DATA_DIR/feitcsi.log"

CH=${1:-36}
WIDTH=${2:-20}

freq_from_channel() {
  local ch=$1
  if (( ch >= 1 && ch <= 13 )); then
    echo $((2407 + ch * 5))
  elif (( ch == 14 )); then
    echo 2484
  else
    echo $((5000 + ch * 5))
  fi
}

VENV_PY="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  VENV_PY="python3"
fi

cleanup() {
  local rc=$?
  echo "[CLEANUP] stopping background processes…"
  [[ -n "${CONV_PID:-}" ]] && kill "$CONV_PID" 2>/dev/null || true
  [[ -n "${FEIT_PID:-}" ]] && sudo kill "$FEIT_PID" 2>/dev/null || true
  wait "${CONV_PID:-}" 2>/dev/null || true
  wait "${FEIT_PID:-}" 2>/dev/null || true
  echo "[CLEANUP] re-enabling Wi‑Fi…"
  sudo rfkill unblock all || true
  sudo systemctl restart NetworkManager || true
  nmcli networking on || true
  nmcli radio wifi on || true
  exit "$rc"
}
trap cleanup INT TERM EXIT

mkdir -p "$DATA_DIR"

echo "[PRE] Unblock rfkill, load iwlwifi, mount debugfs, link iwlwifi, setcap…"
sudo rfkill unblock all || true
sudo modprobe iwlwifi || true
sudo mount -t debugfs debugfs /sys/kernel/debug || true
if [[ ! -d /sys/kernel/debug/iwlwifi ]]; then
  alt=$(sudo find /sys/kernel/debug/ieee80211 -maxdepth 3 -type d -name iwlwifi 2>/dev/null | head -n1 || true)
  if [[ -n "$alt" ]]; then
    sudo ln -sf "$alt" /sys/kernel/debug/iwlwifi || true
  fi
fi
sudo setcap cap_net_admin,cap_net_raw+eip /usr/local/bin/feitcsi || true

echo "[PRE] Disconnect managed Wi‑Fi and bring monitor up…"
nmcli dev disconnect FeitCSIap || true
sudo ip link set FeitCSImon up || true

FREQ=$(freq_from_channel "$CH")
echo "[CAP] Starting FeitCSI on ch $CH ($FREQ MHz), width $WIDTH → $CSI_DAT"
rm -f "$CSI_DAT" "$CSI_LOG"
(
  sudo /usr/local/bin/feitcsi -f "$FREQ" -w "$WIDTH" -o "$CSI_DAT" -v \
    2>&1 | tee -a "$STDLOG"
) &
FEIT_PID=$!

# Wait up to 12s for the .dat file to appear and become non-empty
for i in {1..24}; do
  if [[ -s "$CSI_DAT" ]]; then
    break
  fi
  sleep 0.5
done
if [[ ! -s "$CSI_DAT" ]]; then
  echo "[ERR] $CSI_DAT not created or still empty; recent FeitCSI log:" >&2
  tail -n 80 "$STDLOG" >&2 || true
  exit 1
fi

echo "[CONV] Streaming $CSI_DAT → $CSI_LOG"
"$VENV_PY" "$ROOT_DIR/scripts/dat2json_stream.py" --in "$CSI_DAT" --out "$CSI_LOG" &
CONV_PID=$!

echo "[PIPE] Launching pipeline TUI (log=$CSI_LOG)…"
"$VENV_PY" "$ROOT_DIR/run.py" --tui
