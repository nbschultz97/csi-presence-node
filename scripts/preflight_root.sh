#!/usr/bin/env bash
set -euo pipefail

# Run privileged preflight steps required for FeitCSI capture.
# - Unblock rfkill
# - Load iwlwifi
# - Set reg domain
# - Mount debugfs
# - Ensure FeitCSI binary has required capabilities

RFKILL=${RFKILL:-$(command -v rfkill || echo /usr/sbin/rfkill)}
MODPROBE=${MODPROBE:-$(command -v modprobe || echo /usr/sbin/modprobe)}
IW=${IW:-$(command -v iw || echo /usr/sbin/iw)}
MOUNT=${MOUNT:-$(command -v mount || echo /usr/bin/mount)}
SETCAP=${SETCAP:-$(command -v setcap || echo /usr/sbin/setcap)}
FEITCSI_BIN=${FEITCSI_BIN:-/usr/local/bin/feitcsi}

echo "[root] rfkill unblock all"
"$RFKILL" unblock all || true

echo "[root] modprobe iwlwifi"
"$MODPROBE" iwlwifi || true

echo "[root] iw reg set US"
"$IW" reg set US || true

echo "[root] mount debugfs /sys/kernel/debug"
mountpoint -q /sys/kernel/debug || "$MOUNT" -t debugfs debugfs /sys/kernel/debug || true

# Some distros expose iwlwifi debugfs under /sys/kernel/debug/ieee80211/phyX/iwlwifi
if [[ ! -d /sys/kernel/debug/iwlwifi ]]; then
  alt=$(find /sys/kernel/debug/ieee80211 -maxdepth 3 -type d -name iwlwifi 2>/dev/null | head -n1 || true)
  if [[ -n "$alt" && -d "$alt" ]]; then
    echo "[root] Linking $alt -> /sys/kernel/debug/iwlwifi"
    ln -s "$alt" /sys/kernel/debug/iwlwifi 2>/dev/null || true
  fi
fi

echo "[root] setcap on FeitCSI ($FEITCSI_BIN)"
"$SETCAP" cap_net_admin,cap_net_raw+eip "$FEITCSI_BIN" || true

echo "[root] preflight complete"
