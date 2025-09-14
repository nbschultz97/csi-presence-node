#!/usr/bin/env bash
set -euo pipefail

# Configure passwordless sudo for the CSI GUI.
# Usage: ./scripts/setup_passwordless.sh <username>

USER_NAME=${1:-}
if [[ -z "$USER_NAME" ]]; then
  echo "Usage: $0 <username>" >&2
  exit 1
fi

out_file="/etc/sudoers.d/csi-presence-node"

resolve() {
  local cmd="$1"; shift
  if command -v "$cmd" >/dev/null 2>&1; then
    command -v "$cmd"
  else
    # Fallback to common locations
    case "$cmd" in
      rfkill|modprobe|setcap|iw|ip) echo "/usr/sbin/$cmd" ;;
      systemctl|mount|ln|nmcli) echo "/usr/bin/$cmd" ;;
      *) echo "$cmd" ;;
    esac
  fi
}

RFKILL=$(resolve rfkill)
MODPROBE=$(resolve modprobe)
SETCAP=$(resolve setcap)
IW=$(resolve iw)
IP=$(resolve ip)
SYSTEMCTL=$(resolve systemctl)
MOUNT=$(resolve mount)
LN=$(resolve ln)
NMCLI=$(resolve nmcli)
FEITCSI=${FEITCSI_BIN:-/usr/local/bin/feitcsi}

CONTENT="${USER_NAME} ALL=(root) NOPASSWD: ${RFKILL}, ${MODPROBE}, ${SETCAP}, ${IW}, ${IP}, ${SYSTEMCTL}, ${MOUNT}, ${LN}, ${NMCLI}, ${FEITCSI}"

# Must be root to write sudoers; if not, re-exec via pkexec or sudo
if [[ $EUID -ne 0 ]]; then
  if command -v pkexec >/dev/null 2>&1; then
    exec pkexec bash "$0" "$USER_NAME"
  else
    exec sudo bash "$0" "$USER_NAME"
  fi
fi

set -x
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
printf '%s\n' "$CONTENT" >"$tmp"
install -m 0440 -o root -g root "$tmp" "$out_file"
visudo -cf "$out_file"
set +x
echo "Passwordless sudo configured for $USER_NAME."
