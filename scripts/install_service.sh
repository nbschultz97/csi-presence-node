#!/bin/bash
# Install csi-presence-node as a systemd service
#
# Usage:
#   sudo ./scripts/install_service.sh
#
# This script:
#   1. Copies files to /opt/csi-presence-node
#   2. Creates virtual environment and installs dependencies
#   3. Installs systemd service
#   4. Creates necessary directories and permissions

set -e

# Check for root
if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root (use sudo)"
    exit 1
fi

INSTALL_DIR="/opt/csi-presence-node"
SERVICE_FILE="/etc/systemd/system/csi-presence-node.service"
LOG_DIR="/var/log/csi-presence"
DATA_DIR="$INSTALL_DIR/data"

echo "=== CSI Presence Node Service Installer ==="
echo ""

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(dirname "$SCRIPT_DIR")"

echo "[1/7] Creating installation directory..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$LOG_DIR"

echo "[2/7] Copying files..."
cp -r "$SOURCE_DIR/csi_node" "$INSTALL_DIR/"
cp -r "$SOURCE_DIR/scripts" "$INSTALL_DIR/"
cp -r "$SOURCE_DIR/tests" "$INSTALL_DIR/" 2>/dev/null || true
cp "$SOURCE_DIR/requirements.txt" "$INSTALL_DIR/"
cp "$SOURCE_DIR/setup.sh" "$INSTALL_DIR/" 2>/dev/null || true
cp "$SOURCE_DIR/run.py" "$INSTALL_DIR/"
cp "$SOURCE_DIR/README.md" "$INSTALL_DIR/" 2>/dev/null || true

# Create data directory
mkdir -p "$DATA_DIR"
mkdir -p "$DATA_DIR/baselines"
mkdir -p "$DATA_DIR/runs"

echo "[3/7] Setting up Python virtual environment..."
python3 -m venv "$INSTALL_DIR/.venv"
source "$INSTALL_DIR/.venv/bin/activate"
pip install --upgrade pip
pip install -r "$INSTALL_DIR/requirements.txt"
deactivate

echo "[4/7] Installing systemd service..."
cp "$SOURCE_DIR/systemd/csi-presence-node.service" "$SERVICE_FILE"

# Update paths in service file
sed -i "s|/opt/csi-presence-node|$INSTALL_DIR|g" "$SERVICE_FILE"

echo "[5/7] Setting permissions..."
# Set ownership (keeping root for now, can be changed to dedicated user)
chown -R root:root "$INSTALL_DIR"
chmod 755 "$INSTALL_DIR"
chmod -R 755 "$INSTALL_DIR/scripts"
chmod 644 "$SERVICE_FILE"

# Log directory permissions
chmod 755 "$LOG_DIR"

echo "[6/7] Reloading systemd..."
systemctl daemon-reload

echo "[7/7] Validating installation..."
# Test Python imports
"$INSTALL_DIR/.venv/bin/python" -c "from csi_node import pipeline, atak, udp_streamer, config_validator" && echo "  Python modules: OK"

# Validate config
"$INSTALL_DIR/.venv/bin/python" -m csi_node.config_validator "$INSTALL_DIR/csi_node/config.yaml" --quiet || echo "  Config validation: Warnings present (see above)"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "To configure the service, edit:"
echo "  $INSTALL_DIR/csi_node/config.yaml"
echo ""
echo "To start the service:"
echo "  sudo systemctl start csi-presence-node"
echo ""
echo "To enable on boot:"
echo "  sudo systemctl enable csi-presence-node"
echo ""
echo "To view logs:"
echo "  sudo journalctl -u csi-presence-node -f"
echo ""
echo "Service status:"
echo "  sudo systemctl status csi-presence-node"
echo ""

# Optional: Start service
read -p "Start service now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    systemctl start csi-presence-node
    systemctl status csi-presence-node
fi
