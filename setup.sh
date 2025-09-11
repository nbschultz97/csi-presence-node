#!/bin/bash
set -e

# Automated installer for Vantage Scanner demo on Kali/Linux.
# Creates Python venv, builds FeitCSI and ensures drivers.

sudo apt-get update
sudo apt-get install -y git build-essential python3-venv python3-pip libpcap-dev

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [[ ! -d FeitCSI ]]; then
  git clone https://github.com/KuskoSoft/FeitCSI.git
fi
cd FeitCSI
make
sudo make install
cd ..

sudo modprobe iwlwifi || true

echo "Install complete. Run 'source .venv/bin/activate' and './scripts/demo.sh' to test."
