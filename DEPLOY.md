# VANTAGE System - Hardware Deployment Guide

## System Requirements (MANDATORY)

### Hardware Requirements
- **Intel AX210 WiFi 6E NIC** (REQUIRED - only chip with FeitCSI support)
  - Must have **2 antennas** connected for direction detection
  - PCIe or M.2 (CNVio2) form factor supported
  - Verify: `lspci | grep -i "AX210"`

- **Host System**
  - x86_64 CPU (Intel/AMD)
  - 2GB+ RAM
  - Linux kernel 5.10+ (tested on Kali, Ubuntu 20.04+)

- **Optional**: Display for GUI/TUI (headless operation via JSON logs supported)

### Software Requirements
- **OS**: Kali Linux / Ubuntu / Debian-based distro
- **Kernel Driver**: `iwlwifi` (usually built-in)
- **Packages**: `build-essential`, `libpcap-dev`, `python3`, `git`
- **FeitCSI**: Custom CSI extraction tool (compiled during setup)

---

## Pre-Deployment Checklist

### 1. Verify Hardware Present
```bash
# Check for Intel AX210
lspci | grep -i "ax210\|wireless"
# Expected output: "Intel Corporation Wi-Fi 6 AX210"

# Check antennas connected (both chains should show signal)
iw dev wlan0 station dump  # Should show 2 chains with RSSI values
```

### 2. Verify Kernel Driver Loaded
```bash
# Check iwlwifi loaded
lsmod | grep iwlwifi
# Expected output: iwlwifi module listed

# If not loaded:
sudo modprobe iwlwifi
```

### 3. Verify Debugfs Access (CRITICAL)
```bash
# Check debugfs mounted
mount | grep debugfs
# Expected: debugfs on /sys/kernel/debug

# If not mounted:
sudo mount -t debugfs debugfs /sys/kernel/debug

# Check iwlwifi debugfs interface exists
ls -la /sys/kernel/debug/iwlwifi/
# OR
ls -la /sys/kernel/debug/ieee80211/phy*/iwlwifi/
```

**If `/sys/kernel/debug/iwlwifi/` doesn't exist, FeitCSI CANNOT capture CSI data.**

---

## Installation Steps

### Step 1: Clone Repository
```bash
git clone https://github.com/nbschultz97/csi-presence-node.git
cd csi-presence-node
```

### Step 2: Run Automated Setup
```bash
# This script:
# - Installs system dependencies
# - Clones and compiles FeitCSI from source
# - Installs FeitCSI to /usr/local/bin/feitcsi
# - Creates Python virtualenv
# - Installs Python packages
./setup.sh

# Activate virtualenv
source .venv/bin/activate
```

### Step 3: Grant FeitCSI Capabilities (REQUIRED)
```bash
# Allow FeitCSI to access raw network without sudo
sudo setcap cap_net_admin,cap_net_raw+eip /usr/local/bin/feitcsi

# Verify capabilities set
getcap /usr/local/bin/feitcsi
# Expected: /usr/local/bin/feitcsi cap_net_admin,cap_net_raw=eip
```

### Step 4: Run Preflight Checks
```bash
# Run as root to prepare system
sudo ./scripts/preflight_root.sh
# This script:
# - Unblocks rfkill (if WiFi blocked)
# - Loads iwlwifi module
# - Sets regulatory domain to US
# - Mounts debugfs
# - Creates symlink if needed for iwlwifi debugfs path
# - Grants FeitCSI capabilities
```

### Step 5: Optional - Passwordless Sudo Setup
```bash
# Avoids password prompts during capture
./scripts/setup_passwordless.sh
# OR manually add to /etc/sudoers.d/csi-presence-node:
# yourusername ALL=(root) NOPASSWD: /usr/sbin/rfkill, /usr/sbin/modprobe, /usr/local/bin/feitcsi
```

---

## Hardware Validation Tests

### Test 1: Verify FeitCSI Binary
```bash
which feitcsi
# Expected: /usr/local/bin/feitcsi

feitcsi --help
# Should show usage without errors
```

### Test 2: Test Live CSI Capture (10 seconds)
```bash
# This is the CRITICAL test - if this fails, hardware/driver is not working
./scripts/demo.sh
# Expected output:
# - "Starting FeitCSI capture..."
# - data/csi_raw.dat created and growing
# - JSON presence output printed
# - No errors about missing iwlwifi debugfs
```

**If `demo.sh` fails with "iwlwifi debugfs not readable":**
- Run `sudo ./scripts/preflight_root.sh` first
- Check `ls -la /sys/kernel/debug/iwlwifi/`
- Ensure Intel AX210 is present and not disabled in BIOS

### Test 3: Verify Dual-Antenna Setup
```bash
# Start capture
./scripts/10_csi_capture.sh 36 80 &
sleep 5

# Check CSI data has 2 chains (antennas)
head -20 data/csi_raw.log | grep -o '"rssi":\[[^]]*\]'
# Expected: "rssi":[val1, val2]  <- Two values means dual-antenna working

# Kill capture
pkill feitcsi
```

---

## Deployment Modes

### Mode 1: Live Capture (Field Use)
```bash
# GUI mode (recommended for initial setup)
./scripts/launch_gui.sh

# CLI mode
python run.py --iface wlan0 --pose

# Headless mode (no TUI, JSON output only)
python run.py --iface wlan0 --pose --no-tui > output.log 2>&1
```

### Mode 2: Replay (Testing/Development)
```bash
# Test with bundled sample data (no hardware required)
python run.py --replay data/sample_csi.b64 --pose
```

---

## Field Calibration (REQUIRED FOR ACCURATE METRICS)

### 1. Capture Baseline (Empty Room)
```bash
# IMPORTANT: Room must be completely still, no movement for 60s
# Captures ambient CSI signature to subtract from detections

# CLI:
python -m csi_node.baseline --duration 60

# OR GUI: Tools → "Capture Baseline (60s)"
```

### 2. Calibrate Distance (Path Loss Model)
```bash
# CRITICAL: Your current config has INVALID calibration:
# path_loss_exponent: -0.0911  (WRONG - should be 2.0 to 4.0)
# tx_power_dbm: -24.81         (very low)

# Method A: GUI Calibration Wizard
# Tools → "Calibration Wizard"
# - Prompts for 2 distances (e.g., 1.0m and 3.0m)
# - Auto-captures CSI at each position
# - Computes path loss parameters

# Method B: Manual CLI Calibration
# Step 1: Start capture
./scripts/10_csi_capture.sh &
sleep 5

# Step 2: Stand at 1.0m from sensor, capture for 8 seconds
sleep 8
cp data/csi_raw.log data/cal_1m.log

# Step 3: Move to 3.0m from sensor
sleep 8
cp data/csi_raw.log data/cal_3m.log

# Step 4: Stop capture
pkill feitcsi

# Step 5: Compute calibration
python -m csi_node.calibrate \
  --log1 data/cal_1m.log --d1 1.0 \
  --log2 data/cal_3m.log --d2 3.0 \
  --config csi_node/config.yaml

# Verify new parameters in config.yaml
grep -E "(tx_power|path_loss)" csi_node/config.yaml
```

### 3. Tune Thresholds for Through-Wall Operation
```bash
# For through-wall detection, use 2.4 GHz (better penetration)
# Edit config.yaml or use GUI:

# Recommended through-wall settings:
# - channel: 1 (2.412 GHz)
# - bandwidth: 20 (MHz)
# - window_size: 2.5 (seconds, more stable)
# - variance_threshold: 5.0 (increase if too sensitive)
# - rssi_delta: 3.5 (increase to 4.5 if direction flickers)

# Apply via GUI: Tools → "Through-Wall Preset"
```

---

## Output Data Format (ATAK Integration)

### JSON Output Schema
Each detection window produces one JSON line in `data/presence_log.jsonl`:

```json
{
  "timestamp": "2025-01-07T12:34:56Z",
  "presence": true,
  "pose": "STANDING",
  "direction": "left",
  "distance_m": 2.3,
  "confidence": 0.85
}
```

**Field Definitions:**
- `timestamp`: ISO 8601 UTC time
- `presence`: boolean - true if human movement detected
- `pose`: string - "STANDING" | "CROUCHING" | "PRONE" | "n/a"
- `direction`: string - "left" | "center" | "right" (relative to sensor)
- `distance_m`: float - estimated distance in meters (requires calibration)
- `confidence`: float 0.0-1.0 - exponential moving average of presence signal

### ATAK Integration Options

**Option A: Direct JSON Streaming**
```bash
# Tail live JSON log
tail -f data/presence_log.jsonl | your-atak-parser
```

**Option B: TCP Socket (Requires Custom Script)**
```bash
# Stream JSON over TCP to ATAK client
tail -f data/presence_log.jsonl | nc <atak-host> <port>
```

**Option C: CoT XML Conversion (Recommended)**
```python
# Example CoT converter (pseudo-code)
import json
import socket
import datetime

def json_to_cot(data):
    """Convert VANTAGE JSON to Cursor-on-Target XML"""
    if not data["presence"]:
        return None

    cot_xml = f"""<?xml version="1.0"?>
<event version="2.0" uid="VANTAGE-TARGET-01"
       type="a-h-G" how="m-g"
       time="{data['timestamp']}"
       start="{data['timestamp']}"
       stale="{stale_time}">
  <point lat="{lat}" lon="{lon}" hae="0.0" ce="10.0" le="10.0"/>
  <detail>
    <contact callsign="HUMAN-{data['pose']}"/>
    <track speed="0.0" course="{heading_from_direction(data['direction'])}"/>
    <remarks>Distance: {data['distance_m']:.1f}m, Confidence: {data['confidence']:.2f}</remarks>
  </detail>
</event>"""
    return cot_xml

# Use with: tail -f data/presence_log.jsonl | python cot_bridge.py
```

---

## Troubleshooting

### Problem: "FeitCSI binary not found"
**Cause**: FeitCSI not installed or not in PATH
**Fix**:
```bash
# Check if installed
which feitcsi
ls -la /usr/local/bin/feitcsi

# If missing, rerun setup
cd FeitCSI && make && sudo make install
```

### Problem: "iwlwifi debugfs not readable"
**Cause**: Debugfs not mounted or iwlwifi not loaded
**Fix**:
```bash
sudo mount -t debugfs debugfs /sys/kernel/debug
sudo modprobe iwlwifi
ls -la /sys/kernel/debug/iwlwifi/  # Must exist
```

### Problem: "No CSI data captured"
**Cause**: Intel AX210 not present, disabled, or wrong NIC
**Fix**:
```bash
lspci | grep -i wireless
# Must show "Intel Corporation Wi-Fi 6 AX210"

# Check if disabled in BIOS/UEFI
rfkill list
# If blocked, run: sudo rfkill unblock all
```

### Problem: "Distance always shows tiny values (e-15)"
**Cause**: Distance calibration not performed or failed
**Fix**: Re-run calibration with actual hardware at known distances (see Field Calibration above)

### Problem: "Direction always shows 'center'"
**Cause**: Only one antenna connected, or RSSI delta too high
**Fix**:
```bash
# Check both antennas have signal
iw dev wlan0 station dump

# Lower rssi_delta threshold
# Edit csi_node/config.yaml: rssi_delta: 1.5
```

### Problem: "Presence stays true even when empty"
**Cause**: No baseline captured, or high ambient interference
**Fix**:
```bash
# Capture fresh baseline in empty, still environment
python -m csi_node.baseline --duration 60

# Increase sensitivity thresholds
# Edit config.yaml:
# variance_threshold: 7.0
# pca_threshold: 1.5
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VANTAGE SYSTEM                           │
│                                                             │
│  ┌──────────────┐    ┌─────────────┐    ┌──────────────┐  │
│  │ Intel AX210  │───>│  iwlwifi    │───>│   debugfs    │  │
│  │   (NIC)      │    │  (driver)   │    │  interface   │  │
│  └──────────────┘    └─────────────┘    └───────┬──────┘  │
│                                                  │          │
│                                                  v          │
│                                         ┌────────────────┐  │
│                                         │    FeitCSI     │  │
│                                         │  (CSI extract) │  │
│                                         └────────┬───────┘  │
│                                                  │          │
│                                                  v          │
│                                         ┌────────────────┐  │
│                                         │  csi_raw.dat   │  │
│                                         │  (binary CSI)  │  │
│                                         └────────┬───────┘  │
│                                                  │          │
│                                                  v          │
│                                         ┌────────────────┐  │
│                                         │ dat2json.py    │  │
│                                         │  (converter)   │  │
│                                         └────────┬───────┘  │
│                                                  │          │
│                                                  v          │
│                                         ┌────────────────┐  │
│                                         │ csi_raw.log    │  │
│                                         │  (JSON CSI)    │  │
│                                         └────────┬───────┘  │
│                                                  │          │
│                                                  v          │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              CSI Processing Pipeline                  │ │
│  │  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌────────┐ │ │
│  │  │ Variance│─>│   PCA   │─>│   RSSI   │─>│  Pose  │ │ │
│  │  │ Detect  │  │ Detect  │  │ Distance │  │ Class  │ │ │
│  │  └─────────┘  └─────────┘  └──────────┘  └────────┘ │ │
│  └───────────────────────────┬───────────────────────────┘ │
│                              │                             │
│                              v                             │
│                    ┌──────────────────┐                    │
│                    │ presence_log.jsonl│                   │
│                    │  (Intelligence)  │                    │
│                    └────────┬─────────┘                    │
│                             │                              │
└─────────────────────────────┼──────────────────────────────┘
                              │
                              v
                    ┌──────────────────┐
                    │  ATAK / CoT      │
                    │  Integration     │
                    └──────────────────┘
```

---

## Performance Notes

### Through-Wall Capability
- **1 interior wall**: Good presence/direction, degraded distance accuracy
- **2+ walls**: Marginal, highly dependent on construction
- **Best config**: 2.4 GHz (ch 1), 20 MHz, window_size 2.5s

### Pose Detection Accuracy
- **STANDING**: Best accuracy (largest CSI signature)
- **CROUCHING**: Moderate accuracy
- **PRONE**: Lowest accuracy (smallest CSI signature)
- **Note**: Current model is TOY MODEL using LogisticRegression on 3 synthetic points
  - For production, train on real labeled CSI data

### Single-Target Limitation
- System tracks **one combined signal**
- Multiple moving targets produce **merged response**
- Cannot count or independently track multiple people

### Confidence Metric
- **NOT** a calibrated probability
- Exponential moving average (EMA) of binary presence signal
- Stability indicator: high = sustained detection, low = intermittent

---

## Security Considerations

### Passive Operation (Zero Emission)
- System **does NOT transmit** WiFi signals
- Only **receives and analyzes** ambient WiFi CSI
- Covert ISR capability (no detectable RF signature from VANTAGE)

### Access Control
- FeitCSI requires root or `cap_net_admin,cap_net_raw` capabilities
- Limit access to authorized operators only
- Consider SELinux/AppArmor policies for production

### Data Handling
- JSON logs contain PII (human presence/pose/location)
- Implement encryption for stored logs in field deployment
- Secure CoT/ATAK transmission channels (TLS/VPN)

---

## Known Limitations

1. **Hardware locked**: Only works with Intel AX210 (FeitCSI requirement)
2. **Single target**: Cannot track multiple people independently
3. **Motion required**: Stationary targets fade over time (not radar)
4. **No human/object classification**: Responds to any moving RF reflector
5. **Distance accuracy**: Crude RSSI-based estimate, not precision ranging
6. **Through-wall degradation**: Signal loss through dense materials (concrete, metal)
7. **Interference sensitivity**: Performance degrades in high-WiFi-density environments

---

## Support & Development

- **Repository**: https://github.com/nbschultz97/csi-presence-node
- **Issues**: Report hardware issues, bugs, feature requests on GitHub
- **FeitCSI upstream**: https://github.com/KuskoSoft/FeitCSI

---

**Last Updated**: 2026-01-07
**VANTAGE System - Deployable MVP v1.0**
