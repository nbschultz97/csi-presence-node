# Vantage Demo Setup Guide

## What This Demo Shows

**Laptop + WiFi adapter → detect human presence through a wall → show on screen.**

The system uses WiFi Channel State Information (CSI) to detect human presence without any sensors on the target side. It's passive, covert, and works through standard building materials.

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
# Clone and setup
git clone https://github.com/nbschultz97/csi-presence-node.git
cd csi-presence-node
pip install -r requirements.txt

# On Linux with FeitCSI hardware:
bash scripts/00_install_deps.sh
```

### 2. Launch the Web Dashboard

**Replay mode (no hardware needed — great for showing the UI):**
```bash
python run.py --dashboard --replay data/sample_csi.b64
```

**Live mode (with FeitCSI-compatible WiFi adapter):**
```bash
python run.py --dashboard --log data/csi_raw.log
# In another terminal, start capture:
sudo bash scripts/10_csi_capture.sh
```

**With the Tkinter GUI (Linux):**
```bash
python -m csi_node.gui
```

### 3. Open Dashboard

Navigate to **http://localhost:8088** in any browser.

You'll see:
- 🎯 **Large presence indicator** — CLEAR (green) or DETECTED (red, pulsing)
- 📊 **Real-time metrics** — energy ratio, variance ratio, spectral analysis
- 📈 **60-second history chart** — confidence, energy, and variance over time
- 🏃 **Movement classification** — stationary, moving, or breathing detected
- 📋 **Event log** — timestamped detection events

## Demo Hardware Requirements

| Component | Recommended | Notes |
|-----------|-------------|-------|
| Laptop | Any Linux laptop | Ubuntu 22.04+ preferred |
| WiFi Adapter | Intel AX200/AX210 | Must support CSI extraction |
| FeitCSI | v1.0+ | CSI extraction firmware/driver |

### Minimum Setup
- Linux laptop with Intel WiFi (AX200/AX210)
- FeitCSI installed and configured
- Python 3.10+

## Calibration (Recommended)

For best results, calibrate in the actual demo environment:

### Option A: Web Dashboard Calibration
1. Ensure the room on the other side of the wall is **empty**
2. Click **"Calibrate Now"** button on the dashboard
3. Wait 30 seconds
4. Calibration saves automatically to `data/calibration.json`

### Option B: Command-Line Calibration
```bash
# Record empty-room baseline (60 seconds)
python -m csi_node.baseline --log data/csi_raw.log --duration 60 --out data/baseline.npz

# Calibrate RSSI-to-distance (optional, for distance estimates)
# Capture at 1m and 3m known distances:
python -m csi_node.calibrate --log1 data/cal_1m.log --d1 1.0 --log2 data/cal_3m.log --d2 3.0 --config csi_node/config.yaml
```

## Demo Script (Talking Points)

### Setup Phase (before audience)
1. Position laptop on one side of a wall
2. Start capture + dashboard
3. Calibrate (empty room)
4. Verify dashboard shows CLEAR

### Demo Flow
1. **"This is a standard laptop with WiFi. No special sensors, no radar."**
2. **"The wall between us and the next room is [drywall/concrete/etc]."**
3. **"Watch the dashboard..."** → Have someone walk into the room
4. **Dashboard shows DETECTED** with confidence rising
5. **"Notice the movement classification"** → person walks around vs stands still
6. **"This runs at ~30 frames/sec, real-time detection"**
7. **"The system is completely passive — no signals emitted, no hardware on the other side"**

### Key Differentiators to Highlight
- **10x cheaper** than radar alternatives (Lumineye: $60-85K, Vantage: $5-10K)
- **Passive/covert** — no RF emissions to detect
- **Networked** — ATAK/CoT integration for tactical overlay
- **Through standard walls** — drywall, wood, concrete block

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No packets | Check FeitCSI is running: `scripts/10_csi_capture.sh` |
| Dashboard won't connect | Check port 8088 isn't in use: `lsof -i :8088` |
| Low confidence | Recalibrate in the demo environment |
| False positives | Increase `variance_threshold` in `config.yaml` |
| WiFi adapter not found | Run `iw dev` to list interfaces |

## ATAK Integration (Optional)

Enable CoT streaming for tactical overlay:

```yaml
# In csi_node/config.yaml:
atak_enabled: true
atak_port: 4242
sensor_uid: "vantage-001"
sensor_callsign: "VANTAGE-1"
sensor_lat: 40.7128    # Set to demo location
sensor_lon: -74.0060
sensor_heading: 90.0   # Degrees from north
```

Presence detections will appear as markers in ATAK/WinTAK.

## File Structure

```
csi-presence-node/
├── run.py                  # Main entry point
├── DEMO.md                 # This file
├── csi_node/
│   ├── config.yaml         # Configuration
│   ├── pipeline.py         # Core processing pipeline
│   ├── presence.py         # Multi-method presence detector
│   ├── web_dashboard.py    # Web-based real-time dashboard
│   ├── gui.py              # Tkinter GUI (Linux)
│   ├── atak.py             # ATAK/CoT integration
│   ├── preprocessing.py    # Signal conditioning (Hampel + Butterworth)
│   ├── pose_classifier.py  # Pose classification (standing/crouching/prone)
│   ├── calibrate.py        # RSSI distance calibration
│   └── baseline.py         # Empty-room baseline recording
├── data/
│   ├── sample_csi.b64      # Sample data for replay demos
│   └── calibration.json    # Saved calibration (auto-generated)
└── scripts/
    ├── 10_csi_capture.sh   # Start CSI capture
    └── demo.sh             # Quick demo launcher
```
