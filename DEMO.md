# Vantage Demo Setup Guide

## What This Demo Shows

**Laptop + WiFi adapter → detect human presence through a wall → show on screen.**

The system uses WiFi Channel State Information (CSI) to detect human presence without any sensors on the target side. It's passive, covert, and works through standard building materials.

## Quick Start (5 minutes)

### 1. Pre-Flight Check

```bash
# Clone and setup
git clone https://github.com/nbschultz97/csi-presence-node.git
cd csi-presence-node
pip install -r requirements.txt

# Verify everything is ready
python run.py --preflight
```

On **Windows (PowerShell)** — one command does everything:
```powershell
.\demo.ps1                     # Simulation mode
.\demo.ps1 -ThroughWall       # Through-wall demo
.\demo.ps1 -Live -Log .\data\csi_raw.log   # Live mode
```

On **Linux** with FeitCSI hardware:
```bash
bash scripts/00_install_deps.sh
```

### 2. Launch the Web Dashboard

**🎯 Demo mode (no hardware needed — recommended for first demo):**
```bash
python run.py --demo
```
This generates synthetic CSI data with realistic scenarios: empty room → person enters → movement → breathing → exits. Loops automatically.

**Replay mode (using recorded data):**
```bash
python run.py --dashboard --replay data/demo_csi.log
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
- 📈 **60-second history chart** — confidence, energy, variance, and spectral over time
- 🗺️ **Spatial zone view** — sensor, wall, and detected target position
- 🌡️ **Subcarrier energy heatmap** — per-subcarrier activity visualization
- 🏃 **Movement classification** — stationary, moving, or breathing detected
- 📋 **Event log** — timestamped detection events

### New Dashboard Features

**Detection Profiles** — Switch between profiles via the toolbar dropdown:
- **Default** — balanced sensitivity for general use
- **Through-Wall** — lower thresholds optimized for attenuated through-wall signals
- **Same Room** — higher thresholds to reduce false positives in same-room scenarios
- **High Sensitivity** — maximum sensitivity for weak signals or long range

**Data Recording** — Click "⏺ Record Data" to save labeled CSI frames for training ML models. Recordings are saved to `data/recordings/` as JSONL files with timestamps, raw CSI, RSSI, and labels.

**Real-time Streaming** — Dashboard uses Server-Sent Events (SSE) for instant updates with automatic reconnection. Falls back to polling for older browsers.

**Calibration Progress** — Visual progress bar during the 30-second calibration period.

**Multi-Zone Detection** — Splits subcarriers into near/mid/far zones for coarse spatial localization. Shows per-zone confidence bars, primary occupied zone, and total zone count. Useful for demonstrating "the system can tell roughly where in the room they are."

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
2. Click **"📐 Calibrate"** button in the toolbar
3. Watch the progress bar — wait 30 seconds
4. Calibration saves automatically to `data/calibration.json`

### Option B: Command-Line Calibration
```bash
# Record empty-room baseline (60 seconds)
python -m csi_node.baseline --log data/csi_raw.log --duration 60 --out data/baseline.npz

# Calibrate RSSI-to-distance (optional, for distance estimates)
# Capture at 1m and 3m known distances:
python -m csi_node.calibrate --log1 data/cal_1m.log --d1 1.0 --log2 data/cal_3m.log --d2 3.0 --config csi_node/config.yaml
```

## Collecting Training Data

The dashboard's recording feature lets you build labeled datasets for training ML models:

1. Start the dashboard in live mode
2. Click **"⏺ Record Data"** — enter a label (e.g., "empty", "walking", "standing")
3. Perform the labeled activity for 30-60 seconds
4. Click **"⏹ Stop Recording"**
5. Repeat with different labels
6. Recordings are saved to `data/recordings/<label>_<timestamp>.jsonl`

Each recording contains per-frame entries with raw CSI amplitudes, RSSI, timestamps, and your label — ready for model training.

## Demo Script (Talking Points)

### Setup Phase (before audience)
1. Position laptop on one side of a wall
2. Start capture + dashboard
3. Select **Through-Wall** profile from the dropdown
4. Calibrate (empty room)
5. Verify dashboard shows CLEAR and heatmap is quiet

### Demo Flow
1. **"This is a standard laptop with WiFi. No special sensors, no radar."**
2. **"The wall between us and the next room is [drywall/concrete/etc]."**
3. **"Watch the dashboard..."** → Have someone walk into the room
4. **Dashboard shows DETECTED** with confidence rising, heatmap lights up
5. **"Notice the subcarrier heatmap"** → shows which frequencies are affected
6. **"The spatial view shows estimated position"** → target icon moves
7. **"Notice the movement classification"** → person walks around vs stands still
8. **"This runs at ~30 frames/sec, real-time detection"**
9. **"The system is completely passive — no signals emitted, no hardware on the other side"**

### Key Differentiators to Highlight
- **10x cheaper** than radar alternatives (Lumineye: $60-85K, Vantage: $5-10K)
- **Passive/covert** — no RF emissions to detect
- **Networked** — ATAK/CoT integration for tactical overlay
- **Through standard walls** — drywall, wood, concrete block
- **Multiple detection modes** — energy, variance, spectral (breathing detection)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No packets | Check FeitCSI is running: `scripts/10_csi_capture.sh` |
| Dashboard won't connect | Check port 8088 isn't in use: `lsof -i :8088` |
| Low confidence | Switch to **Through-Wall** or **High Sensitivity** profile |
| False positives | Switch to **Same Room** profile or recalibrate |
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
├── run.py                  # Main entry point (--demo, --dashboard, --preflight)
├── demo.ps1                # Windows PowerShell quick-start
├── DEMO.md                 # This file
├── csi_node/
│   ├── config.yaml         # Configuration
│   ├── pipeline.py         # Core processing pipeline
│   ├── presence.py         # Multi-method presence detector (energy/variance/spectral)
│   ├── web_dashboard.py    # Web dashboard with SSE, heatmap, zone viz
│   ├── gui.py              # Tkinter GUI (Linux)
│   ├── atak.py             # ATAK/CoT integration
│   ├── preprocessing.py    # Signal conditioning (Hampel + Butterworth)
│   ├── simulator.py        # Synthetic CSI generator for demos
│   ├── pose_classifier.py  # Pose classification (standing/crouching/prone)
│   ├── zone_detector.py    # Multi-zone spatial detection (near/mid/far)
│   ├── preflight.py        # Pre-flight demo readiness check
│   ├── calibrate.py        # RSSI distance calibration
│   └── baseline.py         # Empty-room baseline recording
├── data/
│   ├── demo_csi.log        # Sample data for replay demos
│   ├── calibration.json    # Saved calibration (auto-generated)
│   └── recordings/         # Labeled training data (from dashboard recording)
└── scripts/
    ├── 10_csi_capture.sh   # Start CSI capture
    └── demo.sh             # Quick demo launcher
```
