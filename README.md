# csi-presence-node

Vantage Scanner demo node. Streams WiFi CSI from an Intel AX210 via
[FeitCSI](https://github.com/KuskoSoft/FeitCSI) and outputs presence, direction
and a coarse pose estimate (standing / crouching / prone).

## Hardware

* Intel AX210 NIC with two antennas
* Kali/Linux or similar with kernel iwlwifi driver
* Optional: display for curses TUI

## Quick start

The `setup.sh` script installs dependencies, builds FeitCSI and creates a Python
virtual environment:

```bash
./setup.sh
source .venv/bin/activate
```

Verify the install with the bundled demo which captures for ten seconds and
prints JSON results:

```bash
./scripts/demo.sh --with-pose
```

## Running

Start live capture using the Python interface (replace `wlan0` with your AX210
interface):

```bash
python run.py --iface wlan0 --tui --pose
```

For offline playback of a saved log:

```bash
python run.py --replay data/sample_csi.b64 --tui --pose
```

Each window emits a JSON object like:

```json
{"timestamp": "2024-01-01T00:00:00", "presence": true,
 "pose": "standing", "direction": "left", "distance_m": 1.6,
 "confidence": 0.78}
```

Logs are written to `data/presence_log.jsonl` and rotate when exceeding 1 MB.

Baseline amplitudes for an empty room improve stability:

```bash
python -m csi_node.baseline --duration 60
```

## Demo Best Practices (through‑wall and room demos)

Follow this checklist for reliable results:

- Use the GUI and click Tools → “Through‑Wall Preset”. This sets 2.4 GHz
  channel 1, 20 MHz, enables Dat mode, applies an RSSI offset, and uses a
  steadier 2.5 s window. Direction threshold (`rssi_delta`) is raised for
  stability.
- Capture a baseline in an empty room if feasible:
  - Run: `python -m csi_node.baseline --duration 60` (from the repo)
  - This improves presence stability and reduces idle false positives.
- Calibrate distance in the same environment:
  - In the GUI: Tools → “Calibrate Distance…”, supply two short logs at known
    distances (e.g., 1.0 m and 3.0 m). The GUI writes `tx_power_dbm` and
    `path_loss_exponent` into `csi_node/config.yaml` and shows a calibration
    badge.
- Use the “Live Tracking” window for audience display:
  - Tools → “Show Tracking Window” — this shows only the latest frame
    (presence, direction, distance, confidence) without log spam.
- If direction flickers, raise `rssi_delta` (e.g., 3.5 → 4.5 dB). If presence
  is too sensitive, raise `variance_threshold` and `pca_threshold` a notch.

Troubleshooting quick fixes:
- If distance looks too small or doesn’t change with motion, (re)run
  calibration in the demo location.
- If presence stays true when nobody moves, record a baseline and increase
  `variance_threshold`/`pca_threshold`. A longer window (2.5–3.0 s) also helps.
- Prefer 2.4 GHz (ch1/20 MHz) through one interior wall for stronger signals.

## Capabilities and limits

- Presence: detects motion-induced changes in Wi‑Fi channels using CSI variance
  and PCA. It will respond strongly to a moving person, but also to moving
  objects (e.g., doors, fans). It does not classify “human vs object”. When a
  person becomes completely still, presence confidence will fall over time.
- Direction: coarse left/center/right based on the difference between the two
  antenna chains’ average RSSI. This is a relative left/right cue, not an
  absolute angle.
- Distance: ballpark estimate from average RSSI using a log‑distance model.
  Requires calibration for your room to be meaningful. It is not centimeter‑
  accurate ranging, but a rough “nearer/farther” indicator.
- Through‑wall: Wi‑Fi penetrates most interior walls; presence and coarse
  direction typically still work through one wall, but confidence and distance
  accuracy degrade. Prefer 2.4 GHz (e.g., channel 1, 20 MHz) for better
  penetration. Results depend on construction and interference.
- Multiple people: the current demo is single‑target. Multiple movers will
  produce a combined response; it cannot count people or track them
  independently.
- Confidence: an exponential moving average (EMA) of the binary presence signal
  (0/1). It is a stability score, not a calibrated probability.

For safety‑critical or security use, additional sensing and validation are
recommended. This repo is a demo/prototype.

### Direction and distance notes

- Direction is inferred from per‑chain RSSI delta. In file/"Dat mode", the
  converter now estimates RSSI per chain from CSI magnitudes so direction works
  without the live extractor.
- Distance uses a crude log‑distance model driven by average RSSI; tune it for
  your room via calibration below.

#### Calibrate distance (tx_power, path loss exponent)

Collect two short CSI logs at known distances (e.g., 1.0 m and 3.0 m):

```bash
# Start capture (Dat mode creates data/csi_raw.log with rssi values)
./scripts/10_csi_capture.sh &
sleep 8; cp data/csi_raw.log data/csi_raw_d1.log  # at distance d1
# Move to distance d2, wait a few seconds
sleep 8; cp data/csi_raw.log data/csi_raw_d2.log  # at distance d2
kill %1 || true

# Estimate parameters and write them to config
python -m csi_node.calibrate \
  --log1 data/csi_raw_d1.log --d1 1.0 \
  --log2 data/csi_raw_d2.log --d2 3.0 \
  --config csi_node/config.yaml
```

The pipeline reads `tx_power_dbm` and `path_loss_exponent` from
`csi_node/config.yaml` to map RSSI to meters.

### Calibration pipeline explained

Calibration here is not “training a model” — it sets two parameters for the
RSSI→distance curve and optionally establishes a static CSI baseline:

- Baseline: `python -m csi_node.baseline --duration 60` records empty‑room
  CSI to subtract static structure. Improves presence stability.
- Distance calibration: `python -m csi_node.calibrate` (also available from
  the GUI Tools menu) computes:
  - `path_loss_exponent` n from two positions at distances d1, d2
  - `tx_power_dbm` (expected RSSI at 1 m)
  These are written into `csi_node/config.yaml`. Re‑running calibration updates
  the parameters; it does not accumulate a dataset.

You can also “replay” saved logs via the GUI (Replay mode) or CLI to validate
changes. Replay does not change calibration by itself — only the calibrator
script writes new parameters.

### Thresholds guide

Most behavior is controlled by a few intuitive thresholds (editable from the
GUI: Tools → “Edit Thresholds…”):

- `variance_threshold` (default 5.0): motion sensitivity from raw CSI
  amplitude variance. Increase to reduce false positives; decrease to catch
  subtler motion.
- `pca_threshold` (default 1.0): motion sensitivity from the first PCA
  component; complements variance. Increase to be stricter.
- `rssi_delta` (default 2.0 dB): direction sensitivity. Larger values make
  left/right decisions fewer and more stable; smaller values make the system
  switch sides more readily.
- `tx_power_dbm` and `path_loss_exponent`: set by the calibration helper.
  These control the mapping from RSSI to meters.
- `dat_rssi_offset` (GUI field “RSSI offset”): shifts the derived RSSI scale in
  Dat mode so numbers look dBm‑ish. It does not affect relative direction but
  does affect distance unless you calibrate; after calibration, this offset is
  less important.
- `window_size` / `window_hop`: temporal smoothing vs. responsiveness. Larger
  windows are stabler; smaller windows respond faster.

Recommended starting points for through‑wall demos:

- Channel: 1 (2.412 GHz), Width: 20 MHz — better penetration, less DFS.
- Run baseline with no people moving.
- Calibrate distance once at ~1 m and ~3 m on your demo path.
- If direction flickers, raise `rssi_delta` to ~3–4 dB.
- If presence is too jumpy, raise `variance_threshold` slightly (e.g., +1).

### Typical demo flow (GUI)

1) Open the GUI. In Live Settings, check “Dat mode”. Set “RSSI offset” ≈ −60.
2) Tools → “Edit Thresholds…” if you want to pre‑tune `rssi_delta` or others.
3) Click Start. Verify presence toggles with motion and left/right flips as you
   move.
4) Tools → “Calibrate Distance…”: select two short logs at 1 m and 3 m,
   enter distances, and apply. Distance numbers will be more meaningful.
5) For through‑wall, set Channel 1 / Width 20 MHz and place the device so it
   receives traffic passing through the wall; repeat calibration if the setting
   changes.

## GUI Reference

- Live Settings: pick device, channel, width, and toggle Dat mode; set RSSI
  offset (Dat mode only) and optional pose.
- Controls: Start/Stop, Autoscroll, copy/save/clear log. A status line shows
  passwordless sudo and calibration status.
- Tools menu:
  - Diagnostics: collects a detailed system/driver/capture log file under
    `data/`.
  - Setup Passwordless sudo…: guided sudoers entry for smoother capture.
  - Fix Wi‑Fi Profile…: clears stale interface bindings then reconnects.
  - Through‑Wall Preset: applies stable through‑wall defaults.
  - Calibrate Distance…: estimates `tx_power_dbm` and `path_loss_exponent`
    from two logs at known distances and writes them to the config.
  - Edit Thresholds…: updates `variance_threshold`, `pca_threshold`,
    `rssi_delta`, `tx_power_dbm`, `path_loss_exponent`, and `dat_rssi_offset`.
  - Show Tracking Window: opens a clean live view of the latest frame.
  - Instructions: opens this README for quick guidance.

## GUI launcher

A minimal Tk GUI is included to start/stop capture and replay without the
terminal.

Run it:

```bash
# Install Tk if missing (Debian/Ubuntu/Kali)
sudo apt-get install -y python3-tk

# Launch the GUI (uses the project virtualenv if present)
./scripts/launch_gui.sh
```

Add a desktop shortcut:

```bash
# If your checkout path differs, edit Exec/Path inside:
sed -n '1,120p' scripts/csi-presence-gui.desktop

# Then copy to your user applications dir
cp scripts/csi-presence-gui.desktop ~/.local/share/applications/
```

Notes:
- Live capture requires the FeitCSI binary and NIC privileges. Grant caps once
  to avoid sudo: `sudo setcap cap_net_admin,cap_net_raw+eip /usr/local/bin/feitcsi`.
- The GUI streams logs and JSON outputs; it avoids the curses TUI.
- Auto‑preflight: in Live mode the GUI will automatically disable Wi‑Fi,
  unblock rfkill, load `iwlwifi`, set the reg domain, and attempt to grant
  capabilities to FeitCSI before starting capture. It retries widths (chosen →
  80 → 40 → 20 MHz) until `data/csi_raw.log` is detected.
- Auto‑postflight: when you click Stop, the GUI re‑enables Wi‑Fi, restarts
  NetworkManager (via systemd or service), turns networking back on, and tries
  to reconnect to your previously active Wi‑Fi connection(s).
- Diagnostics: click “Diagnostics” to generate `data/diagnostics-YYYYMMDD-HHMMSS.txt`
  with system info, Wi‑Fi state, FeitCSI caps, recent logs, and filtered
  kernel messages. The GUI opens the file after saving.

### Permissions and password prompts

FeitCSI relies on iwlwifi debugfs at `/sys/kernel/debug/iwlwifi` and wireless
privileges. The GUI tries to run preflight and capture with `pkexec` (GUI
authorization) or `sudo -n` where possible. To minimize prompts:

- Grant FeitCSI capabilities (once):

  sudo setcap cap_net_admin,cap_net_raw+eip /usr/local/bin/feitcsi

- Ensure debugfs is mounted (the GUI attempts this):

  sudo mount -t debugfs debugfs /sys/kernel/debug

- Optional: allow passwordless sudo for specific commands. Create a file via
  `sudo visudo -f /etc/sudoers.d/csi-presence-node` with:

  Cmnd_Alias CSI_CMDS = \
    /usr/sbin/rfkill unblock all, \
    /usr/sbin/modprobe iwlwifi, \
    /usr/sbin/setcap cap_net_admin,cap_net_raw+eip /usr/local/bin/feitcsi, \
    /usr/bin/iw reg set US, \
    /bin/systemctl restart NetworkManager, \
    /bin/systemctl start NetworkManager, \
    /usr/bin/mount -t debugfs debugfs /sys/kernel/debug, \
    /usr/local/bin/feitcsi *

  yourusername ALL=(root) NOPASSWD: CSI_CMDS

Replace `yourusername` with your login name. Reopen the GUI after changes.

If reconnection does not happen automatically
- Your desktop may require interactive authorization for restarting
  NetworkManager. In that case, either approve the `pkexec` prompt or pre‑grant
  permissions, or reconnect manually with:
  - `nmcli networking on && nmcli radio wifi on`
  - `nmcli connection up id <YourSSID>` (or `nmcli connection up uuid <UUID>`)

## Demo mode

`demo.sh` runs a quick self‑test: it checks the FeitCSI binary, captures CSI for
10 s and tails the JSON log. Use `--replay <file>` to run the pipeline on a
recorded capture instead.

## Tests

An offline regression test is included:

```bash
python tests/test_offline.py
```

## Configuration

Edit `csi_node/config.yaml` for thresholds and file paths. Results include
presence flag, left/center/right direction, coarse pose label and a crude
RSSI‑based distance estimate.

## License

MIT
