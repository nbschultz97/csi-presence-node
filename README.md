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
