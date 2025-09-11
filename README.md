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
