# csi-presence-node

Minimal CSI presence and pose logger for Intel AX210 using [FeitCSI](https://github.com/KuskoSoft/FeitCSI).

## Hardware

* Intel AX210 or other NIC capable of CSI extraction
* System with monitor-mode permissions

## Quick start

Clone the repository and run the single setup script. It installs Python
dependencies, verifies hardware, builds FeitCSI if missing and launches the
realtime pipeline:

```bash
./setup_and_run.sh
```

Example output:

```json
{"timestamp": "2024-01-01T00:00:00", "presence": true, "pose": "standing", "direction": "L", "confidence": 0.73}
```

All frames are also appended to `logs/session_<date>.json` for later analysis.

## FeitCSI install notes

The setup script checks `pip show feitcsi`. If the package is absent it clones
`https://github.com/KuskoSoft/FeitCSI`, builds the binary with `cargo build
--release` and installs the Python bindings with `pip install .` from the
`python` subdirectory.

## Configuration

Edit `csi_node/config.yaml` to adjust thresholds, window sizes and paths.
Baseline and CSV outputs rotate when exceeding 1 MB. When only a single RSSI
chain is reported, direction defaults to `C`.

## Tests

Offline regression test using a bundled sample log:

```bash
python tests/test_offline.py
```

## Troubleshooting

* Ensure the wireless interface supports monitor mode and appears in `iw dev`.
* Run the script with sufficient permissions to switch channels.
* If FeitCSI capture fails, re-check driver patches and firmware.
* On `Run scripts/10_csi_capture.sh first`, ensure CSI capture is active and
  writing to the configured log.
