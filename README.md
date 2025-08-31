# csi-presence-node

Minimal CSI presence logger for Intel AX210 using FeitCSI.

## Setup

1. Install dependencies and verify driver:
   ```bash
   scripts/00_install_deps.sh
   ```
2. Start CSI capture (channel 36 / 80 MHz by default):
   ```bash
   scripts/10_csi_capture.sh 36 80
   ```
3. In an empty room, record baseline:
   ```bash
   python -m csi_node.baseline --duration 60
   ```
4. Run the realtime pipeline:
   ```bash
   python -m csi_node.pipeline
   ```

Output is written to `./data/presence_log.csv`.

## Configuration

Edit `csi_node/config.yaml` for thresholds, windows, and file paths. Baseline and output files rotate when exceeding 1 MB.

## Tests

Offline test using a bundled sample log:
```bash
python tests/test_offline.py
```

## Acceptance tests

* Empty room for 60 s ⇒ `presence` stays `0`.
* Walk in front for 30 s ⇒ `presence` flips to `1` for most windows.
* Stand left/right/center of antenna ⇒ `direction` mostly `L`/`R`/`C` respectively.
* CPU < 20% on one core at 10 Hz windows, memory stable.
