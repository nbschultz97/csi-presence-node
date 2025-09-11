# csi-presence-node

Minimal CSI presence logger for Intel AX210 using FeitCSI.

## Setup

1. Install dependencies and verify driver. The script drops `--user` when a
   virtual environment is active (`$VIRTUAL_ENV`); otherwise it installs with
   `pip --user`, which also respects PEP 668 `EXTERNALLY-MANAGED` markers:
   ```bash
   scripts/00_install_deps.sh
   ```
2. Ensure the FeitCSI binary is installed. If it's not on your `PATH`, export `FEITCSI_BIN` to its absolute path:
   ```bash
   export FEITCSI_BIN=/path/to/feitcsi
   ```
3. Start CSI capture (channel 36 / 80 MHz / BCC coding by default):
   ```bash
   scripts/10_csi_capture.sh 36 80 LDPC
   ```
   The third argument or `FEITCSI_CODING` env var selects the coding scheme
   (`LDPC` or `BCC`), which the script forwards to FeitCSI via `--coding`.
   On launch the script checks `[ -x "$FEITCSI_BIN" ]` or
   `command -v "$FEITCSI_BIN"` to ensure the binary is available. If
   missing, it prints `FeitCSI binary not found; set
   FEITCSI_BIN=/path/to/feitcsi` and exits non-zero.
4. In an empty room, record baseline:
   ```bash
   python -m csi_node.baseline --duration 60
   ```
5. Run the realtime pipeline:
   ```bash
   python -m csi_node.pipeline
   ```
   The pipeline aborts with `Run scripts/10_csi_capture.sh first` if the CSI
   log file is missing.

Output is written to `./data/presence_log.csv`.

## Quick Demo (Pose + TUI)

Run the live pipeline with pose classification and the curses dashboard:

```bash
./scripts/demo.sh --with-pose
```

If live capture fails or you want to use a recorded log:

```bash
./scripts/demo.sh --with-pose --replay data/sample_csi.dat
```

The demo prints presence, direction, pose and confidence to the terminal and
logs results to `data/presence_log.csv`.

## Configuration

Edit `csi_node/config.yaml` for thresholds, windows, and file paths. Baseline and output files rotate when exceeding 1 MB.
If the capture reports only a single RSSI chain, the pipeline records `NaN`
for `rssi0`/`rssi1` and defaults the direction label to `C`.

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

## Troubleshooting

If the baseline recorder or pipeline prints `Run scripts/10_csi_capture.sh first`,
start CSI capture with `scripts/10_csi_capture.sh` so the log file exists before
retrying.

## Training a better pose model

Collect training data as `data/pose_train.npz` with arrays `X` and `y` then
run:

```bash
python -m csi_node.pose_classifier --train --in data/pose_train.npz --out models/wipose.joblib
```

Place the resulting model under `models/` and rerun the demo to use it.
