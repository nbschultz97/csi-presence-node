#!/usr/bin/env bash
# Automated setup and launch script for the CSI presence node.
#
# The script verifies Python and pip, creates a virtual environment, installs
# runtime dependencies, ensures FeitCSI is available, performs a quick capture
# test and then launches the main pipeline.

set -e

# Check Python version
if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 not found" >&2
    exit 1
fi
PY_OK=$(python3 - <<'PY'
import sys
print(int(sys.version_info >= (3,10)))
PY
)
if [ "$PY_OK" -ne 1 ]; then
    echo "Python 3.10+ required" >&2
    exit 1
fi

# Check pip
if ! python3 -m pip --version >/dev/null 2>&1; then
    echo "pip for Python 3 not installed" >&2
    exit 1
fi

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pandas numpy matplotlib torch scikit-learn PyQt5 tk >/dev/null 2>&1 || true

# Ensure FeitCSI is installed
if ! python -m pip show feitcsi >/dev/null 2>&1; then
    echo "FeitCSI not found; cloning and building..."
    git clone --depth 1 https://github.com/KuskoSoft/FeitCSI.git
    (
        cd FeitCSI && \
        cargo build --release && \
        cd python && python -m pip install .
    )
fi

# Verify CSI-capable NIC
if ! iw dev 2>/dev/null | grep -q Interface; then
    echo "No CSI-capable wireless interface detected (e.g. Intel AX210)" >&2
    exit 1
fi

# Quick capture test
TEST_LOG=$(mktemp)
if feitcsi capture --count 1 --out "$TEST_LOG" >/dev/null 2>&1; then
    echo "FeitCSI capture test succeeded" >&2
else
    echo "FeitCSI capture test failed" >&2
    exit 1
fi

# Launch main application
python -m csi_node.pipeline
