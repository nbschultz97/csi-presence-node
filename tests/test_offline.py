"""Offline pipeline test for ``tests/data/sample.log``.

The log contains 10 packets: five baseline frames followed by five motion
frames with higher variance and a left RSSI bias. Presence should be flagged
on five frames and the direction distribution should be {"C": 6, "L": 4,
"R": 0}. Set ``CSI_TEST_OFFLINE_PRINT=1`` to dump the DataFrame and metrics
for manual inspection during development.
"""

import os
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import yaml
from csi_node import pipeline


# Expected counts derived from ``sample.log``. The first five packets are
# baseline (presence==0) and the last five represent motion biased left.
EXPECTED_PRESENCE_COUNT = 5
EXPECTED_DIRECTION_COUNTS = {"C": 6, "L": 4, "R": 0}


def _run() -> None:
    """Run pipeline on sample.log and optionally print summary metrics."""
    cfg = yaml.safe_load(open("csi_node/config.yaml"))
    df = pipeline.run_offline("tests/data/sample.log", cfg)

    # Count frames where presence was detected.
    presence_count = int(df["presence"].eq(1).sum())
    # Direction label distribution across all frames.
    direction_counts = df["direction"].value_counts().to_dict()

    if os.getenv("CSI_TEST_OFFLINE_PRINT"):
        print(df)
        print("presence==1:", presence_count)
        print("direction counts:", direction_counts)

    assert (
        presence_count == EXPECTED_PRESENCE_COUNT
    ), f"expected {EXPECTED_PRESENCE_COUNT} presence events, got {presence_count}"

    for label, expected in EXPECTED_DIRECTION_COUNTS.items():
        observed = direction_counts.get(label, 0)
        assert (
            observed == expected
        ), f"expected {expected} '{label}' direction frames, got {observed}"


def test_offline() -> None:
    """Pytest entry point using default assertions."""
    _run()


if __name__ == "__main__":
    _run()
