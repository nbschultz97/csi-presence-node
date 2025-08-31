"""Offline pipeline test for sample.log.

The log contains 10 packets: five baseline frames followed by five frames
with higher variance and a left RSSI bias. We expect the presence flag to
be set on five frames and the direction distribution to be {"C": 6, "L": 4,
"R": 0} when processed with the default configuration.
"""

import argparse
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import yaml
from csi_node import pipeline


EXPECTED_PRESENCE_COUNT = 5
EXPECTED_DIRECTION_COUNTS = {"C": 6, "L": 4, "R": 0}


def _run(print_metrics: bool = False) -> None:
    """Run pipeline on sample.log and optionally print summary metrics."""
    cfg = yaml.safe_load(open("csi_node/config.yaml"))
    df = pipeline.run_offline("tests/data/sample.log", cfg)

    # presence==1 should occur once per motion frame
    presence_count = int(df["presence"].eq(1).sum())
    # Direction label distribution
    direction_counts = df["direction"].value_counts().to_dict()

    if print_metrics:
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Print presence and direction counts",
    )
    args = parser.parse_args()
    _run(print_metrics=args.metrics)
