"""Offline pipeline test for sample.log.

The log contains 10 packets: five baseline frames followed by five frames
with higher variance and a left RSSI bias. We expect five presence events and
four "L" direction labels (no "R" events) when processed with the default
configuration.
"""

import argparse
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import yaml
from csi_node import pipeline


def _run(print_metrics: bool = False) -> None:
    """Run pipeline on sample.log and optionally print summary metrics."""
    cfg = yaml.safe_load(open("csi_node/config.yaml"))
    df = pipeline.run_offline("tests/data/sample.log", cfg)

    # presence==1 should occur once per motion frame (five total)
    presence_count = int(df["presence"].eq(1).sum())

    # Direction labels should be 6 "C" frames and 4 "L" frames; no "R"
    direction_counts = df["direction"].value_counts().to_dict()

    if print_metrics:
        print(df)
        print("presence==1:", presence_count)
        print("direction counts:", direction_counts)

    assert presence_count == 5
    assert direction_counts.get("L", 0) == 4
    assert direction_counts.get("C", 0) == 6
    assert direction_counts.get("R", 0) == 0


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
