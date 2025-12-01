"""Parse base64 CSI log and verify presence/direction counts."""

import pathlib
import sys

import yaml

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from csi_node import pipeline, replay

EXPECTED_PRESENCE_COUNT = 5
EXPECTED_DIRECTION_COUNTS = {"C": 10, "L": 0, "R": 0}

def test_parse_and_presence() -> None:
    tmp_path = replay.decode_b64_capture(pathlib.Path("data/sample_csi.b64"))
    try:
        cfg = yaml.safe_load(open("csi_node/config.yaml"))
        df = pipeline.run_offline(str(tmp_path), cfg)
        presence_count = int(df["presence"].eq(1).sum())
        direction_counts = df["direction"].value_counts().to_dict()
        assert {"bearing_deg", "bearing_label", "distance", "distance_filtered"}.issubset(df.columns)
        assert set(df["bearing_label"].dropna().unique()).issubset({"left", "right", "center"})
        assert presence_count == EXPECTED_PRESENCE_COUNT
        for label, expected in EXPECTED_DIRECTION_COUNTS.items():
            assert direction_counts.get(label, 0) == expected
    finally:
        tmp_path.unlink(missing_ok=True)
