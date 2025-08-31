import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import yaml
import pytest
from csi_node import pipeline


def test_single_chain_rssi():
    cfg = yaml.safe_load(open("csi_node/config.yaml"))
    buffer = [
        {"ts": 0.0, "csi": np.zeros(3), "rssi": [-40]},
        {"ts": 0.1, "csi": np.zeros(3), "rssi": [-41]},
    ]
    result = pipeline.compute_window(buffer, 0.0, 0.1, None, cfg)
    assert result["direction"] == "C"
    assert np.isnan(result["rssi1"])
    assert result["rssi0"] == pytest.approx(-40.5)


def test_mixed_chain_defaults_center():
    cfg = yaml.safe_load(open("csi_node/config.yaml"))
    buffer = [
        {"ts": 0.0, "csi": np.zeros(3), "rssi": [-40, -42]},
        {"ts": 0.1, "csi": np.zeros(3), "rssi": [-41]},
    ]
    result = pipeline.compute_window(buffer, 0.0, 0.1, None, cfg)
    assert result["direction"] == "C"
    assert np.isnan(result["rssi1"])

