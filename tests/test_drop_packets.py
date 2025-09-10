import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import yaml
from csi_node import pipeline

def test_discard_invalid_csi(capsys):
    cfg = yaml.safe_load(open("csi_node/config.yaml"))
    cfg["log_dropped"] = True
    buffer = [
        {"ts": 0.0, "csi": np.zeros((3, 3))},
        {"ts": 0.1, "csi": np.array([])},
        {"ts": 0.2, "csi": np.zeros((4, 4))},
        {"ts": 0.3, "csi": np.zeros((3, 3))},
    ]
    result = pipeline.compute_window(buffer, 0.0, 0.3, None, cfg)
    assert result is not None
    captured = capsys.readouterr()
    assert "Dropped 2 packets" in captured.err
