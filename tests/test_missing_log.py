import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import yaml
import pytest
from csi_node import baseline, pipeline


def test_baseline_missing_file(tmp_path, capsys):
    missing = tmp_path / "missing.log"
    out = tmp_path / "out.npz"
    with pytest.raises(SystemExit):
        baseline.record(missing, duration=0.1, outfile=out, wait=0)
    captured = capsys.readouterr()
    assert "log file" in captured.err


def test_run_offline_missing_file(tmp_path):
    cfg = yaml.safe_load(open("csi_node/config.yaml"))
    cfg["log_wait"] = 0
    missing = tmp_path / "missing.log"
    with pytest.raises(FileNotFoundError):
        pipeline.run_offline(missing, cfg)
