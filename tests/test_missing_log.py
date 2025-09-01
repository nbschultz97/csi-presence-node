import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import yaml
import pytest
from csi_node import baseline, pipeline


def test_baseline_missing_file(tmp_path, capsys):
    missing = tmp_path / "missing.log"
    out = tmp_path / "out.npz"
    baseline.record(missing, duration=0.1, outfile=out, wait=0)
    captured = capsys.readouterr()
    assert "Run scripts/10_csi_capture.sh first" in captured.out


def test_run_offline_missing_file(tmp_path):
    cfg = yaml.safe_load(open("csi_node/config.yaml"))
    cfg["log_wait"] = 0
    missing = tmp_path / "missing.log"
    with pytest.raises(FileNotFoundError) as exc:
        pipeline.run_offline(missing, cfg)
    assert "Run scripts/10_csi_capture.sh first" in str(exc.value)


def test_pipeline_run_missing_file(tmp_path, capsys):
    cfg = yaml.safe_load(open("csi_node/config.yaml"))
    cfg["log_file"] = str(tmp_path / "missing.log")
    cfg["log_wait"] = 0
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    with pytest.raises(SystemExit):
        pipeline.run(str(cfg_path))
    captured = capsys.readouterr()
    assert "Run scripts/10_csi_capture.sh first" in captured.err


def test_pipeline_run_missing_file_after_wait(tmp_path, monkeypatch):
    cfg = yaml.safe_load(open("csi_node/config.yaml"))
    cfg["log_file"] = str(tmp_path / "missing.log")
    cfg["log_wait"] = 0

    # Force pipeline.run to proceed without the log existing
    monkeypatch.setattr(pipeline.utils, "wait_for_file", lambda p, w: True)

    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    with pytest.raises(FileNotFoundError) as exc:
        pipeline.run(str(cfg_path))
    assert "Run scripts/10_csi_capture.sh first" in str(exc.value)
