import sys
import pathlib
import os
import time
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import yaml
import pytest
from csi_node import baseline, pipeline

ERR_MSG = "CSI capture not running or log idle"


def test_baseline_missing_file(tmp_path, capsys):
    missing = tmp_path / "missing.log"
    out = tmp_path / "out.npz"
    with pytest.raises(SystemExit) as exc:
        baseline.record(missing, duration=0.1, outfile=out, wait=0)
    captured = capsys.readouterr()
    assert ERR_MSG in captured.err
    assert exc.value.code == baseline.CAPTURE_EXIT_CODE


def test_run_offline_missing_file(tmp_path):
    cfg = yaml.safe_load(open("csi_node/config.yaml"))
    cfg["log_wait"] = 0
    missing = tmp_path / "missing.log"
    with pytest.raises(FileNotFoundError) as exc:
        pipeline.run_offline(missing, cfg)
    assert ERR_MSG in str(exc.value)


def test_pipeline_run_missing_file(tmp_path, capsys):
    cfg = yaml.safe_load(open("csi_node/config.yaml"))
    cfg["log_file"] = str(tmp_path / "missing.log")
    cfg["log_wait"] = 0
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    with pytest.raises(SystemExit) as exc:
        pipeline.run(str(cfg_path))
    captured = capsys.readouterr()
    assert ERR_MSG in captured.err
    assert exc.value.code == pipeline.CAPTURE_EXIT_CODE


def test_pipeline_run_missing_file_after_wait(tmp_path, monkeypatch, capsys):
    cfg = yaml.safe_load(open("csi_node/config.yaml"))
    cfg["log_file"] = str(tmp_path / "missing.log")
    cfg["log_wait"] = 0

    # Force pipeline.run to proceed without the log existing
    monkeypatch.setattr(pipeline.utils, "wait_for_file", lambda p, w: True)

    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    with pytest.raises(SystemExit) as exc:
        pipeline.run(str(cfg_path))
    captured = capsys.readouterr()
    assert ERR_MSG in captured.err
    assert exc.value.code == pipeline.CAPTURE_EXIT_CODE


def test_baseline_stale_file(tmp_path, capsys):
    log = tmp_path / "csi_raw.log"
    log.write_text("")
    old = time.time() - 10
    os.utime(log, (old, old))
    out = tmp_path / "out.npz"
    with pytest.raises(SystemExit) as exc:
        baseline.record(log, duration=0.1, outfile=out, wait=0)
    captured = capsys.readouterr()
    assert ERR_MSG in captured.err
    assert exc.value.code == baseline.CAPTURE_EXIT_CODE


def test_pipeline_run_stale_file(tmp_path, capsys):
    log = tmp_path / "csi_raw.log"
    log.write_text("")
    old = time.time() - 10
    os.utime(log, (old, old))
    cfg = yaml.safe_load(open("csi_node/config.yaml"))
    cfg["log_file"] = str(log)
    cfg["log_wait"] = 0
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    with pytest.raises(SystemExit) as exc:
        pipeline.run(str(cfg_path))
    captured = capsys.readouterr()
    assert ERR_MSG in captured.err
    assert exc.value.code == pipeline.CAPTURE_EXIT_CODE
