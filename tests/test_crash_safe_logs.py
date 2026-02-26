"""Verify crash-safe logging emits per-run artifacts."""

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from csi_node import pipeline  # noqa: E402


def _make_pkt(ts: float, value: float) -> dict:
    csi = np.full((2, 3), value, dtype=float)
    return {"ts": ts, "rssi": [-40.0 + value * 0.1, -42.0], "csi": csi}


def _faulty_source():
    for idx in range(6):
        yield _make_pkt(idx * 0.2, idx + 1)
    raise RuntimeError("forced crash")


def test_fatal_log_written(tmp_path):
    out_path = tmp_path / "presence_log.jsonl"
    with pytest.raises(RuntimeError):
        pipeline.run_demo(source=_faulty_source(), out=str(out_path), window=0.5)

    runs_dir = tmp_path / "runs"
    run_dirs = list(runs_dir.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    log_file = run_dir / "presence_log.jsonl"
    assert log_file.exists()
    lines = log_file.read_text().strip().splitlines()
    assert lines

    fatal_path = run_dir / "fatal.json"
    assert fatal_path.exists()
    fatal = json.loads(fatal_path.read_text())
    assert fatal["exception"]["type"] == "RuntimeError"
    assert "forced crash" in fatal["exception"]["message"]
    assert any("RuntimeError: forced crash" in line for line in fatal["exception"]["stack"])
    assert fatal["last_entry"] is not None
    assert fatal["last_frame"] is not None
    ts = fatal["last_frame"].get("ts")
    assert isinstance(ts, (int, float))
    assert ts >= 0.0
    if fatal.get("last_packet"):
        assert "csi" in fatal["last_packet"]

    if out_path.is_symlink():
        resolved = out_path.resolve()
        assert resolved == log_file.resolve()
    elif out_path.exists():
        # On Windows, symlinks may not be created (requires privileges).
        # The log file should still exist in the run directory.
        assert log_file.exists()
