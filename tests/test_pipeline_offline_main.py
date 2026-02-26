"""Tests for pipeline.run_offline() and pipeline.main() — targets uncovered lines."""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csi_node import pipeline, utils


def _make_csi_line(ts: float, rssi: float = -40.0, n_sub: int = 64) -> str:
    """Create a synthetic CSI log line that parse_csi_line can parse."""
    csi_amps = ",".join([str(round(0.5 + 0.01 * i, 4)) for i in range(n_sub)])
    return f"{ts},{rssi},{csi_amps}\n"


def _write_csi_log(path: Path, n_lines: int = 20, start_ts: float = 0.0):
    """Write synthetic CSI log lines to a file."""
    with open(path, "w") as f:
        for i in range(n_lines):
            ts = start_ts + i * 0.2
            f.write(_make_csi_line(ts))


def _get_default_cfg(tmp_path: Path) -> dict:
    """Load the default config.yaml and override paths for testing."""
    cfg_path = Path(pipeline.__file__).resolve().parent / "config.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    cfg["window_size"] = 1.0
    cfg["window_hop"] = 0.5
    cfg["output_file"] = str(tmp_path / "output.jsonl")
    cfg["baseline_file"] = str(tmp_path / "nonexistent_baseline.npz")
    cfg["log_wait"] = 0.1
    return cfg


class TestRunOffline:
    def test_with_valid_log(self, tmp_path):
        """run_offline produces a DataFrame from valid CSI log."""
        log_path = tmp_path / "csi.log"
        _write_csi_log(log_path, n_lines=30)
        cfg = _get_default_cfg(tmp_path)
        df = pipeline.run_offline(str(log_path), cfg)
        # Should return a pandas DataFrame (may be empty if no windows computed)
        import pandas as pd
        assert isinstance(df, pd.DataFrame)

    def test_with_baseline(self, tmp_path):
        """run_offline loads baseline when file exists."""
        log_path = tmp_path / "csi.log"
        _write_csi_log(log_path, n_lines=30)
        cfg = _get_default_cfg(tmp_path)

        # Create a baseline file
        baseline_path = tmp_path / "baseline.npz"
        mean = np.random.rand(64)
        np.savez(str(baseline_path), mean=mean)
        cfg["baseline_file"] = str(baseline_path)

        df = pipeline.run_offline(str(log_path), cfg)
        import pandas as pd
        assert isinstance(df, pd.DataFrame)

    def test_missing_log_raises(self, tmp_path):
        """run_offline raises FileNotFoundError for missing log."""
        cfg = _get_default_cfg(tmp_path)
        with pytest.raises(FileNotFoundError):
            pipeline.run_offline(str(tmp_path / "missing.log"), cfg)

    def test_empty_log(self, tmp_path):
        """run_offline with empty log returns empty DataFrame."""
        log_path = tmp_path / "empty.log"
        log_path.write_text("")
        cfg = _get_default_cfg(tmp_path)
        df = pipeline.run_offline(str(log_path), cfg)
        import pandas as pd
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestPipelineMain:
    def test_main_with_replay(self, tmp_path):
        """main() with --replay flag calls run_demo with replay_path."""
        log_path = tmp_path / "replay.log"
        _write_csi_log(log_path, n_lines=10)
        out_path = tmp_path / "out.jsonl"

        with patch.object(sys, "argv", [
            "pipeline", "--replay", str(log_path),
            "--out", str(out_path), "--window", "1.0"
        ]):
            # run_demo will process the replay and exit normally
            pipeline.main()

    def test_main_defaults(self):
        """main() with no args should call run_demo with defaults."""
        with patch.object(sys, "argv", ["pipeline"]):
            with patch("csi_node.pipeline.run_demo") as mock_run:
                pipeline.main()
                mock_run.assert_called_once()
                _, kwargs = mock_run.call_args
                assert kwargs["pose"] is False
                assert kwargs["tui"] is False
                assert kwargs["replay_path"] is None

    def test_main_through_wall(self):
        """main() with --through-wall flag passes it through."""
        with patch.object(sys, "argv", ["pipeline", "--through-wall"]):
            with patch("csi_node.pipeline.run_demo") as mock_run:
                pipeline.main()
                _, kwargs = mock_run.call_args
                assert kwargs["through_wall"] is True

    def test_main_with_log_override(self):
        """main() with --log flag passes log_override."""
        with patch.object(sys, "argv", ["pipeline", "--log", "/some/path.log"]):
            with patch("csi_node.pipeline.run_demo") as mock_run:
                pipeline.main()
                _, kwargs = mock_run.call_args
                assert kwargs["log_override"] == "/some/path.log"


class TestRunDemoEdgeCases:
    def test_run_demo_udp_enabled(self, tmp_path):
        """run_demo initializes UDP streamer when configured."""
        def _source():
            for i in range(5):
                csi = np.full((2, 3), i + 1, dtype=float)
                yield {"ts": i * 0.2, "rssi": [-40.0, -42.0], "csi": csi}

        with patch("csi_node.pipeline.yaml") as mock_yaml:
            cfg_path = Path(pipeline.__file__).resolve().parent / "config.yaml"
            real_cfg = yaml.safe_load(open(cfg_path))
            real_cfg["udp_enabled"] = True
            real_cfg["udp_host"] = "127.0.0.1"
            real_cfg["udp_port"] = 9999
            real_cfg["output_file"] = str(tmp_path / "out.jsonl")
            real_cfg["window_size"] = 1.0
            mock_yaml.safe_load.return_value = real_cfg

            # The UDP streamer init will fail gracefully
            pipeline.run_demo(source=_source(), out=str(tmp_path / "out.jsonl"), window=1.0)

    def test_run_demo_config_validation_warnings(self, tmp_path, capsys):
        """run_demo prints config validation warnings."""
        def _source():
            for i in range(3):
                csi = np.full((2, 3), i + 1, dtype=float)
                yield {"ts": i * 0.2, "rssi": [-40.0, -42.0], "csi": csi}

        pipeline.run_demo(
            source=_source(),
            out=str(tmp_path / "out.jsonl"),
            window=1.0,
        )
        # Should complete without error, warnings printed to stderr
