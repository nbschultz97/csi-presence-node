"""Coverage tests for data_collector — interactive_collect, collect_single_pose, main."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from csi_node.data_collector import (
    interactive_collect,
    collect_single_pose,
    main,
)


class TestInteractiveCollect:
    def test_collects_all_poses(self, tmp_path):
        log = tmp_path / "csi.log"
        log.write_text("")
        out = tmp_path / "out.npz"

        fake_sample = {
            "features": np.zeros(14),
            "label": 0,
            "timestamp": "2025-01-01T00:00:00Z",
            "n_packets": 10,
        }

        with patch("csi_node.data_collector.collect_sample", return_value=fake_sample), \
             patch("builtins.input", return_value=""), \
             patch("time.sleep"):
            interactive_collect(log, out, samples_per_pose=2, window_size=1.0)

        data = np.load(out)
        assert data["X"].shape[0] == 6  # 2 samples * 3 poses
        assert len(data["y"]) == 6

    def test_retries_on_timeout(self, tmp_path):
        log = tmp_path / "csi.log"
        log.write_text("")
        out = tmp_path / "out.npz"

        call_count = [0]

        def fake_collect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return None  # timeout
            return {
                "features": np.zeros(14),
                "label": 0,
                "timestamp": "2025-01-01T00:00:00Z",
                "n_packets": 10,
            }

        with patch("csi_node.data_collector.collect_sample", side_effect=fake_collect), \
             patch("builtins.input", return_value=""), \
             patch("time.sleep"):
            interactive_collect(log, out, samples_per_pose=1, window_size=1.0)

        data = np.load(out)
        assert data["X"].shape[0] == 3  # 1 sample * 3 poses


class TestCollectSinglePose:
    def test_valid_pose(self, tmp_path):
        log = tmp_path / "csi.log"
        log.write_text("")
        out = tmp_path / "out.npz"

        fake_sample = {
            "features": np.ones(14),
            "label": 0,
            "timestamp": "2025-01-01T00:00:00Z",
            "n_packets": 10,
        }

        with patch("csi_node.data_collector.collect_sample", return_value=fake_sample), \
             patch("builtins.input", return_value=""), \
             patch("time.sleep"):
            collect_single_pose(log, "standing", 3, out, window_size=1.0)

        data = np.load(out)
        assert data["X"].shape[0] == 3

    def test_unknown_pose_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            collect_single_pose(tmp_path / "log", "flying", 1, tmp_path / "out.npz")

    def test_retries_on_timeout(self, tmp_path):
        log = tmp_path / "csi.log"
        log.write_text("")
        out = tmp_path / "out.npz"

        calls = [0]

        def fake(path, label, ws):
            calls[0] += 1
            if calls[0] == 1:
                return None
            return {"features": np.zeros(14), "label": label,
                    "timestamp": "t", "n_packets": 5}

        with patch("csi_node.data_collector.collect_sample", side_effect=fake), \
             patch("builtins.input", return_value=""), \
             patch("time.sleep"):
            collect_single_pose(log, "prone", 1, out)

        assert out.exists()


class TestMainCLI:
    def test_merge_mode(self, tmp_path):
        f1 = tmp_path / "a.npz"
        np.savez(f1, X=np.ones((2, 5)), y=np.array([0, 1]))
        out = tmp_path / "merged.npz"

        with patch("sys.argv", ["prog", "--merge", str(f1), "--output", str(out)]):
            main()

        assert out.exists()

    def test_pose_mode(self, tmp_path):
        log = tmp_path / "csi.log"
        log.write_text("")
        out = tmp_path / "out.npz"

        with patch("sys.argv", ["prog", "--pose", "standing", "--samples", "1",
                                 "--log", str(log), "--output", str(out)]), \
             patch("csi_node.data_collector.collect_single_pose") as mock_csp:
            main()
            mock_csp.assert_called_once()

    def test_interactive_mode(self, tmp_path):
        with patch("sys.argv", ["prog", "--output", str(tmp_path / "out.npz")]), \
             patch("csi_node.data_collector.interactive_collect") as mock_ic:
            main()
            mock_ic.assert_called_once()
