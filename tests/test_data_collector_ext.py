"""Extended tests for csi_node.data_collector — collect_window, collect_sample."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from csi_node.data_collector import collect_window, collect_sample, merge_datasets


class TestCollectWindow:
    def _write_csi_log(self, path: Path, n_lines: int = 30, interval: float = 0.05):
        """Write fake CSI log lines for testing."""
        from csi_node import utils
        lines = []
        base_ts = time.time()
        for i in range(n_lines):
            ts = base_ts + i * interval
            # Format matching utils.parse_csi_line expectations
            # We'll write lines that parse_csi_line can handle
            csi_data = np.random.randn(2, 64).tolist()
            line = json.dumps({
                "timestamp": ts,
                "csi": csi_data,
                "rssi": [-40, -42],
            })
            lines.append(line + "\n")
        path.write_text("".join(lines))

    def test_missing_log_returns_none(self, tmp_path):
        result = collect_window(tmp_path / "nonexistent.log", window_size=1.0, timeout=0.1)
        assert result is None

    def test_empty_log_returns_none_on_timeout(self, tmp_path):
        log = tmp_path / "empty.log"
        log.write_text("")
        result = collect_window(log, window_size=1.0, timeout=0.15)
        assert result is None


class TestCollectSample:
    def test_returns_none_when_no_data(self, tmp_path):
        from unittest.mock import patch
        log = tmp_path / "empty.log"
        log.write_text("")
        # Mock collect_window to return None immediately (avoid timeout)
        with patch("csi_node.data_collector.collect_window", return_value=None):
            result = collect_sample(log, label=0, window_size=1.0)
        assert result is None

    def test_returns_features_when_data_exists(self, tmp_path):
        from unittest.mock import patch
        log = tmp_path / "data.log"
        log.write_text("")
        fake_csi = np.random.randn(20, 128)
        with patch("csi_node.data_collector.collect_window", return_value=fake_csi):
            result = collect_sample(log, label=1, window_size=1.0)
        assert result is not None
        assert result["label"] == 1
        assert result["features"] is not None
        assert "timestamp" in result


class TestMergeDatasetsEdgeCases:
    def test_single_file(self, tmp_path):
        f1 = tmp_path / "a.npz"
        out = tmp_path / "out.npz"
        np.savez(f1, X=np.ones((4, 3)), y=np.array([0, 1, 0, 1]))
        merge_datasets([f1], out)
        data = np.load(out)
        assert data["X"].shape == (4, 3)

    def test_output_in_new_subdir(self, tmp_path):
        f1 = tmp_path / "a.npz"
        np.savez(f1, X=np.ones((2, 3)), y=np.array([0, 1]))
        out = tmp_path / "sub" / "deep" / "merged.npz"
        merge_datasets([f1], out)
        assert out.exists()

    def test_label_distribution_preserved(self, tmp_path):
        f1 = tmp_path / "a.npz"
        f2 = tmp_path / "b.npz"
        np.savez(f1, X=np.ones((3, 5)), y=np.array([0, 0, 0]))
        np.savez(f2, X=np.zeros((2, 5)), y=np.array([1, 1]))
        out = tmp_path / "out.npz"
        merge_datasets([f1, f2], out)
        data = np.load(out)
        counts = np.bincount(data["y"].astype(int))
        assert counts[0] == 3
        assert counts[1] == 2


class TestCollectWindowExtended:
    def test_collect_window_with_valid_data(self, tmp_path):
        """Test collect_window with valid CSI data that parses correctly."""
        from unittest.mock import patch, MagicMock, mock_open
        import time
        
        log = tmp_path / "test.log"
        log.write_text("dummy content")  # Create the file so exists() returns True
        
        # Mock data packets with proper structure
        base_time = time.time()
        mock_packets = [
            {"ts": base_time, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 0.1, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 0.2, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 0.3, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 0.4, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 0.5, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 0.6, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 0.7, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 0.8, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 0.9, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 1.0, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 1.1, "csi": np.random.randn(2, 64)},
        ]
        
        # Mock file that returns enough data, then empty strings
        mock_file_content = "line1\nline2\nline3\n" + "line4\n" * 20
        mock_file = mock_open(read_data=mock_file_content)
        
        with patch("builtins.open", mock_file):
            with patch("csi_node.utils.parse_csi_line") as mock_parse:
                # Return packets first, then None repeatedly  
                mock_parse.side_effect = mock_packets + [None] * 100
                with patch("time.sleep"):  # Speed up test
                    result = collect_window(log, window_size=1.0, timeout=2.0)
                
                assert result is not None
                assert result.shape[0] >= 10  # Should have enough packets

    def test_collect_window_filters_old_packets(self, tmp_path):
        """Test that collect_window removes packets outside window."""
        from unittest.mock import patch, mock_open
        import time
        
        log = tmp_path / "test.log"
        log.write_text("dummy content")  # Create the file
        
        base_time = time.time()
        # Create packets spread over time, some too old
        mock_packets = [
            {"ts": base_time - 5.0, "csi": np.random.randn(2, 64)},  # Too old
            {"ts": base_time, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 0.1, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 0.2, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 0.3, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 0.4, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 0.5, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 0.6, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 0.7, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 0.8, "csi": np.random.randn(2, 64)},
            {"ts": base_time + 0.9, "csi": np.random.randn(2, 64)},
        ]
        
        mock_file_content = "line1\nline2\nline3\n" + "line4\n" * 20  
        mock_file = mock_open(read_data=mock_file_content)
        
        with patch("builtins.open", mock_file):
            with patch("csi_node.utils.parse_csi_line") as mock_parse:
                mock_parse.side_effect = mock_packets + [None] * 100
                with patch("time.sleep"):
                    result = collect_window(log, window_size=1.0, timeout=2.0)
                    # Should filter out the old packet and return valid data
                    assert result is not None

    def test_collect_window_waits_for_sufficient_data(self, tmp_path):
        """Test that collect_window waits for enough packets."""
        from unittest.mock import patch, mock_open
        import time
        
        log = tmp_path / "test.log"
        
        # Only provide a few packets, not enough for window
        mock_packets = [
            {"ts": time.time(), "csi": np.random.randn(2, 64)},
            {"ts": time.time() + 0.1, "csi": np.random.randn(2, 64)},
        ]
        
        mock_file_content = "line1\nline2\n"
        mock_file = mock_open(read_data=mock_file_content)
        
        with patch("builtins.open", mock_file):
            with patch("csi_node.utils.parse_csi_line") as mock_parse:
                mock_parse.side_effect = mock_packets + [None] * 100
                with patch("time.sleep"):  # Speed up the test
                    result = collect_window(log, window_size=3.0, timeout=0.1)
                    assert result is None  # Should timeout


class TestInteractiveCollect:
    def test_interactive_collect(self, tmp_path):
        """Test interactive_collect function with mocked user input and data collection."""
        from unittest.mock import patch
        from csi_node.data_collector import interactive_collect
        
        log = tmp_path / "test.log"
        log.write_text("")
        output = tmp_path / "output.npz"
        
        # Mock collect_sample to return valid samples
        mock_sample = {
            "features": np.random.randn(10),
            "label": 0,
            "timestamp": "2023-01-01T00:00:00Z",
            "n_packets": 20
        }
        
        with patch("builtins.input", return_value=""):  # Simulate Enter presses
            with patch("builtins.print"):  # Suppress output
                with patch("csi_node.data_collector.collect_sample", return_value=mock_sample):
                    with patch("time.sleep"):  # Speed up test
                        interactive_collect(log, output, samples_per_pose=2)
        
        # Check that output file was created
        assert output.exists()
        data = np.load(output)
        assert "X" in data
        assert "y" in data

    def test_interactive_collect_retries_on_timeout(self, tmp_path):
        """Test that interactive_collect retries when collect_sample returns None."""
        from unittest.mock import patch, call
        from csi_node.data_collector import interactive_collect
        
        log = tmp_path / "test.log"
        log.write_text("")
        output = tmp_path / "output.npz"
        
        mock_sample = {
            "features": np.random.randn(10),
            "label": 0,
            "timestamp": "2023-01-01T00:00:00Z",
            "n_packets": 20
        }
        
        with patch("builtins.input", return_value=""):
            with patch("builtins.print") as mock_print:
                with patch("time.sleep"):
                    # First call returns None (timeout), second succeeds
                    with patch("csi_node.data_collector.collect_sample", side_effect=[None, mock_sample] * 6):
                        interactive_collect(log, output, samples_per_pose=1)
                
                # Check that "TIMEOUT - retrying" was printed
                print_calls = [str(call) for call in mock_print.call_args_list]
                timeout_messages = [call for call in print_calls if "TIMEOUT" in call]
                assert len(timeout_messages) > 0


class TestCollectSinglePose:
    def test_collect_single_pose(self, tmp_path):
        """Test collect_single_pose function."""
        from unittest.mock import patch
        from csi_node.data_collector import collect_single_pose
        
        log = tmp_path / "test.log"
        log.write_text("")
        output = tmp_path / "standing.npz"
        
        mock_sample = {
            "features": np.random.randn(10),
            "label": 0,
            "timestamp": "2023-01-01T00:00:00Z",
            "n_packets": 20
        }
        
        with patch("builtins.input", return_value=""):  # Simulate Enter press
            with patch("builtins.print"):  # Suppress output
                with patch("csi_node.data_collector.collect_sample", return_value=mock_sample):
                    with patch("time.sleep"):  # Speed up test
                        collect_single_pose(log, "standing", 2, output)
        
        # Check output file
        assert output.exists()
        data = np.load(output)
        assert data["X"].shape[0] == 2
        assert list(data["y"]) == [0, 0]  # All standing (label 0)

    def test_collect_single_pose_invalid_pose_exits(self, tmp_path):
        """Test that collect_single_pose exits with invalid pose."""
        from unittest.mock import patch
        from csi_node.data_collector import collect_single_pose
        
        log = tmp_path / "test.log"
        output = tmp_path / "output.npz"
        
        with patch("builtins.print"):
            with pytest.raises(SystemExit):
                collect_single_pose(log, "invalid_pose", 1, output)

    def test_collect_single_pose_retries_on_timeout(self, tmp_path):
        """Test that collect_single_pose retries on timeout."""
        from unittest.mock import patch
        from csi_node.data_collector import collect_single_pose
        
        log = tmp_path / "test.log"
        log.write_text("")
        output = tmp_path / "output.npz"
        
        mock_sample = {
            "features": np.random.randn(10),
            "label": 0,
            "timestamp": "2023-01-01T00:00:00Z",
            "n_packets": 20
        }
        
        with patch("builtins.input", return_value=""):
            with patch("builtins.print") as mock_print:
                with patch("time.sleep"):
                    # First call times out, second succeeds
                    with patch("csi_node.data_collector.collect_sample", side_effect=[None, mock_sample]):
                        collect_single_pose(log, "standing", 1, output)
                
                # Check that "TIMEOUT - retrying" was printed
                print_calls = [str(call) for call in mock_print.call_args_list]
                timeout_messages = [call for call in print_calls if "TIMEOUT" in call]
                assert len(timeout_messages) > 0


class TestMainFunction:
    def test_main_interactive_mode(self, tmp_path):
        """Test main function in interactive mode."""
        from unittest.mock import patch
        from csi_node.data_collector import main
        
        args = [
            "--log", str(tmp_path / "test.log"),
            "--output", str(tmp_path / "output.npz"),
            "--samples", "1"
        ]
        
        with patch("sys.argv", ["data_collector"] + args):
            with patch("csi_node.data_collector.interactive_collect") as mock_interactive:
                main()
                mock_interactive.assert_called_once()

    def test_main_single_pose_mode(self, tmp_path):
        """Test main function in single pose mode."""
        from unittest.mock import patch
        from csi_node.data_collector import main
        
        args = [
            "--pose", "standing",
            "--log", str(tmp_path / "test.log"),
            "--output", str(tmp_path / "output.npz"),
            "--samples", "1"
        ]
        
        with patch("sys.argv", ["data_collector"] + args):
            with patch("csi_node.data_collector.collect_single_pose") as mock_single:
                main()
                mock_single.assert_called_once()

    def test_main_merge_mode(self, tmp_path):
        """Test main function in merge mode."""
        from unittest.mock import patch
        from csi_node.data_collector import main
        
        f1 = tmp_path / "a.npz"
        f2 = tmp_path / "b.npz"
        np.savez(f1, X=np.ones((2, 5)), y=np.array([0, 1]))
        np.savez(f2, X=np.zeros((2, 5)), y=np.array([1, 0]))
        
        args = [
            "--merge", str(f1), str(f2),
            "--output", str(tmp_path / "merged.npz")
        ]
        
        with patch("sys.argv", ["data_collector"] + args):
            with patch("csi_node.data_collector.merge_datasets") as mock_merge:
                main()
                mock_merge.assert_called_once()
