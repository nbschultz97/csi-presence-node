"""Tests for gui.py - ProcessLogger and static utility methods.

The GUI module (tkinter) is not imported directly to avoid display requirements.
We test the non-GUI logic by importing selectively or exercising static methods.
"""

import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import pytest


# Import just the module-level constants and ProcessLogger
# We can't import App directly without tkinter display, so we import the module
# and test what we can.
import importlib


@pytest.fixture
def gui_module():
    """Import gui module with tkinter mocked to avoid display requirement."""
    import unittest.mock as um
    # Mock tkinter to prevent display errors
    mock_tk = MagicMock()
    with patch.dict('sys.modules', {
        'tkinter': mock_tk,
        'tkinter.ttk': mock_tk,
        'tkinter.filedialog': mock_tk,
        'tkinter.messagebox': mock_tk,
        'tkinter.simpledialog': mock_tk,
        'tkinter.scrolledtext': mock_tk,
    }):
        # Force reimport
        if 'csi_node.gui' in sys.modules:
            del sys.modules['csi_node.gui']
        from csi_node import gui
        yield gui


class TestProcessLogger:
    """Test ProcessLogger without tkinter dependency."""

    def test_process_logger_pump_stdout(self, gui_module):
        """ProcessLogger should pump stdout lines into queue."""
        q = queue.Queue()
        # Create a mock process with stdout that yields lines
        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline = MagicMock(side_effect=["line1\n", "line2\n", ""])
        mock_proc.stderr = None
        mock_proc.poll.return_value = None

        plog = gui_module.ProcessLogger("TEST", mock_proc, q)
        plog.start()

        # Wait for pump thread to finish
        time.sleep(0.3)

        items = []
        while not q.empty():
            items.append(q.get_nowait())

        assert ("TEST", "line1") in items
        assert ("TEST", "line2") in items

    def test_process_logger_stop_terminates(self, gui_module):
        """ProcessLogger.stop() should terminate the process."""
        q = queue.Queue()
        mock_proc = MagicMock()
        mock_proc.stdout = None
        mock_proc.stderr = None
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = 0

        plog = gui_module.ProcessLogger("TEST", mock_proc, q)
        plog.stop()

        mock_proc.terminate.assert_called_once()

    def test_process_logger_stop_already_exited(self, gui_module):
        """ProcessLogger.stop() should handle already-exited process."""
        q = queue.Queue()
        mock_proc = MagicMock()
        mock_proc.stdout = None
        mock_proc.stderr = None
        mock_proc.poll.return_value = 0  # Already exited

        plog = gui_module.ProcessLogger("TEST", mock_proc, q)
        plog.stop()  # Should not raise

    def test_process_logger_wait(self, gui_module):
        """ProcessLogger.wait() should return process exit code."""
        q = queue.Queue()
        mock_proc = MagicMock()
        mock_proc.stdout = None
        mock_proc.stderr = None
        mock_proc.wait.return_value = 42

        plog = gui_module.ProcessLogger("TEST", mock_proc, q)
        assert plog.wait() == 42

    def test_process_logger_wait_exception(self, gui_module):
        """ProcessLogger.wait() should return -1 on exception."""
        q = queue.Queue()
        mock_proc = MagicMock()
        mock_proc.stdout = None
        mock_proc.stderr = None
        mock_proc.wait.side_effect = OSError("broken")

        plog = gui_module.ProcessLogger("TEST", mock_proc, q)
        assert plog.wait() == -1

    def test_process_logger_stop_kill_on_timeout(self, gui_module):
        """ProcessLogger.stop() should kill if terminate doesn't work."""
        q = queue.Queue()
        mock_proc = MagicMock()
        mock_proc.stdout = None
        mock_proc.stderr = None
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = subprocess.TimeoutExpired("cmd", 0.5)

        plog = gui_module.ProcessLogger("TEST", mock_proc, q)
        plog.stop()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()


class TestStaticMethods:
    """Test App static methods via the module."""

    def test_channel_to_freq_2ghz(self, gui_module):
        """Channel 1-13 should map to 2.4 GHz band."""
        App = gui_module.App
        assert App._channel_to_freq(1) == 2412
        assert App._channel_to_freq(6) == 2437
        assert App._channel_to_freq(11) == 2462
        assert App._channel_to_freq(13) == 2472

    def test_channel_to_freq_ch14(self, gui_module):
        """Channel 14 should be 2484 MHz."""
        assert gui_module.App._channel_to_freq(14) == 2484

    def test_channel_to_freq_5ghz(self, gui_module):
        """5 GHz channels should use 5000 + ch*5."""
        assert gui_module.App._channel_to_freq(36) == 5180
        assert gui_module.App._channel_to_freq(149) == 5745

    def test_wait_for_file_exists(self, gui_module, tmp_path):
        """_wait_for_file should return True when file exists and is non-empty."""
        f = tmp_path / "test.dat"
        f.write_text("data")
        assert gui_module.App._wait_for_file(f, 1.0) is True

    def test_wait_for_file_not_exists(self, gui_module, tmp_path):
        """_wait_for_file should return False when file doesn't exist."""
        f = tmp_path / "nonexistent.dat"
        assert gui_module.App._wait_for_file(f, 0.3) is False

    def test_wait_for_file_empty(self, gui_module, tmp_path):
        """_wait_for_file should return False for empty file within timeout."""
        f = tmp_path / "empty.dat"
        f.write_text("")
        assert gui_module.App._wait_for_file(f, 0.3) is False


class TestModuleConstants:
    """Test module-level constants."""

    def test_repo_root_exists(self, gui_module):
        """REPO_ROOT should point to the repository root."""
        assert gui_module.REPO_ROOT.exists()

    def test_scripts_dir(self, gui_module):
        """SCRIPTS_DIR should be under REPO_ROOT."""
        assert str(gui_module.SCRIPTS_DIR).startswith(str(gui_module.REPO_ROOT))
