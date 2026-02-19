"""Tests for csi_node.tui — curses TUI dashboard."""
from __future__ import annotations

import sys
import unittest
from threading import Event
from unittest.mock import MagicMock, patch


class TestTuiRun(unittest.TestCase):
    """Exercise the curses TUI run() function with mocked curses."""

    def _run_tui(self, state, stop, log_path, mode, replay_path=None):
        """Import and run tui with curses mocked at the module level."""
        mock_curses = MagicMock()
        stdscr = MagicMock()

        def wrapper_side_effect(fn):
            fn(stdscr)

        mock_curses.wrapper.side_effect = wrapper_side_effect

        with patch.dict("sys.modules", {"curses": mock_curses}):
            # Force reimport to pick up mocked curses
            if "csi_node.tui" in sys.modules:
                del sys.modules["csi_node.tui"]
            from csi_node.tui import run
            run(state, stop, log_path, mode, replay_path)

        return stdscr, mock_curses

    def test_run_quits_on_q_key(self):
        state = {"presence": "YES", "presence_conf": 0.95, "direction": "L",
                 "rssi_delta": -2.5, "pose": "standing", "pose_conf": 0.8}
        stop = Event()

        mock_curses = MagicMock()
        stdscr = MagicMock()
        stdscr.getch.return_value = ord("q")
        mock_curses.wrapper.side_effect = lambda fn: fn(stdscr)

        with patch.dict("sys.modules", {"curses": mock_curses}):
            if "csi_node.tui" in sys.modules:
                del sys.modules["csi_node.tui"]
            from csi_node.tui import run
            run(state, stop, "/tmp/test.log", "simulation")

        self.assertTrue(stop.is_set())
        stdscr.erase.assert_called()

    def test_run_with_replay_path(self):
        stop = Event()
        mock_curses = MagicMock()
        stdscr = MagicMock()
        stdscr.getch.return_value = ord("Q")
        mock_curses.wrapper.side_effect = lambda fn: fn(stdscr)

        with patch.dict("sys.modules", {"curses": mock_curses}):
            if "csi_node.tui" in sys.modules:
                del sys.modules["csi_node.tui"]
            from csi_node.tui import run
            run({}, stop, "/tmp/test.log", "replay", replay_path="/data/replay.log")

        self.assertTrue(stop.is_set())

    def test_run_stops_when_event_set(self):
        stop = Event()
        stop.set()
        mock_curses = MagicMock()
        stdscr = MagicMock()
        stdscr.getch.return_value = -1
        mock_curses.wrapper.side_effect = lambda fn: fn(stdscr)

        with patch.dict("sys.modules", {"curses": mock_curses}):
            if "csi_node.tui" in sys.modules:
                del sys.modules["csi_node.tui"]
            from csi_node.tui import run
            run({}, stop, "/tmp/test.log", "live")

    def test_run_default_state_values(self):
        stop = Event()
        mock_curses = MagicMock()
        stdscr = MagicMock()
        call_count = [0]

        def getch_side_effect():
            call_count[0] += 1
            return ord("q") if call_count[0] >= 2 else -1

        stdscr.getch.side_effect = getch_side_effect
        mock_curses.wrapper.side_effect = lambda fn: fn(stdscr)

        with patch.dict("sys.modules", {"curses": mock_curses}):
            if "csi_node.tui" in sys.modules:
                del sys.modules["csi_node.tui"]
            from csi_node.tui import run
            run({}, stop, "/tmp/log.txt", "simulation")

        self.assertTrue(stop.is_set())


if __name__ == "__main__":
    unittest.main()
