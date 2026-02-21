"""Tests for the curses-based TUI module."""
import sys
import time
from threading import Event, Thread
from unittest.mock import MagicMock, patch, call

import pytest


# TUI requires curses; skip entirely if unavailable (Windows CI)
curses_available = True
try:
    import curses as _curses  # noqa: F401
except ImportError:
    curses_available = False

if curses_available:
    from csi_node import tui


@pytest.fixture
def state():
    return {
        "presence": "YES",
        "presence_conf": 0.85,
        "direction": "L",
        "rssi_delta": 3.2,
        "pose": "standing",
        "pose_conf": 0.72,
    }


@pytest.fixture
def stop_event():
    return Event()


# ---- Helper: fake curses stdscr ----

class FakeStdscr:
    """Minimal curses stdscr mock that returns 'q' on second getch."""

    def __init__(self, quit_after=2):
        self._call = 0
        self._quit_after = quit_after
        self.lines = []

    def erase(self):
        self.lines.clear()

    def addstr(self, y, x, text):
        self.lines.append((y, x, text))

    def refresh(self):
        pass

    def timeout(self, ms):
        pass

    def nodelay(self, flag):
        pass

    def getch(self):
        self._call += 1
        if self._call >= self._quit_after:
            return ord("q")
        return -1  # no key


@pytest.mark.skipif(not curses_available, reason="curses not available")
class TestTuiRun:
    """Test the tui.run() function."""

    def test_quit_on_q_key(self, state, stop_event):
        """TUI should stop when 'q' is pressed."""
        fake = FakeStdscr(quit_after=2)

        with patch("curses.wrapper", side_effect=lambda fn: fn(fake)):
            with patch("curses.curs_set"):
                tui.run(state, stop_event, "/tmp/test.jsonl", "LIVE")

        assert stop_event.is_set()

    def test_quit_on_Q_key(self, state, stop_event):
        """TUI should also stop on uppercase Q."""
        class QStdscr(FakeStdscr):
            def getch(self):
                self._call += 1
                if self._call >= 2:
                    return ord("Q")
                return -1

        fake = QStdscr()

        with patch("curses.wrapper", side_effect=lambda fn: fn(fake)):
            with patch("curses.curs_set"):
                tui.run(state, stop_event, "/tmp/test.jsonl", "LIVE")

        assert stop_event.is_set()

    def test_displays_state_values(self, state, stop_event):
        """TUI should render presence, direction, pose, and log path."""
        fake = FakeStdscr(quit_after=2)

        with patch("curses.wrapper", side_effect=lambda fn: fn(fake)):
            with patch("curses.curs_set"):
                tui.run(state, stop_event, "/tmp/out.jsonl", "LIVE")

        rendered = " ".join(text for _, _, text in fake.lines)
        assert "YES" in rendered
        assert "0.85" in rendered
        assert "L" in rendered
        assert "standing" in rendered
        assert "/tmp/out.jsonl" in rendered
        assert "LIVE" in rendered

    def test_replay_mode_shown(self, state, stop_event):
        """When replay_path is provided, TUI should show REPLAY."""
        fake = FakeStdscr(quit_after=2)

        with patch("curses.wrapper", side_effect=lambda fn: fn(fake)):
            with patch("curses.curs_set"):
                tui.run(state, stop_event, "/tmp/out.jsonl", "REPLAY", replay_path="/data/test.b64")

        rendered = " ".join(text for _, _, text in fake.lines)
        assert "REPLAY" in rendered
        assert "/data/test.b64" in rendered

    def test_stop_event_terminates_loop(self, state):
        """TUI should exit when stop event is set externally."""
        stop = Event()

        class SlowStdscr(FakeStdscr):
            def getch(self):
                # Never press q — rely on stop event
                stop.set()
                return -1

        fake = SlowStdscr()

        with patch("curses.wrapper", side_effect=lambda fn: fn(fake)):
            with patch("curses.curs_set"):
                tui.run(state, stop, "/tmp/out.jsonl", "LIVE")

        assert stop.is_set()

    def test_displays_quit_hint(self, state, stop_event):
        """TUI should show [q]=quit."""
        fake = FakeStdscr(quit_after=2)

        with patch("curses.wrapper", side_effect=lambda fn: fn(fake)):
            with patch("curses.curs_set"):
                tui.run(state, stop_event, "/tmp/out.jsonl", "LIVE")

        rendered = " ".join(text for _, _, text in fake.lines)
        assert "[q]=quit" in rendered

    def test_missing_state_keys_defaults(self, stop_event):
        """TUI should handle missing state keys with defaults."""
        empty_state = {}
        fake = FakeStdscr(quit_after=2)

        with patch("curses.wrapper", side_effect=lambda fn: fn(fake)):
            with patch("curses.curs_set"):
                tui.run(empty_state, stop_event, "/tmp/out.jsonl", "LIVE")

        # Should not crash — defaults used
        rendered = " ".join(text for _, _, text in fake.lines)
        assert "Presence" in rendered
