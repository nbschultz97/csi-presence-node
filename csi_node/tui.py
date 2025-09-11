from __future__ import annotations
"""Simple curses dashboard for realtime pipeline visualization."""

import curses
import time
from threading import Event
from typing import Dict, Optional


def run(state: Dict[str, float], stop: Event, log_path: str, mode: str, replay_path: Optional[str] = None) -> None:
    """Run curses TUI until ``stop`` is set."""

    def _main(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        while not stop.is_set():
            stdscr.erase()
            stdscr.addstr(0, 0, f"Time: {time.strftime('%H:%M:%S')}")
            stdscr.addstr(
                1,
                0,
                f"Presence: {state.get('presence', 'NO'):>3}      Confidence: {state.get('presence_conf', 0.0):.2f}",
            )
            stdscr.addstr(
                2,
                0,
                f"Direction: {state.get('direction', 'C'):>5}    RSSIΔ: {state.get('rssi_delta', 0.0):+5.1f} dB",
            )
            stdscr.addstr(
                3,
                0,
                f"Pose: {state.get('pose', 'N/A'):>9} Confidence: {state.get('pose_conf', 0.0):.2f}",
            )
            stdscr.addstr(4, 0, f"Log: {log_path}")
            if replay_path:
                stdscr.addstr(5, 0, f"Mode: REPLAY: {replay_path}")
            else:
                stdscr.addstr(5, 0, f"Mode: {mode}")
            stdscr.addstr(7, 0, "[q]=quit")
            stdscr.refresh()
            stdscr.timeout(250)  # ~4 Hz refresh
            ch = stdscr.getch()
            if ch in (ord('q'), ord('Q')):
                stop.set()
                break

    curses.wrapper(_main)
