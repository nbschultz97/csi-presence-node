"""Tests for the web dashboard module."""
import json
import threading
import time
import urllib.request

import pytest

from csi_node.web_dashboard import DashboardHandler, _dashboard_state, _state_lock


class TestDashboardHTML:
    def test_html_contains_vantage(self):
        from csi_node.web_dashboard import DASHBOARD_HTML
        assert "VANTAGE" in DASHBOARD_HTML
        assert "Through-Wall" in DASHBOARD_HTML

    def test_html_has_api_endpoint(self):
        from csi_node.web_dashboard import DASHBOARD_HTML
        assert "/api/state" in DASHBOARD_HTML
        assert "/api/calibrate" in DASHBOARD_HTML


class TestDashboardState:
    def test_initial_state(self):
        assert "current" in _dashboard_state
        assert "history" in _dashboard_state
        assert _dashboard_state["started"] is False

    def test_state_lock_works(self):
        with _state_lock:
            _dashboard_state["started"] = True
        assert _dashboard_state["started"] is True
        _dashboard_state["started"] = False  # Reset
