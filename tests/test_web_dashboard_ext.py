"""Extended tests for csi_node.web_dashboard — HTTP handler, state, calibration."""
from __future__ import annotations

import io
import json
import threading
from http.server import HTTPServer
from unittest.mock import patch, MagicMock
import urllib.request

import pytest

from csi_node.web_dashboard import (
    DashboardHandler,
    DASHBOARD_HTML,
    _dashboard_state,
    _state_lock,
    _log_entries,
    _finish_calibration,
)


def _make_server(port=0):
    """Create a test HTTP server on a random port."""
    server = HTTPServer(("127.0.0.1", port), DashboardHandler)
    return server


class TestDashboardHandlerGET:
    @pytest.fixture(autouse=True)
    def _server(self):
        self.server = _make_server()
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        yield
        self.server.shutdown()

    def test_index_returns_html(self):
        resp = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/")
        assert resp.status == 200
        body = resp.read().decode()
        assert "VANTAGE" in body

    def test_index_html_path(self):
        resp = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/index.html")
        assert resp.status == 200

    def test_api_state_returns_json(self):
        resp = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/state")
        assert resp.status == 200
        data = json.loads(resp.read())
        assert "current" in data
        assert "history" in data
        assert "log" in data

    def test_api_state_includes_log_entries(self):
        _log_entries.clear()
        _log_entries.append({"timestamp": "12:00:00", "presence": True, "confidence": 0.9})
        resp = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/state")
        data = json.loads(resp.read())
        assert len(data["log"]) >= 1
        assert data["log"][-1]["presence"] is True
        _log_entries.clear()

    def test_404_on_unknown_path(self):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/nope")
        assert exc_info.value.code == 404


class TestDashboardHandlerPOST:
    @pytest.fixture(autouse=True)
    def _server(self):
        self.server = _make_server()
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        yield
        self.server.shutdown()

    def test_calibrate_endpoint(self):
        import csi_node.web_dashboard as wd
        mock_det = MagicMock()
        original = wd._detector
        wd._detector = mock_det
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.port}/api/calibrate",
                data=b"",
                method="POST",
            )
            resp = urllib.request.urlopen(req)
            assert resp.status == 200
            data = json.loads(resp.read())
            assert data["status"] == "calibrating"
            mock_det.calibrate_start.assert_called_once()
        finally:
            wd._detector = original

    def test_post_404_on_unknown(self):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/nope",
            data=b"",
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 404


class TestFinishCalibration:
    def test_no_detector_is_noop(self):
        import csi_node.web_dashboard as wd
        original = wd._detector
        wd._detector = None
        _finish_calibration()  # Should not raise
        wd._detector = original

    def test_successful_calibration_saves(self, tmp_path):
        import csi_node.web_dashboard as wd
        mock_det = MagicMock()
        mock_det.calibrate_finish.return_value = True
        mock_det.save_calibration = MagicMock()
        original = wd._detector
        wd._detector = mock_det
        try:
            _finish_calibration()
            mock_det.calibrate_finish.assert_called_once()
            mock_det.save_calibration.assert_called_once()
        finally:
            wd._detector = original

    def test_failed_calibration_no_save(self):
        import csi_node.web_dashboard as wd
        mock_det = MagicMock()
        mock_det.calibrate_finish.return_value = False
        original = wd._detector
        wd._detector = mock_det
        try:
            _finish_calibration()
            mock_det.save_calibration.assert_not_called()
        finally:
            wd._detector = original


class TestDashboardHTML:
    def test_has_chart_canvas(self):
        assert '<canvas id="chart"' in DASHBOARD_HTML

    def test_has_polling_script(self):
        assert "pollFallback()" in DASHBOARD_HTML or "poll()" in DASHBOARD_HTML
        assert "fetch('/api/state')" in DASHBOARD_HTML

    def test_has_calibration_button(self):
        assert "startCalibration()" in DASHBOARD_HTML

    def test_has_movement_classes(self):
        assert "movement-moving" in DASHBOARD_HTML
        assert "movement-stationary" in DASHBOARD_HTML
