"""Tests for web_dashboard POST endpoints and helper functions."""
import json
import io
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pytest

import csi_node.web_dashboard as wd


class FakeWfile:
    """Fake writable file for HTTP handler."""
    def __init__(self):
        self.data = b""
    def write(self, b):
        self.data += b
    def flush(self):
        pass


class FakeRfile:
    """Fake readable file for HTTP handler."""
    def __init__(self, data=b""):
        self._io = io.BytesIO(data)
    def read(self, n=-1):
        return self._io.read(n)


def make_handler(method="POST", path="/", body=b""):
    """Create a DashboardHandler without actually listening on a socket."""
    handler = object.__new__(wd.DashboardHandler)
    handler.command = method
    handler.path = path
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = FakeRfile(body)
    handler.wfile = FakeWfile()
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.client_address = ("127.0.0.1", 9999)
    handler.server = MagicMock()
    handler.close_connection = True
    # Patch log_request to avoid side effects
    handler.log_request = lambda *a, **kw: None
    return handler


class TestCalibrationHelpers:
    def test_update_cal_progress(self):
        orig = wd._dashboard_state.get("calibration_progress")
        wd._update_cal_progress(0.5)
        assert wd._dashboard_state["calibration_progress"] == 0.5
        wd._update_cal_progress(0.0)

    def test_finish_calibration_no_detector(self):
        """_finish_calibration should not crash when _detector is None."""
        old = wd._detector
        wd._detector = None
        try:
            wd._finish_calibration()  # should not raise
        finally:
            wd._detector = old

    def test_finish_calibration_success(self, tmp_path):
        old = wd._detector
        mock_det = MagicMock()
        mock_det.calibrate_finish.return_value = True
        mock_det.save_calibration = MagicMock()
        wd._detector = mock_det
        try:
            with patch.object(Path, "resolve", return_value=tmp_path / "csi_node" / "web_dashboard.py"):
                # Just verify no crash — the path logic uses __file__
                wd._finish_calibration()
            mock_det.calibrate_finish.assert_called_once()
        finally:
            wd._detector = old

    def test_finish_calibration_failure(self):
        old = wd._detector
        mock_det = MagicMock()
        mock_det.calibrate_finish.return_value = False
        wd._detector = mock_det
        try:
            wd._finish_calibration()
            mock_det.calibrate_finish.assert_called_once()
            # save_calibration should NOT be called on failure
            mock_det.save_calibration.assert_not_called()
        finally:
            wd._detector = old


class TestHandleProfile:
    def test_switch_profile_default(self):
        old = wd._detector
        mock_det = MagicMock()
        wd._detector = mock_det
        try:
            body = json.dumps({"profile": "sensitive"}).encode()
            h = make_handler("POST", "/api/profile", body)
            h._handle_profile()
            resp = json.loads(h.wfile.data.split(b"\r\n\r\n")[-1])
            assert resp["profile"] == "sensitive"
        finally:
            wd._detector = old

    def test_switch_profile_no_detector(self):
        old = wd._detector
        wd._detector = None
        try:
            body = json.dumps({"profile": "default"}).encode()
            h = make_handler("POST", "/api/profile", body)
            h._handle_profile()
            resp = json.loads(h.wfile.data.split(b"\r\n\r\n")[-1])
            assert resp["profile"] == "default"
        finally:
            wd._detector = old


class TestHandleEnvList:
    @patch("csi_node.web_dashboard.EnvironmentManager")
    def test_list_profiles(self, MockMgr):
        MockMgr.return_value.list_profiles.return_value = ["default", "office"]
        h = make_handler("POST", "/api/environment/list")
        h._handle_env_list()
        resp = json.loads(h.wfile.data.split(b"\r\n\r\n")[-1])
        assert resp["profiles"] == ["default", "office"]


class TestHandleEnvSave:
    def test_save_no_detector(self):
        old = wd._detector
        wd._detector = None
        try:
            body = json.dumps({"name": "test"}).encode()
            h = make_handler("POST", "/api/environment/save", body)
            h._handle_env_save()
            resp = json.loads(h.wfile.data.split(b"\r\n\r\n")[-1])
            assert resp["success"] is False
        finally:
            wd._detector = old

    def test_save_uncalibrated_detector(self):
        old = wd._detector
        mock_det = MagicMock()
        mock_det.calibrated = False
        wd._detector = mock_det
        try:
            body = json.dumps({"name": "test"}).encode()
            h = make_handler("POST", "/api/environment/save", body)
            h._handle_env_save()
            resp = json.loads(h.wfile.data.split(b"\r\n\r\n")[-1])
            assert resp["success"] is False
        finally:
            wd._detector = old

    @patch("csi_node.web_dashboard.EnvironmentManager")
    def test_save_calibrated_detector(self, MockMgr):
        old = wd._detector
        mock_det = MagicMock()
        mock_det.calibrated = True
        wd._detector = mock_det
        MockMgr.return_value.save.return_value = "/tmp/test.json"
        try:
            body = json.dumps({"name": "test", "wall_type": "drywall"}).encode()
            h = make_handler("POST", "/api/environment/save", body)
            h._handle_env_save()
            resp = json.loads(h.wfile.data.split(b"\r\n\r\n")[-1])
            assert resp["success"] is True
        finally:
            wd._detector = old


class TestHandleEnvLoad:
    def test_load_no_detector(self):
        old = wd._detector
        wd._detector = None
        try:
            body = json.dumps({"name": "test"}).encode()
            h = make_handler("POST", "/api/environment/load", body)
            h._handle_env_load()
            resp = json.loads(h.wfile.data.split(b"\r\n\r\n")[-1])
            assert resp["success"] is False
        finally:
            wd._detector = old

    @patch("csi_node.web_dashboard.EnvironmentManager")
    def test_load_success(self, MockMgr):
        old = wd._detector
        mock_det = MagicMock()
        wd._detector = mock_det
        MockMgr.return_value.load.return_value = True
        try:
            body = json.dumps({"name": "office"}).encode()
            h = make_handler("POST", "/api/environment/load", body)
            h._handle_env_load()
            resp = json.loads(h.wfile.data.split(b"\r\n\r\n")[-1])
            assert resp["success"] is True
            assert resp["name"] == "office"
        finally:
            wd._detector = old


class TestHandleRecord:
    def test_start_recording(self, tmp_path):
        old_recording = wd._recording
        old_file = wd._record_file
        wd._recording = False
        wd._record_file = None
        try:
            body = json.dumps({"label": "test"}).encode()
            h = make_handler("POST", "/api/record", body)
            with patch("csi_node.web_dashboard.Path") as MockPath:
                mock_rec_dir = MagicMock()
                MockPath.return_value.resolve.return_value.parent.parent.__truediv__ = lambda s, x: tmp_path / x
                # Simplify: just call _handle_record and check state
                with patch("builtins.open", mock_open()):
                    h._handle_record()
            resp = json.loads(h.wfile.data.split(b"\r\n\r\n")[-1])
            assert resp["recording"] is True
        finally:
            wd._recording = old_recording
            wd._record_file = old_file

    def test_stop_recording(self):
        old_recording = wd._recording
        old_file = wd._record_file
        old_count = wd._record_count
        mock_file = MagicMock()
        wd._recording = True
        wd._record_file = mock_file
        wd._record_count = 42
        try:
            h = make_handler("POST", "/api/record", b"{}")
            h._handle_record()
            resp = json.loads(h.wfile.data.split(b"\r\n\r\n")[-1])
            assert resp["recording"] is False
            assert resp["frames_saved"] == 42
            mock_file.close.assert_called_once()
        finally:
            wd._recording = old_recording
            wd._record_file = old_file
            wd._record_count = old_count


class TestDoPost:
    def test_post_calibrate_no_detector(self):
        old = wd._detector
        wd._detector = None
        try:
            h = make_handler("POST", "/api/calibrate")
            h.do_POST()
            resp = json.loads(h.wfile.data.split(b"\r\n\r\n")[-1])
            assert resp["status"] == "calibrating"
        finally:
            wd._detector = old

    def test_post_calibrate_with_detector(self):
        old = wd._detector
        mock_det = MagicMock()
        wd._detector = mock_det
        try:
            h = make_handler("POST", "/api/calibrate")
            h.do_POST()
            mock_det.calibrate_start.assert_called_once()
        finally:
            wd._detector = old

    def test_post_unknown_404(self):
        h = make_handler("POST", "/api/unknown")
        h.send_error = MagicMock()
        h.do_POST()
        h.send_error.assert_called_with(404)


class TestDoGet:
    def test_get_api_state(self):
        h = make_handler("GET", "/api/state")
        h.do_GET()
        body = h.wfile.data.split(b"\r\n\r\n")[-1]
        data = json.loads(body)
        assert isinstance(data, dict)

    def test_get_unknown_404(self):
        h = make_handler("GET", "/unknown")
        h.send_error = MagicMock()
        h.do_GET()
        h.send_error.assert_called_with(404)


class TestRunDashboard:
    @patch("csi_node.web_dashboard.HTTPServer")
    @patch("csi_node.web_dashboard._pipeline_thread")
    def test_run_dashboard_starts_server(self, mock_pipeline, MockHTTPServer):
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt
        MockHTTPServer.return_value = mock_server

        wd.run_dashboard(port=9999, simulate=True)
        MockHTTPServer.assert_called_once()
        mock_server.shutdown.assert_called_once()


class TestMain:
    @patch("csi_node.web_dashboard.run_dashboard")
    def test_main_defaults(self, mock_run):
        with patch("sys.argv", ["web_dashboard"]):
            wd.main()
        mock_run.assert_called_once_with(
            port=8088,
            replay_path=None,
            log_path=None,
            speed=1.0,
            simulate=False,
            through_wall=False,
        )

    @patch("csi_node.web_dashboard.run_dashboard")
    def test_main_with_args(self, mock_run):
        with patch("sys.argv", ["web_dashboard", "--port", "3000", "--simulate", "--through-wall"]):
            wd.main()
        mock_run.assert_called_once_with(
            port=3000,
            replay_path=None,
            log_path=None,
            speed=1.0,
            simulate=True,
            through_wall=True,
        )
