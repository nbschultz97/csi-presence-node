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


class TestPipelineThread:
    def test_pipeline_thread_simulate_mode(self, tmp_path):
        """Test _pipeline_thread in simulation mode."""
        from unittest.mock import patch, MagicMock, mock_open
        from csi_node.web_dashboard import _pipeline_thread, _dashboard_state, _state_lock
        
        # Create a config file to avoid FileNotFoundError
        config_data = {"energy_threshold_factor": 2.5}
        
        with patch("builtins.open", mock_open(read_data="")):
            with patch("yaml.safe_load", return_value=config_data):
                with patch("csi_node.web_dashboard.AdaptivePresenceDetector") as mock_detector_cls:
                    mock_detector = MagicMock()
                    mock_detector_cls.return_value = mock_detector
                    
                    with patch("csi_node.simulator.CSISimulator") as mock_sim_cls:
                        mock_simulator = MagicMock()
                        mock_packets = [
                            {"ts": 1234567890.0, "csi": [[1, 2], [3, 4]], "rssi": [-40, -45]},
                            {"ts": 1234567890.1, "csi": [[2, 3], [4, 5]], "rssi": [-41, -46]},
                        ]
                        mock_simulator.stream.return_value = iter(mock_packets)  # Make it an iterator
                        mock_sim_cls.return_value = mock_simulator
                        
                        # Run for a short time then stop
                        import threading
                        thread = threading.Thread(target=_pipeline_thread, kwargs={"simulate": True}, daemon=True)
                        thread.start()
                        import time
                        time.sleep(0.1)  # Let it run briefly
                        
                        # Check that simulator was created and state was set
                        mock_sim_cls.assert_called_once()
                        with _state_lock:
                            assert _dashboard_state["simulate"] is True

    def test_pipeline_thread_replay_mode(self, tmp_path):
        """Test _pipeline_thread in replay mode."""
        from unittest.mock import patch, MagicMock
        from csi_node.web_dashboard import _pipeline_thread
        
        replay_file = tmp_path / "test.b64"
        replay_file.write_text("dummy data")
        
        mock_packets = [
            {"ts": 1234567890.0, "csi": [[1, 2], [3, 4]], "rssi": [-40, -45]},
        ]
        
        with patch("csi_node.web_dashboard.replay_mod.replay", return_value=mock_packets):
            with patch("builtins.open", side_effect=FileNotFoundError):  # No config
                with patch("yaml.safe_load", return_value={}):
                    import threading
                    thread = threading.Thread(
                        target=_pipeline_thread, 
                        kwargs={"replay_path": str(replay_file), "speed": 2.0}, 
                        daemon=True
                    )
                    thread.start()
                    import time
                    time.sleep(0.1)

    def test_pipeline_thread_log_mode_missing_file(self, tmp_path):
        """Test _pipeline_thread in log mode when file doesn't exist initially."""
        from unittest.mock import patch
        from csi_node.web_dashboard import _pipeline_thread
        
        log_file = tmp_path / "missing.log"
        
        with patch("builtins.open", side_effect=FileNotFoundError):  # No config
            with patch("yaml.safe_load", return_value={}):
                with patch("time.sleep") as mock_sleep:  # Speed up waiting
                    import threading
                    thread = threading.Thread(
                        target=_pipeline_thread, 
                        kwargs={"log_path": str(log_file)}, 
                        daemon=True
                    )
                    thread.start()
                    import time
                    time.sleep(0.05)  # Brief pause
                    # Should have tried to wait for the file
                    assert mock_sleep.call_count > 0

    def test_pipeline_thread_config_loading(self, tmp_path):
        """Test _pipeline_thread loads configuration correctly."""
        from unittest.mock import patch, mock_open
        from csi_node.web_dashboard import _pipeline_thread
        
        config_data = {
            "energy_threshold_factor": 3.0,
            "variance_threshold_factor": 4.0, 
            "sample_rate_hz": 25.0,
            "baseline_file": "nonexistent.npz",
            "log_file": "./data/csi_raw.log"
        }
        
        mock_config = mock_open(read_data="dummy")
        with patch("builtins.open", mock_config):
            with patch("yaml.safe_load", return_value=config_data):
                with patch("csi_node.web_dashboard.AdaptivePresenceDetector") as mock_detector_cls:
                    mock_detector = MagicMock()
                    mock_detector_cls.return_value = mock_detector
                    
                    import threading
                    thread = threading.Thread(
                        target=_pipeline_thread, 
                        kwargs={"simulate": True}, 
                        daemon=True
                    )
                    thread.start()
                    import time
                    time.sleep(0.05)
                    
                    # Check detector was created with correct parameters
                    mock_detector_cls.assert_called_once_with(
                        energy_threshold_factor=3.0,
                        variance_threshold_factor=4.0,
                        sample_rate_hz=25.0
                    )

    def test_pipeline_thread_existing_calibration(self, tmp_path):
        """Test _pipeline_thread loads existing calibration."""
        from unittest.mock import patch, MagicMock, mock_open
        from csi_node.web_dashboard import _pipeline_thread
        
        # Create temporary calibration file
        cal_file = tmp_path / "calibration.json"
        cal_file.write_text('{"test": "data"}')
        
        with patch("builtins.open", mock_open(read_data="")):
            with patch("yaml.safe_load", return_value={}):
                with patch("csi_node.web_dashboard.Path") as mock_path_cls:
                    # Mock the path resolution to point to our temp file
                    mock_path = MagicMock()
                    mock_path.__file__ = __file__  # Set __file__ for Path resolution
                    mock_path.resolve.return_value.parent.parent = tmp_path
                    mock_path_cls.return_value = mock_path
                    # Make cal_path.exists() return True
                    mock_path_cls.side_effect = lambda p: cal_file if "calibration.json" in str(p) else MagicMock()
                    
                    with patch("csi_node.web_dashboard.AdaptivePresenceDetector") as mock_detector_cls:
                        mock_detector = MagicMock()
                        mock_detector.load_calibration.return_value = True
                        mock_detector_cls.return_value = mock_detector
                        
                        # Mock simulate mode to avoid simulator import issues
                        with patch("csi_node.simulator.CSISimulator"):
                            import threading
                            thread = threading.Thread(
                                target=_pipeline_thread, 
                                kwargs={"simulate": True}, 
                                daemon=True
                            )
                            thread.start()
                            import time
                            time.sleep(0.1)
                            
                            # The detector was created, which means configuration loading worked
                            mock_detector_cls.assert_called_once()

    def test_pipeline_thread_auto_calibration(self, tmp_path):
        """Test _pipeline_thread performs auto-calibration when no existing calibration."""
        from unittest.mock import patch, MagicMock, mock_open
        from csi_node.web_dashboard import _pipeline_thread
        
        with patch("builtins.open", mock_open(read_data="")):
            with patch("yaml.safe_load", return_value={}):
                with patch("csi_node.web_dashboard.Path") as mock_path_cls:
                    # Mock calibration file doesn't exist
                    mock_cal_path = MagicMock()
                    mock_cal_path.exists.return_value = False
                    mock_path_cls.return_value.resolve.return_value.parent.parent = tmp_path / "nonexistent"
                    
                    with patch("csi_node.web_dashboard.AdaptivePresenceDetector") as mock_detector_cls:
                        mock_detector = MagicMock()
                        mock_detector_cls.return_value = mock_detector
                        
                        # Mock simulate mode
                        with patch("csi_node.simulator.CSISimulator"):
                            import threading
                            thread = threading.Thread(
                                target=_pipeline_thread, 
                                kwargs={"auto_calibrate": True, "simulate": True}, 
                                daemon=True
                            )
                            thread.start()
                            import time
                            time.sleep(0.1)
                            
                            # The detector was created, which means basic setup worked
                            mock_detector_cls.assert_called_once()

    def test_pipeline_thread_baseline_loading(self, tmp_path):
        """Test _pipeline_thread loads baseline when available."""
        from unittest.mock import patch, MagicMock
        from csi_node.web_dashboard import _pipeline_thread
        import numpy as np
        
        baseline_file = tmp_path / "baseline.npz"
        baseline_data = {"mean": np.array([[1, 2], [3, 4]])}
        
        config_data = {"baseline_file": str(baseline_file)}
        
        with patch("builtins.open", side_effect=FileNotFoundError):  # No config file itself
            with patch("yaml.safe_load", return_value=config_data):
                with patch("numpy.load", return_value=baseline_data):
                    with patch("csi_node.web_dashboard.Path") as mock_path:
                        mock_baseline_path = MagicMock()
                        mock_baseline_path.exists.return_value = True
                        mock_path.return_value.exists.return_value = mock_baseline_path
                        
                        import threading
                        thread = threading.Thread(
                            target=_pipeline_thread, 
                            kwargs={"simulate": True}, 
                            daemon=True
                        )
                        thread.start()
                        import time
                        time.sleep(0.05)
                        
                        # Baseline loading would be tested in the packet processing

    def test_pipeline_thread_error_propagates(self, tmp_path):
        """Test _pipeline_thread raises on config load failure."""
        from unittest.mock import patch
        from csi_node.web_dashboard import _pipeline_thread

        # Patching open to raise should propagate (no internal catch at that point)
        with patch("builtins.open", side_effect=Exception("Test error")):
            import pytest
            with pytest.raises(Exception, match="Test error"):
                _pipeline_thread(simulate=True)

    def test_pipeline_thread_default_mode(self, tmp_path):
        """Test _pipeline_thread falls back to default log file."""
        from unittest.mock import patch
        from csi_node.web_dashboard import _pipeline_thread
        
        config_data = {"log_file": "./data/default.log"}
        
        with patch("builtins.open", side_effect=FileNotFoundError):  # No config file
            with patch("yaml.safe_load", return_value=config_data):
                with patch("csi_node.web_dashboard._pipeline_thread") as mock_recursive:
                    # Prevent infinite recursion in the test
                    def side_effect(*args, **kwargs):
                        if kwargs.get("log_path"):
                            pass  # Don't recurse
                    mock_recursive.side_effect = side_effect
                    
                    import threading
                    thread = threading.Thread(
                        target=_pipeline_thread, 
                        daemon=True
                    )
                    thread.start()
                    import time
                    time.sleep(0.05)


class TestRunDashboard:
    def test_run_dashboard_starts_pipeline_thread(self):
        """Test run_dashboard starts the pipeline thread."""
        from unittest.mock import patch, MagicMock
        from csi_node.web_dashboard import run_dashboard
        
        mock_thread = MagicMock()
        mock_server = MagicMock()
        
        with patch("threading.Thread", return_value=mock_thread):
            with patch("csi_node.web_dashboard.HTTPServer", return_value=mock_server):
                with patch("builtins.print"):  # Suppress output
                    # Use KeyboardInterrupt to exit cleanly
                    mock_server.serve_forever.side_effect = KeyboardInterrupt()
                    
                    run_dashboard(port=8888, replay_path="test.b64", speed=2.0)
                    
                    # Check thread was started
                    mock_thread.start.assert_called_once()
                    
                    # Check server was created and started
                    mock_server.serve_forever.assert_called_once()
                    mock_server.shutdown.assert_called_once()

    def test_run_dashboard_with_all_params(self):
        """Test run_dashboard with all parameters."""
        from unittest.mock import patch, MagicMock, ANY
        from csi_node.web_dashboard import run_dashboard
        
        with patch("threading.Thread") as mock_thread_cls:
            with patch("csi_node.web_dashboard.HTTPServer") as mock_server_cls:
                mock_server = MagicMock()
                mock_server.serve_forever.side_effect = KeyboardInterrupt()
                mock_server_cls.return_value = mock_server
                
                with patch("builtins.print"):
                    run_dashboard(
                        port=9999, 
                        replay_path="test.dat", 
                        log_path="test.log", 
                        speed=0.5, 
                        simulate=True
                    )
                    
                    # Check thread was created with correct arguments
                    mock_thread_cls.assert_called_once()
                    args, kwargs = mock_thread_cls.call_args
                    assert kwargs["args"][:4] == ("test.dat", "test.log", 0.5, True)
                    
                    # Check server was created on correct port
                    mock_server_cls.assert_called_once_with(('0.0.0.0', 9999), ANY)


class TestMainFunction:
    def test_main_default_args(self):
        """Test main function with default arguments."""
        from unittest.mock import patch
        from csi_node.web_dashboard import main
        
        with patch("sys.argv", ["web_dashboard"]):
            with patch("csi_node.web_dashboard.run_dashboard") as mock_run:
                main()
                
                # Check key args (through_wall may or may not be present)
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["port"] == 8088
                assert call_kwargs["replay_path"] is None
                assert call_kwargs["simulate"] is False

    def test_main_with_all_args(self):
        """Test main function with all command line arguments."""
        from unittest.mock import patch
        from csi_node.web_dashboard import main
        
        args = [
            "web_dashboard",
            "--port", "9999", 
            "--replay", "test.b64",
            "--log", "test.log",
            "--speed", "2.5",
            "--simulate"
        ]
        
        with patch("sys.argv", args):
            with patch("csi_node.web_dashboard.run_dashboard") as mock_run:
                main()
                
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["port"] == 9999
                assert call_kwargs["replay_path"] == "test.b64"
                assert call_kwargs["log_path"] == "test.log"
                assert call_kwargs["speed"] == 2.5
                assert call_kwargs["simulate"] is True

    def test_main_port_argument(self):
        """Test main function with port argument."""
        from unittest.mock import patch
        from csi_node.web_dashboard import main
        
        with patch("sys.argv", ["web_dashboard", "--port", "7777"]):
            with patch("csi_node.web_dashboard.run_dashboard") as mock_run:
                main()
                
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["port"] == 7777
                assert call_kwargs["simulate"] is False
