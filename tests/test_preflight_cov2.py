"""Test coverage for preflight.py - Focus on ImportError branches and edge cases."""

import unittest
from unittest.mock import Mock, patch, mock_open
import sys
import json
import time
from pathlib import Path
import tempfile

from csi_node.preflight import (
    CheckResult,
    check_python_version,
    check_numpy,
    check_scipy,
    check_watchdog, 
    check_yaml,
    check_config,
    check_calibration,
    check_environments,
    check_demo_data,
    check_port,
    run_preflight,
    main
)


class TestPreflightImportErrors(unittest.TestCase):
    """Test ImportError handling in preflight checks."""

    def test_check_numpy_import_error(self):
        """Test check_numpy when numpy import fails."""
        with patch.dict(sys.modules, {'numpy': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'numpy'")):
                result = check_numpy()
                
                self.assertFalse(result.passed)
                self.assertEqual(result.name, "NumPy")
                self.assertEqual(result.message, "not installed")
                self.assertTrue(result.fixable)
                self.assertEqual(result.fix_hint, "pip install numpy")

    def test_check_scipy_import_error(self):
        """Test check_scipy when scipy import fails."""
        with patch.dict(sys.modules, {'scipy': None, 'scipy.signal': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'scipy'")):
                result = check_scipy()
                
                self.assertFalse(result.passed)
                self.assertEqual(result.name, "SciPy")
                self.assertEqual(result.message, "not installed — Butterworth filter unavailable")
                self.assertTrue(result.fixable)
                self.assertEqual(result.fix_hint, "pip install scipy")

    def test_check_watchdog_import_error(self):
        """Test check_watchdog when watchdog import fails."""
        with patch.dict(sys.modules, {'watchdog': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'watchdog'")):
                result = check_watchdog()
                
                self.assertFalse(result.passed)
                self.assertEqual(result.name, "Watchdog")
                self.assertEqual(result.message, "not installed")
                self.assertTrue(result.fixable)
                self.assertEqual(result.fix_hint, "pip install watchdog")

    def test_check_yaml_import_error(self):
        """Test check_yaml when yaml import fails."""
        with patch.dict(sys.modules, {'yaml': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'yaml'")):
                result = check_yaml()
                
                self.assertFalse(result.passed)
                self.assertEqual(result.name, "PyYAML")
                self.assertEqual(result.message, "not installed")
                self.assertTrue(result.fixable)
                self.assertEqual(result.fix_hint, "pip install pyyaml")

    def test_check_numpy_success(self):
        """Test check_numpy with successful import."""
        mock_numpy = Mock()
        mock_numpy.__version__ = "1.21.0"
        
        with patch.dict(sys.modules, {'numpy': mock_numpy}):
            result = check_numpy()
            
            self.assertTrue(result.passed)
            self.assertEqual(result.name, "NumPy")
            self.assertEqual(result.message, "v1.21.0")

    def test_check_scipy_success(self):
        """Test check_scipy with successful import."""
        mock_scipy = Mock()
        mock_scipy.__version__ = "1.7.0"
        mock_signal = Mock()
        mock_signal.butter = Mock()
        mock_signal.sosfilt = Mock()
        
        with patch.dict(sys.modules, {'scipy': mock_scipy, 'scipy.signal': mock_signal}):
            result = check_scipy()
            
            self.assertTrue(result.passed)
            self.assertEqual(result.name, "SciPy (signal processing)")
            self.assertEqual(result.message, "v1.7.0")

    def test_check_watchdog_success(self):
        """Test check_watchdog with successful import."""
        mock_watchdog = Mock()
        mock_watchdog.__version__ = "2.1.0"
        
        with patch.dict(sys.modules, {'watchdog': mock_watchdog}):
            result = check_watchdog()
            
            self.assertTrue(result.passed)
            self.assertEqual(result.name, "Watchdog (file monitoring)")
            self.assertEqual(result.message, "v2.1.0")

    def test_check_watchdog_no_version(self):
        """Test check_watchdog when version attribute is missing."""
        mock_watchdog = Mock()
        del mock_watchdog.__version__  # Remove version attribute
        
        with patch.dict(sys.modules, {'watchdog': mock_watchdog}):
            result = check_watchdog()
            
            self.assertTrue(result.passed)
            self.assertEqual(result.name, "Watchdog (file monitoring)")
            self.assertEqual(result.message, "vinstalled")

    def test_check_yaml_success(self):
        """Test check_yaml with successful import."""
        mock_yaml = Mock()
        
        with patch.dict(sys.modules, {'yaml': mock_yaml}):
            result = check_yaml()
            
            self.assertTrue(result.passed)
            self.assertEqual(result.name, "PyYAML")
            self.assertEqual(result.message, "OK")


class TestConfigEdgeCases(unittest.TestCase):
    """Test config validation edge cases."""

    @patch('pathlib.Path.exists')
    def test_check_config_missing_file(self, mock_exists):
        """Test check_config when config file doesn't exist."""
        mock_exists.return_value = False
        
        result = check_config()
        
        self.assertFalse(result.passed)
        self.assertEqual(result.name, "Config")
        self.assertTrue("Missing:" in result.message)

    @patch('pathlib.Path.exists')
    @patch('builtins.open')
    @patch('yaml.safe_load')
    def test_check_config_negative_path_loss_exponent(self, mock_yaml_load, mock_open_func, mock_exists):
        """Test check_config with negative path_loss_exponent."""
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            'path_loss_exponent': -1.5,  # Invalid negative value
            'channel': 36,
            'bandwidth': '80MHz'
        }
        
        result = check_config()
        
        self.assertFalse(result.passed)
        self.assertEqual(result.name, "Config")
        self.assertIn("path_loss_exponent is negative", result.message)

    @patch('pathlib.Path.exists')
    @patch('builtins.open')
    @patch('yaml.safe_load')
    def test_check_config_no_channel(self, mock_yaml_load, mock_open_func, mock_exists):
        """Test check_config with no channel set."""
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            'path_loss_exponent': 2.0,
            'channel': None,  # No channel
            'bandwidth': '80MHz'
        }
        
        result = check_config()
        
        self.assertFalse(result.passed)
        self.assertEqual(result.name, "Config")
        self.assertIn("no channel set", result.message)

    @patch('pathlib.Path.exists')
    @patch('builtins.open')
    @patch('yaml.safe_load')
    def test_check_config_multiple_issues(self, mock_yaml_load, mock_open_func, mock_exists):
        """Test check_config with multiple issues."""
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            'path_loss_exponent': -2.0,  # Negative
            'channel': '',  # Empty channel
            'bandwidth': '80MHz'
        }
        
        result = check_config()
        
        self.assertFalse(result.passed)
        self.assertEqual(result.name, "Config")
        self.assertIn("path_loss_exponent is negative", result.message)
        self.assertIn("no channel set", result.message)

    @patch('pathlib.Path.exists')
    @patch('builtins.open')
    @patch('yaml.safe_load')
    def test_check_config_yaml_exception(self, mock_yaml_load, mock_open_func, mock_exists):
        """Test check_config when YAML parsing raises exception."""
        mock_exists.return_value = True
        mock_yaml_load.side_effect = Exception("YAML parse error")
        
        result = check_config()
        
        self.assertFalse(result.passed)
        self.assertEqual(result.name, "Config")
        self.assertEqual(result.message, "YAML parse error")

    @patch('pathlib.Path.exists')
    @patch('builtins.open')
    @patch('yaml.safe_load')
    def test_check_config_success(self, mock_yaml_load, mock_open_func, mock_exists):
        """Test check_config with valid config."""
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            'path_loss_exponent': 2.0,
            'channel': 36,
            'bandwidth': '80MHz'
        }
        
        result = check_config()
        
        self.assertTrue(result.passed)
        self.assertEqual(result.name, "Config")
        self.assertEqual(result.message, "ch36 @ 80MHzMHz")


class TestCalibrationEdgeCases(unittest.TestCase):
    """Test calibration validation edge cases."""

    @patch('pathlib.Path.exists')
    def test_check_calibration_no_file(self, mock_exists):
        """Test check_calibration when calibration file doesn't exist."""
        mock_exists.return_value = False
        
        result = check_calibration()
        
        self.assertFalse(result.passed)
        self.assertEqual(result.name, "Calibration")
        self.assertIn("No calibration file", result.message)
        self.assertIn("Run: python run.py --demo", result.fix_hint)

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.read_text')
    @patch('time.time')
    def test_check_calibration_stale(self, mock_time, mock_read_text, mock_exists):
        """Test check_calibration with stale calibration (>24h old)."""
        mock_exists.return_value = True
        mock_time.return_value = 1000000.0  # Current time
        
        # Create calibration data that's 25 hours old
        old_timestamp = 1000000.0 - (25 * 3600)  # 25 hours ago
        cal_data = {
            'timestamp': old_timestamp,
            'baseline_energy': 1.5
        }
        mock_read_text.return_value = json.dumps(cal_data)
        
        result = check_calibration()
        
        self.assertFalse(result.passed)
        self.assertEqual(result.name, "Calibration")
        self.assertIn("Stale (25h old)", result.message)

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.read_text')
    def test_check_calibration_corrupt_file(self, mock_read_text, mock_exists):
        """Test check_calibration with corrupt JSON file."""
        mock_exists.return_value = True
        mock_read_text.return_value = "invalid json{{"
        
        result = check_calibration()
        
        self.assertFalse(result.passed)
        self.assertEqual(result.name, "Calibration")
        self.assertIn("Corrupt:", result.message)

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.read_text')
    @patch('time.time')
    def test_check_calibration_success(self, mock_time, mock_read_text, mock_exists):
        """Test check_calibration with valid recent calibration."""
        mock_exists.return_value = True
        mock_time.return_value = 1000000.0  # Current time
        
        # Create calibration data that's 2 hours old
        recent_timestamp = 1000000.0 - (2 * 3600)  # 2 hours ago
        cal_data = {
            'timestamp': recent_timestamp,
            'baseline_energy': 1.5
        }
        mock_read_text.return_value = json.dumps(cal_data)
        
        result = check_calibration()
        
        self.assertTrue(result.passed)
        self.assertEqual(result.name, "Calibration")
        self.assertIn("Energy baseline: 1.5, 2.0h old", result.message)

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.read_text')
    @patch('time.time')
    def test_check_calibration_no_timestamp(self, mock_time, mock_read_text, mock_exists):
        """Test check_calibration with missing timestamp."""
        mock_exists.return_value = True
        mock_time.return_value = 1000000.0
        
        cal_data = {
            'baseline_energy': 1.5
            # No timestamp
        }
        mock_read_text.return_value = json.dumps(cal_data)
        
        result = check_calibration()
        
        # Should handle missing timestamp gracefully (inf hours old -> stale)
        self.assertFalse(result.passed)
        self.assertIn("Stale", result.message)


class TestPortCheck(unittest.TestCase):
    """Test port checking functionality."""

    @patch('socket.socket')
    def test_check_port_available(self, mock_socket_class):
        """Test check_port when port is available."""
        mock_socket = Mock()
        mock_socket.connect_ex.return_value = 1  # Connection refused (port available)
        mock_socket.close = Mock()
        mock_socket_class.return_value = mock_socket
        
        result = check_port(8088)
        
        self.assertTrue(result.passed)
        self.assertEqual(result.name, "Port")
        self.assertEqual(result.message, "Port 8088 available")
        mock_socket.close.assert_called_once()

    @patch('socket.socket')
    def test_check_port_in_use(self, mock_socket_class):
        """Test check_port when port is in use."""
        mock_socket = Mock()
        mock_socket.connect_ex.return_value = 0  # Connection successful (port in use)
        mock_socket.close = Mock()
        mock_socket_class.return_value = mock_socket
        
        result = check_port(8088)
        
        self.assertFalse(result.passed)
        self.assertEqual(result.name, "Port")
        self.assertEqual(result.message, "Port 8088 already in use")
        self.assertEqual(result.fix_hint, "Use --port 8089")
        mock_socket.close.assert_called_once()

    @patch('socket.socket')
    def test_check_port_exception(self, mock_socket_class):
        """Test check_port when socket operations raise exception."""
        mock_socket_class.side_effect = Exception("Socket error")
        
        result = check_port(8088)
        
        self.assertTrue(result.passed)  # Should gracefully handle exception
        self.assertEqual(result.name, "Port")
        self.assertEqual(result.message, "Port 8088 (check skipped)")


class TestOtherChecks(unittest.TestCase):
    """Test other check functions."""

    def test_check_python_version_valid(self):
        """Test check_python_version with valid version."""
        from collections import namedtuple
        VersionInfo = namedtuple('VersionInfo', ['major', 'minor', 'micro'])
        with patch('sys.version_info', VersionInfo(3, 11, 0)):
            result = check_python_version()
            
            self.assertTrue(result.passed)
            self.assertEqual(result.name, "Python Version")
            self.assertEqual(result.message, "Python 3.11.0")

    def test_check_python_version_invalid(self):
        """Test check_python_version with invalid version."""
        from collections import namedtuple
        VersionInfo = namedtuple('VersionInfo', ['major', 'minor', 'micro'])
        with patch('sys.version_info', VersionInfo(3, 9, 0)):
            result = check_python_version()
            
            self.assertFalse(result.passed)
            self.assertEqual(result.name, "Python Version")
            self.assertIn("need 3.10+", result.message)

    @patch('pathlib.Path.exists')
    def test_check_environments_no_dir(self, mock_exists):
        """Test check_environments when env directory doesn't exist."""
        mock_exists.return_value = False
        
        result = check_environments()
        
        self.assertTrue(result.passed)  # OK for first demo
        self.assertEqual(result.name, "Environment Profiles")
        self.assertEqual(result.message, "None saved (OK for first demo)")

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.glob')
    def test_check_environments_no_profiles(self, mock_glob, mock_exists):
        """Test check_environments when directory exists but no profiles."""
        mock_exists.return_value = True
        mock_glob.return_value = []  # No profiles
        
        result = check_environments()
        
        self.assertTrue(result.passed)
        self.assertEqual(result.name, "Environment Profiles")
        self.assertEqual(result.message, "None saved")

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.glob')
    def test_check_environments_with_profiles(self, mock_glob, mock_exists):
        """Test check_environments with existing profiles."""
        mock_exists.return_value = True
        
        # Mock profile files
        mock_profiles = [Mock(), Mock()]
        mock_profiles[0].stem = 'home'
        mock_profiles[1].stem = 'office'
        mock_glob.return_value = mock_profiles
        
        result = check_environments()
        
        self.assertTrue(result.passed)
        self.assertEqual(result.name, "Environment Profiles")
        self.assertEqual(result.message, "2 saved: home, office")

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    def test_check_demo_data_exists(self, mock_stat, mock_exists):
        """Test check_demo_data when demo log exists."""
        mock_exists.return_value = True
        mock_stat_obj = Mock()
        mock_stat_obj.st_size = 2048  # 2KB
        mock_stat.return_value = mock_stat_obj
        
        result = check_demo_data()
        
        self.assertTrue(result.passed)
        self.assertEqual(result.name, "Demo Data")
        self.assertEqual(result.message, "demo_csi.log (2 KB)")

    @patch('pathlib.Path.exists')
    def test_check_demo_data_not_exists(self, mock_exists):
        """Test check_demo_data when demo log doesn't exist."""
        mock_exists.return_value = False
        
        result = check_demo_data()
        
        self.assertTrue(result.passed)  # Still OK, simulation works
        self.assertEqual(result.name, "Demo Data")
        self.assertEqual(result.message, "No replay data (simulation mode still works)")


class TestMainFunction(unittest.TestCase):
    """Test main function and run_preflight."""

    def test_run_preflight(self):
        """Test run_preflight returns all check results."""
        with patch('csi_node.preflight.check_python_version') as mock_py, \
             patch('csi_node.preflight.check_numpy') as mock_numpy, \
             patch('csi_node.preflight.check_scipy') as mock_scipy, \
             patch('csi_node.preflight.check_watchdog') as mock_watchdog, \
             patch('csi_node.preflight.check_yaml') as mock_yaml, \
             patch('csi_node.preflight.check_config') as mock_config, \
             patch('csi_node.preflight.check_calibration') as mock_cal, \
             patch('csi_node.preflight.check_environments') as mock_env, \
             patch('csi_node.preflight.check_demo_data') as mock_demo, \
             patch('csi_node.preflight.check_port') as mock_port:
            
            # Mock all checks to return passed results
            for mock_check in [mock_py, mock_numpy, mock_scipy, mock_watchdog, 
                             mock_yaml, mock_config, mock_cal, mock_env, mock_demo, mock_port]:
                mock_check.return_value = CheckResult("Test", True, "OK")
            
            results = run_preflight(8088)
            
            self.assertEqual(len(results), 10)  # All checks
            mock_port.assert_called_with(8088)

    @patch('argparse.ArgumentParser.parse_known_args')
    @patch('csi_node.preflight.run_preflight')
    @patch('builtins.print')
    def test_main_all_pass(self, mock_print, mock_run_preflight, mock_parse_args):
        """Test main function when all checks pass."""
        mock_args = Mock()
        mock_args.port = 8088
        mock_parse_args.return_value = (mock_args, [])
        
        # Mock all checks passing
        mock_run_preflight.return_value = [
            CheckResult("Test1", True, "OK"),
            CheckResult("Test2", True, "OK")
        ]
        
        result = main()
        
        self.assertEqual(result, 0)  # Success exit code
        mock_run_preflight.assert_called_with(8088)

    @patch('argparse.ArgumentParser.parse_known_args')
    @patch('csi_node.preflight.run_preflight')
    @patch('builtins.print')
    def test_main_some_fail(self, mock_print, mock_run_preflight, mock_parse_args):
        """Test main function when some checks fail."""
        mock_args = Mock()
        mock_args.port = 8088
        mock_parse_args.return_value = (mock_args, [])
        
        # Mock some checks failing
        mock_run_preflight.return_value = [
            CheckResult("Test1", True, "OK"),
            CheckResult("Test2", False, "Failed", fixable=True, fix_hint="Fix this")
        ]
        
        result = main()
        
        self.assertEqual(result, 1)  # Failure exit code


if __name__ == '__main__':
    unittest.main()