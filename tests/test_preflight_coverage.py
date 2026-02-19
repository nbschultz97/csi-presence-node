"""Tests for csi_node.preflight — pre-flight demo checks."""
from __future__ import annotations

import json
import socket
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

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
    main,
)


class TestCheckPythonVersion(unittest.TestCase):
    def test_current_python_passes(self):
        result = check_python_version()
        self.assertTrue(result.passed)
        self.assertIn("3.", result.message)


class TestCheckImports(unittest.TestCase):
    def test_numpy_available(self):
        result = check_numpy()
        self.assertTrue(result.passed)

    def test_scipy_available(self):
        result = check_scipy()
        self.assertTrue(result.passed)

    def test_watchdog_available(self):
        result = check_watchdog()
        self.assertTrue(result.passed)

    def test_yaml_available(self):
        result = check_yaml()
        self.assertTrue(result.passed)


class TestCheckConfig(unittest.TestCase):
    def test_config_exists(self):
        result = check_config()
        self.assertTrue(result.passed)


class TestCheckCalibration(unittest.TestCase):
    def test_calibration_result_type(self):
        result = check_calibration()
        self.assertIsInstance(result, CheckResult)
        self.assertEqual(result.name, "Calibration")


class TestCheckEnvironments(unittest.TestCase):
    def test_environments_result(self):
        result = check_environments()
        self.assertIsInstance(result, CheckResult)
        self.assertTrue(result.passed)


class TestCheckDemoData(unittest.TestCase):
    def test_demo_data_result(self):
        result = check_demo_data()
        self.assertIsInstance(result, CheckResult)
        self.assertTrue(result.passed)


class TestCheckPort(unittest.TestCase):
    def test_available_port(self):
        result = check_port(59123)
        self.assertTrue(result.passed)

    def test_port_in_use(self):
        """Bind a port then check it shows as in use."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.listen(1)
        try:
            result = check_port(port)
            self.assertFalse(result.passed)
            self.assertIn("already in use", result.message)
        finally:
            s.close()


class TestRunPreflight(unittest.TestCase):
    def test_returns_list_of_checks(self):
        results = run_preflight(59124)
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) >= 8)
        for r in results:
            self.assertIsInstance(r, CheckResult)


class TestMain(unittest.TestCase):
    @patch("csi_node.preflight.run_preflight")
    def test_main_all_pass(self, mock_run):
        mock_run.return_value = [
            CheckResult("Test", True, "OK"),
            CheckResult("Test2", True, "OK"),
        ]
        with patch("sys.argv", ["preflight", "--port", "59125"]):
            ret = main()
        self.assertEqual(ret, 0)

    @patch("csi_node.preflight.run_preflight")
    def test_main_some_fail(self, mock_run):
        mock_run.return_value = [
            CheckResult("Test", True, "OK"),
            CheckResult("Bad", False, "broken", True, "fix it"),
        ]
        with patch("sys.argv", ["preflight"]):
            ret = main()
        self.assertEqual(ret, 1)


if __name__ == "__main__":
    unittest.main()
