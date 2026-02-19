"""Tests for pre-flight check module."""
import pytest
from csi_node.preflight import (
    check_python_version,
    check_numpy,
    check_scipy,
    check_yaml,
    check_config,
    check_demo_data,
    run_preflight,
)


def test_python_version():
    r = check_python_version()
    assert r.passed  # We're running tests, so Python is fine


def test_numpy():
    r = check_numpy()
    assert r.passed


def test_scipy():
    r = check_scipy()
    assert r.passed


def test_yaml():
    r = check_yaml()
    assert r.passed


def test_config():
    r = check_config()
    assert r.passed


def test_demo_data():
    r = check_demo_data()
    # Should pass either way (file exists or doesn't)
    assert r.passed


def test_full_preflight():
    results = run_preflight(port=19999)  # Use uncommon port
    assert len(results) >= 8
    # At minimum, Python + numpy + scipy + yaml + config should pass
    passed = [r for r in results if r.passed]
    assert len(passed) >= 5
