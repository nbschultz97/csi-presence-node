"""Tests for csi_node.preprocessing — Hampel filter + Butterworth bandpass."""
import numpy as np
import pytest

from csi_node.preprocessing import (
    hampel_filter,
    butterworth_bandpass,
    condition_signal,
)


class TestHampelFilter:
    def test_no_outliers_mostly_unchanged(self):
        """Clean low-variance signal should pass through mostly unchanged."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 0.01, (50, 4))
        result = hampel_filter(x, window_size=5, n_sigma=5.0)
        # Most values should be unchanged
        diff = np.abs(result - x)
        assert np.mean(diff < 0.1) > 0.9

    def test_outlier_replaced(self):
        """A single extreme outlier should be replaced with the local median."""
        x = np.zeros((20, 1))
        x[10, 0] = 100.0  # spike
        result = hampel_filter(x, window_size=5, n_sigma=3.0)
        assert abs(result[10, 0]) < 1.0, "Outlier was not suppressed"

    def test_output_shape_matches_input(self):
        x = np.ones((30, 8))
        result = hampel_filter(x)
        assert result.shape == x.shape

    def test_does_not_modify_input(self):
        x = np.array([[0.0], [0.0], [100.0], [0.0], [0.0]])
        x_orig = x.copy()
        hampel_filter(x)
        np.testing.assert_array_equal(x, x_orig)


class TestButterworthBandpass:
    def test_short_array_returns_input(self):
        """Too-short arrays should pass through unchanged."""
        x = np.ones((5, 2))
        result = butterworth_bandpass(x, fs=30.0)
        np.testing.assert_array_equal(result, x)

    def test_output_shape(self):
        x = np.random.default_rng(0).normal(size=(200, 4))
        result = butterworth_bandpass(x, fs=30.0, low=0.1, high=10.0)
        assert result.shape == x.shape

    def test_invalid_band_returns_input(self):
        """If low >= high, input should be returned unchanged."""
        x = np.ones((100, 2))
        # low > high → invalid band
        result = butterworth_bandpass(x, fs=30.0, low=12.0, high=5.0)
        np.testing.assert_array_equal(result, x)

    def test_dc_removed(self):
        """DC (constant) component should be attenuated by bandpass."""
        x = np.full((200, 1), 5.0)
        result = butterworth_bandpass(x, fs=30.0, low=0.5, high=10.0)
        assert np.abs(result[-1, 0]) < 1.0, "DC was not attenuated"


class TestConditionSignal:
    def test_1d_input_handled(self):
        """1-D input should be reshaped and processed without error."""
        x = np.random.default_rng(1).normal(size=(100,))
        result = condition_signal(x, fs=30.0)
        assert result.ndim == 2
        assert result.shape[0] == 100

    def test_pipeline_runs(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(200, 4))
        x[50, 2] = 999.0  # outlier
        result = condition_signal(x, fs=30.0)
        assert result.shape == x.shape
        # Outlier should be gone
        assert abs(result[50, 2]) < 50.0
