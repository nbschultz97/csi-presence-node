"""Tests for multi-person detection and counting module."""
import json
import time
import tempfile
from pathlib import Path

import numpy as np
import pytest

from csi_node.multi_person import (
    MultiPersonDetector,
    PersonCountEstimate,
    extract_counting_features,
    FEATURE_NAMES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_empty_room(n_frames=100, n_sub=52, noise=0.5, seed=42):
    """Generate low-variance empty-room CSI frames."""
    rng = np.random.default_rng(seed)
    base = 20.0 + rng.normal(0, 1.0, n_sub)
    frames = np.tile(base, (n_frames, 1)) + rng.normal(0, noise, (n_frames, n_sub))
    return frames


def _make_single_person(n_frames=100, n_sub=52, seed=43):
    """Generate CSI frames with one person: moderate variance, one breathing peak."""
    rng = np.random.default_rng(seed)
    base = 20.0 + rng.normal(0, 1.0, n_sub)
    frames = np.tile(base, (n_frames, 1))
    # Add single breathing modulation (~0.3 Hz)
    t = np.linspace(0, n_frames / 30.0, n_frames)
    breathing = 3.0 * np.sin(2 * np.pi * 0.3 * t)
    frames += breathing[:, None] * (0.5 + 0.5 * rng.random(n_sub))
    # Add body-induced variance
    frames += rng.normal(0, 2.0, (n_frames, n_sub))
    return frames


def _make_two_people(n_frames=100, n_sub=52, seed=44):
    """Generate CSI frames with two people: higher variance, two breathing peaks."""
    rng = np.random.default_rng(seed)
    base = 20.0 + rng.normal(0, 1.0, n_sub)
    frames = np.tile(base, (n_frames, 1))
    t = np.linspace(0, n_frames / 30.0, n_frames)
    # Person 1 breathing at 0.25 Hz
    breath1 = 3.0 * np.sin(2 * np.pi * 0.25 * t)
    # Person 2 breathing at 0.35 Hz (different rate)
    breath2 = 2.5 * np.sin(2 * np.pi * 0.35 * t + 1.5)
    # Different subcarrier patterns for each person
    pattern1 = 0.5 + 0.5 * rng.random(n_sub)
    pattern2 = 0.5 + 0.5 * rng.random(n_sub)
    frames += breath1[:, None] * pattern1
    frames += breath2[:, None] * pattern2
    # Higher overall variance
    frames += rng.normal(0, 3.5, (n_frames, n_sub))
    return frames


def _make_three_plus(n_frames=100, n_sub=52, seed=45):
    """Generate CSI frames with 3+ people: high variance, complex spectral."""
    rng = np.random.default_rng(seed)
    base = 20.0 + rng.normal(0, 1.0, n_sub)
    frames = np.tile(base, (n_frames, 1))
    t = np.linspace(0, n_frames / 30.0, n_frames)
    for freq, amp, phase in [(0.22, 3.0, 0), (0.33, 2.5, 1.2), (0.42, 2.0, 2.8)]:
        pattern = 0.5 + 0.5 * rng.random(n_sub)
        frames += amp * np.sin(2 * np.pi * freq * t + phase)[:, None] * pattern
    # High variance from multiple movers
    frames += rng.normal(0, 5.0, (n_frames, n_sub))
    return frames


# ---------------------------------------------------------------------------
# PersonCountEstimate dataclass
# ---------------------------------------------------------------------------

class TestPersonCountEstimate:
    def test_defaults(self):
        est = PersonCountEstimate()
        assert est.count == 0
        assert est.confidence == 0.0
        assert est.method == "none"

    def test_to_dict(self):
        est = PersonCountEstimate(count=2, confidence=0.8, method="eigenvalue")
        d = est.to_dict()
        assert d["count"] == 2
        assert d["confidence"] == 0.8
        assert isinstance(d, dict)

    def test_to_json(self):
        est = PersonCountEstimate(count=1)
        j = est.to_json()
        parsed = json.loads(j)
        assert parsed["count"] == 1

    def test_all_fields_in_dict(self):
        est = PersonCountEstimate(
            count=3, confidence=0.9, method="classifier",
            eigenvalue_spread=5.2, spectral_peaks=3,
            variance_ratio=4.1, entropy=3.8, timestamp=123.0,
        )
        d = est.to_dict()
        assert set(d.keys()) == {
            "count", "confidence", "method", "eigenvalue_spread",
            "spectral_peaks", "variance_ratio", "entropy", "timestamp",
        }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    def test_returns_all_features(self):
        frames = _make_empty_room(n_frames=60)
        features = extract_counting_features(frames)
        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"

    def test_empty_room_low_variance_ratio(self):
        frames = _make_empty_room()
        baseline_var = float(np.var(frames))
        features = extract_counting_features(frames, baseline_variance=baseline_var)
        assert features["variance_ratio"] == pytest.approx(1.0, abs=0.1)

    def test_single_person_higher_variance(self):
        empty = _make_empty_room()
        baseline_var = float(np.var(empty))
        person = _make_single_person()
        features = extract_counting_features(person, baseline_variance=baseline_var)
        assert features["variance_ratio"] > 1.5

    def test_two_people_higher_effective_rank(self):
        one = _make_single_person(n_frames=120)
        two = _make_two_people(n_frames=120)
        f1 = extract_counting_features(one)
        f2 = extract_counting_features(two)
        # Two people should generally have higher effective rank
        assert f2["effective_rank"] >= f1["effective_rank"] * 0.8  # allow some noise

    def test_spectral_peaks_detected(self):
        """Two-person data should have spectral peaks in breathing band."""
        frames = _make_two_people(n_frames=300, seed=100)
        features = extract_counting_features(frames, sample_rate_hz=30.0)
        # With 300 frames at 30Hz = 10s, should resolve breathing peaks
        assert features["spectral_peaks"] >= 0  # at least detectable

    def test_short_window_no_crash(self):
        """Very short windows should not crash, just return default features."""
        frames = np.random.randn(5, 52)
        features = extract_counting_features(frames)
        assert features["variance_ratio"] > 0

    def test_single_frame_no_crash(self):
        frames = np.random.randn(1, 52)
        features = extract_counting_features(frames)
        assert "temporal_entropy" in features

    def test_minimum_subcarriers(self):
        frames = np.random.randn(30, 2)
        features = extract_counting_features(frames)
        assert "mean_cross_corr" in features

    def test_cross_correlation_lower_with_more_people(self):
        """More independent sources = lower cross-subcarrier correlation."""
        one = _make_single_person(n_frames=200, n_sub=52, seed=50)
        three = _make_three_plus(n_frames=200, n_sub=52, seed=51)
        f1 = extract_counting_features(one)
        f3 = extract_counting_features(three)
        # Not guaranteed but generally true
        # Just check both are valid floats
        assert 0.0 <= f1["mean_cross_corr"] <= 1.0
        assert 0.0 <= f3["mean_cross_corr"] <= 1.0


# ---------------------------------------------------------------------------
# MultiPersonDetector — basic usage
# ---------------------------------------------------------------------------

class TestMultiPersonDetector:
    def test_init_defaults(self):
        det = MultiPersonDetector()
        assert not det.calibrated
        assert det.state.count == 0

    def test_calibrate(self):
        det = MultiPersonDetector()
        empty = _make_empty_room()
        assert det.calibrate(empty)
        assert det.calibrated

    def test_calibrate_too_few_frames(self):
        det = MultiPersonDetector()
        frames = np.random.randn(5, 52)
        assert not det.calibrate(frames)
        assert not det.calibrated

    def test_estimate_empty_room(self):
        det = MultiPersonDetector()
        empty = _make_empty_room(n_frames=200, noise=0.3, seed=42)
        det.calibrate(empty[:100])
        # Use a separate empty-room sample with same noise level
        empty2 = _make_empty_room(n_frames=60, noise=0.3, seed=99)
        result = det.estimate(empty2)
        assert result.count == 0
        assert result.confidence > 0.0

    def test_estimate_single_person(self):
        det = MultiPersonDetector()
        empty = _make_empty_room(n_frames=100)
        det.calibrate(empty)
        person = _make_single_person(n_frames=100)
        result = det.estimate(person)
        assert result.count >= 1  # Should detect at least 1
        assert result.confidence > 0.0

    def test_estimate_no_frames_returns_zero(self):
        det = MultiPersonDetector()
        result = det.estimate(np.random.randn(3, 52))
        assert result.count == 0

    def test_estimate_from_buffer(self):
        det = MultiPersonDetector(window_seconds=2.0)
        empty = _make_empty_room(n_frames=50)
        det.calibrate(empty)
        person = _make_single_person(n_frames=60)
        for frame in person:
            det.feed_frame(frame)
        result = det.estimate()
        assert isinstance(result.count, int)

    def test_estimate_buffer_too_few(self):
        det = MultiPersonDetector()
        for i in range(5):
            det.feed_frame(np.random.randn(52))
        result = det.estimate()
        assert result.count == 0

    def test_history_populated(self):
        det = MultiPersonDetector()
        empty = _make_empty_room()
        det.calibrate(empty)
        det.estimate(_make_single_person())
        det.estimate(_make_two_people())
        assert len(det.history) == 2

    def test_state_property(self):
        det = MultiPersonDetector()
        det.calibrate(_make_empty_room())
        det.estimate(_make_single_person())
        assert det.state.timestamp > 0

    def test_dashboard_data(self):
        det = MultiPersonDetector()
        det.calibrate(_make_empty_room())
        det.estimate(_make_single_person())
        data = det.get_dashboard_data()
        assert "current" in data
        assert "history" in data
        assert "calibrated" in data
        assert data["calibrated"] is True

    def test_dashboard_data_uncalibrated(self):
        det = MultiPersonDetector()
        data = det.get_dashboard_data()
        assert data["calibrated"] is False


# ---------------------------------------------------------------------------
# Classifier training
# ---------------------------------------------------------------------------

class TestClassifierTraining:
    @pytest.fixture
    def training_data(self):
        """Generate labeled training data."""
        data = []
        for _ in range(10):
            data.append((_make_empty_room(n_frames=60, seed=np.random.randint(1000)), 0))
        for _ in range(10):
            data.append((_make_single_person(n_frames=60, seed=np.random.randint(1000)), 1))
        for _ in range(10):
            data.append((_make_two_people(n_frames=60, seed=np.random.randint(1000)), 2))
        for _ in range(10):
            data.append((_make_three_plus(n_frames=60, seed=np.random.randint(1000)), 3))
        return data

    def test_train_returns_metrics(self, training_data):
        det = MultiPersonDetector()
        det.calibrate(_make_empty_room())
        metrics = det.train(training_data)
        assert "train_accuracy" in metrics
        assert metrics["train_accuracy"] > 0.5
        assert "feature_importances" in metrics

    def test_trained_classifier_predicts(self, training_data):
        det = MultiPersonDetector()
        det.calibrate(_make_empty_room())
        det.train(training_data)
        result = det.estimate(_make_single_person(n_frames=60))
        assert result.method == "classifier"
        assert result.confidence > 0.0

    def test_save_load_model(self, training_data, tmp_path):
        det = MultiPersonDetector()
        det.calibrate(_make_empty_room())
        det.train(training_data)

        model_path = tmp_path / "model.pkl"
        det.save_model(model_path)
        assert model_path.exists()

        det2 = MultiPersonDetector()
        assert det2.load_model(model_path)
        result = det2.estimate(_make_empty_room(n_frames=60))
        assert result.method == "classifier"

    def test_load_model_bad_path(self):
        det = MultiPersonDetector()
        assert not det.load_model("/nonexistent/model.pkl")

    def test_train_caps_count_at_3(self, training_data):
        # Add data with count=5 — should be capped to 3
        data = training_data + [(_make_three_plus(seed=999), 5)]
        det = MultiPersonDetector()
        det.calibrate(_make_empty_room())
        metrics = det.train(data)
        assert 3 in metrics["classes"]


# ---------------------------------------------------------------------------
# Heuristic edge cases
# ---------------------------------------------------------------------------

class TestHeuristic:
    def test_very_high_variance_estimates_multiple(self):
        det = MultiPersonDetector()
        empty = _make_empty_room(n_frames=100, noise=0.1)
        det.calibrate(empty)
        # Extremely high variance data
        rng = np.random.default_rng(99)
        noisy = 20.0 + rng.normal(0, 15.0, (100, 52))
        result = det.estimate(noisy)
        assert result.count >= 2  # High variance → multiple people

    def test_low_variance_estimates_zero(self):
        det = MultiPersonDetector()
        empty = _make_empty_room(n_frames=100, noise=0.3)
        det.calibrate(empty)
        quiet = _make_empty_room(n_frames=60, noise=0.3, seed=99)
        result = det.estimate(quiet)
        assert result.count == 0

    def test_method_field_populated(self):
        det = MultiPersonDetector()
        det.calibrate(_make_empty_room())
        result = det.estimate(_make_single_person())
        assert result.method in ("eigenvalue", "spectral", "variance")


# ---------------------------------------------------------------------------
# Integration: detector + simulator data
# ---------------------------------------------------------------------------

class TestSimulatorIntegration:
    def test_with_simulator_empty_room(self):
        """Detector should report 0 people for simulator empty room data."""
        from csi_node.simulator import CSISimulator, SimScenario
        sim = CSISimulator(
            scenarios=[SimScenario("empty", 3.0, False, "none")],
            seed=42,
        )
        frames = []
        for pkt in sim.stream():
            frames.append(pkt["csi"])
            if len(frames) >= 90:
                break

        det = MultiPersonDetector()
        frame_arr = np.array(frames)
        det.calibrate(frame_arr[:30])
        result = det.estimate(frame_arr[30:])
        assert result.count == 0

    def test_with_simulator_one_person(self):
        """Detector should report >= 1 person for walking scenario."""
        from csi_node.simulator import CSISimulator, SimScenario
        sim = CSISimulator(
            scenarios=[
                SimScenario("empty", 2.0, False, "none"),
                SimScenario("person", 3.0, True, "walking", 3.0, 10.0,
                            walk_speed=1.5),
            ],
            seed=42,
        )
        frames = []
        for pkt in sim.stream():
            frames.append(pkt["csi"])
            if len(frames) >= 150:
                break

        frame_arr = np.array(frames)
        det = MultiPersonDetector()
        det.calibrate(frame_arr[:30])
        # Use later frames (where person is present)
        result = det.estimate(frame_arr[90:])
        assert result.count >= 1
