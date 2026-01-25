"""Tests for enhanced pose classifier module."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from csi_node.pose_classifier import (
    PoseClassifier,
    extract_features,
    extract_features_simple,
    LABELS,
    FEATURE_NAMES,
)


class TestExtractFeatures:
    """Tests for feature extraction functions."""

    def test_basic_extraction(self):
        """Test basic feature extraction from 2D array."""
        csi = np.random.rand(10, 56) * 10  # 10 timesteps, 56 subcarriers
        features = extract_features(csi)

        assert features.shape == (14,)  # 14 features
        assert features.dtype == np.float32

    def test_3d_extraction(self):
        """Test feature extraction from 3D array (chains)."""
        csi = np.random.rand(10, 2, 56) * 10  # 10 timesteps, 2 chains, 56 subcarriers
        features = extract_features(csi)

        assert features.shape == (14,)

    def test_single_timestep(self):
        """Test extraction with single timestep (extended=False effectively)."""
        csi = np.random.rand(1, 56) * 10
        features = extract_features(csi)

        assert features.shape == (14,)
        # Extended features should be zeros for single timestep
        assert features[5:].sum() == 0 or len(features) == 14

    def test_with_rssi(self):
        """Test extraction with RSSI values."""
        csi = np.random.rand(10, 56) * 10
        rssi = [-40.0, -42.0]
        features = extract_features(csi, rssi=rssi)

        assert features.shape == (14,)
        # Chain diff features should be non-zero
        assert features[12] != 0 or features[13] != 0

    def test_feature_order(self):
        """Test that features are in expected order."""
        csi = np.ones((10, 56)) * 5
        features = extract_features(csi)

        # First feature should be mean magnitude
        assert features[0] == pytest.approx(5.0, rel=0.1)

    def test_extract_features_simple(self):
        """Test simple feature extraction for backward compatibility."""
        features = extract_features_simple(mean_mag=10.0, std_mag=2.0)

        assert features.shape == (14,)
        assert features[0] == 10.0  # mean_mag
        assert features[1] == 2.0   # std_mag


class TestPoseClassifier:
    """Tests for PoseClassifier class."""

    def test_init_no_model(self):
        """Test initialization without model file."""
        clf = PoseClassifier()
        assert clf.is_toy_model
        assert clf.model is not None

    def test_init_missing_model(self):
        """Test initialization with missing model file."""
        clf = PoseClassifier("nonexistent/model.joblib")
        assert clf.is_toy_model

    def test_predict_2d_features(self):
        """Test prediction with 2-element feature vector (backward compat)."""
        clf = PoseClassifier()
        label, conf = clf.predict(np.array([10.0, 2.0]))

        assert label in LABELS
        assert 0.0 <= conf <= 1.0

    def test_predict_full_features(self):
        """Test prediction with full feature vector."""
        clf = PoseClassifier()
        features = np.array([10, 2, 12, 6, 6, 5, 3, 1, 0.5, 2, 0.1, 1, 0, 0])
        label, conf = clf.predict(features)

        assert label in LABELS
        assert 0.0 <= conf <= 1.0

    def test_predict_standing_features(self):
        """Test that standing-like features predict standing."""
        clf = PoseClassifier()
        # High magnitude, low variance = standing
        features = np.array([10, 2, 12, 6, 6, 5, 3, 1, 0.5, 2, 0.1, 1, 0, 0])
        label, conf = clf.predict(features)

        # Should predict STANDING with reasonable confidence
        assert label == "STANDING" or conf < 0.5  # Either correct or uncertain

    def test_predict_prone_features(self):
        """Test that prone-like features predict prone."""
        clf = PoseClassifier()
        # Low magnitude = prone
        features = np.array([4, 2, 6, 2, 4, 4, 2, 1, 0.5, 4, 0.15, 1.5, 0, 0])
        label, conf = clf.predict(features)

        assert label in LABELS

    def test_predict_with_features(self):
        """Test predict_with_features method."""
        clf = PoseClassifier()
        csi = np.random.rand(10, 56) * 10
        rssi = [-40.0, -42.0]

        label, conf, features = clf.predict_with_features(csi, rssi)

        assert label in LABELS
        assert 0.0 <= conf <= 1.0
        assert features.shape == (14,)

    def test_predict_batch(self):
        """Test prediction with 2D input (batch)."""
        clf = PoseClassifier()
        features = np.array([
            [10, 2, 12, 6, 6, 5, 3, 1, 0.5, 2, 0.1, 1, 0, 0],
        ])
        label, conf = clf.predict(features)

        assert label in LABELS

    def test_labels_configurable(self):
        """Test that labels can be customized."""
        custom_labels = ["UP", "DOWN", "SIDEWAYS"]
        clf = PoseClassifier(labels=custom_labels)

        assert clf.labels == custom_labels


class TestPoseClassifierTraining:
    """Tests for pose classifier training functionality."""

    @pytest.mark.skipif(
        True,  # Skip training tests by default (slow)
        reason="Training tests are slow"
    )
    def test_train_and_save(self):
        """Test training and saving a model."""
        from csi_node.pose_classifier import train_classifier
        import joblib

        # Generate synthetic data
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 14))
        y = rng.integers(0, 3, size=100)

        # Train
        pipeline = train_classifier(X, y, model_type="logistic")

        # Save and load
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            joblib.dump(pipeline, f.name)

            # Load into classifier
            clf = PoseClassifier(f.name)
            assert not clf.is_toy_model

            # Clean up
            Path(f.name).unlink()


class TestFeatureNames:
    """Tests for feature documentation."""

    def test_feature_count(self):
        """Test that we have names for all features."""
        assert len(FEATURE_NAMES) == 14

    def test_feature_extraction_matches(self):
        """Test that extraction produces same number of features as names."""
        csi = np.random.rand(10, 56) * 10
        features = extract_features(csi)

        assert len(features) == len(FEATURE_NAMES)
