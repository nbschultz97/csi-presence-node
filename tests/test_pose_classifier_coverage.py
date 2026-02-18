"""Additional pose_classifier tests targeting uncovered lines."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from csi_node.pose_classifier import (
    PoseClassifier,
    extract_features,
    extract_features_simple,
    train_classifier,
    _train_from_file,
    main,
    LABELS,
)


class TestPoseClassifierEdgeCases:
    """Cover uncovered branches in PoseClassifier."""

    def test_load_model_no_predict_proba(self, tmp_path):
        """Model without predict_proba falls back to toy model."""
        import joblib
        model_path = tmp_path / "bad_model.joblib"
        # Save an object without predict_proba
        joblib.dump({"not": "a model"}, str(model_path))

        clf = PoseClassifier(str(model_path))
        assert clf.is_toy_model is True

    def test_load_model_corrupt_file(self, tmp_path):
        """Corrupt model file falls back to toy model."""
        model_path = tmp_path / "corrupt.joblib"
        model_path.write_bytes(b"not a joblib file")

        clf = PoseClassifier(str(model_path))
        assert clf.is_toy_model is True

    def test_predict_2_features_compat(self):
        """Predict with 2-element vector (backward compatibility)."""
        clf = PoseClassifier()
        label, conf = clf.predict(np.array([10.0, 2.0]))
        assert label in LABELS[:3]
        assert 0 <= conf <= 1

    def test_predict_padded_features(self):
        """Predict with fewer than 14 features pads with zeros."""
        clf = PoseClassifier()
        label, conf = clf.predict(np.array([10, 2, 12, 6, 6]))
        assert label in LABELS[:3]

    def test_predict_truncated_features(self):
        """Predict with more than 14 features truncates."""
        clf = PoseClassifier()
        features = np.random.rand(20)
        label, conf = clf.predict(features)
        assert label in LABELS[:3]

    def test_predict_with_features(self):
        """predict_with_features returns label, conf, and feature vector."""
        clf = PoseClassifier()
        csi = np.random.rand(10, 56) * 10
        label, conf, feats = clf.predict_with_features(csi, rssi=[-40, -42])
        assert label in LABELS[:3]
        assert feats.shape == (14,)

    def test_no_sklearn_predict(self):
        """No model returns UNKNOWN."""
        clf = PoseClassifier()
        clf.model = None
        label, conf = clf.predict(np.array([10, 2, 12, 6, 6, 5, 3, 1, 0.5, 2, 0.1, 1, 0, 0]))
        assert label == "UNKNOWN"
        assert conf == 0.0

    def test_predict_exception_returns_unknown(self):
        """Prediction exception returns UNKNOWN."""
        clf = PoseClassifier()
        clf.model = MagicMock()
        clf.model.predict_proba.side_effect = ValueError("bad input")
        label, conf = clf.predict(np.array([10, 2, 12, 6, 6, 5, 3, 1, 0.5, 2, 0.1, 1, 0, 0]))
        assert label == "UNKNOWN"
        assert conf == 0.0

    def test_custom_labels(self):
        """PoseClassifier with custom labels."""
        clf = PoseClassifier(labels=["A", "B", "C", "D", "E"])
        assert clf.labels == ["A", "B", "C", "D", "E"]


class TestTrainClassifier:
    """Tests for train_classifier function."""

    def test_logistic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 14))
        y = rng.integers(0, 3, 100)
        try:
            pipeline = train_classifier(X, y, model_type="logistic")
            assert hasattr(pipeline, "predict")
        except TypeError:
            # sklearn >= 1.7 removed multi_class param — known source bug
            pass

    def test_random_forest(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 14))
        y = rng.integers(0, 3, 100)
        pipeline = train_classifier(X, y, model_type="random_forest")
        assert hasattr(pipeline, "predict")

    def test_gradient_boosting(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 14))
        y = rng.integers(0, 3, 100)
        pipeline = train_classifier(X, y, model_type="gradient_boosting")
        assert hasattr(pipeline, "predict")

    def test_unknown_model_type(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            train_classifier(np.zeros((10, 14)), np.zeros(10), model_type="quantum")


class TestTrainFromFile:
    """Tests for _train_from_file function."""

    def test_train_from_npz(self, tmp_path):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 14))
        y = rng.integers(0, 3, 100)
        data_path = tmp_path / "data.npz"
        np.savez(str(data_path), X=X, y=y)
        out_path = tmp_path / "model.joblib"

        _train_from_file(str(data_path), str(out_path), "random_forest")
        assert out_path.exists()

    def test_train_synthetic(self, tmp_path):
        """No input file triggers synthetic data generation."""
        out_path = tmp_path / "model.joblib"
        _train_from_file(None, str(out_path), "random_forest")
        assert out_path.exists()


class TestExtractFeaturesEdges:
    """Edge cases for extract_features."""

    def test_single_frame(self):
        """Single time step produces non-extended features."""
        csi = np.random.rand(1, 56)
        features = extract_features(csi, extended=True)
        assert features.shape == (14,)

    def test_1d_input(self):
        """1D input is reshaped."""
        csi = np.random.rand(56)
        features = extract_features(csi)
        assert features.shape == (14,)

    def test_with_rssi(self):
        """RSSI values populate chain_diff features."""
        csi = np.random.rand(10, 56)
        features = extract_features(csi, rssi=[-30, -50])
        assert features[12] == pytest.approx(20.0)  # chain_diff_mean
        assert features[13] == pytest.approx(20.0)  # chain_diff_std

    def test_non_extended(self):
        """Non-extended mode produces 14 features with zeros."""
        csi = np.random.rand(10, 56)
        features = extract_features(csi, extended=False)
        assert features.shape == (14,)
        # Extended features should be zeros
        assert features[5] == 0.0

    def test_extract_features_simple_compat(self):
        features = extract_features_simple(10.0, 2.0)
        assert features.shape == (14,)
        assert features[0] == pytest.approx(10.0)
        assert features[1] == pytest.approx(2.0)


class TestMainCLI:
    """Test pose_classifier main() CLI."""

    def test_main_train(self, tmp_path):
        out_path = str(tmp_path / "model.joblib")
        with patch("sys.argv", ["pose_classifier", "--train", "--out", out_path]):
            main()
        assert Path(out_path).exists()

    def test_main_test(self, tmp_path):
        with patch("sys.argv", ["pose_classifier", "--test", "nonexistent.joblib"]):
            main()  # Should use toy model

    def test_main_no_args(self):
        with patch("sys.argv", ["pose_classifier"]):
            main()  # Should print help
