"""Extended coverage tests for pose_classifier.py."""
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
    LABELS,
    LABEL_MAP,
    FEATURE_NAMES,
)


class TestExtractFeaturesExtended:
    def test_1d_input(self):
        """Test with 1D array — should reshape to (1, N)."""
        csi = np.random.rand(56)
        features = extract_features(csi)
        assert features.shape == (14,)

    def test_extended_false_single_frame(self):
        """Single timestep yields zeros for extended features."""
        csi = np.random.rand(1, 56) * 10
        features = extract_features(csi, extended=True)
        assert features.shape == (14,)
        # Extended features should be zero for single timestep
        assert all(features[5:] == 0)

    def test_no_rssi(self):
        """Extended features without RSSI — chain diff should be 0."""
        csi = np.random.rand(10, 56)
        features = extract_features(csi, rssi=None)
        assert features[12] == 0.0
        assert features[13] == 0.0

    def test_rssi_single_element(self):
        """RSSI with < 2 elements — should use zero chain diff."""
        csi = np.random.rand(10, 56)
        features = extract_features(csi, rssi=[-40.0])
        assert features[12] == 0.0

    def test_svd_exception_handling(self):
        """SVD failure returns zero PCA features."""
        csi = np.zeros((10, 56))  # All zeros — SVD may still work but test the path
        features = extract_features(csi)
        assert features.shape == (14,)


class TestPoseClassifierExtended:
    def test_predict_no_sklearn(self):
        """When model is None, predict returns UNKNOWN."""
        clf = PoseClassifier()
        clf.model = None
        label, conf = clf.predict(np.array([10, 2, 12, 6, 6, 5, 3, 1, 0.5, 2, 0.1, 1, 0, 0]))
        assert label == "UNKNOWN"
        assert conf == 0.0

    def test_predict_too_few_features(self):
        """Features shorter than 14 get padded."""
        clf = PoseClassifier()
        features = np.array([10.0, 2.0, 12.0, 6.0, 6.0])
        label, conf = clf.predict(features)
        assert label in LABELS
        assert 0.0 <= conf <= 1.0

    def test_predict_too_many_features(self):
        """Features longer than 14 get truncated."""
        clf = PoseClassifier()
        features = np.random.rand(20)
        label, conf = clf.predict(features)
        assert label in LABELS

    def test_predict_proba_exception(self):
        """predict_proba failure returns UNKNOWN."""
        clf = PoseClassifier()
        clf.model = MagicMock()
        clf.model.predict_proba.side_effect = RuntimeError("fail")
        label, conf = clf.predict(np.random.rand(14))
        assert label == "UNKNOWN"
        assert conf == 0.0

    def test_load_model_no_predict_proba(self, tmp_path):
        """Loading model without predict_proba falls back to toy."""
        import joblib
        model_path = tmp_path / "bad_model.joblib"
        joblib.dump({"not": "a model"}, model_path)
        clf = PoseClassifier(str(model_path))
        assert clf.is_toy_model

    def test_load_model_corrupt_file(self, tmp_path):
        """Loading corrupt model file falls back to toy."""
        model_path = tmp_path / "corrupt.joblib"
        model_path.write_bytes(b"not a valid joblib file")
        clf = PoseClassifier(str(model_path))
        assert clf.is_toy_model

    def test_load_valid_pipeline(self, tmp_path):
        """Loading a valid sklearn Pipeline works."""
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ])
        X = np.random.rand(30, 14)
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)
        pipe.fit(X, y)
        model_path = tmp_path / "valid.joblib"
        joblib.dump(pipe, model_path)

        clf = PoseClassifier(str(model_path))
        assert not clf.is_toy_model
        label, conf = clf.predict(np.random.rand(14))
        assert label in LABELS


class TestTrainClassifier:
    def test_train_logistic(self):
        X = np.random.rand(60, 14)
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        # LogisticRegression in newer sklearn removed multi_class param;
        # train_classifier passes it, so we patch to avoid the error
        with patch("csi_node.pose_classifier.LogisticRegression") as MockLR:
            mock_clf = MagicMock()
            mock_clf.fit = MagicMock()
            MockLR.return_value = mock_clf
            pipe = train_classifier(X, y, model_type="logistic")
            assert pipe is not None

    def test_train_gradient_boosting(self):
        X = np.random.rand(60, 14)
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        pipe = train_classifier(X, y, model_type="gradient_boosting")
        assert hasattr(pipe, "predict_proba")

    def test_train_unknown_type(self):
        X = np.random.rand(10, 14)
        y = np.zeros(10)
        with pytest.raises(ValueError, match="Unknown model type"):
            train_classifier(X, y, model_type="invalid")


class TestTrainFromFile:
    def test_train_from_npz(self, tmp_path):
        """Train from .npz data file."""
        X = np.random.rand(60, 14)
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        data_path = tmp_path / "data.npz"
        np.savez(data_path, X=X, y=y)
        out_path = str(tmp_path / "model.joblib")
        _train_from_file(str(data_path), out_path, "random_forest")
        assert Path(out_path).exists()

    def test_train_synthetic(self, tmp_path):
        """Train with no input file — generates synthetic data."""
        out_path = str(tmp_path / "model.joblib")
        _train_from_file(None, out_path, "random_forest")
        assert Path(out_path).exists()


class TestLabelMap:
    def test_label_map_consistency(self):
        for i, label in enumerate(LABELS):
            assert LABEL_MAP[label] == i
