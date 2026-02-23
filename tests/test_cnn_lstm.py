"""Tests for CNN-LSTM model architecture."""

import numpy as np
import pytest

from csi_node.cnn_lstm import (
    CnnLstmConfig,
    CsiCnnLstm,
    ACTIVITY_LABELS,
    generate_synthetic_training_data,
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Config tests ──────────────────────────────────────────────────────────


class TestCnnLstmConfig:
    def test_defaults(self):
        cfg = CnnLstmConfig()
        assert cfg.n_subcarriers == 64
        assert cfg.n_time_steps == 50
        assert cfg.n_classes == 6
        assert cfg.input_features == 64

    def test_input_features_multi_chain(self):
        cfg = CnnLstmConfig(n_subcarriers=64, n_chains=2)
        assert cfg.input_features == 128

    def test_to_from_dict(self):
        cfg = CnnLstmConfig(n_subcarriers=32, lstm_hidden=64)
        d = cfg.to_dict()
        cfg2 = CnnLstmConfig.from_dict(d)
        assert cfg2.n_subcarriers == 32
        assert cfg2.lstm_hidden == 64

    def test_save_load(self, tmp_path):
        cfg = CnnLstmConfig(n_classes=4)
        path = str(tmp_path / "config.json")
        cfg.save(path)
        cfg2 = CnnLstmConfig.load(path)
        assert cfg2.n_classes == 4

    def test_labels_default(self):
        cfg = CnnLstmConfig()
        assert cfg.labels == list(ACTIVITY_LABELS)
        assert len(cfg.labels) == 6

    def test_from_dict_ignores_extra_keys(self):
        d = {"n_subcarriers": 32, "unknown_key": 999}
        cfg = CnnLstmConfig.from_dict(d)
        assert cfg.n_subcarriers == 32


# ── Synthetic data generation ─────────────────────────────────────────────


class TestSyntheticData:
    def test_shape(self):
        cfg = CnnLstmConfig(n_subcarriers=16, n_time_steps=20, n_classes=3,
                            labels=["EMPTY", "STANDING", "WALKING"])
        X, y = generate_synthetic_training_data(cfg, n_samples_per_class=10)
        assert X.shape == (30, 20, 16)
        assert y.shape == (30,)

    def test_all_classes_present(self):
        cfg = CnnLstmConfig()
        X, y = generate_synthetic_training_data(cfg, n_samples_per_class=5)
        for i in range(cfg.n_classes):
            assert i in y

    def test_deterministic_with_seed(self):
        cfg = CnnLstmConfig(n_subcarriers=8, n_time_steps=10, n_classes=2,
                            labels=["EMPTY", "STANDING"])
        X1, y1 = generate_synthetic_training_data(cfg, n_samples_per_class=5, seed=123)
        X2, y2 = generate_synthetic_training_data(cfg, n_samples_per_class=5, seed=123)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_dtype(self):
        cfg = CnnLstmConfig(n_subcarriers=8, n_time_steps=10)
        X, y = generate_synthetic_training_data(cfg, n_samples_per_class=3)
        assert X.dtype == np.float32
        assert y.dtype == np.int64


# ── Model instantiation ──────────────────────────────────────────────────


class TestCsiCnnLstmInit:
    def test_create_default(self):
        model = CsiCnnLstm()
        assert not model.is_trained
        assert model.config.n_classes == 6

    def test_create_custom_config(self):
        cfg = CnnLstmConfig(n_subcarriers=32, n_classes=3, labels=["A", "B", "C"])
        model = CsiCnnLstm(config=cfg)
        assert model.config.n_subcarriers == 32

    def test_summary(self):
        model = CsiCnnLstm()
        s = model.summary()
        assert "CNN-LSTM" in s
        assert "Parameters" in s

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_parameter_count_positive(self):
        model = CsiCnnLstm()
        assert model.parameter_count > 0

    def test_parameter_count_without_torch(self):
        # Even if torch available, we can check the property exists
        model = CsiCnnLstm()
        assert isinstance(model.parameter_count, int)


# ── Input preparation ─────────────────────────────────────────────────────


class TestPrepareInput:
    def setup_method(self):
        self.model = CsiCnnLstm(CnnLstmConfig(n_subcarriers=16, n_time_steps=10))

    def test_2d_input(self):
        x = np.random.randn(10, 16).astype(np.float32)
        out = self.model._prepare_input(x)
        assert out.shape == (1, 10, 16)

    def test_3d_input(self):
        x = np.random.randn(5, 10, 16).astype(np.float32)
        out = self.model._prepare_input(x)
        assert out.shape == (5, 10, 16)

    def test_4d_input_flattened(self):
        x = np.random.randn(3, 10, 2, 8).astype(np.float32)
        out = self.model._prepare_input(x)
        assert out.shape == (3, 10, 16)

    def test_time_padding(self):
        x = np.random.randn(5, 16).astype(np.float32)  # only 5 steps, need 10
        out = self.model._prepare_input(x)
        assert out.shape == (1, 10, 16)
        # First 5 rows should be zero (padding)
        np.testing.assert_array_equal(out[0, :5, :], 0)

    def test_time_truncation(self):
        x = np.random.randn(20, 16).astype(np.float32)  # 20 steps, need 10
        out = self.model._prepare_input(x)
        assert out.shape == (1, 10, 16)

    def test_feature_padding(self):
        x = np.random.randn(10, 8).astype(np.float32)  # only 8 features, need 16
        out = self.model._prepare_input(x)
        assert out.shape == (1, 10, 16)

    def test_feature_truncation(self):
        x = np.random.randn(10, 32).astype(np.float32)  # 32 features, need 16
        out = self.model._prepare_input(x)
        assert out.shape == (1, 10, 16)

    def test_1d_input_raises(self):
        with pytest.raises(ValueError, match="Expected 2D"):
            self.model._prepare_input(np.array([1, 2, 3]))


# ── Prediction (with or without PyTorch) ──────────────────────────────────


class TestPrediction:
    def test_predict_returns_label_and_confidence(self):
        model = CsiCnnLstm(CnnLstmConfig(n_subcarriers=16, n_time_steps=10))
        x = np.random.randn(10, 16).astype(np.float32)
        label, conf = model.predict(x)
        assert isinstance(label, str)
        assert 0.0 <= conf <= 1.0

    def test_predict_proba_shape(self):
        cfg = CnnLstmConfig(n_subcarriers=16, n_time_steps=10, n_classes=4,
                            labels=["A", "B", "C", "D"])
        model = CsiCnnLstm(config=cfg)
        x = np.random.randn(10, 16).astype(np.float32)
        probs = model.predict_proba(x)
        assert probs.shape == (4,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_predict_batch(self):
        cfg = CnnLstmConfig(n_subcarriers=16, n_time_steps=10)
        model = CsiCnnLstm(config=cfg)
        X = np.random.randn(5, 10, 16).astype(np.float32)
        results = model.predict_batch(X)
        assert len(results) == 5
        for label, conf in results:
            assert isinstance(label, str)
            assert 0.0 <= conf <= 1.0


# ── Training (PyTorch only) ──────────────────────────────────────────────


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestTraining:
    def test_train_synthetic(self):
        cfg = CnnLstmConfig(
            n_subcarriers=16, n_time_steps=10, n_classes=3,
            labels=["EMPTY", "STANDING", "WALKING"],
            conv1_filters=8, conv2_filters=16,
            lstm_hidden=16, lstm_layers=1, dense_units=8,
        )
        model = CsiCnnLstm(config=cfg)
        X, y = generate_synthetic_training_data(cfg, n_samples_per_class=20)
        history = model.fit(X, y, epochs=3, verbose=False)
        assert model.is_trained
        assert len(history) == 3
        assert "train_loss" in history[0]
        assert "val_acc" in history[-1]

    def test_predict_after_training(self):
        cfg = CnnLstmConfig(
            n_subcarriers=8, n_time_steps=10, n_classes=2,
            labels=["EMPTY", "STANDING"],
            conv1_filters=4, conv2_filters=8,
            lstm_hidden=8, lstm_layers=1, dense_units=4,
        )
        model = CsiCnnLstm(config=cfg)
        X, y = generate_synthetic_training_data(cfg, n_samples_per_class=30)
        model.fit(X, y, epochs=5, verbose=False)
        label, conf = model.predict(X[0])
        assert label in cfg.labels
        assert conf > 0

    def test_save_load_roundtrip(self, tmp_path):
        cfg = CnnLstmConfig(
            n_subcarriers=8, n_time_steps=10, n_classes=2,
            labels=["A", "B"],
            conv1_filters=4, conv2_filters=8,
            lstm_hidden=8, lstm_layers=1, dense_units=4,
        )
        model = CsiCnnLstm(config=cfg)
        X, y = generate_synthetic_training_data(cfg, n_samples_per_class=10)
        model.fit(X, y, epochs=2, verbose=False)

        path = str(tmp_path / "model.pt")
        model.save(path)

        model2 = CsiCnnLstm(model_path=path)
        assert model2.is_trained
        assert model2.config.n_classes == 2

        # Predictions should match
        x = X[0]
        l1, c1 = model.predict(x)
        l2, c2 = model2.predict(x)
        assert l1 == l2
        assert abs(c1 - c2) < 1e-5


# ── No-PyTorch fallback ──────────────────────────────────────────────────


class TestFallbackBehavior:
    def test_predict_without_trained_model(self):
        """Even untrained model should return valid output."""
        model = CsiCnnLstm(CnnLstmConfig(n_subcarriers=8, n_time_steps=5))
        x = np.random.randn(5, 8).astype(np.float32)
        label, conf = model.predict(x)
        assert isinstance(label, str)
        assert conf >= 0

    def test_summary_always_works(self):
        model = CsiCnnLstm()
        s = model.summary()
        assert len(s) > 50
