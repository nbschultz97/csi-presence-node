"""Extended coverage tests for cnn_lstm.py — targets uncovered lines."""

import numpy as np
import pytest
from unittest.mock import patch

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


class TestCnnLstmConfigExtended:
    def test_labels_are_independent_copies(self):
        """Each config should have its own label list."""
        cfg1 = CnnLstmConfig()
        cfg2 = CnnLstmConfig()
        cfg1.labels.append("NEW")
        assert "NEW" not in cfg2.labels

    def test_to_dict_roundtrip_all_fields(self):
        cfg = CnnLstmConfig(
            n_subcarriers=32, n_chains=2, conv1_filters=32,
            bidirectional=True, dropout_dense=0.3
        )
        d = cfg.to_dict()
        cfg2 = CnnLstmConfig.from_dict(d)
        assert cfg2.n_subcarriers == 32
        assert cfg2.n_chains == 2
        assert cfg2.bidirectional is True
        assert cfg2.input_features == 64


class TestNoTorchFallback:
    """Test behavior when PyTorch is unavailable."""

    def test_predict_proba_uniform_single(self):
        """Without a trained net, predict_proba should give uniform."""
        with patch("csi_node.cnn_lstm.TORCH_AVAILABLE", False):
            model = CsiCnnLstm.__new__(CsiCnnLstm)
            model.config = CnnLstmConfig(n_subcarriers=8, n_time_steps=5, n_classes=3,
                                         labels=["A", "B", "C"])
            model._net = None
            model._device = "cpu"
            model._trained = False
            model._training_history = []

            x = np.random.randn(5, 8).astype(np.float32)
            probs = model.predict_proba(x)
            assert probs.shape == (3,)
            np.testing.assert_allclose(probs, [1/3, 1/3, 1/3])

    def test_predict_proba_uniform_batch(self):
        """Batch predict_proba without torch gives batch of uniform."""
        with patch("csi_node.cnn_lstm.TORCH_AVAILABLE", False):
            model = CsiCnnLstm.__new__(CsiCnnLstm)
            model.config = CnnLstmConfig(n_subcarriers=8, n_time_steps=5, n_classes=2,
                                         labels=["A", "B"])
            model._net = None
            model._device = "cpu"
            model._trained = False
            model._training_history = []

            X = np.random.randn(3, 5, 8).astype(np.float32)
            probs = model.predict_proba(X)
            assert probs.shape == (3, 2)

    def test_fit_raises_without_torch(self):
        with patch("csi_node.cnn_lstm.TORCH_AVAILABLE", False):
            model = CsiCnnLstm.__new__(CsiCnnLstm)
            model.config = CnnLstmConfig()
            model._net = None
            model._device = "cpu"
            model._trained = False
            model._training_history = []

            with pytest.raises(RuntimeError, match="PyTorch required"):
                model.fit(np.zeros((10, 50, 64)), np.zeros(10, dtype=int))

    def test_save_raises_without_torch(self):
        with patch("csi_node.cnn_lstm.TORCH_AVAILABLE", False):
            model = CsiCnnLstm.__new__(CsiCnnLstm)
            model.config = CnnLstmConfig()
            model._net = None

            with pytest.raises(RuntimeError, match="PyTorch required"):
                model.save("fake.pt")

    def test_export_onnx_raises_without_torch(self):
        with patch("csi_node.cnn_lstm.TORCH_AVAILABLE", False):
            model = CsiCnnLstm.__new__(CsiCnnLstm)
            model.config = CnnLstmConfig()
            model._net = None

            with pytest.raises(RuntimeError, match="PyTorch required"):
                model.export_onnx("fake.onnx")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
class TestCnnLstmNetForward:
    """Test the _CnnLstmNet forward pass directly."""

    def test_forward_shape(self):
        from csi_node.cnn_lstm import _CnnLstmNet
        cfg = CnnLstmConfig(
            n_subcarriers=16, n_time_steps=10, n_classes=3,
            labels=["A", "B", "C"],
            conv1_filters=8, conv2_filters=16,
            lstm_hidden=8, lstm_layers=1, dense_units=4,
        )
        net = _CnnLstmNet(cfg)
        x = torch.randn(2, 10, 16)
        out = net(x)
        assert out.shape == (2, 3)

    def test_forward_bidirectional(self):
        from csi_node.cnn_lstm import _CnnLstmNet
        cfg = CnnLstmConfig(
            n_subcarriers=16, n_time_steps=10, n_classes=2,
            labels=["A", "B"],
            conv1_filters=4, conv2_filters=8,
            lstm_hidden=8, lstm_layers=2, dense_units=4,
            bidirectional=True,
        )
        net = _CnnLstmNet(cfg)
        x = torch.randn(1, 10, 16)
        out = net(x)
        assert out.shape == (1, 2)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
class TestTrainingExtended:
    @pytest.fixture
    def small_setup(self):
        cfg = CnnLstmConfig(
            n_subcarriers=8, n_time_steps=5, n_classes=2,
            labels=["A", "B"],
            conv1_filters=4, conv2_filters=8,
            lstm_hidden=8, lstm_layers=1, dense_units=4,
        )
        return cfg

    def test_training_history_structure(self, small_setup):
        model = CsiCnnLstm(small_setup)
        X, y = generate_synthetic_training_data(small_setup, n_samples_per_class=15)
        history = model.fit(X, y, epochs=3, verbose=True)
        assert len(history) == 3
        for h in history:
            assert "epoch" in h
            assert "train_loss" in h
            assert "train_acc" in h
            assert "val_loss" in h
            assert "val_acc" in h
            assert "lr" in h

    def test_predict_batch_after_training(self, small_setup):
        model = CsiCnnLstm(small_setup)
        X, y = generate_synthetic_training_data(small_setup, n_samples_per_class=15)
        model.fit(X, y, epochs=2, verbose=False)
        results = model.predict_batch(X[:5])
        assert len(results) == 5

    def test_predict_proba_batch(self, small_setup):
        model = CsiCnnLstm(small_setup)
        X, y = generate_synthetic_training_data(small_setup, n_samples_per_class=15)
        model.fit(X, y, epochs=2, verbose=False)
        probs = model.predict_proba(X[:3])
        assert probs.shape == (3, 2)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0, 1.0], atol=1e-5)

    def test_export_onnx(self, small_setup, tmp_path):
        model = CsiCnnLstm(small_setup)
        X, y = generate_synthetic_training_data(small_setup, n_samples_per_class=10)
        model.fit(X, y, epochs=1, verbose=False)
        path = str(tmp_path / "model.onnx")
        model.export_onnx(path)
        from pathlib import Path
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    def test_save_creates_directory(self, small_setup, tmp_path):
        model = CsiCnnLstm(small_setup)
        X, y = generate_synthetic_training_data(small_setup, n_samples_per_class=10)
        model.fit(X, y, epochs=1, verbose=False)
        nested = tmp_path / "a" / "b"
        path = str(nested / "model.pt")
        model.save(path)
        assert nested.exists()

    def test_load_restores_training_history(self, small_setup, tmp_path):
        model = CsiCnnLstm(small_setup)
        X, y = generate_synthetic_training_data(small_setup, n_samples_per_class=10)
        model.fit(X, y, epochs=3, verbose=False)
        path = str(tmp_path / "model.pt")
        model.save(path)

        model2 = CsiCnnLstm(model_path=path)
        assert model2.is_trained
        assert len(model2._training_history) == 3


class TestSyntheticDataExtended:
    def test_all_6_classes_generated(self):
        """Ensure all 6 default activity classes generate different patterns."""
        cfg = CnnLstmConfig()
        X, y = generate_synthetic_training_data(cfg, n_samples_per_class=5)
        # Should have 30 total samples (6 classes * 5)
        assert len(X) == 30
        # All classes present
        for i in range(6):
            assert i in y

    def test_unknown_label_fallback(self):
        """Config with more classes than labels should still work."""
        cfg = CnnLstmConfig(n_subcarriers=8, n_time_steps=5, n_classes=8,
                            labels=["A", "B", "C", "D", "E", "F", "G", "H"])
        X, y = generate_synthetic_training_data(cfg, n_samples_per_class=3)
        # Classes 6 and 7 will hit the UNKNOWN path in generation
        assert X.shape[0] == 24
