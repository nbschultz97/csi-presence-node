"""Extended coverage tests for onnx_runtime.py — targets uncovered lines."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from csi_node.onnx_runtime import InferenceStats, DEFAULT_LABELS


class TestInferenceStats:
    def test_to_dict_complete(self):
        stats = InferenceStats(
            n_iterations=50,
            mean_ms=1.5,
            median_ms=1.4,
            p95_ms=2.0,
            p99_ms=2.5,
            min_ms=0.8,
            max_ms=3.0,
            throughput_fps=666.7,
        )
        d = stats.to_dict()
        assert d["n_iterations"] == 50
        assert d["throughput_fps"] == 666.7
        assert len(d) == 8


class TestDefaultLabels:
    def test_default_labels_content(self):
        assert "EMPTY" in DEFAULT_LABELS
        assert "BREATHING" in DEFAULT_LABELS
        assert len(DEFAULT_LABELS) == 6


class TestOnnxEngineInputPreparation:
    """Test _prepare_input edge cases via mocked engine."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mocked ORT session."""
        with patch("csi_node.onnx_runtime.ORT_AVAILABLE", True):
            with patch("csi_node.onnx_runtime.ort") as mock_ort:
                mock_session = MagicMock()
                mock_input = MagicMock()
                mock_input.name = "input"
                mock_output = MagicMock()
                mock_output.name = "output"
                mock_session.get_inputs.return_value = [mock_input]
                mock_session.get_outputs.return_value = [mock_output]
                mock_session.get_providers.return_value = ["CPUExecutionProvider"]
                mock_ort.InferenceSession.return_value = mock_session
                mock_ort.SessionOptions.return_value = MagicMock()
                mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
                mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

                from csi_node.onnx_runtime import OnnxInferenceEngine

                with patch.object(OnnxInferenceEngine, '__init__', lambda self, *a, **kw: None):
                    engine = OnnxInferenceEngine.__new__(OnnxInferenceEngine)
                    engine.n_time_steps = 10
                    engine.n_features = 16
                    engine.n_classes = 3
                    engine.labels = ["EMPTY", "STANDING", "WALKING"]
                    engine._session = mock_session
                    engine._input_name = "input"
                    engine._output_name = "output"
                    engine._metadata = {}
                    engine.model_path = "fake.onnx"
                    return engine

    def test_prepare_4d_input(self, mock_engine):
        """4D input should be reshaped to 3D."""
        x = np.random.randn(2, 10, 2, 8).astype(np.float32)
        result = mock_engine._prepare_input(x)
        assert result.shape == (2, 10, 16)

    def test_prepare_invalid_ndim(self, mock_engine):
        """5D input should raise."""
        x = np.random.randn(1, 2, 3, 4, 5).astype(np.float32)
        with pytest.raises(ValueError, match="Expected 2-4D"):
            mock_engine._prepare_input(x)

    def test_prepare_feature_padding(self, mock_engine):
        """Input with fewer features should be padded."""
        x = np.random.randn(10, 8).astype(np.float32)
        result = mock_engine._prepare_input(x)
        assert result.shape == (1, 10, 16)
        # Last 8 features should be zero
        np.testing.assert_array_equal(result[0, :, 8:], 0)

    def test_prepare_feature_truncation(self, mock_engine):
        """Input with more features should be truncated."""
        x = np.random.randn(10, 32).astype(np.float32)
        result = mock_engine._prepare_input(x)
        assert result.shape == (1, 10, 16)

    def test_prepare_time_padding(self, mock_engine):
        """Short time series should be zero-padded at the beginning."""
        x = np.random.randn(3, 16).astype(np.float32)
        result = mock_engine._prepare_input(x)
        assert result.shape == (1, 10, 16)
        np.testing.assert_array_equal(result[0, :7, :], 0)

    def test_prepare_time_truncation(self, mock_engine):
        """Long time series takes last N steps."""
        x = np.random.randn(20, 16).astype(np.float32)
        result = mock_engine._prepare_input(x)
        assert result.shape == (1, 10, 16)

    def test_predict_returns_label_and_confidence(self, mock_engine):
        """predict() should return (str, float)."""
        logits = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
        mock_engine._session.run.return_value = [logits]

        x = np.random.randn(10, 16).astype(np.float32)
        label, conf = mock_engine.predict(x)
        assert label == "STANDING"  # index 1 has highest logit
        assert 0 < conf <= 1.0

    def test_predict_proba_batch(self, mock_engine):
        """predict_proba with batch input returns batch output."""
        logits = np.array([[1.0, 2.0, 0.5], [0.5, 0.1, 3.0]], dtype=np.float32)
        mock_engine._session.run.return_value = [logits]

        x = np.random.randn(2, 10, 16).astype(np.float32)
        probs = mock_engine.predict_proba(x)
        assert probs.shape == (2, 3)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-5)

    def test_predict_batch_method(self, mock_engine):
        """predict_batch returns list of tuples."""
        logits = np.array([[1.0, 2.0, 0.5], [3.0, 0.1, 0.2]], dtype=np.float32)
        mock_engine._session.run.return_value = [logits]

        x = np.random.randn(2, 10, 16).astype(np.float32)
        results = mock_engine.predict_batch(x)
        assert len(results) == 2
        assert results[0][0] == "STANDING"
        assert results[1][0] == "EMPTY"

    def test_predict_single_as_batch(self, mock_engine):
        """predict_batch with single sample."""
        logits = np.array([[0.1, 0.2, 5.0]], dtype=np.float32)
        mock_engine._session.run.return_value = [logits]

        x = np.random.randn(1, 10, 16).astype(np.float32)
        results = mock_engine.predict_batch(x)
        assert len(results) == 1
        assert results[0][0] == "WALKING"


class TestDetectProviders:
    def test_detect_cuda_provider(self):
        with patch("csi_node.onnx_runtime.ORT_AVAILABLE", True):
            with patch("csi_node.onnx_runtime.ort") as mock_ort:
                mock_ort.get_available_providers.return_value = [
                    "CUDAExecutionProvider", "CPUExecutionProvider"
                ]
                from csi_node.onnx_runtime import OnnxInferenceEngine
                providers = OnnxInferenceEngine._detect_providers()
                assert "CUDAExecutionProvider" in providers
                assert "CPUExecutionProvider" in providers

    def test_detect_tensorrt_provider(self):
        with patch("csi_node.onnx_runtime.ORT_AVAILABLE", True):
            with patch("csi_node.onnx_runtime.ort") as mock_ort:
                mock_ort.get_available_providers.return_value = [
                    "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"
                ]
                from csi_node.onnx_runtime import OnnxInferenceEngine
                providers = OnnxInferenceEngine._detect_providers()
                # TensorRT should be first
                assert providers[0] == "TensorrtExecutionProvider"

    def test_detect_cpu_only(self):
        with patch("csi_node.onnx_runtime.ORT_AVAILABLE", True):
            with patch("csi_node.onnx_runtime.ort") as mock_ort:
                mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
                from csi_node.onnx_runtime import OnnxInferenceEngine
                providers = OnnxInferenceEngine._detect_providers()
                assert providers == ["CPUExecutionProvider"]

    def test_detect_no_ort(self):
        with patch("csi_node.onnx_runtime.ORT_AVAILABLE", False):
            from csi_node.onnx_runtime import OnnxInferenceEngine
            providers = OnnxInferenceEngine._detect_providers()
            assert providers == ["CPUExecutionProvider"]


class TestOnnxEngineInit:
    """Test __init__ edge cases with mocked ORT."""

    def test_init_with_metadata(self, tmp_path):
        """Init loads metadata correctly."""
        import json
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps({
            "n_time_steps": 20,
            "n_features": 32,
            "labels": ["A", "B"],
        }))

        with patch("csi_node.onnx_runtime.ORT_AVAILABLE", True):
            with patch("csi_node.onnx_runtime.ort") as mock_ort:
                mock_session = MagicMock()
                mock_input = MagicMock()
                mock_input.name = "input"
                mock_output = MagicMock()
                mock_output.name = "output"
                mock_session.get_inputs.return_value = [mock_input]
                mock_session.get_outputs.return_value = [mock_output]
                mock_session.get_providers.return_value = ["CPUExecutionProvider"]
                mock_ort.InferenceSession.return_value = mock_session
                mock_ort.SessionOptions.return_value = MagicMock()
                mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
                mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

                # Create a fake model file
                model_path = tmp_path / "model.onnx"
                model_path.write_bytes(b"fake")

                from csi_node.onnx_runtime import OnnxInferenceEngine
                engine = OnnxInferenceEngine(
                    str(model_path),
                    metadata_path=str(meta_path),
                )
                assert engine.n_time_steps == 20
                assert engine.n_features == 32
                assert engine.labels == ["A", "B"]

    def test_init_custom_labels(self, tmp_path):
        """Custom labels override defaults."""
        with patch("csi_node.onnx_runtime.ORT_AVAILABLE", True):
            with patch("csi_node.onnx_runtime.ort") as mock_ort:
                mock_session = MagicMock()
                mock_input = MagicMock()
                mock_input.name = "input"
                mock_output = MagicMock()
                mock_output.name = "output"
                mock_session.get_inputs.return_value = [mock_input]
                mock_session.get_outputs.return_value = [mock_output]
                mock_session.get_providers.return_value = ["CPUExecutionProvider"]
                mock_ort.InferenceSession.return_value = mock_session
                mock_ort.SessionOptions.return_value = MagicMock()
                mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
                mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

                model_path = tmp_path / "model.onnx"
                model_path.write_bytes(b"fake")

                from csi_node.onnx_runtime import OnnxInferenceEngine
                engine = OnnxInferenceEngine(
                    str(model_path),
                    labels=["X", "Y"],
                    n_time_steps=5,
                    n_features=8,
                )
                assert engine.labels == ["X", "Y"]
                assert engine.n_classes == 2

    def test_init_custom_providers(self, tmp_path):
        """Custom providers list passed through."""
        with patch("csi_node.onnx_runtime.ORT_AVAILABLE", True):
            with patch("csi_node.onnx_runtime.ort") as mock_ort:
                mock_session = MagicMock()
                mock_input = MagicMock()
                mock_input.name = "input"
                mock_output = MagicMock()
                mock_output.name = "output"
                mock_session.get_inputs.return_value = [mock_input]
                mock_session.get_outputs.return_value = [mock_output]
                mock_session.get_providers.return_value = ["CPUExecutionProvider"]
                mock_ort.InferenceSession.return_value = mock_session
                mock_ort.SessionOptions.return_value = MagicMock()
                mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99

                model_path = tmp_path / "model.onnx"
                model_path.write_bytes(b"fake")

                from csi_node.onnx_runtime import OnnxInferenceEngine
                engine = OnnxInferenceEngine(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                )
                # Should not call _detect_providers
                mock_ort.get_available_providers.assert_not_called()

    def test_metadata_property(self, tmp_path):
        """metadata property returns copy of internal dict."""
        import json
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps({"key": "value"}))

        with patch("csi_node.onnx_runtime.ORT_AVAILABLE", True):
            with patch("csi_node.onnx_runtime.ort") as mock_ort:
                mock_session = MagicMock()
                mock_input = MagicMock()
                mock_input.name = "input"
                mock_output = MagicMock()
                mock_output.name = "output"
                mock_session.get_inputs.return_value = [mock_input]
                mock_session.get_outputs.return_value = [mock_output]
                mock_session.get_providers.return_value = ["CPUExecutionProvider"]
                mock_ort.InferenceSession.return_value = mock_session
                mock_ort.SessionOptions.return_value = MagicMock()
                mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
                mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

                model_path = tmp_path / "model.onnx"
                model_path.write_bytes(b"fake")

                from csi_node.onnx_runtime import OnnxInferenceEngine
                engine = OnnxInferenceEngine(str(model_path), metadata_path=str(meta_path))
                meta = engine.metadata
                assert meta["key"] == "value"
                # Should be a copy
                meta["key"] = "modified"
                assert engine.metadata["key"] == "value"


class TestBenchmark:
    def test_benchmark_with_mock(self):
        """Test benchmark returns correct stats structure."""
        from csi_node.onnx_runtime import OnnxInferenceEngine

        with patch.object(OnnxInferenceEngine, '__init__', lambda self, *a, **kw: None):
            engine = OnnxInferenceEngine.__new__(OnnxInferenceEngine)
            engine.n_time_steps = 10
            engine.n_features = 16
            engine._session = MagicMock()
            engine._input_name = "input"
            engine._output_name = "output"
            engine._session.run.return_value = [np.zeros((1, 3))]

            stats = engine.benchmark(n_iterations=5, batch_size=2, warmup=1)
            assert stats.n_iterations == 5
            assert stats.mean_ms >= 0
            assert stats.throughput_fps > 0
            # warmup(1) + timed(5) = 6 calls
            assert engine._session.run.call_count == 6
