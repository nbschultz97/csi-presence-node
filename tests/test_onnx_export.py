"""Tests for ONNX export pipeline and runtime inference engine."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from csi_node.cnn_lstm import CnnLstmConfig, CsiCnnLstm, generate_synthetic_training_data
from csi_node.export import OnnxExportPipeline, ExportResult

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

try:
    import onnx as onnx_lib
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


# ── ExportResult tests ───────────────────────────────────────────────


class TestExportResult:
    def test_defaults(self):
        r = ExportResult(success=True)
        assert r.success
        assert r.errors == []
        assert r.model_size_bytes == 0

    def test_to_dict(self):
        r = ExportResult(success=False, errors=["boom"])
        d = r.to_dict()
        assert d["success"] is False
        assert d["errors"] == ["boom"]

    def test_post_init_none_errors(self):
        r = ExportResult(success=True, errors=None)
        assert r.errors == []


# ── OnnxExportPipeline tests (no torch/onnx required) ────────────────


class TestExportPipelineNoTorch:
    @patch("csi_node.export.TORCH_AVAILABLE", False)
    def test_train_and_export_no_torch(self, tmp_path):
        pipeline = OnnxExportPipeline()
        result = pipeline.train_and_export(str(tmp_path))
        assert not result.success
        assert "PyTorch" in result.errors[0]

    @patch("csi_node.export.TORCH_AVAILABLE", False)
    def test_export_from_checkpoint_no_torch(self, tmp_path):
        pipeline = OnnxExportPipeline()
        result = pipeline.export_from_checkpoint("fake.pt", str(tmp_path))
        assert not result.success


# ── OnnxExportPipeline with torch ────────────────────────────────────


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestExportPipelineWithTorch:
    @pytest.fixture
    def small_config(self):
        return CnnLstmConfig(
            n_subcarriers=16,
            n_time_steps=10,
            n_classes=3,
            labels=["EMPTY", "STANDING", "WALKING"],
            conv1_filters=8,
            conv2_filters=16,
            lstm_hidden=16,
            lstm_layers=1,
            dense_units=8,
        )

    @pytest.fixture
    def trained_checkpoint(self, small_config, tmp_path):
        model = CsiCnnLstm(small_config)
        X, y = generate_synthetic_training_data(small_config, n_samples_per_class=20)
        model.fit(X, y, epochs=2, verbose=False)
        path = str(tmp_path / "test_model.pt")
        model.save(path)
        return path, model

    def test_train_and_export(self, small_config, tmp_path):
        pipeline = OnnxExportPipeline(small_config)
        result = pipeline.train_and_export(
            str(tmp_path), epochs=2, n_samples_per_class=20, validate=False
        )
        assert result.success
        assert result.onnx_path is not None
        assert Path(result.onnx_path).exists()
        assert result.metadata_path is not None
        assert result.model_size_bytes > 0
        assert result.n_parameters > 0

        # Check metadata
        meta = json.loads(Path(result.metadata_path).read_text())
        assert meta["model_type"] == "cnn_lstm"
        assert meta["n_classes"] == 3
        assert meta["labels"] == ["EMPTY", "STANDING", "WALKING"]

    def test_export_from_checkpoint(self, trained_checkpoint, tmp_path):
        ckpt_path, _ = trained_checkpoint
        out = tmp_path / "export_out"
        pipeline = OnnxExportPipeline()
        result = pipeline.export_from_checkpoint(
            ckpt_path, str(out), validate=False
        )
        assert result.success
        assert Path(result.onnx_path).exists()

    @pytest.mark.skipif(not ORT_AVAILABLE, reason="onnxruntime not installed")
    def test_export_with_validation(self, small_config, tmp_path):
        pipeline = OnnxExportPipeline(small_config)
        result = pipeline.train_and_export(
            str(tmp_path), epochs=2, n_samples_per_class=20, validate=True
        )
        assert result.success
        assert result.validation_passed
        assert result.max_abs_diff < 0.01  # Should be very close

    def test_export_creates_output_dir(self, small_config, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        pipeline = OnnxExportPipeline(small_config)
        result = pipeline.train_and_export(
            str(nested), epochs=1, n_samples_per_class=10, validate=False
        )
        assert result.success
        assert nested.exists()


# ── ONNX Runtime inference engine tests ──────────────────────────────


@pytest.mark.skipif(
    not (TORCH_AVAILABLE and ORT_AVAILABLE),
    reason="PyTorch + onnxruntime required",
)
class TestOnnxInferenceEngine:
    @pytest.fixture
    def onnx_model(self, tmp_path):
        """Train a small model and export to ONNX."""
        cfg = CnnLstmConfig(
            n_subcarriers=16,
            n_time_steps=10,
            n_classes=3,
            labels=["EMPTY", "STANDING", "WALKING"],
            conv1_filters=8,
            conv2_filters=16,
            lstm_hidden=16,
            lstm_layers=1,
            dense_units=8,
        )
        model = CsiCnnLstm(cfg)
        X, y = generate_synthetic_training_data(cfg, n_samples_per_class=20)
        model.fit(X, y, epochs=2, verbose=False)

        onnx_path = str(tmp_path / "model.onnx")
        meta_path = str(tmp_path / "metadata.json")

        model.export_onnx(onnx_path)
        meta = {
            "n_time_steps": cfg.n_time_steps,
            "n_features": cfg.input_features,
            "labels": list(cfg.labels),
        }
        Path(meta_path).write_text(json.dumps(meta))
        return onnx_path, meta_path, cfg

    def test_load_and_predict(self, onnx_model):
        from csi_node.onnx_runtime import OnnxInferenceEngine

        onnx_path, meta_path, cfg = onnx_model
        engine = OnnxInferenceEngine(onnx_path, metadata_path=meta_path)

        x = np.random.randn(cfg.n_time_steps, cfg.input_features).astype(np.float32)
        label, confidence = engine.predict(x)
        assert label in cfg.labels
        assert 0.0 <= confidence <= 1.0

    def test_predict_proba(self, onnx_model):
        from csi_node.onnx_runtime import OnnxInferenceEngine

        onnx_path, meta_path, cfg = onnx_model
        engine = OnnxInferenceEngine(onnx_path, metadata_path=meta_path)

        x = np.random.randn(cfg.n_time_steps, cfg.input_features).astype(np.float32)
        probs = engine.predict_proba(x)
        assert probs.shape == (cfg.n_classes,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_predict_batch(self, onnx_model):
        from csi_node.onnx_runtime import OnnxInferenceEngine

        onnx_path, meta_path, cfg = onnx_model
        engine = OnnxInferenceEngine(onnx_path, metadata_path=meta_path)

        X = np.random.randn(5, cfg.n_time_steps, cfg.input_features).astype(np.float32)
        results = engine.predict_batch(X)
        assert len(results) == 5
        for label, conf in results:
            assert label in cfg.labels

    def test_benchmark(self, onnx_model):
        from csi_node.onnx_runtime import OnnxInferenceEngine

        onnx_path, meta_path, cfg = onnx_model
        engine = OnnxInferenceEngine(onnx_path, metadata_path=meta_path)

        stats = engine.benchmark(n_iterations=10, warmup=2)
        assert stats.n_iterations == 10
        assert stats.mean_ms > 0
        assert stats.throughput_fps > 0
        d = stats.to_dict()
        assert "mean_ms" in d

    def test_input_padding(self, onnx_model):
        from csi_node.onnx_runtime import OnnxInferenceEngine

        onnx_path, meta_path, cfg = onnx_model
        engine = OnnxInferenceEngine(onnx_path, metadata_path=meta_path)

        # Shorter input — should be padded
        x = np.random.randn(5, cfg.input_features).astype(np.float32)
        label, conf = engine.predict(x)
        assert label in cfg.labels

    def test_input_truncation(self, onnx_model):
        from csi_node.onnx_runtime import OnnxInferenceEngine

        onnx_path, meta_path, cfg = onnx_model
        engine = OnnxInferenceEngine(onnx_path, metadata_path=meta_path)

        # Longer input — should be truncated
        x = np.random.randn(cfg.n_time_steps * 3, cfg.input_features).astype(np.float32)
        label, conf = engine.predict(x)
        assert label in cfg.labels

    def test_missing_model_file(self):
        from csi_node.onnx_runtime import OnnxInferenceEngine

        with pytest.raises(FileNotFoundError):
            OnnxInferenceEngine("nonexistent.onnx")

    def test_no_metadata(self, onnx_model):
        from csi_node.onnx_runtime import OnnxInferenceEngine

        onnx_path, _, cfg = onnx_model
        engine = OnnxInferenceEngine(
            onnx_path,
            labels=list(cfg.labels),
            n_time_steps=cfg.n_time_steps,
            n_features=cfg.input_features,
        )
        x = np.random.randn(cfg.n_time_steps, cfg.input_features).astype(np.float32)
        label, conf = engine.predict(x)
        assert label in cfg.labels

    def test_providers_property(self, onnx_model):
        from csi_node.onnx_runtime import OnnxInferenceEngine

        onnx_path, meta_path, _ = onnx_model
        engine = OnnxInferenceEngine(onnx_path, metadata_path=meta_path)
        assert "CPUExecutionProvider" in engine.providers

    def test_metadata_property(self, onnx_model):
        from csi_node.onnx_runtime import OnnxInferenceEngine

        onnx_path, meta_path, _ = onnx_model
        engine = OnnxInferenceEngine(onnx_path, metadata_path=meta_path)
        meta = engine.metadata
        assert "n_time_steps" in meta


# ── OnnxInferenceEngine without onnxruntime ──────────────────────────


class TestOnnxEngineNoRuntime:
    @patch("csi_node.onnx_runtime.ORT_AVAILABLE", False)
    def test_raises_without_onnxruntime(self):
        from csi_node.onnx_runtime import OnnxInferenceEngine

        with pytest.raises(RuntimeError, match="onnxruntime"):
            OnnxInferenceEngine("dummy.onnx")


# ── CLI entry point test ─────────────────────────────────────────────


class TestExportCLI:
    def test_no_args_error(self):
        """CLI with no args should fail."""
        from csi_node.export import main
        import sys

        with patch.object(sys, "argv", ["export"]):
            with pytest.raises(SystemExit):
                main()
