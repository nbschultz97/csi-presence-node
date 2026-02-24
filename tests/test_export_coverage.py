"""Extended coverage tests for export.py — targets uncovered lines."""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

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


class TestExportResultEdgeCases:
    def test_errors_default(self):
        r = ExportResult(success=True)
        assert r.errors == []

    def test_to_dict_all_fields(self):
        r = ExportResult(
            success=True,
            onnx_path="/a/b.onnx",
            metadata_path="/a/meta.json",
            model_size_bytes=1024,
            n_parameters=5000,
            validation_passed=True,
            max_abs_diff=0.001,
            mean_abs_diff=0.0005,
            export_time_ms=150.0,
            errors=[],
        )
        d = r.to_dict()
        assert d["onnx_path"] == "/a/b.onnx"
        assert d["n_parameters"] == 5000


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
class TestExportModelInternal:
    """Test _export_model edge cases."""

    @pytest.fixture
    def small_config(self):
        from csi_node.cnn_lstm import CnnLstmConfig
        return CnnLstmConfig(
            n_subcarriers=8, n_time_steps=5, n_classes=2,
            labels=["EMPTY", "STANDING"],
            conv1_filters=4, conv2_filters=8,
            lstm_hidden=8, lstm_layers=1, dense_units=4,
        )

    @pytest.fixture
    def trained_model(self, small_config):
        from csi_node.cnn_lstm import CsiCnnLstm, generate_synthetic_training_data
        model = CsiCnnLstm(small_config)
        X, y = generate_synthetic_training_data(small_config, n_samples_per_class=10)
        model.fit(X, y, epochs=1, verbose=False)
        return model

    def test_export_without_validation(self, trained_model, tmp_path):
        pipeline = OnnxExportPipeline(trained_model.config)
        result = pipeline._export_model(trained_model, tmp_path, validate=False)
        assert result.success
        assert result.validation_passed  # Skipped = True

    def test_export_exception_handling(self, small_config, tmp_path):
        """Export with broken model should capture error."""
        from csi_node.cnn_lstm import CsiCnnLstm
        model = CsiCnnLstm(small_config)
        # Break the model's export
        with patch.object(model, 'export_onnx', side_effect=RuntimeError("boom")):
            pipeline = OnnxExportPipeline(small_config)
            result = pipeline._export_model(model, tmp_path, validate=False)
            assert not result.success
            assert "boom" in result.errors[0]
            assert result.export_time_ms >= 0

    @pytest.mark.skipif(not ORT_AVAILABLE, reason="onnxruntime required")
    def test_export_with_validation_failure(self, trained_model, tmp_path):
        """Test validation failure path."""
        pipeline = OnnxExportPipeline(trained_model.config)
        # Monkey-patch _validate_outputs to return failure
        pipeline._validate_outputs = lambda m, p, n_samples=10: {
            "passed": False,
            "max_abs_diff": 999.0,
            "mean_abs_diff": 500.0,
        }
        result = pipeline._export_model(trained_model, tmp_path, validate=True)
        assert result.success  # Export itself succeeds
        assert not result.validation_passed
        assert "Validation failed" in result.errors[0]

    def test_export_no_onnx_checker(self, trained_model, tmp_path):
        """Export succeeds even without onnx checker."""
        pipeline = OnnxExportPipeline(trained_model.config)
        with patch("csi_node.export.ONNX_AVAILABLE", False):
            result = pipeline._export_model(trained_model, tmp_path, validate=False)
            assert result.success

    @pytest.mark.skipif(not ORT_AVAILABLE, reason="onnxruntime required")
    def test_validate_outputs(self, trained_model, tmp_path):
        """_validate_outputs compares PyTorch and ORT outputs."""
        pipeline = OnnxExportPipeline(trained_model.config)
        onnx_path = tmp_path / "model.onnx"
        trained_model.export_onnx(str(onnx_path))

        result = pipeline._validate_outputs(trained_model, str(onnx_path), n_samples=3)
        assert "passed" in result
        assert "max_abs_diff" in result
        assert "mean_abs_diff" in result
        assert result["max_abs_diff"] >= 0

    def test_export_metadata_content(self, trained_model, tmp_path):
        """Verify metadata JSON has all expected fields."""
        pipeline = OnnxExportPipeline(trained_model.config)
        result = pipeline._export_model(trained_model, tmp_path, validate=False)
        assert result.metadata_path is not None
        meta = json.loads(Path(result.metadata_path).read_text())
        assert meta["model_type"] == "cnn_lstm"
        assert meta["opset_version"] == 14
        assert "config" in meta
        assert meta["labels"] == trained_model.config.labels


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
class TestExportFromCheckpoint:
    def test_roundtrip(self, tmp_path):
        from csi_node.cnn_lstm import CnnLstmConfig, CsiCnnLstm, generate_synthetic_training_data
        cfg = CnnLstmConfig(
            n_subcarriers=8, n_time_steps=5, n_classes=2,
            labels=["A", "B"],
            conv1_filters=4, conv2_filters=8,
            lstm_hidden=8, lstm_layers=1, dense_units=4,
        )
        model = CsiCnnLstm(cfg)
        X, y = generate_synthetic_training_data(cfg, n_samples_per_class=10)
        model.fit(X, y, epochs=1, verbose=False)

        ckpt = str(tmp_path / "model.pt")
        model.save(ckpt)

        out = tmp_path / "export"
        pipeline = OnnxExportPipeline()
        result = pipeline.export_from_checkpoint(ckpt, str(out), validate=False)
        assert result.success
        assert Path(result.onnx_path).exists()


class TestExportCLIExtended:
    def test_cli_train_mode(self, tmp_path):
        """CLI --train flag triggers train_and_export."""
        from csi_node.export import main
        with patch.object(sys, "argv", [
            "export", "--train", "--output", str(tmp_path), "--epochs", "1", "--no-validate"
        ]):
            if TORCH_AVAILABLE:
                main()  # Should complete without error
                assert (tmp_path / "cnn_lstm.onnx").exists()
            else:
                main()  # Should print failure result

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    def test_cli_checkpoint_mode(self, tmp_path):
        from csi_node.cnn_lstm import CnnLstmConfig, CsiCnnLstm, generate_synthetic_training_data
        cfg = CnnLstmConfig(
            n_subcarriers=8, n_time_steps=5, n_classes=2,
            labels=["A", "B"],
            conv1_filters=4, conv2_filters=8,
            lstm_hidden=8, lstm_layers=1, dense_units=4,
        )
        model = CsiCnnLstm(cfg)
        X, y = generate_synthetic_training_data(cfg, n_samples_per_class=10)
        model.fit(X, y, epochs=1, verbose=False)
        ckpt = str(tmp_path / "model.pt")
        model.save(ckpt)

        out = tmp_path / "cli_out"
        from csi_node.export import main
        with patch.object(sys, "argv", [
            "export", "--checkpoint", ckpt, "--output", str(out), "--no-validate"
        ]):
            main()
            assert (out / "cnn_lstm.onnx").exists()
