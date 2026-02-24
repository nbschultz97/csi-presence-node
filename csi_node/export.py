"""ONNX export pipeline for Vantage CNN-LSTM models.

Complete pipeline: train (or load) → export ONNX → validate → package
for edge deployment. Produces a deployment bundle with model, metadata,
and optional quantized variants.

Usage:
    from csi_node.export import OnnxExportPipeline

    pipeline = OnnxExportPipeline()
    result = pipeline.export_from_checkpoint("models/cnn_lstm.pt", "deploy/")
    # or
    result = pipeline.train_and_export("deploy/", epochs=30)

CLI:
    python -m csi_node.export --checkpoint models/cnn_lstm.pt --output deploy/
    python -m csi_node.export --train --output deploy/ --epochs 30
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from csi_node.cnn_lstm import CsiCnnLstm, CnnLstmConfig, generate_synthetic_training_data

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    onnx = None  # type: ignore
    ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ort = None  # type: ignore
    ORT_AVAILABLE = False


@dataclass
class ExportResult:
    """Result of an ONNX export operation."""
    success: bool
    onnx_path: Optional[str] = None
    metadata_path: Optional[str] = None
    model_size_bytes: int = 0
    n_parameters: int = 0
    validation_passed: bool = False
    max_abs_diff: float = 0.0
    mean_abs_diff: float = 0.0
    export_time_ms: float = 0.0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OnnxExportPipeline:
    """Pipeline for exporting CNN-LSTM models to ONNX format.

    Handles the complete workflow: load/train → export → validate →
    package with metadata for edge deployment.
    """

    # Tolerance for numerical validation (PyTorch vs ONNX Runtime output)
    VALIDATION_ATOL = 1e-4
    VALIDATION_RTOL = 1e-3

    def __init__(self, config: Optional[CnnLstmConfig] = None):
        self.config = config or CnnLstmConfig()

    def train_and_export(
        self,
        output_dir: str,
        epochs: int = 30,
        n_samples_per_class: int = 200,
        validate: bool = True,
    ) -> ExportResult:
        """Train on synthetic data and export to ONNX.

        Args:
            output_dir: Directory for output files
            epochs: Training epochs
            n_samples_per_class: Synthetic samples per class
            validate: Run ONNX vs PyTorch validation

        Returns:
            ExportResult with paths and validation info
        """
        if not TORCH_AVAILABLE:
            return ExportResult(
                success=False,
                errors=["PyTorch required for training"],
            )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Train
        logger.info("Training CNN-LSTM on synthetic data...")
        model = CsiCnnLstm(self.config)
        X, y = generate_synthetic_training_data(
            self.config, n_samples_per_class=n_samples_per_class
        )
        model.fit(X, y, epochs=epochs, verbose=False)

        # Save checkpoint
        checkpoint_path = str(out / "cnn_lstm.pt")
        model.save(checkpoint_path)

        # Export
        return self._export_model(model, out, validate=validate)

    def export_from_checkpoint(
        self,
        checkpoint_path: str,
        output_dir: str,
        validate: bool = True,
    ) -> ExportResult:
        """Export a pre-trained checkpoint to ONNX.

        Args:
            checkpoint_path: Path to .pt checkpoint
            output_dir: Directory for output files
            validate: Run ONNX vs PyTorch validation

        Returns:
            ExportResult
        """
        if not TORCH_AVAILABLE:
            return ExportResult(
                success=False,
                errors=["PyTorch required to load checkpoint"],
            )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        model = CsiCnnLstm(model_path=checkpoint_path)
        return self._export_model(model, out, validate=validate)

    def _export_model(
        self,
        model: CsiCnnLstm,
        output_dir: Path,
        validate: bool = True,
    ) -> ExportResult:
        """Core export logic."""
        result = ExportResult(success=False)
        onnx_path = output_dir / "cnn_lstm.onnx"
        meta_path = output_dir / "model_metadata.json"

        start = time.perf_counter()

        try:
            # Export ONNX
            model.export_onnx(str(onnx_path))
            result.onnx_path = str(onnx_path)
            result.model_size_bytes = onnx_path.stat().st_size
            result.n_parameters = model.parameter_count

            # Write metadata
            metadata = {
                "model_type": "cnn_lstm",
                "n_time_steps": model.config.n_time_steps,
                "n_features": model.config.input_features,
                "n_classes": model.config.n_classes,
                "labels": list(model.config.labels),
                "config": model.config.to_dict(),
                "model_size_bytes": result.model_size_bytes,
                "n_parameters": result.n_parameters,
                "opset_version": 14,
            }
            meta_path.write_text(json.dumps(metadata, indent=2))
            result.metadata_path = str(meta_path)

            # Validate ONNX model structure
            if ONNX_AVAILABLE:
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                logger.info("ONNX model structure validation passed")

            # Validate numerical accuracy against PyTorch
            if validate and ORT_AVAILABLE and TORCH_AVAILABLE:
                val_result = self._validate_outputs(model, str(onnx_path))
                result.validation_passed = val_result["passed"]
                result.max_abs_diff = val_result["max_abs_diff"]
                result.mean_abs_diff = val_result["mean_abs_diff"]

                if not result.validation_passed:
                    result.errors.append(
                        f"Validation failed: max diff {result.max_abs_diff:.6f} "
                        f"> tolerance {self.VALIDATION_ATOL}"
                    )
            elif not validate:
                result.validation_passed = True  # Skipped

            result.success = True
            result.export_time_ms = (time.perf_counter() - start) * 1000

        except Exception as e:
            result.errors.append(str(e))
            result.export_time_ms = (time.perf_counter() - start) * 1000
            logger.error(f"Export failed: {e}")

        return result

    def _validate_outputs(
        self,
        model: CsiCnnLstm,
        onnx_path: str,
        n_samples: int = 10,
    ) -> Dict[str, Any]:
        """Compare PyTorch and ONNX Runtime outputs."""
        import torch as _torch

        session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        max_diff = 0.0
        diffs = []

        model._net.eval()
        for _ in range(n_samples):
            x_np = np.random.randn(
                1, model.config.n_time_steps, model.config.input_features
            ).astype(np.float32)

            # PyTorch output
            with _torch.no_grad():
                x_t = _torch.FloatTensor(x_np).to(model._device)
                pt_out = model._net(x_t).cpu().numpy()

            # ONNX Runtime output
            ort_out = session.run([output_name], {input_name: x_np})[0]

            diff = np.abs(pt_out - ort_out)
            max_diff = max(max_diff, float(np.max(diff)))
            diffs.append(float(np.mean(diff)))

        mean_diff = float(np.mean(diffs))
        passed = max_diff < self.VALIDATION_ATOL or np.allclose(
            pt_out, ort_out, atol=self.VALIDATION_ATOL, rtol=self.VALIDATION_RTOL
        )

        logger.info(
            f"Validation: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
            f"passed={passed}"
        )

        return {
            "passed": passed,
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
        }


def main():
    """CLI entry point for ONNX export."""
    import argparse

    parser = argparse.ArgumentParser(description="Export CNN-LSTM to ONNX")
    parser.add_argument("--checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("--train", action="store_true", help="Train on synthetic data first")
    parser.add_argument("--output", default="models/onnx", help="Output directory")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    pipeline = OnnxExportPipeline()

    if args.train:
        result = pipeline.train_and_export(
            args.output, epochs=args.epochs, validate=not args.no_validate
        )
    elif args.checkpoint:
        result = pipeline.export_from_checkpoint(
            args.checkpoint, args.output, validate=not args.no_validate
        )
    else:
        parser.error("Specify --checkpoint or --train")

    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
