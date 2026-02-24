"""ONNX Runtime inference for CNN-LSTM models on edge devices.

Provides PyTorch-free inference using ONNX Runtime, enabling deployment
on resource-constrained edge devices (Raspberry Pi, Jetson, NUC) without
the full PyTorch dependency.

Usage:
    from csi_node.onnx_runtime import OnnxInferenceEngine

    engine = OnnxInferenceEngine("models/cnn_lstm.onnx")
    label, confidence = engine.predict(csi_window)
    probs = engine.predict_proba(csi_window)

    # Benchmark
    stats = engine.benchmark(n_iterations=100)
    print(f"Mean latency: {stats['mean_ms']:.1f}ms")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# Try importing ONNX Runtime
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ort = None  # type: ignore
    ORT_AVAILABLE = False

# Default labels (must match training config)
DEFAULT_LABELS = [
    "EMPTY", "STANDING", "WALKING", "SITTING", "PRONE", "BREATHING",
]


@dataclass
class InferenceStats:
    """Performance statistics from benchmarking."""
    n_iterations: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_fps: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "n_iterations": self.n_iterations,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "throughput_fps": self.throughput_fps,
        }


class OnnxInferenceEngine:
    """ONNX Runtime inference engine for CSI CNN-LSTM models.

    Provides lightweight, PyTorch-free inference suitable for edge
    deployment. Supports CPU and optional GPU (CUDA/TensorRT) execution.

    Args:
        model_path: Path to .onnx model file
        labels: Activity class labels (order must match training)
        metadata_path: Optional path to model metadata JSON
        providers: ONNX Runtime execution providers
                   (default: auto-detect best available)
        n_time_steps: Expected time steps (overridden by metadata if available)
        n_features: Expected features per step (overridden by metadata)
    """

    def __init__(
        self,
        model_path: str,
        labels: Optional[List[str]] = None,
        metadata_path: Optional[str] = None,
        providers: Optional[List[str]] = None,
        n_time_steps: int = 50,
        n_features: int = 64,
    ):
        if not ORT_AVAILABLE:
            raise RuntimeError(
                "onnxruntime is required for ONNX inference. "
                "Install with: pip install onnxruntime"
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        # Load metadata if available
        self._metadata: Dict[str, Any] = {}
        if metadata_path:
            meta_p = Path(metadata_path)
            if meta_p.exists():
                self._metadata = json.loads(meta_p.read_text())

        # Config from metadata or defaults
        self.labels = (
            labels
            or self._metadata.get("labels")
            or list(DEFAULT_LABELS)
        )
        self.n_time_steps = self._metadata.get("n_time_steps", n_time_steps)
        self.n_features = self._metadata.get("n_features", n_features)
        self.n_classes = len(self.labels)

        # Auto-detect providers
        if providers is None:
            providers = self._detect_providers()

        # Create session
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2  # Edge device friendly

        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=providers
        )

        # Cache input/output names
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        active_providers = self._session.get_providers()
        logger.info(
            f"ONNX engine loaded: {model_path} "
            f"(providers: {active_providers}, "
            f"input: [{self.n_time_steps}, {self.n_features}], "
            f"classes: {self.n_classes})"
        )

    @staticmethod
    def _detect_providers() -> List[str]:
        """Detect best available execution providers."""
        available = ort.get_available_providers() if ORT_AVAILABLE else []
        providers = []
        # Prefer GPU if available
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "TensorrtExecutionProvider" in available:
            providers.insert(0, "TensorrtExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        """Normalize input to (batch, time_steps, features) float32."""
        if x.ndim == 2:
            x = x[np.newaxis, :, :]
        elif x.ndim == 4:
            b, t, c, s = x.shape
            x = x.reshape(b, t, c * s)
        elif x.ndim != 3:
            raise ValueError(f"Expected 2-4D input, got {x.ndim}D shape {x.shape}")

        _, t, f = x.shape

        # Pad/truncate time steps
        if t > self.n_time_steps:
            x = x[:, -self.n_time_steps:, :]
        elif t < self.n_time_steps:
            pad = np.zeros((x.shape[0], self.n_time_steps - t, f), dtype=np.float32)
            x = np.concatenate([pad, x], axis=1)

        # Pad/truncate features
        if f > self.n_features:
            x = x[:, :, :self.n_features]
        elif f < self.n_features:
            pad = np.zeros((x.shape[0], x.shape[1], self.n_features - f), dtype=np.float32)
            x = np.concatenate([x, pad], axis=2)

        return x.astype(np.float32)

    def predict(self, x: np.ndarray) -> Tuple[str, float]:
        """Predict activity class for a CSI window.

        Args:
            x: CSI amplitude data, shape (time_steps, features) or
               (batch, time_steps, features)

        Returns:
            Tuple of (label, confidence)
        """
        probs = self.predict_proba(x)
        if probs.ndim > 1:
            probs = probs[0]
        idx = int(np.argmax(probs))
        label = self.labels[idx] if idx < len(self.labels) else "UNKNOWN"
        return label, float(probs[idx])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Get class probabilities for a CSI window.

        Args:
            x: CSI amplitude data

        Returns:
            Probability array of shape (n_classes,) or (batch, n_classes)
        """
        x = self._prepare_input(x)
        logits = self._session.run(
            [self._output_name], {self._input_name: x}
        )[0]

        # Softmax
        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp / np.sum(exp, axis=-1, keepdims=True)

        if probs.shape[0] == 1:
            return probs[0]
        return probs

    def predict_batch(self, X: np.ndarray) -> List[Tuple[str, float]]:
        """Predict activity for a batch of CSI windows."""
        probs = self.predict_proba(X)
        if probs.ndim == 1:
            probs = probs[np.newaxis, :]
        results = []
        for p in probs:
            idx = int(np.argmax(p))
            label = self.labels[idx] if idx < len(self.labels) else "UNKNOWN"
            results.append((label, float(p[idx])))
        return results

    def benchmark(
        self,
        n_iterations: int = 100,
        batch_size: int = 1,
        warmup: int = 10,
    ) -> InferenceStats:
        """Benchmark inference latency.

        Args:
            n_iterations: Number of timed iterations
            batch_size: Batch size for each iteration
            warmup: Warmup iterations (not timed)

        Returns:
            InferenceStats with latency metrics
        """
        dummy = np.random.randn(
            batch_size, self.n_time_steps, self.n_features
        ).astype(np.float32)

        # Warmup
        for _ in range(warmup):
            self._session.run([self._output_name], {self._input_name: dummy})

        # Timed runs
        latencies = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            self._session.run([self._output_name], {self._input_name: dummy})
            latencies.append((time.perf_counter() - start) * 1000)

        arr = np.array(latencies)
        return InferenceStats(
            n_iterations=n_iterations,
            mean_ms=float(np.mean(arr)),
            median_ms=float(np.median(arr)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            throughput_fps=1000.0 / float(np.mean(arr)) * batch_size,
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model metadata."""
        return dict(self._metadata)

    @property
    def providers(self) -> List[str]:
        """Active execution providers."""
        return self._session.get_providers()
