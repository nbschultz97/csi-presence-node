"""CNN-LSTM deep learning model for WiFi CSI activity recognition.

Implements the CNN-LSTM architecture described in the Vantage patent for
human activity recognition and presence detection from CSI amplitude data.

Architecture:
    Input: (batch, time_steps, subcarriers) — CSI amplitude windows
    → Conv1D layers: local spatial-frequency feature extraction
    → LSTM layers: temporal sequence modeling
    → Dense classifier: activity/pose prediction

Supports two backends:
    1. PyTorch (preferred): Full training and inference
    2. NumPy (fallback): Inference-only with pre-trained weights

Usage:
    from csi_node.cnn_lstm import CsiCnnLstm, CnnLstmConfig

    config = CnnLstmConfig(n_subcarriers=64, n_classes=5)
    model = CsiCnnLstm(config)

    # Training (PyTorch backend)
    model.fit(X_train, y_train, epochs=50)

    # Inference
    label, confidence = model.predict(csi_window)

    # Export
    model.save("models/cnn_lstm.pt")
    model.export_onnx("models/cnn_lstm.onnx")
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Activity labels (superset — classifier may use a subset)
ACTIVITY_LABELS = [
    "EMPTY",       # No one present
    "STANDING",    # Stationary upright
    "WALKING",     # Active movement
    "SITTING",     # Seated / crouching
    "PRONE",       # Lying down
    "BREATHING",   # Stationary with micro-movement (breathing detection)
]


@dataclass
class CnnLstmConfig:
    """Configuration for the CNN-LSTM model."""

    # Input dimensions
    n_subcarriers: int = 64       # Number of CSI subcarriers per sample
    n_time_steps: int = 50        # Sliding window length (samples)
    n_chains: int = 1             # Number of antenna chains (flattened with subcarriers)

    # CNN layers
    conv1_filters: int = 64
    conv1_kernel: int = 5
    conv2_filters: int = 128
    conv2_kernel: int = 3
    pool_size: int = 2
    dropout_cnn: float = 0.3

    # LSTM layers
    lstm_hidden: int = 128
    lstm_layers: int = 2
    dropout_lstm: float = 0.3
    bidirectional: bool = False

    # Classifier head
    dense_units: int = 64
    n_classes: int = 6           # Number of activity classes
    dropout_dense: float = 0.5

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-4

    # Labels
    labels: List[str] = field(default_factory=lambda: list(ACTIVITY_LABELS))

    @property
    def input_features(self) -> int:
        """Total input features per time step."""
        return self.n_subcarriers * self.n_chains

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CnnLstmConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "CnnLstmConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


if TORCH_AVAILABLE:
    class _CnnLstmNet(nn.Module):
        """PyTorch CNN-LSTM network for CSI activity recognition."""

        def __init__(self, config: CnnLstmConfig):
            super().__init__()
            self.config = config

            # CNN feature extractor (operates on each time step)
            # Input: (batch, 1, subcarriers) per time step
            self.cnn = nn.Sequential(
                # Conv block 1
                nn.Conv1d(1, config.conv1_filters, config.conv1_kernel, padding="same"),
                nn.BatchNorm1d(config.conv1_filters),
                nn.ReLU(),
                nn.MaxPool1d(config.pool_size),
                nn.Dropout(config.dropout_cnn),
                # Conv block 2
                nn.Conv1d(config.conv1_filters, config.conv2_filters, config.conv2_kernel, padding="same"),
                nn.BatchNorm1d(config.conv2_filters),
                nn.ReLU(),
                nn.MaxPool1d(config.pool_size),
                nn.Dropout(config.dropout_cnn),
            )

            # Calculate CNN output size
            cnn_out_size = config.input_features // (config.pool_size ** 2)
            self._cnn_flat_size = config.conv2_filters * cnn_out_size

            # LSTM temporal modeling
            self.lstm = nn.LSTM(
                input_size=self._cnn_flat_size,
                hidden_size=config.lstm_hidden,
                num_layers=config.lstm_layers,
                batch_first=True,
                dropout=config.dropout_lstm if config.lstm_layers > 1 else 0.0,
                bidirectional=config.bidirectional,
            )

            lstm_out = config.lstm_hidden * (2 if config.bidirectional else 1)

            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(lstm_out, config.dense_units),
                nn.ReLU(),
                nn.Dropout(config.dropout_dense),
                nn.Linear(config.dense_units, config.n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Input tensor of shape (batch, time_steps, features)

            Returns:
                Logits of shape (batch, n_classes)
            """
            batch_size, time_steps, features = x.shape

            # Apply CNN to each time step
            # Reshape: (batch * time_steps, 1, features)
            x_cnn = x.reshape(batch_size * time_steps, 1, features)
            cnn_out = self.cnn(x_cnn)

            # Flatten CNN output per time step
            cnn_flat = cnn_out.reshape(batch_size, time_steps, -1)

            # LSTM over time
            lstm_out, _ = self.lstm(cnn_flat)

            # Take last time step output
            last_hidden = lstm_out[:, -1, :]

            # Classify
            logits = self.classifier(last_hidden)
            return logits


class CsiCnnLstm:
    """High-level CNN-LSTM model for CSI activity recognition.

    Wraps the PyTorch network with training, inference, save/load, and
    ONNX export capabilities. Falls back to a simple NumPy-based inference
    mode when PyTorch is not available.
    """

    def __init__(
        self,
        config: Optional[CnnLstmConfig] = None,
        model_path: Optional[str] = None,
    ):
        """Initialize the CNN-LSTM model.

        Args:
            config: Model configuration. If None, uses defaults.
            model_path: Path to load pre-trained weights from.
        """
        self.config = config or CnnLstmConfig()
        self._net = None
        self._device = "cpu"
        self._trained = False
        self._training_history: List[Dict[str, float]] = []

        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self._device = "cuda"
            self._net = _CnnLstmNet(self.config).to(self._device)
            logger.info(
                f"CNN-LSTM initialized on {self._device} "
                f"({sum(p.numel() for p in self._net.parameters())} parameters)"
            )
        else:
            logger.warning(
                "PyTorch not available — CNN-LSTM in inference-only mode "
                "(requires pre-exported weights)"
            )

        if model_path:
            self.load(model_path)

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained or loaded."""
        return self._trained

    @property
    def parameter_count(self) -> int:
        """Total trainable parameters."""
        if self._net is not None and TORCH_AVAILABLE:
            return sum(p.numel() for p in self._net.parameters())
        return 0

    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        """Normalize input shape to (batch, time_steps, features).

        Args:
            x: Input array of shape:
                - (time_steps, features): single sample
                - (batch, time_steps, features): batch
                - (time_steps, chains, subcarriers): raw CSI

        Returns:
            Array of shape (batch, time_steps, features)
        """
        if x.ndim == 2:
            # Single sample: (time_steps, features)
            x = x[np.newaxis, :, :]
        elif x.ndim == 4:
            # (batch, time_steps, chains, subcarriers) → flatten last two
            b, t, c, s = x.shape
            x = x.reshape(b, t, c * s)
        elif x.ndim != 3:
            raise ValueError(
                f"Expected 2D, 3D, or 4D input, got {x.ndim}D with shape {x.shape}"
            )

        # Truncate or pad time steps
        _, t, f = x.shape
        target_t = self.config.n_time_steps
        target_f = self.config.input_features

        if t > target_t:
            x = x[:, -target_t:, :]  # Take last N steps
        elif t < target_t:
            pad = np.zeros((x.shape[0], target_t - t, f))
            x = np.concatenate([pad, x], axis=1)

        if f > target_f:
            x = x[:, :, :target_f]
        elif f < target_f:
            pad = np.zeros((x.shape[0], x.shape[1], target_f - f))
            x = np.concatenate([x, pad], axis=2)

        return x

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        validation_split: float = 0.2,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """Train the model on labeled CSI data.

        Args:
            X: Training data of shape (n_samples, time_steps, features)
            y: Labels of shape (n_samples,) — integer class indices
            epochs: Number of training epochs
            validation_split: Fraction of data for validation
            verbose: Print training progress

        Returns:
            Training history as list of dicts with loss/accuracy per epoch

        Raises:
            RuntimeError: If PyTorch is not available
        """
        if not TORCH_AVAILABLE or self._net is None:
            raise RuntimeError("PyTorch required for training")

        X = self._prepare_input(X)

        # Split train/val
        n = len(X)
        n_val = max(1, int(n * validation_split))
        indices = np.random.permutation(n)
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train = torch.FloatTensor(X[train_idx]).to(self._device)
        y_train = torch.LongTensor(y[train_idx]).to(self._device)
        X_val = torch.FloatTensor(X[val_idx]).to(self._device)
        y_val = torch.LongTensor(y[val_idx]).to(self._device)

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True
        )

        optimizer = optim.Adam(
            self._net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        criterion = nn.CrossEntropyLoss()

        self._training_history = []

        for epoch in range(epochs):
            # Training
            self._net.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = self._net(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * len(y_batch)
                train_correct += (logits.argmax(1) == y_batch).sum().item()
                train_total += len(y_batch)

            # Validation
            self._net.eval()
            with torch.no_grad():
                val_logits = self._net(X_val)
                val_loss = criterion(val_logits, y_val).item()
                val_correct = (val_logits.argmax(1) == y_val).sum().item()

            train_loss /= train_total
            train_acc = train_correct / train_total
            val_acc = val_correct / len(y_val)

            scheduler.step(val_loss)

            epoch_stats = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
            self._training_history.append(epoch_stats)

            if verbose and (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} — "
                    f"loss: {train_loss:.4f}, acc: {train_acc:.3f}, "
                    f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.3f}"
                )

        self._trained = True
        logger.info(
            f"Training complete — final val_acc: {val_acc:.3f} "
            f"({self.parameter_count} parameters)"
        )
        return self._training_history

    def predict(self, x: np.ndarray) -> Tuple[str, float]:
        """Predict activity class for a CSI window.

        Args:
            x: CSI amplitude window of shape (time_steps, features) or
               (batch, time_steps, features)

        Returns:
            Tuple of (label_string, confidence)
        """
        probs = self.predict_proba(x)
        if probs.ndim == 1:
            idx = int(np.argmax(probs))
            label = self.config.labels[idx] if idx < len(self.config.labels) else "UNKNOWN"
            return label, float(probs[idx])
        else:
            # Batch — return first sample
            idx = int(np.argmax(probs[0]))
            label = self.config.labels[idx] if idx < len(self.config.labels) else "UNKNOWN"
            return label, float(probs[0, idx])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Get class probabilities for a CSI window.

        Args:
            x: CSI amplitude window

        Returns:
            Probability array of shape (n_classes,) or (batch, n_classes)
        """
        x = self._prepare_input(x)

        if not TORCH_AVAILABLE or self._net is None:
            # Fallback: uniform distribution
            n = self.config.n_classes
            batch_size = x.shape[0]
            uniform = np.ones(n) / n
            if batch_size == 1:
                return uniform
            return np.tile(uniform, (batch_size, 1))
        self._net.eval()

        with torch.no_grad():
            x_t = torch.FloatTensor(x).to(self._device)
            logits = self._net(x_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        if probs.shape[0] == 1:
            return probs[0]
        return probs

    def predict_batch(self, X: np.ndarray) -> List[Tuple[str, float]]:
        """Predict activity for a batch of CSI windows.

        Args:
            X: Batch of shape (batch, time_steps, features)

        Returns:
            List of (label, confidence) tuples
        """
        probs = self.predict_proba(X)
        if probs.ndim == 1:
            probs = probs[np.newaxis, :]

        results = []
        for p in probs:
            idx = int(np.argmax(p))
            label = self.config.labels[idx] if idx < len(self.config.labels) else "UNKNOWN"
            results.append((label, float(p[idx])))
        return results

    def save(self, path: str) -> None:
        """Save model weights and config.

        Args:
            path: Output path (.pt file)
        """
        if not TORCH_AVAILABLE or self._net is None:
            raise RuntimeError("PyTorch required to save model")

        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config": self.config.to_dict(),
            "state_dict": self._net.state_dict(),
            "trained": self._trained,
            "training_history": self._training_history,
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model weights from checkpoint.

        Args:
            path: Path to .pt checkpoint file
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available — cannot load .pt model")
            return

        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        self.config = CnnLstmConfig.from_dict(checkpoint["config"])
        self._net = _CnnLstmNet(self.config).to(self._device)
        self._net.load_state_dict(checkpoint["state_dict"])
        self._trained = checkpoint.get("trained", True)
        self._training_history = checkpoint.get("training_history", [])
        logger.info(f"Model loaded from {path} ({self.parameter_count} parameters)")

    def export_onnx(self, path: str) -> None:
        """Export model to ONNX format for edge deployment.

        Args:
            path: Output .onnx file path
        """
        if not TORCH_AVAILABLE or self._net is None:
            raise RuntimeError("PyTorch required for ONNX export")

        self._net.eval()
        dummy = torch.randn(
            1, self.config.n_time_steps, self.config.input_features
        ).to(self._device)

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            self._net,
            dummy,
            path,
            input_names=["csi_input"],
            output_names=["activity_logits"],
            dynamic_axes={
                "csi_input": {0: "batch_size"},
                "activity_logits": {0: "batch_size"},
            },
            opset_version=14,
        )
        logger.info(f"ONNX model exported to {path}")

    def summary(self) -> str:
        """Get a human-readable model summary."""
        lines = [
            f"CNN-LSTM Model Summary",
            f"{'='*50}",
            f"Backend:        {'PyTorch' if TORCH_AVAILABLE else 'NumPy (limited)'}",
            f"Device:         {self._device}",
            f"Trained:        {self._trained}",
            f"Parameters:     {self.parameter_count:,}",
            f"",
            f"Input Shape:    ({self.config.n_time_steps}, {self.config.input_features})",
            f"  Time steps:   {self.config.n_time_steps}",
            f"  Subcarriers:  {self.config.n_subcarriers}",
            f"  Chains:       {self.config.n_chains}",
            f"",
            f"CNN:",
            f"  Conv1:        {self.config.conv1_filters} filters, kernel={self.config.conv1_kernel}",
            f"  Conv2:        {self.config.conv2_filters} filters, kernel={self.config.conv2_kernel}",
            f"  Pool:         {self.config.pool_size}",
            f"  Dropout:      {self.config.dropout_cnn}",
            f"",
            f"LSTM:",
            f"  Hidden:       {self.config.lstm_hidden}",
            f"  Layers:       {self.config.lstm_layers}",
            f"  Bidirectional:{self.config.bidirectional}",
            f"  Dropout:      {self.config.dropout_lstm}",
            f"",
            f"Classifier:",
            f"  Dense:        {self.config.dense_units}",
            f"  Classes:      {self.config.n_classes} ({', '.join(self.config.labels)})",
            f"  Dropout:      {self.config.dropout_dense}",
        ]
        return "\n".join(lines)


def generate_synthetic_training_data(
    config: CnnLstmConfig,
    n_samples_per_class: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic CSI training data for each activity class.

    Creates realistic-ish CSI amplitude patterns based on expected signal
    characteristics for each activity. Useful for architecture validation
    and demo purposes — real training data needed for deployment.

    Args:
        config: Model configuration
        n_samples_per_class: Samples per activity class
        seed: Random seed

    Returns:
        Tuple of (X, y) where X is (n_samples, time_steps, features)
        and y is (n_samples,) integer labels
    """
    rng = np.random.default_rng(seed)
    X_all = []
    y_all = []

    for class_idx in range(config.n_classes):
        label = config.labels[class_idx] if class_idx < len(config.labels) else "UNKNOWN"

        for _ in range(n_samples_per_class):
            # Base signal: ambient WiFi CSI pattern
            t = np.linspace(0, 2 * np.pi, config.n_time_steps)
            base = rng.uniform(5, 15, config.input_features)

            window = np.zeros((config.n_time_steps, config.input_features))

            if label == "EMPTY":
                # Stable signal with minor environmental noise
                for j in range(config.input_features):
                    window[:, j] = base[j] + rng.normal(0, 0.3, config.n_time_steps)

            elif label == "STANDING":
                # Moderate attenuation, slight breathing modulation
                for j in range(config.input_features):
                    breathing = 0.5 * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 2 * np.pi))
                    window[:, j] = base[j] * 0.7 + breathing + rng.normal(0, 0.5, config.n_time_steps)

            elif label == "WALKING":
                # Large periodic fluctuations from body movement
                for j in range(config.input_features):
                    walk_freq = rng.uniform(0.8, 1.5)  # ~1 Hz step frequency
                    movement = 3.0 * np.sin(2 * np.pi * walk_freq * t + rng.uniform(0, 2 * np.pi))
                    window[:, j] = base[j] * 0.6 + movement + rng.normal(0, 1.0, config.n_time_steps)

            elif label == "SITTING":
                # Lower signal, occasional small shifts
                for j in range(config.input_features):
                    shift = 0.3 * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 2 * np.pi))
                    window[:, j] = base[j] * 0.5 + shift + rng.normal(0, 0.4, config.n_time_steps)

            elif label == "PRONE":
                # Lowest signal (body parallel to floor), very subtle breathing
                for j in range(config.input_features):
                    breath = 0.2 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 2 * np.pi))
                    window[:, j] = base[j] * 0.3 + breath + rng.normal(0, 0.3, config.n_time_steps)

            elif label == "BREATHING":
                # Stationary but with clear respiratory signal (0.15-0.4 Hz)
                breath_rate = rng.uniform(0.15, 0.4)
                for j in range(config.input_features):
                    breath = 0.8 * np.sin(2 * np.pi * breath_rate * t + rng.uniform(0, 2 * np.pi))
                    window[:, j] = base[j] * 0.65 + breath + rng.normal(0, 0.2, config.n_time_steps)

            X_all.append(window)
            y_all.append(class_idx)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int64)

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]
