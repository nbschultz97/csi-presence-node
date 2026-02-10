"""Enhanced pose classifier with improved feature extraction and model support.

Supports multiple model backends and advanced feature extraction from CSI data.

Usage:
    from csi_node.pose_classifier import PoseClassifier

    # Load pre-trained model if available
    clf = PoseClassifier("models/wipose.joblib")

    # Predict from CSI features
    label, confidence = clf.predict(features)

Training:
    python -m csi_node.pose_classifier --train --in data/labeled.npz --out models/wipose.joblib
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Pose labels
LABELS = ["STANDING", "CROUCHING", "PRONE", "SITTING", "WALKING"]
LABEL_MAP = {label: idx for idx, label in enumerate(LABELS)}

# Feature names for interpretability
FEATURE_NAMES = [
    "mean_magnitude",
    "std_magnitude",
    "max_magnitude",
    "min_magnitude",
    "range_magnitude",
    "variance",
    "pca1",
    "pca2",
    "pca3",
    "spectral_entropy",
    "zero_crossing_rate",
    "temporal_variance",
    "chain_diff_mean",
    "chain_diff_std",
]


def extract_features(
    csi_window: np.ndarray,
    rssi: Optional[List[float]] = None,
    extended: bool = True,
) -> np.ndarray:
    """Extract features from a CSI window for pose classification.

    Args:
        csi_window: CSI amplitude array of shape (time_steps, chains, subcarriers)
                    or (time_steps, features)
        rssi: Optional per-chain RSSI values [rssi0, rssi1]
        extended: If True, compute extended feature set

    Returns:
        Feature vector as 1D numpy array
    """
    # Flatten to (time_steps, features) if needed
    if csi_window.ndim == 3:
        n_steps = csi_window.shape[0]
        csi_flat = csi_window.reshape(n_steps, -1)
    elif csi_window.ndim == 2:
        csi_flat = csi_window
    else:
        csi_flat = csi_window.reshape(1, -1)

    # Take absolute values (magnitude)
    amps = np.abs(csi_flat)

    features = []

    # Basic amplitude statistics
    mean_mag = float(np.mean(amps))
    std_mag = float(np.std(amps))
    max_mag = float(np.max(amps))
    min_mag = float(np.min(amps))
    range_mag = max_mag - min_mag

    features.extend([mean_mag, std_mag, max_mag, min_mag, range_mag])

    if extended and amps.shape[0] > 1:
        # Variance across time
        variance = float(np.var(amps))
        features.append(variance)

        # PCA eigenvalues (motion patterns)
        try:
            centered = amps - np.mean(amps, axis=0, keepdims=True)
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            eigenvalues = (s ** 2) / max(amps.shape[0] - 1, 1)
            pca1 = float(eigenvalues[0]) if len(eigenvalues) > 0 else 0.0
            pca2 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
            pca3 = float(eigenvalues[2]) if len(eigenvalues) > 2 else 0.0
        except Exception:
            pca1 = pca2 = pca3 = 0.0
        features.extend([pca1, pca2, pca3])

        # Spectral entropy (complexity measure)
        try:
            # FFT along time axis for each subcarrier
            fft_result = np.fft.fft(amps, axis=0)
            power_spectrum = np.abs(fft_result) ** 2
            power_spectrum = power_spectrum / (np.sum(power_spectrum) + 1e-10)
            spectral_entropy = float(-np.sum(power_spectrum * np.log2(power_spectrum + 1e-10)))
        except Exception:
            spectral_entropy = 0.0
        features.append(spectral_entropy)

        # Zero crossing rate (movement activity)
        try:
            mean_subtracted = amps - np.mean(amps, axis=0, keepdims=True)
            signs = np.sign(mean_subtracted)
            zero_crossings = np.sum(np.abs(np.diff(signs, axis=0))) / 2
            zcr = float(zero_crossings / (amps.shape[0] - 1)) if amps.shape[0] > 1 else 0.0
        except Exception:
            zcr = 0.0
        features.append(zcr)

        # Temporal variance (change over time)
        try:
            temporal_var = float(np.mean(np.var(amps, axis=0)))
        except Exception:
            temporal_var = 0.0
        features.append(temporal_var)

        # Per-chain difference (if available from original shape)
        if rssi is not None and len(rssi) >= 2:
            chain_diff_mean = float(rssi[0] - rssi[1])
            chain_diff_std = float(abs(rssi[0] - rssi[1]))
        else:
            chain_diff_mean = 0.0
            chain_diff_std = 0.0
        features.extend([chain_diff_mean, chain_diff_std])

    else:
        # Fill with zeros for non-extended or single-frame
        features.extend([0.0] * 9)

    return np.array(features, dtype=np.float32)


def extract_features_simple(mean_mag: float, std_mag: float) -> np.ndarray:
    """Extract minimal features for backward compatibility.

    Args:
        mean_mag: Mean CSI magnitude
        std_mag: Standard deviation of CSI magnitude

    Returns:
        Feature vector with basic features and zeros for extended
    """
    return np.array([
        mean_mag, std_mag, mean_mag, 0.0, std_mag,  # Basic stats
        0.0, 0.0, 0.0, 0.0,  # PCA features
        0.0, 0.0, 0.0,  # Spectral/temporal
        0.0, 0.0,  # Chain diff
    ], dtype=np.float32)


class PoseClassifier:
    """Pose classifier with support for multiple model types.

    Supports:
    - Pre-trained joblib models (sklearn pipelines or classifiers)
    - Fallback to simple rule-based or toy model if no model available

    The classifier expects a feature vector from extract_features().
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "auto",
        labels: Optional[List[str]] = None,
    ):
        """Initialize the pose classifier.

        Args:
            model_path: Path to joblib model file (optional)
            model_type: Model type ("auto", "sklearn", "rules")
            labels: Custom pose labels (default: STANDING/CROUCHING/PRONE)
        """
        self.model = None
        self.model_type = model_type
        self.labels = labels or LABELS[:3]  # Default to 3 poses for compatibility
        self.scaler = None
        self._using_toy_model = False

        # Try to load pre-trained model
        if model_path and Path(model_path).exists():
            try:
                loaded = joblib.load(model_path)
                if isinstance(loaded, Pipeline):
                    self.model = loaded
                elif hasattr(loaded, "predict_proba"):
                    self.model = loaded
                else:
                    logging.warning(f"Loaded model has no predict_proba, using toy model")
                    self.model = None
            except Exception as e:
                logging.warning(f"Could not load model from {model_path}: {e}")

        # Fall back to toy model if no model loaded
        if self.model is None:
            self.model = self._create_toy_model()
            self._using_toy_model = True

    def _create_toy_model(self):
        """Create a production-architecture toy model for demo/fallback.

        Uses RandomForestClassifier (production-ready architecture) trained on
        synthetic data.  The feature distributions are hand-tuned approximations
        of expected CSI patterns for each pose.  Replace with a model trained
        on real labeled data for deployment.
        """
        if not SKLEARN_AVAILABLE:
            return None

        rng = np.random.default_rng(42)
        n_samples = 200  # per class

        # Synthetic feature distributions per pose (14 features each).
        # Columns match FEATURE_NAMES order.
        pose_profiles = {
            # STANDING: strong upright signal, moderate variance
            0: [10, 2, 12, 6, 6, 5, 3, 1, 0.5, 2, 0.1, 1, 0, 0],
            # CROUCHING: lower magnitude, higher variance (compressed body)
            1: [7, 3, 10, 4, 6, 8, 5, 2, 1.0, 3, 0.2, 2, 0, 0],
            # PRONE: low magnitude, spread features (body parallel to ground)
            2: [4, 2, 6, 2, 4, 4, 2, 1, 0.5, 4, 0.15, 1.5, 0, 0],
        }

        X_parts, y_parts = [], []
        for label, means in pose_profiles.items():
            X_parts.append(rng.normal(means, 1.0, (n_samples, 14)))
            y_parts.append(np.full(n_samples, label))

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)

        # Production architecture: RandomForest with StandardScaler pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )),
        ])
        pipeline.fit(X, y)
        return pipeline

    def predict(self, X_window: np.ndarray) -> Tuple[str, float]:
        """Predict pose label and confidence.

        Args:
            X_window: Feature vector from extract_features() or simple [mean, std]

        Returns:
            Tuple of (label, confidence)
        """
        if self.model is None:
            # No model available, return default
            return "UNKNOWN", 0.0

        # Handle simple 2-feature input (backward compatibility)
        if X_window.ndim == 1 and len(X_window) == 2:
            X_window = extract_features_simple(X_window[0], X_window[1])

        if X_window.ndim == 1:
            X_window = X_window.reshape(1, -1)

        # Ensure we have the right number of features
        expected_features = 14
        if X_window.shape[1] < expected_features:
            # Pad with zeros
            padding = np.zeros((X_window.shape[0], expected_features - X_window.shape[1]))
            X_window = np.hstack([X_window, padding])
        elif X_window.shape[1] > expected_features:
            # Truncate
            X_window = X_window[:, :expected_features]

        try:
            proba = self.model.predict_proba(X_window)[0]
            idx = int(np.argmax(proba))
            label = self.labels[idx] if idx < len(self.labels) else "UNKNOWN"
            conf = float(proba[idx])
            return label, conf
        except Exception as e:
            logging.warning(f"Prediction failed: {e}")
            return "UNKNOWN", 0.0

    def predict_with_features(
        self,
        csi_window: np.ndarray,
        rssi: Optional[List[float]] = None,
    ) -> Tuple[str, float, np.ndarray]:
        """Predict pose with feature extraction.

        Args:
            csi_window: Raw CSI window array
            rssi: Optional RSSI values

        Returns:
            Tuple of (label, confidence, features)
        """
        features = extract_features(csi_window, rssi, extended=True)
        label, conf = self.predict(features)
        return label, conf, features

    @property
    def is_toy_model(self) -> bool:
        """Whether the classifier is using a toy model."""
        return self._using_toy_model


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "random_forest",
    **kwargs,
) -> Pipeline:
    """Train a pose classifier.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)
        model_type: Type of classifier ("logistic", "random_forest", "gradient_boosting")
        **kwargs: Additional arguments for the classifier

    Returns:
        Trained sklearn Pipeline
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for training")

    # Create classifier
    if model_type == "logistic":
        clf = LogisticRegression(max_iter=1000, multi_class="multinomial", **kwargs)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42, **kwargs)
    elif model_type == "gradient_boosting":
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create pipeline with scaling
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])

    # Train
    pipeline.fit(X, y)

    return pipeline


def _train_from_file(in_path: Optional[str], out_path: str, model_type: str) -> None:
    """Train from data file and save model."""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for training")

    if in_path and Path(in_path).exists():
        data = np.load(in_path)
        X, y = data["X"], data["y"]
        print(f"Loaded {len(X)} samples from {in_path}")
    else:
        # Generate synthetic data for demo
        print("No training data provided, generating synthetic dataset...")
        rng = np.random.default_rng(42)
        n_per_class = 200

        X_list = []
        y_list = []

        # Generate features for each pose class
        # These are rough approximations of expected feature distributions
        pose_params = {
            0: {"mean_mag": 10, "std_mag": 2, "variance": 5, "pca1": 3},  # Standing
            1: {"mean_mag": 7, "std_mag": 3, "variance": 8, "pca1": 5},   # Crouching
            2: {"mean_mag": 4, "std_mag": 2, "variance": 4, "pca1": 2},   # Prone
        }

        for label, params in pose_params.items():
            for _ in range(n_per_class):
                features = np.array([
                    rng.normal(params["mean_mag"], 1),
                    rng.normal(params["std_mag"], 0.5),
                    rng.normal(params["mean_mag"] + 2, 1),  # max
                    rng.normal(params["mean_mag"] - 4, 1),  # min
                    rng.normal(6, 1),  # range
                    rng.normal(params["variance"], 1),
                    rng.normal(params["pca1"], 0.5),
                    rng.normal(params["pca1"] * 0.5, 0.3),  # pca2
                    rng.normal(params["pca1"] * 0.2, 0.2),  # pca3
                    rng.normal(3, 0.5),  # spectral entropy
                    rng.normal(0.15, 0.05),  # zcr
                    rng.normal(1.5, 0.3),  # temporal var
                    rng.normal(0, 1),  # chain diff mean
                    rng.normal(1, 0.5),  # chain diff std
                ])
                X_list.append(features)
                y_list.append(label)

        X = np.array(X_list)
        y = np.array(y_list)

    # Train
    print(f"Training {model_type} classifier...")
    pipeline = train_classifier(X, y, model_type=model_type)

    # Evaluate (simple train accuracy)
    accuracy = pipeline.score(X, y)
    print(f"Training accuracy: {accuracy:.2%}")

    # Save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    print(f"Saved model to {out_path}")


def main() -> None:
    """CLI entry point for pose classifier training and testing."""
    parser = argparse.ArgumentParser(description="Pose classifier utility")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--in", dest="in_path", default=None, help="Training data (.npz)")
    parser.add_argument("--out", dest="out_path", default="models/wipose.joblib", help="Output model")
    parser.add_argument(
        "--model-type",
        choices=["logistic", "random_forest", "gradient_boosting"],
        default="random_forest",
        help="Classifier type",
    )
    parser.add_argument("--test", type=str, default=None, help="Test model on data")
    args = parser.parse_args()

    if args.train:
        _train_from_file(args.in_path, args.out_path, args.model_type)
    elif args.test:
        # Load and test model
        clf = PoseClassifier(args.test if Path(args.test).suffix == ".joblib" else "models/wipose.joblib")
        print(f"Using toy model: {clf.is_toy_model}")

        # Test with sample features
        test_features = np.array([10, 2, 12, 6, 6, 5, 3, 1, 0.5, 2, 0.1, 1, 0, 0])
        label, conf = clf.predict(test_features)
        print(f"Test prediction: {label} (confidence: {conf:.2%})")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
