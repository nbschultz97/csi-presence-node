"""Multi-person detection and counting from WiFi CSI data.

Estimates the number of people in a monitored area using CSI statistical
features. Approach: CSI variance, spectral complexity, and spatial diversity
increase with additional occupants due to independent body reflections
creating distinct multipath signatures.

Methods:
1. **Variance decomposition**: Multi-person environments show higher variance
   with distinct temporal patterns (independent motion = higher entropy).
2. **Spectral peak counting**: Each person's breathing/movement contributes
   frequency peaks; count peaks in 0.1-2 Hz band.
3. **Eigenvalue spread**: PCA on subcarrier covariance — more people = more
   significant eigenvalues (each body is an independent reflector).
4. **Fusion classifier**: RandomForest trained on the above features maps
   to person count (0, 1, 2, 3+).

References:
- Wang et al., "E-eyes: Device-free location-oriented activity identification
  using fine-grained WiFi signatures" (2014)
- Xi et al., "Electronic frog eye: Counting crowd using WiFi" (2014)

Usage:
    from csi_node.multi_person import MultiPersonDetector
    detector = MultiPersonDetector()
    detector.calibrate(empty_room_frames)
    result = detector.estimate(recent_frames)
    print(f"Estimated {result.count} people (confidence: {result.confidence:.0%})")
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class PersonCountEstimate:
    """Result of multi-person estimation."""
    count: int = 0                  # Estimated number of people (0, 1, 2, 3 = "3+")
    confidence: float = 0.0         # Overall confidence [0, 1]
    method: str = "none"            # Primary method that drove the estimate
    eigenvalue_spread: float = 0.0  # PCA eigenvalue spread metric
    spectral_peaks: int = 0         # Number of detected spectral peaks
    variance_ratio: float = 0.0     # Variance relative to baseline
    entropy: float = 0.0            # Temporal entropy of CSI signal
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def extract_counting_features(frames: np.ndarray,
                               sample_rate_hz: float = 30.0,
                               baseline_variance: float = 1.0) -> dict:
    """Extract features useful for person counting from a window of CSI frames.

    Args:
        frames: np.ndarray of shape (n_frames, n_subcarriers).
        sample_rate_hz: CSI packet rate in Hz.
        baseline_variance: Empty-room variance for normalization.

    Returns:
        Dict of feature name → float value.
    """
    n_frames, n_sub = frames.shape
    features = {}

    # 1. Overall variance ratio
    total_var = float(np.var(frames))
    features["variance_ratio"] = total_var / max(baseline_variance, 1e-10)

    # 2. Mean per-subcarrier variance
    sub_variances = np.var(frames, axis=0)
    features["mean_subcarrier_var"] = float(np.mean(sub_variances))
    features["std_subcarrier_var"] = float(np.std(sub_variances))

    # 3. Eigenvalue spread (PCA on subcarrier covariance)
    # More people → more significant principal components
    centered = frames - np.mean(frames, axis=0, keepdims=True)
    if n_frames > 2 and n_sub > 1:
        cov = np.cov(centered.T)
        eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(cov)))[::-1]
        # Normalize
        eig_sum = np.sum(eigenvalues) + 1e-10
        eig_normalized = eigenvalues / eig_sum

        # Number of eigenvalues explaining >95% variance
        cumsum = np.cumsum(eig_normalized)
        n_components_95 = int(np.searchsorted(cumsum, 0.95)) + 1
        features["n_components_95"] = float(n_components_95)

        # Effective rank (exponential of spectral entropy)
        eig_pos = eig_normalized[eig_normalized > 1e-10]
        spectral_entropy = -float(np.sum(eig_pos * np.log(eig_pos)))
        features["spectral_entropy"] = spectral_entropy
        features["effective_rank"] = float(np.exp(spectral_entropy))

        # Ratio of top eigenvalue to sum (lower = more spread = more people)
        features["top_eig_ratio"] = float(eig_normalized[0])
        # Ratio of top 2 eigenvalues
        features["top2_eig_ratio"] = float(np.sum(eig_normalized[:2])) if len(eig_normalized) >= 2 else 1.0
    else:
        features["n_components_95"] = 1.0
        features["spectral_entropy"] = 0.0
        features["effective_rank"] = 1.0
        features["top_eig_ratio"] = 1.0
        features["top2_eig_ratio"] = 1.0

    # 4. Spectral peak counting (breathing/movement bands)
    if n_frames >= 16:
        # Average across subcarriers for a single time series
        avg_signal = np.mean(frames, axis=1)
        avg_signal = avg_signal - np.mean(avg_signal)

        fft_vals = np.fft.rfft(avg_signal)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n_frames, d=1.0 / sample_rate_hz)

        # Human-relevant band: 0.1 - 2.0 Hz (breathing + slow movement)
        mask = (freqs >= 0.1) & (freqs <= 2.0)
        band_power = power[mask]
        band_freqs = freqs[mask]

        if len(band_power) > 2:
            # Simple peak detection: local maxima above median
            median_power = np.median(band_power)
            threshold = median_power * 3.0
            peaks = 0
            for i in range(1, len(band_power) - 1):
                if (band_power[i] > band_power[i-1] and
                    band_power[i] > band_power[i+1] and
                    band_power[i] > threshold):
                    peaks += 1
            features["spectral_peaks"] = float(peaks)
            features["band_power_total"] = float(np.sum(band_power))
            features["band_power_max"] = float(np.max(band_power))
        else:
            features["spectral_peaks"] = 0.0
            features["band_power_total"] = 0.0
            features["band_power_max"] = 0.0
    else:
        features["spectral_peaks"] = 0.0
        features["band_power_total"] = 0.0
        features["band_power_max"] = 0.0

    # 5. Temporal entropy (higher with multiple independent movers)
    if n_frames > 1:
        diffs = np.diff(frames, axis=0)
        diff_energy = np.sum(diffs ** 2, axis=1)
        diff_energy = diff_energy / (np.sum(diff_energy) + 1e-10)
        diff_pos = diff_energy[diff_energy > 1e-10]
        temporal_entropy = -float(np.sum(diff_pos * np.log(diff_pos)))
        features["temporal_entropy"] = temporal_entropy
    else:
        features["temporal_entropy"] = 0.0

    # 6. Cross-subcarrier correlation (lower avg correlation = more people)
    if n_frames > 2 and n_sub > 1:
        # Sample a subset of subcarrier pairs for efficiency
        n_pairs = min(50, n_sub * (n_sub - 1) // 2)
        rng = np.random.default_rng(42)
        correlations = []
        for _ in range(n_pairs):
            i, j = rng.integers(0, n_sub, size=2)
            if i != j:
                r = np.corrcoef(frames[:, i], frames[:, j])[0, 1]
                if not np.isnan(r):
                    correlations.append(abs(r))
        features["mean_cross_corr"] = float(np.mean(correlations)) if correlations else 1.0
    else:
        features["mean_cross_corr"] = 1.0

    return features


# Feature names in canonical order for classifier
FEATURE_NAMES = [
    "variance_ratio", "mean_subcarrier_var", "std_subcarrier_var",
    "n_components_95", "spectral_entropy", "effective_rank",
    "top_eig_ratio", "top2_eig_ratio",
    "spectral_peaks", "band_power_total", "band_power_max",
    "temporal_entropy", "mean_cross_corr",
]


class MultiPersonDetector:
    """Estimates the number of people from CSI data.

    Uses a combination of statistical signal features and an optional
    trained classifier. Without training, falls back to heuristic rules
    based on eigenvalue spread and spectral peaks.

    Args:
        sample_rate_hz: Expected CSI packet rate.
        window_seconds: Analysis window length.
        history_seconds: How much history to keep for dashboard.
    """

    def __init__(
        self,
        sample_rate_hz: float = 30.0,
        window_seconds: float = 5.0,
        history_seconds: float = 120.0,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.window_seconds = window_seconds

        # Baseline from calibration
        self._baseline_variance: float = 1.0
        self._calibrated: bool = False

        # Frame buffer
        buf_size = int(sample_rate_hz * window_seconds * 1.5)
        self._frame_buffer: deque = deque(maxlen=buf_size)

        # History
        max_hist = int(history_seconds * 0.5)  # ~1 estimate per 2 sec
        self._history: deque[PersonCountEstimate] = deque(maxlen=max_hist)
        self._state = PersonCountEstimate()

        # Optional trained classifier
        self._classifier: Optional[RandomForestClassifier] = None
        self._scaler: Optional[StandardScaler] = None

    @property
    def state(self) -> PersonCountEstimate:
        return self._state

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def history(self) -> list[PersonCountEstimate]:
        return list(self._history)

    def calibrate(self, frames: np.ndarray) -> bool:
        """Calibrate with empty-room CSI frames.

        Args:
            frames: np.ndarray of shape (n_frames, n_subcarriers).

        Returns:
            True if calibration succeeded.
        """
        if len(frames) < 10:
            return False
        self._baseline_variance = float(np.var(frames))
        self._baseline_variance = max(self._baseline_variance, 1e-6)
        self._calibrated = True
        return True

    def feed_frame(self, amps: np.ndarray) -> None:
        """Add a single CSI frame to the buffer without running estimation."""
        self._frame_buffer.append(amps.flatten().astype(np.float64))

    def estimate(self, frames: Optional[np.ndarray] = None,
                 timestamp: Optional[float] = None) -> PersonCountEstimate:
        """Estimate the number of people.

        Args:
            frames: Optional explicit frame window. If None, uses internal buffer.
            timestamp: Optional timestamp.

        Returns:
            PersonCountEstimate with count and confidence.
        """
        ts = timestamp or time.time()

        if frames is None:
            if len(self._frame_buffer) < 15:
                self._state = PersonCountEstimate(timestamp=ts)
                return self._state
            frames = np.array(list(self._frame_buffer))

        if frames.ndim == 1:
            frames = frames.reshape(1, -1)
        if len(frames) < 10:
            self._state = PersonCountEstimate(timestamp=ts)
            return self._state

        features = extract_counting_features(
            frames,
            sample_rate_hz=self.sample_rate_hz,
            baseline_variance=self._baseline_variance,
        )

        # Use classifier if trained, otherwise heuristic
        if self._classifier is not None and self._scaler is not None:
            result = self._classify(features, ts)
        else:
            result = self._heuristic(features, ts)

        self._state = result
        self._history.append(result)
        return result

    def _heuristic(self, features: dict, ts: float) -> PersonCountEstimate:
        """Rule-based person count estimation.

        Heuristic thresholds derived from literature on WiFi CSI
        person counting (Xi et al. 2014, Wang et al. 2014).
        """
        var_ratio = features["variance_ratio"]
        effective_rank = features["effective_rank"]
        spectral_peaks = int(features["spectral_peaks"])
        temporal_entropy = features["temporal_entropy"]
        top_eig_ratio = features["top_eig_ratio"]
        mean_cross_corr = features["mean_cross_corr"]

        # Score system: accumulate evidence for each count
        scores = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

        # Variance ratio scoring
        if var_ratio < 1.5:
            scores[0] += 3.0
        elif var_ratio < 3.0:
            scores[1] += 2.0
        elif var_ratio < 5.0:
            scores[2] += 2.0
        else:
            scores[3] += 2.0

        # Effective rank scoring (more people = higher rank)
        if effective_rank < 2.0:
            scores[0] += 2.0
            scores[1] += 1.0
        elif effective_rank < 4.0:
            scores[1] += 2.0
        elif effective_rank < 7.0:
            scores[2] += 2.0
        else:
            scores[3] += 2.0

        # Spectral peaks (each person contributes ~1 breathing peak)
        if spectral_peaks == 0:
            scores[0] += 2.0
        elif spectral_peaks == 1:
            scores[1] += 2.0
        elif spectral_peaks == 2:
            scores[2] += 2.5
        else:
            scores[3] += 2.5

        # Top eigenvalue ratio (lower = more spread = more people)
        if top_eig_ratio > 0.8:
            scores[0] += 1.5
            scores[1] += 0.5
        elif top_eig_ratio > 0.5:
            scores[1] += 1.5
        elif top_eig_ratio > 0.3:
            scores[2] += 1.5
        else:
            scores[3] += 1.5

        # Cross-correlation (lower = more independent sources = more people)
        if mean_cross_corr > 0.8:
            scores[0] += 1.0
            scores[1] += 0.5
        elif mean_cross_corr > 0.5:
            scores[1] += 1.0
        elif mean_cross_corr > 0.3:
            scores[2] += 1.0
        else:
            scores[3] += 1.0

        # Temporal entropy
        if temporal_entropy < 2.0:
            scores[0] += 1.0
        elif temporal_entropy < 3.5:
            scores[1] += 1.0
        elif temporal_entropy < 5.0:
            scores[2] += 1.0
        else:
            scores[3] += 1.0

        # Pick winner
        total = sum(scores.values()) + 1e-10
        best_count = max(scores, key=scores.get)  # type: ignore[arg-type]
        confidence = scores[best_count] / total

        # Determine primary method
        method_scores = {
            "eigenvalue": abs(scores.get(best_count, 0)),
            "spectral": spectral_peaks * 1.5,
            "variance": var_ratio,
        }
        method = max(method_scores, key=method_scores.get)  # type: ignore[arg-type]

        return PersonCountEstimate(
            count=best_count,
            confidence=confidence,
            method=method,
            eigenvalue_spread=features["effective_rank"],
            spectral_peaks=spectral_peaks,
            variance_ratio=var_ratio,
            entropy=temporal_entropy,
            timestamp=ts,
        )

    def _classify(self, features: dict, ts: float) -> PersonCountEstimate:
        """Use trained classifier for estimation."""
        feature_vec = np.array([[features[name] for name in FEATURE_NAMES]])
        scaled = self._scaler.transform(feature_vec)
        prediction = int(self._classifier.predict(scaled)[0])
        probas = self._classifier.predict_proba(scaled)[0]
        confidence = float(np.max(probas))

        return PersonCountEstimate(
            count=prediction,
            confidence=confidence,
            method="classifier",
            eigenvalue_spread=features["effective_rank"],
            spectral_peaks=int(features["spectral_peaks"]),
            variance_ratio=features["variance_ratio"],
            entropy=features["temporal_entropy"],
            timestamp=ts,
        )

    def train(self, training_data: list[tuple[np.ndarray, int]],
              sample_rate_hz: Optional[float] = None) -> dict:
        """Train the person count classifier on labeled data.

        Args:
            training_data: List of (frames_window, person_count) tuples.
                frames_window shape: (n_frames, n_subcarriers).
                person_count: 0, 1, 2, or 3 (3+).
            sample_rate_hz: Override sample rate for feature extraction.

        Returns:
            Dict with training metrics.
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for classifier training")

        sr = sample_rate_hz or self.sample_rate_hz
        X = []
        y = []
        for frames, count in training_data:
            features = extract_counting_features(
                frames, sample_rate_hz=sr,
                baseline_variance=self._baseline_variance,
            )
            X.append([features[name] for name in FEATURE_NAMES])
            y.append(min(count, 3))  # Cap at 3+

        X_arr = np.array(X)
        y_arr = np.array(y)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_arr)

        self._classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight="balanced",
        )
        self._classifier.fit(X_scaled, y_arr)

        # Training accuracy
        train_acc = float(self._classifier.score(X_scaled, y_arr))

        return {
            "n_samples": len(y_arr),
            "classes": sorted(set(y_arr.tolist())),
            "train_accuracy": train_acc,
            "feature_importances": dict(zip(
                FEATURE_NAMES,
                self._classifier.feature_importances_.tolist(),
            )),
        }

    def save_model(self, path: str | Path) -> None:
        """Save trained classifier and scaler to JSON-compatible format."""
        import pickle
        data = {
            "classifier": self._classifier,
            "scaler": self._scaler,
            "baseline_variance": self._baseline_variance,
        }
        Path(path).write_bytes(pickle.dumps(data))

    def load_model(self, path: str | Path) -> bool:
        """Load a trained classifier. Returns True if successful."""
        import pickle
        try:
            data = pickle.loads(Path(path).read_bytes())
            self._classifier = data["classifier"]
            self._scaler = data["scaler"]
            self._baseline_variance = data.get("baseline_variance", 1.0)
            self._calibrated = True
            return True
        except Exception:
            return False

    def get_dashboard_data(self) -> dict:
        """Get data formatted for the web dashboard."""
        history = list(self._history)[-60:]
        return {
            "current": self._state.to_dict(),
            "history": {
                "timestamps": [h.timestamp for h in history],
                "count": [h.count for h in history],
                "confidence": [h.confidence for h in history],
                "eigenvalue_spread": [h.eigenvalue_spread for h in history],
                "spectral_peaks": [h.spectral_peaks for h in history],
            },
            "calibrated": self._calibrated,
            "baseline_variance": self._baseline_variance,
        }
