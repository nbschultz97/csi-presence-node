from __future__ import annotations
"""Minimal pose classifier with optional training and joblib model loading."""

from pathlib import Path
from typing import Tuple
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

LABELS = ["STANDING", "CROUCHING"]


class PoseClassifier:
    """Two-class pose classifier returning label and confidence.

    If ``model_path`` exists, the model is loaded via ``joblib``. Otherwise a
    tiny deterministic logistic regression model is instantiated so the demo
    remains functional even without training data.
    """

    def __init__(self, model_path: str | None = None):
        self.model = None
        if model_path and Path(model_path).exists():
            try:
                self.model = joblib.load(model_path)
            except Exception:
                pass
        if self.model is None:
            self.model = self._toy_model()

    @staticmethod
    def _toy_model() -> LogisticRegression:
        """Return a deterministic logistic regression model."""
        clf = LogisticRegression()
        # Deterministic dataset: two points, two classes.
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 1])
        clf.fit(X, y)
        return clf

    def predict(self, X_window: np.ndarray) -> Tuple[str, float]:
        """Predict pose label and confidence for a feature window."""
        if X_window.ndim == 1:
            X_window = X_window.reshape(1, -1)
        proba = self.model.predict_proba(X_window)[0]
        idx = int(np.argmax(proba))
        label = LABELS[idx]
        conf = float(proba[idx])
        return label, conf


def _train(in_path: str | None, out_path: str) -> None:
    """Train a logistic regression classifier and save via joblib."""
    if in_path and Path(in_path).exists():
        data = np.load(in_path)
        X, y = data["X"], data["y"]
    else:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(32, 2))
        y = rng.integers(0, 2, size=32)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pose classifier utility")
    parser.add_argument("--train", action="store_true", help="train model")
    parser.add_argument("--in", dest="in_path", default=None, help="training npz")
    parser.add_argument("--out", dest="out_path", default="models/wipose.joblib")
    args = parser.parse_args()
    if args.train:
        _train(args.in_path, args.out_path)


if __name__ == "__main__":
    main()
