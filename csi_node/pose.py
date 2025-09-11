"""WiPose-based pose estimation interface.

This module wraps the WiPose model (if installed) so that the main pipeline
can request pose predictions on CSI amplitude windows. The integration is
kept lightweight to avoid breaking when the model is unavailable."""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:  # WiPose may not be installed in minimal environments
    import torch  # noqa: F401
    from wipose import WiPose
except Exception:  # pragma: no cover - best effort import
    WiPose = None


class PoseEstimator:
    """Wrapper around the optional WiPose model.

    When WiPose is not installed, the estimator returns ``("unknown", 0.0)``
    for any input. The interface mirrors what ``pipeline.compute_window``
    expects: ``predict`` accepts an ``np.ndarray`` of CSI amplitudes and
    returns a ``(pose, confidence)`` tuple.
    """

    def __init__(self) -> None:
        self.model = None
        if WiPose is not None:
            try:
                # WiPose internally loads its pretrained weights.
                self.model = WiPose()
            except Exception:
                # Model failed to load; fall back to dummy behaviour.
                self.model = None

    def predict(self, amps: np.ndarray) -> Tuple[str, float]:
        """Return pose label and confidence for the provided amplitudes."""
        if self.model is None:
            return "unknown", 0.0
        try:
            tensor = np.asarray(amps, dtype="float32")
            tensor = tensor.reshape(1, *tensor.shape)
            pose, conf = self.model.predict(tensor)
            return str(pose), float(conf)
        except Exception:
            # Any failure should not crash the pipeline.
            return "unknown", 0.0
