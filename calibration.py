"""
Probability calibration utilities.

Uses scikit-learn if available (LogisticRegression or IsotonicRegression)
and falls back to a bounded affine transform when unavailable or not fitted.

This module avoids network calls and large dependencies beyond sklearn.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    import numpy as np
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    np = None  # type: ignore


class Calibrator:
    """Lightweight per-hazard probability calibrator.

    If sklearn is available and a calibration file exists, apply the learned
    mapping; otherwise use a safe bounded affine transform.
    """

    def __init__(self, model_path: str = "calibration.pkl") -> None:
        self.model_path = model_path
        self.per_hazard = {}  # type: Dict[str, object]
        self._loaded = False
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.per_hazard = pickle.load(f)
                self._loaded = True
            except Exception:
                self.per_hazard = {}
                self._loaded = False

    def is_ready(self) -> bool:
        return self._loaded and bool(self.per_hazard)

    def apply(self, hazard: str, p: float) -> float:
        """Apply calibration to a single probability value in [0,1]."""
        p = float(max(0.0, min(1.0, p)))
        model = self.per_hazard.get(hazard)
        if SKLEARN_AVAILABLE and model is not None and np is not None:
            try:
                if isinstance(model, (LogisticRegression, IsotonicRegression)):
                    pred = float(model.predict_proba(np.array([[p]]))[:, 1][0]) if hasattr(model, "predict_proba") else float(model.transform([p])[0])
                    return max(0.0, min(1.0, pred))
            except Exception:
                pass
        # Fallback: bounded affine transform to avoid overconfidence
        return max(0.55, min(0.98, p * 0.85 + 0.15))


