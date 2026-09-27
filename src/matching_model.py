"""Matching Model module supporting Random Forest and alternative classifiers.

Provides:
- Abstracted matching model interface
- Factory for model creation (Random Forest, LightGBM, Gradient Boosting)
- Class imbalance handling (balanced weights)
- Feature importance extraction
- Model persistence (joblib) with metadata
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

# Optional LightGBM support
try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False


class MatchingClassifier:
    """Wrapper around scikit-learn compatible binary classifiers for entity matching."""

    def __init__(
        self,
        model_type: str = "random_forest",
        feature_names: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        self.model_type = model_type.lower()
        self.feature_names = feature_names or []
        self.model_params = model_params or {}
        self.model = self._init_underlying_model()
        self.is_fitted = False

    def _init_underlying_model(self):
        """Initialize the underlying classifier based on model_type."""

        if self.model_type in ("rf", "random_forest", "randomforest"):
            from sklearn.ensemble import RandomForestClassifier

            default_params = {
                "n_estimators": 100,
                "max_depth": 14,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            }

            default_params.update(self.model_params)

            return RandomForestClassifier(**default_params)

        elif self.model_type in ("lgb", "lightgbm"):

            if not _HAS_LIGHTGBM:
                raise ImportError(
                    "lightgbm is not installed. "
                    "Install it or use 'random_forest'."
                )

            default_params = {
                "n_estimators": 150,
                "learning_rate": 0.08,
                "num_leaves": 31,
                "max_depth": 8,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }

            default_params.update(self.model_params)

            return lgb.LGBMClassifier(**default_params)

        elif self.model_type in ("gb", "gradient_boosting"):
            from sklearn.ensemble import GradientBoostingClassifier

            default_params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "random_state": 42,
            }

            default_params.update(self.model_params)

            return GradientBoostingClassifier(**default_params)

        else:
            raise ValueError(
                f"Unknown model type: {self.model_type}. "
                f"Choose from: random_forest, lightgbm, gradient_boosting"
            )

    def fit(self, X, y):
        """Train the matching model."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """Return probability predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        return self.model.predict_proba(X)

    def predict(self, X):
        """Return class predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        return self.model.predict(X)

    def get_feature_importance(self):
        """Return feature importance values as a dictionary."""

        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        if hasattr(self.model, "feature_importances_"):
            return dict(
                zip(
                    self.feature_names,
                    self.model.feature_importances_
                )
            )

        return {}

    def get_feature_importances(self):
        """Compatibility alias for get_feature_importance."""
        return self.get_feature_importance()

    def save(self, filepath, threshold=0.5):
        """Save trained model and metadata."""

        payload = {
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "model_params": self.model_params,
            "model": self.model,
            "is_fitted": self.is_fitted,
            "threshold": threshold,
        }

        directory = os.path.dirname(filepath)

        if directory:
            os.makedirs(directory, exist_ok=True)

        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath):
        """Load trained model and threshold."""

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Model file not found: {filepath}"
            )

        payload = joblib.load(filepath)

        instance = cls(
            model_type=payload.get("model_type", "random_forest"),
            feature_names=payload.get("feature_names", []),
            model_params=payload.get("model_params", {}),
        )

        instance.model = payload["model"]
        instance.is_fitted = payload.get("is_fitted", True)

        threshold = payload.get("threshold", 0.5)

        return instance, threshold