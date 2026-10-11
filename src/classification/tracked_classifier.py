from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted
from sklearn.exceptions import NotFittedError

from log.resource_tracker import track_step_resources


class TrackedClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Proxy class that wraps any sklearn classifier to track fit and predict resources.
    """

    def __init__(self, model: BaseEstimator):
        self.model = model
        self.model_name = model.__class__.__name__

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any):
        with track_step_resources(f"Training Model: {self.model_name}"):
            self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X: np.ndarray) -> Any:
        with track_step_resources(
            f"Inference Model: {self.model_name} (Batch Size: {len(X)})"
        ):
            return self.model.predict(X)

    def __getattr__(self, name: str):
        return getattr(self.model, name)

    def __sklearn_is_fitted__(self) -> bool:
        """
        Allows scikit-learn Pipeline to correctly detect if the underlying model is fitted.
        """
        try:
            check_is_fitted(self.model)
            return True
        except NotFittedError:
            return False
