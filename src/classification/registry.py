from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from log.resource_tracker import track_step_resources


class TrackedClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Proxy class that wraps any sklearn classifier to track fit and predict resources.
    """

    def __init__(self, model: BaseEstimator):
        self.model = model
        self.model_name = model.__class__.__name__

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        with track_step_resources(f"Training Model: {self.model_name}"):
            self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X: np.ndarray):
        with track_step_resources(
            f"Inference Model: {self.model_name} (Batch Size: {len(X)})"
        ):
            return self.model.predict(X)

    def __getattr__(self, name: str):
        # Forward any other method calls (like predict_proba) to the underlying model
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


class ClassifierRegistry:
    """
    Factory that returns a classifier object based on its name as a string.
    Available classifiers:
        - "svm"
        - "random_forest"
        - "knn"
    """

    _registry: dict[str, type[Any]] = {
        "svm": SVC,
        "svm_linear": LinearSVC,
        "random_forest": RandomForestClassifier,
        "knn": KNeighborsClassifier,
    }

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Any:
        """
        Returns a classifier object based on its name as a string.
        :param name: Name of the classifier.
        :param kwargs: Arguments that the classifier accepts in its constructor.
        :return: Classifier object.
        """
        name = name.lower()

        # Linear kernel doesn't accept some params so they're disabled
        if name == "svm" and kwargs.get("kernel") == "linear":
            name = "svm_linear"
            kwargs.pop("kernel", None)
            kwargs.pop("gamma", None)
            kwargs.pop("degree", None)
            kwargs.pop("shrinking", None)
            kwargs.pop("probability", None)

        model_class = cls._registry.get(name.lower())

        if not model_class:
            valid_options = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Classifier '{name}' not found. Available: {valid_options}"
            )

        raw_model = model_class(**kwargs)
        return TrackedClassifierWrapper(raw_model)
