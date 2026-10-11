from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from classification.tracked_classifier import TrackedClassifierWrapper


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
        "logistic_regression": LogisticRegression,
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
