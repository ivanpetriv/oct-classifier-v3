from typing import Any
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


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
        "random_forest": RandomForestClassifier,
        "knn": KNeighborsClassifier,
    }

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Any:
        """
        Returns a feature extractor object based on its name as a string.
        :param name: Name of the classifier.
        :param kwargs: Arguments that the classifier accepts in its constructor.
        :return: Feature extractor object.
        """
        model_class = cls._registry.get(name.lower())

        if not model_class:
            valid_options = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Classifier '{name}' not found. Available: {valid_options}"
            )

        return model_class(**kwargs)
