from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from log.resource_tracker import track_step_resources


class BaseFeaturesExtractor(ABC, BaseEstimator, TransformerMixin):
    """
    Base class for extracting features from an image.
    """

    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Converts an image to a histogram.
        :param image: Numpy 3D tensor representing image pixels (width*height*channels).
        :return: Numpy array representing the extracted features.
        """
        ...

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        all_features = []
        extractor_name = self.__class__.__name__

        with track_step_resources(f"Feature Extraction Batch ({extractor_name})"):
            for i, image in enumerate(X):
                image_features = self.extract(image)
                all_features.append(image_features.ravel())

        return np.array(all_features)
