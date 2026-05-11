from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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
        features = []

        for image in X:
            feat = self.extract(image)
            features.append(feat)
        return np.array(features)
