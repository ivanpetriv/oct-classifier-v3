from abc import ABC, abstractmethod
import time
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
        all_features = []

        start_time = time.perf_counter()
        for image in X:
            image_features = self.extract(image)
            all_features.append(image_features.ravel())
        end_time = time.perf_counter()
        print(f"Processed {len(all_features)} images in {(end_time - start_time):.3f}s")
        return np.array(all_features)
