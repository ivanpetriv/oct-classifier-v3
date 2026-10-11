import os
from abc import ABC, abstractmethod

import numpy as np
from rich.progress import track
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
        # extractor_name = self.__class__.__name__

        # hide_progress = os.environ.get("DISABLE_EXTRACTOR_BARS", "0") == "1"

        for image in X:
            image_features = self.extract(image)
            all_features.append(image_features.ravel())

        features = np.array(all_features)
        print(
            f"{self.__class__.__name__}: "
            f"input_images={len(X)}, "
            f"output={features.shape}, "
            f"features/image={features.shape[1]}"
        )

        # with track_step_resources(f"Feature Extraction Batch ({extractor_name})"):
        #     iterator = track(
        #         X,
        #         description=f"[cyan]Extracting {extractor_name}...",
        #         disable=hide_progress,
        #     )

        #     for image in iterator:
        #         image_features = self.extract(image)
        #         all_features.append(image_features.ravel())

        return np.array(all_features)
